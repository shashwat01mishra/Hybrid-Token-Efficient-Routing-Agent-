#!/usr/bin/env python3
"""
Lightweight API and static file web server for the Routing Agent Dashboard.
Wired up specifically to use the SUBMISSION stack files (local_model_gguf.py,
remote_client_submission.py, and router_submission.py) as defined in the
submission Dockerfile to match the hackathon pipeline exactly.
"""

import os
import sys
import json
import time
import traceback
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path

# Add project root directory to python path
project_dir = Path(__file__).parent.absolute()
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Try loading env variables
try:
    from dotenv import load_dotenv
    load_dotenv(project_dir / ".env")
except ImportError:
    pass

# Import agent submission stack components
import config
import math_tool
import prompt_templates
import verify
import router_submission
import remote_client_submission

# Import GGUF local model
try:
    import local_model_gguf
except Exception as e:
    print(f"[ERROR] Failed to import local_model_gguf: {e}", file=sys.stderr)
    local_model_gguf = None


def run_routing_trace(task: str) -> dict:
    """
    Runs the prompt through the agent routing loop and records detailed trace
    telemetry for visual display on the web frontend.
    """
    from datetime import datetime, timezone
    
    category = prompt_templates.classify_category(task, math_tool.is_math_prompt)
    
    try:
        remote_available = remote_client_submission.is_available()
    except Exception:
        remote_available = False
        
    trace = {
        "task": task,
        "category": category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "local_model": config.LOCAL_MODEL_PATH,
        "remote_model": config.FIREWORKS_MODEL,
        "mean_threshold": config.MEAN_LOGPROB_THRESHOLD,
        "min_threshold": config.MIN_LOGPROB_THRESHOLD,
        "remote_available": remote_available,
    }
    
    if local_model_gguf is None:
        raise RuntimeError("local_model_gguf is not imported successfully. Check llama-cpp-python installation.")

    # 1. Math solving path (Deterministic, bypasses routing)
    if category == "math":
        start_time = time.perf_counter()
        
        def _local_generate_plain(prompt: str) -> str:
            text, _features = local_model_gguf.generate(prompt)
            return text
            
        try:
            result = math_tool.solve_math_task(task, _local_generate_plain)
            eval_success = True
            error_msg = ""
        except Exception as e:
            # Degrade gracefully
            result = "[Math solver failure]"
            eval_success = False
            error_msg = str(e)
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Gather mock/real features for frontend gauges
        _, mock_features = local_model_gguf.generate(task)
        local_result = {
            "text": result,
            "mean_logprob": float(mock_features.get("mean_logprob", 0.0)),
            "min_logprob": float(mock_features.get("min_logprob", 0.0)),
            "entropy_mean": float(mock_features.get("entropy_mean", 0.0)),
            "top2_margin_mean": float(mock_features.get("top2_margin_mean", 0.0)),
            "num_tokens": len(result.split()),
            "latency_ms": latency_ms,
        }
        
        trace.update({
            "final_route": "math",
            "final_answer": result,
            "total_latency_ms": latency_ms,
            "total_cost_usd": 0.0,
            "math_info": {
                "extracted_expr": task,
                "cleaned_expr": "Extracted & evaluated by AST solver",
                "eval_success": eval_success,
                "error_msg": error_msg
            },
            "local_generation": local_result,
            "routing_decision": {
                "escalate": False,
                "reason": "Math tasks bypass escalation and are evaluated deterministically via Python AST."
            }
        })
        
        # Log results to runs/agent_log.jsonl
        record = {
            "route": "math",
            "reason": "Math task solved locally via AST",
            "answer": result,
            "cost_usd": 0.0,
            "latency_ms": latency_ms,
            "task": task,
            "local_mean_logprob": local_result["mean_logprob"],
            "local_min_logprob": local_result["min_logprob"],
            "timestamp": trace["timestamp"],
            "category": "math"
        }
        _log_to_file(record)
        return trace

    # 2. General pathway: Run local GGUF model inference
    system_prompt = prompt_templates.get_template(category)
    start_local = time.perf_counter()
    local_text, local_features = local_model_gguf.generate(task, system_prompt=system_prompt)
    local_latency_ms = (time.perf_counter() - start_local) * 1000
    
    mean_lp = float(local_features.get("mean_logprob", 0.0))
    min_lp = float(local_features.get("min_logprob", 0.0))
    entropy_mean = float(local_features.get("entropy_mean", 0.0))
    top2_margin_mean = float(local_features.get("top2_margin_mean", 0.0))

    local_result = {
        "text": local_text,
        "mean_logprob": mean_lp,
        "min_logprob": min_lp,
        "entropy_mean": entropy_mean,
        "top2_margin_mean": top2_margin_mean,
        "num_tokens": len(local_text.split()),
        "latency_ms": local_latency_ms,
    }
    trace["local_generation"] = local_result
    
    # Check syntax if code debugging or generation
    syntax_error_occurred = False
    syntax_error_msg = ""
    
    if category in ("code_debugging", "code_generation"):
        needs_attention, syntax_error_msg = verify.verify_code_answer(local_text)
        if needs_attention:
            syntax_error_occurred = True
            trace["code_verify_failed"] = True
            trace["code_verify_error"] = syntax_error_msg
            
    # Check router decision (uses router_submission.py)
    should_escalate = router_submission.decide(category, local_features, remote_available)
    
    trace["router_decisions"] = {
        "submission_router": {
            "escalate": should_escalate,
            "reason": (
                f"Category: {category} (escalation-eligible: {category in config.ESCALATION_ELIGIBLE_CATEGORIES}). "
                f"Confidence check: mean_lp={mean_lp:.3f} (threshold {config.MEAN_LOGPROB_THRESHOLD}), "
                f"min_lp={min_lp:.3f} (threshold {config.MIN_LOGPROB_THRESHOLD}). "
                f"Remote available: {remote_available}."
            )
        }
    }
    
    # Execute routing / retry logic matching harness.py
    if syntax_error_occurred:
        retry_prompt = verify.build_retry_prompt(task, local_text, syntax_error_msg)
        if remote_available:
            start_remote = time.perf_counter()
            try:
                remote_text = remote_client_submission.query_fireworks(retry_prompt, system_prompt=system_prompt)
                remote_latency = (time.perf_counter() - start_remote) * 1000
                total_latency = local_latency_ms + remote_latency
                
                # Estimate cost for stats visualization
                num_tokens = len(retry_prompt.split()) + len(remote_text.split()) + 20
                cost_usd = (num_tokens / 1000) * config.FIREWORKS_PRICE_PER_1K_TOKENS
                
                trace.update({
                    "final_route": "verify_escalate_remote",
                    "final_answer": remote_text,
                    "total_latency_ms": total_latency,
                    "total_cost_usd": cost_usd,
                    "remote_generation": {
                        "text": remote_text,
                        "latency_ms": remote_latency,
                        "cost_usd": cost_usd,
                        "num_tokens": num_tokens
                    },
                    "routing_decision": {
                        "escalate": True,
                        "reason": f"Code verification syntax error: {syntax_error_msg}. Escalated to remote client."
                    }
                })
            except Exception as e:
                trace.update({
                    "final_route": "verify_escalate_failed",
                    "final_answer": local_text,
                    "total_latency_ms": local_latency_ms,
                    "total_cost_usd": 0.0,
                    "routing_decision": {
                        "escalate": True,
                        "reason": f"Syntax verification failed. Remote recovery error: {e}. Fell back to local answer."
                    }
                })
        else:
            # Local retry
            start_retry = time.perf_counter()
            try:
                retried_text, _features2 = local_model_gguf.generate(retry_prompt, system_prompt=system_prompt)
                retry_latency = (time.perf_counter() - start_retry) * 1000
                
                trace.update({
                    "final_route": "verify_retry_local",
                    "final_answer": retried_text,
                    "total_latency_ms": local_latency_ms + retry_latency,
                    "total_cost_usd": 0.0,
                    "routing_decision": {
                        "escalate": False,
                        "reason": f"Code syntax error: {syntax_error_msg}. Local retry executed."
                    }
                })
            except Exception as e:
                trace.update({
                    "final_route": "verify_retry_failed",
                    "final_answer": local_text,
                    "total_latency_ms": local_latency_ms,
                    "total_cost_usd": 0.0,
                    "routing_decision": {
                        "escalate": False,
                        "reason": f"Code syntax error. Local retry failed: {e}."
                    }
                })
    elif should_escalate:
        # Confidence escalation
        start_remote = time.perf_counter()
        try:
            remote_text = remote_client_submission.query_fireworks(task, system_prompt=system_prompt)
            remote_latency = (time.perf_counter() - start_remote) * 1000
            total_latency = local_latency_ms + remote_latency
            
            num_tokens = len(task.split()) + len(remote_text.split()) + 20
            cost_usd = (num_tokens / 1000) * config.FIREWORKS_PRICE_PER_1K_TOKENS
            
            trace.update({
                "final_route": "remote",
                "final_answer": remote_text,
                "total_latency_ms": total_latency,
                "total_cost_usd": cost_usd,
                "remote_generation": {
                    "text": remote_text,
                    "latency_ms": remote_latency,
                    "cost_usd": cost_usd,
                    "num_tokens": num_tokens
                },
                "routing_decision": {
                    "escalate": True,
                    "reason": trace["router_decisions"]["submission_router"]["reason"]
                }
            })
        except Exception as e:
            trace.update({
                "final_route": "remote_failed",
                "final_answer": local_text,
                "total_latency_ms": local_latency_ms,
                "total_cost_usd": 0.0,
                "routing_decision": {
                    "escalate": True,
                    "reason": f"Escalation failed: {e}. Defaulted back to local."
                }
            })
    else:
        # Standard local route path
        trace.update({
            "final_route": "local",
            "final_answer": local_text,
            "total_latency_ms": local_latency_ms,
            "total_cost_usd": 0.0,
            "routing_decision": {
                "escalate": False,
                "reason": trace["router_decisions"]["submission_router"]["reason"]
            }
        })
        
    # Write to agent_log.jsonl to keep dashboard in sync with file logs
    record = {
        "route": trace["final_route"],
        "reason": trace["routing_decision"]["reason"],
        "answer": trace["final_answer"],
        "cost_usd": trace["total_cost_usd"],
        "latency_ms": trace["total_latency_ms"],
        "task": task,
        "local_mean_logprob": mean_lp,
        "local_min_logprob": min_lp,
        "timestamp": trace["timestamp"],
        "category": category
    }
    _log_to_file(record)
    return trace


def _log_to_file(record: dict) -> None:
    log_path = Path(__file__).parent / "runs" / "agent_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


class AgentDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        if self.path == "/api/config":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            data = {
                "LOCAL_MODEL": config.LOCAL_MODEL_PATH,
                "LOCAL_BACKEND": "llama-cpp-python",
                "FIREWORKS_MODEL": config.FIREWORKS_MODEL,
                "MEAN_LOGPROB_THRESHOLD": config.MEAN_LOGPROB_THRESHOLD,
                "MIN_LOGPROB_THRESHOLD": config.MIN_LOGPROB_THRESHOLD,
                "MOCK_LOCAL_MODEL": config.MOCK_LOCAL_MODEL,
                "MOCK_REMOTE_CLIENT": config.MOCK_REMOTE_CLIENT
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif self.path == "/api/logs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            logs = []
            log_path = Path(__file__).parent / "runs" / "agent_log.jsonl"
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                logs.append(json.loads(line))
                            except Exception:
                                pass
            # Return reversed to have newest logs first
            self.wfile.write(json.dumps(list(reversed(logs))).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                task = payload.get("task", "")
                
                # Execute tracing runner
                trace = run_routing_trace(task)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(trace).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err_response = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                self.wfile.write(json.dumps(err_response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, AgentDashboardHandler)
    print(f"\n=======================================================")
    print(f"Routing Agent Dashboard server successfully initialized!")
    print(f"URL: http://localhost:{port}")
    print(f"Local Model: {config.LOCAL_MODEL_PATH} (GGUF)")
    print(f"Press Ctrl+C to terminate.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...", file=sys.stderr)
        httpd.server_close()
        sys.exit(0)


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", 8080))
    if len(sys.argv) > 1:
        try:
            port_num = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port_num)
