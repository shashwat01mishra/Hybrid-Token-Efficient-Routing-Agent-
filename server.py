#!/usr/bin/env python3
"""
Lightweight API and static file web server for the Routing Agent Dashboard.
Uses only Python standard libraries to avoid package dependencies.
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

# Import agent components
import config
import agent_loop
import math_tool
import prompt_templates
import verify
import router
import router_submission
import remote_client

# Soft fallback if Firework API key is missing and mock remote is disabled
if not config.FIREWORKS_API_KEY and not config.MOCK_REMOTE_CLIENT:
    print("[WARNING] FIREWORKS_API_KEY is not set. Automatically enabling MOCK_REMOTE_CLIENT for testing.", file=sys.stderr)
    config.MOCK_REMOTE_CLIENT = True


def run_routing_trace(task: str) -> dict:
    """
    Runs the prompt through the agent routing loop and records detailed trace
    telemetry for visual display on the web frontend.
    """
    from datetime import datetime, timezone
    
    category = prompt_templates.classify_category(task, math_tool.is_math_prompt)
    local_model = agent_loop._get_local_model()
    
    trace = {
        "task": task,
        "category": category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "local_model": config.LOCAL_MODEL,
        "remote_model": config.FIREWORKS_MODEL,
        "mean_threshold": config.MEAN_LOGPROB_THRESHOLD,
        "min_threshold": config.MIN_LOGPROB_THRESHOLD,
    }
    
    # 1. Math solving path (Deterministic, bypasses routing)
    if category == "math":
        start_time = time.perf_counter()
        extraction_prompt = math_tool.build_extraction_prompt(task)
        local_result = local_model.generate(extraction_prompt)
        raw_expr = local_result.get("text", "")
        expr = math_tool.clean_extracted_expression(raw_expr)
        
        eval_success = False
        result = ""
        error_msg = ""
        if expr:
            try:
                val = math_tool.safe_eval(expr)
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                result = str(val)
                eval_success = True
            except Exception as e:
                result = raw_expr
                error_msg = str(e)
        else:
            result = raw_expr
            error_msg = "Could not extract mathematical expression"
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        trace.update({
            "final_route": "math",
            "final_answer": result,
            "total_latency_ms": latency_ms,
            "total_cost_usd": 0.0,
            "math_info": {
                "extracted_expr": raw_expr,
                "cleaned_expr": expr,
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
            "local_mean_logprob": local_result.get("mean_logprob", 0.0),
            "local_min_logprob": local_result.get("min_logprob", 0.0),
            "timestamp": trace["timestamp"],
            "category": "math"
        }
        agent_loop._log(record)
        return trace

    # 2. General pathway: Run local model inference
    local_result = local_model.generate(task)
    trace["local_generation"] = local_result
    
    mean_lp = local_result.get("mean_logprob", 0.0)
    min_lp = local_result.get("min_logprob", 0.0)
    local_text = local_result.get("text", "")
    
    # Check syntax if code debugging or generation
    syntax_error_occurred = False
    syntax_error_msg = ""
    
    if category in ("code_debugging", "code_generation"):
        needs_attention, syntax_error_msg = verify.verify_code_answer(local_text)
        if needs_attention:
            syntax_error_occurred = True
            trace["code_verify_failed"] = True
            trace["code_verify_error"] = syntax_error_msg
            
    # Compute router decisions (dev vs submission)
    dev_decision = router.decide(mean_lp, min_lp)
    
    remote_available = not config.MOCK_REMOTE_CLIENT or config.FIREWORKS_API_KEY != ""
    sub_escalate = router_submission.decide(category, local_result, remote_available)
    
    trace["router_decisions"] = {
        "dev_router": {
            "escalate": dev_decision.escalate,
            "reason": dev_decision.reason
        },
        "submission_router": {
            "escalate": sub_escalate,
            "reason": f"Category eligibility: {category in config.ESCALATION_ELIGIBLE_CATEGORIES}. Confidence checks: mean_lp={mean_lp:.3f} (threshold {config.MEAN_LOGPROB_THRESHOLD}), min_lp={min_lp:.3f} (threshold {config.MIN_LOGPROB_THRESHOLD})"
        }
    }
    
    # We follow the dev router logic for execution pathway, but incorporate syntax checking.
    # If code syntax check fails, we automatically escalate to remote client (mimics harness.py retry).
    escalate = dev_decision.escalate or syntax_error_occurred
    
    if syntax_error_occurred and remote_available:
        retry_prompt = verify.build_retry_prompt(task, local_text, syntax_error_msg)
        start_time = time.perf_counter()
        try:
            remote_res = remote_client.query_fireworks(retry_prompt)
            latency_ms = local_result.get("latency_ms", 0.0) + (time.perf_counter() - start_time) * 1000
            
            trace.update({
                "final_route": "verify_escalate_remote",
                "final_answer": remote_res.get("text", ""),
                "total_latency_ms": latency_ms,
                "total_cost_usd": remote_res.get("cost_usd", 0.0),
                "remote_generation": remote_res,
                "routing_decision": {
                    "escalate": True,
                    "reason": f"Syntax verification failed: {syntax_error_msg}. Escalate-on-error triggered."
                }
            })
        except Exception as e:
            trace.update({
                "final_route": "verify_escalate_failed",
                "final_answer": local_text,
                "total_latency_ms": local_result.get("latency_ms", 0.0),
                "total_cost_usd": 0.0,
                "routing_decision": {
                    "escalate": True,
                    "reason": f"Syntax verification failed. Remote recovery error: {e}"
                }
            })
    elif escalate:
        # Escalation due to low confidence scores
        start_time = time.perf_counter()
        try:
            remote_res = remote_client.query_fireworks(task)
            latency_ms = local_result.get("latency_ms", 0.0) + (time.perf_counter() - start_time) * 1000
            
            trace.update({
                "final_route": "remote",
                "final_answer": remote_res.get("text", ""),
                "total_latency_ms": latency_ms,
                "total_cost_usd": remote_res.get("cost_usd", 0.0),
                "remote_generation": remote_res,
                "routing_decision": {
                    "escalate": True,
                    "reason": dev_decision.reason
                }
            })
        except Exception as e:
            trace.update({
                "final_route": "remote_failed",
                "final_answer": local_text,
                "total_latency_ms": local_result.get("latency_ms", 0.0),
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
            "total_latency_ms": local_result.get("latency_ms", 0.0),
            "total_cost_usd": 0.0,
            "routing_decision": {
                "escalate": False,
                "reason": dev_decision.reason
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
    agent_loop._log(record)
    return trace


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
                "LOCAL_MODEL": config.LOCAL_MODEL,
                "LOCAL_BACKEND": config.LOCAL_BACKEND,
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
    print(f"Local config: {config.LOCAL_MODEL} ({config.LOCAL_BACKEND})")
    print(f"Press Ctrl+C to terminate.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...", file=sys.stderr)
        httpd.server_close()
        sys.exit(0)


if __name__ == "__main__":
    port_num = 8080
    if len(sys.argv) > 1:
        try:
            port_num = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port_num)
