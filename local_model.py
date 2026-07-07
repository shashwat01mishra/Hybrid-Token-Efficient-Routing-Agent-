"""
Loads the local model, generates a response, and extracts per-token
log-probabilities as a side effect of generation — no extra forward pass,
no extra cost. Those logprobs are the raw material for the routing
decision in router.py.

Heavy imports (torch, transformers) are done lazily inside __init__ so
that MOCK_LOCAL_MODEL=1 works without either installed.
"""
import time
import hashlib

from config import LOCAL_MODEL, MAX_NEW_TOKENS, MOCK_LOCAL_MODEL


class LocalModel:
    def __init__(self):
        self.mock = MOCK_LOCAL_MODEL
        if self.mock:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> dict:
        """
        Returns:
            text: the generated answer
            mean_logprob: average per-token log-probability (smooth
                confidence signal across the whole answer)
            min_logprob: log-probability of the single weakest token
                (catches a fabricated number/entity that the mean can hide)
            num_tokens: number of generated tokens
            latency_ms: wall-clock generation time
        """
        start = time.perf_counter()
        if self.mock:
            return self._mock_generate(prompt, start)

        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Per-token logprob of the token actually chosen at each step.
        logprobs = []
        for step_scores, token_id in zip(output.scores, generated_ids):
            step_logprobs = torch.log_softmax(step_scores[0], dim=-1)
            logprobs.append(step_logprobs[token_id].item())

        latency_ms = (time.perf_counter() - start) * 1000

        if not logprobs:
            # Degenerate case: empty generation. Treat as maximally
            # unconfident so the router escalates rather than silently
            # returning nothing.
            return {
                "text": text,
                "mean_logprob": float("-inf"),
                "min_logprob": float("-inf"),
                "num_tokens": 0,
                "latency_ms": latency_ms,
            }

        return {
            "text": text,
            "mean_logprob": sum(logprobs) / len(logprobs),
            "min_logprob": min(logprobs),
            "num_tokens": len(logprobs),
            "latency_ms": latency_ms,
        }

    def _mock_generate(self, prompt: str, start: float) -> dict:
        """
        Deterministic fake output for pipeline testing without model
        weights or a GPU. Confidence is derived from a hash of the prompt
        so behavior is reproducible run-to-run, and both the "keep local"
        and "escalate" branches get exercised across different prompts.
        """
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        mean_lp = -0.1 - (h % 100) / 100        # ranges roughly -0.1 to -1.09
        min_lp = mean_lp - (h % 50) / 20         # a bit worse than the mean
        latency_ms = (time.perf_counter() - start) * 1000 + 5
        return {
            "text": f"[mock local answer for: {prompt[:40]}]",
            "mean_logprob": mean_lp,
            "min_logprob": min_lp,
            "num_tokens": 42,
            "latency_ms": latency_ms,
        }
