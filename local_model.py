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

from config import LOCAL_MODEL, MAX_NEW_TOKENS, MOCK_LOCAL_MODEL, LOCAL_BACKEND


def _degenerate_result(text: str, latency_ms: float) -> dict:
    """Shared empty-generation fallback for both backends. -10.0 rather
    than -inf — maximally unconfident so the router escalates, but still
    a finite float so json.dumps doesn't emit non-standard "-Infinity"
    tokens downstream in the log file."""
    return {
        "text": text,
        "mean_logprob": -10.0,
        "min_logprob": -10.0,
        "entropy_mean": 0.0,
        "top2_margin_mean": 0.0,
        "worst_decile_mean": -10.0,
        "logprob_variance": 0.0,
        "eos_logprob_last": -10.0,
        "num_tokens": 0,
        "latency_ms": latency_ms,
    }


def _shape_result(text: str, logprobs: list, entropies: list,
                   top2_margins: list, eos_logprobs: list,
                   latency_ms: float) -> dict:
    """Shared feature aggregation for both backends — same schema either
    way, so calibrate.py / agent_loop.py never need to know which backend
    ran."""
    if not logprobs:
        return _degenerate_result(text, latency_ms)

    n = len(logprobs)
    mean_lp = sum(logprobs) / n
    variance = sum((x - mean_lp) ** 2 for x in logprobs) / n
    worst_decile_n = max(1, n // 10)
    worst_decile_mean = sum(sorted(logprobs)[:worst_decile_n]) / worst_decile_n

    return {
        "text": text,
        "mean_logprob": mean_lp,
        "min_logprob": min(logprobs),
        "entropy_mean": sum(entropies) / n,
        "top2_margin_mean": sum(top2_margins) / n,
        "worst_decile_mean": worst_decile_mean,
        "logprob_variance": variance,
        "eos_logprob_last": eos_logprobs[-1] if eos_logprobs else -10.0,
        "num_tokens": n,
        "latency_ms": latency_ms,
    }


class LocalModel:
    def __init__(self):
        self.mock = MOCK_LOCAL_MODEL
        if self.mock:
            return

        self.backend = LOCAL_BACKEND
        if self.backend == "mlx":
            self._init_mlx()
        else:
            self._init_transformers()

    def _init_mlx(self):
        # mlx-lm is Apple Silicon only and deliberately NOT in
        # requirements.txt (installing it on the AMD Linux instance would
        # just fail) — install separately: pip install mlx-lm
        from mlx_lm import load
        self.mlx_model, self.mlx_tokenizer = load(LOCAL_MODEL)

    def _init_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU via Metal
        else:
            self.device = "cpu"

        if self.device == "cuda":
            torch_dtype = torch.bfloat16
        elif self.device == "mps":
            # bfloat16 support on Metal has historically been spottier than
            # float16 across PyTorch/macOS version combos — float16 is the
            # safer default here.
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL, torch_dtype=torch_dtype
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
            entropy_mean, top2_margin_mean, worst_decile_mean,
            logprob_variance, eos_logprob_last: extra confidence features,
                logged for a future learned router but not routed on yet
            num_tokens: number of generated tokens
            latency_ms: wall-clock generation time
        """
        start = time.perf_counter()
        if self.mock:
            return self._mock_generate(prompt, start)
        if self.backend == "mlx":
            return self._generate_mlx(prompt, start)
        return self._generate_transformers(prompt, start)

    def _generate_mlx(self, prompt: str, start: float) -> dict:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.mlx_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        prompt_tokens = mx.array(self.mlx_tokenizer.encode(prompt_str))

        eos_ids = getattr(self.mlx_tokenizer, "eos_token_ids", None)
        if not eos_ids:
            eos_ids = {self.mlx_tokenizer.eos_token_id}
        eos_id_for_margin = next(iter(eos_ids))

        sampler = make_sampler(temp=0.0)  # greedy — matches do_sample=False
        # on the transformers path, so both backends make the same
        # deterministic choice at every step and their confidence numbers
        # are actually comparable.

        logprobs_list, entropies, top2_margins, eos_logprobs = [], [], [], []
        generated_ids = []

        for token, logprobs in generate_step(
            prompt_tokens, self.mlx_model,
            max_tokens=MAX_NEW_TOKENS, sampler=sampler,
        ):
            token_id = token.item()
            if token_id in eos_ids:
                break

            logprobs_list.append(logprobs[token_id].item())

            probs = mx.exp(logprobs)
            entropies.append(-(probs * logprobs).sum().item())

            sorted_lp = mx.sort(logprobs)[::-1]
            top2_margins.append((sorted_lp[0] - sorted_lp[1]).item())
            eos_logprobs.append(logprobs[eos_id_for_margin].item())

            generated_ids.append(token_id)

        text = self.mlx_tokenizer.decode(generated_ids)
        latency_ms = (time.perf_counter() - start) * 1000

        return _shape_result(text, logprobs_list, entropies, top2_margins,
                              eos_logprobs, latency_ms)

    def _generate_transformers(self, prompt: str, start: float) -> dict:
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
        eos_id = self.tokenizer.eos_token_id

        # Per-token logprob of the token actually chosen at each step, plus
        # richer signals that fall out of the same distribution for free:
        # entropy (how spread out the model's options were), top-2 margin
        # (winner vs. runner-up — since do_sample=False the chosen token IS
        # the argmax, so this is genuinely new information, not a restate
        # of the chosen-token logprob), and eos logprob at each step (how
        # much the model "wanted" to stop here).
        logprobs, entropies, top2_margins, eos_logprobs = [], [], [], []
        for step_scores, token_id in zip(output.scores, generated_ids):
            step_logprobs = torch.log_softmax(step_scores[0], dim=-1)
            logprobs.append(step_logprobs[token_id].item())

            probs = step_logprobs.exp()
            entropies.append(-(probs * step_logprobs).sum().item())

            top2 = torch.topk(step_logprobs, k=2)
            top2_margins.append((top2.values[0] - top2.values[1]).item())

            if eos_id is not None:
                eos_logprobs.append(step_logprobs[eos_id].item())

        latency_ms = (time.perf_counter() - start) * 1000
        return _shape_result(text, logprobs, entropies, top2_margins,
                              eos_logprobs, latency_ms)

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

        # Independent hash seeds per field — deliberately not derived from
        # mean_lp/min_lp, so a mock sweep over these fields can't discover
        # a spurious correlation that only exists because it's all one hash.
        h2 = int(hashlib.sha256((prompt + "::entropy").encode()).hexdigest(), 16)
        h3 = int(hashlib.sha256((prompt + "::margin").encode()).hexdigest(), 16)
        h4 = int(hashlib.sha256((prompt + "::eos").encode()).hexdigest(), 16)

        entropy_mean = 0.2 + (h2 % 100) / 40      # roughly 0.2 - 2.7
        top2_margin_mean = 0.1 + (h3 % 100) / 50  # roughly 0.1 - 2.1
        eos_logprob_last = -0.5 - (h4 % 100) / 20 # roughly -0.5 - -5.5
        worst_decile_mean = min_lp - (h % 30) / 20
        logprob_variance = ((mean_lp - min_lp) ** 2) / 3

        latency_ms = (time.perf_counter() - start) * 1000 + 5
        return {
            "text": f"[mock local answer for: {prompt[:40]}]",
            "mean_logprob": mean_lp,
            "min_logprob": min_lp,
            "entropy_mean": entropy_mean,
            "top2_margin_mean": top2_margin_mean,
            "worst_decile_mean": worst_decile_mean,
            "logprob_variance": logprob_variance,
            "eos_logprob_last": eos_logprob_last,
            "num_tokens": 42,
            "latency_ms": latency_ms,
        }
