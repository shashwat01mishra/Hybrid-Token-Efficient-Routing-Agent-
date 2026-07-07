"""
Local tier: runs a small open-weights model and returns not just the text,
but the per-token log-probabilities of the tokens it actually generated.
That's the entire signal the router needs, and it costs nothing extra --
it falls out of the forward pass HF already does during generation.
"""
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CFG


@dataclass
class LocalResult:
    text: str
    token_logprobs: List[float]  # log P(token | context) for each generated token
    mean_logprob: float
    min_logprob: float


class LocalModel:
    """Thin wrapper so agent_loop.py doesn't need to know any HF internals."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.local_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            CFG.local_model_name,
            torch_dtype=torch.float16 if CFG.local_device != "cpu" else torch.float32,
        ).to(CFG.local_device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> LocalResult:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(CFG.local_device)

        out = self.model.generate(
            input_ids,
            max_new_tokens=CFG.local_max_new_tokens,
            do_sample=False,  # greedy -- deterministic confidence, no sampling noise in the logprob signal
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = out.sequences[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # out.scores is a tuple of per-step logits, one tensor per generated token
        token_logprobs = []
        for step_logits, tok_id in zip(out.scores, gen_ids):
            logprob = torch.log_softmax(step_logits[0], dim=-1)[tok_id].item()
            token_logprobs.append(logprob)

        if not token_logprobs:
            # degenerate case: model emitted EOS immediately, treat as maximally unconfident
            token_logprobs = [-10.0]

        mean_lp = sum(token_logprobs) / len(token_logprobs)
        min_lp = min(token_logprobs)

        return LocalResult(
            text=text,
            token_logprobs=token_logprobs,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
        )
