"""
Deterministic verification layer for code categories (code_debugging,
code_generation). This is NOT a judge — it doesn't check semantic
correctness — but a syntactically broken answer is a certain failure on
the accuracy gate, and this catches that case for free (pure Python
compile(), zero tokens, near-zero latency).

Several other Track 1 submissions visible on the public leaderboard use an
"answer verified by execution" pattern for exactly this reason — a cheap,
deterministic check before spending anything (local time or remote tokens)
trusting an answer that's already provably wrong.

This intentionally only checks Python syntax validity, not runtime
correctness — full execution would require a sandboxed runner, which is
out of scope given the time/complexity budget. Catching a hard syntax
error is still meaningfully better than nothing, since a syntax error
means the accuracy gate fails with certainty.
"""
import re

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str):
    """Pull the first fenced code block out of a response. Returns None if
    no fenced block is found (falls back to treating the whole response as
    code only if it looks code-like, otherwise skips verification rather
    than false-flagging prose as broken code)."""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    return None


def check_python_syntax(code: str):
    """Returns (is_valid: bool, error_message: str)."""
    try:
        compile(code, "<answer>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        # Extremely rare (e.g. null bytes) — treat as invalid rather than
        # crash the harness.
        return False, f"{type(e).__name__}: {e}"


def verify_code_answer(answer_text: str):
    """
    Returns (needs_attention: bool, reason: str).
    needs_attention=False whenever there's no extractable code block to
    check (prose-only answers, or code without fences) — verification is
    a bonus signal, not a requirement, so absence of a block is never
    treated as a failure.
    """
    code = extract_code_block(answer_text)
    if code is None:
        return False, ""
    is_valid, error_msg = check_python_syntax(code)
    if is_valid:
        return False, ""
    return True, error_msg


def build_retry_prompt(original_prompt: str, broken_answer: str, error_msg: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Your previous answer had a syntax error and cannot run: {error_msg}\n"
        f"Previous answer:\n{broken_answer}\n\n"
        f"Provide a corrected version. Respond with a single fenced code block only."
    )
