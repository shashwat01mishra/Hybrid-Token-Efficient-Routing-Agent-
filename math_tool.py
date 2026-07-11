"""
Deterministic fix for the confirmed failure mode: small models are fluently
and confidently WRONG on multi-digit arithmetic (well-documented even at
frontier scale — this is architectural, not a small-model quirk). No
logprob-confidence threshold catches it because a wrong product looks exactly
as smooth to generate as a right one.

Approach: the LLM extracts the arithmetic expression from the word problem,
Python evaluates it exactly via a restricted AST walker (never eval()).
Only touches math-flagged prompts; every other category passes through
completely untouched, zero extra tokens or latency cost.
"""
import ast
import operator
import re

MATH_HINT_RE = re.compile(
    r"(\d+\s*[\+\-\*x×÷/]\s*\d+)|"  # explicit "19 x 33" / "19*33" style
    r"\b(calculate|compute|what is|how much|how many|sum of|product of|"
    r"percent(age)?|total cost|difference between|multiplied by|divided by|"
    r"add|subtract|multiply|divide)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d")

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def is_math_prompt(prompt: str) -> bool:
    """Cheap heuristic: needs at least one digit AND either an explicit
    operator pattern or a math-signal keyword. Avoids false-positiving on
    prompts that merely mention a number (e.g. 'Named entities in this: ...
    2024 was a big year')."""
    if not NUMBER_RE.search(prompt):
        return False
    return bool(MATH_HINT_RE.search(prompt))


class UnsafeExpressionError(Exception):
    pass


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise UnsafeExpressionError(f"Non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise UnsafeExpressionError(f"Disallowed operator: {op_type}")
        return _ALLOWED_BINOPS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise UnsafeExpressionError(f"Disallowed unary operator: {op_type}")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand))
    raise UnsafeExpressionError(f"Disallowed expression node: {type(node)}")


def safe_eval(expression: str):
    """
    Restricted AST walker. Only numeric literals and +,-,*,/,**,%, unary +/-,
    and parentheses are permitted. Explicitly rejects names, calls, attribute
    access, subscripts, comprehensions — anything that could be used for
    code injection (__import__, open(), etc.) is a parse-time or eval-time
    error, never silently executed.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Could not parse expression: {e}")
    return _eval_node(tree.body)


EXTRACTION_INSTRUCTION = (
    "Extract the arithmetic expression implied by this problem. Respond with "
    "ONLY the numeric expression using +, -, *, /, (, ) — no words, no "
    "explanation, no equals sign, no units.\n"
    "Example: 'If a store sells 12 boxes at $9 each, what is the total?' -> 12*9\n"
    "Problem: "
)


def build_extraction_prompt(original_prompt: str) -> str:
    return EXTRACTION_INSTRUCTION + original_prompt


_EXPR_CLEAN_RE = re.compile(r"[^0-9\.\+\-\*/\(\)\s%]")


def clean_extracted_expression(raw: str) -> str:
    """Strip anything the model added besides the expression itself
    (stray words, trailing periods, '=' followed by the answer, etc.)."""
    raw = raw.strip()
    # If the model included "= <answer>", drop everything from '=' onward.
    if "=" in raw:
        raw = raw.split("=")[0]
    raw = _EXPR_CLEAN_RE.sub("", raw).strip()
    raw = raw.replace("%", "/100")
    return raw


def solve_math_task(original_prompt: str, local_generate_fn) -> str:
    """
    local_generate_fn(prompt: str) -> str — the caller's local model
    generation function, kept as a plain callable so this module has no
    dependency on which backend (gguf/transformers/mock) is in use.

    Falls back to returning the raw extraction attempt if evaluation fails,
    rather than raising — this must never crash the harness.
    """
    extraction_prompt = build_extraction_prompt(original_prompt)
    raw_expr = local_generate_fn(extraction_prompt)
    expr = clean_extracted_expression(raw_expr)
    if not expr:
        return raw_expr.strip()
    try:
        result = safe_eval(expr)
    except UnsafeExpressionError:
        return raw_expr.strip()
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)
