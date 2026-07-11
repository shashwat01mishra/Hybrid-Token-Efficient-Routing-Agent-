"""
Category detection and category-specific prompt scaffolding.

The real grading harness's I/O contract only gives us {"task_id", "prompt"} —
no category label. So detection has to work purely on prompt text, the same
way math_tool.py detects arithmetic. This must stay cheap: regex/keyword only,
no LLM call, since classification itself must cost zero tokens and near-zero
latency under the 30s-per-request / 9-minute-total budget.

Detection order matters — check more specific/unambiguous signals first,
fall through to "factual" as the default since it's the safest catch-all
for a general-purpose small model.
"""
import re

CODE_BLOCK_RE = re.compile(r"```|def |function\s+\w+\s*\(|class \w+|Traceback")
DEBUG_WORDS_RE = re.compile(
    r"\b(bug|fix this|debug|error in|not working|wrong output|why does this fail|"
    r"expected .* but got|throws? an? (error|exception))\b",
    re.IGNORECASE,
)
CODEGEN_WORDS_RE = re.compile(
    r"\b(write a function|write a program|implement a|write code (that|to)|"
    r"create a function|generate code)\b",
    re.IGNORECASE,
)
SENTIMENT_RE = re.compile(
    r"\b(sentiment|positive or negative|classify.*(review|tweet|feedback)|"
    r"is this (review|comment|feedback) (positive|negative))\b",
    re.IGNORECASE,
)
SUMMARIZE_RE = re.compile(
    r"\b(summari[sz]e|summary|tl;?dr|in a few sentences|condense the following)\b",
    re.IGNORECASE,
)
NER_RE = re.compile(
    r"\b(named entit(y|ies)|extract (the )?(entities|people|organi[sz]ations|locations)|"
    r"identify (the )?(people|places|organi[sz]ations) (mentioned|in))\b",
    re.IGNORECASE,
)
LOGIC_RE = re.compile(
    r"\b(true or false|if .* then|syllogism|deduce|logically (follows|valid)|"
    r"who is lying|which statement|riddle|puzzle|premise)\b",
    re.IGNORECASE,
)

CATEGORIES = (
    "math",
    "code_debugging",
    "code_generation",
    "sentiment",
    "summarization",
    "ner",
    "logical_reasoning",
    "factual",
)


def classify_category(prompt: str, is_math_fn) -> str:
    """
    is_math_fn: math_tool.is_math_prompt, passed in to avoid a circular
    import and to keep math detection as the single source of truth there.
    """
    if is_math_fn(prompt):
        return "math"

    has_code = bool(CODE_BLOCK_RE.search(prompt))
    if has_code and DEBUG_WORDS_RE.search(prompt):
        return "code_debugging"
    if CODEGEN_WORDS_RE.search(prompt):
        return "code_generation"
    if has_code and not DEBUG_WORDS_RE.search(prompt) and not CODEGEN_WORDS_RE.search(prompt):
        # Ambiguous code block with no clear debug/gen signal — debugging is
        # the safer assumption since gen prompts almost always contain
        # "write"/"implement"/"create" explicitly.
        return "code_debugging"
    if SENTIMENT_RE.search(prompt):
        return "sentiment"
    if SUMMARIZE_RE.search(prompt):
        return "summarization"
    if NER_RE.search(prompt):
        return "ner"
    if LOGIC_RE.search(prompt):
        return "logical_reasoning"

    return "factual"


# Category-specific system prompts. Kept short deliberately — every extra
# token here costs local inference time too (2 vCPU, no GPU), not just
# Fireworks tokens on escalation.

_TEMPLATES = {
    "factual": (
        "Answer the question directly and concisely. State only the answer, "
        "with a brief supporting fact if useful."
    ),
    "sentiment": (
        "Classify the sentiment as Positive, Negative, or Neutral. "
        "Give the label first, then a one-sentence justification."
    ),
    "summarization": (
        "Summarize the text in 2-3 sentences. Prioritize covering the key "
        "points over compressing aggressively — do not omit a central fact "
        "to save words."
    ),
    "ner": (
        "Extract named entities from the text. Respond ONLY with a JSON list "
        'of objects: [{"entity": "...", "type": "PERSON|ORG|LOCATION|DATE|MISC"}]. '
        "No extra text before or after the JSON.\n"
        'Example: Text: "Marie Curie worked at the Sorbonne in Paris." -> '
        '[{"entity": "Marie Curie", "type": "PERSON"}, {"entity": "Sorbonne", "type": "ORG"}, '
        '{"entity": "Paris", "type": "LOCATION"}]'
    ),
    "code_debugging": (
        "Debug the code step by step. First state what the code is SUPPOSED to "
        "do. Then trace what it ACTUALLY does. Then identify exactly where they "
        "diverge. Only then give the fix, clearly marked as 'Fix:'."
    ),
    "code_generation": (
        "Write clean, correct code that satisfies the request. Include the code "
        "in a single code block. Add a one-line comment only where logic isn't "
        "obvious."
    ),
    "logical_reasoning": (
        "Reason step by step before answering. List each logical step on its "
        "own line, numbered. Then give the final answer clearly marked as "
        "'Answer:'.\n"
        "Example: Premise: All cats are mammals. Premise: Whiskers is a cat. "
        "1. All cats are mammals. 2. Whiskers is a cat. 3. Therefore Whiskers is "
        "a mammal. Answer: Whiskers is a mammal."
    ),
}


def get_template(category: str) -> str:
    return _TEMPLATES.get(category, _TEMPLATES["factual"])
