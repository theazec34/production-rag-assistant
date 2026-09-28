"""Simple, explicit prompt-injection heuristics.

Heuristics reduce obvious abuse; they are not a complete security boundary.
"""

import re
from dataclasses import dataclass

_PATTERNS = {
    "instruction override": re.compile(
        r"\b(ignore|disregard|override)\b.{0,50}\b"
        r"(previous|prior|system|developer|instructions?|rules?)\b",
        re.IGNORECASE,
    ),
    "prompt extraction": re.compile(
        r"\b(reveal|show|print|repeat|leak)\b.{0,50}\b"
        r"(system prompt|hidden prompt|developer message|instructions?)\b",
        re.IGNORECASE,
    ),
    "role injection": re.compile(
        r"(<\|?(system|assistant|developer)\|?>|\[(system|assistant|developer)\])",
        re.IGNORECASE,
    ),
    "context exfiltration": re.compile(
        r"\b(exfiltrate|dump)\b.{0,40}\b(context|documents?|secrets?|credentials?)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class SafetyCheck:
    safe: bool
    reason: str


def check_question(question: str) -> SafetyCheck:
    for label, pattern in _PATTERNS.items():
        if pattern.search(question):
            return SafetyCheck(False, f"Detected possible {label}.")
    return SafetyCheck(True, "No prompt-injection pattern detected.")
