"""Deterministic ambiguity signals for a standalone software requirement."""

import re
from dataclasses import dataclass
from typing import TypedDict


class Issue(TypedDict):
    text: str
    category: str
    explanation: str
    start: int
    end: int
    weight: int


class AmbiguityResult(TypedDict):
    ambiguous: bool
    score: int
    severity: str
    issues: list[Issue]


WEIGHTS = {
    "vague_term": 2,
    "weak_quantifier": 1,
    "weak_obligation": 1,
    "undefined_comparative": 2,
    "unclear_reference": 2,
    "open_ended_phrase": 3,
}


@dataclass(frozen=True)
class Rule:
    category: str
    pattern: re.Pattern[str]
    explanation: str
    context: str = ""


def phrases(category: str, terms: tuple[str, ...], explanation: str, context: str = "") -> Rule:
    # Allow whitespace and typographic hyphens between words, with full token boundaries.
    alternatives = [
        r"[\s\-‐‑]+".join(map(re.escape, re.split(r"[\s-]+", term)))
        for term in sorted(terms, key=len, reverse=True)
    ]
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)
    return Rule(category, pattern, explanation, context)


RULES = (
    phrases(
        "vague_term", ("quickly", "rapidly"),
        "The requirement does not specify a measurable response-time threshold.", "time",
    ),
    phrases(
        "vague_term",
        ("efficiently", "intuitive", "user-friendly", "adequate", "appropriate", "sufficient", "reasonable"),
        "The requirement leaves this quality judgment without a testable acceptance criterion.",
    ),
    phrases(
        "weak_quantifier", ("some", "several", "many", "usually", "generally", "often"),
        "The requirement leaves the amount, frequency, or covered cases unspecified.",
    ),
    phrases(
        "weak_quantifier", ("approximately",),
        "The requirement gives an approximation without a defined tolerance.", "approximation",
    ),
    phrases(
        "weak_obligation", ("if possible", "as needed", "where appropriate", "may", "might", "could", "should"),
        "The wording leaves the obligation or the condition that triggers it unclear.", "modality",
    ),
    phrases(
        "undefined_comparative", ("faster", "better", "higher", "lower", "minimal", "optimal"),
        "The comparison or optimum lacks a measurable reference or acceptance threshold.", "comparison",
    ),
    Rule(
        "unclear_reference",
        re.compile(
            r"^\s*(?:(?:if|when|unless)\s+)?(?P<target>it|this|that|they|these)\s+"
            r"(?:shall|must|should|may|might|could|is|are|was|were|will|fails?|occurs?|happens?)\b",
            re.IGNORECASE,
        ),
        "The opening pronoun has no explicit antecedent in the requirement.",
        "reference",
    ),
    phrases(
        "open_ended_phrase", ("as soon as possible",),
        "The requirement gives no objective deadline for this action.", "time",
    ),
    phrases(
        "open_ended_phrase", ("as appropriate", "as required", "if necessary", "and so on"),
        "The requirement leaves a condition or part of the scope open to interpretation.", "citation",
    ),
    Rule(
        "open_ended_phrase", re.compile(r"(?<!\w)etc\.(?!\w)", re.IGNORECASE),
        "The list is open-ended. Specify which additional items it includes.",
    ),
)

QUOTED = re.compile(r'''"[^"\n]*"|“[^”\n]*”|(?<!\w)'[^'\n]*'(?!\w)|\x60[^\x60\n]*\x60''')
NUMBER = r"[-+]?\d+(?:[.,]\d+)?"
DURATION = NUMBER + r"\s*(?:ms|milliseconds?|s|seconds?|min|minutes?|h|hours?)(?!\w)"
TIMING_LIMIT = re.compile(
    r"^\s*[,(:–-]?\s*(?:(?:i\.e\.|defined as|but|and)\s+)?"
    r"(?:within|at most|in under|in no more than|no later than)\s+" + DURATION,
    re.IGNORECASE,
)
TOLERANCE = re.compile(
    r"^\s+" + NUMBER + r"(?:\s+[a-z]+)?\s*[,(\[]?\s*"
    r"(?:with (?:a )?tolerance of\s+)?(?:±|\+/-)\s*" + NUMBER,
    re.IGNORECASE,
)
MEASUREMENT = NUMBER + r"\s*(?:%|ms|milliseconds?|s|seconds?|min|minutes?|hours?|[kmg]i?b)(?!\w)"
NUMERIC_REFERENCE = re.compile(r"^\s+than\s+" + MEASUREMENT, re.IGNORECASE)
NUMERIC_DIFFERENCE = re.compile(MEASUREMENT + r"\s*$", re.IGNORECASE)
PERMITTED_ACTOR = re.compile(
    r"\b(?:users?|administrators?|operators?|auditors?|customers?|managers?|employees?)\s*$",
    re.IGNORECASE,
)
CITED_REQUIREMENT = re.compile(
    r"^\s+(?:by|under|in)\s+(?:(?i:RFC|ISO|IEEE|NIST)\s+\d+|[A-Z][A-Z0-9]+-\d+)\b"
)
EXPLETIVE_IT = re.compile(
    r"^\s+(?:(?:shall|must|should|may|will)\s+be|is|was)\s+"
    r"(?:possible|required|necessary|mandatory|recommended|permitted|acceptable)\b",
    re.IGNORECASE,
)


def has_resolving_context(context: str, hit: str, before: str, after: str) -> bool:
    if context == "time":
        return bool(TIMING_LIMIT.match(after))
    if context == "approximation":
        return bool(TOLERANCE.match(after))
    if context == "modality" and hit.casefold() == "may":
        return bool(
            PERMITTED_ACTOR.search(before)
            or re.match(r"\s+only\b", after, re.IGNORECASE)
            or re.search(r"\b\d{1,2}\s*$", before)
            or re.match(r"\s+\d{4}\b", after)
        )
    if context == "comparison":
        return bool(
            NUMERIC_REFERENCE.match(after)
            or (NUMERIC_DIFFERENCE.search(before) and re.match(r"\s+than\b", after, re.IGNORECASE))
        )
    if context == "reference":
        return hit.casefold() == "it" and bool(EXPLETIVE_IT.match(after))
    if context == "citation":
        return hit.casefold() == "as required" and bool(CITED_REQUIREMENT.match(after))
    return False


def find_issues(text: str) -> list[Issue]:
    literals = [match.span() for match in QUOTED.finditer(text)]
    candidates: list[Issue] = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            start, end = match.span("target") if "target" in match.re.groupindex else match.span()
            if any(left <= start < right for left, right in literals):
                continue
            hit = text[start:end]
            if has_resolving_context(rule.context, hit, text[:start], text[end:]):
                continue
            candidates.append(
                Issue(text=hit, category=rule.category, explanation=rule.explanation,
                      start=start, end=end, weight=WEIGHTS[rule.category])
            )
    # A compound trigger owns its span, so it cannot also score its embedded words.
    candidates.sort(key=lambda issue: (issue["start"], -(issue["end"] - issue["start"])))
    distinct_spans = []
    end = 0
    for issue in candidates:
        if issue["start"] >= end:
            distinct_spans.append(issue)
            end = issue["end"]
    return distinct_spans


def detect_ambiguity(text: str, *, vague_only: bool = False) -> AmbiguityResult:
    """Return local rule matches and a 0–10 heuristic score for non-blank text."""
    if not isinstance(text, str):
        raise TypeError("Requirement text must be a string.")
    if not text.strip():
        raise ValueError("Requirement text must not be blank.")

    issues = find_issues(text)
    if vague_only:
        issues = [issue for issue in issues if issue["category"] == "vague_term"]
    unique_weights = {
        (issue["category"], re.sub(r"[\s\-‐‑]+", " ", issue["text"].casefold())): issue["weight"]
        for issue in issues
    }
    score = min(10, sum(unique_weights.values()))
    severity = "high" if score >= 6 else "medium" if score >= 3 else "low"
    return {"ambiguous": bool(issues), "score": score, "severity": severity, "issues": issues}
