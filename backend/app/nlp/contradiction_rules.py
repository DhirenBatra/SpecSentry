"""Flag explicit conflicts in clauses with matching actors and context."""

import re
from decimal import Decimal

from .contradictions import checked_text

NEGATIVE = ("shall not", "must not", "may not", "shall never", "must never", "is forbidden to", "is not permitted to")
MANDATORY = ("shall", "must")
POSITIVE = MANDATORY + ("may", "is permitted to")
CLAUSE = re.compile(rf"(?P<subject>.+?) (?P<modal>{'|'.join(NEGATIVE + POSITIVE)}) (?P<body>.+)")

LOWER = ("at least", "no less than", "more than", "greater than")
UPPER = ("at most", "no more than", "no longer than", "within", "less than", "under")
EXCLUSIVE = ("more than", "greater than", "less than", "under")
BOUND = re.compile(
    rf"(?<!\w)(?P<operator>{'|'.join(('exactly',) + LOWER + UPPER)}) "
    r"(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?|(?:concurrent )?users?|requests?|records?|bits?|bytes?|percent|%)(?!\w)"
)
CONDITIONAL = re.compile(r"\b(?:if|when|whenever|while|unless|until)\b")
INFINITY = Decimal("Infinity")


def _clause(text):
    return CLAUSE.fullmatch(" ".join(checked_text(text).casefold().split()).rstrip(".!?"))


def _bound(body):
    matches = list(BOUND.finditer(body))
    if len(matches) != 1 or CONDITIONAL.search(body):
        return None
    match = matches[0]
    operator, number, unit = match.group("operator", "number", "unit")
    value = Decimal(number)
    context = (body[:match.start()], body[match.end():], unit)
    inclusive = operator not in EXCLUSIVE
    if operator == "exactly":
        interval = (value, value, True, True)
    elif operator in LOWER:
        interval = (value, INFINITY, inclusive, False)
    else:
        interval = (-INFINITY, value, False, inclusive)
    return context, interval


def _before(left, right):
    _, end, _, end_closed = left
    start, _, start_closed, _ = right
    return end < start or (end == start and not (end_closed and start_closed))


def explicit_conflict(requirement_a: str, requirement_b: str) -> str | None:
    """Return the rule name for a matched conflict, or None."""
    first = _clause(requirement_a)
    second = _clause(requirement_b)
    if first is None or second is None or first["subject"] != second["subject"]:
        return None
    modal_a, body_a = first.group("modal", "body")
    modal_b, body_b = second.group("modal", "body")
    if body_a == body_b and ((modal_a in NEGATIVE) != (modal_b in NEGATIVE)):
        return "opposite_obligations"
    if modal_a in MANDATORY and modal_b in MANDATORY:
        bound_a, bound_b = _bound(body_a), _bound(body_b)
        if bound_a is not None and bound_b is not None and bound_a[0] == bound_b[0]:
            if _before(bound_a[1], bound_b[1]) or _before(bound_b[1], bound_a[1]):
                return "disjoint_numeric_bounds"
    return None
