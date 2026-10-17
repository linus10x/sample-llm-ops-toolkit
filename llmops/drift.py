"""Score drift between a baseline window and a current window."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _shares(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        placed = False
        for i in range(len(edges) - 1):
            last = i == len(edges) - 2
            if edges[i] <= value < edges[i + 1] or (last and value <= edges[i + 1]):
                counts[i] += 1
                placed = True
                break
        if not placed:
            if value < edges[0]:
                counts[0] += 1
            else:
                counts[-1] += 1
    total = sum(counts) or 1
    # Empty bins use a small floor so the log stays defined.
    floor = 0.001
    shares = [max(floor, count / total) for count in counts]
    norm = sum(shares)
    return [share / norm for share in shares]


def population_stability(baseline: list[float], current: list[float], bins: int = 5) -> float:
    if not baseline or not current:
        raise ValueError("both windows need scores")
    lo = min(min(baseline), min(current))
    hi = max(max(baseline), max(current))
    if hi == lo:
        return 0.0
    step = (hi - lo) / bins
    edges = [lo + step * i for i in range(bins + 1)]
    base = _shares(baseline, edges)
    curr = _shares(current, edges)
    psi = 0.0
    for b, c in zip(base, curr):
        psi += (c - b) * math.log(c / b)
    return psi


def mean_shift(baseline: list[float], current: list[float]) -> float:
    return (sum(current) / len(current)) - (sum(baseline) / len(baseline))


@dataclass(frozen=True)
class DriftReport:
    psi: float
    mean_shift: float
    alert: bool
    reasons: tuple[str, ...]


def compare(
    baseline: list[float],
    current: list[float],
    psi_limit: float = 0.25,
    mean_limit: float = 0.10,
) -> DriftReport:
    psi = population_stability(baseline, current)
    shift = mean_shift(baseline, current)
    reasons = []
    if psi >= psi_limit:
        reasons.append(f"psi {psi:.4f} >= {psi_limit}")
    if abs(shift) >= mean_limit:
        reasons.append(f"mean_shift {shift:.4f} abs >= {mean_limit}")
    return DriftReport(psi=psi, mean_shift=shift, alert=bool(reasons), reasons=tuple(reasons))
