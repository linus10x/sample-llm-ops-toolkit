"""Eval harness. The responder is injected so CI does not call a model."""

from __future__ import annotations

import re
from dataclasses import dataclass


def token_f1(output: str, expected: str) -> float:
    def toks(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    pred = toks(output)
    gold = toks(expected)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common: dict[str, int] = {}
    for token in gold:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in pred:
        if common.get(token, 0):
            overlap += 1
            common[token] -= 1
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_case(output: str, expected: str, mode: str = "contains") -> float:
    if mode == "exact":
        return 1.0 if output.strip() == expected.strip() else 0.0
    if mode == "contains":
        return 1.0 if expected.lower() in output.lower() else 0.0
    if mode == "f1":
        return token_f1(output, expected)
    raise ValueError(f"unknown mode {mode}")


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    score: float
    output: str


@dataclass(frozen=True)
class RunReport:
    prompt_name: str
    prompt_version: int
    results: tuple[CaseResult, ...]

    @property
    def mean(self) -> float:
        if not self.results:
            return 0.0
        return sum(row.score for row in self.results) / len(self.results)


class LocalResponder:
    """Fills {slots}. A stand-in, not a hosted model."""

    def __call__(self, template: str, variables: dict) -> str:
        missing = [key for key in re.findall(r"{([a-z_]+)}", template) if key not in variables]
        if missing:
            raise KeyError(",".join(missing))
        return template.format(**variables)


class EvalHarness:
    def __init__(self, responder) -> None:
        self.responder = responder

    def run(self, prompt, cases: list[dict]) -> RunReport:
        results = []
        for case in cases:
            output = self.responder(prompt.template, case["variables"])
            results.append(
                CaseResult(
                    case_id=case["id"],
                    score=score_case(output, case["expected"], case.get("mode", "contains")),
                    output=output,
                )
            )
        return RunReport(prompt.name, prompt.version, tuple(results))
