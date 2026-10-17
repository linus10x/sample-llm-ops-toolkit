"""Token cost ledger. Prices are a sample table, not an invoice."""

from __future__ import annotations

from dataclasses import dataclass

# USD per token. Sample only.
PRICES = {
    "sample-local": {"input": 0.000002, "output": 0.000008},
    "sample-large": {"input": 0.000015, "output": 0.000060},
}


class CostError(ValueError):
    pass


@dataclass(frozen=True)
class Charge:
    day: str
    model: str
    input_tokens: int
    output_tokens: int
    usd: float


class CostMonitor:
    def __init__(self, prices: dict | None = None) -> None:
        self.prices = prices or PRICES
        self.charges: list[Charge] = []

    def record(self, day: str, model: str, input_tokens: int, output_tokens: int) -> Charge:
        if model not in self.prices:
            raise CostError(f"unknown model {model}")
        if input_tokens < 0 or output_tokens < 0:
            raise CostError("token counts must be nonnegative")
        rate = self.prices[model]
        usd = input_tokens * rate["input"] + output_tokens * rate["output"]
        charge = Charge(day, model, input_tokens, output_tokens, round(usd, 8))
        self.charges.append(charge)
        return charge

    def spend(self, day: str | None = None) -> float:
        rows = self.charges if day is None else [row for row in self.charges if row.day == day]
        return round(sum(row.usd for row in rows), 8)

    def over_budget(self, day: str, budget_usd: float) -> bool:
        return self.spend(day) > budget_usd

    def by_model(self, day: str) -> dict[str, float]:
        totals: dict[str, float] = {}
        for row in self.charges:
            if row.day != day:
                continue
            totals[row.model] = round(totals.get(row.model, 0.0) + row.usd, 8)
        return totals
