"""Register two prompt versions, score them, and check drift and cost."""

from llmops.cost import CostMonitor
from llmops.drift import compare
from llmops.harness import EvalHarness, LocalResponder
from llmops.registry import PromptRegistry


CASES = [
    {
        "id": "hours",
        "variables": {"desk": "sample desk", "hours": "8 to 6 Central"},
        "expected": "8 to 6 Central",
        "mode": "contains",
    },
    {
        "id": "keep",
        "variables": {"desk": "sample desk", "hours": "90 days"},
        "expected": "90 days",
        "mode": "contains",
    },
]


def main() -> int:
    registry = PromptRegistry()
    v1 = registry.add(
        "desk-note",
        "The {desk} keeps the stated window: {hours}.",
        note="first wording",
    )
    v2 = registry.add(
        "desk-note",
        "Window for the {desk}: {hours}. Reply with that window only.",
        note="tighter wording",
    )
    registry.promote("desk-note", v2.version)
    harness = EvalHarness(LocalResponder())
    base = harness.run(registry.get("desk-note", v1.version), CASES)
    current = harness.run(registry.get("desk-note"), CASES)
    report = compare([row.score for row in base.results], [row.score for row in current.results])
    monitor = CostMonitor()
    monitor.record("2026-10-16", "sample-local", 1000, 200)
    monitor.record("2026-10-16", "sample-local", 400000, 100000)
    print(f"active {registry.active_version('desk-note')}")
    print(f"v1_mean {base.mean:.3f} v2_mean {current.mean:.3f} drift_alert {report.alert}")
    print(f"day_spend {monitor.spend('2026-10-16'):.6f} over_1usd {monitor.over_budget('2026-10-16', 1.0)}")
    if registry.active_version("desk-note") != 2 or base.mean < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
