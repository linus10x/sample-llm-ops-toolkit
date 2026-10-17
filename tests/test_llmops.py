import unittest

from llmops.cost import CostError, CostMonitor
from llmops.drift import compare, population_stability
from llmops.harness import EvalHarness, LocalResponder, score_case
from llmops.registry import PromptRegistry, RegistryError


class RegistryTests(unittest.TestCase):
    def test_versions_are_immutable_and_promotion_selects_active(self):
        reg = PromptRegistry()
        first = reg.add("answer", "Say {fact}.")
        second = reg.add("answer", "Reply only: {fact}.")
        self.assertEqual(first.version, 1)
        self.assertEqual(reg.active_version("answer"), 1)
        reg.promote("answer", 2)
        self.assertEqual(reg.get("answer").version, 2)
        self.assertEqual(reg.get("answer", 1).template, "Say {fact}.")
        again = reg.add("answer", "Say {fact}.")
        self.assertEqual(again.version, 1)
        self.assertEqual(second.version, 2)

    def test_blank_template_rejected(self):
        with self.assertRaises(RegistryError):
            PromptRegistry().add("answer", "  ")


class HarnessTests(unittest.TestCase):
    def test_contains_and_f1(self):
        self.assertEqual(score_case("kept for 90 days", "90 days"), 1.0)
        self.assertEqual(score_case("nope", "90 days", mode="exact"), 0.0)
        self.assertGreater(score_case("kept for 90 days", "90 days", mode="f1"), 0.4)

    def test_harness_mean_on_local_responder(self):
        reg = PromptRegistry()
        prompt = reg.add("note", "Window: {hours}.")
        report = EvalHarness(LocalResponder()).run(
            prompt,
            [{"id": "a", "variables": {"hours": "90 days"}, "expected": "90 days"}],
        )
        self.assertEqual(report.mean, 1.0)
        self.assertEqual(report.prompt_version, 1)

    def test_missing_slot_raises(self):
        with self.assertRaises(KeyError):
            LocalResponder()("Window: {hours}.", {})


class DriftTests(unittest.TestCase):
    def test_identical_windows_do_not_alert(self):
        report = compare([1, 1, 1, 0.9], [1, 1, 0.9, 1])
        self.assertFalse(report.alert)
        self.assertLess(report.psi, 0.25)

    def test_shifted_window_alerts(self):
        report = compare([0.9, 0.95, 1.0, 0.92], [0.1, 0.2, 0.0, 0.15])
        self.assertTrue(report.alert)
        self.assertLess(report.mean_shift, 0)

    def test_psi_zero_when_constant(self):
        self.assertEqual(population_stability([0.5, 0.5], [0.5, 0.5]), 0.0)


class CostTests(unittest.TestCase):
    def test_budget_and_unknown_model(self):
        mon = CostMonitor()
        mon.record("2026-10-16", "sample-local", 1_000_000, 0)
        self.assertAlmostEqual(mon.spend("2026-10-16"), 2.0)
        self.assertTrue(mon.over_budget("2026-10-16", 1.0))
        self.assertFalse(mon.over_budget("2026-10-17", 1.0))
        with self.assertRaises(CostError):
            mon.record("2026-10-16", "missing", 1, 1)
        self.assertEqual(mon.by_model("2026-10-16")["sample-local"], 2.0)


if __name__ == "__main__":
    unittest.main()
