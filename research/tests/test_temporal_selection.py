"""Synthetic timing, continuity and selection tests; no historical reads."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from temporal_study.selection import Candidate, CandidatePath, SelectionPlan, select_candidates


def token(number):
    return f"{number:064x}"


class SelectionTests(unittest.TestCase):
    def setUp(self):
        # Intentionally tiny dates for executable guards, not a study proposal.
        self.plan = SelectionPlan(
            protocol_id="synthetic", family_id="temporal", seeds=(1, 2),
            candidates=(Candidate("strong", {"rho": .1}), Candidate("weak", {"rho": .01})),
            initial_decision="2000-12-31", validation_first_return="2001-07-31",
            validation_last_return="2001-12-31", selection_available_at="2001-12-31",
            first_evaluation_decision="2001-12-31", max_available_decision="2001-11-30",
            max_available_outcome="2001-12-31", initial_pretrade_sha256=token(0),
            gamma=10., exposure_label="synthetic previously exposed fixture")
        self.paths = [self.path(seed, candidate.name) for seed in self.plan.seeds for candidate in self.plan.candidates]

    def path(self, seed, candidate):
        dates = pd.date_range("2000-12-31", "2001-11-30", freq="ME")
        offset = 100 * seed + (10 if candidate == "weak" else 20)
        states = [token(0)] + [token(offset + i) for i in range(1, 13)]
        table = pd.DataFrame(dict(
            decision_eom=dates, return_eom=dates + pd.offsets.MonthEnd(1),
            parent_cutoff="2000-12-31", training_last_decision="2000-11-30",
            training_outcomes_available_at="2000-12-31", gross_excess=.012,
            impact_cost=.002, net_excess=.01, pretrade_sha256=states[:-1],
            next_pretrade_sha256=states[1:], holdings_reset=False))
        return CandidatePath(seed, candidate, table, {f"source/{seed}/{candidate}": token(10)}, True)

    def run_selection(self, paths=None, plan=None):
        return select_candidates(plan=plan or self.plan, paths=self.paths if paths is None else paths)

    def test_same_close_selection_and_declared_ties(self):
        result = self.run_selection()
        self.assertEqual([x["candidate"] for x in result["selected_by_seed"]], ["strong", "strong"])
        self.assertAlmostEqual(result["selected_by_seed"][0]["annual_net_utility"], .12)
        self.assertEqual(result["candidate_seed_paths_inspected"], 4)
        self.assertEqual(result["total_candidate_selection_months"], 24)
        self.assertFalse(result["selection_pools_seeds"])
        self.assertFalse(result["selected_configuration_known_at_initial_decision"])
        self.assertEqual(len(result["receipt_sha256"]), 64)

    def test_each_seed_selects_its_own_winner(self):
        self.paths[1].monthly.loc[6:, "gross_excess"] = .022
        self.paths[1].monthly.loc[6:, "net_excess"] = .02
        result = self.run_selection()
        self.assertEqual([x["candidate"] for x in result["selected_by_seed"]], ["weak", "strong"])

    def test_development_returns_do_not_choose_candidate(self):
        before = self.run_selection()
        self.paths[1].monthly.loc[:5, "gross_excess"] = 100.
        self.paths[1].monthly.loc[:5, "net_excess"] = 99.998
        after = self.run_selection()
        self.assertEqual(before["selected_by_seed"], after["selected_by_seed"])
        self.assertNotEqual(before["receipt_sha256"], after["receipt_sha256"])

    def test_future_rows_rejected_before_nonfinite_values_scored(self):
        future = self.paths[0].monthly.iloc[[-1]].copy()
        future["decision_eom"] = pd.Timestamp("2001-12-31")
        future["return_eom"] = pd.Timestamp("2002-01-31")
        future["net_excess"] = np.nan
        table = pd.concat([self.paths[0].monthly, future], ignore_index=True)
        altered = replace(self.paths[0], monthly=table)
        with self.assertRaisesRegex(ValueError, "Future rows"):
            self.run_selection([altered, *self.paths[1:]])

    def test_missing_duplicate_or_out_of_order_month_rejected(self):
        for table in (self.paths[0].monthly.iloc[1:],
                      pd.concat([self.paths[0].monthly, self.paths[0].monthly.iloc[[0]]]),
                      self.paths[0].monthly.iloc[::-1]):
            with self.subTest(length=len(table)):
                with self.assertRaisesRegex(ValueError, "complete ordered continuous"):
                    self.run_selection([replace(self.paths[0], monthly=table), *self.paths[1:]])

    def test_missing_or_duplicate_seed_candidate_rejected(self):
        for paths in (self.paths[:-1], [*self.paths, self.paths[0]]):
            with self.assertRaisesRegex(ValueError, "Exactly every"):
                self.run_selection(paths)

    def test_new_initial_holdings_or_reset_rejected(self):
        self.paths[0].monthly.loc[0, "pretrade_sha256"] = token(9)
        with self.assertRaisesRegex(ValueError, "same accepted holdings"):
            self.run_selection()
        self.paths[0].monthly.loc[0, "pretrade_sha256"] = token(0)
        self.paths[0].monthly.loc[1, "holdings_reset"] = True
        with self.assertRaisesRegex(ValueError, "resets"):
            self.run_selection()

    def test_broken_own_path_continuity_rejected(self):
        self.paths[0].monthly.loc[4, "pretrade_sha256"] = token(99)
        with self.assertRaisesRegex(ValueError, "carry continuously"):
            self.run_selection()

    def test_later_parent_or_unmatured_training_rejected(self):
        self.paths[0].monthly.loc[0, "training_outcomes_available_at"] = "2001-01-31"
        with self.assertRaisesRegex(ValueError, "mature"):
            self.run_selection()
        self.paths[0].monthly.loc[0, "training_outcomes_available_at"] = "2000-12-31"
        self.paths[0].monthly.loc[11, "parent_cutoff"] = "2001-11-30"
        with self.assertRaisesRegex(ValueError, "exact archived annual parent"):
            self.run_selection()

    def test_evaluation_cannot_precede_selection_availability(self):
        with self.assertRaisesRegex(ValueError, "available before"):
            replace(self.plan, first_evaluation_decision="2001-11-30")
        with self.assertRaisesRegex(ValueError, "available before"):
            replace(self.plan, max_available_outcome="2001-11-30")

    def test_risk_and_costs_enter_selection_criterion(self):
        weak = self.paths[1].monthly
        weak.loc[6:, "net_excess"] = [-.19, .21] * 3
        weak.loc[6:, "gross_excess"] = weak.loc[6:, "net_excess"] + weak.loc[6:, "impact_cost"]
        result = self.run_selection()
        self.assertEqual(result["selected_by_seed"][0]["candidate"], "strong")
        self.assertAlmostEqual(result["selected_by_seed"][0]["candidate_scores"][1]["annual_net_utility"], -2.28)
        weak.loc[6:, "gross_excess"] = .017
        weak.loc[6:, "impact_cost"] = .009
        weak.loc[6:, "net_excess"] = .008
        self.assertEqual(self.run_selection()["selected_by_seed"][0]["candidate"], "strong")

    def test_accounting_and_first_trade_required(self):
        self.paths[0].monthly.loc[1, "net_excess"] = 2.
        with self.assertRaisesRegex(ValueError, "reconcile"):
            self.run_selection()
        with self.assertRaisesRegex(ValueError, "initial trade"):
            replace(self.paths[0], first_trade_charged=False)

    def test_input_order_independent_and_content_hashes_detect_changes(self):
        first = self.run_selection()
        second = self.run_selection(list(reversed(self.paths)))
        self.assertEqual(first, second)
        table = self.paths[0].monthly
        table.loc[0, "next_pretrade_sha256"] = token(991)
        table.loc[1, "pretrade_sha256"] = token(991)
        third = self.run_selection()
        self.assertEqual(first["selected_by_seed"], third["selected_by_seed"])
        self.assertNotEqual(first["receipt_sha256"], third["receipt_sha256"])

    def test_duplicate_configuration_and_bad_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "configurations must be unique"):
            replace(self.plan, candidates=(Candidate("one", {"rho": .1}), Candidate("two", {"rho": .1})))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            replace(self.paths[0], artifact_sha256={"source": "unverified"})
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(self.plan, gamma=0.)

    def test_container_and_monthly_seed_labels_must_agree(self):
        self.paths[0].monthly["seed"] = 2
        with self.assertRaisesRegex(ValueError, "seed labels"):
            self.run_selection()

    def test_wrong_method_cannot_be_relabelled_as_another_candidate(self):
        self.paths[0].monthly["method"] = "weak"
        with self.assertRaisesRegex(ValueError, "method labels"):
            self.run_selection()

    def test_continuous_path_runner_outputs_feed_selector(self):
        from feature_geometry.features import fingerprint
        from temporal_study.paths import run_paths
        from tests.test_temporal_paths import PathTests
        from threadpoolctl import threadpool_limits
        source, initial, policies, options = PathTests().problem()
        with threadpool_limits(limits=1):
            result = run_paths(source, source.dates, policies, initial, **options)
        plan = replace(self.plan, candidates=(Candidate("parent", {"fixture": 1.}),
                                               Candidate("learned", {"fixture": 2.})),
                       initial_decision="2000-11-30", validation_first_return="2001-01-31",
                       validation_last_return="2001-02-28", selection_available_at="2001-02-28",
                       first_evaluation_decision="2001-02-28", max_available_decision="2001-01-31",
                       max_available_outcome="2001-02-28",
                       initial_pretrade_sha256=fingerprint(initial.ids, initial.holdings))
        paths = []
        for seed in plan.seeds:
            for candidate in plan.candidates:
                frame = result["monthly"]
                frame = frame[(frame.seed == seed) & (frame.method == candidate.name)]
                paths.append(CandidatePath(seed, candidate.name, frame,
                                           {"synthetic_runner_receipt": token(1)},
                                           result["manifest"]["first_trade_charged"]))
        selected = select_candidates(plan=plan, paths=paths)
        self.assertEqual(selected["candidate_seed_paths_inspected"], 4)
        self.assertTrue(all(row["complete_path_months"] == 3 for row in selected["candidate_exposure"]))


if __name__ == "__main__":
    unittest.main()
