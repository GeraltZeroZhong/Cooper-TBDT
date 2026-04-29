from __future__ import annotations

import unittest

from evopoint_da.docking_eval.metrics import select_top1_and_rank, summarize_delta, summarize_top1


class DockingEvalMetricTests(unittest.TestCase):
    def test_select_top1_by_rank(self) -> None:
        rows = [
            {"target_id": "A", "rank": "2", "rmsd": "3.0"},
            {"target_id": "A", "rank": "1", "rmsd": "1.5"},
            {"target_id": "B", "rank": "1", "rmsd": "2.5"},
        ]
        top1, ranked = select_top1_and_rank(rows, "target_id", "rank", None, "lower_better")
        self.assertEqual([row["rmsd"] for row in top1], ["1.5", "2.5"])
        self.assertEqual(ranked["A"][0]["_ranked_position"], "1")

    def test_summarize_top1_and_delta(self) -> None:
        summary, values = summarize_top1(
            [{"rmsd": "1.0"}, {"rmsd": "3.0"}],
            rmsd_col="rmsd",
            threshold=2.0,
            n_iter=20,
            seed=1,
        )
        self.assertEqual(values, [1.0, 3.0])
        self.assertEqual(summary.n_success, 1)
        self.assertAlmostEqual(summary.success_rate, 0.5)

        delta, cooper_tbdt_values, af2_values, delta_values = summarize_delta(
            [
                {"cooper_tbdt": "-8.0", "af2": "-7.0"},
                {"cooper_tbdt": "-6.0", "af2": "-7.0"},
            ],
            cooper_tbdt_col="cooper_tbdt",
            af2_col="af2",
        )
        self.assertEqual(cooper_tbdt_values, [-8.0, -6.0])
        self.assertEqual(af2_values, [-7.0, -7.0])
        self.assertEqual(delta_values, [-1.0, 1.0])
        self.assertEqual(delta.n_improved, 1)


if __name__ == "__main__":
    unittest.main()
