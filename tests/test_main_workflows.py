from __future__ import annotations

import unittest

from main import DEFAULT_BASELINE_CHECKPOINT, _download_benchmark_args, _prediction_args, _workflow_parser


class MainWorkflowTests(unittest.TestCase):
    def test_download_benchmark_defaults_to_all_tiers_with_pae(self) -> None:
        args = _workflow_parser("download_benchmark").parse_args([])

        cmd = _download_benchmark_args(args)

        self.assertIn("--download-pae", cmd)
        self.assertIn("--sync-tier-manifests", cmd)
        self.assertEqual(cmd.count("--tier"), 3)
        self.assertIn("gold", cmd)
        self.assertIn("silver", cmd)
        self.assertIn("bronze", cmd)

    def test_download_benchmark_can_select_gold_only(self) -> None:
        args = _workflow_parser("download_benchmark").parse_args(["--tier", "gold"])

        cmd = _download_benchmark_args(args)

        self.assertEqual(cmd.count("--tier"), 1)
        self.assertIn("gold", cmd)
        self.assertNotIn("silver", cmd)
        self.assertNotIn("bronze", cmd)

    def test_baseline_predict_forwards_prediction_only_arguments(self) -> None:
        args = _workflow_parser("baseline_predict").parse_args(
            [
                "--data-dir",
                "graphs",
                "--output-dir",
                "preds",
                "--split",
                "test",
            ]
        )

        cmd = _prediction_args(args, args.ckpt)

        self.assertEqual(cmd[cmd.index("--ckpt") + 1], str(DEFAULT_BASELINE_CHECKPOINT))
        self.assertEqual(cmd[cmd.index("--data-dir") + 1], "graphs")
        self.assertEqual(cmd[cmd.index("--output-dir") + 1], "preds")
        self.assertNotIn("--predictions", cmd)

    def test_baseline_predict_allows_checkpoint_override(self) -> None:
        args = _workflow_parser("baseline_predict").parse_args(["--ckpt", "checkpoints/custom.ckpt"])

        cmd = _prediction_args(args, args.ckpt)

        self.assertEqual(cmd[cmd.index("--ckpt") + 1], "checkpoints/custom.ckpt")


if __name__ == "__main__":
    unittest.main()
