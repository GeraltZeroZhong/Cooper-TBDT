from __future__ import annotations

import unittest

import torch

from evopoint_da.models.module import EvoPointLitModule, _as_raw_plddt


class ModelHelperTests(unittest.TestCase):
    def test_as_raw_plddt_accepts_raw_or_normalized_values(self) -> None:
        normalized = _as_raw_plddt(torch.tensor([[0.8], [0.4]]))
        raw = _as_raw_plddt(torch.tensor([80.0, 40.0]))
        torch.testing.assert_close(normalized, raw)

    def test_predict_displacement_does_not_apply_multiplier_by_default(self) -> None:
        module = EvoPointLitModule(
            in_channels=1,
            hidden_dim=4,
            num_layers=1,
            coord_scale=3.0,
            inference_disp_multiplier=2.0,
        )
        module.forward = lambda _batch: torch.ones((2, 3), dtype=torch.float32)

        pred = module.predict_displacement(batch=object())
        legacy_pred = module.predict_displacement(batch=object(), apply_inference_multiplier=True)

        torch.testing.assert_close(pred, torch.full((2, 3), 3.0))
        torch.testing.assert_close(legacy_pred, torch.full((2, 3), 6.0))


if __name__ == "__main__":
    unittest.main()
