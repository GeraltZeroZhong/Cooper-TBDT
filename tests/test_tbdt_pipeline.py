from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from evopoint_da.data.alignment import compute_displacement_target
from evopoint_da.data.dataset import EvoPointDataset, build_split_file_lists
from evopoint_da.data.graph import gvp_edge_scalar_dim
from evopoint_da.data.tbdt import (
    REGION_BARREL_CORE,
    REGION_PLUG,
    REGION_TONB_BOX,
    REGION_VOCAB,
    STATE_VOCAB,
    build_region_features,
    get_region_residue_ids,
    state_id,
)
from evopoint_da.models.module import EvoPointLitModule
from evopoint_da.pipeline.eval_tbdt_state import evaluate
from evopoint_da.pipeline.eval_tbdt_classification_curves import evaluate as evaluate_classification_curves
from evopoint_da.pipeline.build_tbdt_template_baselines import PairSample, _transfer_prediction
from evopoint_da.pipeline.build_tbdt_coordinate_baselines import build_baselines as build_coordinate_baselines
from evopoint_da.pipeline.build_tbdt_external_baselines import _anm_mobility, _gnm_mobility, _parse_fpocket_info
from evopoint_da.pipeline.build_tbdt_structure_template_baselines import _parse_foldseek_hits, _parse_usalign_tabular


def _chain(sequence: str, coords: np.ndarray, chain_id: str = "A") -> dict:
    return {
        chain_id: {
            "sequence": sequence,
            "coords": coords.astype(np.float32),
            "plddts": np.full(len(sequence), 90.0, dtype=np.float32),
            "residue_ids": [f"{chain_id}_{i + 1}" for i in range(len(sequence))],
        }
    }


def _strict_graph_fields(n: int) -> dict[str, torch.Tensor]:
    return {
        "plddt": torch.full((n, 1), 80.0, dtype=torch.float32),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.zeros((0, 2), dtype=torch.float32),
        "node_v": torch.zeros((n, 3, 3), dtype=torch.float32),
        "edge_s": torch.zeros((0, gvp_edge_scalar_dim()), dtype=torch.float32),
        "edge_v": torch.zeros((0, 1, 3), dtype=torch.float32),
    }


class TBDTPipelineTests(unittest.TestCase):
    def test_core_selector_aligns_on_barrel_core_but_returns_all_displacements(self) -> None:
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        coords = np.array(
            [[float(i), float((i * i) % 7), float(i % 5)] for i in range(len(sequence))],
            dtype=np.float32,
        )
        translation = np.array([3.0, -2.0, 0.5], dtype=np.float32)
        holo = coords + translation
        holo[10] += np.array([0.0, 2.5, 0.0], dtype=np.float32)

        delta, residue_ids, _aligned, _af2_idx, _holo_idx, chain_id = compute_displacement_target(
            _chain(sequence, coords),
            _chain(sequence, holo),
            alignment_residue_ids={f"A_{i}" for i in range(1, 7)},
        )

        self.assertEqual(chain_id, "A")
        self.assertEqual(residue_ids[10], "A_11")
        np.testing.assert_allclose(delta[:6], np.zeros((6, 3)), atol=1e-5)
        np.testing.assert_allclose(delta[10], np.array([0.0, 2.5, 0.0], dtype=np.float32), atol=1e-5)

    def test_region_annotation_builds_masks_ids_and_weights(self) -> None:
        annotation = {
            "default_chain": "A",
            "regions": {
                REGION_BARREL_CORE: [{"ranges": [[1, 2]]}],
                REGION_PLUG: [{"residues": [3, 4]}],
                REGION_TONB_BOX: [{"residue_ids": ["A_5"]}],
            },
            "eval_regions": [REGION_PLUG, REGION_TONB_BOX],
            "loss_weights": {"default": 0.3, REGION_BARREL_CORE: 0.1, REGION_PLUG: 2.0, REGION_TONB_BOX: 3.0},
        }
        residue_ids = [f"A_{i}" for i in range(1, 7)]

        features = build_region_features(residue_ids, annotation, default_chain="A")

        self.assertEqual(get_region_residue_ids(annotation, REGION_BARREL_CORE, default_chain="A"), {"A_1", "A_2"})
        self.assertEqual(features["region_id"][0], REGION_VOCAB[REGION_BARREL_CORE])
        self.assertEqual(features["region_id"][2], REGION_VOCAB[REGION_PLUG])
        self.assertEqual(features["region_id"][4], REGION_VOCAB[REGION_TONB_BOX])
        np.testing.assert_array_equal(features["eval_mask"], np.array([False, False, True, True, True, False]))
        np.testing.assert_allclose(features["loss_weight"], np.array([0.1, 0.1, 2.0, 2.0, 3.0, 0.3], dtype=np.float32))
        self.assertEqual(state_id("productive_substrate_bound"), STATE_VOCAB["productive_substrate_bound"])

    def test_dataset_loads_tbdt_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            torch.save(
                {
                    "x": torch.ones((3, 4), dtype=torch.float32),
                    "pos": torch.zeros((3, 3), dtype=torch.float32),
                    "y_delta": torch.zeros((3, 3), dtype=torch.float32),
                    **_strict_graph_fields(3),
                    "region_id": torch.tensor([1, 2, 4], dtype=torch.long),
                    "family_id": torch.tensor(2, dtype=torch.long),
                    "state_id": torch.tensor(4, dtype=torch.long),
                    "substrate_id": torch.tensor(2, dtype=torch.long),
                    "loss_weight": torch.tensor([0.1, 2.0, 3.0], dtype=torch.float32),
                    "plug_mask": torch.tensor([False, True, False]),
                    "tonb_box_mask": torch.tensor([False, False, True]),
                    "eval_mask": torch.tensor([False, True, True]),
                },
                root / "sample.pt",
            )

            sample = EvoPointDataset(str(root), split="all")[0]

            self.assertEqual(sample.family_id.item(), 2)
            torch.testing.assert_close(sample.loss_weight, torch.tensor([0.1, 2.0, 3.0]))
            torch.testing.assert_close(sample.plug_mask, torch.tensor([False, True, False]))
            torch.testing.assert_close(sample.eval_mask, torch.tensor([False, True, True]))

    def test_metadata_split_file_lists_use_manifest_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, split in [("a", "train"), ("b", "val"), ("c", "test")]:
                torch.save(
                    {
                        "x": torch.ones((1, 4), dtype=torch.float32),
                        "pos": torch.zeros((1, 3), dtype=torch.float32),
                        "y_delta": torch.zeros((1, 3), dtype=torch.float32),
                        "metadata": {"manifest_row": {"split": split}},
                    },
                    root / f"{name}.pt",
                )

            split_files = build_split_file_lists(
                str(root),
                split_ranges={
                    "train": [0.0, 0.7],
                    "val": [0.7, 0.85],
                    "test": [0.85, 1.0],
                    "all": [0.0, 1.0],
                },
                split_seed=42,
                split_source="metadata",
            )

            self.assertEqual([Path(p).name for p in split_files["train"]], ["a.pt"])
            self.assertEqual([Path(p).name for p in split_files["val"]], ["b.pt"])
            self.assertEqual([Path(p).name for p in split_files["test"]], ["c.pt"])
            self.assertEqual(len(split_files["all"]), 3)

    def test_model_tbdt_conditioning_accepts_graph_and_node_ids(self) -> None:
        batch = Data(
            x=torch.ones((3, 4), dtype=torch.float32),
            pos=torch.zeros((3, 3), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float32),
            node_v=torch.zeros((3, 3, 3), dtype=torch.float32),
            edge_s=torch.zeros((0, gvp_edge_scalar_dim()), dtype=torch.float32),
            edge_v=torch.zeros((0, 1, 3), dtype=torch.float32),
            region_id=torch.tensor([1, 2, 4], dtype=torch.long),
            family_id=torch.tensor([2], dtype=torch.long),
            state_id=torch.tensor([4], dtype=torch.long),
            substrate_id=torch.tensor([2], dtype=torch.long),
        )
        module = EvoPointLitModule(
            in_channels=8,
            base_in_channels=4,
            use_tbdt_conditioning=True,
            condition_embedding_dim=1,
            region_vocab_size=8,
            family_vocab_size=8,
            state_vocab_size=8,
            substrate_vocab_size=8,
            hidden_dim=8,
            num_layers=1,
        )

        pred = module.forward(batch)

        self.assertEqual(tuple(pred.shape), (3, 3))
        self.assertTrue(bool(torch.isfinite(pred).all()))

    def test_eval_tbdt_state_reports_region_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_path = root / "sample.pt"
            out_json = root / "metrics.json"
            torch.save(
                {
                    "y_delta": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32),
                    "plug_mask": torch.tensor([True, False]),
                    "barrel_core_mask": torch.tensor([False, True]),
                },
                sample_path,
            )
            args = argparse.Namespace(
                inputs=[str(sample_path)],
                predictions=None,
                output_json=str(out_json),
                output_csv=None,
                region_json=None,
                include_all_region=True,
            )

            report = evaluate(args)

            plug = report["aggregate_by_region"]["plug"]
            self.assertEqual(plug["n_residues"], 1)
            self.assertEqual(plug["prediction_error_rms"], plug["zero_error_rms"])
            self.assertTrue(out_json.exists())

    def test_eval_tbdt_state_reports_paired_delta_and_derived_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_path = root / "sample.pt"
            pred_dir = root / "pred"
            pred_dir.mkdir()
            out_json = root / "metrics.json"
            paired_csv = root / "paired.csv"
            tonb_csv = root / "tonb.csv"
            torch.save(
                {
                    "pos": torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "y_delta": torch.tensor(
                        [
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "plug_mask": torch.tensor([True, True, True, True, False]),
                    "tonb_box_mask": torch.tensor([True, False, False, False, False]),
                    "barrel_core_mask": torch.tensor([False, False, False, False, True]),
                },
                sample_path,
            )
            torch.save(
                {
                    "pred_delta": torch.tensor(
                        [
                            [0.5, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    )
                },
                pred_dir / "sample.pt",
            )
            args = argparse.Namespace(
                inputs=[str(sample_path)],
                predictions=str(pred_dir),
                output_json=str(out_json),
                output_csv=None,
                region_json=None,
                include_all_region=True,
                direction_threshold=1.0,
                add_derived_regions=True,
                plug_apical_fraction=0.5,
                plug_extension_residues=2,
                bootstrap_iter=20,
                bootstrap_seed=42,
                paired_delta_csv=str(paired_csv),
                tonb_metrics_csv=str(tonb_csv),
                tonb_exposure_threshold=0.5,
            )

            report = evaluate(args)

            self.assertIn("plug_apical_loop", report["aggregate_by_region"])
            self.assertIn("paired_delta_by_region", report)
            self.assertLess(report["paired_delta_by_region"]["plug"]["median_delta_rmsd_method_minus_raw"], 0.0)
            self.assertEqual(report["tonb_state_summary"]["n_targets"], 1)
            self.assertTrue(paired_csv.exists())
            self.assertTrue(tonb_csv.exists())

    def test_template_transfer_rotates_displacement_vectors_into_target_frame(self) -> None:
        seq = "ACDE"
        donor_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        target_pos = donor_pos @ rotation.T
        donor_delta = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float32)
        expected_delta = torch.tensor([[0.0, 1.0, 0.0]] * 4, dtype=torch.float32)
        donor = PairSample(
            path=Path("donor.pt"),
            stem="donor",
            sequence=seq,
            af2_pos=donor_pos,
            y_delta=donor_delta,
            barrel_core_mask=torch.ones(4, dtype=torch.bool),
            metadata={"manifest_row": {"family": "tbdt", "state_label": "apo", "uniprot_id": "D"}},
        )
        target = PairSample(
            path=Path("target.pt"),
            stem="target",
            sequence=seq,
            af2_pos=target_pos,
            y_delta=torch.zeros((4, 3), dtype=torch.float32),
            barrel_core_mask=torch.ones(4, dtype=torch.bool),
            metadata={"manifest_row": {"family": "tbdt", "state_label": "apo", "uniprot_id": "T"}},
        )

        transfer = _transfer_prediction(donor, target)

        self.assertIsNotNone(transfer)
        torch.testing.assert_close(transfer["prediction"], expected_delta, atol=1e-5, rtol=1e-5)
        self.assertEqual(transfer["fit_region"], "barrel_core")

    def test_coordinate_baselines_fit_train_val_and_predict_test_without_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "graphs"
            out_dir = root / "baselines"
            data_dir.mkdir()

            pos = torch.tensor(
                [
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [0.0, 2.0, 1.5],
                ],
                dtype=torch.float32,
            )
            masks = {
                "barrel_core_mask": torch.tensor([True, True, True, False, False, False]),
                "plug_mask": torch.tensor([False, False, False, True, True, False]),
                "tonb_box_mask": torch.tensor([False, False, False, True, False, False]),
                "extracellular_loop_mask": torch.tensor([False, False, False, False, True, False]),
                "substrate_contact_mask": torch.tensor([False, False, False, False, False, True]),
                "eval_mask": torch.tensor([False, False, False, True, True, True]),
            }

            def write_sample(name: str, split: str, plug_value: float) -> None:
                y_delta = torch.zeros((6, 3), dtype=torch.float32)
                y_delta[masks["plug_mask"]] = torch.tensor([plug_value, 0.0, 0.0])
                x = torch.zeros((6, 16), dtype=torch.float32)
                x[:, 15] = 1.0
                torch.save(
                    {
                        "pos": pos,
                        "y_delta": y_delta,
                        "plddt": torch.full((6, 1), 85.0, dtype=torch.float32),
                        "x": x,
                        "metadata": {
                            "manifest_row": {
                                "split": split,
                                "family": "feca",
                                "state_label": "apo",
                                "substrate_class": "none",
                            }
                        },
                        **masks,
                    },
                    data_dir / f"{name}.pt",
                )

            write_sample("train_sample", "train", 1.0)
            write_sample("val_sample", "val", 3.0)
            write_sample("test_sample", "test", 100.0)

            args = argparse.Namespace(
                data_dir=str(data_dir),
                split="test",
                donor_split=["train", "val"],
                split_source="metadata",
                split_seed=42,
                baseline=["global_region_mean", "region_centroid_shift", "barrel_frame_ridge"],
                output_root=str(out_dir),
                report_path=None,
                region_priority="tonb_box,plug_extension_nt,plug_apical_loop,plug,barrel_core,eval,all",
                ridge_alpha=1.0,
                max_prediction_norm=0.0,
                skip_eval=True,
                include_all_region=True,
                add_derived_regions=True,
                plug_apical_fraction=0.5,
                plug_extension_residues=1,
                direction_threshold=1.0,
                tonb_exposure_threshold=1.0,
                bootstrap_iter=20,
                bootstrap_seed=42,
            )

            report = build_coordinate_baselines(args)

            self.assertEqual(report["target_count"], 1)
            self.assertEqual(report["donor_count"], 2)
            mean_pred = torch.load(
                out_dir / "global_region_mean" / "test_sample.pt",
                map_location="cpu",
                weights_only=False,
            )["pred_delta"]
            torch.testing.assert_close(mean_pred[masks["plug_mask"]], torch.tensor([[2.0, 0.0, 0.0]] * 2))

            ridge_pred = torch.load(
                out_dir / "barrel_frame_ridge" / "test_sample.pt",
                map_location="cpu",
                weights_only=False,
            )["pred_delta"]
            self.assertEqual(tuple(ridge_pred.shape), (6, 3))
            self.assertTrue(bool(torch.isfinite(ridge_pred).all()))

    def test_external_score_baseline_classification_and_enm_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_path = root / "sample.pt"
            score_dir = root / "scores"
            out_dir = root / "curves"
            score_dir.mkdir()
            pos = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            torch.save(
                {
                    "pos": pos,
                    "y_delta": torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "eval_mask": torch.tensor([True, True, True, True, True]),
                    "plddt": torch.full((5, 1), 90.0, dtype=torch.float32),
                    "x": torch.zeros((5, 144), dtype=torch.float32),
                },
                sample_path,
            )
            torch.save({"score": torch.tensor([0.0, 0.1, 0.9, 0.8, 0.2], dtype=torch.float32)}, score_dir / "sample.pt")

            gnm = _gnm_mobility(pos, cutoff=2.1, mode_count=2)
            anm = _anm_mobility(pos, cutoff=2.1, mode_count=2)
            self.assertEqual(tuple(gnm.shape), (5,))
            self.assertEqual(tuple(anm.shape), (5,))
            self.assertTrue(bool(torch.isfinite(gnm).all()))
            self.assertTrue(bool(torch.isfinite(anm).all()))

            report = evaluate_classification_curves(
                argparse.Namespace(
                    inputs=[str(sample_path)],
                    sample_list=None,
                    prediction=[],
                    score_baseline=[f"toy_score={score_dir}"],
                    external_baseline=[],
                    region=["eval"],
                    positive_threshold=1.0,
                    sasa_feature_index=129,
                    rsa_feature_index=130,
                    out_dir=str(out_dir),
                    dpi=80,
                )
            )

            toy = next(row for row in report["summary"] if row["method"] == "toy_score")
            self.assertEqual(toy["n_positive"], 2)
            self.assertGreater(toy["average_precision"], 0.9)

    def test_external_tool_output_parsers_rank_and_extract_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            foldseek_hits = root / "hits.tsv"
            foldseek_hits.write_text(
                "\n".join(
                    [
                        "query\tdonor_b\t0.70\t0.60\t0.55\t100\t1e-5",
                        "query\tdonor_a\t0.90\t0.80\t0.75\t200\t1e-20",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            foldseek = _parse_foldseek_hits(foldseek_hits)

            self.assertEqual([hit.donor_stem for hit in foldseek], ["donor_a", "donor_b"])
            self.assertAlmostEqual(foldseek[0].metadata["foldseek_alntmscore"], 0.90)

            usalign = _parse_usalign_tabular(
                "\n".join(
                    [
                        "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali",
                        "target.pdb:1,A\tdonor_b.pdb:1,A\t0.50\t0.40\t3.0\t0.1\t0.1\t0.1\t10\t11\t9",
                        "target.pdb:1,A\tdonor_a.pdb:1,A\t0.80\t0.70\t2.0\t0.2\t0.2\t0.2\t10\t11\t9",
                    ]
                )
            )

            self.assertEqual([hit.donor_stem for hit in usalign], ["donor_a", "donor_b"])
            self.assertAlmostEqual(usalign[0].metadata["usalign_tm_target_norm"], 0.80)

            info = root / "fpocket_info.txt"
            info.write_text(
                "Pocket 1 :\n\tScore : \t0.123\n\tDruggability Score : \t0.456\n\n"
                "Pocket 2 :\n\tScore : \t0.789\n",
                encoding="utf-8",
            )
            fpocket = _parse_fpocket_info(info)

            self.assertAlmostEqual(fpocket[1]["score"], 0.123)
            self.assertAlmostEqual(fpocket[1]["drug_score"], 0.456)
            self.assertAlmostEqual(fpocket[2]["score"], 0.789)


if __name__ == "__main__":
    unittest.main()
