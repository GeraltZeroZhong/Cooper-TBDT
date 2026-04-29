from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evopoint_da.docking_eval.chem import DockingBox
from evopoint_da.docking_eval.pipeline import (
    DockingPipelineConfig,
    StructureSpec,
    build_pipeline_summary,
    infer_default_structure_specs,
    parse_structure_specs,
    resolve_docking_box,
    run_docking_pipeline,
)
from evopoint_da.docking_eval.vina_runner import parse_vina_pdbqt_scores, write_box_config


class DockingPipelineHelperTests(unittest.TestCase):
    def test_parse_structure_specs(self) -> None:
        specs = parse_structure_specs(["af2=receptor_af2", "cooper_tbdt=receptor_hs"])
        self.assertEqual(specs, [StructureSpec("af2", "receptor_af2"), StructureSpec("cooper_tbdt", "receptor_hs")])

    def test_infer_default_structure_specs(self) -> None:
        rows = [{"target_id": "A", "receptor_af2": "a.pdb", "receptor_cooper_tbdt": "b.pdb"}]
        specs = infer_default_structure_specs(rows)
        self.assertEqual([spec.label for spec in specs], ["af2", "cooper_tbdt"])

    def test_parse_vina_scores(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.pdbqt"
            path.write_text(
                "\n".join(
                    [
                        "MODEL 1",
                        "REMARK VINA RESULT:     -8.200      0.000      0.000",
                        "ENDMDL",
                        "MODEL 2",
                        "REMARK VINA RESULT:     -7.100      1.200      2.000",
                        "ENDMDL",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(parse_vina_pdbqt_scores(path), [-8.2, -7.1])

    def test_resolve_explicit_box(self) -> None:
        cfg = DockingPipelineConfig(
            manifest=Path("manifest.csv"),
            output_dir=Path("out"),
            structures=[StructureSpec("af2", "receptor_af2")],
        )
        row = {
            "center_x": "1",
            "center_y": "2",
            "center_z": "3",
            "size_x": "10",
            "size_y": "11",
            "size_z": "12",
        }
        box, source = resolve_docking_box(row, cfg, Path("."))
        self.assertEqual(box, DockingBox(1.0, 2.0, 3.0, 10.0, 11.0, 12.0))
        self.assertEqual(source, "manifest_explicit")

    def test_write_box_config(self) -> None:
        with TemporaryDirectory() as tmp:
            path = write_box_config(DockingBox(1, 2, 3, 10, 11, 12), Path(tmp) / "box.txt")
            self.assertIn("center_x = 1.000", path.read_text(encoding="utf-8"))

    def test_build_pipeline_summary_compares_success_and_scores(self) -> None:
        structures = [StructureSpec("af2", "receptor_af2"), StructureSpec("cooper_tbdt", "receptor_cooper_tbdt")]
        cfg = DockingPipelineConfig(
            manifest=Path("manifest.csv"),
            output_dir=Path("out"),
            structures=structures,
            bootstrap_iter=20,
        )
        pose_rows = [
            {"target_id": "A", "structure": "af2", "rank": 1, "score": -6.0, "rmsd": 3.0, "pose_valid": 1},
            {"target_id": "A", "structure": "cooper_tbdt", "rank": 1, "score": -8.0, "rmsd": 1.0, "pose_valid": 1},
            {"target_id": "B", "structure": "af2", "rank": 1, "score": -5.0, "rmsd": 2.5, "pose_valid": 1},
            {"target_id": "B", "structure": "cooper_tbdt", "rank": 1, "score": -7.0, "rmsd": 1.5, "pose_valid": 1},
        ]
        score_rows = [
            {"target_id": "A", "score_af2": -6.0, "score_cooper_tbdt": -8.0},
            {"target_id": "B", "score_af2": -5.0, "score_cooper_tbdt": -7.0},
        ]
        summary = build_pipeline_summary(pose_rows, score_rows, structures, cfg, failures=[])
        self.assertEqual(summary["by_structure"]["af2"]["top1_success"]["n_success"], 0)
        self.assertEqual(summary["by_structure"]["cooper_tbdt"]["top1_success"]["n_success"], 2)
        self.assertAlmostEqual(summary["success_comparison"]["cooper_tbdt_minus_af2_success_rate"], 1.0)
        self.assertEqual(summary["delta_score"]["n_improved"], 2)

    def test_dry_run_pipeline_writes_planned_outputs_without_tools(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            receptor = root / "receptor.pdb"
            receptor.write_text(
                "\n".join(
                    [
                        "ATOM      1  N   ALA A   1      11.104  13.207   9.002  1.00 20.00           N",
                        "ATOM      2  CA  ALA A   1      12.560  13.303   9.103  1.00 20.00           C",
                        "ATOM      3  C   ALA A   1      13.074  12.281  10.112  1.00 20.00           C",
                        "ATOM      4  O   ALA A   1      12.401  11.285  10.322  1.00 20.00           O",
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ligand = root / "ligand.sdf"
            ligand.write_text("", encoding="utf-8")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "\n".join(
                    [
                        "target_id,receptor_af2,receptor_cooper_tbdt,ligand_sdf,reference_ligand_sdf,center_x,center_y,center_z,size_x,size_y,size_z",
                        "T1,receptor.pdb,receptor.pdb,ligand.sdf,ligand.sdf,0,0,0,20,20,20",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = DockingPipelineConfig(
                manifest=manifest,
                output_dir=root / "out",
                structures=[StructureSpec("af2", "receptor_af2"), StructureSpec("cooper_tbdt", "receptor_cooper_tbdt")],
                dry_run=True,
            )
            summary = run_docking_pipeline(cfg)
            self.assertEqual(summary["n_pose_rows"], 2)
            self.assertTrue((root / "out" / "poses_all.csv").exists())
            self.assertTrue((root / "out" / "targets" / "T1" / "af2" / "docking.log").exists())


if __name__ == "__main__":
    unittest.main()
