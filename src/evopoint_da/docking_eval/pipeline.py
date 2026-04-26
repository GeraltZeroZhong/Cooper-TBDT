from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chem import DockingBox, compute_pose_rmsds, infer_box_from_ligand_sdf, prepare_ligand_sdf, split_sdf_poses
from .io_utils import read_table, write_csv
from .metrics import (
    compute_first_hit_stats,
    per_target_pose_metrics,
    select_top1_and_rank,
    summarize_delta,
    summarize_top1,
    summarize_topn_success,
    summarize_topn_success_and_valid,
)
from .vina_runner import (
    export_docking_results_with_meeko,
    infer_box_from_bound_heterogen_pdb,
    parse_vina_pdbqt_scores,
    prepare_ligand_with_meeko,
    prepare_receptor_with_meeko,
    run_vina,
    write_box_config,
)


@dataclass(frozen=True)
class StructureSpec:
    label: str
    receptor_col: str


@dataclass
class DockingPipelineConfig:
    manifest: Path
    output_dir: Path
    structures: list[StructureSpec]
    target_col: str = "target_id"
    ligand_col: str = "ligand_sdf"
    reference_ligand_col: str = "reference_ligand_sdf"
    box_source_pdb_col: str = "box_source_pdb"
    rmsd_threshold: float = 2.0
    topn_levels: list[int] | None = None
    box_padding_angstrom: float = 8.0
    box_min_size_angstrom: float = 16.0
    exhaustiveness: int = 8
    num_modes: int = 9
    energy_range: float = 3.0
    vina_seed: int = 20260408
    ligand_seed: int = 42
    bootstrap_iter: int = 2000
    bootstrap_seed: int = 42
    reuse: bool = False
    skip_failed: bool = False
    dry_run: bool = False
    limit: int | None = None


def parse_structure_specs(specs: list[str]) -> list[StructureSpec]:
    parsed: list[StructureSpec] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Structure spec must be LABEL=COLUMN, got: {spec}")
        label, col = spec.split("=", 1)
        label = label.strip()
        col = col.strip()
        if not label or not col:
            raise ValueError(f"Structure spec must be LABEL=COLUMN, got: {spec}")
        parsed.append(StructureSpec(label=label, receptor_col=col))
    if not parsed:
        raise ValueError("At least one structure spec is required.")
    return parsed


def infer_default_structure_specs(rows: list[dict[str, str]]) -> list[StructureSpec]:
    if not rows:
        raise ValueError("Manifest is empty.")
    columns = set(rows[0].keys())
    candidates = {
        "af2": ["receptor_af2", "af2_receptor", "af2_pdb", "receptor_af2_pdb", "af2"],
        "holoshift": [
            "receptor_holoshift",
            "holoshift_receptor",
            "holoshift_pdb",
            "receptor_holoshift_pdb",
            "holoshift",
        ],
    }
    specs = []
    for label, names in candidates.items():
        found = next((name for name in names if name in columns), None)
        if found:
            specs.append(StructureSpec(label=label, receptor_col=found))
    if specs:
        return specs
    raise ValueError(
        "Could not infer receptor columns. Pass --structure af2=receptor_af2 "
        "--structure holoshift=receptor_holoshift, or use matching manifest column names."
    )


def _safe_path_part(value: str) -> str:
    text = str(value).strip() or "target"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _nonempty(row: dict[str, str], key: str | None) -> str | None:
    if key is None or key not in row:
        return None
    value = str(row.get(key, "")).strip()
    return value or None


def _resolve_path(raw: str | None, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _box_from_explicit_columns(row: dict[str, str]) -> DockingBox | None:
    aliases = [
        ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"),
        ("search_center_x", "search_center_y", "search_center_z", "search_size_x", "search_size_y", "search_size_z"),
        ("box_center_x", "box_center_y", "box_center_z", "box_size_x", "box_size_y", "box_size_z"),
    ]
    for keys in aliases:
        values = [_float_or_none(row.get(key)) for key in keys]
        if all(value is not None for value in values):
            cx, cy, cz, sx, sy, sz = [float(value) for value in values if value is not None]
            return DockingBox(cx, cy, cz, sx, sy, sz)
    return None


def resolve_docking_box(row: dict[str, str], cfg: DockingPipelineConfig, manifest_dir: Path) -> tuple[DockingBox, str]:
    explicit = _box_from_explicit_columns(row)
    if explicit is not None:
        return explicit, "manifest_explicit"

    reference_raw = _nonempty(row, cfg.reference_ligand_col)
    reference_path = _resolve_path(reference_raw, manifest_dir)
    if reference_path is not None and reference_path.exists():
        return (
            infer_box_from_ligand_sdf(
                reference_path,
                padding_angstrom=cfg.box_padding_angstrom,
                min_size_angstrom=cfg.box_min_size_angstrom,
            ),
            "reference_ligand_sdf",
        )

    source_raw = _nonempty(row, cfg.box_source_pdb_col)
    source_path = _resolve_path(source_raw, manifest_dir)
    if source_path is not None and source_path.exists():
        return (
            infer_box_from_bound_heterogen_pdb(
                source_path,
                padding_angstrom=cfg.box_padding_angstrom,
                min_size_angstrom=cfg.box_min_size_angstrom,
            ),
            "bound_heterogen_pdb",
        )

    raise ValueError(
        "No docking box could be resolved. Provide center/size columns, "
        "a reference ligand SDF, or a box_source_pdb with bound ligand."
    )


def _row_base(
    *,
    target_id: str,
    structure_label: str,
    receptor_pdb: Path,
    ligand_sdf: Path,
    reference_ligand_sdf: Path | None,
    box: DockingBox,
    box_source: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target_id": target_id,
        "structure": structure_label,
        "receptor_pdb": str(receptor_pdb),
        "ligand_sdf": str(ligand_sdf),
        "reference_ligand_sdf": str(reference_ligand_sdf) if reference_ligand_sdf else "",
        "box_source": box_source,
    }
    row.update({f"box_{k}": v for k, v in box.as_dict().items()})
    return row


def _run_structure_docking(
    *,
    target_id: str,
    structure: StructureSpec,
    receptor_pdb: Path,
    ligand_pdbqt: Path,
    ligand_sdf: Path,
    reference_ligand_sdf: Path | None,
    box: DockingBox,
    box_source: str,
    target_dir: Path,
    cfg: DockingPipelineConfig,
) -> list[dict[str, Any]]:
    structure_dir = target_dir / structure.label
    structure_dir.mkdir(parents=True, exist_ok=True)
    log_path = structure_dir / "docking.log"
    receptor_pdbqt = structure_dir / "prepared_receptor.pdbqt"
    receptor_json = structure_dir / "prepared_receptor.json"
    docking_pdbqt = structure_dir / "docking_results.pdbqt"
    docking_sdf = structure_dir / "docking_results.sdf"
    pose_dir = structure_dir / "poses"

    write_box_config(box, structure_dir / "docking_box.txt")

    if not cfg.reuse or not receptor_pdbqt.exists():
        prepare_receptor_with_meeko(
            receptor_pdb,
            receptor_pdbqt,
            receptor_json,
            log_path,
            dry_run=cfg.dry_run,
        )
    if not cfg.reuse or not docking_pdbqt.exists():
        run_vina(
            receptor_pdbqt,
            ligand_pdbqt,
            box,
            docking_pdbqt,
            log_path,
            exhaustiveness=cfg.exhaustiveness,
            num_modes=cfg.num_modes,
            energy_range=cfg.energy_range,
            seed=cfg.vina_seed,
            dry_run=cfg.dry_run,
        )

    base = _row_base(
        target_id=target_id,
        structure_label=structure.label,
        receptor_pdb=receptor_pdb,
        ligand_sdf=ligand_sdf,
        reference_ligand_sdf=reference_ligand_sdf,
        box=box,
        box_source=box_source,
    )
    base.update(
        {
            "receptor_pdbqt": str(receptor_pdbqt),
            "receptor_json": str(receptor_json),
            "docking_pdbqt": str(docking_pdbqt),
            "docking_sdf": str(docking_sdf),
            "docking_log": str(log_path),
        }
    )

    if cfg.dry_run:
        return [{**base, "rank": "", "score": "", "rmsd": "", "pose_sdf": "", "pose_valid": 0, "status": "dry_run"}]

    if not docking_pdbqt.exists() or docking_pdbqt.stat().st_size == 0:
        raise RuntimeError(f"Vina produced no pose file: {docking_pdbqt}")

    scores = parse_vina_pdbqt_scores(docking_pdbqt)
    if not cfg.reuse or not docking_sdf.exists():
        export_docking_results_with_meeko(docking_pdbqt, docking_sdf, log_path)

    pose_paths = split_sdf_poses(docking_sdf, pose_dir, prefix=f"{target_id}_{structure.label}_pose")
    rmsds: list[float | None] = []
    if reference_ligand_sdf is not None:
        rmsds = list(compute_pose_rmsds(docking_sdf, reference_ligand_sdf))

    n_rows = max(len(scores), len(pose_paths), len(rmsds))
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        score = scores[idx] if idx < len(scores) else None
        rmsd = rmsds[idx] if idx < len(rmsds) else None
        pose_path = pose_paths[idx] if idx < len(pose_paths) else None
        rows.append(
            {
                **base,
                "rank": idx + 1,
                "score": score if score is not None else "",
                "rmsd": rmsd if rmsd is not None and not math.isnan(float(rmsd)) else "",
                "success_at_threshold": (
                    int(float(rmsd) < cfg.rmsd_threshold)
                    if rmsd is not None and not math.isnan(float(rmsd))
                    else ""
                ),
                "pose_sdf": str(pose_path) if pose_path else "",
                "pose_valid": int(pose_path is not None and pose_path.exists()),
                "status": "ok",
                "receptor_pdbqt": str(receptor_pdbqt),
                "receptor_json": str(receptor_json),
                "docking_pdbqt": str(docking_pdbqt),
                "docking_sdf": str(docking_sdf),
                "docking_log": str(log_path),
            }
        )
    return rows


def _prepare_target_ligand(
    *,
    target_dir: Path,
    ligand_sdf: Path,
    cfg: DockingPipelineConfig,
) -> tuple[Path, Path]:
    ligand_dir = target_dir / "ligand"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    prepared_sdf = ligand_dir / "prepared_ligand.sdf"
    ligand_pdbqt = ligand_dir / "prepared_ligand.pdbqt"
    log_path = ligand_dir / "ligand_prep.log"

    if cfg.dry_run:
        prepare_ligand_with_meeko(ligand_sdf, ligand_pdbqt, log_path, dry_run=True)
        return ligand_sdf, ligand_pdbqt

    if not cfg.reuse or not prepared_sdf.exists():
        prepare_ligand_sdf(ligand_sdf, prepared_sdf, random_seed=cfg.ligand_seed)
    if not cfg.reuse or not ligand_pdbqt.exists():
        prepare_ligand_with_meeko(prepared_sdf, ligand_pdbqt, log_path)
    return prepared_sdf, ligand_pdbqt


def _top1_by_structure(pose_rows: list[dict[str, Any]], structures: list[StructureSpec]) -> dict[str, dict[str, Any]]:
    top1: dict[str, dict[str, Any]] = {}
    for structure in structures:
        rows = [row for row in pose_rows if row.get("structure") == structure.label and row.get("rank") == 1]
        top1[structure.label] = {str(row["target_id"]): row for row in rows if row.get("target_id") is not None}
    return top1


def _build_score_rows(pose_rows: list[dict[str, Any]], structures: list[StructureSpec]) -> list[dict[str, Any]]:
    by_structure = _top1_by_structure(pose_rows, structures)
    target_ids = sorted({str(row.get("target_id", "")) for row in pose_rows if str(row.get("target_id", "")).strip()})
    score_rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        row: dict[str, Any] = {"target_id": target_id}
        for structure in structures:
            top = by_structure.get(structure.label, {}).get(target_id)
            if not top:
                continue
            row[f"score_{structure.label}"] = top.get("score", "")
            row[f"top1_rmsd_{structure.label}"] = top.get("rmsd", "")
            row[f"top1_success_{structure.label}"] = top.get("success_at_threshold", "")
        if "score_holoshift" in row and "score_af2" in row:
            hs = _float_or_none(row["score_holoshift"])
            af2 = _float_or_none(row["score_af2"])
            row["delta_score"] = (hs - af2) if hs is not None and af2 is not None else ""
        score_rows.append(row)
    return score_rows


def _pose_rows_for_metrics(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in rows if str(row.get("rmsd", "")).strip()]


def build_pipeline_summary(
    pose_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    structures: list[StructureSpec],
    cfg: DockingPipelineConfig,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    topn_levels = cfg.topn_levels or [1, 2, 3, 5, 10]
    summary: dict[str, Any] = {
        "meta": {
            "manifest": str(cfg.manifest),
            "output_dir": str(cfg.output_dir),
            "rmsd_threshold": cfg.rmsd_threshold,
            "topn_levels": topn_levels,
            "score_direction": "lower_better",
            "formula": "delta_score = score_holoshift - score_af2",
            "interpretation": "lower Vina score is better; delta < 0 means HoloShift improved over AF2",
            "structures": [asdict(structure) for structure in structures],
        },
        "by_structure": {},
        "n_pose_rows": len(pose_rows),
        "n_score_rows": len(score_rows),
        "failures": failures,
    }

    for structure in structures:
        structure_rows = _pose_rows_for_metrics([row for row in pose_rows if row.get("structure") == structure.label])
        if not structure_rows:
            summary["by_structure"][structure.label] = {"n_pose_rows": 0}
            continue
        top1_rows, ranked_by_target = select_top1_and_rank(
            structure_rows,
            target_col="target_id",
            rank_col="rank",
            pose_score_col="score",
            score_direction="lower_better",
        )
        top1_summary, rmsd_values = summarize_top1(
            top1_rows=top1_rows,
            rmsd_col="rmsd",
            threshold=cfg.rmsd_threshold,
            n_iter=cfg.bootstrap_iter,
            seed=cfg.bootstrap_seed,
        )
        target_metrics = per_target_pose_metrics(
            ranked_by_target=ranked_by_target,
            target_col="target_id",
            rmsd_col="rmsd",
            topn_levels=topn_levels,
            threshold=cfg.rmsd_threshold,
            pose_score_col="score",
            pose_valid_col="pose_valid",
        )
        summary["by_structure"][structure.label] = {
            "top1_success": asdict(top1_summary),
            "topn_success": summarize_topn_success(
                ranked_by_target=ranked_by_target,
                rmsd_col="rmsd",
                threshold=cfg.rmsd_threshold,
                topn_levels=topn_levels,
            ),
            "first_hit_rank": compute_first_hit_stats(
                ranked_by_target=ranked_by_target,
                rmsd_col="rmsd",
                threshold=cfg.rmsd_threshold,
                topn_levels=topn_levels,
            ),
            "topn_success_and_valid": summarize_topn_success_and_valid(
                ranked_by_target=ranked_by_target,
                rmsd_col="rmsd",
                valid_col="pose_valid",
                threshold=cfg.rmsd_threshold,
                topn_levels=topn_levels,
            ),
            "top1_success_curve": {
                f"rmsd_lt_{th:.1f}A": (
                    sum(1 for x in rmsd_values if x < th) / len(rmsd_values) if rmsd_values else float("nan")
                )
                for th in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            },
            "n_target_metrics_rows": len(target_metrics),
        }

    score_rows_str = [{k: str(v) for k, v in row.items()} for row in score_rows]
    if score_rows_str and any("score_holoshift" in row and "score_af2" in row for row in score_rows_str):
        delta_summary, hs_values, af2_values, delta_values = summarize_delta(
            score_rows_str,
            hs_col="score_holoshift",
            af2_col="score_af2",
        )
        summary["delta_score"] = asdict(delta_summary)
        summary["delta_score_extra"] = {
            "n_holoshift_better": sum(1 for h, a in zip(hs_values, af2_values) if h < a),
            "n_af2_better": sum(1 for h, a in zip(hs_values, af2_values) if h > a),
            "n_tied": sum(1 for h, a in zip(hs_values, af2_values) if h == a),
        }

    if "af2" in summary["by_structure"] and "holoshift" in summary["by_structure"]:
        af2_success = summary["by_structure"]["af2"].get("top1_success", {}).get("success_rate")
        hs_success = summary["by_structure"]["holoshift"].get("top1_success", {}).get("success_rate")
        if af2_success is not None and hs_success is not None:
            summary["success_comparison"] = {
                "holoshift_minus_af2_success_rate": float(hs_success) - float(af2_success),
                "af2_success_rate": af2_success,
                "holoshift_success_rate": hs_success,
            }

    return summary


def pipeline_summary_to_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Docking Power / Pose Prediction Pipeline Report", "", "## Meta"]
    for key, value in summary.get("meta", {}).items():
        if key == "structures":
            continue
        lines.append(f"- **{key}**: {value}")

    by_structure = summary.get("by_structure", {})
    if isinstance(by_structure, dict) and by_structure:
        lines += ["", "## Top-1 Success By Structure"]
        for label, payload in by_structure.items():
            top1 = payload.get("top1_success", {}) if isinstance(payload, dict) else {}
            if not top1:
                lines.append(f"- {label}: no valid pose RMSD rows")
                continue
            lines.append(
                f"- {label}: {top1['success_rate_percent']:.2f}% "
                f"({top1['n_success']}/{top1['n_targets']}), mean RMSD={top1['rmsd_mean']:.3f} A"
            )

    if "success_comparison" in summary:
        cmp_payload = summary["success_comparison"]
        lines += [
            "",
            "## Success Comparison",
            f"- HoloShift - AF2 success rate: {cmp_payload['holoshift_minus_af2_success_rate']:.4f}",
        ]

    if "delta_score" in summary:
        delta = summary["delta_score"]
        lines += [
            "",
            "## Delta Vina Score",
            f"- Improvement rate (delta<0): {delta['improvement_rate_percent']:.2f}% "
            f"({delta['n_improved']}/{delta['n_targets']})",
            f"- mean delta: {delta['mean_delta']:.4f}",
        ]

    failures = summary.get("failures", [])
    if failures:
        lines += ["", "## Failures", f"- {len(failures)} target/structure failures recorded."]
    return "\n".join(lines) + "\n"


def run_docking_pipeline(cfg: DockingPipelineConfig) -> dict[str, Any]:
    rows = read_table(cfg.manifest)
    if cfg.limit is not None:
        rows = rows[: cfg.limit]
    if not rows:
        raise ValueError("Manifest is empty.")

    manifest_dir = cfg.manifest.parent
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pose_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in rows:
        target_id = _nonempty(row, cfg.target_col)
        if not target_id:
            raise ValueError(f"Manifest row missing target id column {cfg.target_col!r}: {row}")
        target_dir = cfg.output_dir / "targets" / _safe_path_part(target_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            reference_raw = _nonempty(row, cfg.reference_ligand_col)
            reference_ligand_sdf = _resolve_path(reference_raw, manifest_dir)
            ligand_raw = _nonempty(row, cfg.ligand_col)
            ligand_sdf = _resolve_path(ligand_raw, manifest_dir)
            if ligand_sdf is None:
                raise ValueError("No ligand SDF was provided.")
            if reference_ligand_sdf is None:
                raise ValueError("No reference ligand SDF was provided.")
            if not cfg.dry_run and not ligand_sdf.exists():
                raise FileNotFoundError(f"Ligand SDF not found: {ligand_sdf}")
            if not reference_ligand_sdf.exists():
                raise FileNotFoundError(f"Reference ligand SDF not found: {reference_ligand_sdf}")

            box, box_source = resolve_docking_box(row, cfg, manifest_dir)
            prepared_ligand_sdf, ligand_pdbqt = _prepare_target_ligand(
                target_dir=target_dir,
                ligand_sdf=ligand_sdf,
                cfg=cfg,
            )

            for structure in cfg.structures:
                receptor_raw = _nonempty(row, structure.receptor_col)
                receptor_pdb = _resolve_path(receptor_raw, manifest_dir)
                if receptor_pdb is None:
                    raise ValueError(f"Missing receptor column {structure.receptor_col!r}.")
                if not cfg.dry_run and not receptor_pdb.exists():
                    raise FileNotFoundError(f"Receptor PDB not found: {receptor_pdb}")
                try:
                    pose_rows.extend(
                        _run_structure_docking(
                            target_id=target_id,
                            structure=structure,
                            receptor_pdb=receptor_pdb,
                            ligand_pdbqt=ligand_pdbqt,
                            ligand_sdf=prepared_ligand_sdf,
                            reference_ligand_sdf=reference_ligand_sdf,
                            box=box,
                            box_source=box_source,
                            target_dir=target_dir,
                            cfg=cfg,
                        )
                    )
                except Exception as exc:
                    failure = {
                        "target_id": target_id,
                        "structure": structure.label,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    if not cfg.skip_failed:
                        raise
        except Exception as exc:
            failure = {"target_id": target_id, "structure": "", "error": str(exc)}
            failures.append(failure)
            if not cfg.skip_failed:
                raise

    score_rows = _build_score_rows(pose_rows, cfg.structures)
    summary = build_pipeline_summary(pose_rows, score_rows, cfg.structures, cfg, failures)

    write_csv(cfg.output_dir / "poses_all.csv", pose_rows)
    for structure in cfg.structures:
        write_csv(
            cfg.output_dir / f"poses_{structure.label}.csv",
            [row for row in pose_rows if row.get("structure") == structure.label],
        )
    write_csv(cfg.output_dir / "scores.csv", score_rows)
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (cfg.output_dir / "summary.md").write_text(pipeline_summary_to_markdown(summary), encoding="utf-8")
    return summary
