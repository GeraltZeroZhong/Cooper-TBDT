from __future__ import annotations

import os
import pickle
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import freesasa
import numpy as np
import torch
from Bio.PDB import DSSP, MMCIFParser, PDBParser
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from .structure import STANDARD_AA


class PCAReducer:
    def __init__(self, n_components: int = 128):
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, data_list: List[torch.Tensor]) -> None:
        if not data_list:
            raise ValueError("Cannot fit PCA with an empty data list.")
        x = torch.cat(data_list, dim=0).detach().cpu().numpy()
        self.pca.fit(x)
        self.is_fitted = True

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted")
        return torch.from_numpy(self.pca.transform(x.detach().cpu().numpy())).float()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PCA model not found: {path}")
        with open(path, "rb") as f:
            self.pca = pickle.load(f)
        self.is_fitted = True


class ESMFeatureExtractor:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ESM model weights not found: {model_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein
            from esm.tokenization import EsmSequenceTokenizer
        except Exception as e:
            raise RuntimeError(f"ESM package unavailable: {e}") from e
        self.ESMProtein = ESMProtein
        self.tokenizer = EsmSequenceTokenizer()
        self.model = ESMC(tokenizer=self.tokenizer, d_model=1152, n_layers=36, n_heads=18).to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()

    @torch.no_grad()
    def extract_residue_embeddings(self, sequence: str) -> torch.Tensor:
        sequence = sequence[:1022]
        protein = self.ESMProtein(sequence=sequence)
        tokenized = self.model.encode(protein).sequence.unsqueeze(0).to(self.device)
        out = self.model(tokenized)
        return out.embeddings[0, 1:-1].cpu()


def compute_sasa_with_freesasa(structure_path: str) -> Dict[str, float]:
    structure = freesasa.Structure(structure_path)
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()
    per_res = {}
    for chain_id, residues in tqdm(residue_areas.items(), desc="FreeSASA chains", unit="chain"):
        for res_id, residue_area in tqdm(residues.items(), desc=f"Chain {chain_id} residues", unit="res", leave=False):
            key = f"{chain_id}_{str(res_id).strip()}"
            per_res[key] = float(residue_area.total)
    return per_res


AA_MAX_ACC = {
    "ALA": 121.0,
    "ARG": 265.0,
    "ASN": 187.0,
    "ASP": 187.0,
    "CYS": 148.0,
    "GLN": 214.0,
    "GLU": 214.0,
    "GLY": 97.0,
    "HIS": 216.0,
    "ILE": 195.0,
    "LEU": 191.0,
    "LYS": 230.0,
    "MET": 203.0,
    "PHE": 228.0,
    "PRO": 154.0,
    "SER": 143.0,
    "THR": 163.0,
    "TRP": 264.0,
    "TYR": 255.0,
    "VAL": 165.0,
}


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1) + 1e-8
    b1_u = b1 / b1_norm
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_u, v), w)
    return float(np.arctan2(y, x))


def compute_structural_node_features(
    structure_path: str,
    residue_ids: List[str],
    neighbor_radius: float = 10.0,
    surface_sasa_threshold: float = 1.0,
    require_dssp: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute per-residue geometric/structural features aligned to residue_ids."""
    parser = PDBParser(QUIET=True) if structure_path.lower().endswith((".pdb", ".ent")) else MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", structure_path)
    model = next(structure.get_models())

    fs_structure = freesasa.Structure(structure_path)
    fs_result = freesasa.calc(fs_structure)
    residue_areas = fs_result.residueAreas()

    per_res_sasa: Dict[str, float] = {}
    for chain_id, residues in residue_areas.items():
        for res_id, residue_area in residues.items():
            per_res_sasa[f"{chain_id}_{str(res_id).strip()}"] = float(residue_area.total)

    ca_coord_map: Dict[str, np.ndarray] = {}
    resname_map: Dict[str, str] = {}
    backbone_map: Dict[str, Dict[str, np.ndarray]] = {}
    for chain in model:
        for res in chain:
            resname = res.get_resname().strip().upper()
            if res.id[0] != " " or resname not in STANDARD_AA or "CA" not in res:
                continue
            ins_code = res.id[2].strip()
            rid = f"{chain.id}_{res.id[1]}{ins_code}"
            ca_coord_map[rid] = res["CA"].get_coord().astype(np.float32)
            resname_map[rid] = resname
            backbone_map[rid] = {
                atom_name: res[atom_name].get_coord().astype(np.float32)
                for atom_name in ("N", "CA", "C")
                if atom_name in res
            }

    n = len(residue_ids)
    residue_to_index = {rid: i for i, rid in enumerate(residue_ids)}
    sasa = np.zeros((n, 1), dtype=np.float32)
    rsa = np.zeros((n, 1), dtype=np.float32)
    depth = np.zeros((n, 1), dtype=np.float32)
    coord_num = np.zeros((n, 1), dtype=np.float32)
    hse_up = np.zeros((n, 1), dtype=np.float32)
    hse_down = np.zeros((n, 1), dtype=np.float32)
    dihed = np.zeros((n, 6), dtype=np.float32)
    dssp_3 = np.zeros((n, 3), dtype=np.float32)
    dssp_3[:, 2] = 1.0

    ca_coords = np.array(
        [ca_coord_map[rid] if rid in ca_coord_map else np.zeros(3, dtype=np.float32) for rid in residue_ids],
        dtype=np.float32,
    )
    valid_mask = np.array([rid in ca_coord_map for rid in residue_ids], dtype=bool)

    for i, rid in enumerate(residue_ids):
        s = float(per_res_sasa.get(rid, 0.0))
        sasa[i, 0] = s
        max_acc = AA_MAX_ACC.get(resname_map.get(rid, "GLY"), 180.0)
        rsa[i, 0] = np.clip(s / max_acc, 0.0, 1.5)

    surface_indices = [i for i in range(n) if valid_mask[i] and sasa[i, 0] > surface_sasa_threshold]
    if surface_indices:
        surface_tree = cKDTree(ca_coords[surface_indices])
        for i in range(n):
            if not valid_mask[i]:
                continue
            d, _ = surface_tree.query(ca_coords[i], k=1)
            depth[i, 0] = float(d)

    if valid_mask.any():
        tree = cKDTree(ca_coords[valid_mask])
        valid_indices = np.where(valid_mask)[0]
        for i in valid_indices:
            center = ca_coords[i]
            neigh_local = tree.query_ball_point(center, r=neighbor_radius)
            neigh = [valid_indices[j] for j in neigh_local if valid_indices[j] != i]
            coord_num[i, 0] = float(len(neigh))

            if 0 < i < n - 1 and valid_mask[i - 1] and valid_mask[i + 1]:
                axis = ca_coords[i + 1] - ca_coords[i - 1]
            elif i < n - 1 and valid_mask[i + 1]:
                axis = ca_coords[i + 1] - ca_coords[i]
            elif i > 0 and valid_mask[i - 1]:
                axis = ca_coords[i] - ca_coords[i - 1]
            else:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            axis_norm = np.linalg.norm(axis) + 1e-8
            axis_u = axis / axis_norm
            up = 0
            down = 0
            for j in neigh:
                v = ca_coords[j] - center
                if np.dot(v, axis_u) >= 0:
                    up += 1
                else:
                    down += 1
            hse_up[i, 0] = float(up)
            hse_down[i, 0] = float(down)

    for i in range(n):
        rid_i = residue_ids[i]
        bb_i = backbone_map.get(rid_i, {})
        phi = psi = omega = 0.0

        if i > 0:
            rid_prev = residue_ids[i - 1]
            bb_prev = backbone_map.get(rid_prev, {})
            if "C" in bb_prev and "N" in bb_i and "CA" in bb_i and "C" in bb_i:
                phi = _dihedral_angle(bb_prev["C"], bb_i["N"], bb_i["CA"], bb_i["C"])
            if "CA" in bb_prev and "C" in bb_prev and "N" in bb_i and "CA" in bb_i:
                omega = _dihedral_angle(bb_prev["CA"], bb_prev["C"], bb_i["N"], bb_i["CA"])
        if i < n - 1:
            rid_next = residue_ids[i + 1]
            bb_next = backbone_map.get(rid_next, {})
            if "N" in bb_i and "CA" in bb_i and "C" in bb_i and "N" in bb_next:
                psi = _dihedral_angle(bb_i["N"], bb_i["CA"], bb_i["C"], bb_next["N"])

        dihed[i, 0] = np.sin(phi)
        dihed[i, 1] = np.cos(phi)
        dihed[i, 2] = np.sin(psi)
        dihed[i, 3] = np.cos(psi)
        dihed[i, 4] = np.sin(omega)
        dihed[i, 5] = np.cos(omega)

    dssp_tmp_path: Optional[str] = None
    try:
        lower_path = structure_path.lower()
        if lower_path.endswith((".pdb", ".ent")):
            dssp_file_type = "PDB"
            dssp_input_path = structure_path
            with open(structure_path, "r", encoding="utf-8") as handle:
                first_line = handle.readline()
            if not first_line.startswith("HEADER"):
                with NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as tmp:
                    tmp.write("HEADER    HOLOSHIFT DSSP INPUT\n")
                    with open(structure_path, "r", encoding="utf-8") as src:
                        for line in src:
                            tmp.write(line)
                    dssp_tmp_path = tmp.name
                dssp_input_path = dssp_tmp_path
        else:
            dssp_file_type = "MMCIF"
            dssp_input_path = structure_path

        dssp = DSSP(model, dssp_input_path, dssp="mkdssp", file_type=dssp_file_type)
        for dssp_key in dssp.keys():
            chain_id = dssp_key[0]
            resseq = dssp_key[1][1]
            icode = (dssp_key[1][2] or "").strip()
            rid = f"{chain_id}_{resseq}{icode}"
            if rid not in residue_to_index:
                continue
            idx = residue_to_index[rid]
            ss = dssp[dssp_key][2]
            dssp_3[idx, :] = 0.0
            if ss in {"H", "G", "I"}:
                dssp_3[idx, 0] = 1.0
            elif ss in {"E", "B"}:
                dssp_3[idx, 1] = 1.0
            else:
                dssp_3[idx, 2] = 1.0
    except Exception as e:
        message = f"DSSP unavailable for {structure_path}: {e}."
        if require_dssp:
            raise RuntimeError(message) from e
        print(f"[warning] {message} Falling back to coil state.")
    finally:
        if dssp_tmp_path:
            try:
                os.remove(dssp_tmp_path)
            except OSError:
                pass

    return {
        "sasa": torch.from_numpy(sasa),
        "rsa": torch.from_numpy(rsa),
        "residue_depth": torch.from_numpy(depth),
        "coordination_number": torch.from_numpy(coord_num),
        "hse": torch.from_numpy(np.concatenate([hse_up, hse_down], axis=1)),
        "dihedral_sincos": torch.from_numpy(dihed),
        "dssp_3state": torch.from_numpy(dssp_3),
    }
