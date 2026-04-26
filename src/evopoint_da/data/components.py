"""Backward-compatible imports for data utilities.

New code should import directly from the focused modules:
`structure`, `alignment`, `features`, or `graph`.
"""

from .alignment import apply_transform, compute_displacement_target, iterative_kabsch, kabsch_rotation
from .features import (
    AA_MAX_ACC,
    ESMFeatureExtractor,
    PCAReducer,
    compute_sasa_with_freesasa,
    compute_structural_node_features,
)
from .graph import (
    build_ca_node_vectors,
    build_gvp_edge_features,
    build_gvp_graph_features,
    build_knn_edges,
    parse_pae_matrix,
)
from .structure import STANDARD_AA, StructureParser, format_residue_id, parse_residue_id, select_chain

__all__ = [
    "AA_MAX_ACC",
    "ESMFeatureExtractor",
    "PCAReducer",
    "STANDARD_AA",
    "StructureParser",
    "apply_transform",
    "build_ca_node_vectors",
    "build_gvp_edge_features",
    "build_gvp_graph_features",
    "build_knn_edges",
    "compute_displacement_target",
    "compute_sasa_with_freesasa",
    "compute_structural_node_features",
    "format_residue_id",
    "iterative_kabsch",
    "kabsch_rotation",
    "parse_pae_matrix",
    "parse_residue_id",
    "select_chain",
]
