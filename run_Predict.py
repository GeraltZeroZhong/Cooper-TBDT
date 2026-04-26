import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evopoint_da.pipeline.predict import (  # noqa: E402
    _build_auto_features,
    _feature_edges_for_node_count,
    _load_prediction_features,
    _summarize_prediction_bins,
    build_arg_parser,
    predict_and_relax,
    predict_displacement,
)


def get_args():
    return build_arg_parser().parse_args()


if __name__ == "__main__":
    predict_and_relax(get_args())
