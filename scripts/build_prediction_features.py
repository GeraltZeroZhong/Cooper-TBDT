#!/usr/bin/env python
"""CLI wrapper for inference-only HoloShift feature construction."""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from evopoint_da.pipeline.build_prediction_features import main  # noqa: E402


if __name__ == "__main__":
    main()
