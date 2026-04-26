from __future__ import annotations

import unittest

from evopoint_da.data.structure import parse_residue_id, select_chain
from evopoint_da.utils.binning import build_bin_ranges, parse_float_edges


class BinningAndStructureTests(unittest.TestCase):
    def test_build_bin_ranges_validates_edges(self) -> None:
        self.assertEqual(
            build_bin_ranges([0.0, 0.5, 1.0]),
            [(0.0, 0.5, "0to0p5"), (0.5, 1.0, "0p5to1"), (1.0, None, "gt1")],
        )
        with self.assertRaises(ValueError):
            build_bin_ranges([0.0, 0.0])

    def test_parse_float_edges(self) -> None:
        self.assertEqual(parse_float_edges("0,1,2.5"), [0.0, 1.0, 2.5])
        with self.assertRaises(ValueError):
            parse_float_edges("1,1")

    def test_parse_residue_id_preserves_underscored_chain(self) -> None:
        self.assertEqual(parse_residue_id("A_B_-12C"), ("A_B", -12, "C"))
        self.assertEqual(parse_residue_id("plain"), ("plain", 0, ""))

    def test_select_chain_chooses_longest_or_requested(self) -> None:
        chains = {
            "A": {"coords": [1, 2]},
            "B": {"coords": [1, 2, 3]},
        }
        self.assertEqual(select_chain(chains)[0], "B")
        self.assertEqual(select_chain(chains, "A")[0], "A")
        with self.assertRaises(ValueError):
            select_chain(chains, "Z")


if __name__ == "__main__":
    unittest.main()
