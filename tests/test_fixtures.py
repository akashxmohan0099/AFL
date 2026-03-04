import unittest

import pandas as pd

import pipeline


class TestFixtures(unittest.TestCase):
    def test_fixture_match_ids_pair_home_away(self):
        fx = pd.read_csv("data/fixtures/round_1_2026.csv")
        fx2 = pipeline._ensure_fixture_match_ids(fx)

        # Expect 2 rows per match (home + away) and a single match_id per match key.
        fx2["_key"] = fx2.apply(
            lambda r: (
                tuple(sorted([str(r["team"]), str(r["opponent"])])),
                str(r["venue"]).strip().lower(),
                str(r["date"]),
            ),
            axis=1,
        )

        for _, g in fx2.groupby("_key", sort=False):
            self.assertEqual(len(g), 2, "Each match should have exactly 2 fixture rows")
            self.assertEqual(g["match_id"].nunique(), 1, "Home/away rows must share match_id")


if __name__ == "__main__":
    unittest.main()

