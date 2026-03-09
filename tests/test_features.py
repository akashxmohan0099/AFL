import unittest

import numpy as np
import pandas as pd

import features


class TestFeatures(unittest.TestCase):
    def test_shifted_rolling_slope_matches_polyfit_behavior(self):
        series = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.0])

        expected = series.shift(1).rolling(5, min_periods=3).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )
        actual = features._rolling_linear_slope_shifted(series, window=5, min_periods=3)

        pd.testing.assert_series_equal(actual, expected)

    def test_rolling_features_use_history_for_synthetic_row(self):
        # Minimal schema for add_rolling_features roll_cols.
        rows = [
            # Historical matches
            {"player": "Player A", "team": "Team X", "year": 2025, "date": "2025-01-01", "KI": 10, "HB": 10, "MK": 5, "DI": 20, "GL": 1, "BH": 0, "HO": 0, "TK": 3, "RB": 2, "IF": 2, "CL": 4, "CG": 0, "FF": 0, "FA": 0, "BR": 0, "CP": 10, "UP": 0, "CM": 0, "MI": 1, "one_pct": 0, "BO": 0, "GA": 1, "pct_played": 80, "did_not_play": False},
            {"player": "Player A", "team": "Team X", "year": 2025, "date": "2025-01-08", "KI": 9, "HB": 9, "MK": 4, "DI": 18, "GL": 0, "BH": 1, "HO": 0, "TK": 2, "RB": 1, "IF": 1, "CL": 3, "CG": 0, "FF": 1, "FA": 0, "BR": 0, "CP": 8, "UP": 0, "CM": 0, "MI": 0, "one_pct": 1, "BO": 0, "GA": 0, "pct_played": 82, "did_not_play": False},
            {"player": "Player A", "team": "Team X", "year": 2025, "date": "2025-01-15", "KI": 12, "HB": 13, "MK": 6, "DI": 25, "GL": 2, "BH": 0, "HO": 0, "TK": 4, "RB": 3, "IF": 3, "CL": 5, "CG": 0, "FF": 0, "FA": 0, "BR": 0, "CP": 12, "UP": 1, "CM": 1, "MI": 2, "one_pct": 0, "BO": 0, "GA": 2, "pct_played": 78, "did_not_play": False},
            # Synthetic future match (stats unknown)
            {"player": "Player A", "team": "Team X", "year": 2025, "date": "2025-01-22", "KI": None, "HB": None, "MK": None, "DI": None, "GL": None, "BH": None, "HO": None, "TK": None, "RB": None, "IF": None, "CL": None, "CG": None, "FF": None, "FA": None, "BR": None, "CP": None, "UP": None, "CM": None, "MI": None, "one_pct": None, "BO": None, "GA": None, "pct_played": None, "did_not_play": False},
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])

        out = features.add_rolling_features(df)

        # For the synthetic row, player_gl_avg_3 should be the mean of previous 3 GL values: (1+0+2)/3 = 1.0
        synth = out.sort_values("date").iloc[-1]
        self.assertIn("player_gl_avg_3", out.columns)
        self.assertAlmostEqual(float(synth["player_gl_avg_3"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
