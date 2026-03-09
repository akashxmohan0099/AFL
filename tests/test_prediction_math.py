import unittest

import pandas as pd

from prediction_math import audit_prediction_frame, reconcile_goal_distribution
from validate import ValidationError, validate_predictions


class TestPredictionMath(unittest.TestCase):
    def test_reconcile_goal_distribution_repairs_exported_probabilities(self):
        df = pd.DataFrame(
            {
                "player": ["A"],
                "team": ["T1"],
                "predicted_goals": [1.2],
                "predicted_behinds": [0.4],
                "p_scorer": [0.9],
                "p_goals_0": [0.2],
                "p_goals_1": [0.3],
                "p_goals_2": [0.2],
                "p_goals_3": [0.1],
                "p_goals_4": [0.05],
                "p_goals_5": [0.05],
                "p_goals_6": [0.05],
                "p_goals_7plus": [0.05],
                "p_2plus_goals": [0.55],
                "p_3plus_goals": [0.32],
            }
        )

        before = audit_prediction_frame(df)
        self.assertGreater(before["goal_zero_consistency_max_abs_error"], 0.0)

        reconcile_goal_distribution(df, round_dp=4)
        after = audit_prediction_frame(df)

        self.assertEqual(after["goal_threshold_monotonic_violations"], 0)
        self.assertEqual(after["goal_zero_consistency_max_abs_error"], 0.0)
        self.assertEqual(after["goal_2plus_consistency_max_abs_error"], 0.0)
        self.assertEqual(after["goal_3plus_consistency_max_abs_error"], 0.0)

    def test_validate_predictions_rejects_inconsistent_goal_distribution(self):
        df = pd.DataFrame(
            {
                "player": ["A"],
                "team": ["T1"],
                "predicted_goals": [1.2],
                "predicted_behinds": [0.4],
                "p_scorer": [0.9],
                "p_goals_0": [0.2],
                "p_goals_1": [0.3],
                "p_goals_2": [0.2],
                "p_goals_3": [0.1],
                "p_goals_4": [0.05],
                "p_goals_5": [0.05],
                "p_goals_6": [0.05],
                "p_goals_7plus": [0.05],
                "p_2plus_goals": [0.55],
                "p_3plus_goals": [0.32],
            }
        )

        with self.assertRaises(ValidationError):
            validate_predictions(df)


if __name__ == "__main__":
    unittest.main()
