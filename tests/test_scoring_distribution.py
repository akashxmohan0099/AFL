import types
import unittest

import numpy as np
import pandas as pd

import config
from model import AFLScoringModel


class TestScoringDistribution(unittest.TestCase):
    def test_pmf_and_threshold_consistency(self):
        m = AFLScoringModel()
        m.feature_cols = ["f1"]

        # Monkeypatch the ensemble to avoid needing trained weights.
        def _fake_ensemble_predict(self, X_raw, X_scaled, target="goals", df=None, X_clean=None):
            if target == "goals":
                scorer_prob = np.array([0.3, 1.0], dtype=float)
                mean_if = np.array([2.5, 1.0], dtype=float)
                pred = scorer_prob * mean_if
                return pred, scorer_prob, mean_if
            pred_behinds = np.array([0.4, 0.2], dtype=float)
            return pred_behinds, None, None

        m._ensemble_predict = types.MethodType(_fake_ensemble_predict, m)

        df = pd.DataFrame(
            {
                "player": ["A", "B"],
                "team": ["T1", "T2"],
                "opponent": ["O1", "O2"],
                "venue": ["V", "V"],
                "round_number": [1, 1],
                "match_id": [1, 2],
                "f1": [0.0, 1.0],
            }
        )

        out = m.predict_distributions(df, store=None, feature_cols=["f1"])

        max_k = config.GOAL_DISTRIBUTION_MAX_K
        pmf_cols = [f"p_goals_{k}" for k in range(max_k)] + [f"p_goals_{max_k}plus"]
        pmf_sum = out[pmf_cols].sum(axis=1).to_numpy(dtype=float)
        # PMFs are rounded to 4dp in output, so allow a small tolerance.
        self.assertTrue(np.allclose(pmf_sum, 1.0, atol=0.02))

        # In the raw distribution, P(1+) should equal the stage-1 scorer probability.
        self.assertTrue(np.allclose(out["p_1plus_goals_raw"], out["p_scorer_raw"], atol=1e-6))

        # With no calibration store provided, p_scorer should equal p_1plus_goals.
        self.assertTrue(np.allclose(out["p_1plus_goals"], out["p_scorer"], atol=1e-6))

        # P(0) should be exactly 1 - P(1+) in this two-stage mixture.
        self.assertTrue(np.allclose(out["p_goals_0"], 1.0 - out["p_scorer_raw"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

