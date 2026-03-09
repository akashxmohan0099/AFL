import types
import unittest

import numpy as np
import pandas as pd

import config
from model import AFLScoringModel


class _GoalCalibrator:
    def has_calibrator(self, target_name):
        return target_name in {"1plus_goals", "2plus_goals", "3plus_goals"}

    def transform(self, target_name, preds):
        preds = np.asarray(preds, dtype=float)
        if target_name == "1plus_goals":
            return np.full_like(preds, 0.92, dtype=float)
        if target_name == "2plus_goals":
            return np.full_like(preds, 0.71, dtype=float)
        if target_name == "3plus_goals":
            return np.full_like(preds, 0.48, dtype=float)
        return preds


class _GoalStore:
    def load_isotonic_calibrator(self, run_id=None):
        return _GoalCalibrator()


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

    def test_calibrated_thresholds_keep_exported_goal_pmf_consistent(self):
        m = AFLScoringModel()
        m.feature_cols = ["f1"]
        original_skip_targets = config.ISOTONIC_SKIP_TARGETS
        config.ISOTONIC_SKIP_TARGETS = set()

        try:
            def _fake_ensemble_predict(self, X_raw, X_scaled, target="goals", df=None, X_clean=None):
                if target == "goals":
                    scorer_prob = np.array([0.55], dtype=float)
                    mean_if = np.array([2.8], dtype=float)
                    pred = scorer_prob * mean_if
                    return pred, scorer_prob, mean_if
                pred_behinds = np.array([0.4], dtype=float)
                return pred_behinds, None, None

            m._ensemble_predict = types.MethodType(_fake_ensemble_predict, m)

            df = pd.DataFrame(
                {
                    "player": ["A"],
                    "team": ["T1"],
                    "opponent": ["O1"],
                    "venue": ["V"],
                    "round_number": [1],
                    "match_id": [1],
                    "f1": [0.0],
                }
            )

            out = m.predict_distributions(df, store=_GoalStore(), feature_cols=["f1"])
            p0 = out["p_goals_0"].to_numpy(dtype=float)
            p1 = out["p_goals_1"].to_numpy(dtype=float)
            p2 = out["p_goals_2"].to_numpy(dtype=float)

            self.assertTrue(np.allclose(out["p_scorer"], 0.92, atol=1e-4))
            self.assertTrue(np.allclose(p0, 1.0 - out["p_scorer"].to_numpy(dtype=float), atol=1e-4))
            self.assertTrue(
                np.allclose(
                    out["p_2plus_goals"].to_numpy(dtype=float),
                    1.0 - p0 - p1,
                    atol=1e-4,
                )
            )
            self.assertTrue(
                np.allclose(
                    out["p_3plus_goals"].to_numpy(dtype=float),
                    1.0 - p0 - p1 - p2,
                    atol=1e-4,
                )
            )
            pmf_cols = [f"p_goals_{k}" for k in range(config.GOAL_DISTRIBUTION_MAX_K)] + [f"p_goals_{config.GOAL_DISTRIBUTION_MAX_K}plus"]
            self.assertTrue(np.allclose(out[pmf_cols].sum(axis=1), 1.0, atol=0.001))
        finally:
            config.ISOTONIC_SKIP_TARGETS = original_skip_targets


if __name__ == "__main__":
    unittest.main()
