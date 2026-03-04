import types
import unittest

import numpy as np
import pandas as pd

import config
from model import AFLDisposalModel, AFLMarksModel, AFLScoringModel


class _DummyCalibrator:
    def has_calibrator(self, target_name):
        return True

    def transform(self, target_name, preds):
        preds = np.asarray(preds, dtype=float)

        # Goals: deliberately violate ordering (2+ > 1+).
        if target_name == "1plus_goals":
            return np.full_like(preds, 0.6, dtype=float)
        if target_name == "2plus_goals":
            return np.full_like(preds, 0.7, dtype=float)
        if target_name == "3plus_goals":
            return np.full_like(preds, 0.65, dtype=float)

        # Disposals: include out-of-range values to ensure clip(), and violate ordering (25+ > 20+).
        if target_name.endswith("plus_disp"):
            if target_name.startswith("10"):
                return np.full_like(preds, 1.2, dtype=float)
            if target_name.startswith("15"):
                return np.full_like(preds, -0.1, dtype=float)
            if target_name.startswith("20"):
                return np.full_like(preds, 0.2, dtype=float)
            if target_name.startswith("25"):
                return np.full_like(preds, 0.3, dtype=float)
            if target_name.startswith("30"):
                return np.full_like(preds, 0.1, dtype=float)
            return preds

        # Marks: violate ordering (3+ > 2+).
        if target_name.endswith("plus_mk"):
            if target_name.startswith("2"):
                return np.full_like(preds, 0.4, dtype=float)
            if target_name.startswith("3"):
                return np.full_like(preds, 0.5, dtype=float)
            if target_name.startswith("4"):
                return np.full_like(preds, 0.45, dtype=float)
            if target_name.startswith("5"):
                return np.full_like(preds, 0.6, dtype=float)
            return preds

        return preds


class _DummyStore:
    def load_isotonic_calibrator(self, run_id=None):
        return _DummyCalibrator()

    def get_lambda_calibration(self, target, lam):
        # Should not be used in isotonic mode for goals/disposals/marks.
        return lam


class _NoLambdaStore(_DummyStore):
    def get_lambda_calibration(self, target, lam):
        raise AssertionError("get_lambda_calibration() should not be called in isotonic mode")


class TestThresholdMonotonicity(unittest.TestCase):
    def test_scoring_thresholds_monotonic_after_calibration(self):
        m = AFLScoringModel()
        m.feature_cols = ["f1"]

        def _fake_ensemble_predict(self, X_raw, X_scaled, target="goals", df=None, X_clean=None):
            if target == "goals":
                scorer_prob = np.array([0.3], dtype=float)
                mean_if = np.array([2.5], dtype=float)
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

        out = m.predict_distributions(df, store=_DummyStore(), feature_cols=["f1"])
        self.assertTrue((out["p_1plus_goals"] >= out["p_2plus_goals"]).all())
        self.assertTrue((out["p_2plus_goals"] >= out["p_3plus_goals"]).all())
        self.assertTrue(np.all((out["p_1plus_goals"] >= 0) & (out["p_1plus_goals"] <= 1)))
        self.assertTrue(np.allclose(out["p_scorer"], out["p_1plus_goals"]))

    def test_disposal_thresholds_monotonic_after_calibration(self):
        m = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION)
        m.feature_cols = ["f1"]

        def _fake_predict_raw(self, X_raw, X_scaled, df=None):
            return np.array([25.0, 25.0], dtype=float)

        m._predict_raw = types.MethodType(_fake_predict_raw, m)

        df = pd.DataFrame(
            {
                "player": ["A", "B"],
                "team": ["T1", "T2"],
                "match_id": [1, 2],
                "f1": [0.0, 1.0],
            }
        )

        out = m.predict_distributions(df, store=_DummyStore(), feature_cols=["f1"])
        cols = [f"p_{t}plus_disp" for t in config.DISPOSAL_THRESHOLDS]
        arr = out[cols].to_numpy(dtype=float)
        self.assertTrue(np.all(arr[:, :-1] >= arr[:, 1:]))
        self.assertTrue(np.all((arr >= 0.0) & (arr <= 1.0)))

    def test_marks_thresholds_monotonic_and_no_lambda_calibration_in_isotonic_mode(self):
        m = AFLMarksModel()
        m.feature_cols = ["f1"]

        def _fake_predict_raw(self, X_raw, X_scaled, df=None):
            return np.array([4.0], dtype=float)

        m._predict_raw = types.MethodType(_fake_predict_raw, m)

        df = pd.DataFrame(
            {
                "player": ["A"],
                "team": ["T1"],
                "match_id": [1],
                "f1": [0.0],
            }
        )

        out = m.predict_distributions(df, store=_NoLambdaStore(), feature_cols=["f1"])
        cols = [f"p_{t}plus_mk" for t in config.MARKS_THRESHOLDS]
        arr = out[cols].to_numpy(dtype=float)
        self.assertTrue(np.all(arr[:, :-1] >= arr[:, 1:]))
        self.assertTrue(np.all((arr >= 0.0) & (arr <= 1.0)))


if __name__ == "__main__":
    unittest.main()

