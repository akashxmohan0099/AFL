"""
AFL Prediction Pipeline — Model Training, Evaluation & Prediction
==================================================================
Dual Poisson/GBT ensemble for goals and behinds prediction.

Architecture:
  - Goals model:  80% PoissonRegressor + 20% HistGradientBoostingRegressor
  - Behinds model: same architecture (behinds are more stochastic)
  - Disposal model: separate Poisson/GBT ensemble with own tuned params
  - Game winner model: Elo + HistGBT with hybrid market prior

Training:
  - Time-based split: train 2015-2023, validate 2024
  - Walk-forward within training (per-round)
  - Must beat naive baseline (career goal average)

Output per prediction:
  - predicted_goals, predicted_behinds, predicted_score
  - 80% confidence interval from Poisson distribution
  - Top-3 driving features
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from prediction_math import reconcile_goal_distribution

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Isotonic Calibration
# ---------------------------------------------------------------------------

class CalibratedPredictor:
    """Post-hoc probability calibration using isotonic regression.

    Fits separate IsotonicRegression models per threshold target
    (e.g., "1plus_goals", "25plus_disp"). Given (predicted_prob, actual_binary)
    pairs, learns a monotone mapping from raw probabilities to calibrated ones.
    """

    def __init__(self):
        self._calibrators = {}  # target_name → IsotonicRegression

    def fit(self, target_name, preds, actuals):
        """Fit isotonic calibrator for a specific threshold target.

        Args:
            target_name: e.g. "1plus_goals", "25plus_disp"
            preds: array of predicted probabilities
            actuals: array of binary outcomes (0/1)
        """
        preds = np.asarray(preds, dtype=float)
        actuals = np.asarray(actuals, dtype=float)
        valid = ~np.isnan(preds) & ~np.isnan(actuals)
        if valid.sum() < getattr(config, "ISOTONIC_MIN_SAMPLES", 100):
            return  # not enough data to fit

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(preds[valid], actuals[valid])
        self._calibrators[target_name] = ir

    def transform(self, target_name, preds):
        """Apply isotonic calibration to predicted probabilities.

        Returns calibrated probabilities, or original if no calibrator fitted.
        """
        if target_name not in self._calibrators:
            return preds
        preds = np.asarray(preds, dtype=float)
        return self._calibrators[target_name].predict(preds)

    def has_calibrator(self, target_name):
        """Check if a calibrator exists for the given target."""
        return target_name in self._calibrators

    @property
    def targets(self):
        """List of fitted target names."""
        return list(self._calibrators.keys())


def _prepare_features(df, feature_cols, scaler=None, fit_scaler=False):
    """Prepare feature matrices for the dual Poisson/GBT ensemble.

    Returns (X_raw, X_clean, X_scaled):
      X_raw:    inf→NaN, NaN preserved (for HistGBT which handles NaN natively)
      X_clean:  NaN→0 (for StandardScaler / PoissonRegressor / standard GBT)
      X_scaled: scaler output from X_clean (None if no scaler provided)
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns: {missing[:10]}")
    X_raw = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    X_clean = X_raw.fillna(0)
    X_scaled = None
    if scaler is not None:
        X_scaled = scaler.fit_transform(X_clean) if fit_scaler else scaler.transform(X_clean)
    return X_raw, X_clean, X_scaled


def _model_supports_nan(model):
    """Return True when estimator supports NaN inputs natively."""
    try:
        tags = model._get_tags()
        if "allow_nan" in tags:
            return bool(tags["allow_nan"])
    except Exception:
        pass
    name = model.__class__.__name__
    return "HistGradientBoosting" in name


def _predict_with_compatible_input(model, X_raw, X_clean):
    """Predict using raw or cleaned matrix based on model NaN support."""
    X = X_raw if _model_supports_nan(model) else X_clean
    return model.predict(X)


def _predict_proba_with_compatible_input(model, X_raw, X_clean):
    """Predict probabilities using raw or cleaned matrix based on NaN support."""
    X = X_raw if _model_supports_nan(model) else X_clean
    return model.predict_proba(X)[:, 1]


def _enforce_non_increasing_probs(df: pd.DataFrame, cols, round_dp: int = 4) -> None:
    """Enforce row-wise monotonicity for threshold probabilities.

    For probability thresholds, higher thresholds must not have higher
    probabilities (e.g. P(3+) <= P(2+) <= P(1+)).

    This matters because thresholds are often calibrated independently
    (e.g. via isotonic regression), which can otherwise violate ordering.
    """
    if df is None or df.empty:
        return

    present = [c for c in cols if c in df.columns]
    if not present:
        return

    # Always clip any present columns into [0, 1] first.
    if len(present) == 1:
        c = present[0]
        df[c] = np.round(np.clip(df[c].astype(float), 0.0, 1.0), round_dp)
        return

    arr = df[present].to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    # Enforce non-increasing sequence across columns (left-to-right).
    arr = np.minimum.accumulate(arr, axis=1)
    df[present] = np.round(arr, round_dp)



class AFLScoringModel:
    """Two-stage ensemble model for predicting goals and behinds per player per match.

    Stage 1: Binary classifier predicts P(scorer) — whether a player scores >= 1 goal.
    Stage 2: Poisson + GBT ensemble predicts E[goals | scorer] for predicted scorers.
    Final prediction = P(scorer) * E[goals | scorer].

    This addresses the zero-inflation problem: ~68% of players score 0 goals,
    so a single regression model averages toward the mean. The two-stage approach
    lets Stage 1 learn who scores and Stage 2 learn how many.
    """

    def __init__(self, gbt_params=None, poisson_params=None, ensemble_weights=None):
        self.scorer_clf = None  # Stage 1: binary classifier
        self.goals_poisson = None
        self.goals_gbt = None
        self.behinds_poisson = None
        self.behinds_gbt = None
        self.scaler = None
        self.feature_cols = []
        self.eval_metrics = {}
        self.training_info = {}
        # Instance-level params (override config defaults)
        self.gbt_params = gbt_params or config.HIST_GBT_PARAMS_BACKTEST
        self.poisson_params = poisson_params or config.POISSON_PARAMS
        self.ensemble_weights = ensemble_weights or config.ENSEMBLE_WEIGHTS

    @staticmethod
    def _mixture_quantile(p_scorer, lambda_if_scorer, quantile, max_k=15):
        """Quantile from the two-stage goals mixture.

        Stage 1 predicts p = P(X>=1).
        Stage 2 predicts E[X | X>=1] (lambda_if_scorer).

        We model the scorer distribution with a *shifted Poisson*:
          X = 0 w.p. (1-p)
          X = 1 + Pois(mu) w.p. p
        where mu = max(lambda_if_scorer - 1, 0).
        """
        p = float(np.clip(p_scorer, 0.0, 1.0))
        mean_if = max(float(lambda_if_scorer), 1.0)
        mu = max(mean_if - 1.0, 0.001)
        cdf = 0.0
        for k in range(max_k + 1):
            if k == 0:
                pmf_k = 1.0 - p
            else:
                pmf_k = p * poisson.pmf(k - 1, mu)
            cdf += pmf_k
            if cdf >= quantile:
                return k
        return max_k

    def _check_scorer_calibration(self, X_val, y_val_is_scorer):
        """Check calibration of scorer classifier and print ECE diagnostic."""
        from sklearn.calibration import calibration_curve
        prob = self.scorer_clf.predict_proba(X_val)[:, 1]
        frac, mean_pred = calibration_curve(y_val_is_scorer, prob, n_bins=10, strategy="uniform")
        ece = sum(abs(f - m) for f, m in zip(frac, mean_pred)) / len(frac)
        if ece > 0.05:
            print(f"  WARNING: Scorer ECE = {ece:.4f} (>0.05 — consider calibration)")
        else:
            print(f"  Scorer calibration ECE: {ece:.4f} (good)")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df, feature_cols=None):
        """Train goals and behinds models on the provided feature matrix.

        Args:
            df: DataFrame with features, targets (GL, BH), and sample_weight.
            feature_cols: list of feature column names. If None, loads from
                         features/feature_columns.json.
        """
        if feature_cols is None:
            feat_path = config.FEATURES_DIR / "feature_columns.json"
            if feat_path.exists():
                with open(feat_path) as f:
                    feature_cols = json.load(f)
            else:
                raise ValueError("No feature_cols provided and no feature_columns.json found")

        self.feature_cols = feature_cols

        # Filter to players with enough matches (relax for small datasets)
        player_counts = df.groupby(["player", "team"])["match_id"].transform("nunique")
        min_matches = min(config.MIN_PLAYER_MATCHES, max(1, int(player_counts.max()) - 1))
        df = df[player_counts >= min_matches].copy()

        # Time-based train/val split
        train_df = df[df["year"] < config.VALIDATION_YEAR].copy()
        val_df = df[df["year"] == config.VALIDATION_YEAR].copy()

        # If validation set empty, use last 20% of data for validation
        if val_df.empty and len(train_df) > 0:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
        elif train_df.empty and val_df.empty:
            # Extremely small dataset: use all for both
            train_df = df.copy()
            val_df = df.copy()
            print("  WARNING: Dataset too small for proper train/val split")

        print(f"Training set: {len(train_df)} rows ({train_df['year'].min()}-{train_df['year'].max()})")
        print(f"Validation set: {len(val_df)} rows")

        # Prepare features
        y_train_goals = train_df["GL"].values
        y_train_behinds = train_df["BH"].values
        y_val_goals = val_df["GL"].values
        y_val_behinds = val_df["BH"].values

        weights_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
        weights_val = val_df["sample_weight"].values if "sample_weight" in val_df.columns else None

        # Prepare feature matrices (X_raw=NaN preserved, X_clean=NaN→0, X_scaled=scaled clean)
        self.scaler = StandardScaler()
        X_train_raw, X_train_clean, X_train_scaled = _prepare_features(
            train_df, feature_cols, scaler=self.scaler, fit_scaler=True
        )
        X_val_raw, X_val_clean, X_val_scaled = _prepare_features(
            val_df, feature_cols, scaler=self.scaler
        )

        # --- Stage 1: Train Scorer Classifier ---
        print("\n--- Stage 1: Training Scorer Classifier ---")
        y_train_is_scorer = (y_train_goals >= 1).astype(int)
        self.scorer_clf = GradientBoostingClassifier(
            n_estimators=config.GBT_PARAMS["n_estimators"],
            max_depth=config.GBT_PARAMS["max_depth"],
            learning_rate=config.GBT_PARAMS["learning_rate"],
            min_samples_leaf=config.GBT_PARAMS["min_samples_leaf"],
            subsample=config.GBT_PARAMS["subsample"],
            random_state=config.RANDOM_SEED,
        )
        self.scorer_clf.fit(X_train_clean, y_train_is_scorer, sample_weight=weights_train)

        y_val_is_scorer = (y_val_goals >= 1).astype(int)
        scorer_acc = np.mean(
            self.scorer_clf.predict(X_val_clean) == y_val_is_scorer
        )
        scorer_prob = self.scorer_clf.predict_proba(X_val_clean)[:, 1]
        print(f"  Scorer classification accuracy: {scorer_acc:.3f}")
        print(f"  Mean P(scorer) on val: {scorer_prob.mean():.3f} "
              f"(actual: {y_val_is_scorer.mean():.3f})")
        self._check_scorer_calibration(X_val_clean, y_val_is_scorer)

        # --- Stage 2: Train Goals Models (on scorers only — GL >= 1) ---
        print("\n--- Stage 2: Training Goals Models (scorers only) ---")

        scorer_mask_train = y_train_goals >= 1
        X_train_scorers = X_train_clean[scorer_mask_train]
        X_train_scaled_scorers = X_train_scaled[scorer_mask_train]
        y_train_goals_scorers = y_train_goals[scorer_mask_train]
        y_train_behinds_scorers = y_train_behinds[scorer_mask_train]
        weights_train_scorers = weights_train[scorer_mask_train] if weights_train is not None else None

        print(f"  Scorer training rows: {scorer_mask_train.sum()} / {len(y_train_goals)}")

        # Poisson regressor
        print("  Training Poisson regressor for goals...")
        self.goals_poisson = PoissonRegressor(
            alpha=config.POISSON_PARAMS["alpha"],
            max_iter=config.POISSON_PARAMS["max_iter"],
        )
        self.goals_poisson.fit(X_train_scaled_scorers, y_train_goals_scorers,
                               sample_weight=weights_train_scorers)

        # GBT regressor
        print("  Training GBT regressor for goals...")
        self.goals_gbt = GradientBoostingRegressor(**config.GBT_PARAMS)
        self.goals_gbt.fit(X_train_scorers, y_train_goals_scorers,
                           sample_weight=weights_train_scorers)

        # --- Train Behinds Models (on all data — behinds are more diffuse) ---
        print("\n--- Training Behinds Models ---")

        print("  Training Poisson regressor for behinds...")
        self.behinds_poisson = PoissonRegressor(
            alpha=config.POISSON_PARAMS["alpha"],
            max_iter=config.POISSON_PARAMS["max_iter"],
        )
        self.behinds_poisson.fit(X_train_scaled, y_train_behinds, sample_weight=weights_train)

        print("  Training GBT regressor for behinds...")
        self.behinds_gbt = GradientBoostingRegressor(**config.GBT_PARAMS)
        self.behinds_gbt.fit(X_train_clean, y_train_behinds, sample_weight=weights_train)

        # --- Evaluate on validation set ---
        print("\n--- Evaluating on validation set ---")
        self.eval_metrics = self._evaluate(
            X_val_raw, X_val_scaled, y_val_goals, y_val_behinds, val_df
        )

        self.training_info = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "n_features": len(feature_cols),
            "train_years": f"{train_df['year'].min()}-{train_df['year'].max()}",
            "val_year": config.VALIDATION_YEAR,
        }

        return self.eval_metrics

    def train_backtest(self, df, feature_cols):
        """Train all sub-models without validation split or metric printing.

        Used by the backtest loop to quickly fit on a training subset.
        Uses HistGradientBoosting for 10-50x faster training than standard GBT.
        """
        self.feature_cols = feature_cols
        hist_params = self.gbt_params

        y_goals = df["GL"].values
        y_behinds = df["BH"].values
        weights = df["sample_weight"].values if "sample_weight" in df.columns else None

        self.scaler = StandardScaler()
        X_raw, X_clean, X_scaled = _prepare_features(
            df, feature_cols, scaler=self.scaler, fit_scaler=True
        )

        y_is_scorer = (y_goals >= 1).astype(int)
        scorer_mask = y_goals >= 1
        X_raw_scorers = X_raw[scorer_mask]
        X_clean_scorers = X_clean[scorer_mask]
        X_scaled_scorers = X_scaled[scorer_mask]
        y_goals_scorers = y_goals[scorer_mask]
        weights_scorers = weights[scorer_mask] if weights is not None else None

        # --- Poisson components ---
        self.goals_poisson = PoissonRegressor(
            alpha=self.poisson_params["alpha"],
            max_iter=self.poisson_params["max_iter"],
        )
        self.goals_poisson.fit(X_scaled_scorers, y_goals_scorers, sample_weight=weights_scorers)

        self.behinds_poisson = PoissonRegressor(
            alpha=self.poisson_params["alpha"],
            max_iter=self.poisson_params["max_iter"],
        )
        self.behinds_poisson.fit(X_scaled, y_behinds, sample_weight=weights)

        # --- Single-model GBT path ---
        self.scorer_clf = HistGradientBoostingClassifier(**hist_params)
        self.scorer_clf.fit(X_raw, y_is_scorer, sample_weight=weights)

        self.goals_gbt = HistGradientBoostingRegressor(**hist_params)
        self.goals_gbt.fit(X_raw_scorers, y_goals_scorers, sample_weight=weights_scorers)

        self.behinds_gbt = HistGradientBoostingRegressor(**hist_params)
        self.behinds_gbt.fit(X_raw, y_behinds, sample_weight=weights)

    def _ensemble_predict(self, X_raw, X_scaled, target="goals", df=None, X_clean=None):
        """Generate ensemble prediction (two-stage for goals, standard for behinds).

        For goals:
          Stage 1: P(scorer) from binary classifier
          Stage 2: E[goals | scorer] from Poisson + GBT ensemble (trained on scorers only)
          Final = P(scorer) * E[goals | scorer]

        When df is provided and contains market_expected_team_goals, the Poisson
        component is blended with a market-implied baseline per player.

        Returns:
          For goals: (predictions, scorer_prob, raw_pred) tuple
            raw_pred = E[goals | scorer] (lambda before multiplying by P(scorer))
          For behinds: (predictions, None, None) tuple
        """
        w_poi = self.ensemble_weights["poisson"]
        w_gbt = self.ensemble_weights["gbt"]
        blend = getattr(config, "MARKET_POISSON_BLEND", 0.0)

        if X_clean is None:
            X_clean = X_raw.fillna(0) if hasattr(X_raw, "fillna") else X_raw

        if target == "goals":
            pred_poi = self.goals_poisson.predict(X_scaled)

            # Market blend for Poisson component
            if blend > 0 and df is not None and "market_expected_team_goals" in df.columns:
                mkt_goals = df["market_expected_team_goals"].values.astype(float)
                # Estimate per-player goal share from career avg (capped at 40%)
                if "career_goal_avg_pre" in df.columns:
                    career_avg = df["career_goal_avg_pre"].fillna(0).values
                    # Average ~10 goals per team per game, so share = career_avg / 10
                    share = np.clip(career_avg / 10.0, 0.0, 0.4)
                else:
                    share = np.full(len(df), 0.1)  # default 10% share
                market_baseline = mkt_goals * share
                has_market = ~np.isnan(mkt_goals) & (mkt_goals > 0)
                pred_poi = np.where(
                    has_market,
                    (1 - blend) * pred_poi + blend * market_baseline,
                    pred_poi,
                )

            pred_gbt = _predict_with_compatible_input(self.goals_gbt, X_raw, X_clean)
            raw_pred = np.clip(w_poi * pred_poi + w_gbt * pred_gbt, 0, None)

            # Two-stage: pred = P(scorer) * E[goals | scorer]
            if self.scorer_clf is not None:
                scorer_prob = _predict_proba_with_compatible_input(
                    self.scorer_clf, X_raw, X_clean
                )
                pred = scorer_prob * raw_pred
                pred = np.clip(pred, 0, None)
            else:
                scorer_prob = np.ones(len(X_raw))
                pred = raw_pred
            return pred, scorer_prob, raw_pred
        else:
            pred_poi = self.behinds_poisson.predict(X_scaled)

            pred_gbt = _predict_with_compatible_input(self.behinds_gbt, X_raw, X_clean)
            pred = np.clip(w_poi * pred_poi + w_gbt * pred_gbt, 0, None)
            return pred, None, None

    def _evaluate(self, X_val, X_val_scaled, y_val_goals, y_val_behinds, val_df):
        """Evaluate on validation set. Returns dict of metrics."""
        pred_goals, scorer_prob, _ = self._ensemble_predict(X_val, X_val_scaled, "goals")
        pred_behinds, _, _ = self._ensemble_predict(X_val, X_val_scaled, "behinds")

        # Baseline: career_goal_avg_pre (pre-game, no leakage)
        baseline_goals = val_df["career_goal_avg_pre"].fillna(0).values

        metrics = {
            "goals_mae": mean_absolute_error(y_val_goals, pred_goals),
            "goals_rmse": np.sqrt(mean_squared_error(y_val_goals, pred_goals)),
            "goals_baseline_mae": mean_absolute_error(y_val_goals, baseline_goals),
            "behinds_mae": mean_absolute_error(y_val_behinds, pred_behinds),
            "behinds_rmse": np.sqrt(mean_squared_error(y_val_behinds, pred_behinds)),
        }

        # Improvement over baseline
        if metrics["goals_baseline_mae"] > 0:
            metrics["goals_improvement_pct"] = (
                (metrics["goals_baseline_mae"] - metrics["goals_mae"])
                / metrics["goals_baseline_mae"] * 100
            )
        else:
            metrics["goals_improvement_pct"] = 0.0

        print(f"\n  Goals MAE:       {metrics['goals_mae']:.4f}")
        print(f"  Goals RMSE:      {metrics['goals_rmse']:.4f}")
        print(f"  Baseline MAE:    {metrics['goals_baseline_mae']:.4f}")
        print(f"  Improvement:     {metrics['goals_improvement_pct']:.1f}%")
        print(f"  Behinds MAE:     {metrics['behinds_mae']:.4f}")
        print(f"  Behinds RMSE:    {metrics['behinds_rmse']:.4f}")

        # Feature importance (from GBT)
        if not hasattr(self.goals_gbt, 'feature_importances_'):
            return metrics
        importances = self.goals_gbt.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print("\n  Top 15 goal features:")
        for i in top_idx:
            print(f"    {self.feature_cols[i]:40s} {importances[i]:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df, feature_cols=None, store=None):
        """Generate predictions for a DataFrame of upcoming matches.

        Args:
            df: DataFrame with the same feature columns as training data.
                Must include 'player', 'team', 'opponent', 'venue', 'round'.
            store: LearningStore instance for isotonic calibration (optional).

        Returns:
            DataFrame with prediction columns added.
        """
        feature_cols = feature_cols or self.feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(df, feature_cols, scaler=self.scaler)

        pred_goals, scorer_prob, lambda_if_scorer = self._ensemble_predict(
            X_raw, X_scaled, "goals", df=df, X_clean=X_clean
        )
        pred_behinds, _, _ = self._ensemble_predict(
            X_raw, X_scaled, "behinds", X_clean=X_clean
        )

        # Pick round column — new schema uses round_number, legacy used round
        round_col = "round_number" if "round_number" in df.columns else "round"
        result = df[["player", "team", "opponent", "venue", round_col]].copy()
        if round_col != "round":
            result = result.rename(columns={round_col: "round"})
        result["p_scorer"] = np.round(scorer_prob, 4)
        result["predicted_goals"] = np.round(pred_goals, 2)
        result["predicted_behinds"] = np.round(pred_behinds, 2)
        result["predicted_score"] = np.round(pred_goals * 6 + pred_behinds, 2)

        # Apply isotonic calibration to p_scorer if available
        if store is not None and getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            if calibrator is not None and calibrator.has_calibrator("1plus_goals"):
                result["p_scorer"] = np.round(calibrator.transform("1plus_goals", result["p_scorer"].values), 4)

        # Confidence intervals from zero-inflated Poisson mixture (80%)
        result["conf_lower_gl"] = [
            self._mixture_quantile(sp, lam, 0.10)
            for sp, lam in zip(scorer_prob, lambda_if_scorer)
        ]
        result["conf_upper_gl"] = [
            self._mixture_quantile(sp, lam, 0.90)
            for sp, lam in zip(scorer_prob, lambda_if_scorer)
        ]

        # Top-3 driving features per prediction
        top_factors = self._explain_predictions(X_clean, feature_cols)
        result["top_factor_1"] = top_factors[0]
        result["top_factor_2"] = top_factors[1]
        result["top_factor_3"] = top_factors[2]

        # Player role and career baseline
        if "player_role" in df.columns:
            result["player_role"] = df["player_role"].values
        if "career_goal_avg_pre" in df.columns:
            result["career_goal_avg"] = df["career_goal_avg_pre"].values

        # Sort by P(scorer) desc (primary), predicted_goals desc (secondary)
        result = result.sort_values(
            ["p_scorer", "predicted_goals"], ascending=[False, False]
        ).reset_index(drop=True)

        # Validate predictions
        from validate import validate_predictions
        validate_predictions(result)

        return result

    def predict_distributions(self, df, store=None, feature_cols=None):
        """Generate full probability distributions for goals and behinds.

        Returns per-player Poisson PMFs, calibrated lambdas, confidence
        intervals, and point estimates — structured for dashboard consumption.

        Args:
            df: DataFrame with features and identifiers.
            store: LearningStore instance for calibration adjustments.
            feature_cols: list of feature column names.

        Returns:
            DataFrame with distribution columns per player.
        """
        feature_cols = feature_cols or self.feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(df, feature_cols, scaler=self.scaler)

        pred_goals, scorer_prob, lambda_if_scorer = self._ensemble_predict(
            X_raw, X_scaled, "goals", df=df, X_clean=X_clean
        )
        pred_behinds, _, _ = self._ensemble_predict(
            X_raw, X_scaled, "behinds", X_clean=X_clean
        )

        max_k = config.GOAL_DISTRIBUTION_MAX_K

        # Build result DataFrame with identifiers
        round_col = "round_number" if "round_number" in df.columns else "round"
        result = pd.DataFrame({
            "player": df["player"].values,
            "team": df["team"].values,
            "opponent": df["opponent"].values,
            "venue": df["venue"].values,
            "round": df[round_col].values,
            "match_id": df["match_id"].values,
        })

        # Point estimates
        result["predicted_goals"] = np.round(pred_goals, 4)
        result["predicted_behinds"] = np.round(pred_behinds, 4)
        result["predicted_score"] = np.round(pred_goals * 6 + pred_behinds, 2)
        # Keep raw scorer probability for learning/calibration; expose calibrated as p_scorer.
        result["p_scorer_raw"] = np.round(scorer_prob, 4)
        result["p_scorer"] = result["p_scorer_raw"].copy()

        # Calibrated lambdas and full PMFs
        lambda_goals = np.zeros(len(df))
        lambda_behinds = np.zeros(len(df))
        goal_pmfs = np.zeros((len(df), max_k + 1))  # k=0..6, plus 7+
        behind_pmfs = np.zeros((len(df), 5))  # k=0..3, plus 4+
        conf_lower_gl = np.zeros(len(df))
        conf_upper_gl = np.zeros(len(df))

        cal_method = getattr(config, "CALIBRATION_METHOD", "bucket")

        for i in range(len(df)):
            # Goals: lambda_if_scorer is interpreted as E[goals | goals>=1] (scorers only).
            sp = float(np.clip(scorer_prob[i], 0.0, 1.0))
            raw_mean_if = max(float(lambda_if_scorer[i]), 1.0)

            # Bucket-based lambda calibration is a heuristic; disable when isotonic is the chosen method.
            if store is not None and cal_method != "isotonic":
                cal_gl_if_scorer = float(store.get_lambda_calibration("goals", raw_mean_if))
            else:
                cal_gl_if_scorer = raw_mean_if
            cal_gl_if_scorer = max(cal_gl_if_scorer, 1.0)

            lambda_goals[i] = sp * cal_gl_if_scorer  # unconditional expected goals

            # Goal PMF: two-stage mixture with a zero-free scorer component.
            # P(X=0) = 1-p, and for k>=1: P(X=k) = p * Pois(k-1; mu)
            # where mu = mean_if - 1 (shifted Poisson).
            goal_pmfs[i, 0] = 1.0 - sp
            mu = max(cal_gl_if_scorer - 1.0, 0.001)
            for k in range(1, max_k):
                goal_pmfs[i, k] = sp * poisson.pmf(k - 1, mu)
            goal_pmfs[i, max_k] = max(1.0 - goal_pmfs[i, :max_k].sum(), 0.0)

            # Behinds: raw lambda = prediction from ensemble (unchanged)
            raw_bh = max(pred_behinds[i], 0.001)
            if store is not None:
                cal_bh = raw_bh  # behinds don't have calibration targets yet
            else:
                cal_bh = raw_bh
            lambda_behinds[i] = cal_bh

            # Behind PMF: P(X=k) for k=0..3, then P(X>=4)
            for k in range(4):
                behind_pmfs[i, k] = poisson.pmf(k, cal_bh)
            behind_pmfs[i, 4] = 1.0 - poisson.cdf(3, cal_bh)

            # Confidence intervals (80%) — zero-inflated Poisson mixture
            conf_lower_gl[i] = self._mixture_quantile(sp, cal_gl_if_scorer, 0.10)
            conf_upper_gl[i] = self._mixture_quantile(sp, cal_gl_if_scorer, 0.90)

        result["lambda_goals"] = np.round(lambda_goals, 4)
        result["lambda_behinds"] = np.round(lambda_behinds, 4)

        # Goal distribution columns
        for k in range(max_k):
            result[f"p_goals_{k}"] = np.round(goal_pmfs[:, k], 4)
        result[f"p_goals_{max_k}plus"] = np.round(goal_pmfs[:, max_k], 4)

        # Behind distribution columns
        for k in range(4):
            result[f"p_behinds_{k}"] = np.round(behind_pmfs[:, k], 4)
        result["p_behinds_4plus"] = np.round(behind_pmfs[:, 4], 4)

        # Goal threshold probabilities: raw values derived from the PMF
        p1_raw = np.clip(1.0 - goal_pmfs[:, 0], 0.0, 1.0)
        p2_raw = np.clip(1.0 - goal_pmfs[:, 0] - goal_pmfs[:, 1], 0.0, 1.0)
        p3_raw = np.clip(1.0 - goal_pmfs[:, 0] - goal_pmfs[:, 1] - goal_pmfs[:, 2], 0.0, 1.0)
        result["p_1plus_goals_raw"] = np.round(p1_raw, 4)
        result["p_2plus_goals_raw"] = np.round(p2_raw, 4)
        result["p_3plus_goals_raw"] = np.round(p3_raw, 4)

        # Default outputs (may be calibrated below)
        result["p_1plus_goals"] = result["p_1plus_goals_raw"].copy()
        result["p_2plus_goals"] = result["p_2plus_goals_raw"].copy()
        result["p_3plus_goals"] = result["p_3plus_goals_raw"].copy()

        # Apply isotonic calibration to goal thresholds if available
        if store is not None and cal_method == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            skip_targets = getattr(config, "ISOTONIC_SKIP_TARGETS", set())
            if calibrator is not None:
                for threshold, name in [(1, "1plus_goals"), (2, "2plus_goals"), (3, "3plus_goals")]:
                    if name in skip_targets:
                        continue
                    col = f"p_{name}"
                    if calibrator.has_calibrator(name) and col in result.columns:
                        result[col] = np.round(calibrator.transform(name, result[col].values), 4)

        # Independent calibration can break cross-threshold ordering; enforce monotonicity.
        _enforce_non_increasing_probs(
            result, ["p_1plus_goals", "p_2plus_goals", "p_3plus_goals"], round_dp=4
        )

        # Keep the exported goal PMF aligned with any calibrated threshold probabilities.
        reconcile_goal_distribution(result, round_dp=4)

        # Keep p_scorer aligned to the (possibly calibrated) 1+ probability.
        result["p_scorer"] = result["p_1plus_goals"].values

        # Confidence intervals
        result["conf_lower_gl"] = conf_lower_gl.astype(int)
        result["conf_upper_gl"] = conf_upper_gl.astype(int)

        # Context columns
        if "career_goal_avg_pre" in df.columns:
            result["career_goal_avg"] = df["career_goal_avg_pre"].values
        if "player_role" in df.columns:
            result["player_role"] = df["player_role"].values

        # Sort by P(scorer) desc
        result = result.sort_values(
            ["p_scorer", "predicted_goals"], ascending=[False, False]
        ).reset_index(drop=True)

        return result

    def _explain_predictions(self, X, feature_cols):
        """For each prediction, identify top-3 features driving it.

        Uses GBT feature importances weighted by actual feature values
        (simple approximation of SHAP without the dependency).
        """
        if not hasattr(self.goals_gbt, 'feature_importances_'):
            return result
        importances = self.goals_gbt.feature_importances_

        top1, top2, top3 = [], [], []

        for i in range(len(X)):
            row = X.iloc[i] if isinstance(X, pd.DataFrame) else X[i]
            # Score = importance * abs(normalized value)
            vals = np.nan_to_num(np.abs(np.array(row, dtype=float)), nan=0.0)
            # Avoid zero values dominating
            scores = importances * (1 + vals / (vals.max() + 1e-9))
            top_idx = np.argsort(scores)[::-1][:3]

            def _fmt(idx):
                name = feature_cols[idx]
                val = row.iloc[idx] if isinstance(row, pd.Series) else row[idx]
                return f"{name}={val:.2f}"

            top1.append(_fmt(top_idx[0]))
            top2.append(_fmt(top_idx[1]))
            top3.append(_fmt(top_idx[2]))

        return top1, top2, top3

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, models_dir=None):
        """Save trained models, scaler, and metadata."""
        models_dir = Path(models_dir or config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        with open(models_dir / "scorer_clf.pkl", "wb") as f:
            pickle.dump(self.scorer_clf, f)
        with open(models_dir / "goals_poisson.pkl", "wb") as f:
            pickle.dump(self.goals_poisson, f)
        with open(models_dir / "goals_gbt.pkl", "wb") as f:
            pickle.dump(self.goals_gbt, f)
        with open(models_dir / "behinds_poisson.pkl", "wb") as f:
            pickle.dump(self.behinds_poisson, f)
        with open(models_dir / "behinds_gbt.pkl", "wb") as f:
            pickle.dump(self.behinds_gbt, f)
        with open(models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        metadata = {
            "feature_cols": self.feature_cols,
            "eval_metrics": self.eval_metrics,
            "training_info": self.training_info,
            "ensemble_weights": config.ENSEMBLE_WEIGHTS,
        }
        with open(models_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Models saved to {models_dir}")

    def load(self, models_dir=None):
        """Load previously trained models."""
        models_dir = Path(models_dir or config.MODELS_DIR)

        # Scorer classifier (may not exist in older saved models)
        scorer_path = models_dir / "scorer_clf.pkl"
        if scorer_path.exists():
            with open(scorer_path, "rb") as f:
                self.scorer_clf = pickle.load(f)
        else:
            self.scorer_clf = None

        with open(models_dir / "goals_poisson.pkl", "rb") as f:
            self.goals_poisson = pickle.load(f)
        with open(models_dir / "goals_gbt.pkl", "rb") as f:
            self.goals_gbt = pickle.load(f)
        with open(models_dir / "behinds_poisson.pkl", "rb") as f:
            self.behinds_poisson = pickle.load(f)
        with open(models_dir / "behinds_gbt.pkl", "rb") as f:
            self.behinds_gbt = pickle.load(f)
        with open(models_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(models_dir / "model_metadata.json") as f:
            metadata = json.load(f)
        self.feature_cols = metadata["feature_cols"]
        self.eval_metrics = metadata.get("eval_metrics", {})
        self.training_info = metadata.get("training_info", {})

        print(f"Models loaded from {models_dir}")
        print(f"  Features: {len(self.feature_cols)}")
        if self.eval_metrics:
            print(f"  Goals MAE: {self.eval_metrics.get('goals_mae', 'N/A')}")

    # ------------------------------------------------------------------
    # Full evaluation report
    # ------------------------------------------------------------------

    def evaluate_detailed(self, df, feature_cols=None):
        """Run detailed evaluation with breakdowns by role, team, etc."""
        feature_cols = feature_cols or self.feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(df, feature_cols, scaler=self.scaler)

        pred_goals, scorer_prob, _ = self._ensemble_predict(
            X_raw, X_scaled, "goals", X_clean=X_clean
        )
        pred_behinds, _, _ = self._ensemble_predict(
            X_raw, X_scaled, "behinds", X_clean=X_clean
        )

        actual_goals = df["GL"].values
        actual_behinds = df["BH"].values
        baseline = df["career_goal_avg_pre"].fillna(0).values

        print("=" * 60)
        print("DETAILED EVALUATION REPORT")
        print("=" * 60)

        # Overall
        print(f"\nOverall ({len(df)} predictions):")
        print(f"  Goals MAE:     {mean_absolute_error(actual_goals, pred_goals):.4f}")
        print(f"  Goals RMSE:    {np.sqrt(mean_squared_error(actual_goals, pred_goals)):.4f}")
        print(f"  Baseline MAE:  {mean_absolute_error(actual_goals, baseline):.4f}")
        print(f"  Behinds MAE:   {mean_absolute_error(actual_behinds, pred_behinds):.4f}")

        # By role
        if "player_role" in df.columns:
            print(f"\nBy player role:")
            for role in df["player_role"].unique():
                mask = df["player_role"] == role
                n = mask.sum()
                if n < 10:
                    continue
                mae = mean_absolute_error(actual_goals[mask], pred_goals[mask])
                bl_mae = mean_absolute_error(actual_goals[mask], baseline[mask])
                print(f"  {role:20s} n={n:5d}  MAE={mae:.4f}  baseline={bl_mae:.4f}")

        # By actual goal count bucket
        print(f"\nBy actual goals scored:")
        for bucket, label in [(0, "0 goals"), (1, "1 goal"), (2, "2 goals"), (3, "3+ goals")]:
            if bucket < 3:
                mask = actual_goals == bucket
            else:
                mask = actual_goals >= bucket
            n = mask.sum()
            if n < 5:
                continue
            mae = mean_absolute_error(actual_goals[mask], pred_goals[mask])
            avg_pred = pred_goals[mask].mean()
            print(f"  {label:10s} n={n:5d}  MAE={mae:.4f}  avg_pred={avg_pred:.2f}")

        return {
            "goals_mae": mean_absolute_error(actual_goals, pred_goals),
            "behinds_mae": mean_absolute_error(actual_behinds, pred_behinds),
            "baseline_mae": mean_absolute_error(actual_goals, baseline),
        }


# ---------------------------------------------------------------------------
# Elo Rating System
# ---------------------------------------------------------------------------

class EloSystem:
    """Running Elo ratings for AFL teams.

    K-factor adapts for finals (higher stakes = more movement).
    Ratings regress toward mean at season start.
    """

    def __init__(self, k_factor=None, home_advantage=None, season_regression=None,
                 initial_rating=1500):
        self.k_factor = k_factor if k_factor is not None else getattr(config, "ELO_K_FACTOR", 30)
        self.home_advantage = home_advantage if home_advantage is not None else getattr(config, "ELO_HOME_ADVANTAGE", 30)
        self.season_regression = season_regression if season_regression is not None else getattr(config, "ELO_SEASON_REGRESSION", 0.5)
        self.initial_rating = initial_rating
        self.ratings = {}  # team → rating
        self._last_year = None

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)

    def expected_win_prob(self, team_rating, opp_rating, is_home=True):
        """Expected win probability using logistic function."""
        diff = team_rating - opp_rating
        if is_home:
            diff += self.home_advantage
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def update(self, team, opponent, margin, is_home=True, is_finals=False,
               year=None):
        """Update ratings after a match. Returns (team_new, opp_new)."""
        # Regress ratings at the start of a new season
        if year is not None and self._last_year is not None and year > self._last_year:
            self._season_regress()
        self._last_year = year

        team_r = self.get_rating(team)
        opp_r = self.get_rating(opponent)

        expected = self.expected_win_prob(team_r, opp_r, is_home)

        # Actual result: 1 for win, 0 for loss, 0.5 for draw
        if margin > 0:
            actual = 1.0
        elif margin < 0:
            actual = 0.0
        else:
            actual = 0.5

        # Margin-scaled K: larger upsets move ratings more
        # MOV multiplier from FiveThirtyEight formula
        # Cap margin at ±60 to reduce blowout distortion
        capped_margin = max(min(abs(margin), 60), 0)
        mov = np.log(capped_margin + 1) * (2.2 / ((team_r - opp_r) * 0.001 + 2.2))
        k = self.k_factor * mov
        if is_finals:
            k *= 1.5

        delta = k * (actual - expected)
        self.ratings[team] = team_r + delta
        self.ratings[opponent] = opp_r - delta

        return self.ratings[team], self.ratings[opponent]

    def _season_regress(self):
        """Regress all ratings toward the mean at season boundary."""
        if not self.ratings:
            return
        mean_r = np.mean(list(self.ratings.values()))
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - self.season_regression)
                + mean_r * self.season_regression
            )

    def compute_all(self, team_match_df):
        """Process all matches chronologically and return ratings per match.

        Args:
            team_match_df: DataFrame with columns [match_id, team, opponent,
                           margin, is_home, is_finals, year, date]

        Returns:
            DataFrame with [match_id, team, elo_pre, elo_post, opp_elo_pre,
                            elo_diff, expected_win_prob]
        """
        # Only process home rows to avoid double-counting
        home_df = team_match_df[team_match_df["is_home"]].sort_values("date").copy()

        records = []
        for _, row in home_df.iterrows():
            team = row["team"]
            opp = row["opponent"]
            team_elo = self.get_rating(team)
            opp_elo = self.get_rating(opp)
            exp_wp = self.expected_win_prob(team_elo, opp_elo, is_home=True)

            # Record pre-match ratings for both teams
            records.append({
                "match_id": row["match_id"],
                "team": team,
                "elo_pre": team_elo,
                "opp_elo_pre": opp_elo,
                "elo_diff": team_elo - opp_elo + self.home_advantage,
                "expected_win_prob": exp_wp,
            })
            records.append({
                "match_id": row["match_id"],
                "team": opp,
                "elo_pre": opp_elo,
                "opp_elo_pre": team_elo,
                "elo_diff": opp_elo - team_elo - self.home_advantage,
                "expected_win_prob": 1 - exp_wp,
            })

            # Update
            self.update(
                team, opp, row["margin"],
                is_home=True,
                is_finals=bool(row.get("is_finals", False)),
                year=int(row.get("year", 0)),
            )

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Game Winner Model
# ---------------------------------------------------------------------------

class AFLGameWinnerModel:
    """Predicts match outcomes (home win / draw / away win).

    Features:
      - Elo ratings and differential
      - Team form metrics (rolling win %, margin, scoring)
      - Head-to-head record
      - Rest days differential
      - Home advantage
      - Venue familiarity
      - Finals flag

    Uses HistGradientBoostingClassifier for speed.
    """

    def __init__(self, gbt_params=None, elo_params=None):
        self.classifier = None
        self.margin_regressor = None
        self.elo_system = EloSystem(**(elo_params or {}))
        self.feature_cols = []
        self.eval_metrics = {}
        self.hybrid_enabled = bool(getattr(config, "WINNER_HYBRID_ENABLED", True))
        self.hybrid_alpha = float(getattr(config, "WINNER_HYBRID_ALPHA", 1.0))
        self.hybrid_beta = float(getattr(config, "WINNER_HYBRID_BETA", 1.0))
        self.hybrid_bias = float(getattr(config, "WINNER_HYBRID_BIAS", 0.0))
        self.market_eps = float(getattr(config, "WINNER_MARKET_EPS", 1e-6))
        # Instance-level params (override config defaults)
        self.gbt_params = gbt_params or getattr(config, "GAME_WINNER_PARAMS_BACKTEST", config.HIST_GBT_PARAMS_BACKTEST)

    def _refresh_hybrid_params(self):
        """Refresh hybrid parameters from config to support runtime toggles."""
        self.hybrid_enabled = bool(getattr(config, "WINNER_HYBRID_ENABLED", True))
        self.hybrid_alpha = float(getattr(config, "WINNER_HYBRID_ALPHA", 1.0))
        self.hybrid_beta = float(getattr(config, "WINNER_HYBRID_BETA", 1.0))
        self.hybrid_bias = float(getattr(config, "WINNER_HYBRID_BIAS", 0.0))
        self.market_eps = float(getattr(config, "WINNER_MARKET_EPS", 1e-6))

    def _market_prior_components(self, game_df):
        """Return market prior probabilities and availability mask."""
        prior = np.full(len(game_df), 0.5, dtype=float)
        available = np.zeros(len(game_df), dtype=bool)
        if "market_home_implied_prob" in game_df.columns:
            raw = pd.to_numeric(game_df["market_home_implied_prob"], errors="coerce").values
            valid = np.isfinite(raw) & (raw >= 0.0) & (raw <= 1.0)
            prior[valid] = raw[valid]
            available = valid
        return prior, available

    def _prob_to_logit(self, probs):
        """Convert probabilities to logits with clipping for numeric safety."""
        eps = max(float(self.market_eps), 1e-12)
        p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _logit_to_prob(logits):
        """Convert logits to probabilities."""
        z = np.clip(np.asarray(logits, dtype=float), -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _combine_hybrid_prob(self, residual_prob, market_prior_prob):
        """Blend market prior and residual model in logit space."""
        if not self.hybrid_enabled:
            return np.asarray(residual_prob, dtype=float)
        market_logit = self._prob_to_logit(market_prior_prob)
        residual_logit = self._prob_to_logit(residual_prob)
        combined = (
            self.hybrid_alpha * market_logit
            + self.hybrid_beta * residual_logit
            + self.hybrid_bias
        )
        return self._logit_to_prob(combined)

    def _predict_prob_components(self, game_df):
        """Predict residual and hybrid probabilities for game winner."""
        if self.classifier is None:
            raise ValueError("Game winner classifier is not trained.")
        self._refresh_hybrid_params()
        X = game_df[self.feature_cols].fillna(0)
        residual_prob = self.classifier.predict_proba(X)[:, 1]
        market_prior_prob, market_available = self._market_prior_components(game_df)
        hybrid_prob = self._combine_hybrid_prob(residual_prob, market_prior_prob)
        return hybrid_prob, residual_prob, market_prior_prob, market_available

    def build_game_features(self, team_match_df, elo_df=None,
                            player_predictions_df=None):
        """Build game-level features from team_match data.

        Args:
            team_match_df: One row per (team, match).
            elo_df: Pre-computed Elo ratings (optional).
            player_predictions_df: Player-level predictions with columns
                [match_id, team, predicted_goals, predicted_disposals].
                If provided, aggregated per team as game-level features.

        Returns one row per match (not per team) with columns for both
        home and away team stats.
        """
        df = team_match_df.sort_values("date").copy()

        if elo_df is None:
            elo_df = self.elo_system.compute_all(df)

        # Merge Elo ratings
        df = df.merge(elo_df[["match_id", "team", "elo_pre", "opp_elo_pre",
                               "elo_diff", "expected_win_prob"]],
                       on=["match_id", "team"], how="left")

        # Rolling team features
        grouped = df.groupby("team", observed=True)

        # Win percentage (last 5, 10)
        df["won"] = (df["margin"] > 0).astype(int)
        for w in [5, 10]:
            df[f"team_win_pct_{w}"] = grouped["won"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )

        # Margin averages
        for w in [5, 10]:
            df[f"team_margin_avg_{w}"] = grouped["margin"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )

        # Scoring averages
        df[f"team_scoring_avg_5"] = grouped["score"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        df[f"team_conceded_avg_5"] = grouped["opp_score"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

        # Form trajectory (slope of margin over last 5)
        df["team_form_trend"] = grouped["margin"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
            )
        ).fillna(0)

        # Head-to-head features
        h2h_group = df.groupby(["team", "opponent"], observed=True)
        df["h2h_win_rate"] = h2h_group["won"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ).fillna(0.5)
        df["h2h_margin_avg"] = h2h_group["margin"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ).fillna(0)
        df["h2h_games"] = h2h_group["won"].transform(
            lambda s: s.shift(1).expanding().count()
        ).fillna(0)

        # Rest days (already computed in team_match_df)
        df["rest_days"] = df["rest_days"].fillna(7)

        # --- Venue-specific home advantage ---
        # Compute historical home win rate at each venue using only past data
        home_matches = df[df["is_home"]].copy()
        home_matches["_home_won"] = (home_matches["margin"] > 0).astype(int)
        home_matches = home_matches.sort_values("date")
        venue_grp = home_matches.groupby("venue", observed=True)
        home_matches["venue_home_win_rate"] = venue_grp["_home_won"].transform(
            lambda s: s.shift(1).expanding(min_periods=5).mean()
        )
        venue_home = home_matches[["match_id", "venue_home_win_rate"]].drop_duplicates(
            subset=["match_id"]
        )
        df = df.merge(venue_home, on="match_id", how="left")
        df["venue_home_win_rate"] = df["venue_home_win_rate"].fillna(0.5)

        # Now pivot to one row per match (home perspective)
        home = df[df["is_home"]].copy()
        away = df[~df["is_home"]].copy()

        # Select feature columns
        feat_cols = [
            "elo_pre", "opp_elo_pre", "elo_diff", "expected_win_prob",
            "team_win_pct_5", "team_win_pct_10",
            "team_margin_avg_5", "team_margin_avg_10",
            "team_scoring_avg_5", "team_conceded_avg_5",
            "team_form_trend",
            "h2h_win_rate", "h2h_margin_avg", "h2h_games",
            "rest_days", "is_finals", "attendance",
        ]

        # Game-level features (not per-team, added after pivot)
        game_level_cols = ["venue_home_win_rate"]

        # Rename away features with 'away_' prefix
        home_feats = home[["match_id", "date", "year", "round_number",
                           "team", "opponent", "venue", "margin"]
                          + feat_cols + game_level_cols].copy()
        home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feat_cols})

        away_feats = away[["match_id"] + feat_cols].copy()
        away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feat_cols})

        game_df = home_feats.merge(away_feats, on="match_id", how="inner")

        # Join market odds features (match-level, one row per match)
        odds_path = Path(config.BASE_STORE_DIR) / "odds.parquet"
        if odds_path.exists():
            odds_df = pd.read_parquet(odds_path)
            odds_cols = [
                "match_id",
                "market_home_implied_prob", "market_away_implied_prob",
                "market_handicap", "market_total_score",
                "market_confidence", "odds_movement_home", "odds_movement_line",
                "betfair_home_implied_prob",
            ]
            odds_df = odds_df[[c for c in odds_cols if c in odds_df.columns]]
            game_df = game_df.merge(odds_df, on="match_id", how="left")

        # --- Aggregated player predictions as game features ---
        player_pred_features = []
        if player_predictions_df is not None:
            pp = player_predictions_df.copy()
            # Aggregate per (match_id, team)
            agg_cols = {}
            if "predicted_goals" in pp.columns:
                agg_cols["predicted_goals"] = "sum"
            if "predicted_disposals" in pp.columns:
                agg_cols["predicted_disposals"] = "sum"

            if agg_cols:
                team_preds = (
                    pp.groupby(["match_id", "team"], observed=True)
                    .agg(**{f"team_{k}": (k, v) for k, v in agg_cols.items()})
                    .reset_index()
                )

                # Merge for home team
                home_preds = team_preds.rename(
                    columns={c: f"home_{c}" for c in team_preds.columns
                             if c.startswith("team_predicted")}
                )
                game_df = game_df.merge(
                    home_preds, left_on=["match_id", "team"],
                    right_on=["match_id", "team"], how="left",
                )

                # Merge for away team
                away_preds = team_preds.rename(
                    columns={"team": "away_team_pp",
                             **{c: f"away_{c}" for c in team_preds.columns
                                if c.startswith("team_predicted")}}
                )
                game_df = game_df.merge(
                    away_preds, left_on=["match_id", "opponent"],
                    right_on=["match_id", "away_team_pp"], how="left",
                )
                game_df = game_df.drop(columns=["away_team_pp"], errors="ignore")

                # Differentials and feature list
                for stat in agg_cols:
                    col = f"team_{stat}"
                    h = f"home_{col}"
                    a = f"away_{col}"
                    d = f"diff_{col}"
                    game_df[h] = game_df[h].fillna(0)
                    game_df[a] = game_df[a].fillna(0)
                    game_df[d] = game_df[h] - game_df[a]
                    player_pred_features.extend([h, a, d])

        # Differential features
        for col in ["elo_pre", "team_win_pct_5", "team_margin_avg_5",
                     "team_scoring_avg_5", "rest_days", "team_form_trend"]:
            game_df[f"diff_{col}"] = game_df[f"home_{col}"] - game_df[f"away_{col}"]

        # Target: home win (1), draw (0.5 treated as loss), away win (0)
        game_df["home_win"] = (game_df["margin"] > 0).astype(int)

        # Feature columns for the model
        model_features = (
            [f"home_{c}" for c in feat_cols]
            + [f"away_{c}" for c in feat_cols]
            + [f"diff_{col}" for col in ["elo_pre", "team_win_pct_5",
                "team_margin_avg_5", "team_scoring_avg_5",
                "rest_days", "team_form_trend"]]
            + player_pred_features
        )

        # Add market odds features (graceful fallback if odds.parquet missing)
        odds_feature_names = [
            "market_home_implied_prob", "market_away_implied_prob",
            "market_handicap", "market_total_score",
            "market_confidence", "odds_movement_home", "odds_movement_line",
            "betfair_home_implied_prob",
        ]
        available_odds = [c for c in odds_feature_names if c in game_df.columns]
        # In hybrid mode, use market implied probabilities as prior (not residual inputs).
        if bool(getattr(config, "WINNER_HYBRID_ENABLED", True)):
            available_odds = [
                c for c in available_odds
                if c not in {"market_home_implied_prob", "market_away_implied_prob"}
            ]

        # Betfair-bookmaker divergence: sharp money signal
        if "betfair_home_implied_prob" in game_df.columns and "market_home_implied_prob" in game_df.columns:
            game_df["market_betfair_divergence"] = (
                game_df["betfair_home_implied_prob"] - game_df["market_home_implied_prob"]
            )
            available_odds = available_odds + ["market_betfair_divergence"]

        model_features = model_features + available_odds

        # Add game-level features (not duplicated per team)
        available_game_level = [c for c in game_level_cols if c in game_df.columns]
        model_features = model_features + available_game_level

        return game_df, model_features

    def train(self, team_match_df, player_predictions_df=None):
        """Train game winner model on team-match data."""
        print("Building game-level features...")
        elo_df = self.elo_system.compute_all(team_match_df)
        game_df, model_features = self.build_game_features(
            team_match_df, elo_df,
            player_predictions_df=player_predictions_df,
        )

        self.feature_cols = model_features

        # Time-based split
        train = game_df[game_df["year"] < config.VALIDATION_YEAR].copy()
        val = game_df[game_df["year"] == config.VALIDATION_YEAR].copy()

        if val.empty or train.empty:
            print("  Not enough data for train/val split")
            return {}

        X_train = train[model_features].fillna(0)
        X_val = val[model_features].fillna(0)
        y_train = train["home_win"].values
        y_val = val["home_win"].values

        print(f"  Training: {len(train)} games, Validation: {len(val)} games")

        gw_params = getattr(config, "GAME_WINNER_PARAMS", {
            "max_iter": 200, "max_depth": 4, "learning_rate": 0.05,
            "min_samples_leaf": 10, "random_state": config.RANDOM_SEED,
        })
        self.classifier = HistGradientBoostingClassifier(**gw_params)
        self.classifier.fit(X_train, y_train)
        self._refresh_hybrid_params()

        # Evaluate (residual + hybrid paths)
        residual_prob = self.classifier.predict_proba(X_val)[:, 1]
        market_prior_prob, market_available = self._market_prior_components(val)
        hybrid_prob = self._combine_hybrid_prob(residual_prob, market_prior_prob)

        residual_class = (residual_prob > 0.5).astype(int)
        hybrid_class = (hybrid_prob > 0.5).astype(int)
        residual_acc = float(np.mean(residual_class == y_val))
        accuracy = float(np.mean(hybrid_class == y_val))

        try:
            residual_auc = float(roc_auc_score(y_val, residual_prob))
        except ValueError:
            residual_auc = float("nan")
        try:
            auc = float(roc_auc_score(y_val, hybrid_prob))
        except ValueError:
            auc = float("nan")
        residual_brier = float(np.mean((residual_prob - y_val) ** 2))
        hybrid_brier = float(np.mean((hybrid_prob - y_val) ** 2))

        # Elo baseline accuracy
        elo_col = "home_expected_win_prob"
        if elo_col in val.columns:
            elo_pred = (val[elo_col].fillna(0.5) > 0.5).astype(int)
            elo_acc = np.mean(elo_pred == y_val)
        else:
            elo_acc = float("nan")

        # Market baseline comparison (if odds features present)
        market_acc = float("nan")
        market_auc = float("nan")
        if market_available.sum() > 10:
            mkt_prob = market_prior_prob
            mkt_pred = (mkt_prob > 0.5).astype(int)
            market_acc = float(np.mean(mkt_pred[market_available] == y_val[market_available]))
            try:
                market_auc = float(roc_auc_score(y_val[market_available], mkt_prob[market_available]))
            except ValueError:
                pass
            market_brier = float(np.mean((mkt_prob[market_available] - y_val[market_available]) ** 2))
        else:
            market_brier = float("nan")

        self.eval_metrics = {
            "accuracy": accuracy,
            "auc": auc if not np.isnan(auc) else None,
            "residual_accuracy": residual_acc,
            "residual_auc": residual_auc if not np.isnan(residual_auc) else None,
            "hybrid_brier": hybrid_brier,
            "residual_brier": residual_brier,
            "elo_accuracy": elo_acc if not np.isnan(elo_acc) else None,
            "market_accuracy": market_acc if not np.isnan(market_acc) else None,
            "market_auc": market_auc if not np.isnan(market_auc) else None,
            "market_brier": market_brier if not np.isnan(market_brier) else None,
            "hybrid_enabled": bool(self.hybrid_enabled),
            "market_coverage": float(market_available.mean()),
            "n_val_games": len(val),
            "n_features": len(model_features),
        }

        print(f"  Game Winner (hybrid) acc: {accuracy:.3f}  AUC: {auc:.4f}  Brier: {hybrid_brier:.4f}")
        print(f"  Residual-only acc:         {residual_acc:.3f}  AUC: {residual_auc:.4f}  Brier: {residual_brier:.4f}")
        if not np.isnan(elo_acc):
            print(f"  Elo baseline acc:     {elo_acc:.3f}")
        if not np.isnan(market_acc):
            print(f"  Market baseline acc:  {market_acc:.3f}  AUC: {market_auc:.4f}  Brier: {market_brier:.4f}")
            edge = accuracy - market_acc
            print(f"  Model edge vs market: {edge:+.3f}")

        # Top features
        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print("  Top 10 game winner features:")
            for i in top_idx:
                print(f"    {model_features[i]:40s} {importances[i]:.4f}")

        return self.eval_metrics

    def predict(self, team_match_df, upcoming_matches=None,
                player_predictions_df=None, store=None):
        """Predict game winners.

        If upcoming_matches provided, uses that. Otherwise predicts on
        the latest available data.
        """
        elo_df = self.elo_system.compute_all(team_match_df)
        game_df, _ = self.build_game_features(
            team_match_df, elo_df,
            player_predictions_df=player_predictions_df,
        )

        if upcoming_matches is not None:
            pred_df = upcoming_matches
        else:
            pred_df = game_df

        hybrid_prob, residual_prob, market_prior_prob, market_available = (
            self._predict_prob_components(pred_df)
        )

        # Apply isotonic calibration to hybrid prob if available
        if store is not None and getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            if calibrator is not None and calibrator.has_calibrator("game_winner"):
                hybrid_prob = calibrator.transform("game_winner", hybrid_prob)

        result = pred_df[["match_id", "team", "opponent", "venue"]].copy()
        result["market_prior_prob_home"] = np.round(market_prior_prob, 4)
        result["residual_prob_home"] = np.round(residual_prob, 4)
        result["hybrid_prob_home"] = np.round(hybrid_prob, 4)
        result["market_prior_available"] = market_available.astype(int)
        result["home_win_prob"] = np.round(hybrid_prob, 4)
        result["away_win_prob"] = np.round(1 - hybrid_prob, 4)
        result["predicted_winner"] = np.where(
            hybrid_prob > 0.5, result["team"], result["opponent"]
        )

        return result

    def train_backtest(self, team_match_df, player_predictions_df=None):
        """Fast training for backtest/sequential loop.

        Trains both a classifier (win/loss) and regressor (margin) using
        HistGBT for speed. Returns the EloSystem for reuse.
        """
        elo_df = self.elo_system.compute_all(team_match_df)
        game_df, model_features = self.build_game_features(
            team_match_df, elo_df,
            player_predictions_df=player_predictions_df,
        )
        self.feature_cols = model_features

        X = game_df[model_features].fillna(0)
        y_win = game_df["home_win"].values
        y_margin = game_df["margin"].values

        self.classifier = HistGradientBoostingClassifier(**self.gbt_params)
        self.classifier.fit(X, y_win)
        self._refresh_hybrid_params()

        self.margin_regressor = HistGradientBoostingRegressor(**self.gbt_params)
        self.margin_regressor.fit(X, y_margin)

        return self.elo_system

    def predict_with_margin(self, team_match_df, player_predictions_df=None,
                            store=None):
        """Predict game winners with margin estimates.

        Returns DataFrame with: match_id, home_team, away_team, venue,
        home_win_prob, away_win_prob, predicted_margin, predicted_winner,
        home_elo, away_elo.
        """
        elo_df = self.elo_system.compute_all(team_match_df)
        game_df, _ = self.build_game_features(
            team_match_df, elo_df,
            player_predictions_df=player_predictions_df,
        )

        if game_df.empty or self.classifier is None:
            return pd.DataFrame()

        missing = [c for c in self.feature_cols if c not in game_df.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} game winner feature columns: {missing[:10]}")

        hybrid_prob, residual_prob, market_prior_prob, market_available = (
            self._predict_prob_components(game_df)
        )

        # Apply isotonic calibration to hybrid prob if available
        if store is not None and getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            if calibrator is not None and calibrator.has_calibrator("game_winner"):
                hybrid_prob = calibrator.transform("game_winner", hybrid_prob)
        X = game_df[self.feature_cols].fillna(0)

        # Margin prediction
        if self.margin_regressor is not None:
            pred_margin = self.margin_regressor.predict(X)
        else:
            pred_margin = np.zeros(len(game_df))

        result = pd.DataFrame({
            "match_id": game_df["match_id"].values,
            "home_team": game_df["team"].values,
            "away_team": game_df["opponent"].values,
            "venue": game_df["venue"].values,
            "market_prior_prob_home": np.round(market_prior_prob, 4),
            "residual_prob_home": np.round(residual_prob, 4),
            "hybrid_prob_home": np.round(hybrid_prob, 4),
            "market_prior_available": market_available.astype(int),
            "home_win_prob": np.round(hybrid_prob, 4),
            "away_win_prob": np.round(1 - hybrid_prob, 4),
            "predicted_margin": np.round(pred_margin, 1),
            "predicted_winner": np.where(
                hybrid_prob > 0.5,
                game_df["team"].values,
                game_df["opponent"].values,
            ),
            "home_elo": game_df["home_elo_pre"].values if "home_elo_pre" in game_df.columns else 1500.0,
            "away_elo": game_df["away_elo_pre"].values if "away_elo_pre" in game_df.columns else 1500.0,
        })

        return result

    def save(self, models_dir=None):
        """Save game winner model."""
        models_dir = Path(models_dir or config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        with open(models_dir / "game_winner_clf.pkl", "wb") as f:
            pickle.dump(self.classifier, f)
        with open(models_dir / "elo_system.pkl", "wb") as f:
            pickle.dump(self.elo_system, f)
        if self.margin_regressor is not None:
            with open(models_dir / "game_winner_margin.pkl", "wb") as f:
                pickle.dump(self.margin_regressor, f)

        metadata = {
            "feature_cols": self.feature_cols,
            "eval_metrics": self.eval_metrics,
            "elo_ratings": {k: round(v, 1) for k, v in self.elo_system.ratings.items()},
            "has_margin_regressor": self.margin_regressor is not None,
            "hybrid": {
                "enabled": bool(self.hybrid_enabled),
                "alpha": float(self.hybrid_alpha),
                "beta": float(self.hybrid_beta),
                "bias": float(self.hybrid_bias),
                "market_eps": float(self.market_eps),
            },
        }
        with open(models_dir / "game_winner_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Game winner model saved to {models_dir}")

    def load(self, models_dir=None):
        """Load game winner model."""
        models_dir = Path(models_dir or config.MODELS_DIR)

        with open(models_dir / "game_winner_clf.pkl", "rb") as f:
            self.classifier = pickle.load(f)
        with open(models_dir / "elo_system.pkl", "rb") as f:
            self.elo_system = pickle.load(f)

        margin_path = models_dir / "game_winner_margin.pkl"
        if margin_path.exists():
            with open(margin_path, "rb") as f:
                self.margin_regressor = pickle.load(f)
        else:
            self.margin_regressor = None

        with open(models_dir / "game_winner_metadata.json") as f:
            metadata = json.load(f)
        self.feature_cols = metadata["feature_cols"]
        self.eval_metrics = metadata.get("eval_metrics", {})
        print(f"Game winner model loaded from {models_dir}")


class AFLDisposalModel:
    """Model for predicting player disposals per match.

    Architecture:
      - Poisson + GBT ensemble for E[disposals]
      - Probability thresholds from configurable distribution CDF

    Supported distributions for threshold probabilities:
      - 'poisson':  P(X>=k) = 1 - Poisson.CDF(k-1, lambda)
      - 'gaussian': P(X>=k) = 1 - Norm.CDF(k-0.5, mu, sigma)  [continuity corrected]
      - 'negbin':   P(X>=k) = 1 - NegBin.CDF(k-1, r, p)

    Disposals are more predictable than goals (higher volume, lower variance),
    so we use a single-stage regression (no scorer classifier needed).
    """

    def __init__(self, distribution="poisson", gbt_params=None, poisson_params=None,
                 ensemble_weights=None):
        self.distribution = distribution  # 'poisson', 'gaussian', 'negbin'
        self.disp_poisson = None
        self.disp_gbt = None
        self.scaler = None
        self.feature_cols = []
        self.eval_metrics = {}
        self.training_info = {}
        # Distribution-specific parameters (estimated from training residuals)
        self._residual_std = None   # for Gaussian: global residual std
        self._std_params = None     # (a, b) for std(mu) = a + b*sqrt(mu) (optional)
        self._negbin_r = None       # for NegBin: dispersion parameter r
        # Instance-level params (override config defaults)
        self.gbt_params = gbt_params or getattr(config, "DISPOSAL_GBT_PARAMS_BACKTEST", config.HIST_GBT_PARAMS_BACKTEST)
        self.poisson_params = poisson_params or getattr(config, "DISPOSAL_POISSON_PARAMS", config.POISSON_PARAMS)
        self.ensemble_weights = ensemble_weights or config.ENSEMBLE_WEIGHTS

    def train(self, df, feature_cols=None):
        """Train disposal prediction models."""
        if feature_cols is None:
            feat_path = config.FEATURES_DIR / "feature_columns.json"
            if feat_path.exists():
                with open(feat_path) as f:
                    feature_cols = json.load(f)
            else:
                raise ValueError("No feature_cols provided and no feature_columns.json found")

        self.feature_cols = feature_cols

        # Time-based split
        train_df = df[df["year"] < config.VALIDATION_YEAR].copy()
        val_df = df[df["year"] == config.VALIDATION_YEAR].copy()

        if val_df.empty and len(train_df) > 0:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()

        print(f"Training disposal model: {len(train_df)} train, {len(val_df)} val rows")

        y_train = train_df["DI"].values
        y_val = val_df["DI"].values
        weights_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

        self.scaler = StandardScaler()
        X_train_raw, X_train_clean, X_train_scaled = _prepare_features(
            train_df, feature_cols, scaler=self.scaler, fit_scaler=True
        )
        X_val_raw, X_val_clean, X_val_scaled = _prepare_features(
            val_df, feature_cols, scaler=self.scaler
        )

        # Poisson regressor
        print("  Training Poisson regressor for disposals...")
        self.disp_poisson = PoissonRegressor(
            alpha=self.poisson_params["alpha"],
            max_iter=self.poisson_params["max_iter"],
        )
        self.disp_poisson.fit(X_train_scaled, y_train, sample_weight=weights_train)

        # GBT regressor
        print("  Training GBT regressor for disposals...")
        # Use HistGBT consistently with backtest path and instance-level params.
        self.disp_gbt = HistGradientBoostingRegressor(**self.gbt_params)
        self.disp_gbt.fit(X_train_raw, y_train, sample_weight=weights_train)

        # Estimate distribution-specific parameters from training data
        if self.distribution != "poisson":
            self._estimate_distribution_params(X_train_raw, X_train_scaled, y_train)

        # Evaluate
        pred = self._predict_raw(X_val_raw, X_val_scaled)
        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        baseline = y_train.mean()
        baseline_mae = mean_absolute_error(y_val, np.full_like(y_val, baseline, dtype=float))

        self.eval_metrics = {
            "disp_mae": mae,
            "disp_rmse": rmse,
            "disp_baseline_mae": baseline_mae,
        }
        self.training_info = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "n_features": len(feature_cols),
        }

        print(f"  Disposals MAE: {mae:.4f}  RMSE: {rmse:.4f}  Baseline: {baseline_mae:.4f}")

        # Top features
        if hasattr(self.disp_gbt, "feature_importances_"):
            importances = self.disp_gbt.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print("  Top 10 disposal features:")
            for i in top_idx:
                print(f"    {feature_cols[i]:40s} {importances[i]:.4f}")

        return self.eval_metrics

    def train_backtest(self, df, feature_cols):
        """Fast training for backtest loop."""
        self.feature_cols = feature_cols
        hist_params = self.gbt_params

        y = df["DI"].values
        weights = df["sample_weight"].values if "sample_weight" in df.columns else None

        self.scaler = StandardScaler()
        X_raw, X_clean, X_scaled = _prepare_features(
            df, feature_cols, scaler=self.scaler, fit_scaler=True
        )

        # Poisson component
        self.disp_poisson = PoissonRegressor(
            alpha=self.poisson_params["alpha"],
            max_iter=self.poisson_params["max_iter"],
        )
        self.disp_poisson.fit(X_scaled, y, sample_weight=weights)

        # Single-model GBT
        self.disp_gbt = HistGradientBoostingRegressor(**hist_params)
        self.disp_gbt.fit(X_raw, y, sample_weight=weights)

        # Estimate distribution-specific parameters
        if self.distribution != "poisson":
            self._estimate_distribution_params(X_raw, X_scaled, y)

    def _estimate_distribution_params(self, X_raw, X_scaled, y_actual):
        """Estimate distribution-specific parameters from training residuals."""
        pred = self._predict_raw(X_raw, X_scaled)
        residuals = y_actual - pred

        if self.distribution == "gaussian":
            # Estimate residual std — use per-prediction std model:
            # Group by predicted level (binned) to estimate heteroscedastic std
            self._residual_std = float(np.std(residuals))
            # Also estimate std as function of predicted mean for better calibration
            # std ≈ a + b * sqrt(predicted)
            from scipy.optimize import curve_fit
            # Bin predictions and compute std per bin
            pred_clipped = np.clip(pred, 1, None)
            bins = np.percentile(pred_clipped, np.arange(0, 101, 10))
            bins = np.unique(bins)
            bin_idx = np.digitize(pred_clipped, bins)
            bin_stds = []
            bin_means = []
            for b in range(1, len(bins)):
                mask = bin_idx == b
                if mask.sum() > 30:
                    bin_stds.append(np.std(residuals[mask]))
                    bin_means.append(np.mean(pred_clipped[mask]))
            if len(bin_stds) >= 3:
                # Fit std = a + b * sqrt(mean)
                bin_means = np.array(bin_means)
                bin_stds = np.array(bin_stds)
                try:
                    def std_model(x, a, b):
                        return a + b * np.sqrt(x)
                    popt, _ = curve_fit(std_model, bin_means, bin_stds, p0=[2.0, 0.5])
                    self._std_params = (float(popt[0]), float(popt[1]))
                except Exception:
                    self._std_params = None
            else:
                self._std_params = None

        elif self.distribution == "negbin":
            # Estimate NegBin dispersion parameter r from residuals
            # Var(Y) = mu + mu^2/r  →  r = mu^2 / (Var(Y) - mu)
            # Estimate per-prediction variance
            pred_clipped = np.clip(pred, 0.5, None)
            overall_var = np.var(y_actual)
            overall_mean = np.mean(y_actual)
            if overall_var > overall_mean:
                self._negbin_r = float(overall_mean ** 2 / (overall_var - overall_mean))
                print(f"  NegBin r = {self._negbin_r:.2f} (var={overall_var:.2f}, mean={overall_mean:.2f})")
            else:
                self._negbin_r = 100.0
                print(f"  NegBin: var ({overall_var:.2f}) <= mean ({overall_mean:.2f}), using r=100 (≈Poisson)")

    def _get_std_for_mu(self, mu):
        """Get estimated std for a given predicted mean (Gaussian distribution)."""
        if self._std_params is not None:
            a, b = self._std_params
            return max(a + b * np.sqrt(max(mu, 0.1)), 1.0)
        if self._residual_std is not None:
            return max(float(self._residual_std), 1.0)
        # Fallback when the model is untrained or metadata is missing.
        return max(1.0, 0.5 * np.sqrt(max(mu, 0.1)) + 3.0)

    def _gaussian_tail_prob(self, mu, threshold, sigma):
        """Gaussian disposal probability with optional upper-tail correction."""
        from scipy.stats import norm, skewnorm

        x = threshold - 0.5  # continuity correction for discrete threshold
        p_base = 1.0 - norm.cdf(x, loc=mu, scale=sigma)

        enabled = bool(getattr(config, "DISPOSAL_UPPER_TAIL_ENABLED", False))
        tail_thresholds = set(getattr(config, "DISPOSAL_UPPER_TAIL_THRESHOLDS", []))
        if not enabled or threshold not in tail_thresholds:
            return float(np.clip(p_base, 0.0, 1.0))

        std_mult = float(getattr(config, "DISPOSAL_UPPER_TAIL_STD_MULTIPLIER", 1.0))
        skew_alpha = float(getattr(config, "DISPOSAL_UPPER_TAIL_SKEW_ALPHA", 0.0))
        sigma_tail = max(sigma * std_mult, 1e-6)
        p_tail = 1.0 - skewnorm.cdf(x, a=skew_alpha, loc=mu, scale=sigma_tail)

        # Guardrail: correction should only widen the upper tail.
        p = max(p_base, p_tail)

        # 30+ tends to be underpredicted in the bulk but overconfident at the extreme tail.
        # Apply a light scale-up with a hard cap to improve Brier at 30+.
        if threshold == 30:
            scale = float(getattr(config, "DISPOSAL_30PLUS_PROB_SCALE", 1.0))
            cap = float(getattr(config, "DISPOSAL_30PLUS_PROB_CAP", 1.0))
            p = min(cap, p * scale)

        return float(np.clip(p, 0.0, 1.0))

    def _threshold_prob(self, mu, threshold):
        """Compute P(X >= threshold) using the configured distribution."""
        mu = max(mu, 0.01)
        if self.distribution == "poisson":
            return 1 - poisson.cdf(threshold - 1, mu)
        elif self.distribution == "gaussian":
            sigma = self._get_std_for_mu(mu)
            return self._gaussian_tail_prob(mu, threshold, sigma)
        elif self.distribution == "negbin":
            from scipy.stats import nbinom
            r = self._negbin_r if self._negbin_r is not None else 100.0
            # scipy nbinom: P(X=k) with params (n=r, p=r/(r+mu))
            p_success = r / (r + mu)
            return 1 - nbinom.cdf(threshold - 1, r, p_success)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _confidence_interval(self, mu, lo_q=0.10, hi_q=0.90):
        """Compute confidence interval quantiles using configured distribution."""
        mu = max(mu, 0.01)
        if self.distribution == "poisson":
            return int(poisson.ppf(lo_q, mu)), int(poisson.ppf(hi_q, mu))
        elif self.distribution == "gaussian":
            from scipy.stats import norm
            sigma = self._get_std_for_mu(mu)
            return max(0, int(norm.ppf(lo_q, mu, sigma))), int(norm.ppf(hi_q, mu, sigma))
        elif self.distribution == "negbin":
            from scipy.stats import nbinom
            r = self._negbin_r if self._negbin_r is not None else 100.0
            p_success = r / (r + mu)
            return int(nbinom.ppf(lo_q, r, p_success)), int(nbinom.ppf(hi_q, r, p_success))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _predict_raw(self, X_raw, X_scaled, df=None):
        """Raw ensemble prediction for disposals.

        When df is provided with market_total_score, Poisson component is
        blended with a market-pace proxy for disposal volume.
        """
        w_poi = self.ensemble_weights["poisson"]
        w_gbt = self.ensemble_weights["gbt"]
        blend = getattr(config, "MARKET_POISSON_BLEND", 0.0)

        pred_poi = self.disp_poisson.predict(X_scaled)

        # Market blend: use total_score as game-pace proxy
        if blend > 0 and df is not None and "market_total_score" in df.columns:
            total_score = df["market_total_score"].values.astype(float)
            # Average game total is ~170 points; disposals scale ~linearly with pace
            avg_total = 170.0
            pace_factor = total_score / avg_total
            has_market = ~np.isnan(total_score) & (total_score > 0)
            # Scale Poisson prediction by pace factor
            market_adj_poi = pred_poi * pace_factor
            pred_poi = np.where(
                has_market,
                (1 - blend) * pred_poi + blend * market_adj_poi,
                pred_poi,
            )

        X_clean = X_raw.fillna(0) if hasattr(X_raw, "fillna") else X_raw
        pred_gbt = _predict_with_compatible_input(self.disp_gbt, X_raw, X_clean)
        return np.clip(w_poi * pred_poi + w_gbt * pred_gbt, 0, None)

    def predict(self, df, feature_cols=None, store=None):
        """Generate disposal predictions with probability thresholds.

        Args:
            df: DataFrame with features.
            feature_cols: list of feature column names.
            store: LearningStore instance for isotonic calibration (optional).
        """
        feature_cols = feature_cols or self.feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(df, feature_cols, scaler=self.scaler)

        pred_disp = self._predict_raw(X_raw, X_scaled, df=df)

        round_col = "round_number" if "round_number" in df.columns else "round"
        result = df[["player", "team", "opponent", "venue", round_col]].copy()
        if round_col != "round":
            result = result.rename(columns={round_col: "round"})

        result["predicted_disposals"] = np.round(pred_disp, 2)

        # Probability thresholds from configured distribution CDF
        thresholds = config.MODEL_TARGETS["disposals"]["thresholds"]
        for t in thresholds:
            result[f"p_{t}plus_disp"] = np.round(
                [self._threshold_prob(mu, t) for mu in pred_disp], 4
            )

        # Apply isotonic calibration if available
        if store is not None and getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            if calibrator is not None:
                for t in thresholds:
                    tgt = f"{t}plus_disp"
                    col = f"p_{t}plus_disp"
                    if calibrator.has_calibrator(tgt) and col in result.columns:
                        result[col] = np.round(calibrator.transform(tgt, result[col].values), 4)

        _enforce_non_increasing_probs(
            result,
            [f"p_{t}plus_disp" for t in thresholds],
            round_dp=4,
        )

        # Confidence intervals (80%)
        ci = [self._confidence_interval(mu) for mu in pred_disp]
        result["conf_lower_di"] = [c[0] for c in ci]
        result["conf_upper_di"] = [c[1] for c in ci]

        if "player_role" in df.columns:
            result["player_role"] = df["player_role"].values

        result = result.sort_values("predicted_disposals", ascending=False).reset_index(drop=True)
        return result

    def predict_distributions(self, df, store=None, feature_cols=None):
        """Generate disposal predictions with expanded thresholds and calibration.

        Returns per-player disposal predictions with calibrated lambda,
        probability thresholds for 10+/15+/20+/25+/30+, and confidence intervals.
        """
        feature_cols = feature_cols or self.feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(df, feature_cols, scaler=self.scaler)
        pred_disp = self._predict_raw(X_raw, X_scaled, df=df)

        round_col = "round_number" if "round_number" in df.columns else "round"
        result = pd.DataFrame({
            "player": df["player"].values,
            "team": df["team"].values,
            "match_id": df["match_id"].values,
        })

        # Calibrate lambda
        lambda_disp = np.zeros(len(df))
        cal_method = getattr(config, "CALIBRATION_METHOD", "bucket")
        for i in range(len(df)):
            raw_lam = max(pred_disp[i], 0.001)
            # Bucket-based lambda calibration is a heuristic; disable when isotonic is used.
            if store is not None and cal_method != "isotonic":
                lambda_disp[i] = store.get_lambda_calibration("disposals", raw_lam)
            else:
                lambda_disp[i] = raw_lam

        result["predicted_disposals"] = np.round(pred_disp, 2)
        result["lambda_disposals"] = np.round(lambda_disp, 4)

        # Expanded thresholds using configured distribution
        thresholds = config.DISPOSAL_THRESHOLDS
        for t in thresholds:
            raw_probs = np.array([self._threshold_prob(mu, t) for mu in lambda_disp])
            result[f"p_{t}plus_disp"] = np.round(raw_probs, 4)

        # Apply isotonic calibration if available (replaces heuristic 30+ adjustments)
        if store is not None and getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            skip_targets = getattr(config, "ISOTONIC_SKIP_TARGETS", set())
            if calibrator is not None:
                for t in thresholds:
                    tgt = f"{t}plus_disp"
                    col = f"p_{t}plus_disp"
                    if tgt in skip_targets:
                        continue
                    if calibrator.has_calibrator(tgt):
                        result[col] = np.round(
                            calibrator.transform(tgt, result[col].values), 4
                        )

        _enforce_non_increasing_probs(
            result,
            [f"p_{t}plus_disp" for t in thresholds],
            round_dp=4,
        )

        # Confidence intervals (80%)
        ci = [self._confidence_interval(mu) for mu in lambda_disp]
        result["conf_lower_di"] = [c[0] for c in ci]
        result["conf_upper_di"] = [c[1] for c in ci]

        return result

    def save(self, models_dir=None):
        """Save trained disposal models."""
        models_dir = Path(models_dir or config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        with open(models_dir / "disp_poisson.pkl", "wb") as f:
            pickle.dump(self.disp_poisson, f)
        with open(models_dir / "disp_gbt.pkl", "wb") as f:
            pickle.dump(self.disp_gbt, f)
        with open(models_dir / "disp_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        metadata = {
            "feature_cols": self.feature_cols,
            "eval_metrics": self.eval_metrics,
            "training_info": self.training_info,
            "distribution": self.distribution,
            "residual_std": self._residual_std,
            "std_params": self._std_params if hasattr(self, '_std_params') else None,
            "negbin_r": self._negbin_r,
        }
        with open(models_dir / "disp_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Disposal models saved to {models_dir}")

    def load(self, models_dir=None):
        """Load previously trained disposal models."""
        models_dir = Path(models_dir or config.MODELS_DIR)

        with open(models_dir / "disp_poisson.pkl", "rb") as f:
            self.disp_poisson = pickle.load(f)
        with open(models_dir / "disp_gbt.pkl", "rb") as f:
            self.disp_gbt = pickle.load(f)
        with open(models_dir / "disp_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(models_dir / "disp_model_metadata.json") as f:
            metadata = json.load(f)
        self.feature_cols = metadata["feature_cols"]
        self.eval_metrics = metadata.get("eval_metrics", {})
        self.training_info = metadata.get("training_info", {})
        self.distribution = metadata.get("distribution", "poisson")
        self._residual_std = metadata.get("residual_std")
        self._std_params = metadata.get("std_params")
        self._negbin_r = metadata.get("negbin_r")

        print(f"Disposal models loaded from {models_dir}")


class AFLMarksModel:
    """Model for predicting player marks per match.

    Architecture:
      - Poisson + GBT ensemble for E[marks]
      - Probability thresholds from configurable distribution CDF

    Supported distributions for threshold probabilities:
      - 'poisson':  P(X>=k) = 1 - Poisson.CDF(k-1, lambda)
      - 'gaussian': P(X>=k) = 1 - Norm.CDF(k-0.5, mu, sigma)  [continuity corrected]
      - 'negbin':   P(X>=k) = 1 - NegBin.CDF(k-1, r, p)

    Marks have lower volume and variance than disposals (mean ~4, std ~2.6).
    """

    def __init__(self, distribution="poisson", gbt_params=None, poisson_params=None,
                 ensemble_weights=None):
        self.distribution = distribution
        self.marks_poisson = None
        self.marks_gbt = None
        self.mark_taker_clf = None
        self._mark_taker_threshold = getattr(config, "MARKS_TAKER_THRESHOLD", 5)
        self._mark_taker_blend = getattr(config, "MARKS_TAKER_BLEND", 0.3)
        self.scaler = None
        self.feature_cols = []
        self.eval_metrics = {}
        self.training_info = {}
        self._residual_std = None
        self._std_params = None
        self._negbin_r = None
        self.gbt_params = gbt_params or getattr(config, "MARKS_GBT_PARAMS_BACKTEST", config.HIST_GBT_PARAMS_BACKTEST)
        self.poisson_params = poisson_params or getattr(config, "MARKS_POISSON_PARAMS", config.POISSON_PARAMS)
        self.ensemble_weights = ensemble_weights or config.ENSEMBLE_WEIGHTS

    def train(self, df, feature_cols=None):
        """Train marks prediction models."""
        if feature_cols is None:
            feat_path = config.FEATURES_DIR / "feature_columns.json"
            if feat_path.exists():
                with open(feat_path) as f:
                    feature_cols = json.load(f)
            else:
                raise ValueError("No feature_cols provided and no feature_columns.json found")

        self.feature_cols = feature_cols

        train_df = df[df["year"] < config.VALIDATION_YEAR].copy()
        val_df = df[df["year"] == config.VALIDATION_YEAR].copy()

        if val_df.empty and len(train_df) > 0:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()

        print(f"Training marks model: {len(train_df)} train, {len(val_df)} val rows")

        y_train = train_df["MK"].values
        y_val = val_df["MK"].values
        weights_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None

        self.scaler = StandardScaler()
        X_train_raw, X_train_clean, X_train_scaled = _prepare_features(
            train_df, feature_cols, scaler=self.scaler, fit_scaler=True
        )
        X_val_raw, X_val_clean, X_val_scaled = _prepare_features(
            val_df, feature_cols, scaler=self.scaler
        )

        print("  Training Poisson regressor for marks...")
        self.marks_poisson = PoissonRegressor(
            alpha=self.poisson_params["alpha"],
            max_iter=self.poisson_params["max_iter"],
        )
        self.marks_poisson.fit(X_train_scaled, y_train, sample_weight=weights_train)

        print("  Training GBT regressor for marks...")
        self.marks_gbt = HistGradientBoostingRegressor(**self.gbt_params)
        self.marks_gbt.fit(X_train_raw, y_train, sample_weight=weights_train)

        # Train mark-taker classifier (soft two-stage)
        if self._mark_taker_blend > 0:
            y_mt = (y_train >= self._mark_taker_threshold).astype(int)
            mt_rate = y_mt.mean()
            print(f"  Training mark-taker classifier (>={self._mark_taker_threshold}): {mt_rate:.1%} positive rate")
            self.mark_taker_clf = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, random_state=config.RANDOM_SEED,
            )
            self.mark_taker_clf.fit(X_train_raw, y_mt, sample_weight=weights_train)
            if len(X_val_raw) > 0:
                y_mt_val = (y_val >= self._mark_taker_threshold).astype(int)
                mt_val_prob = _predict_proba_with_compatible_input(
                    self.mark_taker_clf, X_val_raw, X_val_raw.fillna(0) if hasattr(X_val_raw, "fillna") else X_val_raw
                )
                from sklearn.metrics import roc_auc_score
                try:
                    mt_auc = roc_auc_score(y_mt_val, mt_val_prob)
                    print(f"  Mark-taker AUC: {mt_auc:.4f}")
                except Exception:
                    pass

        if self.distribution != "poisson":
            self._estimate_distribution_params(X_train_raw, X_train_scaled, y_train)

        pred = self._predict_raw(X_val_raw, X_val_scaled)
        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        baseline = y_train.mean()
        baseline_mae = mean_absolute_error(y_val, np.full_like(y_val, baseline, dtype=float))

        self.eval_metrics = {
            "marks_mae": mae,
            "marks_rmse": rmse,
            "marks_baseline_mae": baseline_mae,
        }
        self.training_info = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "n_features": len(feature_cols),
        }

        print(f"  Marks MAE: {mae:.4f}  RMSE: {rmse:.4f}  Baseline: {baseline_mae:.4f}")

        if hasattr(self.marks_gbt, "feature_importances_"):
            importances = self.marks_gbt.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print("  Top 10 marks features:")
            for i in top_idx:
                print(f"    {feature_cols[i]:40s} {importances[i]:.4f}")

        return self.eval_metrics

    def train_backtest(self, df, feature_cols):
        """Fast training for backtest loop."""
        self.feature_cols = feature_cols
        hist_params = self.gbt_params

        y = df["MK"].values
        weights = df["sample_weight"].values if "sample_weight" in df.columns else None

        self.scaler = StandardScaler()
        X_raw, X_clean, X_scaled = _prepare_features(
            df, feature_cols, scaler=self.scaler, fit_scaler=True
        )

        self.marks_poisson = PoissonRegressor(
            alpha=self.poisson_params["alpha"],
            max_iter=self.poisson_params["max_iter"],
        )
        self.marks_poisson.fit(X_scaled, y, sample_weight=weights)

        self.marks_gbt = HistGradientBoostingRegressor(**hist_params)
        self.marks_gbt.fit(X_raw, y, sample_weight=weights)

        # Train mark-taker classifier (soft two-stage)
        if self._mark_taker_blend > 0:
            y_mt = (y >= self._mark_taker_threshold).astype(int)
            self.mark_taker_clf = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, random_state=config.RANDOM_SEED,
            )
            self.mark_taker_clf.fit(X_raw, y_mt, sample_weight=weights)

        if self.distribution != "poisson":
            self._estimate_distribution_params(X_raw, X_scaled, y)

    def _estimate_distribution_params(self, X_raw, X_scaled, y_actual):
        """Estimate distribution-specific parameters from training residuals."""
        pred = self._predict_raw(X_raw, X_scaled)
        residuals = y_actual - pred

        if self.distribution == "gaussian":
            self._residual_std = float(np.std(residuals))
            from scipy.optimize import curve_fit
            pred_clipped = np.clip(pred, 0.5, None)
            bins = np.percentile(pred_clipped, np.arange(0, 101, 10))
            bins = np.unique(bins)
            bin_idx = np.digitize(pred_clipped, bins)
            bin_stds = []
            bin_means = []
            for b in range(1, len(bins)):
                mask = bin_idx == b
                if mask.sum() > 30:
                    bin_stds.append(np.std(residuals[mask]))
                    bin_means.append(np.mean(pred_clipped[mask]))
            if len(bin_stds) >= 3:
                bin_means = np.array(bin_means)
                bin_stds = np.array(bin_stds)
                try:
                    def std_model(x, a, b):
                        return a + b * np.sqrt(x)
                    popt, _ = curve_fit(std_model, bin_means, bin_stds, p0=[1.0, 0.3])
                    self._std_params = (float(popt[0]), float(popt[1]))
                except Exception:
                    self._std_params = None
            else:
                self._std_params = None

        elif self.distribution == "negbin":
            pred_clipped = np.clip(pred, 0.5, None)
            overall_var = np.var(y_actual)
            overall_mean = np.mean(y_actual)
            if overall_var > overall_mean:
                self._negbin_r = float(overall_mean ** 2 / (overall_var - overall_mean))
                print(f"  NegBin r = {self._negbin_r:.2f} (var={overall_var:.2f}, mean={overall_mean:.2f})")
            else:
                self._negbin_r = 100.0
                print(f"  NegBin: var ({overall_var:.2f}) <= mean ({overall_mean:.2f}), using r=100 (≈Poisson)")

    def _get_std_for_mu(self, mu):
        """Get estimated std for a given predicted mean (Gaussian distribution)."""
        if self._std_params is not None:
            a, b = self._std_params
            return max(a + b * np.sqrt(max(mu, 0.1)), 0.5)
        if self._residual_std is not None:
            return max(float(self._residual_std), 0.5)
        # Fallback when the model is untrained or metadata is missing.
        return max(0.5, 0.6 * np.sqrt(max(mu, 0.1)) + 1.5)

    def _threshold_prob(self, mu, threshold):
        """Compute P(X >= threshold) using the configured distribution."""
        mu = max(mu, 0.01)
        if self.distribution == "poisson":
            return 1 - poisson.cdf(threshold - 1, mu)
        elif self.distribution == "gaussian":
            from scipy.stats import norm
            sigma = self._get_std_for_mu(mu)
            x = threshold - 0.5  # continuity correction
            return float(np.clip(1.0 - norm.cdf(x, loc=mu, scale=sigma), 0.0, 1.0))
        elif self.distribution == "negbin":
            from scipy.stats import nbinom
            r = self._negbin_r if self._negbin_r is not None else 100.0
            p_success = r / (r + mu)
            return 1 - nbinom.cdf(threshold - 1, r, p_success)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _confidence_interval(self, mu, lo_q=0.10, hi_q=0.90):
        """Compute confidence interval quantiles using configured distribution."""
        mu = max(mu, 0.01)
        if self.distribution == "poisson":
            return int(poisson.ppf(lo_q, mu)), int(poisson.ppf(hi_q, mu))
        elif self.distribution == "gaussian":
            from scipy.stats import norm
            sigma = self._get_std_for_mu(mu)
            return max(0, int(norm.ppf(lo_q, mu, sigma))), int(norm.ppf(hi_q, mu, sigma))
        elif self.distribution == "negbin":
            from scipy.stats import nbinom
            r = self._negbin_r if self._negbin_r is not None else 100.0
            p_success = r / (r + mu)
            return int(nbinom.ppf(lo_q, r, p_success)), int(nbinom.ppf(hi_q, r, p_success))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _predict_raw(self, X_raw, X_scaled, df=None):
        """Raw ensemble prediction for marks with optional mark-taker soft blend."""
        w_poi = self.ensemble_weights["poisson"]
        w_gbt = self.ensemble_weights["gbt"]

        pred_poi = self.marks_poisson.predict(X_scaled)
        X_clean = X_raw.fillna(0) if hasattr(X_raw, "fillna") else X_raw
        pred_gbt = _predict_with_compatible_input(self.marks_gbt, X_raw, X_clean)
        base_pred = np.clip(w_poi * pred_poi + w_gbt * pred_gbt, 0, None)

        # Soft two-stage: adjust predictions based on mark-taker probability
        if self.mark_taker_clf is not None and self._mark_taker_blend > 0:
            mt_prob = _predict_proba_with_compatible_input(self.mark_taker_clf, X_raw, X_clean)
            adjustment = 1.0 + self._mark_taker_blend * (mt_prob - 0.5)
            base_pred = base_pred * np.clip(adjustment, 0.5, 2.0)

        return base_pred

    def predict(self, df, feature_cols=None, store=None):
        """Generate marks predictions with probability thresholds."""
        feature_cols = feature_cols or self.feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(df, feature_cols, scaler=self.scaler)

        pred_marks = self._predict_raw(X_raw, X_scaled, df=df)

        round_col = "round_number" if "round_number" in df.columns else "round"
        result = df[["player", "team", "opponent", "venue", round_col]].copy()
        if round_col != "round":
            result = result.rename(columns={round_col: "round"})

        result["predicted_marks"] = np.round(pred_marks, 2)

        thresholds = config.MODEL_TARGETS["marks"]["thresholds"]
        for t in thresholds:
            result[f"p_{t}plus_mk"] = np.round(
                [self._threshold_prob(mu, t) for mu in pred_marks], 4
            )

        if store is not None and getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            if calibrator is not None:
                for t in thresholds:
                    tgt = f"{t}plus_mk"
                    col = f"p_{t}plus_mk"
                    if calibrator.has_calibrator(tgt) and col in result.columns:
                        result[col] = np.round(calibrator.transform(tgt, result[col].values), 4)

        _enforce_non_increasing_probs(
            result,
            [f"p_{t}plus_mk" for t in thresholds],
            round_dp=4,
        )

        ci = [self._confidence_interval(mu) for mu in pred_marks]
        result["conf_lower_mk"] = [c[0] for c in ci]
        result["conf_upper_mk"] = [c[1] for c in ci]

        if "player_role" in df.columns:
            result["player_role"] = df["player_role"].values

        result = result.sort_values("predicted_marks", ascending=False).reset_index(drop=True)
        return result

    def predict_distributions(self, df, store=None, feature_cols=None):
        """Generate marks predictions with expanded thresholds and calibration.

        Returns per-player marks predictions with calibrated lambda,
        probability thresholds for 3+/5+/7+, and confidence intervals.
        """
        feature_cols = feature_cols or self.feature_cols
        X_raw, X_clean, X_scaled = _prepare_features(df, feature_cols, scaler=self.scaler)
        pred_marks = self._predict_raw(X_raw, X_scaled, df=df)

        result = pd.DataFrame({
            "player": df["player"].values,
            "team": df["team"].values,
            "match_id": df["match_id"].values,
        })

        # Add mark-taker probability column
        if self.mark_taker_clf is not None:
            X_clean = X_raw.fillna(0) if hasattr(X_raw, "fillna") else X_raw
            result["p_mark_taker"] = np.round(
                _predict_proba_with_compatible_input(self.mark_taker_clf, X_raw, X_clean), 4
            )

        lambda_marks = np.zeros(len(df))
        cal_method = getattr(config, "CALIBRATION_METHOD", "bucket")
        for i in range(len(df)):
            raw_lam = max(pred_marks[i], 0.001)
            # Bucket-based lambda calibration is a heuristic; disable when isotonic is used.
            if store is not None and cal_method != "isotonic":
                lambda_marks[i] = store.get_lambda_calibration("marks", raw_lam)
            else:
                lambda_marks[i] = raw_lam

        result["predicted_marks"] = np.round(pred_marks, 2)
        result["lambda_marks"] = np.round(lambda_marks, 4)

        thresholds = config.MARKS_THRESHOLDS
        for t in thresholds:
            raw_probs = np.array([self._threshold_prob(mu, t) for mu in lambda_marks])
            result[f"p_{t}plus_mk"] = np.round(raw_probs, 4)

        if store is not None and getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
            calibrator = store.load_isotonic_calibrator() if hasattr(store, "load_isotonic_calibrator") else None
            skip_targets = getattr(config, "ISOTONIC_SKIP_TARGETS", set())
            if calibrator is not None:
                for t in thresholds:
                    tgt = f"{t}plus_mk"
                    col = f"p_{t}plus_mk"
                    if tgt in skip_targets:
                        continue
                    if calibrator.has_calibrator(tgt):
                        result[col] = np.round(
                            calibrator.transform(tgt, result[col].values), 4
                        )

        _enforce_non_increasing_probs(
            result,
            [f"p_{t}plus_mk" for t in thresholds],
            round_dp=4,
        )

        ci = [self._confidence_interval(mu) for mu in lambda_marks]
        result["conf_lower_mk"] = [c[0] for c in ci]
        result["conf_upper_mk"] = [c[1] for c in ci]

        return result

    def save(self, models_dir=None):
        """Save trained marks models."""
        models_dir = Path(models_dir or config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        with open(models_dir / "marks_poisson.pkl", "wb") as f:
            pickle.dump(self.marks_poisson, f)
        with open(models_dir / "marks_gbt.pkl", "wb") as f:
            pickle.dump(self.marks_gbt, f)
        with open(models_dir / "marks_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        if self.mark_taker_clf is not None:
            with open(models_dir / "marks_taker_clf.pkl", "wb") as f:
                pickle.dump(self.mark_taker_clf, f)

        metadata = {
            "feature_cols": self.feature_cols,
            "eval_metrics": self.eval_metrics,
            "training_info": self.training_info,
            "distribution": self.distribution,
            "residual_std": self._residual_std,
            "std_params": self._std_params if hasattr(self, '_std_params') else None,
            "negbin_r": self._negbin_r,
            "mark_taker_threshold": self._mark_taker_threshold,
            "mark_taker_blend": self._mark_taker_blend,
        }
        with open(models_dir / "marks_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Marks models saved to {models_dir}")

    def load(self, models_dir=None):
        """Load previously trained marks models."""
        models_dir = Path(models_dir or config.MODELS_DIR)

        with open(models_dir / "marks_poisson.pkl", "rb") as f:
            self.marks_poisson = pickle.load(f)
        with open(models_dir / "marks_gbt.pkl", "rb") as f:
            self.marks_gbt = pickle.load(f)
        with open(models_dir / "marks_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        mt_path = models_dir / "marks_taker_clf.pkl"
        if mt_path.exists():
            with open(mt_path, "rb") as f:
                self.mark_taker_clf = pickle.load(f)

        with open(models_dir / "marks_model_metadata.json") as f:
            metadata = json.load(f)
        self.feature_cols = metadata["feature_cols"]
        self.eval_metrics = metadata.get("eval_metrics", {})
        self.training_info = metadata.get("training_info", {})
        self.distribution = metadata.get("distribution", "poisson")
        self._residual_std = metadata.get("residual_std")
        self._std_params = metadata.get("std_params")
        self._negbin_r = metadata.get("negbin_r")
        self._mark_taker_threshold = metadata.get("mark_taker_threshold", 5)
        self._mark_taker_blend = metadata.get("mark_taker_blend", 0.3)

        print(f"Marks models loaded from {models_dir}")


# ---------------------------------------------------------------------------
# Monte Carlo Simulation Engine
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """Game-correlated Monte Carlo simulation layer.

    Runs N simulations per game, drawing player goals/disposals from
    the fitted scoring and disposal models, then applies a game-level
    correlation factor so that players on dominant teams get scaled up.

    Usage:
        sim = MonteCarloSimulator(scoring_model, disposal_model, winner_model)
        sim.estimate_correlation_factors(historical_df)
        mc_df = sim.simulate_round(predictions_df, game_preds_df, n_sims=10000)
    """

    # Default margin buckets and scaling factors (overridden by estimate_correlation_factors)
    DEFAULT_MARGIN_BINS = np.array([0, 10, 20, 40, 200])
    DEFAULT_SCALE_FACTORS = np.array([1.00, 1.03, 1.07, 1.12])  # per bucket

    def __init__(self, scoring_model=None, disposal_model=None, winner_model=None):
        self.scoring_model = scoring_model
        self.disposal_model = disposal_model
        self.winner_model = winner_model

        # Correlation factors: mapping from margin bucket → (goals_scale, disp_scale)
        self.margin_bins = self.DEFAULT_MARGIN_BINS.copy()
        self.goals_scale = self.DEFAULT_SCALE_FACTORS.copy()
        self.disp_scale = self.DEFAULT_SCALE_FACTORS.copy()

        # Margin distribution std for game-level draws
        self.margin_std = 30.0  # typical AFL game margin std

    def estimate_correlation_factors(self, feature_df, team_match_df=None):
        """Estimate correlation factors from historical data.

        Groups player performances by game margin buckets and measures
        how individual stats deviate from predictions at each margin level.

        Args:
            feature_df: Feature matrix with actuals (GL, DI) and predictions
                        or at minimum GL, DI, match_id, team columns.
            team_match_df: team_matches.parquet with margin per (team, match).
        """
        if team_match_df is None:
            print("  [MC] No team_match data — using default correlation factors")
            return

        # Get team margin per match
        tm = team_match_df[["match_id", "team", "margin", "score"]].copy()

        # Merge player data with team margin
        df = feature_df[["match_id", "player", "team", "GL", "DI"]].copy()
        df = df.merge(tm[["match_id", "team", "margin"]], on=["match_id", "team"], how="inner")

        # Need per-player expected values. Use simple career averages as proxy.
        career = df.groupby("player", observed=True).agg(
            career_gl=("GL", "mean"),
            career_di=("DI", "mean"),
            n_games=("GL", "count"),
        ).reset_index()
        career = career[career["n_games"] >= 10]  # need enough history
        df = df.merge(career[["player", "career_gl", "career_di"]], on="player", how="inner")

        # Bucket by absolute margin (winning side only)
        df["is_winning_side"] = df["margin"] > 0
        df["abs_margin"] = df["margin"].abs()

        bins = [0, 10, 20, 40, 200]
        labels = ["0-10", "10-20", "20-40", "40+"]
        df["margin_bucket"] = pd.cut(df["abs_margin"], bins=bins, labels=labels, right=True)

        # Compute scaling per bucket for winning side
        # For goals (sparse): use mean(actual) / mean(expected) ratio
        # For disposals: use mean(actual) / mean(expected) ratio
        goals_scales = []
        disp_scales = []
        for bucket in labels:
            bucket_df = df[(df["margin_bucket"] == bucket) & df["is_winning_side"]]
            if len(bucket_df) < 50:
                goals_scales.append(1.0)
                disp_scales.append(1.0)
                continue
            # Mean-of-means ratio: how much do stats deviate from career avg
            gl_ratio = float(bucket_df["GL"].mean() / bucket_df["career_gl"].mean())
            di_ratio = float(bucket_df["DI"].mean() / bucket_df["career_di"].mean())
            goals_scales.append(gl_ratio)
            disp_scales.append(di_ratio)

        self.margin_bins = np.array(bins)
        self.goals_scale = np.array(goals_scales, dtype=np.float64)
        self.disp_scale = np.array(disp_scales, dtype=np.float64)

        # Estimate margin std from historical data
        if team_match_df is not None and "margin" in team_match_df.columns:
            home_margins = team_match_df[team_match_df["is_home"]]["margin"]
            if len(home_margins) > 50:
                self.margin_std = float(home_margins.std())

        print(f"  [MC] Correlation factors estimated from {len(df)} player-games")
        print(f"       Goals scaling by margin bucket: {dict(zip(labels, np.round(self.goals_scale, 3)))}")
        print(f"       Disp scaling by margin bucket:  {dict(zip(labels, np.round(self.disp_scale, 3)))}")
        print(f"       Margin std: {self.margin_std:.1f}")

    def _get_correlation_scale(self, margin_draws, is_home_team, stat_type="goals"):
        """Vectorized: return per-simulation scaling factor based on simulated margin.

        Args:
            margin_draws: (n_sims,) array of simulated home margins.
            is_home_team: bool — True if this player is on the home team.
            stat_type: 'goals' or 'disp'

        Returns:
            (n_sims,) array of scaling factors.
        """
        scales = self.goals_scale if stat_type == "goals" else self.disp_scale

        # Team margin from player's perspective
        team_margin = margin_draws if is_home_team else -margin_draws

        # For winning side: scale up. For losing side: scale down (inverse).
        abs_margin = np.abs(team_margin)
        # Digitize into buckets: bins = [0, 10, 20, 40, 200]
        bucket_idx = np.digitize(abs_margin, self.margin_bins[1:-1])  # 0, 1, 2, 3
        bucket_idx = np.clip(bucket_idx, 0, len(scales) - 1)

        raw_scale = scales[bucket_idx]

        # Winning side gets scale factor; losing side gets inverse
        is_winning = team_margin > 0
        result = np.where(is_winning, raw_scale, 1.0 / np.maximum(raw_scale, 0.5))

        # Close games (margin 0-5): no scaling
        close_game = abs_margin <= 5
        result[close_game] = 1.0

        return result

    def simulate_round(self, predictions_df, game_preds_df=None, n_sims=10000):
        """Run Monte Carlo simulations for a round of games.

        Args:
            predictions_df: Merged predictions DataFrame with columns:
                player, team, opponent, venue, round, match_id,
                p_scorer (or p_scorer_raw), lambda_goals, lambda_behinds,
                predicted_disposals, plus disposal std params.
            game_preds_df: Game-level predictions with columns:
                match_id, home_team, home_win_prob, predicted_margin.
                If None, uses 50/50 with margin=0.
            n_sims: Number of simulations per game.

        Returns:
            DataFrame with per-player MC probabilities alongside direct model probs.
        """
        df = predictions_df.copy()
        n_players = len(df)

        if n_players == 0:
            return pd.DataFrame()

        # Extract match-level info
        matches = df[["match_id", "team"]].drop_duplicates()
        match_ids = df["match_id"].unique()

        # Build game-level params: P(home_win), predicted_margin, margin_std
        game_params = {}
        for mid in match_ids:
            if game_preds_df is not None and len(game_preds_df) > 0:
                gp = game_preds_df[game_preds_df["match_id"] == mid]
                if len(gp) > 0:
                    row = gp.iloc[0]
                    game_params[mid] = {
                        "home_win_prob": float(row.get("home_win_prob", 0.5)),
                        "predicted_margin": float(row.get("predicted_margin", 0.0)),
                        "home_team": row.get("home_team", ""),
                    }
                    continue
            # Fallback: use team home status from predictions
            game_params[mid] = {
                "home_win_prob": 0.5,
                "predicted_margin": 0.0,
                "home_team": "",
            }

        # ── Vectorized simulation ─────────────────────────────────────
        # Pre-extract player parameters
        p_scorer = df["p_scorer"].values if "p_scorer" in df.columns else df["p_scorer_raw"].values
        p_scorer = np.clip(p_scorer, 0.0, 1.0)

        # Lambda for shifted Poisson: goals | scorer ~ 1 + Poisson(mu)
        # lambda_goals = p_scorer * lambda_if_scorer, so lambda_if_scorer = lambda_goals / p_scorer
        lambda_goals = df["lambda_goals"].values if "lambda_goals" in df.columns else df["predicted_goals"].values
        lambda_if_scorer = np.where(
            p_scorer > 0.01,
            np.maximum(lambda_goals / p_scorer, 1.0),
            1.0,
        )

        # Disposal parameters
        pred_disp = df["predicted_disposals"].values if "predicted_disposals" in df.columns else np.full(n_players, 15.0)
        # Get disposal std: use lambda_disposals and _get_std_for_mu logic
        lambda_disp = df["lambda_disposals"].values if "lambda_disposals" in df.columns else pred_disp.copy()
        # Compute std per player: std = a + b * sqrt(mu)
        disp_std = np.zeros(n_players)
        if self.disposal_model is not None and hasattr(self.disposal_model, "_std_params") and self.disposal_model._std_params is not None:
            a, b = self.disposal_model._std_params
            disp_std = np.maximum(a + b * np.sqrt(np.maximum(lambda_disp, 0.1)), 1.0)
        elif self.disposal_model is not None and hasattr(self.disposal_model, "_residual_std") and self.disposal_model._residual_std is not None:
            disp_std = np.full(n_players, max(float(self.disposal_model._residual_std), 1.0))
        else:
            # Fallback: std ≈ 0.5*sqrt(mu) + 3
            disp_std = np.maximum(0.5 * np.sqrt(np.maximum(lambda_disp, 0.1)) + 3.0, 1.0)

        # Player-to-match mapping
        player_match_ids = df["match_id"].values
        player_is_home = np.zeros(n_players, dtype=bool)
        if "is_home" in df.columns:
            player_is_home = df["is_home"].values.astype(bool)
        else:
            # Infer from game_params
            for i in range(n_players):
                mid = player_match_ids[i]
                gp = game_params.get(mid, {})
                player_is_home[i] = (df.iloc[i]["team"] == gp.get("home_team", ""))

        # ── Run simulations per match (vectorized across players) ─────
        # Accumulators for threshold counts
        goals_counts = np.zeros((n_players, 4))  # >=1, >=2, >=3, >=4
        disp_counts = np.zeros((n_players, 5))   # >=10, >=15, >=20, >=25, >=30

        # Group players by match for correlated simulation
        match_player_indices = {}
        for i in range(n_players):
            mid = int(player_match_ids[i])
            if mid not in match_player_indices:
                match_player_indices[mid] = []
            match_player_indices[mid].append(i)

        rng = np.random.default_rng(42)

        for mid, player_idxs in match_player_indices.items():
            idxs = np.array(player_idxs)
            n_p = len(idxs)
            gp = game_params.get(mid, {"home_win_prob": 0.5, "predicted_margin": 0.0})

            # Draw game-level margins for all sims
            pred_margin = gp["predicted_margin"]
            margin_draws = rng.normal(pred_margin, self.margin_std, size=n_sims)

            # Get player params for this match
            p_sc = p_scorer[idxs]           # (n_p,)
            lam_if = lambda_if_scorer[idxs]  # (n_p,)
            mu_disp = lambda_disp[idxs]      # (n_p,)
            std_disp = disp_std[idxs]        # (n_p,)
            is_home = player_is_home[idxs]   # (n_p,)

            # Process in chunks to manage memory: (n_p, n_sims)
            chunk_size = min(n_sims, 2500)
            n_chunks = (n_sims + chunk_size - 1) // chunk_size

            for chunk_i in range(n_chunks):
                start = chunk_i * chunk_size
                end = min(start + chunk_size, n_sims)
                cs = end - start
                margin_chunk = margin_draws[start:end]  # (cs,)

                # --- Goals simulation ---
                # Stage 1: Bernoulli draw for scorer
                scorer_draws = rng.random((n_p, cs)) < p_sc[:, None]  # (n_p, cs)

                # Stage 2: Shifted Poisson for goals | scorer
                mu_shifted = np.maximum(lam_if[:, None] - 1.0, 0.001)  # (n_p, 1)
                pois_draws = rng.poisson(mu_shifted, size=(n_p, cs))    # (n_p, cs)
                sim_goals = np.where(scorer_draws, 1 + pois_draws, 0)   # (n_p, cs)

                # --- Disposal simulation ---
                sim_disp = rng.normal(
                    mu_disp[:, None],
                    std_disp[:, None],
                    size=(n_p, cs),
                )  # (n_p, cs)

                # --- Apply correlation scaling ---
                for j in range(n_p):
                    gl_scale = self._get_correlation_scale(margin_chunk, is_home[j], "goals")  # (cs,)
                    di_scale = self._get_correlation_scale(margin_chunk, is_home[j], "disp")   # (cs,)
                    sim_goals[j] = np.round(sim_goals[j].astype(np.float64) * gl_scale).astype(int)
                    sim_disp[j] = sim_disp[j] * di_scale

                # Clip disposals at 0
                sim_disp = np.maximum(sim_disp, 0)

                # --- Accumulate threshold counts ---
                for t_i, threshold in enumerate([1, 2, 3, 4]):
                    goals_counts[idxs, t_i] += (sim_goals >= threshold).sum(axis=1)
                for t_i, threshold in enumerate([10, 15, 20, 25, 30]):
                    disp_counts[idxs, t_i] += (sim_disp >= threshold).sum(axis=1)

        # ── Compute probabilities ─────────────────────────────────────
        mc_p_1plus = goals_counts[:, 0] / n_sims
        mc_p_2plus = goals_counts[:, 1] / n_sims
        mc_p_3plus = goals_counts[:, 2] / n_sims
        mc_p_4plus = goals_counts[:, 3] / n_sims
        mc_p_10plus_disp = disp_counts[:, 0] / n_sims
        mc_p_15plus_disp = disp_counts[:, 1] / n_sims
        mc_p_20plus_disp = disp_counts[:, 2] / n_sims
        mc_p_25plus_disp = disp_counts[:, 3] / n_sims
        mc_p_30plus_disp = disp_counts[:, 4] / n_sims

        # ── Build result DataFrame ────────────────────────────────────
        round_col = "round" if "round" in df.columns else "round_number"
        result = df[["player", "team", "opponent", "venue", round_col, "match_id"]].copy()
        if round_col != "round":
            result = result.rename(columns={round_col: "round"})

        # Direct model probabilities (for comparison)
        result["direct_p_1plus_goals"] = df.get("p_1plus_goals", df.get("p_scorer", np.nan)).values
        result["direct_p_2plus_goals"] = df.get("p_2plus_goals", np.full(n_players, np.nan)).values
        result["direct_p_3plus_goals"] = df.get("p_3plus_goals", np.full(n_players, np.nan)).values
        result["direct_p_15plus_disp"] = df.get("p_15plus_disp", np.full(n_players, np.nan)).values
        result["direct_p_20plus_disp"] = df.get("p_20plus_disp", np.full(n_players, np.nan)).values
        result["direct_p_25plus_disp"] = df.get("p_25plus_disp", np.full(n_players, np.nan)).values
        result["direct_p_30plus_disp"] = df.get("p_30plus_disp", np.full(n_players, np.nan)).values

        # Monte Carlo probabilities
        result["mc_p_1plus_goals"] = np.round(mc_p_1plus, 4)
        result["mc_p_2plus_goals"] = np.round(mc_p_2plus, 4)
        result["mc_p_3plus_goals"] = np.round(mc_p_3plus, 4)
        result["mc_p_4plus_goals"] = np.round(mc_p_4plus, 4)
        result["mc_p_10plus_disp"] = np.round(mc_p_10plus_disp, 4)
        result["mc_p_15plus_disp"] = np.round(mc_p_15plus_disp, 4)
        result["mc_p_20plus_disp"] = np.round(mc_p_20plus_disp, 4)
        result["mc_p_25plus_disp"] = np.round(mc_p_25plus_disp, 4)
        result["mc_p_30plus_disp"] = np.round(mc_p_30plus_disp, 4)

        # Predicted values for reference
        result["predicted_goals"] = df["predicted_goals"].values
        result["predicted_disposals"] = pred_disp

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    model = AFLScoringModel()

    if "--evaluate" in sys.argv:
        # Load model and evaluate on validation data
        model.load()
        feat_df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
        val_df = feat_df[feat_df["year"] == config.VALIDATION_YEAR]
        model.evaluate_detailed(val_df)
    else:
        # Train
        feat_df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
        model.train(feat_df)
        model.save()
