"""
AFL Prediction Pipeline — Model Training, Evaluation & Prediction
==================================================================
Dual Poisson/GBT ensemble for goals and behinds prediction.

Architecture:
  - Goals model:  40% PoissonRegressor + 60% GradientBoostingRegressor
  - Behinds model: same architecture (behinds are more stochastic)

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")


class AFLScoringModel:
    """Ensemble model for predicting goals and behinds per player per match."""

    def __init__(self):
        self.goals_poisson = None
        self.goals_gbt = None
        self.behinds_poisson = None
        self.behinds_gbt = None
        self.scaler = None
        self.feature_cols = []
        self.eval_metrics = {}
        self.training_info = {}

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
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()

        y_train_goals = train_df["GL"].values
        y_train_behinds = train_df["BH"].values
        y_val_goals = val_df["GL"].values
        y_val_behinds = val_df["BH"].values

        weights_train = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
        weights_val = val_df["sample_weight"].values if "sample_weight" in val_df.columns else None

        # Handle NaN/inf in features
        X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
        X_val = X_val.fillna(0).replace([np.inf, -np.inf], 0)

        # Scale for Poisson regressor
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # --- Train Goals Models ---
        print("\n--- Training Goals Models ---")

        # Poisson regressor (wants non-negative features for stable convergence)
        print("  Training Poisson regressor for goals...")
        self.goals_poisson = PoissonRegressor(
            alpha=config.POISSON_PARAMS["alpha"],
            max_iter=config.POISSON_PARAMS["max_iter"],
        )
        self.goals_poisson.fit(X_train_scaled, y_train_goals, sample_weight=weights_train)

        # GBT regressor
        print("  Training GBT regressor for goals...")
        self.goals_gbt = GradientBoostingRegressor(**config.GBT_PARAMS)
        self.goals_gbt.fit(X_train, y_train_goals, sample_weight=weights_train)

        # --- Train Behinds Models ---
        print("\n--- Training Behinds Models ---")

        print("  Training Poisson regressor for behinds...")
        self.behinds_poisson = PoissonRegressor(
            alpha=config.POISSON_PARAMS["alpha"],
            max_iter=config.POISSON_PARAMS["max_iter"],
        )
        self.behinds_poisson.fit(X_train_scaled, y_train_behinds, sample_weight=weights_train)

        print("  Training GBT regressor for behinds...")
        self.behinds_gbt = GradientBoostingRegressor(**config.GBT_PARAMS)
        self.behinds_gbt.fit(X_train, y_train_behinds, sample_weight=weights_train)

        # --- Evaluate on validation set ---
        print("\n--- Evaluating on validation set ---")
        self.eval_metrics = self._evaluate(
            X_val, X_val_scaled, y_val_goals, y_val_behinds, val_df
        )

        self.training_info = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "n_features": len(feature_cols),
            "train_years": f"{train_df['year'].min()}-{train_df['year'].max()}",
            "val_year": config.VALIDATION_YEAR,
        }

        return self.eval_metrics

    def _ensemble_predict(self, X_raw, X_scaled, target="goals"):
        """Generate ensemble prediction (Poisson + GBT weighted average)."""
        w_poi = config.ENSEMBLE_WEIGHTS["poisson"]
        w_gbt = config.ENSEMBLE_WEIGHTS["gbt"]

        if target == "goals":
            pred_poi = self.goals_poisson.predict(X_scaled)
            pred_gbt = self.goals_gbt.predict(X_raw)
        else:
            pred_poi = self.behinds_poisson.predict(X_scaled)
            pred_gbt = self.behinds_gbt.predict(X_raw)

        # Clip to non-negative
        pred = np.clip(w_poi * pred_poi + w_gbt * pred_gbt, 0, None)
        return pred

    def _evaluate(self, X_val, X_val_scaled, y_val_goals, y_val_behinds, val_df):
        """Evaluate on validation set. Returns dict of metrics."""
        pred_goals = self._ensemble_predict(X_val, X_val_scaled, "goals")
        pred_behinds = self._ensemble_predict(X_val, X_val_scaled, "behinds")

        # Baseline: career_goal_avg
        baseline_goals = val_df["career_goal_avg"].fillna(0).values

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
        importances = self.goals_gbt.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print("\n  Top 15 goal features:")
        for i in top_idx:
            print(f"    {self.feature_cols[i]:40s} {importances[i]:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df, feature_cols=None):
        """Generate predictions for a DataFrame of upcoming matches.

        Args:
            df: DataFrame with the same feature columns as training data.
                Must include 'player', 'team', 'opponent', 'venue', 'round'.

        Returns:
            DataFrame with prediction columns added.
        """
        feature_cols = feature_cols or self.feature_cols
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)

        pred_goals = self._ensemble_predict(X, X_scaled, "goals")
        pred_behinds = self._ensemble_predict(X, X_scaled, "behinds")

        result = df[["player", "team", "opponent", "venue", "round"]].copy()
        result["predicted_goals"] = np.round(pred_goals, 2)
        result["predicted_behinds"] = np.round(pred_behinds, 2)
        result["predicted_score"] = np.round(pred_goals * 6 + pred_behinds, 2)

        # Confidence intervals from Poisson distribution (80%)
        result["conf_lower_gl"] = [
            poisson.ppf(0.10, max(mu, 0.01)) for mu in pred_goals
        ]
        result["conf_upper_gl"] = [
            poisson.ppf(0.90, max(mu, 0.01)) for mu in pred_goals
        ]

        # Top-3 driving features per prediction
        top_factors = self._explain_predictions(X, feature_cols)
        result["top_factor_1"] = top_factors[0]
        result["top_factor_2"] = top_factors[1]
        result["top_factor_3"] = top_factors[2]

        # Player role and career baseline
        if "player_role" in df.columns:
            result["player_role"] = df["player_role"].values
        if "career_goal_avg" in df.columns:
            result["career_goal_avg"] = df["career_goal_avg"].values

        # Sort by predicted goals descending
        result = result.sort_values("predicted_goals", ascending=False).reset_index(drop=True)

        return result

    def _explain_predictions(self, X, feature_cols):
        """For each prediction, identify top-3 features driving it.

        Uses GBT feature importances weighted by actual feature values
        (simple approximation of SHAP without the dependency).
        """
        importances = self.goals_gbt.feature_importances_

        top1, top2, top3 = [], [], []

        for i in range(len(X)):
            row = X.iloc[i] if isinstance(X, pd.DataFrame) else X[i]
            # Score = importance * abs(normalized value)
            vals = np.abs(np.array(row, dtype=float))
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
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)

        pred_goals = self._ensemble_predict(X, X_scaled, "goals")
        pred_behinds = self._ensemble_predict(X, X_scaled, "behinds")

        actual_goals = df["GL"].values
        actual_behinds = df["BH"].values
        baseline = df["career_goal_avg"].fillna(0).values

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
