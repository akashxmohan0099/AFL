"""Stacked meta-ensemble for AFL prediction.

Level 0: Diverse base models (XGBoost, LightGBM, HistGBT, Poisson)
Level 1: Ridge regression meta-learner on out-of-fold predictions

Walk-forward OOF strategy (no random splits — time series data):
  Fold 1: train 2015-2020, predict 2021
  Fold 2: train 2015-2021, predict 2022
  Fold 3: train 2015-2022, predict 2023
  Fold 4: train 2015-2023, predict 2024

Called from model.py's train_backtest() and _predict_raw() when
config.STACKING_ENABLED is True.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stacked Ensemble
# ---------------------------------------------------------------------------

class StackedEnsemble:
    """Two-level stacking for count prediction (goals, disposals, or marks).

    Level 0 base models:
      - XGBoost (tree-based, handles missing values natively)
      - LightGBM (leaf-wise growth, fast, complementary to XGB)
      - HistGradientBoosting (sklearn native, no extra deps)
      - PoissonRegressor (linear, captures count nature of targets)

    Level 1 meta-learner:
      - Ridge regression on out-of-fold base predictions
      - StandardScaler on meta features for stable regularization
    """

    def __init__(self, target="GL"):
        """Initialize stacked ensemble.

        Args:
            target: Column name of the prediction target.
                    One of "GL" (goals), "DI" (disposals), "MK" (marks).
        """
        self.target = target
        self.base_models = {}       # name -> trained model (final, on all data)
        self.meta_model = None      # Ridge meta-learner
        self.meta_scaler = None     # StandardScaler for meta features
        self.poisson_scaler = None  # StandardScaler for Poisson input
        self.base_names = []        # ordered list of base model names

    # ------------------------------------------------------------------
    # Full fit with walk-forward OOF
    # ------------------------------------------------------------------

    def fit(self, df, feature_cols, sample_weight=None):
        """Train all base models and meta-learner using walk-forward OOF.

        Walk-forward folds (NO random splits — time series data):
          Fold 1: train 2015-2020, predict 2021
          Fold 2: train 2015-2021, predict 2022
          Fold 3: train 2015-2022, predict 2023
          Fold 4: train 2015-2023, predict 2024

        The number of folds is controlled by config.STACKING_N_FOLDS.
        After OOF collection, a Ridge meta-learner is fitted on the
        stacked OOF predictions. Finally, base models are retrained on
        the full dataset for production use.

        Args:
            df: DataFrame with features, target column, and "year" column.
            feature_cols: List of feature column names.
            sample_weight: Optional array of sample weights (len == len(df)).
        """
        y = df[self.target].values
        years = sorted(df["year"].unique())
        n_folds = getattr(config, "STACKING_N_FOLDS", 4)

        # Use last n_folds years as validation folds
        val_years = years[-n_folds:]

        # Prepare raw feature matrix once (avoid repeated .replace calls)
        X_raw_all = df[feature_cols].copy()
        X_raw_all = X_raw_all.replace([np.inf, -np.inf], np.nan)
        X_clean_all = X_raw_all.fillna(0)

        # Fit Poisson scaler on full data (for final model training later)
        self.poisson_scaler = StandardScaler()
        self.poisson_scaler.fit(X_clean_all)

        # Containers for out-of-fold predictions
        model_names = ["xgb", "lgbm", "histgbt", "poisson"]
        oof_preds = {name: np.full(len(df), np.nan) for name in model_names}

        for val_year in val_years:
            train_mask = df["year"] < val_year
            val_mask = df["year"] == val_year

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue

            X_train_raw = X_raw_all[train_mask].values
            X_val_raw = X_raw_all[val_mask].values
            X_train_clean = X_clean_all[train_mask].values
            X_val_clean = X_clean_all[val_mask].values

            # Per-fold scaler for Poisson (fit only on fold's training data)
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train_clean)
            X_val_scaled = fold_scaler.transform(X_val_clean)

            y_train = y[train_mask]
            w_train = sample_weight[train_mask] if sample_weight is not None else None

            val_idx = np.where(val_mask)[0]

            # --- XGBoost ---
            try:
                from xgboost import XGBRegressor
                xgb_params = getattr(config, "XGB_PARAMS_BACKTEST", {})
                xgb = XGBRegressor(**xgb_params)
                xgb.fit(X_train_raw, y_train, sample_weight=w_train)
                oof_preds["xgb"][val_idx] = np.clip(xgb.predict(X_val_raw), 0, None)
            except (ImportError, Exception) as e:
                logger.debug(f"XGBoost not available ({e}), skipping in stacking fold")

            # --- LightGBM ---
            try:
                from lightgbm import LGBMRegressor
                lgbm_params = getattr(config, "LGBM_PARAMS_BACKTEST", {})
                lgbm = LGBMRegressor(**lgbm_params)
                lgbm.fit(X_train_raw, y_train, sample_weight=w_train)
                oof_preds["lgbm"][val_idx] = np.clip(lgbm.predict(X_val_raw), 0, None)
            except ImportError:
                logger.debug("LightGBM not available, skipping in stacking fold")

            # --- HistGradientBoosting ---
            histgbt_params = getattr(config, "HIST_GBT_PARAMS_BACKTEST", {})
            hgbt = HistGradientBoostingRegressor(**histgbt_params)
            hgbt.fit(X_train_raw, y_train, sample_weight=w_train)
            oof_preds["histgbt"][val_idx] = np.clip(hgbt.predict(X_val_raw), 0, None)

            # --- Poisson ---
            poi_params = getattr(config, "POISSON_PARAMS", {"alpha": 0.014, "max_iter": 1000})
            poi = PoissonRegressor(**poi_params)
            poi.fit(X_train_scaled, y_train, sample_weight=w_train)
            oof_preds["poisson"][val_idx] = np.clip(poi.predict(X_val_scaled), 0, None)

        # Build meta features from OOF predictions
        valid_names = [name for name in model_names
                       if not np.all(np.isnan(oof_preds[name]))]
        self.base_names = valid_names

        # Only use rows with complete OOF predictions (validation years)
        oof_matrix = np.column_stack([oof_preds[name] for name in valid_names])
        valid_rows = ~np.any(np.isnan(oof_matrix), axis=1)

        if valid_rows.sum() < 100:
            logger.warning(
                f"Stacking [{self.target}]: only {valid_rows.sum()} valid OOF rows, "
                "falling back to simple ensemble (no meta-learner)"
            )
            self.meta_model = None
            self._fit_base_models_final(df, feature_cols, y, sample_weight)
            return

        Z_oof = oof_matrix[valid_rows]
        y_oof = y[valid_rows]

        # Log OOF MAE per base model for diagnostics
        for i, name in enumerate(valid_names):
            mae = mean_absolute_error(y_oof, Z_oof[:, i])
            logger.info(f"Stacking [{self.target}] OOF MAE for {name}: {mae:.4f}")

        # Fit meta-learner (Ridge on scaled meta features)
        alpha = getattr(config, "STACKING_META_ALPHA", 1.0)
        self.meta_scaler = StandardScaler()
        Z_scaled = self.meta_scaler.fit_transform(Z_oof)
        self.meta_model = Ridge(alpha=alpha)
        self.meta_model.fit(Z_scaled, y_oof)

        logger.info(
            f"Stacking [{self.target}] meta weights: "
            + ", ".join(f"{name}={coef:.3f}"
                        for name, coef in zip(valid_names, self.meta_model.coef_))
            + f" | intercept={self.meta_model.intercept_:.3f}"
        )

        # Retrain base models on ALL data for production predictions
        self._fit_base_models_final(df, feature_cols, y, sample_weight)

    # ------------------------------------------------------------------
    # Backtest / sequential fit (simplified, heavy regularization)
    # ------------------------------------------------------------------

    def fit_backtest(self, train_df, feature_cols, sample_weight=None):
        """Simplified fit for backtest/sequential — no OOF, heavy regularization.

        In sequential mode the training set grows each round, so walk-forward
        OOF is impractical. Instead, fit all base models on the training data
        and train the Ridge meta-learner on in-sample base predictions with
        heavy regularization (config.STACKING_META_ALPHA_BACKTEST, default 10.0)
        to avoid overfitting.

        Args:
            train_df: Training DataFrame with features, target, and "year".
            feature_cols: List of feature column names.
            sample_weight: Optional array of sample weights.
        """
        y = train_df[self.target].values
        X_raw_df = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_clean = X_raw_df.fillna(0).values
        X_raw = X_raw_df.values

        self.poisson_scaler = StandardScaler()
        X_scaled = self.poisson_scaler.fit_transform(X_clean)

        w = sample_weight

        # Train all base models and collect in-sample predictions
        self.base_models = {}
        base_preds = {}

        # --- XGBoost ---
        try:
            from xgboost import XGBRegressor
            xgb_params = getattr(config, "XGB_PARAMS_BACKTEST", {})
            self.base_models["xgb"] = XGBRegressor(**xgb_params)
            self.base_models["xgb"].fit(X_raw, y, sample_weight=w)
            base_preds["xgb"] = np.clip(self.base_models["xgb"].predict(X_raw), 0, None)
        except ImportError:
            logger.debug("XGBoost not available, skipping in stacking backtest")

        # --- LightGBM ---
        try:
            from lightgbm import LGBMRegressor
            lgbm_params = getattr(config, "LGBM_PARAMS_BACKTEST", {})
            self.base_models["lgbm"] = LGBMRegressor(**lgbm_params)
            self.base_models["lgbm"].fit(X_raw, y, sample_weight=w)
            base_preds["lgbm"] = np.clip(self.base_models["lgbm"].predict(X_raw), 0, None)
        except ImportError:
            logger.debug("LightGBM not available, skipping in stacking backtest")

        # --- HistGradientBoosting ---
        histgbt_params = getattr(config, "HIST_GBT_PARAMS_BACKTEST", {})
        self.base_models["histgbt"] = HistGradientBoostingRegressor(**histgbt_params)
        self.base_models["histgbt"].fit(X_raw, y, sample_weight=w)
        base_preds["histgbt"] = np.clip(self.base_models["histgbt"].predict(X_raw), 0, None)

        # --- Poisson ---
        poi_params = getattr(config, "POISSON_PARAMS", {"alpha": 0.014, "max_iter": 1000})
        self.base_models["poisson"] = PoissonRegressor(**poi_params)
        self.base_models["poisson"].fit(X_scaled, y, sample_weight=w)
        base_preds["poisson"] = np.clip(self.base_models["poisson"].predict(X_scaled), 0, None)

        self.base_names = sorted(base_preds.keys())

        if len(self.base_names) == 0:
            logger.error(f"Stacking [{self.target}]: no base models trained, cannot fit meta-learner")
            self.meta_model = None
            return

        # In-sample meta training with heavy regularization
        Z = np.column_stack([base_preds[name] for name in self.base_names])
        alpha = getattr(config, "STACKING_META_ALPHA_BACKTEST", 10.0)
        self.meta_scaler = StandardScaler()
        Z_scaled = self.meta_scaler.fit_transform(Z)
        self.meta_model = Ridge(alpha=alpha)
        self.meta_model.fit(Z_scaled, y, sample_weight=w)

        logger.info(
            f"Stacking [{self.target}] backtest meta weights: "
            + ", ".join(f"{name}={coef:.3f}"
                        for name, coef in zip(self.base_names, self.meta_model.coef_))
            + f" | intercept={self.meta_model.intercept_:.3f}"
            + f" | alpha={alpha}"
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df, feature_cols):
        """Generate stacked ensemble prediction.

        Runs each base model to produce level-0 predictions, then combines
        them through the Ridge meta-learner. Falls back to simple average
        if the meta-learner was not fitted (e.g., too few OOF rows).

        Args:
            df: DataFrame with feature columns.
            feature_cols: List of feature column names (must match training).

        Returns:
            np.ndarray of predicted values, clipped to >= 0.
        """
        if len(self.base_names) == 0:
            logger.warning(f"Stacking [{self.target}]: no base models, returning zeros")
            return np.zeros(len(df))

        X_raw_df = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_raw = X_raw_df.values
        X_clean = X_raw_df.fillna(0).values
        X_scaled = self.poisson_scaler.transform(X_clean)

        base_preds = {}
        for name in self.base_names:
            model = self.base_models.get(name)
            if model is None:
                continue
            if name == "poisson":
                base_preds[name] = np.clip(model.predict(X_scaled), 0, None)
            else:
                base_preds[name] = np.clip(model.predict(X_raw), 0, None)

        if len(base_preds) == 0:
            logger.warning(f"Stacking [{self.target}]: no base predictions, returning zeros")
            return np.zeros(len(df))

        # Use meta-learner if available and all base models produced predictions
        if self.meta_model is not None and len(base_preds) == len(self.base_names):
            Z = np.column_stack([base_preds[name] for name in self.base_names])
            Z_scaled = self.meta_scaler.transform(Z)
            return np.clip(self.meta_model.predict(Z_scaled), 0, None)
        else:
            # Fallback: simple average of available base predictions
            all_preds = list(base_preds.values())
            logger.info(
                f"Stacking [{self.target}]: meta-learner unavailable, "
                f"using simple average of {len(all_preds)} base models"
            )
            return np.clip(np.mean(all_preds, axis=0), 0, None)

    # ------------------------------------------------------------------
    # Internal: final base model training on full data
    # ------------------------------------------------------------------

    def _fit_base_models_final(self, df, feature_cols, y, sample_weight=None):
        """Train base models on full dataset for production predictions.

        Called after OOF meta-learner fitting. These final models are used
        at predict() time to generate level-0 inputs.

        Args:
            df: Full training DataFrame.
            feature_cols: List of feature column names.
            y: Target values array.
            sample_weight: Optional sample weights.
        """
        X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).values
        X_clean = np.nan_to_num(X_raw, nan=0.0)
        X_scaled = self.poisson_scaler.transform(X_clean)

        # --- XGBoost ---
        try:
            from xgboost import XGBRegressor
            xgb_params = getattr(config, "XGB_PARAMS_BACKTEST", {})
            self.base_models["xgb"] = XGBRegressor(**xgb_params)
            self.base_models["xgb"].fit(X_raw, y, sample_weight=sample_weight)
        except ImportError:
            logger.debug("XGBoost not available for final base model training")

        # --- LightGBM ---
        try:
            from lightgbm import LGBMRegressor
            lgbm_params = getattr(config, "LGBM_PARAMS_BACKTEST", {})
            self.base_models["lgbm"] = LGBMRegressor(**lgbm_params)
            self.base_models["lgbm"].fit(X_raw, y, sample_weight=sample_weight)
        except ImportError:
            logger.debug("LightGBM not available for final base model training")

        # --- HistGradientBoosting ---
        histgbt_params = getattr(config, "HIST_GBT_PARAMS_BACKTEST", {})
        self.base_models["histgbt"] = HistGradientBoostingRegressor(**histgbt_params)
        self.base_models["histgbt"].fit(X_raw, y, sample_weight=sample_weight)

        # --- Poisson ---
        poi_params = getattr(config, "POISSON_PARAMS", {"alpha": 0.014, "max_iter": 1000})
        self.base_models["poisson"] = PoissonRegressor(**poi_params)
        self.base_models["poisson"].fit(X_scaled, y, sample_weight=sample_weight)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save stacking ensemble to disk.

        Creates a pickle file at ``<path>/stacking_<target>.pkl``
        containing all base models, the meta-learner, and scalers.

        Args:
            path: Directory path (will be created if it doesn't exist).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_path = path / f"stacking_{self.target}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({
                "target": self.target,
                "base_names": self.base_names,
                "base_models": self.base_models,
                "meta_model": self.meta_model,
                "meta_scaler": self.meta_scaler,
                "poisson_scaler": self.poisson_scaler,
            }, f)
        logger.info(f"Stacking [{self.target}] saved to {save_path}")

    @classmethod
    def load(cls, path, target="GL"):
        """Load stacking ensemble from disk.

        Args:
            path: Directory containing ``stacking_<target>.pkl``.
            target: Target name to load (default "GL").

        Returns:
            StackedEnsemble instance with all models restored.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        path = Path(path)
        load_path = path / f"stacking_{target}.pkl"
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        obj = cls(target=data["target"])
        obj.base_names = data["base_names"]
        obj.base_models = data["base_models"]
        obj.meta_model = data["meta_model"]
        obj.meta_scaler = data["meta_scaler"]
        obj.poisson_scaler = data["poisson_scaler"]
        logger.info(f"Stacking [{target}] loaded from {load_path}")
        return obj
