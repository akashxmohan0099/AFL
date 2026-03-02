"""
AFL Prediction Pipeline — Data Validation
==========================================
Lightweight checks at each pipeline stage to catch data issues early.
No external dependencies beyond pandas/numpy.
"""

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when data fails a critical validation check."""
    pass


def validate_cleaned(df):
    """Validate the cleaned player_matches DataFrame.

    Checks:
      - Required columns present
      - No duplicate (player, team, match_id) rows
      - Non-negative target columns (GL, BH)
      - Valid dates
      - Minimum match size (at least 1 player per match)

    Raises ValidationError on critical failures, prints warnings otherwise.
    """
    errors = []
    warnings = []

    # Required columns (accept both "date" and "date_iso" for backward compat)
    required = ["player", "team", "match_id", "year",
                 "round_number", "venue", "opponent", "GL", "BH"]
    missing = [c for c in required if c not in df.columns]
    if "date" not in df.columns and "date_iso" not in df.columns:
        missing.append("date/date_iso")
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Duplicate rows
    if not missing:
        dupes = df.duplicated(subset=["player", "team", "match_id"], keep=False)
        n_dupes = dupes.sum()
        if n_dupes > 0:
            errors.append(
                f"{n_dupes} duplicate (player, team, match_id) rows found"
            )

    # Non-negative targets
    if "GL" in df.columns:
        neg_gl = (df["GL"] < 0).sum()
        if neg_gl > 0:
            errors.append(f"{neg_gl} rows with negative GL values")

    if "BH" in df.columns:
        neg_bh = (df["BH"] < 0).sum()
        if neg_bh > 0:
            errors.append(f"{neg_bh} rows with negative BH values")

    # Valid dates
    date_col = "date" if "date" in df.columns else "date_iso"
    if date_col in df.columns:
        null_dates = df[date_col].isna().sum()
        if null_dates > 0:
            warnings.append(f"{null_dates} rows with null dates")

    # Minimum match size
    if "match_id" in df.columns and "player" in df.columns:
        match_sizes = df.groupby("match_id")["player"].count()
        tiny = (match_sizes < 2).sum()
        if tiny > 0:
            warnings.append(f"{tiny} matches with fewer than 2 players")

    # Report
    if warnings:
        for w in warnings:
            print(f"  VALIDATION WARNING: {w}")

    if errors:
        msg = "Cleaned data validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValidationError(msg)

    print("  Validation passed (cleaned data)")


def validate_features(df, feature_cols):
    """Validate the feature matrix before saving.

    Checks:
      - All feature columns exist in the DataFrame
      - No inf values in feature columns
      - No all-NaN feature columns
      - Target columns still valid

    Raises ValidationError on critical failures, prints warnings otherwise.
    """
    errors = []
    warnings = []

    # All feature cols exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        errors.append(f"{len(missing)} feature columns missing: {missing[:10]}")

    present_cols = [c for c in feature_cols if c in df.columns]

    # No inf values
    if present_cols:
        inf_counts = df[present_cols].apply(lambda s: np.isinf(s).sum() if s.dtype.kind == 'f' else 0)
        inf_cols = inf_counts[inf_counts > 0]
        if len(inf_cols) > 0:
            warnings.append(
                f"{len(inf_cols)} feature columns contain inf values: "
                f"{list(inf_cols.index[:5])}"
            )

    # All-NaN columns (warning — can happen with sparse data, model fills with 0)
    if present_cols:
        all_nan = [c for c in present_cols if df[c].isna().all()]
        if all_nan:
            warnings.append(f"{len(all_nan)} feature columns are all NaN: {all_nan[:10]}")

    # Targets still valid
    if "GL" in df.columns:
        neg = (df["GL"] < 0).sum()
        if neg > 0:
            errors.append(f"{neg} rows with negative GL after feature engineering")

    # Report
    if warnings:
        for w in warnings:
            print(f"  VALIDATION WARNING: {w}")

    if errors:
        msg = "Feature matrix validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValidationError(msg)

    print(f"  Validation passed (feature matrix: {len(present_cols)} features, {len(df)} rows)")


def validate_predictions(pred_df):
    """Validate prediction output before returning.

    Checks:
      - Required output columns exist
      - Non-negative predictions

    Raises ValidationError on critical failures.
    """
    errors = []

    required = ["player", "team", "predicted_goals", "predicted_behinds", "p_scorer"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        errors.append(f"Missing required prediction columns: {missing}")

    if "predicted_goals" in pred_df.columns:
        neg = (pred_df["predicted_goals"] < 0).sum()
        if neg > 0:
            errors.append(f"{neg} negative predicted_goals values")

    if "predicted_behinds" in pred_df.columns:
        neg = (pred_df["predicted_behinds"] < 0).sum()
        if neg > 0:
            errors.append(f"{neg} negative predicted_behinds values")

    if "p_scorer" in pred_df.columns:
        out_of_range = ((pred_df["p_scorer"] < 0) | (pred_df["p_scorer"] > 1)).sum()
        if out_of_range > 0:
            errors.append(f"{out_of_range} p_scorer values outside [0, 1]")

    if errors:
        msg = "Prediction validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValidationError(msg)

    print(f"  Validation passed (predictions: {len(pred_df)} rows)")
