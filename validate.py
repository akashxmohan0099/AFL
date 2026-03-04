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


def validate_temporal_integrity(df, feature_cols):
    """Validate that feature columns are pre-match safe.

    Checks:
      - No raw current-match stats slipped into model feature columns
      - career_disp_avg_pre does not mirror same-row DI for first player-team match
      - Obvious post-game marker columns are not in feature set

    Raises ValidationError on critical failures, prints warnings otherwise.
    """
    errors = []
    warnings = []

    stat_cols = {
        "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF", "CL",
        "CG", "FF", "FA", "BR", "CP", "UP", "CM", "MI", "one_pct", "BO", "GA",
    }
    forbidden_exact = (
        stat_cols
        | {f"{c}_rate" for c in stat_cols}
        | {
            "pct_played", "q1_goals", "q1_behinds", "q2_goals", "q2_behinds",
            "q3_goals", "q3_behinds", "q4_goals", "q4_behinds",
        }
    )

    leaked = [c for c in feature_cols if c in forbidden_exact]
    if leaked:
        errors.append(
            f"{len(leaked)} forbidden current-match columns present in features: {leaked[:10]}"
        )

    post_cols = [
        c for c in feature_cols
        if ("_post" in c.lower()) or c.lower().endswith("_result")
    ]
    if post_cols:
        errors.append(
            f"{len(post_cols)} post-match style columns present in features: {post_cols[:10]}"
        )

    # First-row leakage check for career_disp_avg_pre.
    # If this equals same-row DI for first appearances, it indicates lookahead fallback.
    needed = {"player", "team", "date", "DI", "career_disp_avg_pre"}
    if needed.issubset(df.columns):
        order_cols = ["player", "team", "date"] + (["match_id"] if "match_id" in df.columns else [])
        ordered = df.sort_values(order_cols)
        first_rows = ordered.groupby(["player", "team"], observed=True).head(1)
        first_rows = first_rows[["career_disp_avg_pre", "DI"]].dropna()
        if not first_rows.empty:
            same = np.isclose(
                first_rows["career_disp_avg_pre"].to_numpy(dtype=float),
                first_rows["DI"].to_numpy(dtype=float),
                atol=1e-8,
                rtol=0.0,
            )
            n_same = int(same.sum())
            if n_same > 0:
                pct = 100.0 * n_same / len(first_rows)
                msg = (
                    f"career_disp_avg_pre equals current DI on "
                    f"{n_same} first rows ({pct:.2f}%)"
                )
                # A tiny overlap can occur by coincidence (e.g., fallback equals DI).
                if pct > 1.0:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    # Soft signal: excessive defaults for team_venue features often indicates bad joins.
    if "team_venue_win_rate" in df.columns:
        default_rate = float((df["team_venue_win_rate"] == 0.5).mean())
        if default_rate > 0.85:
            warnings.append(
                f"team_venue_win_rate defaulted to 0.5 for {default_rate:.1%} of rows"
            )

    if warnings:
        for w in warnings:
            print(f"  VALIDATION WARNING (temporal): {w}")

    if errors:
        msg = "Temporal integrity validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValidationError(msg)

    print(f"  Validation passed (temporal integrity: {len(feature_cols)} features)")


def validate_umpires(df):
    """Validate umpire data.

    Checks:
      - Required columns present
      - Umpire panel sizes (typically 3 per match)
      - Career games within reasonable range

    Prints warnings for issues, no exceptions raised.
    """
    warnings = []

    required = ["match_id", "umpire_name", "umpire_career_games"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns: {missing}")

    if not missing:
        dupes = df.duplicated(subset=["match_id", "umpire_name"], keep=False).sum()
        if dupes > 0:
            warnings.append(f"{dupes} duplicate (match_id, umpire_name) rows")

        blank_names = (
            df["umpire_name"].isna()
            | (df["umpire_name"].astype(str).str.strip() == "")
        ).sum()
        if blank_names > 0:
            warnings.append(f"{blank_names} rows with blank umpire names")

        # Panel sizes
        panel_sizes = df.groupby("match_id")["umpire_name"].nunique()
        unusual = panel_sizes[(panel_sizes < 3) | (panel_sizes > 4)]
        if len(unusual) > 0:
            warnings.append(
                f"{len(unusual)} matches with unusual panel size "
                f"(min={panel_sizes.min()}, max={panel_sizes.max()})"
            )

        # Career games range
        if "umpire_career_games" in df.columns:
            neg = (df["umpire_career_games"] < 0).sum()
            if neg > 0:
                warnings.append(f"{neg} rows with negative career games")
            zero_rate = float((df["umpire_career_games"] == 0).mean())
            if zero_rate > 0.35:
                warnings.append(
                    f"high zero-rate in umpire_career_games ({zero_rate:.1%})"
                )

    if warnings:
        for w in warnings:
            print(f"  VALIDATION WARNING (umpires): {w}")
    else:
        print(f"  Validation passed (umpires: {len(df)} rows, "
              f"{df['match_id'].nunique()} matches)")


def validate_coaches(df):
    """Validate coach data.

    Checks:
      - Required columns present
      - Win pct within [0, 100]
      - One coach per (match_id, team)
    """
    warnings = []

    required = ["match_id", "team", "coach"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns: {missing}")

    if not missing:
        dupes = df.duplicated(subset=["match_id", "team"], keep=False).sum()
        if dupes > 0:
            warnings.append(f"{dupes} duplicate (match_id, team) rows")

        if "coach_win_pct" in df.columns:
            invalid = ((df["coach_win_pct"] < 0) | (df["coach_win_pct"] > 100)).sum()
            if invalid > 0:
                warnings.append(f"{invalid} rows with win_pct outside [0, 100]")

    if warnings:
        for w in warnings:
            print(f"  VALIDATION WARNING (coaches): {w}")
    else:
        print(f"  Validation passed (coaches: {len(df)} rows)")


def validate_player_profiles(df):
    """Validate player profile data.

    Checks:
      - Height within reasonable range (150-220 cm)
      - Weight within reasonable range (60-130 kg)
    """
    warnings = []

    if "height_cm" in df.columns:
        h = df["height_cm"].dropna()
        out_of_range = ((h < 150) | (h > 220)).sum()
        if out_of_range > 0:
            warnings.append(f"{out_of_range} players with height outside 150-220 cm")

    if "weight_kg" in df.columns:
        w = df["weight_kg"].dropna()
        out_of_range = ((w < 60) | (w > 130)).sum()
        if out_of_range > 0:
            warnings.append(f"{out_of_range} players with weight outside 60-130 kg")

    if warnings:
        for w in warnings:
            print(f"  VALIDATION WARNING (profiles): {w}")
    else:
        print(f"  Validation passed (profiles: {len(df)} rows)")


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
