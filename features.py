"""
AFL Prediction Pipeline — Feature Engineering
===============================================
The core of the pipeline. Takes the cleaned player_matches DataFrame
and computes ~80 features per (player, team, match) row.

Feature categories:
  A. Career / age features
  B. Recency-weighted rolling averages (form)
  C. Venue features
  D. Opponent defense features
  E. Player-vs-defender matchup features
  F. Team context features
  G. Scoring pattern features (from quarter-level data)
  H. Role classification
"""

import numpy as np
import pandas as pd
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MARKET_ENV_STATS_CACHE = None


def _rolling_linear_slope_shifted(series, window=5, min_periods=3):
    """Compute rolling linear slope on a shifted series without `np.polyfit`.

    This matches the prior behavior of `shift(1).rolling(...).apply(polyfit)`:
    any NaN inside the active window yields NaN, which callers typically
    convert to 0 with `fillna(0)`.
    """
    series = pd.to_numeric(series, errors="coerce")

    # Fast path for the only configuration used in this codebase. The
    # original implementation was:
    #   shift(1).rolling(5, min_periods=3).apply(polyfit)
    # Because the leading shifted NaN remained inside early windows, values
    # only became non-null once five full prior observations existed.
    # For x = [0,1,2,3,4], the slope simplifies to:
    #   (-2*y0 - y1 + y3 + 2*y4) / 10
    if window == 5 and min_periods <= 5:
        return (
            -2.0 * series.shift(5)
            - 1.0 * series.shift(4)
            + 1.0 * series.shift(2)
            + 2.0 * series.shift(1)
        ) / 10.0

    values = series.to_numpy(dtype=float, copy=False)
    n_values = len(values)
    result = np.full(n_values, np.nan, dtype=np.float64)
    if n_values == 0:
        return pd.Series(result, index=series.index)

    shifted = np.empty(n_values, dtype=np.float64)
    shifted[0] = np.nan
    if n_values > 1:
        shifted[1:] = values[:-1]

    x_cache = {
        length: np.arange(length, dtype=np.float64)
        for length in range(min_periods, window + 1)
    }
    sum_x_cache = {length: float(x.sum()) for length, x in x_cache.items()}
    denom_cache = {
        length: float(length * np.square(x).sum() - sum_x_cache[length] ** 2)
        for length, x in x_cache.items()
    }

    for end in range(n_values):
        start = max(0, end - window + 1)
        y = shifted[start:end + 1]
        length = len(y)
        if length < min_periods or not np.isfinite(y).all():
            continue

        x = x_cache[length]
        sum_y = float(y.sum())
        sum_xy = float(np.dot(x, y))
        denom = denom_cache[length]
        if denom == 0:
            continue
        result[end] = (length * sum_xy - sum_x_cache[length] * sum_y) / denom

    return pd.Series(result, index=series.index)


def _group_shifted_rolling_mean(df, key_cols, value_col, window, min_periods=1):
    """Rolling mean of prior observations for a grouped column."""
    shifted = df.groupby(key_cols, observed=True)[value_col].shift(1)
    key_arrays = [df[col] for col in key_cols]
    rolled = shifted.groupby(key_arrays, observed=True).rolling(
        window, min_periods=min_periods
    ).mean()
    return rolled.reset_index(level=list(range(len(key_cols))), drop=True)


def _group_shifted_expanding_mean(df, key_cols, value_col, min_periods=1):
    """Expanding mean of prior observations for a grouped column."""
    shifted = df.groupby(key_cols, observed=True)[value_col].shift(1)
    key_arrays = [df[col] for col in key_cols]
    expanded = shifted.groupby(key_arrays, observed=True).expanding(
        min_periods=min_periods
    ).mean()
    return expanded.reset_index(level=list(range(len(key_cols))), drop=True)


def _group_shifted_expanding_count(df, key_cols, value_col):
    """Expanding count of prior observations for a grouped column."""
    shifted = df.groupby(key_cols, observed=True)[value_col].shift(1)
    key_arrays = [df[col] for col in key_cols]
    expanded = shifted.groupby(key_arrays, observed=True).expanding().count()
    return expanded.reset_index(level=list(range(len(key_cols))), drop=True)


def _grouped_shifted_slope_5(grouped, value_col):
    """Closed-form slope over the previous five grouped observations."""
    series_group = grouped[value_col]
    return (
        -2.0 * series_group.shift(5)
        - 1.0 * series_group.shift(4)
        + 1.0 * series_group.shift(2)
        + 2.0 * series_group.shift(1)
    ) / 10.0

def _era_weight(year):
    """Return era weight for a given year based on config.ERA_WEIGHTS."""
    for (lo, hi), w in config.ERA_WEIGHTS.items():
        if lo <= year <= hi:
            return w
    return 0.3  # fallback for years outside defined eras


def _decay_weight(days_ago, half_life=None):
    """Exponential decay weight: 0.5^(days_ago / half_life)."""
    half_life = half_life or config.RECENCY_DECAY_HALF_LIFE
    if pd.isna(days_ago) or days_ago < 0:
        return 0.0
    return 0.5 ** (days_ago / half_life)


def _combined_weight(year, days_ago):
    """Era weight * decay weight — used for all rolling features."""
    return _era_weight(year) * _decay_weight(days_ago)


def _estimate_market_environment_stats():
    """Estimate market-total-score conversion ratios from historical data.

    Returns:
      dict with:
        - points_per_goal_from_market_total
        - points_per_disposal_from_market_total
        - n_matches
    """
    global _MARKET_ENV_STATS_CACHE
    if _MARKET_ENV_STATS_CACHE is not None:
        return _MARKET_ENV_STATS_CACHE

    # Fallbacks if source data is unavailable
    fallback_points_per_goal = float(getattr(config, "MARKET_POINTS_PER_GOAL_FALLBACK", 6.2))
    fallback_points_per_disp = float(getattr(config, "MARKET_POINTS_PER_DISPOSAL_FALLBACK", 0.38))
    stats = {
        "points_per_goal_from_market_total": fallback_points_per_goal,
        "points_per_disposal_from_market_total": fallback_points_per_disp,
        "n_matches": 0,
    }

    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    odds_path = getattr(config, "ODDS_PARQUET_PATH", config.BASE_STORE_DIR / "odds.parquet")
    if not tm_path.exists() or not odds_path.exists():
        _MARKET_ENV_STATS_CACHE = stats
        return stats

    try:
        tm = pd.read_parquet(tm_path, columns=["match_id", "score", "GL", "DI"])
        odds = pd.read_parquet(odds_path, columns=["match_id", "market_total_score"])
    except Exception:
        _MARKET_ENV_STATS_CACHE = stats
        return stats

    match_totals = (
        tm.groupby("match_id", observed=True)
        .agg(
            actual_total_score=("score", "sum"),
            actual_total_goals=("GL", "sum"),
            actual_total_disposals=("DI", "sum"),
        )
        .reset_index()
    )
    merged = match_totals.merge(odds, on="match_id", how="inner")
    merged["market_total_score"] = pd.to_numeric(merged["market_total_score"], errors="coerce")
    merged = merged[
        merged["market_total_score"].notna()
        & (merged["market_total_score"] > 0)
    ].copy()

    if merged.empty:
        _MARKET_ENV_STATS_CACHE = stats
        return stats

    valid_goal = merged["actual_total_goals"] > 0
    if valid_goal.any():
        points_per_goal = (
            merged.loc[valid_goal, "market_total_score"].sum()
            / merged.loc[valid_goal, "actual_total_goals"].sum()
        )
        if np.isfinite(points_per_goal) and points_per_goal > 0:
            stats["points_per_goal_from_market_total"] = float(points_per_goal)

    valid_disp = merged["actual_total_disposals"] > 0
    if valid_disp.any():
        points_per_disp = (
            merged.loc[valid_disp, "market_total_score"].sum()
            / merged.loc[valid_disp, "actual_total_disposals"].sum()
        )
        if np.isfinite(points_per_disp) and points_per_disp > 0:
            stats["points_per_disposal_from_market_total"] = float(points_per_disp)

    stats["n_matches"] = int(len(merged))
    _MARKET_ENV_STATS_CACHE = stats
    return stats


# ---------------------------------------------------------------------------
# A. Career / Age Features
# ---------------------------------------------------------------------------

def add_career_features(df):
    """Add features derived from player_details (already joined in clean.py).
    These columns already exist: age_years, career_games_pre, career_goal_avg_pre, etc.
    We add a few derived ones."""
    df = df.sort_values(["player", "date"]).copy()

    # Age squared (captures non-linear peak-years effect ~25-29)
    df["age_squared"] = df["age_years"] ** 2

    # Capped career goal avg to reduce outlier influence from prolific forwards
    # Uses pre-game version to avoid same-game leakage
    if "career_goal_avg_pre" in df.columns:
        df["career_goal_avg_capped"] = df["career_goal_avg_pre"].clip(upper=2.5)

    # Career disposal average before this match (no leakage via shift(1))
    # Needed for market-implied player disposal share.
    if "career_disp_avg_pre" not in df.columns:
        grp = df.groupby("player", observed=True)["DI"]
        pre_avg = grp.transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        # Cold-start fallback: use past-only league prior by date.
        # This avoids first-match leakage from current-row DI values.
        sort_cols = ["date"] + (["match_id"] if "match_id" in df.columns else [])
        date_sorted = df.sort_values(sort_cols)
        league_prior = date_sorted["DI"].shift(1).expanding(min_periods=1).mean()
        league_prior = league_prior.reindex(df.index)
        fallback_disp = float(getattr(config, "CAREER_DISP_AVG_FALLBACK", 15.0))
        df["career_disp_avg_pre"] = (
            pre_avg.fillna(league_prior).fillna(fallback_disp).astype(np.float32)
        )

    return df


# ---------------------------------------------------------------------------
# B. Recency-Weighted Rolling Averages
# ---------------------------------------------------------------------------

def add_rolling_features(df):
    """Compute weighted rolling averages for player form.

    For each window in ROLLING_WINDOWS, compute the mean of the last N
    matches for key stat columns. Uses shift(1) so the current match
    is never included in its own features.

    Includes both raw-stat and rate-normalised rolling features.
    """
    df = df.sort_values(["player", "team", "date"]).copy()

    # Handle did_not_play rows: filter them out before computing rolling
    # features so windows of 3/5/10 always reference 3/5/10 real games.
    # DNP rows (medical subs, emergencies) have 0 stats and would pollute
    # rolling averages if included.
    if "did_not_play" in df.columns:
        dnp_mask = df["did_not_play"].values
    else:
        dnp_mask = (df["pct_played"] == 0).values if "pct_played" in df.columns else np.zeros(len(df), dtype=bool)

    n_dnp = dnp_mask.sum()
    if n_dnp > 0:
        # Save original index for re-insertion
        df["_orig_idx"] = np.arange(len(df))
        dnp_indices = df.loc[dnp_mask, "_orig_idx"].values

        # Compute rolling features on real-games-only subset
        df_real = df[~dnp_mask].copy()
    else:
        df_real = df

    # Columns to compute rolling averages for (raw stats)
    roll_cols = {
        "GL": "gl", "BH": "bh", "DI": "di", "MK": "mk", "TK": "tk",
        "IF": "if50", "CL": "cl", "HO": "ho", "GA": "ga", "MI": "mi",
        "CM": "cm", "CP": "cp", "FF": "ff", "RB": "rb", "one_pct": "one_pct",
        "pct_played": "tog", "UP": "up",
    }

    # Rate-normalised columns for rolling averages
    rate_roll_cols = {
        "GL_rate": "gl_rate", "BH_rate": "bh_rate", "DI_rate": "di_rate",
        "MK_rate": "mk_rate", "TK_rate": "tk_rate", "IF_rate": "if50_rate",
        "CP_rate": "cp_rate",
    }

    # Group by (player, team) compound key — on real games only
    grouped = df_real.groupby(["player", "team"], observed=True)

    # Key stats get exponentially-weighted rolling means (recent form matters more)
    ewm_cols = {"GL": "gl", "BH": "bh", "MI": "mi", "IF": "if50", "DI": "di", "MK": "mk", "pct_played": "tog"}
    ewm_rate_cols = {"GL_rate": "gl_rate", "DI_rate": "di_rate", "MK_rate": "mk_rate"}

    for window in config.ROLLING_WINDOWS:
        for src_col, feat_name in roll_cols.items():
            col_name = f"player_{feat_name}_avg_{window}"
            min_p = min(2, window)
            df_real[col_name] = grouped[src_col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=min_p).mean()
            )

    # Rate-based rolling averages
    for window in config.ROLLING_WINDOWS:
        for src_col, feat_name in rate_roll_cols.items():
            if src_col in df_real.columns:
                col_name = f"player_{feat_name}_avg_{window}"
                min_p = min(2, window)
                df_real[col_name] = grouped[src_col].transform(
                    lambda s: s.shift(1).rolling(window, min_periods=min_p).mean()
                )

    # Exponentially-weighted averages for key stats
    for src_col, feat_name in ewm_cols.items():
        col_name = f"player_{feat_name}_ewm_5"
        df_real[col_name] = grouped[src_col].transform(
            lambda s: s.shift(1).ewm(span=5, min_periods=2).mean()
        )

    # Rate-based EWM averages
    for src_col, feat_name in ewm_rate_cols.items():
        if src_col in df_real.columns:
            col_name = f"player_{feat_name}_ewm_5"
            df_real[col_name] = grouped[src_col].transform(
                lambda s: s.shift(1).ewm(span=5, min_periods=2).mean()
            )

    # Rolling accuracy: GL / (GL + BH) over last N matches
    for window in config.ROLLING_WINDOWS:
        min_p = min(2, window)
        gl_sum = grouped["GL"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_p).sum()
        )
        bh_sum = grouped["BH"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_p).sum()
        )
        total = gl_sum + bh_sum
        df_real[f"player_accuracy_{window}"] = np.where(total > 0, gl_sum / total, 0.5)

    # ---- Vectorized streak calculations ----
    shifted_gl = grouped["GL"].shift(1)

    # Scoring streak: consecutive matches with GL >= 1
    scored = shifted_gl.ge(1).fillna(False).astype(int)
    epoch = (scored == 0).cumsum()
    df_real["player_gl_streak"] = scored.groupby(epoch, observed=True).cumsum()

    # Exponentially-decayed weighted streak
    decay = config.STREAK_DECAY
    streak = df_real["player_gl_streak"]
    df_real["player_gl_streak_weighted"] = np.where(
        streak > 0,
        (1.0 - decay ** streak) / (1.0 - decay),
        0.0,
    )

    # Cold streak: consecutive 0-goal games
    is_cold = shifted_gl.eq(0).fillna(False).astype(int)
    epoch_cold = (is_cold == 0).cumsum()
    df_real["player_gl_cold_streak"] = is_cold.groupby(epoch_cold, observed=True).cumsum()

    # Form ratio: recent 3-game avg / career avg
    recent_3_avg = grouped["GL"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    career_avg = grouped["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df_real["player_form_ratio"] = np.where(
        career_avg > 0, recent_3_avg / career_avg, 1.0
    )
    df_real["player_form_ratio"] = df_real["player_form_ratio"].clip(0, 5.0)

    # Binary hot/cold flags
    df_real["player_is_hot"] = (df_real["player_form_ratio"] >= config.HOT_THRESHOLD).astype(int)
    df_real["player_is_cold"] = (df_real["player_form_ratio"] <= config.COLD_THRESHOLD).astype(int)

    # Goal volatility over last 5 games
    df_real["player_gl_volatility_5"] = grouped["GL"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)

    # Streak just broke
    prev_streak = df_real.groupby(["player", "team"], observed=True)["player_gl_streak"].shift(1)
    df_real["player_streak_just_broke"] = (
        (df_real["player_gl_streak"] == 0) & (prev_streak >= config.STREAK_BROKE_MIN)
    ).astype(int)

    # Trend: linear slope of GL over last 5 matches
    df_real["player_gl_trend_5"] = _grouped_shifted_slope_5(grouped, "GL").fillna(0)

    pre_roll_updates = {}

    # Days since last match (computed on real games only — skip DNP gaps)
    pre_roll_updates["days_since_last_match"] = grouped["date"].transform(
        lambda s: s.diff().dt.days
    )
    pre_roll_updates["is_returning_from_break"] = (
        pre_roll_updates["days_since_last_match"].fillna(0) > 21
    ).astype(int)

    # Season-to-date goals
    season_group = df_real.groupby(["player", "team", "year"], observed=True)
    pre_roll_updates["season_goals_total"] = season_group["GL"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    # --- Disposal-specific features ---
    pre_roll_updates["player_di_volatility_5"] = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)
    pre_roll_updates["player_di_trend_5"] = _grouped_shifted_slope_5(grouped, "DI").fillna(0)

    # Kick/handball ratio
    for window in [3, 5]:
        ki_sum = grouped["KI"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).sum()
        )
        hb_sum = grouped["HB"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).sum()
        )
        total_kh = ki_sum + hb_sum
        pre_roll_updates[f"player_ki_hb_ratio_{window}"] = np.where(
            total_kh > 0, ki_sum / total_kh, 0.5
        )

    # Season-to-date disposals
    pre_roll_updates["season_disposals_total"] = season_group["DI"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    # Season hit-rate features (% of current-season games meeting thresholds)
    season_key_arrays = [df_real["player"], df_real["team"], df_real["year"]]
    pre_roll_updates["season_gl_1plus_rate"] = season_group["GL"].shift(1).ge(1).groupby(
        season_key_arrays, observed=True
    ).expanding(min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).fillna(0)
    pre_roll_updates["season_disp_20plus_rate"] = season_group["DI"].shift(1).ge(20).groupby(
        season_key_arrays, observed=True
    ).expanding(min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).fillna(0)
    pre_roll_updates["season_mk_3plus_rate"] = season_group["MK"].shift(1).ge(3).groupby(
        season_key_arrays, observed=True
    ).expanding(min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True).fillna(0)

    # Season-to-date rate aggregations
    if "GL_rate" in df_real.columns:
        pre_roll_updates["season_goals_rate_avg"] = season_group["GL_rate"].transform(
            lambda s: s.shift(1).expanding().mean()
        ).fillna(0)
        pre_roll_updates["player_gl_rate_volatility_5"] = grouped["GL_rate"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).std()
        ).fillna(0)
        pre_roll_updates["player_gl_rate_trend_5"] = _grouped_shifted_slope_5(
            grouped, "GL_rate"
        ).fillna(0)
    if "DI_rate" in df_real.columns:
        pre_roll_updates["season_disposals_rate_avg"] = season_group["DI_rate"].transform(
            lambda s: s.shift(1).expanding().mean()
        ).fillna(0)
        pre_roll_updates["player_di_rate_volatility_5"] = grouped["DI_rate"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).std()
        ).fillna(0)
        pre_roll_updates["player_di_rate_trend_5"] = _grouped_shifted_slope_5(
            grouped, "DI_rate"
        ).fillna(0)

    # --- Phase 1A: TOG volatility, trend, and disposal intensity ---
    if "pct_played" in df_real.columns:
        pre_roll_updates["player_tog_volatility_5"] = grouped["pct_played"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).std()
        ).fillna(0)
        pre_roll_updates["player_tog_trend_5"] = _grouped_shifted_slope_5(
            grouped, "pct_played"
        ).fillna(0)

        # Disposal per minute proxy: DI / (pct_played/100 * 120) rolling avg
        # 120 = typical game length in minutes (4 x 30min quarters incl. time-on)
        tog_frac = df_real["pct_played"].clip(lower=5) / 100.0
        di_per_min = df_real["DI"] / (tog_frac * 120.0)
        di_per_min_grouped = di_per_min.groupby([df_real["player"], df_real["team"]], observed=True)
        pre_roll_updates["player_disp_per_minute_5"] = di_per_min_grouped.transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).mean()
        ).fillna(0)

    # --- Phase 1C: Contested disposal ratio ---
    cp_sum_5 = grouped["CP"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).sum()
    )
    di_sum_5 = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).sum()
    )
    pre_roll_updates["player_contested_disp_ratio_5"] = np.where(
        di_sum_5 > 0, cp_sum_5 / di_sum_5, 0.5
    )

    # --- Phase 1G: Kick/handball ratio trend ---
    pre_roll_updates["player_ki_hb_trend_5"] = _grouped_shifted_slope_5(
        grouped, "KI"
    ).fillna(0) - _grouped_shifted_slope_5(grouped, "HB").fillna(0)

    if pre_roll_updates:
        overlap = [col for col in pre_roll_updates if col in df_real.columns]
        if overlap:
            df_real = df_real.drop(columns=overlap, errors="ignore")
        df_real = pd.concat(
            [df_real, pd.DataFrame(pre_roll_updates, index=df_real.index)],
            axis=1,
        ).copy()

    # --- Phase 1H: Disposal + marks rolling features ---
    # Batch these late-stage rolling columns into a side frame so we avoid
    # repeatedly fragmenting a large DataFrame with many single-column inserts.
    rolling_feature_updates = {}
    shifted_di = grouped["DI"].shift(1)

    # High disposal streak: consecutive matches with DI >= 20
    di_high = shifted_di.ge(20).fillna(False).astype(int)
    epoch_di_high = (di_high == 0).cumsum()
    rolling_feature_updates["player_di_streak_high"] = di_high.groupby(
        epoch_di_high, observed=True
    ).cumsum()

    # Low disposal streak: consecutive matches with DI < 12
    di_low = shifted_di.lt(12).fillna(False).astype(int)
    epoch_di_low = (di_low == 0).cumsum()
    rolling_feature_updates["player_di_streak_low"] = di_low.groupby(
        epoch_di_low, observed=True
    ).cumsum()

    # Disposal cold streak: consecutive games with < 15 disposals
    disp_cold = shifted_di.lt(15).fillna(False).astype(int)
    epoch_di_cold = (disp_cold == 0).cumsum()
    rolling_feature_updates["player_disp_cold_streak"] = disp_cold.groupby(
        epoch_di_cold, observed=True
    ).cumsum()

    # Disposal ceiling and floor over last 5 games
    rolling_feature_updates["player_di_ceiling_5"] = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).max()
    ).fillna(0)
    rolling_feature_updates["player_di_floor_5"] = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).min()
    ).fillna(0)

    # Disposal consistency: 1 - (volatility / avg) — consistent players are more predictable
    _di_avg_5 = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )
    _di_std_5 = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)
    rolling_feature_updates["player_di_consistency_5"] = pd.Series(
        np.where(_di_avg_5 > 0, 1.0 - (_di_std_5 / _di_avg_5), 0),
        index=df_real.index,
    ).clip(-1, 1).fillna(0)

    # --- Mark streaks ---
    shifted_mk = grouped["MK"].shift(1)

    # Mark streak: consecutive games with 3+ marks
    mk_3plus = shifted_mk.ge(3).fillna(False).astype(int)
    epoch_mk = (mk_3plus == 0).cumsum()
    rolling_feature_updates["player_mk_streak_3plus"] = mk_3plus.groupby(
        epoch_mk, observed=True
    ).cumsum()

    # Mark cold streak: consecutive games with < 2 marks
    mk_cold = shifted_mk.lt(2).fillna(False).astype(int)
    epoch_mk_cold = (mk_cold == 0).cumsum()
    rolling_feature_updates["player_mk_cold_streak"] = mk_cold.groupby(
        epoch_mk_cold, observed=True
    ).cumsum()

    # --- Marks-specific rolling features ---
    # Marks per disposal ratio (rolling 5)
    mk_sum_5 = grouped["MK"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).sum()
    )
    di_sum_5 = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).sum()
    )
    rolling_feature_updates["player_mk_per_disp_5"] = np.where(
        di_sum_5 > 0, mk_sum_5 / di_sum_5, 0
    )

    # Contested marks ratio: CM / MK (rolling 5)
    if "CM" in df_real.columns:
        cm_sum_5 = grouped["CM"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).sum()
        )
        rolling_feature_updates["player_contested_mk_ratio_5"] = np.where(
            mk_sum_5 > 0, cm_sum_5 / mk_sum_5, 0
        )

    # Mark form ratio: recent 3 avg / career avg (clipped 0-5)
    mk_recent_3 = grouped["MK"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    mk_career = grouped["MK"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    rolling_feature_updates["player_mk_form_ratio"] = pd.Series(
        np.where(mk_career > 0, mk_recent_3 / mk_career, 1.0),
        index=df_real.index,
    ).clip(0, 5.0)

    # Mark volatility over last 5 games
    rolling_feature_updates["player_mk_volatility_5"] = grouped["MK"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)

    # Mark trend: linear slope of MK over last 5 matches
    rolling_feature_updates["player_mk_trend_5"] = _grouped_shifted_slope_5(
        grouped, "MK"
    ).fillna(0)

    # --- Marks: aerial & position proxy features ---
    # One-percenters avg (aerial contest indicator: spoils, contested marks, etc.)
    if "one_pct" in df_real.columns:
        rolling_feature_updates["player_one_pct_avg_5"] = grouped["one_pct"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).mean()
        ).fillna(0)

    # Ruck indicator: hitouts > 0 means this player contests ruck = fewer marks
    if "HO" in df_real.columns:
        ho_sum_5 = grouped["HO"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).sum()
        ).fillna(0)
        rolling_feature_updates["player_is_ruck_5"] = (ho_sum_5 > 0).astype(np.float32)

    # Mark ceiling and floor over last 5 games
    rolling_feature_updates["player_mk_ceiling_5"] = grouped["MK"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).max()
    ).fillna(0)
    rolling_feature_updates["player_mk_floor_5"] = grouped["MK"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).min()
    ).fillna(0)

    # Contested marks per game avg (absolute, not ratio) — for mark-taker signal
    if "CM" in df_real.columns:
        rolling_feature_updates["player_cm_avg_5"] = grouped["CM"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).mean()
        ).fillna(0)

    # Disposal form ratio: last 3 avg / last 10 avg
    di_recent_3 = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    di_career = grouped["DI"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    rolling_feature_updates["player_di_form_ratio"] = pd.Series(
        np.where(di_career > 0, di_recent_3 / di_career, 1.0),
        index=df_real.index,
    ).clip(0, 5.0)

    if rolling_feature_updates:
        overlap = [col for col in rolling_feature_updates if col in df_real.columns]
        if overlap:
            df_real = df_real.drop(columns=overlap, errors="ignore")
        df_real = pd.concat(
            [df_real, pd.DataFrame(rolling_feature_updates, index=df_real.index)],
            axis=1,
        ).copy()

    # --- Exclude DNP rows from output ---
    if n_dnp > 0:
        df = df_real.drop(columns=["_orig_idx"], errors="ignore")
        print(f"  Excluded {n_dnp} did_not_play rows from rolling calculations")
    else:
        df = df_real

    return df


# ---------------------------------------------------------------------------
# C. Venue Features
# ---------------------------------------------------------------------------

def add_venue_features(df):
    """Player performance at specific venues vs their overall average."""
    df = df.sort_values(["player", "team", "date"]).copy()

    # Build lookup: (player, team, venue) → historical stats
    # We need to compute these as expanding means up to (but not including)
    # the current match, for each (player, team, venue) group.

    venue_group = df.groupby(["player", "team", "venue"], observed=True)

    df["player_gl_at_venue_avg"] = venue_group["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["player_bh_at_venue_avg"] = venue_group["BH"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    # Player's overall GL average (for computing venue diff)
    player_group = df.groupby(["player", "team"], observed=True)
    overall_gl_avg = player_group["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    df["player_gl_venue_diff"] = df["player_gl_at_venue_avg"] - overall_gl_avg

    # Venue average goals per team (regardless of player)
    # — reflects ground dimensions / conditions
    # Temporal: use only past matches via shift(1).expanding()
    match_team_goals = (
        df.groupby(["match_id", "team", "venue"], observed=True)
        .agg(team_total_goals=("GL", "sum"), date=("date", "first"))
        .reset_index()
    )
    match_team_goals = match_team_goals.sort_values("date")
    match_team_goals["venue_avg_goals_per_team"] = (
        match_team_goals.groupby("venue", observed=True)["team_total_goals"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    venue_merge = match_team_goals[["match_id", "team", "venue_avg_goals_per_team"]].drop_duplicates()
    df = df.merge(venue_merge, on=["match_id", "team"], how="left")

    # Fill NaN venue features — use player's career average as fallback
    # instead of 0 (which wrongly implies "scored 0 goals here")
    if "career_goal_avg_pre" in df.columns:
        df["player_gl_at_venue_avg"] = df["player_gl_at_venue_avg"].fillna(
            df["career_goal_avg_pre"]
        )
    else:
        df["player_gl_at_venue_avg"] = df["player_gl_at_venue_avg"].fillna(0)

    df["player_bh_at_venue_avg"] = df["player_bh_at_venue_avg"].fillna(0)
    df["player_gl_venue_diff"] = df["player_gl_venue_diff"].fillna(0)

    # Venue-level average: use overall league average as fallback
    venue_mean = df["venue_avg_goals_per_team"].dropna().mean()
    df["venue_avg_goals_per_team"] = df["venue_avg_goals_per_team"].fillna(
        venue_mean if pd.notna(venue_mean) else 0
    )

    return df


# ---------------------------------------------------------------------------
# D. Opponent Defense Features
# ---------------------------------------------------------------------------

def add_opponent_features(df):
    """How leaky is the opponent's defense? Computed from team-level aggregates."""
    df = df.sort_values(["date"]).copy()

    # Step 1: compute goals conceded per match per team
    # Goals conceded by team X = goals scored by X's opponent in that match
    team_match_goals = (
        df.groupby(["match_id", "team", "opponent"], observed=True)["GL"]
        .sum()
        .reset_index()
        .rename(columns={"GL": "team_goals_scored"})
    )

    # Goals conceded by opponent = goals scored against them by the other team
    # If team A scored X against opponent B, then B conceded X goals
    goals_conceded = team_match_goals.rename(columns={
        "opponent": "conceding_team",
        "team_goals_scored": "goals_conceded",
    })[["match_id", "conceding_team", "goals_conceded"]]

    # Merge match date for sorting
    match_dates = df[["match_id", "date"]].drop_duplicates()
    goals_conceded = goals_conceded.merge(match_dates, on="match_id", how="left")
    goals_conceded = goals_conceded.sort_values("date")

    # Rolling averages of goals conceded
    gc_group = goals_conceded.groupby("conceding_team", observed=True)
    for window in [5, 10]:
        goals_conceded[f"goals_conceded_avg_{window}"] = _group_shifted_rolling_mean(
            goals_conceded, ["conceding_team"], "goals_conceded", window, min_periods=1
        )

    # Map back to the main df: for each row, look up the opponent's
    # defensive stats
    opp_defense = goals_conceded[
        ["match_id", "conceding_team", "goals_conceded_avg_5",
         "goals_conceded_avg_10"]
    ].drop_duplicates()

    df = df.merge(
        opp_defense,
        left_on=["match_id", "opponent"],
        right_on=["match_id", "conceding_team"],
        how="left",
    )
    df = df.rename(columns={
        "goals_conceded_avg_5": "opp_goals_conceded_avg_5",
        "goals_conceded_avg_10": "opp_goals_conceded_avg_10",
    })
    df = df.drop(columns=["conceding_team"], errors="ignore")

    # Step 2: player vs this specific opponent history
    opp_group = df.groupby(["player", "team", "opponent"], observed=True)
    df["player_vs_opp_gl_avg"] = _group_shifted_expanding_mean(
        df, ["player", "team", "opponent"], "GL", min_periods=1
    )
    df["player_vs_opp_games"] = _group_shifted_expanding_count(
        df, ["player", "team", "opponent"], "GL"
    )

    # Diff: player's avg vs opponent minus their overall avg
    overall = _group_shifted_expanding_mean(
        df, ["player", "team"], "GL", min_periods=1
    )
    df["player_vs_opp_gl_diff"] = df["player_vs_opp_gl_avg"] - overall

    # Mask out matchup features where sample too small
    mask = df["player_vs_opp_games"].fillna(0) < config.MATCHUP_MIN_MATCHES
    df.loc[mask, "player_vs_opp_gl_diff"] = 0
    df.loc[mask, "player_vs_opp_gl_avg"] = np.nan

    # Step 2b: player vs this specific opponent — marks history
    df["player_vs_opp_mk_avg"] = _group_shifted_expanding_mean(
        df, ["player", "team", "opponent"], "MK", min_periods=1
    )
    df.loc[mask, "player_vs_opp_mk_avg"] = np.nan

    # Step 3: opponent disposal concession (for disposal prediction)
    team_match_disp = (
        df.groupby(["match_id", "team", "opponent"], observed=True)["DI"]
        .sum()
        .reset_index()
        .rename(columns={"DI": "team_disp_scored"})
    )
    # Disposals conceded by opponent = disposals accumulated against them
    # If team A had X disposals against opponent B, then B conceded X disposals
    disp_conceded = team_match_disp.rename(columns={
        "opponent": "conceding_team",
        "team_disp_scored": "disp_conceded",
    })[["match_id", "conceding_team", "disp_conceded"]]
    disp_conceded = disp_conceded.merge(match_dates, on="match_id", how="left")
    disp_conceded = disp_conceded.sort_values("date")

    dc_group = disp_conceded.groupby("conceding_team", observed=True)
    for window in [5, 10]:
        disp_conceded[f"disp_conceded_avg_{window}"] = _group_shifted_rolling_mean(
            disp_conceded, ["conceding_team"], "disp_conceded", window, min_periods=1
        )

    opp_disp_defense = disp_conceded[
        ["match_id", "conceding_team", "disp_conceded_avg_5", "disp_conceded_avg_10"]
    ].drop_duplicates()

    df = df.merge(
        opp_disp_defense,
        left_on=["match_id", "opponent"],
        right_on=["match_id", "conceding_team"],
        how="left",
    )
    df = df.rename(columns={
        "disp_conceded_avg_5": "opp_disp_conceded_avg_5",
        "disp_conceded_avg_10": "opp_disp_conceded_avg_10",
    })
    df = df.drop(columns=["conceding_team"], errors="ignore")

    # Step 3b: opponent marks concession (for marks prediction)
    team_match_mk = (
        df.groupby(["match_id", "team", "opponent"], observed=True)["MK"]
        .sum()
        .reset_index()
        .rename(columns={"MK": "team_mk_scored"})
    )
    mk_conceded = team_match_mk.rename(columns={
        "opponent": "conceding_team",
        "team_mk_scored": "marks_conceded",
    })[["match_id", "conceding_team", "marks_conceded"]]
    mk_conceded = mk_conceded.merge(match_dates, on="match_id", how="left")
    mk_conceded = mk_conceded.sort_values("date")

    mc_group = mk_conceded.groupby("conceding_team", observed=True)
    for window in [5, 10]:
        mk_conceded[f"marks_conceded_avg_{window}"] = _group_shifted_rolling_mean(
            mk_conceded, ["conceding_team"], "marks_conceded", window, min_periods=1
        )

    opp_mk_defense = mk_conceded[
        ["match_id", "conceding_team", "marks_conceded_avg_5", "marks_conceded_avg_10"]
    ].drop_duplicates()

    df = df.merge(
        opp_mk_defense,
        left_on=["match_id", "opponent"],
        right_on=["match_id", "conceding_team"],
        how="left",
    )
    df = df.rename(columns={
        "marks_conceded_avg_5": "opp_marks_conceded_avg_5",
        "marks_conceded_avg_10": "opp_marks_conceded_avg_10",
    })
    df = df.drop(columns=["conceding_team"], errors="ignore")

    # Step 4: opponent contested possession differential
    # CP won by team vs CP won by their opponent → measures midfield dominance
    team_match_cp = (
        df.groupby(["match_id", "team", "opponent"], observed=True)["CP"]
        .sum()
        .reset_index()
        .rename(columns={"CP": "team_cp"})
    )
    # Get opponent CP for the same match
    opp_cp_lookup = team_match_cp[["match_id", "team", "team_cp"]].rename(
        columns={"team": "opp_team_cp", "team_cp": "opp_cp_in_match"}
    )
    team_match_cp = team_match_cp.merge(
        opp_cp_lookup, left_on=["match_id", "opponent"],
        right_on=["match_id", "opp_team_cp"], how="left",
    )
    team_match_cp["cp_diff"] = team_match_cp["team_cp"] - team_match_cp["opp_cp_in_match"]
    team_match_cp = team_match_cp.merge(match_dates, on="match_id", how="left")
    team_match_cp = team_match_cp.sort_values("date")

    cp_group = team_match_cp.groupby("team", observed=True)
    team_match_cp["cp_diff_avg_5"] = _group_shifted_rolling_mean(
        team_match_cp, ["team"], "cp_diff", 5, min_periods=1
    )

    # Map back: for each row, look up the OPPONENT's CP differential
    # (positive = opponent wins more contested ball → suppresses this player's disposals)
    opp_cp_defense = team_match_cp[
        ["match_id", "team", "cp_diff_avg_5"]
    ].drop_duplicates()
    df = df.merge(
        opp_cp_defense,
        left_on=["match_id", "opponent"],
        right_on=["match_id", "team"],
        how="left",
        suffixes=("", "_opp_cp"),
    )
    df = df.rename(columns={"cp_diff_avg_5": "opp_contested_poss_diff_5"})
    df = df.drop(columns=["team_opp_cp"], errors="ignore")

    # Fill NaNs — use league average for opponent defense, career avg for matchup
    opp_mean_5 = df["opp_goals_conceded_avg_5"].dropna().mean()
    opp_mean_10 = df["opp_goals_conceded_avg_10"].dropna().mean()
    df["opp_goals_conceded_avg_5"] = df["opp_goals_conceded_avg_5"].fillna(
        opp_mean_5 if pd.notna(opp_mean_5) else 0
    )
    df["opp_goals_conceded_avg_10"] = df["opp_goals_conceded_avg_10"].fillna(
        opp_mean_10 if pd.notna(opp_mean_10) else 0
    )
    opp_disp_mean_5 = df["opp_disp_conceded_avg_5"].dropna().mean()
    opp_disp_mean_10 = df["opp_disp_conceded_avg_10"].dropna().mean()
    df["opp_disp_conceded_avg_5"] = df["opp_disp_conceded_avg_5"].fillna(
        opp_disp_mean_5 if pd.notna(opp_disp_mean_5) else 0
    )
    df["opp_disp_conceded_avg_10"] = df["opp_disp_conceded_avg_10"].fillna(
        opp_disp_mean_10 if pd.notna(opp_disp_mean_10) else 0
    )
    opp_mk_mean_5 = df["opp_marks_conceded_avg_5"].dropna().mean()
    opp_mk_mean_10 = df["opp_marks_conceded_avg_10"].dropna().mean()
    df["opp_marks_conceded_avg_5"] = df["opp_marks_conceded_avg_5"].fillna(
        opp_mk_mean_5 if pd.notna(opp_mk_mean_5) else 0
    )
    df["opp_marks_conceded_avg_10"] = df["opp_marks_conceded_avg_10"].fillna(
        opp_mk_mean_10 if pd.notna(opp_mk_mean_10) else 0
    )

    # Opponent contested possession differential: 0 = neutral
    df["opp_contested_poss_diff_5"] = df["opp_contested_poss_diff_5"].fillna(0)

    # --- Phase 1D: Opponent tackle pressure ---
    team_match_tk = (
        df.groupby(["match_id", "team", "opponent"], observed=True)["TK"]
        .sum()
        .reset_index()
        .rename(columns={"TK": "team_tk_total"})
    )
    tk_by_team = team_match_tk[["match_id", "team", "team_tk_total"]].copy()
    tk_by_team = tk_by_team.merge(match_dates, on="match_id", how="left")
    tk_by_team = tk_by_team.sort_values("date")

    tk_group = tk_by_team.groupby("team", observed=True)
    for window in [5, 10]:
        tk_by_team[f"team_tk_avg_{window}"] = _group_shifted_rolling_mean(
            tk_by_team, ["team"], "team_tk_total", window, min_periods=1
        )

    opp_tk = tk_by_team[
        ["match_id", "team", "team_tk_avg_5", "team_tk_avg_10"]
    ].drop_duplicates()

    df = df.merge(
        opp_tk.rename(columns={
            "team": "_opp_tk_team",
            "team_tk_avg_5": "opp_tackle_rate_avg_5",
            "team_tk_avg_10": "opp_tackle_rate_avg_10",
        }),
        left_on=["match_id", "opponent"],
        right_on=["match_id", "_opp_tk_team"],
        how="left",
    )
    df = df.drop(columns=["_opp_tk_team"], errors="ignore")
    for col in ["opp_tackle_rate_avg_5", "opp_tackle_rate_avg_10"]:
        mean_val = df[col].dropna().mean()
        df[col] = df[col].fillna(mean_val if pd.notna(mean_val) else 0)

    # For player vs opponent: fall back to player's overall career average
    if "career_goal_avg_pre" in df.columns:
        df["player_vs_opp_gl_avg"] = df["player_vs_opp_gl_avg"].fillna(
            df["career_goal_avg_pre"]
        )
    else:
        df["player_vs_opp_gl_avg"] = df["player_vs_opp_gl_avg"].fillna(0)
    df["player_vs_opp_gl_diff"] = df["player_vs_opp_gl_diff"].fillna(0)

    # For player vs opponent marks: fall back to player's overall marks avg
    if "player_vs_opp_mk_avg" in df.columns:
        mk_career_avg = _group_shifted_expanding_mean(
            df, ["player", "team"], "MK", min_periods=1
        )
        df["player_vs_opp_mk_avg"] = df["player_vs_opp_mk_avg"].fillna(mk_career_avg).fillna(0)

    return df


# ---------------------------------------------------------------------------
# E. Defender Matchup Features
# ---------------------------------------------------------------------------

def add_matchup_features(df):
    """Estimate forward-vs-defender interactions (vectorized, temporal).

    For each player-match row, computes cumulative stats to determine if
    a player is a key defender using only past data. Then aggregates
    defender count and strength per (match_id, team) and merges as
    opponent stats.
    """
    df = df.sort_values(["player", "team", "date"]).copy()

    grouped = df.groupby(["player", "team"], observed=True)

    # Cumulative stats using only past matches (shift + expanding)
    df["_cum_gl_avg"] = grouped["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    df["_cum_rb_avg"] = grouped["RB"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    df["_cum_one_pct_avg"] = grouped["one_pct"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    df["_cum_games"] = grouped["GL"].transform(
        lambda s: s.shift(1).expanding().count()
    )

    # Flag defender per-row based on cumulative stats
    df["_is_defender"] = (
        (df["_cum_gl_avg"] < 0.3)
        & (df["_cum_rb_avg"] >= config.KEY_DEFENDER_RB_THRESHOLD)
        & (df["_cum_one_pct_avg"] >= config.KEY_DEFENDER_ONE_PCT_THRESHOLD)
        & (df["_cum_games"] >= config.MIN_PLAYER_MATCHES)
    ).astype(int)

    df["_defender_strength"] = df["_is_defender"] * (
        df["_cum_one_pct_avg"].fillna(0) + df["_cum_rb_avg"].fillna(0)
    )

    # Aggregate per (match_id, team): count of defenders and total strength
    team_def = (
        df.groupby(["match_id", "team"], observed=True)
        .agg(
            _team_def_count=("_is_defender", "sum"),
            _team_def_strength=("_defender_strength", "sum"),
        )
        .reset_index()
    )

    # Merge as opponent stats: for each row, look up opponent team's defenders
    df = df.merge(
        team_def.rename(columns={
            "team": "opponent",
            "_team_def_count": "opp_key_defenders_count",
            "_team_def_strength": "opp_defender_strength_score",
        }),
        on=["match_id", "opponent"],
        how="left",
    )

    df["opp_key_defenders_count"] = df["opp_key_defenders_count"].fillna(0).astype(int)
    df["opp_defender_strength_score"] = df["opp_defender_strength_score"].fillna(0.0)

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_cum_") or c in ("_is_defender", "_defender_strength")]
    df = df.drop(columns=temp_cols)

    return df


# ---------------------------------------------------------------------------
# F. Team Context Features
# ---------------------------------------------------------------------------

def add_team_features(df):
    """Team-level form and player's role within the team attack."""
    df = df.sort_values(["date"]).copy()

    # Team goals per match
    team_match = (
        df.groupby(["match_id", "team"], observed=True)
        .agg(
            team_total_goals=("GL", "sum"),
            team_total_behinds=("BH", "sum"),
            team_total_di=("DI", "sum"),
            team_total_if=("IF", "sum"),
            team_total_cl=("CL", "sum"),
        )
        .reset_index()
    )
    match_dates = df[["match_id", "date"]].drop_duplicates()
    team_match = team_match.merge(match_dates, on="match_id", how="left")
    team_match = team_match.sort_values("date")

    # Rolling team goals
    tg = team_match.groupby("team", observed=True)
    for window in [5, 10]:
        team_match[f"team_goals_avg_{window}"] = tg["team_total_goals"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        team_match[f"team_disp_avg_{window}"] = tg["team_total_di"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    team_match["team_goals_avg_pre"] = tg["team_total_goals"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    team_match["team_disp_avg_pre"] = tg["team_total_di"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    team_match["team_if_avg_5"] = tg["team_total_if"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Rolling team clearances (for clearance dominance)
    team_match["team_cl_avg_5"] = tg["team_total_cl"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Join back
    team_cols = [c for c in team_match.columns
                 if c not in ["date", "team_total_goals",
                              "team_total_behinds", "team_total_di", "team_total_if",
                              "team_total_cl"]]
    df = df.merge(
        team_match[team_cols + ["team_total_goals"]],
        on=["match_id", "team"],
        how="left",
    )

    # --- Team clearance dominance: team CL / opponent CL over last 5 ---
    # Get opponent's clearance average for the same match
    opp_cl = team_match[["match_id", "team", "team_cl_avg_5"]].rename(
        columns={"team": "opp_team", "team_cl_avg_5": "opp_cl_avg_5"}
    )
    df = df.merge(opp_cl, left_on=["match_id", "opponent"],
                  right_on=["match_id", "opp_team"], how="left")
    df = df.drop(columns=["opp_team"], errors="ignore")
    num = df["team_cl_avg_5"].fillna(0)
    den = df["opp_cl_avg_5"].fillna(0)
    df["team_clearance_dominance_5"] = np.where(den > 0, num / den, 1.0)
    df = df.drop(columns=["opp_cl_avg_5"], errors="ignore")

    # --- Midfielder quality score: aggregate DI+CL+IF rates of midfield teammates ---
    # Identify midfielders by cumulative stats (DI >= threshold and CL >= threshold)
    _pg = df.sort_values(["player", "team", "date"]).copy()
    _pg_grp = _pg.groupby(["player", "team"], observed=True)
    _pg["_cum_di_avg"] = _pg_grp["DI"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    _pg["_cum_cl_avg"] = _pg_grp["CL"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    _pg["_cum_if_avg"] = _pg_grp["IF"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    _pg["_is_mid"] = (
        (_pg["_cum_di_avg"] >= config.MIDFIELDER_DI_THRESHOLD)
        & (_pg["_cum_cl_avg"] >= config.MIDFIELDER_CL_THRESHOLD)
    ).astype(int)
    _pg["_mid_quality"] = (
        _pg["_cum_di_avg"].fillna(0)
        + _pg["_cum_cl_avg"].fillna(0) * 2
        + _pg["_cum_if_avg"].fillna(0)
    ) * _pg["_is_mid"]

    # Aggregate per (match_id, team): sum of midfielder quality scores
    mid_agg = (
        _pg.groupby(["match_id", "team"], observed=True)
        .agg(
            _mid_count=("_is_mid", "sum"),
            _mid_quality_sum=("_mid_quality", "sum"),
        )
        .reset_index()
    )
    mid_agg["team_mid_quality_score"] = np.where(
        mid_agg["_mid_count"] > 0,
        mid_agg["_mid_quality_sum"] / mid_agg["_mid_count"],
        0.0,
    )
    df = df.merge(
        mid_agg[["match_id", "team", "team_mid_quality_score"]],
        on=["match_id", "team"], how="left",
    )

    # Player goal share: player's goals / team's goals (last 5 matches)
    player_group = df.groupby(["player", "team"], observed=True)
    player_gl_5 = player_group["GL"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    team_gl_5_per_player = df.groupby(["player", "team"], observed=True)["team_total_goals"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    df["player_goal_share_5"] = np.where(
        team_gl_5_per_player > 0,
        player_gl_5 / team_gl_5_per_player,
        0,
    )

    # --- Phase 1B: Player disposal share ---
    # Need team total DI per match for the share calculation
    team_match_di_totals = (
        df.groupby(["match_id", "team"], observed=True)["DI"]
        .sum()
        .reset_index()
        .rename(columns={"DI": "_team_total_di"})
    )
    df = df.merge(team_match_di_totals, on=["match_id", "team"], how="left")

    for window in [5, 10]:
        player_di_w = player_group["DI"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        team_di_w = df.groupby(["player", "team"], observed=True)["_team_total_di"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        df[f"player_disp_share_{window}"] = np.where(
            team_di_w > 0, player_di_w / team_di_w, 0,
        )

    df = df.drop(columns=["_team_total_di"], errors="ignore")

    # --- Phase 1C: Player marks share ---
    team_match_mk_totals = (
        df.groupby(["match_id", "team"], observed=True)["MK"]
        .sum()
        .reset_index()
        .rename(columns={"MK": "_team_total_mk"})
    )
    df = df.merge(team_match_mk_totals, on=["match_id", "team"], how="left")

    for window in [5, 10]:
        player_mk_w = player_group["MK"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        team_mk_w = df.groupby(["player", "team"], observed=True)["_team_total_mk"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        df[f"player_mk_share_{window}"] = np.where(
            team_mk_w > 0, player_mk_w / team_mk_w, 0,
        )

    df = df.drop(columns=["_team_total_mk"], errors="ignore")

    # Team win streak (approximate: did team score more than opponent last N?)
    # We need match results. Compute from team_total_goals vs opponent goals.
    opp_match = (
        df.groupby(["match_id", "team"], observed=True)["GL"].sum().reset_index()
        .rename(columns={"GL": "team_gl", "team": "this_team"})
    )
    # Get opponent's goals for same match
    match_teams = df[["match_id", "team", "opponent"]].drop_duplicates()
    opp_gl = opp_match.rename(columns={"this_team": "opp_team", "team_gl": "opp_gl"})
    match_results = match_teams.merge(
        opp_match, left_on=["match_id", "team"], right_on=["match_id", "this_team"],
        how="left",
    )
    match_results = match_results.merge(
        opp_gl, left_on=["match_id", "opponent"], right_on=["match_id", "opp_team"],
        how="left",
    )
    match_results["won"] = (match_results["team_gl"] > match_results["opp_gl"]).astype(int)

    # Merge dates and compute rolling
    match_results = match_results.merge(match_dates, on="match_id", how="left")
    match_results = match_results.sort_values("date")
    match_results["team_win_pct_5"] = (
        match_results.groupby("team", observed=True)["won"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    match_results["team_margin_avg_5"] = (
        match_results.groupby("team", observed=True)
        .apply(lambda g: (g["team_gl"] - g["opp_gl"]).shift(1).rolling(5, min_periods=1).mean(),
               include_groups=False)
        .reset_index(level=0, drop=True)
    )

    win_cols = match_results[
        ["match_id", "team", "team_win_pct_5", "team_margin_avg_5"]
    ].drop_duplicates()

    df = df.merge(win_cols, on=["match_id", "team"], how="left")

    # Fill NaN
    for col in ["team_goals_avg_5", "team_goals_avg_10", "team_goals_avg_pre",
                "team_disp_avg_5", "team_disp_avg_10", "team_disp_avg_pre", "team_if_avg_5",
                "team_cl_avg_5", "team_clearance_dominance_5",
                "team_mid_quality_score",
                "player_goal_share_5", "player_disp_share_5", "player_disp_share_10",
                "player_mk_share_5", "player_mk_share_10",
                "team_win_pct_5", "team_margin_avg_5"]:
        if col in df.columns:
            fill_val = 1.0 if col == "team_clearance_dominance_5" else 0
            df[col] = df[col].fillna(fill_val)

    # Drop helper column
    df = df.drop(columns=["team_total_goals"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# G. Scoring Pattern Features
# ---------------------------------------------------------------------------

def add_scoring_pattern_features(df):
    """Features from quarter-level scoring data (q1_goals, q2_goals, etc.)."""
    df = df.copy()

    # Quarter goal columns should exist from clean.py join
    q_goal_cols = [f"q{q}_goals" for q in range(1, 5)]
    q_behind_cols = [f"q{q}_behinds" for q in range(1, 5)]

    # Ensure they exist
    for col in q_goal_cols + q_behind_cols:
        if col not in df.columns:
            df[col] = 0

    # Total goals from scoring data (should match GL but computed independently)
    df["scoring_total_goals"] = sum(df[c] for c in q_goal_cols)

    # Cumulative quarter distribution (expanding over player history)
    grouped = df.groupby(["player", "team"], observed=True)

    for q in range(1, 5):
        col = f"q{q}_goals"
        total_qn = grouped[col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).sum()
        )
        total_all = grouped["GL"].transform(
            lambda s: s.shift(1).expanding(min_periods=1).sum()
        )
        df[f"player_q{q}_gl_pct"] = np.where(total_all > 0, total_qn / total_all, 0.25)

    # Late scorer: Q3+Q4 fraction
    df["player_late_scorer_pct"] = df["player_q3_gl_pct"] + df["player_q4_gl_pct"]

    # Multi-goal game rate
    df["_prev_gl_ge2"] = (df["GL"] >= 2).astype(int)
    df["player_multi_goal_rate"] = grouped["_prev_gl_ge2"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df = df.drop(columns=["_prev_gl_ge2", "scoring_total_goals"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# H. Role Classification
# ---------------------------------------------------------------------------

def classify_roles(df):
    """Classify each player's role from their recent stat averages.

    Uses last 10 matches to handle role changes mid-career.
    Simplified to 3 core roles: forward, midfielder, other.
    This avoids the sparse 6-role one-hot encoding where everyone
    got classified as "general" due to high thresholds.
    """
    df = df.sort_values(["player", "team", "date"]).copy()
    grouped = df.groupby(["player", "team"], observed=True)

    # Compute recent averages for classification
    for col in ["GL", "HO", "DI", "CL", "RB", "one_pct", "IF", "MI"]:
        df[f"_recent_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )

    # Simplified 3-role classification
    conditions = [
        # Forward: either key forward (goals + MI) or small forward (goals + IF)
        (df["_recent_GL"] >= config.KEY_FORWARD_GL_THRESHOLD) |
        ((df["_recent_GL"] >= config.SMALL_FORWARD_GL_THRESHOLD) &
         (df["_recent_IF"] >= config.SMALL_FORWARD_IF_THRESHOLD)),
        # Midfielder: high disposals + clearances
        (df["_recent_DI"] >= config.MIDFIELDER_DI_THRESHOLD) &
        (df["_recent_CL"] >= config.MIDFIELDER_CL_THRESHOLD),
    ]
    choices = ["forward", "midfielder"]
    df["player_role"] = np.select(conditions, choices, default="other")

    # One-hot encode (just 3 roles: forward, midfielder, other)
    for role in choices + ["other"]:
        df[f"role_{role}"] = (df["player_role"] == role).astype(int)

    # Continuous "forward score" — more informative than binary
    df["forward_score"] = (
        df["_recent_GL"].fillna(0) * 2
        + df["_recent_MI"].fillna(0)
        + df["_recent_IF"].fillna(0) * 0.5
    )

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_recent_")]
    df = df.drop(columns=temp_cols)

    return df


# ---------------------------------------------------------------------------
# I. Teammate Enabler Features
# ---------------------------------------------------------------------------

def add_enabler_features(df):
    """Proxy for teammate enabler effects (vectorized, temporal).

    Instead of identifying specific enablers (which requires future data),
    we compute each player's cumulative GL average and aggregate per
    (match_id, team) to get teammate scoring context. Then subtract
    self-contribution so each player gets a measure of how strong their
    teammates are as scorers.
    """
    df = df.sort_values(["player", "team", "date"]).copy()

    # Each player's cumulative GL average (temporal, excludes current match)
    grouped = df.groupby(["player", "team"], observed=True)
    df["_player_cum_gl_avg"] = grouped["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).fillna(0)

    # Flag: is this player a high-scoring teammate? (cum GL avg >= 0.5)
    df["_is_high_scorer"] = (df["_player_cum_gl_avg"] >= 0.5).astype(int)

    # Aggregate per (match_id, team): team-level scoring summary
    team_scoring = (
        df.groupby(["match_id", "team"], observed=True)
        .agg(
            _team_high_scorers=("_is_high_scorer", "sum"),
            _team_scoring_sum=("_player_cum_gl_avg", "sum"),
            _team_player_count=("_player_cum_gl_avg", "count"),
        )
        .reset_index()
    )

    df = df.merge(team_scoring, on=["match_id", "team"], how="left")

    # Subtract self-contribution to get teammate stats
    df["teammate_enabler_count"] = (
        df["_team_high_scorers"] - df["_is_high_scorer"]
    ).clip(lower=0).astype(int)

    df["teammate_scoring_avg"] = np.where(
        df["_team_player_count"] > 1,
        (df["_team_scoring_sum"] - df["_player_cum_gl_avg"]) / (df["_team_player_count"] - 1),
        0.0,
    )

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=temp_cols)

    return df


# ---------------------------------------------------------------------------
# J. Interaction Features
# ---------------------------------------------------------------------------

def add_interaction_features(df):
    """Cross-feature interactions that capture matchup-specific effects."""
    df = df.copy()

    # Player scoring ability * opponent defensive weakness
    if "career_goal_avg_capped" in df.columns and "opp_goals_conceded_avg_5" in df.columns:
        df["interact_player_vs_opp_defense"] = (
            df["career_goal_avg_capped"].fillna(0) * df["opp_goals_conceded_avg_5"].fillna(0)
        )

    # Rolling form * opponent conceded (recent form against leaky defense)
    if "player_gl_ewm_5" in df.columns and "opp_goals_conceded_avg_5" in df.columns:
        df["interact_form_vs_defense"] = (
            df["player_gl_ewm_5"].fillna(0) * df["opp_goals_conceded_avg_5"].fillna(0)
        )

    # Home advantage interaction: is_home * career_goal_avg_capped
    if "is_home" in df.columns and "career_goal_avg_capped" in df.columns:
        df["interact_home_scoring"] = (
            df["is_home"] * df["career_goal_avg_capped"].fillna(0)
        )

    # Venue familiarity boost: games at venue * venue scoring diff
    if "player_gl_venue_diff" in df.columns:
        df["interact_venue_boost"] = (
            df["player_gl_venue_diff"].fillna(0)
            * df.get("player_vs_opp_games", pd.Series(0, index=df.index)).fillna(0).clip(upper=5)
        )

    # Team form * player share (player on hot team AND high share = more goals)
    if "team_win_pct_5" in df.columns and "player_goal_share_5" in df.columns:
        df["interact_team_form_share"] = (
            df["team_win_pct_5"].fillna(0) * df["player_goal_share_5"].fillna(0)
        )

    # Hot player vs weak defense
    if "player_is_hot" in df.columns and "opp_goals_conceded_avg_5" in df.columns:
        df["interact_hot_vs_weak_defense"] = (
            df["player_is_hot"] * df["opp_goals_conceded_avg_5"].fillna(0)
        )

    # Weighted streak * forward score
    if "player_gl_streak_weighted" in df.columns and "forward_score" in df.columns:
        df["interact_streak_forward"] = (
            df["player_gl_streak_weighted"].fillna(0) * df["forward_score"].fillna(0)
        )

    # --- Disposal-specific interaction features ---

    # Disposal form × opposition contested possession rate
    if "player_di_ewm_5" in df.columns and "opp_disp_conceded_avg_5" in df.columns:
        df["interact_disp_vs_contested"] = (
            df["player_di_ewm_5"].fillna(0) * df["opp_disp_conceded_avg_5"].fillna(0)
        )

    # Disposal form × expected game pace (team DI averages proxy)
    if "player_di_ewm_5" in df.columns and "team_goals_avg_5" in df.columns:
        df["interact_disp_pace"] = (
            df["player_di_ewm_5"].fillna(0) * df["team_goals_avg_5"].fillna(0)
        )

    # Disposal form × opponent contested possession dominance
    # Higher opp CP diff = opponent wins more contested ball → suppresses disposals
    if "player_di_ewm_5" in df.columns and "opp_contested_poss_diff_5" in df.columns:
        df["interact_disp_vs_cp_diff"] = (
            df["player_di_ewm_5"].fillna(0) * df["opp_contested_poss_diff_5"].fillna(0)
        )

    # Midfielder quality × player scoring: better supply → more goals for forwards
    if "team_mid_quality_score" in df.columns and "forward_score" in df.columns:
        df["interact_mid_supply_forward"] = (
            df["team_mid_quality_score"].fillna(0) * df["forward_score"].fillna(0)
        )

    # --- Phase 1B: Disposal share × team form ---
    if "player_disp_share_5" in df.columns and "team_win_pct_5" in df.columns:
        df["interact_disp_share_team_form"] = (
            df["player_disp_share_5"].fillna(0) * df["team_win_pct_5"].fillna(0)
        )

    # --- Phase 1A: TOG × disposal form ---
    if "player_tog_avg_5" in df.columns and "player_di_ewm_5" in df.columns:
        df["interact_tog_disp_form"] = (
            df["player_tog_avg_5"].fillna(75) * df["player_di_ewm_5"].fillna(0)
        )

    return df


# ---------------------------------------------------------------------------
# K-a. GMM Player Archetypes
# ---------------------------------------------------------------------------

def build_archetypes(df, n_archetypes=None):
    """Cluster players into archetypes using GMM on rate-normalised stat profiles.

    Uses expanding (temporal) averages of rate columns so each player's
    archetype is based only on past data.

    Returns:
      - df with archetype columns added:
        'archetype' (int 0..K-1), 'archetype_prob_0'..'archetype_prob_{K-1}'
      - gmm: fitted GaussianMixture model
      - archetype_names: dict mapping archetype int → descriptive name
    """
    from sklearn.mixture import GaussianMixture

    n_archetypes = n_archetypes or config.N_ARCHETYPES

    df = df.sort_values(["player", "team", "date"]).copy()
    grouped = df.groupby(["player", "team"], observed=True)

    # Build cumulative rate profile per player (expanding average of rate stats)
    rate_features = ["GL_rate", "BH_rate", "DI_rate", "MK_rate", "TK_rate",
                     "IF_rate", "CP_rate", "HO_rate", "RB_rate", "CL_rate"]
    available_rates = [c for c in rate_features if c in df.columns]

    for col in available_rates:
        df[f"_cum_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).expanding(min_periods=config.MIN_PLAYER_MATCHES).mean()
        )

    cum_cols = [f"_cum_{c}" for c in available_rates]

    # Fit GMM on rows that have enough history (non-NaN cumulative stats)
    fit_mask = df[cum_cols].notna().all(axis=1)
    fit_data = df.loc[fit_mask, cum_cols].values

    if len(fit_data) < n_archetypes * 10:
        print(f"  WARNING: Insufficient data for GMM ({len(fit_data)} rows). Skipping archetypes.")
        df["archetype"] = 0
        for k in range(n_archetypes):
            df[f"archetype_prob_{k}"] = 1.0 / n_archetypes if k == 0 else 0.0
        # Drop temp columns
        df = df.drop(columns=cum_cols, errors="ignore")
        return df, None, {}

    # Standardize for GMM
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    fit_data_scaled = scaler.fit_transform(fit_data)

    gmm = GaussianMixture(
        n_components=n_archetypes,
        covariance_type="full",
        n_init=3,
        random_state=config.RANDOM_SEED,
        max_iter=200,
    )
    gmm.fit(fit_data_scaled)

    # Predict archetypes for all rows (NaN rows get archetype=0, uniform probs)
    df["archetype"] = 0
    for k in range(n_archetypes):
        df[f"archetype_prob_{k}"] = 1.0 / n_archetypes

    all_data = df[cum_cols].values
    valid_mask = ~np.isnan(all_data).any(axis=1)
    if valid_mask.sum() > 0:
        valid_scaled = scaler.transform(all_data[valid_mask])
        labels = gmm.predict(valid_scaled)
        probs = gmm.predict_proba(valid_scaled)
        df.loc[valid_mask, "archetype"] = labels
        for k in range(n_archetypes):
            df.loc[valid_mask, f"archetype_prob_{k}"] = probs[:, k]

    # Name archetypes by their dominant stat profile
    archetype_names = _name_archetypes(gmm, scaler, available_rates, n_archetypes)
    print(f"  GMM archetypes: {archetype_names}")

    # Drop temp columns
    df = df.drop(columns=cum_cols, errors="ignore")

    return df, gmm, archetype_names


def _name_archetypes(gmm, scaler, rate_names, n_archetypes):
    """Auto-name archetypes based on which stats are highest in each cluster center."""
    centers = scaler.inverse_transform(gmm.means_)
    # Short stat labels
    stat_labels = {
        "GL_rate": "Goal", "BH_rate": "Behind", "DI_rate": "Disp",
        "MK_rate": "Mark", "TK_rate": "Tackle", "IF_rate": "IF50",
        "CP_rate": "Contest", "HO_rate": "Ruck", "RB_rate": "Rebound",
        "CL_rate": "Clear",
    }
    role_templates = {
        "GL_rate": "Forward", "HO_rate": "Ruck", "DI_rate": "Midfielder",
        "RB_rate": "Defender", "TK_rate": "Tagger", "MK_rate": "Tall",
    }

    names = {}
    for k in range(n_archetypes):
        center = centers[k]
        # Find the stat with highest z-score relative to the overall mean
        z_scores = (center - centers.mean(axis=0)) / (centers.std(axis=0) + 1e-9)
        top_idx = np.argmax(z_scores)
        top_stat = rate_names[top_idx]
        name = role_templates.get(top_stat, stat_labels.get(top_stat, f"Type{k}"))
        # Ensure unique names
        if name in names.values():
            name = f"{name}_{k}"
        names[k] = name
    return names


# ---------------------------------------------------------------------------
# K-b. Opponent Concession Profiles by Archetype
# ---------------------------------------------------------------------------

def add_archetype_concession_features(df):
    """Compute opponent concession profiles broken down by player archetype.

    For each (opponent, archetype) pair, compute rolling average of goals
    and disposals conceded to players of that archetype. Then merge back
    as features for each player based on their archetype assignment.
    """
    if "archetype" not in df.columns:
        return df

    df = df.sort_values("date").copy()

    # Build per-match per-team per-archetype aggregates
    arch_match = (
        df.groupby(["match_id", "team", "opponent", "archetype"], observed=True)
        .agg(
            arch_goals=("GL", "sum"),
            arch_disp=("DI", "sum"),
            arch_count=("player", "count"),
        )
        .reset_index()
    )

    # These goals were scored *by* team *against* opponent
    # Relabel as: opponent conceded these to archetype
    conceded = arch_match.rename(columns={
        "team": "scoring_team",
        "opponent": "conceding_team",
    })

    match_dates = df[["match_id", "date"]].drop_duplicates()
    conceded = conceded.merge(match_dates, on="match_id", how="left")
    conceded = conceded.sort_values("date")

    # Per-archetype goals conceded per match by the conceding team
    conceded["goals_per_arch_player"] = np.where(
        conceded["arch_count"] > 0,
        conceded["arch_goals"] / conceded["arch_count"],
        0,
    )
    conceded["disp_per_arch_player"] = np.where(
        conceded["arch_count"] > 0,
        conceded["arch_disp"] / conceded["arch_count"],
        0,
    )

    # Rolling averages per (conceding_team, archetype)
    cg = conceded.groupby(["conceding_team", "archetype"], observed=True)
    conceded["opp_arch_gl_conceded_avg_5"] = cg["goals_per_arch_player"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    conceded["opp_arch_disp_conceded_avg_5"] = cg["disp_per_arch_player"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Merge back to main df: for each row, look up opponent's concession
    # to player's archetype
    merge_cols = conceded[[
        "match_id", "conceding_team", "archetype",
        "opp_arch_gl_conceded_avg_5", "opp_arch_disp_conceded_avg_5",
    ]].drop_duplicates(subset=["match_id", "conceding_team", "archetype"])

    df = df.merge(
        merge_cols,
        left_on=["match_id", "opponent", "archetype"],
        right_on=["match_id", "conceding_team", "archetype"],
        how="left",
    )
    df = df.drop(columns=["conceding_team"], errors="ignore")

    # Fill NaN with league-wide averages
    for col in ["opp_arch_gl_conceded_avg_5", "opp_arch_disp_conceded_avg_5"]:
        mean_val = df[col].dropna().mean()
        df[col] = df[col].fillna(mean_val if pd.notna(mean_val) else 0)

    return df


# ---------------------------------------------------------------------------
# K-b2. Archetype Disposal Ceiling Features
# ---------------------------------------------------------------------------

def add_archetype_disposal_features(df):
    """Archetype-relative disposal context: ceiling and percentile.

    For each archetype, computes the 90th-percentile disposal count from
    recent historical data. Then computes where each player's rolling
    disposal average sits relative to their archetype's distribution.
    """
    if "archetype" not in df.columns:
        return df

    df = df.sort_values("date").copy()

    # Compute per-archetype rolling disposal statistics (90th pctile, median)
    # Group by archetype across all players, rolling over the last ~200 rows
    # (approx 5 rounds × ~40 players/archetype)
    arch_group = df.groupby("archetype", observed=True)

    df["_arch_di_p90"] = arch_group["DI"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=20).quantile(0.9)
    )
    df["_arch_di_median"] = arch_group["DI"].transform(
        lambda s: s.shift(1).rolling(200, min_periods=20).median()
    )

    # Player's recent disposal avg relative to archetype ceiling
    if "player_di_avg_5" in df.columns:
        df["archetype_di_ceiling_ratio"] = np.where(
            df["_arch_di_p90"].fillna(0) > 0,
            df["player_di_avg_5"].fillna(0) / df["_arch_di_p90"],
            0.0,
        )
    else:
        df["archetype_di_ceiling_ratio"] = 0.0

    # Archetype disposal ceiling as a standalone feature
    df["archetype_di_ceiling_5"] = df["_arch_di_p90"].fillna(0)

    # Clean up
    df = df.drop(columns=["_arch_di_p90", "_arch_di_median"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# K-c. Game Environment Features
# ---------------------------------------------------------------------------

# Indoor/roofed venues — weather-independent scoring conditions
INDOOR_VENUES = {
    "Docklands", "Marvel Stadium", "Etihad Stadium",  # same venue, renamed
    "Adelaide Arena at Jiangwan Stadium",  # exhibition
}


def add_game_environment_features(df):
    """Game-level features that affect scoring conditions."""
    df = df.copy()

    # Indoor/outdoor venue flag
    df["venue_is_indoor"] = df["venue"].astype(str).isin(INDOOR_VENUES).astype(int)

    # Game pace proxy: average team disposals from both teams' recent form
    if "team_goals_avg_5" in df.columns:
        # Build team DI averages per match (proxy for pace)
        team_match_di = (
            df.groupby(["match_id", "team"], observed=True)["DI"]
            .sum()
            .reset_index()
            .rename(columns={"DI": "team_di_total"})
        )
        match_dates = df[["match_id", "date"]].drop_duplicates()
        team_match_di = team_match_di.merge(match_dates, on="match_id", how="left")
        team_match_di = team_match_di.sort_values("date")

        tg = team_match_di.groupby("team", observed=True)
        team_match_di["team_di_avg_5"] = tg["team_di_total"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

        # Merge own team DI average
        di_cols = team_match_di[["match_id", "team", "team_di_avg_5"]].drop_duplicates()
        df = df.merge(di_cols, on=["match_id", "team"], how="left")

        # Merge opponent DI average for pace proxy
        opp_di = di_cols.rename(columns={
            "team": "opponent", "team_di_avg_5": "opp_di_avg_5"
        })
        df = df.merge(opp_di, on=["match_id", "opponent"], how="left")

        # Game pace = (team DI + opp DI) / 2
        df["game_pace_proxy"] = (
            (df["team_di_avg_5"].fillna(0) + df["opp_di_avg_5"].fillna(0)) / 2
        )

        # Clean up temp columns
        df = df.drop(columns=["team_di_avg_5", "opp_di_avg_5"], errors="ignore")

    # --- Expected game closeness / blowout risk ---
    # Uses team margin averages to estimate expected competitiveness
    # In blowout games, winning-team players get ~20% more disposals (garbage time)
    if "team_margin_avg_5" in df.columns:
        # Get opponent's margin average for this match
        opp_margin = (
            df[["match_id", "team", "team_margin_avg_5"]]
            .drop_duplicates(subset=["match_id", "team"])
            .rename(columns={"team": "opp_team_m", "team_margin_avg_5": "opp_margin_avg_5"})
        )
        df = df.merge(
            opp_margin, left_on=["match_id", "opponent"],
            right_on=["match_id", "opp_team_m"], how="left",
        )
        df = df.drop(columns=["opp_team_m"], errors="ignore")

        # Expected margin differential: team margin - opponent margin
        # Large positive = this team expected to dominate
        # Large absolute value = expected blowout
        df["expected_margin_diff"] = (
            df["team_margin_avg_5"].fillna(0) - df["opp_margin_avg_5"].fillna(0)
        )
        df["expected_margin_abs"] = df["expected_margin_diff"].abs()
        # Game closeness: 1 for expected even contest, → 0 for blowout
        df["expected_game_closeness"] = 1.0 / (1.0 + df["expected_margin_abs"])

        df = df.drop(columns=["opp_margin_avg_5"], errors="ignore")

    # --- Phase 1E: Rest days from team_matches ---
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if tm_path.exists():
        tm = pd.read_parquet(tm_path, columns=["match_id", "team", "rest_days"])
        tm = tm.dropna(subset=["rest_days"])
        tm["rest_days"] = pd.to_numeric(tm["rest_days"], errors="coerce")

        # Team rest days
        df = df.merge(
            tm.rename(columns={"rest_days": "team_rest_days"}),
            on=["match_id", "team"], how="left",
        )
        # Opponent rest days
        df = df.merge(
            tm.rename(columns={"team": "_opp_rd", "rest_days": "opp_rest_days"}),
            left_on=["match_id", "opponent"],
            right_on=["match_id", "_opp_rd"], how="left",
        )
        df = df.drop(columns=["_opp_rd"], errors="ignore")

        df["rest_day_differential"] = (
            df["team_rest_days"].fillna(7) - df["opp_rest_days"].fillna(7)
        )
        df["team_rest_days"] = df["team_rest_days"].fillna(7)
        df["opp_rest_days"] = df["opp_rest_days"].fillna(7)

    # Fill NaN
    for col in ["venue_is_indoor", "game_pace_proxy",
                "expected_margin_diff", "expected_margin_abs",
                "team_rest_days", "opp_rest_days", "rest_day_differential"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# K-c2. Ground Dimension Features
# ---------------------------------------------------------------------------

def add_ground_dimension_features(df):
    """Join static ground dimensions by venue name.

    Loads data/venue_dimensions.json (length_m, width_m per venue) and
    computes ground_area and ground_shape_ratio.  Falls back to median
    dimensions if a venue is missing from the lookup.
    """
    import json as _json

    dims_path = Path(config.DATA_DIR) / "venue_dimensions.json"
    if not dims_path.exists():
        print("    WARNING: venue_dimensions.json not found — skipping ground features")
        return df

    with open(dims_path) as f:
        dims = _json.load(f)

    # Build DataFrame from lookup
    rows = []
    for venue, d in dims.items():
        length = d["length_m"]
        width = d["width_m"]
        rows.append({
            "venue": venue,
            "ground_length": length,
            "ground_width": width,
            "ground_area": length * width,
            "ground_shape_ratio": length / width,
        })
    dims_df = pd.DataFrame(rows)

    # Median fallback for any venue not in the lookup
    median_length = dims_df["ground_length"].median()
    median_width = dims_df["ground_width"].median()

    df = df.merge(dims_df, on="venue", how="left")

    missing = df["ground_length"].isna().sum()
    if missing > 0:
        unmapped = df.loc[df["ground_length"].isna(), "venue"].unique()
        print(f"    WARNING: {missing} rows ({len(unmapped)} venues) missing ground dims, "
              f"filling with median: {unmapped.tolist()}")
        df["ground_length"] = df["ground_length"].fillna(median_length)
        df["ground_width"] = df["ground_width"].fillna(median_width)
        df["ground_area"] = df["ground_area"].fillna(median_length * median_width)
        df["ground_shape_ratio"] = df["ground_shape_ratio"].fillna(median_length / median_width)

    # Cast to float32
    for col in ["ground_length", "ground_width", "ground_area", "ground_shape_ratio"]:
        df[col] = df[col].astype(np.float32)

    print(f"    Joined ground dimensions for {len(dims)} venues "
          f"(area range: {dims_df['ground_area'].min():.0f}–{dims_df['ground_area'].max():.0f} m²)")
    return df


# ---------------------------------------------------------------------------
# K-c3. Day / Night Classification
# ---------------------------------------------------------------------------

def add_day_night_features(df):
    """Classify games as day, twilight, or night based on local kick-off time.

    The timestamps in our data are already local to the venue (verified by
    checking Perth Friday night games: 18:10 = 6:10pm AWST, not AEST).

    Classification (local time):
      - night:    kick-off >= 17:00
      - twilight: kick-off >= 16:00 and < 17:00
      - day:      kick-off < 16:00
    """
    df = df.copy()

    local_hour = df["date"].dt.hour
    local_minute = df["date"].dt.minute
    # Use fractional hour for precise boundary (e.g. 16:30 = 16.5)
    local_time_frac = local_hour + local_minute / 60.0

    df["is_night_game"] = (local_time_frac >= 17.0).astype(np.int8)
    df["is_twilight_game"] = ((local_time_frac >= 16.0) & (local_time_frac < 17.0)).astype(np.int8)

    n_night = df["is_night_game"].sum()
    n_twilight = df["is_twilight_game"].sum()
    n_day = len(df) - n_night - n_twilight
    # Count unique matches for summary
    match_counts = df.groupby("match_id", observed=True)[["is_night_game", "is_twilight_game"]].first()
    print(f"    Day/night: {(match_counts['is_night_game']==0).sum() - (match_counts['is_twilight_game']==1).sum()} day, "
          f"{(match_counts['is_twilight_game']==1).sum()} twilight, "
          f"{(match_counts['is_night_game']==1).sum()} night "
          f"(of {len(match_counts)} matches)")

    return df


# ---------------------------------------------------------------------------
# K-c4. Market / Odds Features
# ---------------------------------------------------------------------------

def add_market_features(df):
    """Add market/odds features from odds.parquet as first-class features.

    Match-level features (from bookmaker/Betfair odds):
      - market_home_implied_prob, market_away_implied_prob
      - market_handicap, market_total_score
      - market_confidence, odds_movement_home, odds_movement_line
      - betfair_home_implied_prob

    Player-context derived columns vary by config.MARKET_PLAYER_FEATURE_CONFIG:
      - "full": full chain (environment + player implied goals/disposals)
      - "env_only": keep only market_expected_match_goals + market_expected_team_goals
      - "player_goal_only": keep only market_implied_player_goals from the chain
      - "v31_legacy": revert to v3.1 behavior (market_expected_team_goals = total*team_prob/6)

    Handles missing odds.parquet gracefully (skip with warning).
    """
    if not getattr(config, "MARKET_FEATURES_ENABLED", True):
        print("    Market features disabled in config — skipping")
        return df

    odds_path = getattr(config, "ODDS_PARQUET_PATH", config.BASE_STORE_DIR / "odds.parquet")
    if not odds_path.exists():
        print(f"    Warning: {odds_path} not found — skipping market features")
        return df

    df = df.copy()
    odds = pd.read_parquet(odds_path)
    mode = str(getattr(config, "MARKET_PLAYER_FEATURE_CONFIG", "full")).strip().lower()
    valid_modes = {"full", "env_only", "player_goal_only", "v31_legacy"}
    if mode not in valid_modes:
        print(f"    Warning: unknown MARKET_PLAYER_FEATURE_CONFIG='{mode}', using 'full'")
        mode = "full"
    print(f"    Market feature mode: {mode}")

    odds_cols = [c for c in odds.columns if c != "match_id"]

    # Drop existing odds columns for idempotent re-runs
    existing = [c for c in odds_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    # Also drop derived columns we'll recreate
    for c in [
        "market_team_implied_prob",
        "market_team_scoring_share",
        "market_expected_match_goals",
        "market_expected_team_goals",
        "market_player_goal_share_pre",
        "market_implied_player_goals",
        "market_expected_match_disposals",
        "market_expected_team_disposals",
        "market_player_disposal_share_pre",
        "market_implied_player_disposals",
        "market_is_favourite",
    ]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Merge on match_id
    df = df.merge(odds, on="match_id", how="left")

    # Derive player-context columns
    is_home = df["is_home"].astype(bool) if "is_home" in df.columns else pd.Series(True, index=df.index)
    home_prob = pd.to_numeric(df.get("market_home_implied_prob", np.nan), errors="coerce")
    away_prob = pd.to_numeric(df.get("market_away_implied_prob", np.nan), errors="coerce")
    away_prob = away_prob.where(away_prob.notna(), 1.0 - home_prob)
    team_prob = np.where(is_home, home_prob, away_prob).astype(np.float32)
    df["market_team_implied_prob"] = team_prob

    total = pd.to_numeric(df.get("market_total_score", np.nan), errors="coerce")

    if mode == "v31_legacy":
        # v3.1 behavior: expected team goals directly from implied team win probability
        df["market_expected_team_goals"] = (
            total * pd.Series(team_prob, index=df.index) / 6.0
        ).astype(np.float32)
    else:
        # Split market total by normalized implied probabilities
        prob_sum = (home_prob.fillna(0) + away_prob.fillna(0)).astype(float)
        valid_prob_sum = prob_sum > 0
        home_share = np.where(valid_prob_sum, home_prob / prob_sum, 0.5)
        away_share = np.where(valid_prob_sum, away_prob / prob_sum, 0.5)
        team_scoring_share = np.where(is_home, home_share, away_share).astype(np.float32)

        env = _estimate_market_environment_stats()
        pts_per_goal = float(env["points_per_goal_from_market_total"])
        pts_per_disp = float(env["points_per_disposal_from_market_total"])
        n_env = int(env["n_matches"])
        if n_env > 0:
            print(
                "    Market totals verified: "
                f"{n_env} matches, points/goal={pts_per_goal:.3f}, "
                f"goals/point={1.0/pts_per_goal:.4f}"
            )

        exp_match_goals = total / max(pts_per_goal, 1e-6)
        exp_team_goals = exp_match_goals * team_scoring_share

        if mode == "full":
            df["market_team_scoring_share"] = team_scoring_share

        if mode in {"full", "env_only"}:
            df["market_expected_match_goals"] = exp_match_goals.astype(np.float32)
            df["market_expected_team_goals"] = exp_team_goals.astype(np.float32)

        # Player historical goal share of team scoring
        team_goal_avg = pd.to_numeric(
            df.get("team_goals_avg_pre", df.get("team_goals_avg_10", np.nan)),
            errors="coerce",
        )
        player_goal_avg = pd.to_numeric(df.get("career_goal_avg_pre", np.nan), errors="coerce")
        raw_goal_share = np.where(team_goal_avg > 0, player_goal_avg / team_goal_avg, np.nan)
        fallback_goal_share = pd.to_numeric(df.get("player_goal_share_5", np.nan), errors="coerce")
        raw_goal_share = np.where(np.isfinite(raw_goal_share), raw_goal_share, fallback_goal_share)
        raw_goal_share = np.where(np.isfinite(raw_goal_share), raw_goal_share, 0.08)
        goal_share = np.clip(raw_goal_share, 0.0, 0.6).astype(np.float32)

        if mode == "full":
            df["market_player_goal_share_pre"] = goal_share
        if mode in {"full", "player_goal_only"}:
            df["market_implied_player_goals"] = (
                exp_team_goals * goal_share
            ).astype(np.float32)

        # Disposal market chain only in full mode
        if mode == "full":
            exp_match_disp = total / max(pts_per_disp, 1e-6)
            exp_team_disp = exp_match_disp * team_scoring_share
            df["market_expected_match_disposals"] = exp_match_disp.astype(np.float32)
            df["market_expected_team_disposals"] = exp_team_disp.astype(np.float32)

            team_disp_avg = pd.to_numeric(
                df.get("team_disp_avg_pre", df.get("team_disp_avg_10", np.nan)),
                errors="coerce",
            )
            player_disp_avg = pd.to_numeric(df.get("career_disp_avg_pre", np.nan), errors="coerce")
            raw_disp_share = np.where(team_disp_avg > 0, player_disp_avg / team_disp_avg, np.nan)
            fallback_player_di = pd.to_numeric(df.get("player_di_avg_10", np.nan), errors="coerce")
            fallback_team_di = pd.to_numeric(df.get("team_disp_avg_10", np.nan), errors="coerce")
            fallback_disp_share = np.where(
                fallback_team_di > 0, fallback_player_di / fallback_team_di, np.nan
            )
            raw_disp_share = np.where(np.isfinite(raw_disp_share), raw_disp_share, fallback_disp_share)
            raw_disp_share = np.where(np.isfinite(raw_disp_share), raw_disp_share, 0.06)
            disp_share = np.clip(raw_disp_share, 0.0, 0.25).astype(np.float32)

            df["market_player_disposal_share_pre"] = disp_share
            df["market_implied_player_disposals"] = (
                exp_team_disp * disp_share
            ).astype(np.float32)

    df["market_is_favourite"] = (df["market_team_implied_prob"] > 0.5).astype(np.int8)

    # Market-based game closeness: 1 for coin-flip, → 0 for heavy favourite
    handicap = pd.to_numeric(df.get("market_handicap", np.nan), errors="coerce").fillna(0)
    df["market_game_closeness"] = (1.0 / (1.0 + handicap.abs())).astype(np.float32)

    # Cast odds columns to float32
    for c in odds_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    n_with_odds = df["market_home_implied_prob"].notna().sum()
    n_total = len(df)
    print(f"    Market features: {n_with_odds}/{n_total} rows with odds data "
          f"({n_with_odds/n_total*100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# K-d. Season Era / Rule Regime Features
# ---------------------------------------------------------------------------

def add_era_features(df):
    """Add season-era one-hot, COVID flag, and quarter-length ratio as features.

    Eras encode major AFL rule changes that shift statistical baselines.
    One-hot encoding lets the model learn different intercepts per regime.
    quarter_length_ratio (0.8 for 2020, 1.0 otherwise) tells the model
    that 2020 counting stats are deflated without manually adjusting them.
    """
    df = df.copy()

    # Ensure base columns exist (normally set in clean.py)
    if "season_era" not in df.columns:
        df["season_era"] = df["year"].map(config.ERA_MAP).fillna(
            config.CURRENT_PREDICTION_ERA
        ).astype(np.int8)
    if "is_covid_season" not in df.columns:
        df["is_covid_season"] = (df["year"] == config.COVID_SEASON_YEAR).astype(np.int8)
    if "quarter_length_ratio" not in df.columns:
        df["quarter_length_ratio"] = np.where(
            df["year"] == config.COVID_SEASON_YEAR,
            config.COVID_QUARTER_LENGTH_RATIO, 1.0
        ).astype(np.float32)

    # One-hot encode era (era_1 through era_6)
    for era in sorted(set(config.ERA_MAP.values())):
        df[f"era_{era}"] = (df["season_era"] == era).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# L. Sample Weights
# ---------------------------------------------------------------------------

def add_dynamic_sample_weights(df, pred_year, pred_round):
    """Add sample_weight column with dynamic current-season boosting.

    Computes: weight = era_w × decay_w × current_season_boost × within_season_recency.

    Parameters
    ----------
    df : DataFrame with 'date', 'year', 'round_number' columns
    pred_year : int — the year being predicted
    pred_round : int — the round being predicted (training data is < this)
    """
    df = df.copy()
    now = df["date"].max()

    days_ago = (now - df["date"]).dt.days.fillna(9999)
    era_w = df["year"].apply(_era_weight)
    decay_w = days_ago.apply(_decay_weight)

    # Current-season boost: scales up with how many rounds of current-season data exist
    is_current = df["year"] == pred_year
    rounds_available = max(pred_round - 1, 0)
    boost_val = min(
        config.CURRENT_SEASON_BOOST_BASE + config.CURRENT_SEASON_BOOST_PER_ROUND * rounds_available,
        config.CURRENT_SEASON_BOOST_MAX,
    )
    season_boost = np.where(is_current, boost_val, 1.0)

    # Within-season recency: recent rounds within current season weighted higher
    half_life = config.WITHIN_SEASON_RECENCY_HALF_LIFE
    rounds_ago = pred_round - df["round_number"]
    within_recency = np.where(
        is_current & (rounds_ago >= 0),
        0.5 ** (rounds_ago / half_life),
        1.0,  # prior-season rows: no additional penalty (day-decay handles it)
    )

    df["sample_weight"] = era_w * decay_w * season_boost * within_recency
    # Normalize so mean weight ≈ 1
    mean_w = df["sample_weight"].mean()
    if mean_w > 0:
        df["sample_weight"] = df["sample_weight"] / mean_w

    return df


def add_sample_weights(df):
    """Add sample_weight column for model training.
    Combines era weight * time decay. Delegates to add_dynamic_sample_weights
    using the max year/round in the data."""
    max_year = int(df["year"].max())
    max_round = int(df.loc[df["year"] == max_year, "round_number"].max()) + 1
    return add_dynamic_sample_weights(df, max_year, max_round)


# ---------------------------------------------------------------------------
# L. Weather Features
# ---------------------------------------------------------------------------

def add_weather_features(df):
    """Join weather data to the player-match DataFrame.

    Loads weather.parquet (one row per match) and merges on match_id.
    Includes both raw weather metrics and derived features.
    """
    weather_path = config.BASE_STORE_DIR / "weather.parquet"
    if not weather_path.exists():
        print("    WARNING: weather.parquet not found — skipping weather features")
        return df

    weather = pd.read_parquet(weather_path)

    # Select columns to join — raw metrics + derived features
    # Exclude match_id (join key) from feature list
    weather_feature_cols = [
        # Raw metrics (continuous — let the model decide importance)
        "temperature_avg", "apparent_temperature_avg",
        "precipitation_total", "rain_total",
        "wind_speed_avg", "wind_speed_max", "wind_gusts_max",
        "humidity_avg", "dew_point_avg", "cloud_cover_avg",
        # Derived features
        "is_wet", "is_heavy_rain",
        "wind_severity", "temperature_category",
        "feels_like_delta", "humidity_discomfort",
        "temperature_range", "is_overcast",
        "weather_difficulty_score", "slippery_conditions",
        "wind_direction_variability",
        "is_roofed",
    ]

    # Only keep columns that exist
    available = [c for c in weather_feature_cols if c in weather.columns]
    merge_cols = ["match_id"] + available

    df = df.merge(weather[merge_cols], on="match_id", how="left")

    # Fill any unmatched matches (shouldn't happen, but safety)
    n_missing = df[available[0]].isna().sum() if available else 0
    if n_missing > 0:
        print(f"    WARNING: {n_missing} matches missing weather data — filling with defaults")
        for col in available:
            if df[col].dtype.kind == "f":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)

    print(f"    Joined {len(available)} weather features from {len(weather)} matches")
    return df


# ---------------------------------------------------------------------------
# N. Umpire Features (match-level)
# ---------------------------------------------------------------------------

def add_umpire_features(df):
    """Add umpire panel features: experience, scoring tendency, home bias, familiarity.

    Loads umpires.parquet and computes rolling umpire stats.
    All temporal features use shift(1) to prevent leakage.
    Gracefully skips if umpires.parquet is missing.
    """
    umpire_path = config.BASE_STORE_DIR / "umpires.parquet"
    if not umpire_path.exists():
        print("    umpires.parquet not found — skipping umpire features")
        return df

    umpires = pd.read_parquet(umpire_path)
    if umpires.empty:
        return df

    umpires = umpires.copy()
    umpires["umpire_career_games_pre"] = (
        pd.to_numeric(umpires["umpire_career_games"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .sub(1)
        .clip(lower=0)
    ).astype(np.float32)

    df = df.copy()
    match_dates = df[["match_id", "date"]].drop_duplicates()
    match_dates["date"] = pd.to_datetime(match_dates["date"], errors="coerce")

    # --- Panel average experience (static per match) ---
    panel_exp = (
        umpires.groupby("match_id", observed=True)["umpire_career_games_pre"]
        .mean()
        .reset_index()
        .rename(columns={"umpire_career_games_pre": "umpire_panel_avg_experience"})
    )
    df = df.merge(panel_exp, on="match_id", how="left")
    panel_exp_fallback = umpires["umpire_career_games_pre"].replace(0, np.nan).median()
    if pd.isna(panel_exp_fallback):
        panel_exp_fallback = 0.0
    df["umpire_panel_avg_experience"] = (
        df["umpire_panel_avg_experience"]
        .fillna(panel_exp_fallback)
        .astype(np.float32)
    )

    # --- Umpire scoring tendency ---
    # Need match total goals for each umpire's past matches
    match_goals = (
        df.groupby("match_id", observed=True)["GL"]
        .sum()
        .reset_index()
        .rename(columns={"GL": "match_total_goals"})
    )
    match_goals = (
        match_goals.merge(match_dates, on="match_id", how="left")
        .sort_values(["date", "match_id"])
        .reset_index(drop=True)
    )
    match_goals["league_goals_prior"] = (
        match_goals["match_total_goals"].shift(1).expanding(min_periods=1).mean()
    )
    ump_match = umpires.merge(match_goals, on="match_id", how="left")

    # Match dates for sorting
    ump_match = ump_match.sort_values(["date", "match_id", "umpire_name"])

    lookback = config.UMPIRE_LOOKBACK_MATCHES
    ug = ump_match.groupby("umpire_name", observed=True)
    ump_match["umpire_scoring_tendency"] = ug["match_total_goals"].transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=3).mean()
    )

    # Average across panel for each match
    panel_tendency = (
        ump_match.groupby("match_id", observed=True)["umpire_scoring_tendency"]
        .mean()
        .reset_index()
    )
    df = df.merge(panel_tendency, on="match_id", how="left")
    df = df.merge(
        match_goals[["match_id", "league_goals_prior"]],
        on="match_id", how="left",
    )
    goals_fallback = ump_match["match_total_goals"].dropna().mean()
    if pd.isna(goals_fallback):
        goals_fallback = 22.0
    df["umpire_scoring_tendency"] = (
        df["umpire_scoring_tendency"]
        .fillna(df["league_goals_prior"])
        .fillna(goals_fallback)
        .astype(np.float32)
    )
    df = df.drop(columns=["league_goals_prior"], errors="ignore")

    # --- Umpire home bias ---
    # Need home/away free kicks per match
    ff_by_team = (
        df.groupby(["match_id", "is_home"], observed=True)["FF"]
        .sum()
        .reset_index()
    )
    home_ff = ff_by_team[ff_by_team["is_home"] == 1][["match_id", "FF"]].rename(
        columns={"FF": "home_ff"}
    )
    away_ff = ff_by_team[ff_by_team["is_home"] == 0][["match_id", "FF"]].rename(
        columns={"FF": "away_ff"}
    )
    match_ff = home_ff.merge(away_ff, on="match_id", how="outer").fillna(0)
    match_ff["ff_home_diff"] = match_ff["home_ff"] - match_ff["away_ff"]

    ump_ff = umpires.merge(match_ff[["match_id", "ff_home_diff"]], on="match_id", how="left")
    ump_ff = ump_ff.merge(match_dates, on="match_id", how="left")
    ump_ff = ump_ff.sort_values(["date", "match_id", "umpire_name"])

    ug_ff = ump_ff.groupby("umpire_name", observed=True)
    ump_ff["umpire_home_bias"] = ug_ff["ff_home_diff"].transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=3).mean()
    )

    panel_bias = (
        ump_ff.groupby("match_id", observed=True)["umpire_home_bias"]
        .mean()
        .reset_index()
    )
    df = df.merge(panel_bias, on="match_id", how="left")
    df["umpire_home_bias"] = df["umpire_home_bias"].fillna(0).astype(np.float32)

    # --- Umpire familiarity (per-team) ---
    # For each (umpire, team), count of prior games together
    # We need team info per match — expand umpires to team level
    match_teams = df[["match_id", "team"]].drop_duplicates()
    ump_teams = umpires.merge(match_teams, on="match_id", how="left")
    ump_teams = ump_teams.merge(match_dates, on="match_id", how="left")
    ump_teams = ump_teams.sort_values(["date", "match_id", "umpire_name", "team"])

    ut_grp = ump_teams.groupby(["umpire_name", "team"], observed=True)
    ump_teams["_ump_team_games"] = ut_grp["match_id"].transform(
        lambda s: s.shift(1).expanding().count()
    )

    # Average familiarity across panel for each (match_id, team)
    panel_fam = (
        ump_teams.groupby(["match_id", "team"], observed=True)["_ump_team_games"]
        .mean()
        .reset_index()
        .rename(columns={"_ump_team_games": "umpire_familiarity"})
    )
    df = df.merge(panel_fam, on=["match_id", "team"], how="left")
    df["umpire_familiarity"] = df["umpire_familiarity"].fillna(0).astype(np.float32)

    n_with = df["umpire_panel_avg_experience"].notna().sum()
    print(f"    Umpire features: {n_with}/{len(df)} rows with data")
    return df


# ---------------------------------------------------------------------------
# O. Coach Features (team-level)
# ---------------------------------------------------------------------------

def add_coach_features(df):
    """Add coach features: win rate, experience, tenure, vs-opponent win rate.

    Loads coaches.parquet and computes expanding/rolling coach stats.
    Uses team_matches.parquet for result history.
    Gracefully skips if coaches.parquet is missing.
    """
    coaches_path = config.BASE_STORE_DIR / "coaches.parquet"
    if not coaches_path.exists():
        print("    coaches.parquet not found — skipping coach features")
        return df

    coaches = pd.read_parquet(coaches_path)
    if coaches.empty:
        return df

    df = df.copy()

    # Load team results from team_matches
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if not tm_path.exists():
        print("    team_matches.parquet not found — skipping coach features")
        return df
    tm = pd.read_parquet(tm_path, columns=["match_id", "team", "opponent", "result", "date"])
    tm["date"] = pd.to_datetime(tm["date"])

    # Join coach to team results
    coach_results = tm.merge(
        coaches[["match_id", "team", "coach"]],
        on=["match_id", "team"],
        how="left",
    )
    coach_results = coach_results.dropna(subset=["coach"])
    coach_results = coach_results.sort_values("date")

    if coach_results.empty:
        return df

    coach_results["won"] = (coach_results["result"] == "W").astype(int)

    # --- Coach expanding win rate ---
    cg = coach_results.groupby("coach", observed=True)
    coach_results["coach_win_rate"] = cg["won"].transform(
        lambda s: s.shift(1).expanding(min_periods=config.COACH_MIN_GAMES).mean()
    )

    # --- Coach experience (game count) ---
    coach_results["coach_experience_games"] = cg["won"].transform(
        lambda s: s.shift(1).expanding().count()
    )

    # --- Coach tenure (years with current team) ---
    # Find first game with current team per (coach, team)
    first_game = (
        coach_results.groupby(["coach", "team"], observed=True)["date"]
        .transform("first")
    )
    coach_results["coach_tenure"] = (
        (coach_results["date"] - first_game).dt.days / 365.25
    ).clip(lower=0)

    # --- Coach vs opponent win rate ---
    co_grp = coach_results.groupby(["coach", "opponent"], observed=True)
    coach_results["coach_vs_opp_win_rate"] = co_grp["won"].transform(
        lambda s: s.shift(1).expanding(min_periods=config.COACH_H2H_MIN_MATCHES).mean()
    )

    # Select merge columns
    coach_feat = coach_results[[
        "match_id", "team", "coach_win_rate", "coach_experience_games",
        "coach_tenure", "coach_vs_opp_win_rate",
    ]].drop_duplicates(subset=["match_id", "team"])

    df = df.merge(coach_feat, on=["match_id", "team"], how="left")

    # Fill NaN
    df["coach_win_rate"] = df["coach_win_rate"].fillna(0.5).astype(np.float32)
    df["coach_experience_games"] = df["coach_experience_games"].fillna(0).astype(np.float32)
    df["coach_tenure"] = df["coach_tenure"].fillna(0).astype(np.float32)
    df["coach_vs_opp_win_rate"] = df["coach_vs_opp_win_rate"].fillna(0.5).astype(np.float32)

    n_with = (df["coach_experience_games"] > 0).sum()
    print(f"    Coach features: {n_with}/{len(df)} rows with data")
    return df


# ---------------------------------------------------------------------------
# P. Player Physical Features (player-level)
# ---------------------------------------------------------------------------

def add_physical_features(df):
    """Add player physical features: height, weight, BMI, age_at_match,
    height_for_role, weight_for_role.

    Loads player_profiles.parquet.  Must run after archetypes (Stage K)
    so the 'archetype' column exists for role-relative features.
    Gracefully skips if profiles are missing.
    """
    profiles_path = config.BASE_STORE_DIR / "player_profiles.parquet"
    if not profiles_path.exists():
        print("    player_profiles.parquet not found — skipping physical features")
        return df

    profiles = pd.read_parquet(profiles_path)
    if profiles.empty:
        return df

    df = df.copy()

    # Merge on player name
    prof_cols = ["player", "height_cm", "weight_kg", "dob"]
    available = [c for c in prof_cols if c in profiles.columns]
    df = df.merge(profiles[available], on="player", how="left")

    # Fill missing with config fallbacks
    height_fb = config.PLAYER_PROFILE_HEIGHT_FALLBACK
    weight_fb = config.PLAYER_PROFILE_WEIGHT_FALLBACK

    df["height_cm"] = df["height_cm"].fillna(height_fb).astype(np.float32)
    df["weight_kg"] = df["weight_kg"].fillna(weight_fb).astype(np.float32)

    # BMI
    df["bmi"] = (
        df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
    ).astype(np.float32)

    # Age at match (more precise than age_years from player_details)
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["age_at_match"] = (
            (df["date"] - df["dob"]).dt.days / 365.25
        ).astype(np.float32)
        # Fill missing DOB with existing age_years or fallback
        if "age_years" in df.columns:
            df["age_at_match"] = df["age_at_match"].fillna(
                df["age_years"]
            )
        df["age_at_match"] = df["age_at_match"].fillna(
            config.PLAYER_PROFILE_AGE_FALLBACK
        ).astype(np.float32)
        df = df.drop(columns=["dob"], errors="ignore")
    elif "age_years" in df.columns:
        df["age_at_match"] = df["age_years"].fillna(
            config.PLAYER_PROFILE_AGE_FALLBACK
        ).astype(np.float32)
    else:
        df["age_at_match"] = np.float32(config.PLAYER_PROFILE_AGE_FALLBACK)

    # Height/weight relative to archetype mean
    if "archetype" in df.columns:
        arch_means = df.groupby("archetype", observed=True).agg(
            arch_mean_height=("height_cm", "mean"),
            arch_mean_weight=("weight_kg", "mean"),
        )
        df = df.merge(arch_means, on="archetype", how="left")

        df["height_for_role"] = np.where(
            df["arch_mean_height"] > 0,
            df["height_cm"] / df["arch_mean_height"],
            1.0,
        ).astype(np.float32)
        df["weight_for_role"] = np.where(
            df["arch_mean_weight"] > 0,
            df["weight_kg"] / df["arch_mean_weight"],
            1.0,
        ).astype(np.float32)

        df = df.drop(columns=["arch_mean_height", "arch_mean_weight"], errors="ignore")
    else:
        df["height_for_role"] = np.float32(1.0)
        df["weight_for_role"] = np.float32(1.0)

    n_with = (df["height_cm"] != height_fb).sum()
    print(f"    Physical features: {n_with}/{len(df)} rows with profile data")
    return df


# ---------------------------------------------------------------------------
# Q. Career Split Features (player + context level)
# ---------------------------------------------------------------------------

def add_career_split_features(df):
    """Add career split features from profile scrape data.

    Uses career totals by opponent and by venue to create matchup boost features.
    Gracefully skips if split data is missing.
    """
    if not getattr(config, "CAREER_SPLIT_FEATURES_ENABLED", False):
        print("    career split features disabled in config (leakage guard)")
        return df

    opp_path = config.BASE_STORE_DIR / "career_splits_opponent.parquet"
    venue_path = config.BASE_STORE_DIR / "career_splits_venue.parquet"

    has_opp = opp_path.exists()
    has_venue = venue_path.exists()

    if not has_opp and not has_venue:
        print("    career splits not found — skipping career split features")
        return df

    df = df.copy()
    min_games = config.CAREER_SPLIT_MIN_GAMES

    if has_opp:
        opp_df = pd.read_parquet(opp_path)
        if not opp_df.empty and "P" in opp_df.columns:
            # Filter to minimum games
            opp_df = opp_df[opp_df["P"] >= min_games].copy()

            # Compute per-game averages
            if "GL" in opp_df.columns:
                opp_df["career_vs_opp_gl_avg"] = (
                    opp_df["GL"] / opp_df["P"]
                ).astype(np.float32)
            if "DI" in opp_df.columns:
                opp_df["career_vs_opp_di_avg"] = (
                    opp_df["DI"] / opp_df["P"]
                ).astype(np.float32)

            # Overall career average per player (for boost ratio)
            player_overall = (
                opp_df.groupby("player", observed=True)
                .agg(
                    _total_gl=("GL", "sum") if "GL" in opp_df.columns else ("P", "first"),
                    _total_di=("DI", "sum") if "DI" in opp_df.columns else ("P", "first"),
                    _total_p=("P", "sum"),
                )
                .reset_index()
            )
            if "GL" in opp_df.columns:
                player_overall["_career_gl_avg"] = (
                    player_overall["_total_gl"] / player_overall["_total_p"]
                )
            if "DI" in opp_df.columns:
                player_overall["_career_di_avg"] = (
                    player_overall["_total_di"] / player_overall["_total_p"]
                )

            # Merge opponent averages
            merge_cols = ["player", "opponent"]
            opp_feat_cols = [c for c in ["career_vs_opp_gl_avg", "career_vs_opp_di_avg"]
                            if c in opp_df.columns]
            if opp_feat_cols:
                df = df.merge(
                    opp_df[merge_cols + opp_feat_cols].drop_duplicates(subset=merge_cols),
                    on=["player", "opponent"],
                    how="left",
                )

            # Compute opponent boost ratios
            if "_career_gl_avg" in player_overall.columns and "career_vs_opp_gl_avg" in df.columns:
                df = df.merge(
                    player_overall[["player", "_career_gl_avg"]],
                    on="player", how="left",
                )
                df["career_opponent_boost"] = np.where(
                    df["_career_gl_avg"] > 0,
                    df["career_vs_opp_gl_avg"] / df["_career_gl_avg"],
                    1.0,
                )
                df["career_opponent_boost"] = df["career_opponent_boost"].clip(0.2, 5.0).astype(np.float32)
                df = df.drop(columns=["_career_gl_avg"], errors="ignore")

    if has_venue:
        venue_df = pd.read_parquet(venue_path)
        if not venue_df.empty and "P" in venue_df.columns:
            venue_df = venue_df[venue_df["P"] >= min_games].copy()

            if "GL" in venue_df.columns:
                venue_df["career_at_venue_gl_avg"] = (
                    venue_df["GL"] / venue_df["P"]
                ).astype(np.float32)
            if "DI" in venue_df.columns:
                venue_df["career_at_venue_di_avg"] = (
                    venue_df["DI"] / venue_df["P"]
                ).astype(np.float32)

            # Overall career average per player for venue boost
            player_overall_v = (
                venue_df.groupby("player", observed=True)
                .agg(
                    _total_gl_v=("GL", "sum") if "GL" in venue_df.columns else ("P", "first"),
                    _total_p_v=("P", "sum"),
                )
                .reset_index()
            )
            if "GL" in venue_df.columns:
                player_overall_v["_career_gl_avg_v"] = (
                    player_overall_v["_total_gl_v"] / player_overall_v["_total_p_v"]
                )

            merge_cols_v = ["player", "venue"]
            venue_feat_cols = [c for c in ["career_at_venue_gl_avg", "career_at_venue_di_avg"]
                              if c in venue_df.columns]
            if venue_feat_cols:
                df = df.merge(
                    venue_df[merge_cols_v + venue_feat_cols].drop_duplicates(subset=merge_cols_v),
                    on=["player", "venue"],
                    how="left",
                )

            if "_career_gl_avg_v" in player_overall_v.columns and "career_at_venue_gl_avg" in df.columns:
                df = df.merge(
                    player_overall_v[["player", "_career_gl_avg_v"]],
                    on="player", how="left",
                )
                df["career_venue_boost"] = np.where(
                    df["_career_gl_avg_v"] > 0,
                    df["career_at_venue_gl_avg"] / df["_career_gl_avg_v"],
                    1.0,
                )
                df["career_venue_boost"] = df["career_venue_boost"].clip(0.2, 5.0).astype(np.float32)
                df = df.drop(columns=["_career_gl_avg_v"], errors="ignore")

    # Fill NaN for all career split features
    for col in ["career_vs_opp_gl_avg", "career_vs_opp_di_avg",
                "career_at_venue_gl_avg", "career_at_venue_di_avg",
                "career_opponent_boost", "career_venue_boost"]:
        if col in df.columns:
            fill = 1.0 if "boost" in col else 0.0
            df[col] = df[col].fillna(fill).astype(np.float32)

    n_opp = (df.get("career_vs_opp_gl_avg", pd.Series(0)) > 0).sum() if "career_vs_opp_gl_avg" in df.columns else 0
    n_venue = (df.get("career_at_venue_gl_avg", pd.Series(0)) > 0).sum() if "career_at_venue_gl_avg" in df.columns else 0
    print(f"    Career splits: {n_opp} opp matches, {n_venue} venue matches")
    return df


# ---------------------------------------------------------------------------
# R. Team Venue Features (derived from team_matches.parquet)
# ---------------------------------------------------------------------------

def add_team_venue_features(df):
    """Add team venue record features: win rate, average score at venue.

    Computed from team_matches.parquet — no additional scraping needed.
    Uses a rolling time window and past-only priors by date to avoid lookahead.
    """
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if not tm_path.exists():
        print("    team_matches.parquet not found — skipping team venue features")
        return df

    tm = pd.read_parquet(tm_path, columns=["match_id", "team", "opponent", "venue",
                                             "score", "result", "date", "year"])
    if tm.empty:
        return df

    df = df.copy()
    if "date" not in df.columns:
        print("    date column missing — skipping team venue features")
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    tm = tm.dropna(subset=["date"]).sort_values(["team", "venue", "date", "match_id"]).copy()
    tm["won"] = (tm["result"] == "W").astype(np.int8)

    min_games = max(1, int(config.TEAM_VENUE_MIN_GAMES))
    lookback_days = max(1, int(round(config.TEAM_VENUE_LOOKBACK_YEARS * 365.25)))
    lookback_delta = np.timedelta64(lookback_days, "D")

    # Compute per-(team, venue) rolling metrics using only prior matches
    tm["_tv_win_rate"] = np.nan
    tm["_tv_avg_score"] = np.nan
    for _, grp in tm.groupby(["team", "venue"], observed=True):
        idx = grp.index.to_numpy()
        dates = grp["date"].to_numpy(dtype="datetime64[ns]")
        wins = grp["won"].to_numpy(dtype=np.float64)
        scores = grp["score"].to_numpy(dtype=np.float64)
        out_win = np.full(len(grp), np.nan, dtype=np.float64)
        out_score = np.full(len(grp), np.nan, dtype=np.float64)

        left = 0
        for i in range(len(grp)):
            cutoff = dates[i] - lookback_delta
            while left < i and dates[left] < cutoff:
                left += 1
            n_hist = i - left
            if n_hist >= min_games:
                out_win[i] = wins[left:i].mean()
                out_score[i] = scores[left:i].mean()

        tm.loc[idx, "_tv_win_rate"] = out_win
        tm.loc[idx, "_tv_avg_score"] = out_score

    # Build as-of lookup for requested rows (supports synthetic future match_ids)
    req = df[["match_id", "team", "opponent", "venue", "date"]].drop_duplicates().copy()
    req_valid = req.dropna(subset=["date"]).copy()
    hist = tm[["team", "venue", "date", "_tv_win_rate", "_tv_avg_score"]].copy()
    req_valid["team"] = req_valid["team"].astype(str)
    req_valid["venue"] = req_valid["venue"].astype(str)
    hist["team"] = hist["team"].astype(str)
    hist["venue"] = hist["venue"].astype(str)

    if not req_valid.empty and not hist.empty:
        parts = []
        for (team, venue), req_grp in req_valid.groupby(["team", "venue"], observed=True):
            hist_grp = hist[
                (hist["team"] == team) & (hist["venue"] == venue)
            ][["date", "_tv_win_rate", "_tv_avg_score"]]
            req_sorted = req_grp.sort_values("date")
            if hist_grp.empty:
                merged = req_sorted.copy()
                merged["_tv_win_rate"] = np.nan
                merged["_tv_avg_score"] = np.nan
            else:
                merged = pd.merge_asof(
                    req_sorted,
                    hist_grp.sort_values("date"),
                    on="date",
                    direction="backward",
                    allow_exact_matches=True,
                )
            parts.append(merged)
        req_tv = pd.concat(parts, ignore_index=True) if parts else req_valid.copy()
    else:
        req_tv = req_valid.copy()
        req_tv["_tv_win_rate"] = np.nan
        req_tv["_tv_avg_score"] = np.nan

    tv_feat = req_tv[["match_id", "team", "_tv_win_rate", "_tv_avg_score"]].rename(
        columns={
            "_tv_win_rate": "team_venue_win_rate",
            "_tv_avg_score": "team_venue_avg_score",
        }
    )
    tv_feat["team"] = tv_feat["team"].astype(str)
    df = df.merge(tv_feat, on=["match_id", "team"], how="left")

    # Opponent venue win rate from the same as-of lookup
    opp_tv = tv_feat.rename(
        columns={
            "team": "opponent",
            "team_venue_win_rate": "opponent_venue_win_rate",
        }
    )
    df = df.merge(
        opp_tv[["match_id", "opponent", "opponent_venue_win_rate"]],
        on=["match_id", "opponent"],
        how="left",
    )

    # Past-only league score prior for neutral fallback
    league_tm = tm.sort_values(["date", "match_id", "team"]).copy()
    league_tm["_league_score_prior"] = (
        league_tm["score"].shift(1).expanding(min_periods=1).mean()
    )
    date_key = df[["match_id", "date"]].drop_duplicates().dropna(subset=["date"])
    league_ref = league_tm[["date", "_league_score_prior"]].dropna().sort_values("date")
    if not date_key.empty and not league_ref.empty:
        prior = pd.merge_asof(
            date_key.sort_values("date"),
            league_ref,
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        df = df.merge(prior, on=["match_id", "date"], how="left")
    else:
        df["_league_score_prior"] = np.nan

    # --- Venue advantage differential ---
    df["venue_advantage_diff"] = (
        df["team_venue_win_rate"].fillna(0.5) -
        df["opponent_venue_win_rate"].fillna(0.5)
    ).astype(np.float32)

    # Fill NaN with leakage-safe defaults
    score_fallback = float(getattr(config, "TEAM_VENUE_SCORE_FALLBACK", 80.0))
    df["team_venue_win_rate"] = df["team_venue_win_rate"].fillna(0.5).astype(np.float32)
    df["team_venue_avg_score"] = (
        df["team_venue_avg_score"]
        .fillna(df["_league_score_prior"])
        .fillna(score_fallback)
        .astype(np.float32)
    )
    df["opponent_venue_win_rate"] = df["opponent_venue_win_rate"].fillna(0.5).astype(np.float32)
    df = df.drop(columns=["_league_score_prior"], errors="ignore")

    n_with = (df["team_venue_win_rate"] != 0.5).sum()
    print(f"    Team venue features: {n_with}/{len(df)} rows with venue history")
    return df


# ---------------------------------------------------------------------------
# S. Player Market Odds Features
# ---------------------------------------------------------------------------

def add_player_odds_features(df):
    """Merge player-level Betfair market features (disposal lines, FGS, goals).

    Loads player_odds.parquet and left-joins on (match_id, player).
    Gracefully skips if player_odds.parquet is missing.
    """
    odds_path = config.BASE_STORE_DIR / "player_odds.parquet"
    if not odds_path.exists():
        print("    WARNING: player_odds.parquet not found — skipping player market features")
        print("    Run: python integrate_player_odds.py")
        return df

    odds = pd.read_parquet(odds_path)
    if odds.empty:
        print("    WARNING: player_odds.parquet is empty — skipping player market features")
        return df

    if "match_id" not in odds.columns or "player" not in odds.columns:
        print("    WARNING: player_odds.parquet missing match_id/player keys — skipping")
        return df

    odds = odds.copy()
    odds["match_id"] = pd.to_numeric(odds["match_id"], errors="coerce")
    odds["player"] = odds["player"].astype(str).str.strip()
    odds = odds.dropna(subset=["match_id", "player"])
    odds = odds[(odds["player"] != "") & (odds["player"].str.lower() != "nan")]
    odds["match_id"] = odds["match_id"].astype(np.int64)

    # Player market feature columns
    feature_cols = [c for c in odds.columns if c.startswith("market_")]

    if not feature_cols:
        print("    WARNING: No market_ columns in player_odds.parquet")
        return df

    # Drop existing columns if re-running (idempotent)
    existing = [c for c in feature_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    # De-duplicate newly pulled odds rows to avoid merge row explosion
    dupes = odds.duplicated(subset=["match_id", "player"], keep=False)
    if dupes.any():
        n_keys = odds.loc[dupes, ["match_id", "player"]].drop_duplicates().shape[0]
        print(f"    WARNING: deduplicating {n_keys} duplicate (match_id, player) odds keys")
        odds = odds.drop_duplicates(subset=["match_id", "player"], keep="last")

    # Clamp probability-like columns to [0, 1]
    prob_cols = [c for c in feature_cols if ("implied" in c) or c.endswith("_prob")]
    for col in prob_cols:
        odds[col] = pd.to_numeric(odds[col], errors="coerce").clip(0, 1)

    merge_cols = ["match_id", "player"] + feature_cols
    available = [c for c in merge_cols if c in odds.columns]
    df = df.merge(odds[available], on=["match_id", "player"], how="left")

    # HistGBT handles NaN natively — no fill needed
    n_with = 0
    for col in feature_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            n_with = max(n_with, n)

    print(f"    Player market features: {len(feature_cols)} columns, "
          f"{n_with}/{len(df)} rows with data")
    return df


# ---------------------------------------------------------------------------
# Venue Elevation Features
# ---------------------------------------------------------------------------

VENUE_ELEVATION = {
    "M.C.G.": 30, "Docklands": 8, "Adelaide Oval": 48,
    "Perth Stadium": 14, "Gabba": 27, "Carrara": 6,
    "S.C.G.": 45, "Kardinia Park": 9, "Sydney Showground": 39,
    "York Park": 165, "Bellerive Oval": 4, "Manuka Oval": 578,
    "Marrara Oval": 30, "Eureka Stadium": 435, "Cazaly's Stadium": 5,
    "Traeger Park": 546, "Stadium Australia": 7, "Jiangwan Stadium": 4,
    "Riverway Stadium": 10, "Wellington": 35,
    "Subiaco": 25, "Norwood Oval": 48, "Summit Sports Park": 100,
    "Barossa Oval": 275, "Hands Oval": 435,
}


def add_venue_elevation_features(df):
    """Add venue elevation and high-altitude flag.

    High altitude (>200m): ball travels further, players fatigue faster.
    Only 3+ venues are meaningfully elevated: Manuka (578m), Traeger (546m),
    Eureka/Hands (435m), Barossa (275m).
    """
    df["venue_elevation_m"] = df["venue"].map(VENUE_ELEVATION).astype(np.float32)
    # Fill unknown venues with median
    median_elev = np.nanmedian(list(VENUE_ELEVATION.values()))
    df["venue_elevation_m"] = df["venue_elevation_m"].fillna(median_elev).astype(np.float32)

    df["is_high_altitude"] = (df["venue_elevation_m"] > 200).astype(np.int8)

    n_high = (df["is_high_altitude"] == 1).sum()
    print(f"    Venue elevation: {n_high}/{len(df)} rows at high-altitude venues")
    return df


# ---------------------------------------------------------------------------
# U. Disposal-Specific Interaction Features (runs after weather + ground)
# ---------------------------------------------------------------------------

def add_disposal_interaction_features(df):
    """Cross-feature interactions for disposal prediction.

    Runs after weather, ground dimensions, rest days, and opponent tackle
    features are available. Captures disposal-specific signal combinations.
    """
    df = df.copy()

    # Weather × disposal form
    if "rain_total" in df.columns and "player_di_ewm_5" in df.columns:
        df["interact_rain_disp_form"] = (
            df["rain_total"].fillna(0) * df["player_di_ewm_5"].fillna(0)
        )
    if "wind_speed_avg" in df.columns and "player_ki_hb_ratio_5" in df.columns:
        df["interact_wind_kick_ratio"] = (
            df["wind_speed_avg"].fillna(0) * df["player_ki_hb_ratio_5"].fillna(0.5)
        )
    if "weather_difficulty_score" in df.columns and "player_di_ewm_5" in df.columns:
        df["interact_weather_disp"] = (
            df["weather_difficulty_score"].fillna(0) * df["player_di_ewm_5"].fillna(0)
        )
    if "slippery_conditions" in df.columns and "player_cp_avg_5" in df.columns:
        df["interact_slippery_contested"] = (
            df["slippery_conditions"].fillna(0) * df["player_cp_avg_5"].fillna(0)
        )

    # Kick/handball ratio × ground length
    if "player_ki_hb_ratio_5" in df.columns and "ground_length" in df.columns:
        df["interact_kick_ratio_ground"] = (
            df["player_ki_hb_ratio_5"].fillna(0.5) * df["ground_length"].fillna(165)
        )

    # Disposal form × opponent tackle pressure
    if "player_di_ewm_5" in df.columns and "opp_tackle_rate_avg_5" in df.columns:
        df["interact_disp_vs_opp_tackles"] = (
            df["player_di_ewm_5"].fillna(0) * df["opp_tackle_rate_avg_5"].fillna(0)
        )

    # Disposal form × rest day differential
    if "player_di_ewm_5" in df.columns and "rest_day_differential" in df.columns:
        df["interact_disp_rest_diff"] = (
            df["player_di_ewm_5"].fillna(0) * df["rest_day_differential"].fillna(0)
        )

    count = sum(1 for c in df.columns if c.startswith("interact_") and "disp" in c.lower()
                or c in ("interact_rain_disp_form", "interact_wind_kick_ratio",
                         "interact_weather_disp", "interact_slippery_contested",
                         "interact_kick_ratio_ground"))
    print(f"    Added {count} disposal-specific interaction features")
    return df


# ---------------------------------------------------------------------------
# X. Marks-Specific Interaction Features (runs after weather + ground + opponent)
# ---------------------------------------------------------------------------

def add_marks_interaction_features(df):
    """Cross-feature interactions for marks prediction.

    Runs after weather, ground dimensions, rest days, and opponent marks
    concession features are available.
    """
    df = df.copy()
    count = 0

    # Wind × marks form
    if "wind_speed_avg" in df.columns and "player_mk_ewm_5" in df.columns:
        df["interact_wind_marks"] = (
            df["wind_speed_avg"].fillna(0) * df["player_mk_ewm_5"].fillna(0)
        )
        count += 1

    # Rain × marks form
    if "rain_total" in df.columns and "player_mk_ewm_5" in df.columns:
        df["interact_rain_marks"] = (
            df["rain_total"].fillna(0) * df["player_mk_ewm_5"].fillna(0)
        )
        count += 1

    # Ground area × marks form
    if "ground_area" in df.columns and "player_mk_ewm_5" in df.columns:
        df["interact_ground_area_marks"] = (
            df["ground_area"].fillna(0) * df["player_mk_ewm_5"].fillna(0)
        )
        count += 1

    # Marks form × opponent marks concession
    if "player_mk_ewm_5" in df.columns and "opp_marks_conceded_avg_5" in df.columns:
        df["interact_marks_vs_opp_conceded"] = (
            df["player_mk_ewm_5"].fillna(0) * df["opp_marks_conceded_avg_5"].fillna(0)
        )
        count += 1

    # Marks share × team form
    if "player_mk_share_5" in df.columns and "team_win_pct_5" in df.columns:
        df["interact_mk_share_team_form"] = (
            df["player_mk_share_5"].fillna(0) * df["team_win_pct_5"].fillna(0)
        )
        count += 1

    # Contested mark ratio × opponent tackle pressure
    if "player_contested_mk_ratio_5" in df.columns and "opp_tackle_rate_avg_5" in df.columns:
        df["interact_contested_mk_vs_tackles"] = (
            df["player_contested_mk_ratio_5"].fillna(0) * df["opp_tackle_rate_avg_5"].fillna(0)
        )
        count += 1

    # Marks form × rest day differential
    if "player_mk_ewm_5" in df.columns and "rest_day_differential" in df.columns:
        df["interact_marks_rest_diff"] = (
            df["player_mk_ewm_5"].fillna(0) * df["rest_day_differential"].fillna(0)
        )
        count += 1

    print(f"    Added {count} marks-specific interaction features")
    return df


# ---------------------------------------------------------------------------
# V. FootyWire Advanced Stats Features
# ---------------------------------------------------------------------------

def add_footywire_features(df):
    """Add features from FootyWire advanced stats (ED, DE%, CCL, SCL, TO, MG, TOG%, ITC).

    Loads footywire_advanced.parquet and computes rolling averages of
    advanced stats that are NOT available from AFLTables. All features
    use shift(1) to prevent leakage.

    Gracefully skips if footywire_advanced.parquet is missing (pre-scrape).
    """
    fw_path = config.BASE_STORE_DIR / "footywire_advanced.parquet"
    if not fw_path.exists():
        print("    footywire_advanced.parquet not found — skipping FootyWire features")
        return df

    fw = pd.read_parquet(fw_path)
    if fw.empty:
        return df

    df = df.copy()

    # Merge FootyWire stats onto the main DataFrame
    fw_cols = ["ED", "DE_pct", "CCL", "SCL", "TO", "MG", "SI", "ITC", "T5", "TOG_pct"]
    # Prefix with fw_ to avoid collision with existing columns
    fw_rename = {c: f"_fw_{c}" for c in fw_cols}
    fw_merge = fw[["match_id", "player"] + fw_cols].rename(columns=fw_rename)

    df = df.merge(fw_merge, on=["match_id", "player"], how="left")

    n_with = df["_fw_ED"].notna().sum()
    n_total = len(df)
    print(f"    FootyWire data: {n_with}/{n_total} rows matched ({n_with/n_total*100:.1f}%)")

    if n_with == 0:
        df = df.drop(columns=[f"_fw_{c}" for c in fw_cols], errors="ignore")
        return df

    # Compute rolling features from FootyWire stats
    df = df.sort_values(["player", "team", "date"]).copy()
    grouped = df.groupby(["player", "team"], observed=True)

    # Rolling averages for key advanced stats
    fw_roll_stats = {
        "_fw_ED": "fw_ed",
        "_fw_DE_pct": "fw_de_pct",
        "_fw_CCL": "fw_ccl",
        "_fw_SCL": "fw_scl",
        "_fw_TO": "fw_to",
        "_fw_MG": "fw_mg",
        "_fw_ITC": "fw_itc",
        "_fw_TOG_pct": "fw_tog_pct",
    }

    for src_col, feat_name in fw_roll_stats.items():
        if src_col not in df.columns:
            continue
        for window in [5, 10]:
            col_name = f"player_{feat_name}_avg_{window}"
            min_p = min(2, window)
            df[col_name] = grouped[src_col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=min_p).mean()
            )

    # EWM for key stats
    for src_col, feat_name in [("_fw_ED", "fw_ed"), ("_fw_DE_pct", "fw_de_pct"),
                                ("_fw_TOG_pct", "fw_tog_pct")]:
        if src_col in df.columns:
            df[f"player_{feat_name}_ewm_5"] = grouped[src_col].transform(
                lambda s: s.shift(1).ewm(span=5, min_periods=2).mean()
            )

    # Derived features
    # CCL / (CCL + SCL) ratio — inside-mid vs centre-mid
    if "_fw_CCL" in df.columns and "_fw_SCL" in df.columns:
        ccl_sum = grouped["_fw_CCL"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).sum()
        )
        scl_sum = grouped["_fw_SCL"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).sum()
        )
        total_cl = ccl_sum + scl_sum
        df["player_fw_ccl_scl_ratio_5"] = np.where(total_cl > 0, ccl_sum / total_cl, 0.5)

    # Turnover rate: TO / DI
    if "_fw_TO" in df.columns:
        to_sum = grouped["_fw_TO"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).sum()
        )
        di_sum = grouped["DI"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).sum()
        )
        df["player_fw_turnover_rate_5"] = np.where(di_sum > 0, to_sum / di_sum, 0.0)

    # Disposal efficiency trend
    if "_fw_DE_pct" in df.columns:
        df["player_fw_de_trend_5"] = _grouped_shifted_slope_5(
            grouped, "_fw_DE_pct"
        ).fillna(0)

    # Interaction: disposal efficiency × opponent tackle pressure
    if "player_fw_de_pct_avg_5" in df.columns and "opp_tackle_rate_avg_5" in df.columns:
        df["interact_fw_de_opp_pressure"] = (
            df["player_fw_de_pct_avg_5"].fillna(0) * df["opp_tackle_rate_avg_5"].fillna(0)
        )

    # Interaction: centre clearances × disposal form
    if "player_fw_ccl_avg_5" in df.columns and "player_di_ewm_5" in df.columns:
        df["interact_fw_ccl_disp_form"] = (
            df["player_fw_ccl_avg_5"].fillna(0) * df["player_di_ewm_5"].fillna(0)
        )

    # Drop raw FootyWire columns (keep only rolling features)
    raw_fw_cols = [f"_fw_{c}" for c in fw_cols]
    df = df.drop(columns=raw_fw_cols, errors="ignore")

    # Count new features
    fw_features = [c for c in df.columns if "fw_" in c]
    print(f"    Added {len(fw_features)} FootyWire-derived features")

    return df


# ---------------------------------------------------------------------------
# W. DFS Australia CBA Features
# ---------------------------------------------------------------------------

def add_cba_features(df):
    """Add Centre Bounce Attendance features from DFS Australia data.

    CBA% is the single best proxy for midfield role centrality (2020+ only).
    Gracefully skips if CBA data is not available.
    """
    cba_path = config.BASE_STORE_DIR / "cba_stats.parquet"
    if not cba_path.exists():
        print("    cba_stats.parquet not found — skipping CBA features")
        return df

    cba = pd.read_parquet(cba_path)
    if cba.empty:
        return df

    df = df.copy()

    # Merge CBA stats
    cba_merge = cba[["match_id", "player", "CBA_pct"]].copy()
    cba_merge = cba_merge.rename(columns={"CBA_pct": "_cba_pct"})
    df = df.merge(cba_merge, on=["match_id", "player"], how="left")

    n_with = df["_cba_pct"].notna().sum()
    print(f"    CBA data: {n_with}/{len(df)} rows matched")

    if n_with == 0:
        df = df.drop(columns=["_cba_pct"], errors="ignore")
        return df

    df = df.sort_values(["player", "team", "date"]).copy()
    grouped = df.groupby(["player", "team"], observed=True)

    # Rolling CBA%
    df["player_cba_pct_avg_5"] = grouped["_cba_pct"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )

    # CBA trend
    df["player_cba_trend_5"] = _grouped_shifted_slope_5(grouped, "_cba_pct").fillna(0)

    # CBA × TOG interaction
    if "player_tog_avg_5" in df.columns:
        df["interact_cba_tog"] = (
            df["player_cba_pct_avg_5"].fillna(0) * df["player_tog_avg_5"].fillna(75)
        )

    df = df.drop(columns=["_cba_pct"], errors="ignore")

    cba_features = [c for c in df.columns if "cba" in c.lower()]
    print(f"    Added {len(cba_features)} CBA features")

    return df


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

# Feature columns that the model will train on (assembled after all features added)
FEATURE_COLS = None  # Set dynamically after build

def build_features(df=None, data_dir=None, save=True):
    """Main entry point. Takes cleaned player_games DataFrame
    and returns it with all features added.

    If df is None, loads from base store parquet.
    Uses mtime-based cache: skips rebuild if feature matrix is newer than base data.
    """
    global FEATURE_COLS
    import json

    base_path = config.BASE_STORE_DIR / "player_games.parquet"
    cached_path = config.FEATURES_DIR / "feature_matrix.parquet"
    feat_list_path = config.FEATURES_DIR / "feature_columns.json"

    # Cache check: skip rebuild if features are newer than all source data
    weather_path = config.BASE_STORE_DIR / "weather.parquet"
    dims_path = config.DATA_DIR / "venue_dimensions.json"
    umpires_path = config.BASE_STORE_DIR / "umpires.parquet"
    coaches_path = config.BASE_STORE_DIR / "coaches.parquet"
    profiles_path = config.BASE_STORE_DIR / "player_profiles.parquet"
    opp_splits_path = config.BASE_STORE_DIR / "career_splits_opponent.parquet"
    venue_splits_path = config.BASE_STORE_DIR / "career_splits_venue.parquet"
    player_odds_path = config.BASE_STORE_DIR / "player_odds.parquet"
    footywire_path = config.BASE_STORE_DIR / "footywire_advanced.parquet"
    cba_path = config.BASE_STORE_DIR / "cba_stats.parquet"
    team_changes_path = config.BASE_STORE_DIR / "team_changes.parquet"
    injuries_path = config.BASE_STORE_DIR / "injuries.parquet"
    if save and df is None and cached_path.exists() and base_path.exists():
        cache_mtime = cached_path.stat().st_mtime
        sources_fresh = cache_mtime > base_path.stat().st_mtime
        # Invalidate cache when feature code or configuration changes.
        config_file = getattr(config, "__file__", None)
        code_paths = [Path(__file__)]
        if config_file:
            code_paths.append(Path(config_file))
        for code_path in code_paths:
            try:
                if code_path.exists():
                    sources_fresh = sources_fresh and cache_mtime > code_path.stat().st_mtime
            except Exception:
                pass
        if weather_path.exists():
            sources_fresh = sources_fresh and cache_mtime > weather_path.stat().st_mtime
        if dims_path.exists():
            sources_fresh = sources_fresh and cache_mtime > dims_path.stat().st_mtime
        for extra_path in [umpires_path, coaches_path, profiles_path,
                           opp_splits_path, venue_splits_path, player_odds_path,
                           footywire_path, cba_path, team_changes_path,
                           injuries_path]:
            if extra_path.exists():
                sources_fresh = sources_fresh and cache_mtime > extra_path.stat().st_mtime
        if sources_fresh:
            print("Using cached features (base data unchanged)")
            cached = pd.read_parquet(cached_path)
            if feat_list_path.exists():
                with open(feat_list_path) as f:
                    FEATURE_COLS = json.load(f)
            return cached

    if df is None:
        if base_path.exists():
            df = pd.read_parquet(base_path)
        else:
            # Fallback to legacy path
            cleaned_path = config.CLEANED_DIR / "player_matches.parquet"
            if cleaned_path.exists():
                df = pd.read_parquet(cleaned_path)
            else:
                from clean import build_player_games
                df = build_player_games(data_dir=data_dir, save=True)

    print(f"Building features for {len(df)} player-match rows...")

    # Filter to configured year range
    df = df[df["year"] >= config.HISTORICAL_START_YEAR].copy()
    print(f"  After year filter ({config.HISTORICAL_START_YEAR}+): {len(df)} rows")

    # A. Career features
    print("  [A] Career features...")
    df = add_career_features(df)

    # B. Rolling form features
    print("  [B] Rolling form features...")
    df = add_rolling_features(df)

    # C. Venue features
    print("  [C] Venue features...")
    df = add_venue_features(df)

    # D. Opponent defense features
    print("  [D] Opponent defense features...")
    df = add_opponent_features(df)

    # E. Matchup features (computationally expensive)
    print("  [E] Matchup features...")
    df = add_matchup_features(df)

    # F. Team context features
    print("  [F] Team context features...")
    df = add_team_features(df)

    # G. Scoring pattern features
    print("  [G] Scoring pattern features...")
    df = add_scoring_pattern_features(df)

    # H. Role classification
    print("  [H] Role classification...")
    df = classify_roles(df)

    # I. Teammate enabler features
    print("  [I] Enabler features...")
    df = add_enabler_features(df)

    # J. Interaction features
    print("  [J] Interaction features...")
    df = add_interaction_features(df)

    # K. GMM archetypes
    print("  [K] GMM archetypes...")
    df, gmm_model, archetype_names = build_archetypes(df)

    # K-b. Opponent concession profiles by archetype
    print("  [K-b] Archetype concession profiles...")
    df = add_archetype_concession_features(df)

    # K-b2. Archetype disposal ceiling features
    print("  [K-b2] Archetype disposal ceiling features...")
    df = add_archetype_disposal_features(df)

    # K-c. Game environment features
    print("  [K-c] Game environment features...")
    df = add_game_environment_features(df)

    # K-c2. Ground dimension features
    print("  [K-c2] Ground dimension features...")
    df = add_ground_dimension_features(df)

    # K-c3. Day/night classification
    print("  [K-c3] Day/night classification...")
    df = add_day_night_features(df)

    # K-c4. Market / odds features
    print("  [K-c4] Market / odds features...")
    df = add_market_features(df)

    # Save archetypes to LearningStore if we have a GMM model
    if gmm_model is not None and save:
        from store import LearningStore
        ls = LearningStore()
        arch_df = df[["player", "team", "archetype"]].drop_duplicates(
            subset=["player", "team"], keep="last"
        )
        ls.save_archetypes(arch_df)
        print(f"  Saved {len(arch_df)} archetype assignments to LearningStore")

    # K-d. Era / rule regime features
    print("  [K-d] Era / rule regime features...")
    df = add_era_features(df)

    # L. Weather features
    print("  [L] Weather features...")
    df = add_weather_features(df)

    # M. Sample weights
    print("  [M] Sample weights...")
    df = add_sample_weights(df)

    # N. Umpire features
    print("  [N] Umpire features...")
    df = add_umpire_features(df)

    # O. Coach features
    print("  [O] Coach features...")
    df = add_coach_features(df)

    # P. Player physical features (needs archetype column from K)
    print("  [P] Player physical features...")
    df = add_physical_features(df)

    # Q. Career split features
    print("  [Q] Career split features...")
    df = add_career_split_features(df)

    # R. Team venue features
    print("  [R] Team venue features...")
    df = add_team_venue_features(df)

    # S. Player market odds features
    print("  [S] Player market odds features...")
    df = add_player_odds_features(df)

    # T. Venue elevation features
    print("  [T] Venue elevation features...")
    df = add_venue_elevation_features(df)

    # T2. News / team changes features (ins/outs, injuries, debutants)
    print("  [T2] News / team changes features...")
    from news import add_news_features
    df = add_news_features(df)

    # U. Disposal-specific interaction features (needs weather + ground + rest days)
    print("  [U] Disposal interaction features...")
    df = add_disposal_interaction_features(df)

    # X. Marks-specific interaction features (needs weather + ground + opponent marks concession)
    print("  [X] Marks interaction features...")
    df = add_marks_interaction_features(df)

    # V. FootyWire advanced stats features — DISABLED (Phase 2 A/B test showed BSS regression)
    # FootyWire features add noise without improving disposal probability calibration.
    # Keeping code for future reference but not including in feature matrix.
    # print("  [V] FootyWire advanced stats features...")
    # df = add_footywire_features(df)

    # W. CBA features (DFS Australia, 2020+ only)
    print("  [W] CBA features...")
    df = add_cba_features(df)

    # Assemble feature column list

    # Raw stat columns (current-match values — leaky for prediction)
    # GL and BH are targets but also stats; including them ensures GL_rate/BH_rate get excluded too
    _stat_cols = {
        "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF", "CL", "CG",
        "FF", "FA", "BR", "CP", "UP", "CM", "MI", "one_pct", "BO", "GA",
    }
    # Rate-normalised versions are also leaky (current-match / pct_played)
    _rate_cols = {f"{c}_rate" for c in _stat_cols}

    exclude_cols = {
        # Identifiers
        "match_id", "year", "round_number", "round_label", "venue", "date",
        "team", "opponent", "jumper", "player", "player_role", "player_id",
        # Targets
        "GL", "BH", "DI",
        # Raw + rate current-match stats (leaky — current match values)
        *_stat_cols, "pct_played", *_rate_cols,
        # Quarter scoring (raw — we use the derived features instead)
        "q1_goals", "q1_behinds", "q2_goals", "q2_behinds",
        "q3_goals", "q3_behinds", "q4_goals", "q4_behinds",
        # Weight
        "sample_weight",
        # Raw pre-game career avg (the capped version is the feature)
        "career_goal_avg_pre",
        # Flags
        "did_not_play",
        # Archetype label (probabilities are features, label is identifier)
        "archetype",
        # Era label (one-hot era_1..era_6 are features, raw int is identifier)
        "season_era",
        # ── Pruned features (SHAP + permutation importance ≈ 0 across all 3 models) ──
        "is_roofed", "role_forward", "is_returning_from_break", "venue_is_indoor",
        "player_is_hot", "is_heavy_rain", "player_is_cold", "is_overcast",
        "player_streak_just_broke", "role_other", "role_midfielder",
        "era_1", "precipitation_total", "temperature_category", "is_wet",
        "era_5", "player_bh_avg_3", "player_bh_avg_5",
        "humidity_discomfort",
        # ── Redundant features (r=±1.0 with era_4, keep era_4 as representative) ──
        "is_covid_season", "quarter_length_ratio",
        # Legacy column names (may not exist but harmless to exclude)
        "round", "date_iso", "home_away", "sub_status",
        "team_goal_avg", "team_goals_total", "team_games_total",
        "Age", "Career Games (W-D-L W%)", "Career Goals (Ave.)",
        "team_games", "team_goals",
        # Physical feature intermediates (dob used to derive age_at_match)
        "dob",
    }

    # Accept any numeric/bool dtype (int8/16/32/64, float32/64, uint8, bool)
    FEATURE_COLS = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype.kind in ("f", "i", "u", "b")]

    print(f"\n  Total features: {len(FEATURE_COLS)}")
    print(f"  Dataset shape: {df.shape}")

    # Validate feature matrix
    from validate import validate_features, validate_temporal_integrity
    validate_features(df, FEATURE_COLS)
    validate_temporal_integrity(df, FEATURE_COLS)

    if save:
        # Downcast float64 → float32 for feature columns (halves memory + disk)
        for col in FEATURE_COLS:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        # Fix mixed-type columns that break parquet (e.g. bool + int from fillna)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(float)
                except (ValueError, TypeError):
                    pass

        config.ensure_dirs()
        out_path = config.FEATURES_DIR / "feature_matrix.parquet"
        df.to_parquet(out_path, index=False)

        # Save feature column list
        feat_list_path = config.FEATURES_DIR / "feature_columns.json"
        with open(feat_list_path, "w") as f:
            json.dump(FEATURE_COLS, f, indent=2)

        print(f"  Saved to {out_path}")
        print(f"  Feature matrix: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_features()
    print(f"\nFeature matrix: {df.shape}")
    print(f"\nFeature columns ({len(FEATURE_COLS)}):")
    for c in sorted(FEATURE_COLS):
        print(f"  {c}")
