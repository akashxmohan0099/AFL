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


# ---------------------------------------------------------------------------
# A. Career / Age Features
# ---------------------------------------------------------------------------

def add_career_features(df):
    """Add features derived from player_details (already joined in clean.py).
    These columns already exist: age_years, career_games_pre, career_goal_avg_pre, etc.
    We add a few derived ones."""
    df = df.copy()

    # Age squared (captures non-linear peak-years effect ~25-29)
    df["age_squared"] = df["age_years"] ** 2

    # Capped career goal avg to reduce outlier influence from prolific forwards
    # Uses pre-game version to avoid same-game leakage
    if "career_goal_avg_pre" in df.columns:
        df["career_goal_avg_capped"] = df["career_goal_avg_pre"].clip(upper=2.5)

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
    ewm_cols = {"GL": "gl", "BH": "bh", "MI": "mi", "IF": "if50", "DI": "di", "MK": "mk"}
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
    df_real["player_gl_trend_5"] = grouped["GL"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )
    ).fillna(0)

    # Days since last match (computed on real games only — skip DNP gaps)
    df_real["days_since_last_match"] = grouped["date"].transform(
        lambda s: s.diff().dt.days
    )
    df_real["is_returning_from_break"] = (
        df_real["days_since_last_match"].fillna(0) > 21
    ).astype(int)

    # Season-to-date goals
    season_group = df_real.groupby(["player", "team", "year"], observed=True)
    df_real["season_goals_total"] = season_group["GL"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    # --- Disposal-specific features ---
    df_real["player_di_volatility_5"] = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)

    df_real["player_di_trend_5"] = grouped["DI"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )
    ).fillna(0)

    # Kick/handball ratio
    for window in [3, 5]:
        ki_sum = grouped["KI"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).sum()
        )
        hb_sum = grouped["HB"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).sum()
        )
        total_kh = ki_sum + hb_sum
        df_real[f"player_ki_hb_ratio_{window}"] = np.where(total_kh > 0, ki_sum / total_kh, 0.5)

    # Season-to-date disposals
    df_real["season_disposals_total"] = season_group["DI"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    # Season-to-date rate aggregations
    if "GL_rate" in df_real.columns:
        df_real["season_goals_rate_avg"] = season_group["GL_rate"].transform(
            lambda s: s.shift(1).expanding().mean()
        ).fillna(0)
    if "DI_rate" in df_real.columns:
        df_real["season_disposals_rate_avg"] = season_group["DI_rate"].transform(
            lambda s: s.shift(1).expanding().mean()
        ).fillna(0)

    # Rate-based volatility and trend for goals
    if "GL_rate" in df_real.columns:
        df_real["player_gl_rate_volatility_5"] = grouped["GL_rate"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).std()
        ).fillna(0)
        df_real["player_gl_rate_trend_5"] = grouped["GL_rate"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
            )
        ).fillna(0)

    # Rate-based volatility and trend for disposals
    if "DI_rate" in df_real.columns:
        df_real["player_di_rate_volatility_5"] = grouped["DI_rate"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).std()
        ).fillna(0)
        df_real["player_di_rate_trend_5"] = grouped["DI_rate"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
            )
        ).fillna(0)

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
        goals_conceded[f"goals_conceded_avg_{window}"] = gc_group[
            "goals_conceded"
        ].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
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
    df["player_vs_opp_gl_avg"] = opp_group["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["player_vs_opp_games"] = opp_group["GL"].transform(
        lambda s: s.shift(1).expanding().count()
    )

    # Diff: player's avg vs opponent minus their overall avg
    overall = df.groupby(["player", "team"], observed=True)["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["player_vs_opp_gl_diff"] = df["player_vs_opp_gl_avg"] - overall

    # Mask out matchup features where sample too small
    mask = df["player_vs_opp_games"].fillna(0) < config.MATCHUP_MIN_MATCHES
    df.loc[mask, "player_vs_opp_gl_diff"] = 0
    df.loc[mask, "player_vs_opp_gl_avg"] = np.nan

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
        disp_conceded[f"disp_conceded_avg_{window}"] = dc_group[
            "disp_conceded"
        ].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
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
    team_match_cp["cp_diff_avg_5"] = cp_group["cp_diff"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
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

    # Opponent contested possession differential: 0 = neutral
    df["opp_contested_poss_diff_5"] = df["opp_contested_poss_diff_5"].fillna(0)

    # For player vs opponent: fall back to player's overall career average
    if "career_goal_avg_pre" in df.columns:
        df["player_vs_opp_gl_avg"] = df["player_vs_opp_gl_avg"].fillna(
            df["career_goal_avg_pre"]
        )
    else:
        df["player_vs_opp_gl_avg"] = df["player_vs_opp_gl_avg"].fillna(0)
    df["player_vs_opp_gl_diff"] = df["player_vs_opp_gl_diff"].fillna(0)

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
        .agg(team_total_goals=("GL", "sum"), team_total_behinds=("BH", "sum"),
             team_total_if=("IF", "sum"), team_total_cl=("CL", "sum"))
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
                              "team_total_behinds", "team_total_if",
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
    df["team_clearance_dominance_5"] = np.where(
        df["opp_cl_avg_5"].fillna(0) > 0,
        df["team_cl_avg_5"].fillna(0) / df["opp_cl_avg_5"],
        1.0,
    )
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
    for col in ["team_goals_avg_5", "team_goals_avg_10", "team_if_avg_5",
                "team_cl_avg_5", "team_clearance_dominance_5",
                "team_mid_quality_score",
                "player_goal_share_5", "team_win_pct_5", "team_margin_avg_5"]:
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

        df = df.drop(columns=["opp_margin_avg_5"], errors="ignore")

    # Fill NaN
    for col in ["venue_is_indoor", "game_pace_proxy",
                "expected_margin_diff", "expected_margin_abs"]:
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
    match_counts = df.groupby("match_id")[["is_night_game", "is_twilight_game"]].first()
    print(f"    Day/night: {(match_counts['is_night_game']==0).sum() - (match_counts['is_twilight_game']==1).sum()} day, "
          f"{(match_counts['is_twilight_game']==1).sum()} twilight, "
          f"{(match_counts['is_night_game']==1).sum()} night "
          f"(of {len(match_counts)} matches)")

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

def add_sample_weights(df):
    """Add sample_weight column for model training.
    Combines era weight * time decay."""
    df = df.copy()
    now = df["date"].max()

    days_ago = (now - df["date"]).dt.days.fillna(9999)
    era_w = df["year"].apply(_era_weight)
    decay_w = days_ago.apply(_decay_weight)

    df["sample_weight"] = era_w * decay_w
    # Normalize so mean weight ≈ 1
    mean_w = df["sample_weight"].mean()
    if mean_w > 0:
        df["sample_weight"] = df["sample_weight"] / mean_w

    return df


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
    if save and df is None and cached_path.exists() and base_path.exists():
        cache_mtime = cached_path.stat().st_mtime
        sources_fresh = cache_mtime > base_path.stat().st_mtime
        if weather_path.exists():
            sources_fresh = sources_fresh and cache_mtime > weather_path.stat().st_mtime
        if dims_path.exists():
            sources_fresh = sources_fresh and cache_mtime > dims_path.stat().st_mtime
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
        "humidity_discomfort", "player_mk_avg_5",
        # ── Redundant features (r=±1.0 with era_4, keep era_4 as representative) ──
        "is_covid_season", "quarter_length_ratio",
        # Legacy column names (may not exist but harmless to exclude)
        "round", "date_iso", "home_away", "sub_status",
        "team_goal_avg", "team_goals_total", "team_games_total",
        "Age", "Career Games (W-D-L W%)", "Career Goals (Ave.)",
        "team_games", "team_goals",
    }

    # Accept any numeric/bool dtype (int8/16/32/64, float32/64, uint8, bool)
    FEATURE_COLS = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype.kind in ("f", "i", "u", "b")]

    print(f"\n  Total features: {len(FEATURE_COLS)}")
    print(f"  Dataset shape: {df.shape}")

    # Validate feature matrix
    from validate import validate_features
    validate_features(df, FEATURE_COLS)

    if save:
        # Downcast float64 → float32 for feature columns (halves memory + disk)
        for col in FEATURE_COLS:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

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
