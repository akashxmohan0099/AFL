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
    These columns already exist: age_years, career_games, career_goal_avg, etc.
    We add a few derived ones."""
    df = df.copy()

    # Age squared (captures non-linear peak-years effect ~25-29)
    df["age_squared"] = df["age_years"] ** 2

    # Is first year at current club
    df["is_first_year_at_club"] = (
        df["team_games_total"].fillna(0) <= 22
    ).astype(int)

    # Career accuracy (avoid div-by-zero)
    career_total_shots = df["career_goals_total"].fillna(0)
    # We don't have career behinds directly, so use goal_avg as proxy
    # career_goal_avg is goals per game — higher = more prolific

    return df


# ---------------------------------------------------------------------------
# B. Recency-Weighted Rolling Averages
# ---------------------------------------------------------------------------

def add_rolling_features(df):
    """Compute weighted rolling averages for player form.

    For each window in ROLLING_WINDOWS, compute the mean of the last N
    matches for key stat columns. Uses shift(1) so the current match
    is never included in its own features.
    """
    df = df.sort_values(["player", "team", "date_iso"]).copy()

    # Columns to compute rolling averages for
    roll_cols = {
        "GL": "gl", "BH": "bh", "DI": "di", "MK": "mk", "TK": "tk",
        "IF": "if50", "CL": "cl", "HO": "ho", "GA": "ga", "MI": "mi",
        "CM": "cm", "CP": "cp", "FF": "ff", "RB": "rb", "one_pct": "one_pct",
    }

    # Group by (player, team) compound key
    grouped = df.groupby(["player", "team"])

    for window in config.ROLLING_WINDOWS:
        for src_col, feat_name in roll_cols.items():
            col_name = f"player_{feat_name}_avg_{window}"
            df[col_name] = grouped[src_col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    # Rolling accuracy: GL / (GL + BH) over last N matches
    for window in config.ROLLING_WINDOWS:
        gl_sum = grouped["GL"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        bh_sum = grouped["BH"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        total = gl_sum + bh_sum
        df[f"player_accuracy_{window}"] = np.where(total > 0, gl_sum / total, 0.5)

    # Scoring streak: consecutive matches with GL >= 1
    def _streak(series):
        """Count consecutive non-zero values from the end (shifted)."""
        shifted = series.shift(1)
        result = pd.Series(0, index=series.index, dtype=int)
        streak = 0
        for idx in shifted.index:
            val = shifted.loc[idx]
            if pd.isna(val):
                streak = 0
            elif val >= 1:
                streak += 1
            else:
                streak = 0
            result.loc[idx] = streak
        return result

    df["player_gl_streak"] = grouped["GL"].transform(_streak)

    # Trend: linear slope of GL over last 5 matches
    def _trend_slope(series):
        shifted = series.shift(1)
        result = pd.Series(0.0, index=series.index)
        vals = []
        for idx in shifted.index:
            v = shifted.loc[idx]
            if pd.notna(v):
                vals.append(v)
            if len(vals) >= 5:
                vals = vals[-5:]
            if len(vals) >= 3:
                x = np.arange(len(vals))
                slope = np.polyfit(x, vals, 1)[0]
                result.loc[idx] = slope
        return result

    df["player_gl_trend_5"] = grouped["GL"].transform(_trend_slope)

    # Days since last match
    df["days_since_last_match"] = grouped["date_iso"].transform(
        lambda s: s.diff().dt.days
    )
    df["is_returning_from_break"] = (
        df["days_since_last_match"].fillna(0) > 21
    ).astype(int)

    # Season-to-date goals
    season_group = df.groupby(["player", "team", "year"])
    df["season_goals_total"] = season_group["GL"].transform(
        lambda s: s.shift(1).expanding().sum()
    ).fillna(0)

    return df


# ---------------------------------------------------------------------------
# C. Venue Features
# ---------------------------------------------------------------------------

def add_venue_features(df):
    """Player performance at specific venues vs their overall average."""
    df = df.sort_values(["player", "team", "date_iso"]).copy()

    # Build lookup: (player, team, venue) → historical stats
    # We need to compute these as expanding means up to (but not including)
    # the current match, for each (player, team, venue) group.

    venue_group = df.groupby(["player", "team", "venue"])

    df["player_gl_at_venue_avg"] = venue_group["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["player_bh_at_venue_avg"] = venue_group["BH"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    # Player's overall GL average (for computing venue diff)
    player_group = df.groupby(["player", "team"])
    overall_gl_avg = player_group["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    df["player_gl_venue_diff"] = df["player_gl_at_venue_avg"] - overall_gl_avg

    # Venue average goals per team (regardless of player)
    # — reflects ground dimensions / conditions
    match_team_goals = (
        df.groupby(["match_id", "team", "venue"])["GL"]
        .sum()
        .reset_index()
        .rename(columns={"GL": "team_total_goals"})
    )
    venue_avg = (
        match_team_goals.groupby("venue")["team_total_goals"]
        .mean()
        .reset_index()
        .rename(columns={"team_total_goals": "venue_avg_goals_per_team"})
    )
    df = df.merge(venue_avg, on="venue", how="left")

    # Fill NaN venue features
    for col in ["player_gl_at_venue_avg", "player_bh_at_venue_avg",
                "player_gl_venue_diff", "venue_avg_goals_per_team"]:
        df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# D. Opponent Defense Features
# ---------------------------------------------------------------------------

def add_opponent_features(df):
    """How leaky is the opponent's defense? Computed from team-level aggregates."""
    df = df.sort_values(["date_iso"]).copy()

    # Step 1: compute goals conceded per match per team
    # Goals conceded by team X = goals scored by X's opponent in that match
    team_match_goals = (
        df.groupby(["match_id", "team", "opponent"])["GL"]
        .sum()
        .reset_index()
        .rename(columns={"GL": "team_goals_scored"})
    )

    # Goals conceded by opponent = team_goals_scored
    goals_conceded = team_match_goals.rename(columns={
        "team": "conceding_team",
        "opponent": "team",
        "team_goals_scored": "goals_conceded",
    })[["match_id", "conceding_team", "goals_conceded"]]

    # Merge match date for sorting
    match_dates = df[["match_id", "date_iso"]].drop_duplicates()
    goals_conceded = goals_conceded.merge(match_dates, on="match_id", how="left")
    goals_conceded = goals_conceded.sort_values("date_iso")

    # Rolling averages of goals conceded
    gc_group = goals_conceded.groupby("conceding_team")
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
    opp_group = df.groupby(["player", "team", "opponent"])
    df["player_vs_opp_gl_avg"] = opp_group["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["player_vs_opp_games"] = opp_group["GL"].transform(
        lambda s: s.shift(1).expanding().count()
    )

    # Diff: player's avg vs opponent minus their overall avg
    overall = df.groupby(["player", "team"])["GL"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["player_vs_opp_gl_diff"] = df["player_vs_opp_gl_avg"] - overall

    # Mask out matchup features where sample too small
    mask = df["player_vs_opp_games"].fillna(0) < config.MATCHUP_MIN_MATCHES
    df.loc[mask, "player_vs_opp_gl_diff"] = 0
    df.loc[mask, "player_vs_opp_gl_avg"] = np.nan

    # Fill NaNs
    for col in ["opp_goals_conceded_avg_5", "opp_goals_conceded_avg_10",
                "player_vs_opp_gl_avg", "player_vs_opp_gl_diff"]:
        df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# E. Defender Matchup Features
# ---------------------------------------------------------------------------

def add_matchup_features(df):
    """Estimate forward-vs-defender interactions.

    For each match, we know the full opponent lineup from player_stats.
    We build a matchup matrix:
      - For forward F vs opponent team, compute F's goal avg when
        defender D was in the opposition lineup vs when D was absent.
      - The matchup adjustment is the weighted average of these effects
        for defenders expected in the upcoming match.
    """
    df = df.sort_values(["date_iso", "match_id"]).copy()

    # Build match lineup lookup: match_id → {team: set of players}
    lineup_lookup = (
        df.groupby(["match_id", "team"])["player"]
        .apply(set)
        .reset_index()
        .rename(columns={"player": "lineup"})
    )
    match_lineups = {}
    for _, row in lineup_lookup.iterrows():
        mid = row["match_id"]
        if mid not in match_lineups:
            match_lineups[mid] = {}
        match_lineups[mid][row["team"]] = row["lineup"]

    # Identify key defenders: low goal output, high RB + one_pct
    player_career = (
        df.groupby(["player", "team"])
        .agg(
            gl_avg=("GL", "mean"),
            rb_avg=("RB", "mean"),
            one_pct_avg=("one_pct", "mean"),
            games=("match_id", "nunique"),
        )
        .reset_index()
    )
    defenders = player_career[
        (player_career["gl_avg"] < 0.3)
        & (player_career["rb_avg"] >= config.KEY_DEFENDER_RB_THRESHOLD)
        & (player_career["one_pct_avg"] >= config.KEY_DEFENDER_ONE_PCT_THRESHOLD)
        & (player_career["games"] >= config.MIN_PLAYER_MATCHES)
    ][["player", "team"]].copy()
    defender_set = set(zip(defenders["player"], defenders["team"]))

    # For each match, count key defenders in opponent lineup and compute
    # aggregate defender strength score
    if not defender_set:
        # No defenders found (small dataset) — return zeros
        df["opp_key_defenders_count"] = 0
        df["opp_defender_strength_score"] = 0.0
        return df

    defender_strength = player_career[
        player_career.set_index(["player", "team"]).index.isin(
            pd.MultiIndex.from_tuples(defender_set)
        )
    ].set_index(["player", "team"])

    opp_def_count = []
    opp_def_strength = []

    for _, row in df.iterrows():
        mid = row["match_id"]
        opp = row["opponent"]
        opp_lineup = match_lineups.get(mid, {}).get(opp, set())

        count = 0
        strength = 0.0
        for p in opp_lineup:
            if (p, opp) in defender_set:
                count += 1
                if (p, opp) in defender_strength.index:
                    ds = defender_strength.loc[(p, opp)]
                    strength += ds["one_pct_avg"] + ds["rb_avg"]
        opp_def_count.append(count)
        opp_def_strength.append(strength)

    df["opp_key_defenders_count"] = opp_def_count
    df["opp_defender_strength_score"] = opp_def_strength

    return df


# ---------------------------------------------------------------------------
# F. Team Context Features
# ---------------------------------------------------------------------------

def add_team_features(df):
    """Team-level form and player's role within the team attack."""
    df = df.sort_values(["date_iso"]).copy()

    # Team goals per match
    team_match = (
        df.groupby(["match_id", "team"])
        .agg(team_total_goals=("GL", "sum"), team_total_behinds=("BH", "sum"),
             team_total_if=("IF", "sum"))
        .reset_index()
    )
    match_dates = df[["match_id", "date_iso"]].drop_duplicates()
    team_match = team_match.merge(match_dates, on="match_id", how="left")
    team_match = team_match.sort_values("date_iso")

    # Rolling team goals
    tg = team_match.groupby("team")
    for window in [5, 10]:
        team_match[f"team_goals_avg_{window}"] = tg["team_total_goals"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    team_match["team_if_avg_5"] = tg["team_total_if"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Join back
    team_cols = [c for c in team_match.columns
                 if c not in ["date_iso", "team_total_goals",
                              "team_total_behinds", "team_total_if"]]
    df = df.merge(
        team_match[team_cols + ["team_total_goals"]],
        on=["match_id", "team"],
        how="left",
    )

    # Player goal share: player's goals / team's goals (last 5 matches)
    player_group = df.groupby(["player", "team"])
    player_gl_5 = player_group["GL"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    team_gl_5_per_player = df.groupby(["player", "team"])["team_total_goals"].transform(
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
        df.groupby(["match_id", "team"])["GL"].sum().reset_index()
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
    match_results = match_results.sort_values("date_iso")
    match_results["team_win_pct_5"] = (
        match_results.groupby("team")["won"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    match_results["team_margin_avg_5"] = (
        match_results.groupby("team")
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
                "player_goal_share_5", "team_win_pct_5", "team_margin_avg_5"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

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
    grouped = df.groupby(["player", "team"])

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
    """
    df = df.sort_values(["player", "team", "date_iso"]).copy()
    grouped = df.groupby(["player", "team"])

    # Compute recent averages for classification
    for col in ["GL", "HO", "DI", "CL", "RB", "one_pct", "IF", "MI"]:
        df[f"_recent_{col}"] = grouped[col].transform(
            lambda s: s.rolling(10, min_periods=3).mean()
        )

    conditions = [
        df["_recent_HO"] >= config.RUCK_HO_THRESHOLD,
        (df["_recent_GL"] >= config.KEY_FORWARD_GL_THRESHOLD) &
        (df["_recent_MI"] >= config.KEY_FORWARD_MI_THRESHOLD),
        (df["_recent_GL"] >= config.SMALL_FORWARD_GL_THRESHOLD) &
        (df["_recent_IF"] >= config.SMALL_FORWARD_IF_THRESHOLD),
        (df["_recent_RB"] >= config.KEY_DEFENDER_RB_THRESHOLD) &
        (df["_recent_one_pct"] >= config.KEY_DEFENDER_ONE_PCT_THRESHOLD),
        (df["_recent_DI"] >= config.MIDFIELDER_DI_THRESHOLD) &
        (df["_recent_CL"] >= config.MIDFIELDER_CL_THRESHOLD),
    ]
    choices = ["ruck", "key_forward", "small_forward", "key_defender", "midfielder"]
    df["player_role"] = np.select(conditions, choices, default="general")

    # One-hot encode role for model
    for role in choices + ["general"]:
        df[f"role_{role}"] = (df["player_role"] == role).astype(int)

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.startswith("_recent_")]
    df = df.drop(columns=temp_cols)

    return df


# ---------------------------------------------------------------------------
# I. Teammate Enabler Features
# ---------------------------------------------------------------------------

def add_enabler_features(df):
    """Compute how many of a player's top enablers are in the current lineup.

    An 'enabler' is a teammate whose presence correlates with higher goal
    output for this player. We identify top-N enablers per (player, team)
    by comparing the player's GL average when the enabler plays vs doesn't.
    """
    df = df.sort_values(["date_iso"]).copy()

    # Build match lineup lookup
    lineup_lookup = (
        df.groupby(["match_id", "team"])["player"]
        .apply(set)
        .to_dict()
    )

    # For computational efficiency, only compute enablers for players
    # with enough games
    player_games = df.groupby(["player", "team"])["match_id"].nunique()
    eligible = player_games[player_games >= config.MIN_PLAYER_MATCHES].index

    # Pre-compute: for each (player, team), for each teammate, compute
    # the player's average GL when teammate is present vs absent
    enabler_map = {}  # (player, team) → [list of top enabler names]

    for (player, team) in eligible:
        player_df = df[(df["player"] == player) & (df["team"] == team)]
        if len(player_df) < config.MIN_PLAYER_MATCHES:
            continue

        # Get all teammates who played with this player
        teammate_effects = {}
        match_ids = player_df["match_id"].unique()

        for mid in match_ids:
            teammates = lineup_lookup.get((mid, team), set()) - {player}
            gl = player_df.loc[player_df["match_id"] == mid, "GL"].iloc[0]
            for tm in teammates:
                if tm not in teammate_effects:
                    teammate_effects[tm] = {"with": [], "total": []}
                teammate_effects[tm]["with"].append(gl)

        # Compute effect: avg GL with teammate - overall avg GL
        overall_avg = player_df["GL"].mean()
        effects = []
        for tm, data in teammate_effects.items():
            if len(data["with"]) >= 3:
                with_avg = np.mean(data["with"])
                effects.append((tm, with_avg - overall_avg))

        # Top-N enablers
        effects.sort(key=lambda x: x[1], reverse=True)
        top = [e[0] for e in effects[:config.ENABLER_COUNT]]
        enabler_map[(player, team)] = top

    # Now count how many enablers are present in each match
    enabler_counts = []
    for _, row in df.iterrows():
        player = row["player"]
        team = row["team"]
        mid = row["match_id"]
        enablers = enabler_map.get((player, team), [])
        lineup = lineup_lookup.get((mid, team), set())
        count = sum(1 for e in enablers if e in lineup)
        enabler_counts.append(count)

    df["teammate_enabler_count"] = enabler_counts

    return df


# ---------------------------------------------------------------------------
# J. Sample Weights
# ---------------------------------------------------------------------------

def add_sample_weights(df):
    """Add sample_weight column for model training.
    Combines era weight * time decay."""
    df = df.copy()
    now = df["date_iso"].max()

    days_ago = (now - df["date_iso"]).dt.days.fillna(9999)
    era_w = df["year"].apply(_era_weight)
    decay_w = days_ago.apply(_decay_weight)

    df["sample_weight"] = era_w * decay_w
    # Normalize so mean weight ≈ 1
    mean_w = df["sample_weight"].mean()
    if mean_w > 0:
        df["sample_weight"] = df["sample_weight"] / mean_w

    return df


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

# Feature columns that the model will train on (assembled after all features added)
FEATURE_COLS = None  # Set dynamically after build

def build_features(df=None, data_dir=None, save=True):
    """Main entry point. Takes cleaned player_matches DataFrame
    and returns it with all features added.

    If df is None, loads from cleaned parquet.
    """
    if df is None:
        cleaned_path = config.CLEANED_DIR / "player_matches.parquet"
        if cleaned_path.exists():
            df = pd.read_parquet(cleaned_path)
        else:
            from clean import build_player_matches
            df = build_player_matches(data_dir=data_dir, save=True)

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

    # J. Sample weights
    print("  [J] Sample weights...")
    df = add_sample_weights(df)

    # Assemble feature column list
    global FEATURE_COLS
    exclude_cols = {
        # Identifiers
        "match_id", "year", "round", "round_number", "venue", "date",
        "date_iso", "team", "opponent", "home_away", "jumper", "player",
        "sub_status", "player_role",
        # Targets
        "GL", "BH",
        # Quarter scoring (raw — we use the derived features instead)
        "q1_goals", "q1_behinds", "q2_goals", "q2_behinds",
        "q3_goals", "q3_behinds", "q4_goals", "q4_behinds",
        # Weight
        "sample_weight",
        # Raw career strings
        "Age", "Career Games (W-D-L W%)", "Career Goals (Ave.)",
        "team_games", "team_goals",
    }
    FEATURE_COLS = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in [np.float64, np.int64, np.float32,
                                        np.int32, np.uint8, int, float, bool]]

    print(f"\n  Total features: {len(FEATURE_COLS)}")
    print(f"  Dataset shape: {df.shape}")

    if save:
        config.ensure_dirs()
        out_path = config.FEATURES_DIR / "feature_matrix.parquet"
        df.to_parquet(out_path, index=False)

        # Save feature column list
        import json
        feat_list_path = config.FEATURES_DIR / "feature_columns.json"
        with open(feat_list_path, "w") as f:
            json.dump(FEATURE_COLS, f, indent=2)

        print(f"  Saved to {out_path}")

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
