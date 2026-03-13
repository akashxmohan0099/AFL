"""Monte Carlo simulation service for per-match analysis."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.data_loader import DataCache
from multi import MultiEngine, build_candidate_legs_from_predictions, overlay_real_odds


# Use fewer sims for API (speed vs accuracy trade-off)
API_N_SIMS = 5000


def _get_store(cache: DataCache):
    """Return the best available store (sequential preferred)."""
    return cache.sequential_store or cache.store


def get_match_simulation(
    match_id: int,
    n_sims: Optional[int] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    round_num: Optional[int] = None,
) -> Optional[dict]:
    """Run Monte Carlo simulation for a specific match.

    Supports two lookup modes:
      1. By match_id (completed games in matches.parquet)
      2. By home_team + away_team + round_num (upcoming games from predictions)

    Returns per-player probability distributions and match-level outcomes.
    """
    import config

    n_sims = n_sims or API_N_SIMS
    cache = DataCache.get()
    matches = cache.matches
    store = _get_store(cache)
    if store is None:
        return None

    year = getattr(config, "CURRENT_SEASON_YEAR", 2026)

    # Mode 1: lookup by match_id in matches.parquet
    found_in_matches = False
    if match_id and not matches.empty:
        mask = matches["match_id"] == match_id
        if mask.any():
            m = matches.loc[mask].iloc[0]
            year = int(m["year"])
            round_num = int(m["round_number"])
            home_team = str(m["home_team"])
            away_team = str(m["away_team"])
            found_in_matches = True

    # Mode 2: lookup by team names + round for upcoming games
    if not found_in_matches:
        if not home_team or not away_team or not round_num:
            return None

    preds = store.load_predictions(year=year, round_num=round_num)
    if preds.empty:
        return None

    # Filter predictions to this match
    if found_in_matches and "match_id" in preds.columns:
        match_preds = preds[preds["match_id"] == match_id]
        if match_preds.empty:
            match_preds = preds[preds["team"].isin([home_team, away_team])]
    else:
        # Team-based filter for upcoming games
        match_preds = preds[preds["team"].isin([home_team, away_team])]
        # Further filter by opponent to avoid pulling in wrong match data
        if "opponent" in match_preds.columns:
            match_preds = match_preds[
                match_preds["opponent"].isin([home_team, away_team])
            ]

    if match_preds.empty:
        return None

    # Load game predictions for margin/total
    game_preds = pd.DataFrame()
    try:
        gp = store.load_game_predictions(year=year, round_num=round_num)
        if not gp.empty:
            # Filter to this specific match
            if found_in_matches and "match_id" in gp.columns:
                game_preds = gp[gp["match_id"] == match_id]
            if game_preds.empty and "home_team" in gp.columns:
                game_preds = gp[
                    (gp["home_team"] == home_team) & (gp["away_team"] == away_team)
                ]
    except Exception:
        pass

    # Build candidates and match info
    # Pass match_id=None for team-based lookups so build_candidate_legs doesn't
    # filter on match_id=0 (which would match nothing).
    # Also pass None when predictions use placeholder IDs (negative) that differ
    # from the real match_id — otherwise the filter matches nothing.
    effective_match_id = match_id if found_in_matches else None
    if effective_match_id is not None and "match_id" in match_preds.columns:
        if effective_match_id not in match_preds["match_id"].values:
            effective_match_id = None
    candidates, match_info = build_candidate_legs_from_predictions(
        match_preds, game_preds_df=game_preds, match_id=effective_match_id
    )

    if "players_df" not in match_info:
        return None

    # Overlay real Betfair odds where available (graceful — works without odds data)
    player_odds = cache.player_odds
    if not player_odds.empty and candidates:
        try:
            overlay_real_odds(candidates, player_odds, match_teams=[home_team, away_team])
        except Exception:
            pass  # Odds overlay is optional

    # Run simulation
    engine = MultiEngine(n_sims=n_sims, seed=42)
    traces = engine.simulate_match(
        match_info["players_df"],
        match_info["home_team"],
        match_info["away_team"],
        match_info["predicted_margin"],
        match_info["predicted_total"],
        match_info["home_win_prob"],
    )

    # Build response
    result = {
        "match_id": match_id or 0,
        "home_team": home_team,
        "away_team": away_team,
        "n_sims": n_sims,
        "match_outcomes": _build_match_outcomes(traces),
        "players": _build_player_distributions(traces),
        "suggested_multis": _build_suggested_multis(engine, traces, candidates),
    }

    return result


def _build_match_outcomes(traces: dict) -> dict:
    """Extract match-level outcome distributions from simulation traces."""
    margin = traces["margin"]
    home_score = traces["home_score"]
    away_score = traces["away_score"]
    total = home_score + away_score

    home_wins = float((margin > 0).mean())
    away_wins = float((margin < 0).mean())
    draws = float((margin == 0).mean())

    return {
        "home_win_pct": round(home_wins, 4),
        "away_win_pct": round(away_wins, 4),
        "draw_pct": round(draws, 4),
        "avg_home_score": round(float(home_score.mean()), 1),
        "avg_away_score": round(float(away_score.mean()), 1),
        "avg_total": round(float(total.mean()), 1),
        "avg_margin": round(float(margin.mean()), 1),
        "score_distribution": {
            "home": _percentiles(home_score),
            "away": _percentiles(away_score),
            "total": _percentiles(total),
            "margin": _percentiles(margin),
        },
        "total_brackets": _total_brackets(total),
    }


def _percentiles(arr: np.ndarray) -> dict:
    """Compute p10/p25/p50/p75/p90 for an array."""
    return {
        "p10": round(float(np.percentile(arr, 10)), 1),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
    }


def _total_brackets(total: np.ndarray) -> list:
    """Probability of total score exceeding various thresholds."""
    brackets = []
    for threshold in [120, 130, 140, 150, 160, 170, 180, 190, 200]:
        p_over = float((total >= threshold).mean())
        brackets.append({"threshold": threshold, "p_over": round(p_over, 4)})
    return brackets


def _build_player_distributions(traces: dict) -> list:
    """Extract per-player probability distributions from simulation traces."""
    players = []

    for p in traces["players"]:
        goals = p["goals"]
        disp = p["disp"]
        marks = p["marks"]

        # Goals distribution: P(exactly k) for k=0..4+
        max_goal = min(int(goals.max()), 8)
        goal_dist = []
        for k in range(max_goal + 1):
            goal_dist.append(round(float((goals == k).mean()), 4))
        # Accumulate remainder into last bucket
        if max_goal < 8:
            goal_dist.append(round(float((goals > max_goal).mean()), 4))

        player_data = {
            "player": p["player"],
            "team": p["team"],
            "is_home": p["is_home"],
            "goals": {
                "avg": round(float(goals.mean()), 2),
                "p_1plus": round(float((goals >= 1).mean()), 4),
                "p_2plus": round(float((goals >= 2).mean()), 4),
                "p_3plus": round(float((goals >= 3).mean()), 4),
                "distribution": goal_dist,
            },
            "disposals": {
                "avg": round(float(disp.mean()), 1),
                "p_10plus": round(float((disp >= 10).mean()), 4),
                "p_15plus": round(float((disp >= 15).mean()), 4),
                "p_20plus": round(float((disp >= 20).mean()), 4),
                "p_25plus": round(float((disp >= 25).mean()), 4),
                "p_30plus": round(float((disp >= 30).mean()), 4),
                "percentiles": _percentiles(disp),
            },
            "marks": {
                "avg": round(float(marks.mean()), 1),
                "p_3plus": round(float((marks >= 3).mean()), 4),
                "p_5plus": round(float((marks >= 5).mean()), 4),
                "p_7plus": round(float((marks >= 7).mean()), 4),
                "p_10plus": round(float((marks >= 10).mean()), 4),
                "percentiles": _percentiles(marks.astype(np.float32)),
            },
        }
        players.append(player_data)

    # Sort: home players first, then by predicted goals desc
    players.sort(key=lambda x: (not x["is_home"], -x["goals"]["avg"]))

    return players


def _build_suggested_multis(engine: MultiEngine, traces: dict, candidates: list) -> list:
    """Find top multi-bet suggestions from simulation."""
    if not candidates:
        return []

    best = engine.find_best_multis(traces, candidates, min_edge=0.01, top_n=10)

    suggestions = []
    for combo in best:
        suggestions.append({
            "legs": combo["legs"],
            "n_legs": combo["n_legs"],
            "joint_prob": combo["joint_prob"],
            "indep_prob": combo["indep_prob"],
            "correlation_lift": combo["correlation_lift"],
        })

    return suggestions


def get_round_simulations(year: int, round_num: int, n_sims: int = API_N_SIMS) -> list[dict]:
    """Run lightweight simulations for all matches in a round.

    Returns a list of compact simulation summaries (no per-player data).
    """
    import config

    cache = DataCache.get()
    store = cache.sequential_store or cache.store
    if store is None:
        return []

    preds = store.load_predictions(year=year, round_num=round_num)
    if preds.empty:
        return []

    game_preds = pd.DataFrame()
    try:
        game_preds = store.load_game_predictions(year=year, round_num=round_num)
    except Exception:
        pass

    # Identify distinct matches by (team, opponent) pairs
    if "team" not in preds.columns or "opponent" not in preds.columns:
        return []

    # Get unique home teams from fixture file (authoritative source for home/away)
    fixture_path = Path(config.DATA_DIR) / "fixtures" / f"round_{round_num}_{year}.csv"
    if fixture_path.exists():
        fix_df = pd.read_csv(fixture_path)
        home_teams = fix_df[fix_df["is_home"] == 1]["team"].unique()
    elif "is_home" in preds.columns:
        home_mask = preds["is_home"].astype(bool)
        home_teams = preds.loc[home_mask, "team"].unique()
    else:
        # Deduplicate: for each team pair, only take the first one alphabetically
        seen = set()
        home_teams_list = []
        for _, row in preds[["team", "opponent"]].drop_duplicates().iterrows():
            pair = tuple(sorted([row["team"], row["opponent"]]))
            if pair not in seen:
                seen.add(pair)
                home_teams_list.append(row["team"])
        home_teams = np.array(home_teams_list)

    results = []
    for ht in home_teams:
        ht_rows = preds[preds["team"] == ht]
        if ht_rows.empty:
            continue
        at = ht_rows["opponent"].iloc[0]
        match_preds = preds[
            (preds["team"].isin([ht, at])) & (preds["opponent"].isin([ht, at]))
        ]
        if match_preds.empty:
            continue

        # Get game prediction for this match
        match_game_preds = pd.DataFrame()
        if not game_preds.empty and "home_team" in game_preds.columns:
            match_game_preds = game_preds[
                (game_preds["home_team"] == ht) & (game_preds["away_team"] == at)
            ]

        try:
            candidates, match_info = build_candidate_legs_from_predictions(
                match_preds, game_preds_df=match_game_preds, match_id=None
            )
            if "players_df" not in match_info:
                continue

            engine = MultiEngine(n_sims=n_sims, seed=42)
            traces = engine.simulate_match(
                match_info["players_df"],
                match_info["home_team"],
                match_info["away_team"],
                match_info["predicted_margin"],
                match_info["predicted_total"],
                match_info["home_win_prob"],
            )

            margin = traces["margin"]
            home_score = traces["home_score"]
            away_score = traces["away_score"]
            total = home_score + away_score

            # Top 3 goal scorers from simulation
            player_traces = traces["players"]
            top_scorers = sorted(
                player_traces,
                key=lambda p: float((p["goals"] >= 1).mean()),
                reverse=True,
            )[:3]

            results.append({
                "home_team": ht,
                "away_team": at,
                "n_sims": n_sims,
                "home_win_pct": round(float((margin > 0).mean()), 4),
                "away_win_pct": round(float((margin < 0).mean()), 4),
                "draw_pct": round(float((margin == 0).mean()), 4),
                "avg_total": round(float(total.mean()), 1),
                "avg_margin": round(float(margin.mean()), 1),
                "median_total": round(float(np.median(total)), 1),
                "score_range": {
                    "home": _percentiles(home_score),
                    "away": _percentiles(away_score),
                },
                "top_scorers": [
                    {
                        "player": p["player"],
                        "team": p["team"],
                        "p_1plus": round(float((p["goals"] >= 1).mean()), 4),
                        "avg_goals": round(float(p["goals"].mean()), 2),
                    }
                    for p in top_scorers
                ],
            })
        except Exception:
            continue

    return results
