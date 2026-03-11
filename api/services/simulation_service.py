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


def get_match_simulation(match_id: int, n_sims: Optional[int] = None) -> Optional[dict]:
    """Run Monte Carlo simulation for a specific match.

    Returns per-player probability distributions and match-level outcomes.
    """
    n_sims = n_sims or API_N_SIMS
    cache = DataCache.get()
    matches = cache.matches
    if matches.empty:
        return None

    mask = matches["match_id"] == match_id
    if not mask.any():
        return None

    m = matches.loc[mask].iloc[0]
    year = int(m["year"])
    round_num = int(m["round_number"])
    home_team = str(m["home_team"])
    away_team = str(m["away_team"])

    # Load predictions
    store = _get_store(cache)
    if store is None:
        return None

    preds = store.load_predictions(year=year, round_num=round_num)
    if preds.empty:
        return None

    # Filter to this match
    if "match_id" in preds.columns:
        match_preds = preds[preds["match_id"] == match_id]
        if match_preds.empty:
            # Fallback: filter by teams
            match_preds = preds[
                (preds["team"].isin([home_team, away_team]))
            ]
    else:
        match_preds = preds[preds["team"].isin([home_team, away_team])]

    if match_preds.empty:
        return None

    # Load game predictions for margin/total
    game_preds = pd.DataFrame()
    try:
        game_preds = store.load_game_predictions(year=year, round_num=round_num)
    except Exception:
        pass

    # Build candidates and match info
    candidates, match_info = build_candidate_legs_from_predictions(
        match_preds, game_preds_df=game_preds, match_id=match_id
    )

    if "players_df" not in match_info:
        return None

    # Overlay real Betfair odds where available (graceful — works without odds data)
    player_odds = cache.player_odds
    if not player_odds.empty:
        n_updated = overlay_real_odds(candidates, player_odds, match_teams=[home_team, away_team])
    else:
        n_updated = 0

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
        "match_id": match_id,
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
