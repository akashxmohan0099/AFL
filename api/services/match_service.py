"""Match and game winner prediction services."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from api.data_loader import DataCache


def _get_store(cache: DataCache):
    """Return the best available store (sequential preferred, then learning)."""
    return cache.sequential_store or cache.store


def get_matches_for_round(year: int, round_num: int) -> list[dict]:
    cache = DataCache.get()
    matches = cache.matches
    if matches.empty:
        return []

    mask = (matches["year"] == year) & (matches["round_number"] == round_num)
    round_matches = matches.loc[mask]

    # Load game predictions if available
    game_preds = pd.DataFrame()
    store = _get_store(cache)
    if store is not None:
        game_preds = store.load_game_predictions(year=year, round_num=round_num)

    results = []
    for _, m in round_matches.iterrows():
        entry = {
            "match_id": int(m["match_id"]),
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "venue": m.get("venue", ""),
            "date": str(m["date"])[:10] if pd.notna(m.get("date")) else None,
            "home_score": int(m["home_score"]) if pd.notna(m.get("home_score")) else None,
            "away_score": int(m["away_score"]) if pd.notna(m.get("away_score")) else None,
        }

        # Add game winner predictions
        if not game_preds.empty:
            gp_mask = game_preds["match_id"] == m["match_id"] if "match_id" in game_preds.columns else pd.Series(False, index=game_preds.index)
            if not gp_mask.any() and "home_team" in game_preds.columns:
                gp_mask = (game_preds["home_team"] == m["home_team"]) & (game_preds["away_team"] == m["away_team"])
            gp = game_preds.loc[gp_mask]
            if len(gp) > 0:
                row = gp.iloc[0]
                if "home_win_prob" in row.index and pd.notna(row["home_win_prob"]):
                    entry["home_win_prob"] = round(float(row["home_win_prob"]), 4)
                    entry["away_win_prob"] = round(1 - float(row["home_win_prob"]), 4)
                    entry["predicted_winner"] = m["home_team"] if row["home_win_prob"] > 0.5 else m["away_team"]

        # Determine actual winner
        if entry.get("home_score") is not None and entry.get("away_score") is not None:
            if entry["home_score"] > entry["away_score"]:
                entry["actual_winner"] = m["home_team"]
            elif entry["away_score"] > entry["home_score"]:
                entry["actual_winner"] = m["away_team"]
            else:
                entry["actual_winner"] = "Draw"
            if entry.get("predicted_winner"):
                entry["correct"] = entry["predicted_winner"] == entry["actual_winner"]

        results.append(entry)

    return results


def get_match_detail(match_id: int) -> dict | None:
    cache = DataCache.get()
    matches = cache.matches
    if matches.empty:
        return None

    mask = matches["match_id"] == match_id
    if not mask.any():
        return None

    m = matches.loc[mask].iloc[0]
    result = {
        "match_id": int(m["match_id"]),
        "home_team": m["home_team"],
        "away_team": m["away_team"],
        "venue": m.get("venue", ""),
        "date": str(m["date"])[:10] if pd.notna(m.get("date")) else None,
        "year": int(m["year"]),
        "round_number": int(m["round_number"]),
        "home_score": int(m["home_score"]) if pd.notna(m.get("home_score")) else None,
        "away_score": int(m["away_score"]) if pd.notna(m.get("away_score")) else None,
        "home_players": [],
        "away_players": [],
    }

    # Add team stat totals from team_matches
    tm = cache.team_matches
    if not tm.empty:
        tm_match = tm[tm["match_id"] == match_id]
        stat_cols = ["GL", "BH", "DI", "MK", "TK", "CP", "IF", "CL", "RB"]
        for side, team_name in [("home_team_stats", m["home_team"]), ("away_team_stats", m["away_team"])]:
            team_row = tm_match[tm_match["team"] == team_name]
            if not team_row.empty:
                tr = team_row.iloc[0]
                stats = {
                    "score": int(tr["score"]) if pd.notna(tr.get("score")) else None,
                    "opp_score": int(tr["opp_score"]) if pd.notna(tr.get("opp_score")) else None,
                }
                for col in stat_cols:
                    if col in tr.index and pd.notna(tr[col]):
                        stats[col] = int(tr[col])
                result[side] = stats

    # Load game predictions for win prob
    store = _get_store(cache)
    if store is not None:
        game_preds = store.load_game_predictions(
            year=int(m["year"]), round_num=int(m["round_number"])
        )
        if not game_preds.empty:
            gp_mask = game_preds["match_id"] == match_id if "match_id" in game_preds.columns else pd.Series(False, index=game_preds.index)
            if not gp_mask.any() and "home_team" in game_preds.columns:
                gp_mask = (game_preds["home_team"] == m["home_team"]) & (game_preds["away_team"] == m["away_team"])
            gp = game_preds.loc[gp_mask]
            if len(gp) > 0:
                row = gp.iloc[0]
                if "home_win_prob" in row.index and pd.notna(row["home_win_prob"]):
                    result["home_win_prob"] = round(float(row["home_win_prob"]), 4)
                    result["away_win_prob"] = round(1 - float(row["home_win_prob"]), 4)

        # Load player predictions
        preds = store.load_predictions(
            year=int(m["year"]), round_num=int(m["round_number"])
        )
        if not preds.empty:
            for side, team_name in [("home_players", m["home_team"]), ("away_players", m["away_team"])]:
                team_mask = preds["team"] == team_name
                if "match_id" in preds.columns:
                    exact = team_mask & (preds["match_id"] == match_id)
                    if exact.any():
                        team_mask = exact
                team_preds = preds.loc[team_mask]
                for _, p in team_preds.iterrows():
                    entry = {
                        "player": p.get("player", ""),
                        "team": p.get("team", ""),
                        "opponent": p.get("opponent", ""),
                    }
                    for col in ["predicted_goals", "predicted_disposals", "predicted_marks",
                                 "p_scorer", "p_2plus_goals", "p_3plus_goals",
                                 "p_15plus_disp", "p_20plus_disp", "p_25plus_disp", "p_30plus_disp",
                                 "p_3plus_mk", "p_5plus_mk"]:
                        if col in p.index and pd.notna(p[col]):
                            entry[col] = round(float(p[col]), 4)
                    result[side].append(entry)

    return result
