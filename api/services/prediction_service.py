"""Prediction loading and grouping services."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from api.data_loader import DataCache


def _get_store(cache: DataCache):
    return cache.sequential_store or cache.store


def list_available_rounds(year: int | None = None) -> list[dict]:
    cache = DataCache.get()
    store = _get_store(cache)
    if store is None:
        return []

    rounds = store.list_rounds(subdir="predictions", year=year)
    return [{"year": y, "round_number": r} for y, r in rounds]


def get_round_predictions(year: int, round_num: int) -> list[dict]:
    cache = DataCache.get()
    store = _get_store(cache)
    if store is None:
        return []

    preds = store.load_predictions(year=year, round_num=round_num)
    if preds.empty:
        return []

    results = []
    for _, row in preds.iterrows():
        entry = {
            "player": row.get("player", ""),
            "team": row.get("team", ""),
            "opponent": row.get("opponent", ""),
        }
        # Add all prediction columns
        for col in ["predicted_goals", "predicted_behinds", "predicted_disposals",
                     "predicted_marks", "predicted_score",
                     "p_scorer", "p_2plus_goals", "p_3plus_goals",
                     "lambda_goals", "lambda_disposals",
                     "p_10plus_disp", "p_15plus_disp", "p_20plus_disp",
                     "p_25plus_disp", "p_30plus_disp",
                     "conf_lower_gl", "conf_upper_gl",
                     "conf_lower_di", "conf_upper_di",
                     "player_role", "career_goal_avg",
                     "match_id", "venue", "round"]:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                entry[col] = float(val) if isinstance(val, (float, int)) and col != "match_id" else val
        if "match_id" in entry:
            entry["match_id"] = int(entry["match_id"])
        results.append(entry)

    return results


def get_round_predictions_by_match(year: int, round_num: int) -> dict:
    """Get predictions grouped by match."""
    preds = get_round_predictions(year, round_num)
    if not preds:
        return {}

    grouped = {}
    for p in preds:
        key = p.get("match_id") or f"{p.get('team', '')}_{p.get('opponent', '')}"
        grouped.setdefault(key, []).append(p)

    return grouped
