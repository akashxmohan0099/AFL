"""League-wide services: stat leaders."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.data_loader import DataCache

VALID_STATS = {"GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO",
               "CP", "UP", "IF", "CL", "CG", "FF", "FA", "CM",
               "one_pct", "RB", "MI", "BO", "GA"}


def _r(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)


def get_leaders(stat: str = "GL", year: int = 2025, limit: int = 50, min_games: int = 5) -> dict:
    """Top players by a given stat for a season."""
    cache = DataCache.get()
    pg = cache.player_games

    if pg.empty or stat not in VALID_STATS:
        return {"stat": stat, "year": year, "leaders": []}

    season = pg[pg["year"] == year]
    if season.empty:
        return {"stat": stat, "year": year, "leaders": []}

    if stat not in season.columns:
        return {"stat": stat, "year": year, "leaders": []}

    grouped = season.groupby(["player_id", "player", "team"], observed=True)
    agg = grouped.agg(
        games=(stat, "size"),
        total=(stat, "sum"),
        avg=(stat, "mean"),
        best=(stat, "max"),
    ).reset_index()

    qualified = agg[agg["games"] >= min_games]
    if qualified.empty:
        return {"stat": stat, "year": year, "leaders": []}

    # All stats in VALID_STATS are counting stats — sort by total
    sort_col = "total"
    top = qualified.nlargest(limit, sort_col)

    leaders = []
    for _, r in top.iterrows():
        leaders.append({
            "player_id": str(r["player_id"]),
            "name": str(r["player"]),
            "team": str(r["team"]),
            "games": int(r["games"]),
            "total": int(r["total"]),
            "avg": _r(r["avg"], 1),
            "best": int(r["best"]),
        })

    return {"stat": stat, "year": year, "leaders": leaders}
