"""Team-level services: roster, record, form, splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from api.data_loader import DataCache


def _r(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)


def list_teams(year: int) -> list[dict]:
    """All teams with W/L record and season averages for a given year."""
    cache = DataCache.get()
    tm = cache.team_matches
    if tm.empty:
        return []

    season = tm[tm["year"] == year]
    if season.empty:
        return []

    results = []
    for team, grp in season.groupby("team", observed=True):
        wins = int((grp["result"] == "W").sum())
        losses = int((grp["result"] == "L").sum())
        draws = len(grp) - wins - losses
        results.append({
            "team": str(team),
            "played": len(grp),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "pct": _r(wins / len(grp) * 100, 1) if len(grp) > 0 else 0,
            "avg_score": _r(grp["score"].mean(), 1),
            "avg_conceded": _r(grp["opp_score"].mean(), 1),
            "avg_margin": _r(grp["margin"].mean(), 1) if "margin" in grp.columns else None,
        })

    results.sort(key=lambda x: (-x["wins"], -(x.get("avg_margin") or 0)))
    return results


def get_team_detail(team_name: str, year: int) -> dict | None:
    """Detailed team profile: record, form, splits, top players."""
    cache = DataCache.get()
    tm = cache.team_matches
    pg = cache.player_games

    if tm.empty:
        return None

    team_tm = tm[tm["team"] == team_name]
    if team_tm.empty:
        return None

    season = team_tm[team_tm["year"] == year]

    # Season record
    def _record(games: pd.DataFrame) -> dict:
        if games.empty:
            return {"played": 0, "wins": 0, "losses": 0, "draws": 0}
        wins = int((games["result"] == "W").sum())
        losses = int((games["result"] == "L").sum())
        draws = len(games) - wins - losses
        return {
            "played": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_score": _r(games["score"].mean(), 1),
            "avg_conceded": _r(games["opp_score"].mean(), 1),
            "avg_margin": _r(games["margin"].mean(), 1) if "margin" in games.columns else None,
        }

    record = _record(season)

    # Home/away split
    home_away = {}
    if not season.empty:
        home_games = season[season["is_home"] == True]
        away_games = season[season["is_home"] == False]
        if not home_games.empty:
            home_away["home"] = _record(home_games)
        if not away_games.empty:
            home_away["away"] = _record(away_games)

    # Recent form (last 5 games)
    recent = season.sort_values("date").tail(5)
    form = []
    for _, r in recent.iterrows():
        form.append({
            "result": r.get("result", ""),
            "opponent": r.get("opponent", ""),
            "score": int(r["score"]) if pd.notna(r.get("score")) else None,
            "opp_score": int(r["opp_score"]) if pd.notna(r.get("opp_score")) else None,
            "margin": int(r["margin"]) if pd.notna(r.get("margin")) else None,
            "venue": r.get("venue", ""),
            "is_home": bool(r.get("is_home", False)),
            "round_number": int(r["round_number"]) if pd.notna(r.get("round_number")) else None,
        })

    # Season stat averages from team_matches
    stat_cols = ["GL", "BH", "DI", "MK", "TK", "CP", "IF", "CL", "RB"]
    stats = {}
    if not season.empty:
        for col in stat_cols:
            if col in season.columns:
                vals = season[col].dropna()
                if not vals.empty:
                    stats[f"avg_{col.lower()}"] = _r(vals.mean(), 1)

    # Top players (season)
    top_players = []
    if not pg.empty:
        team_pg = pg[(pg["team"] == team_name) & (pg["year"] == year)]
        if not team_pg.empty:
            player_agg = team_pg.groupby(["player_id", "player"], observed=True).agg(
                games=("GL", "size"),
                total_gl=("GL", "sum"),
                avg_gl=("GL", "mean"),
                total_di=("DI", "sum"),
                avg_di=("DI", "mean"),
                total_mk=("MK", "sum"),
                avg_mk=("MK", "mean"),
                avg_tk=("TK", "mean"),
            ).reset_index()

            for _, r in player_agg.nlargest(15, "games").iterrows():
                top_players.append({
                    "player_id": str(r["player_id"]),
                    "name": str(r["player"]),
                    "games": int(r["games"]),
                    "total_goals": int(r["total_gl"]),
                    "avg_goals": _r(r["avg_gl"]),
                    "total_disposals": int(r["total_di"]),
                    "avg_disposals": _r(r["avg_di"], 1),
                    "avg_marks": _r(r["avg_mk"], 1),
                    "avg_tackles": _r(r["avg_tk"], 1),
                })

    # Opponent splits (season)
    opponent_splits = []
    if not season.empty:
        for opp, grp in season.groupby("opponent", observed=True):
            opponent_splits.append({
                "opponent": str(opp),
                **_record(grp),
            })
        opponent_splits.sort(key=lambda x: -x["played"])

    # Season history (last 5 years)
    season_history = []
    for y in range(year, max(year - 5, 2014), -1):
        yr_data = team_tm[team_tm["year"] == y]
        if yr_data.empty:
            continue
        season_history.append({
            "year": y,
            **_record(yr_data),
        })

    return {
        "team": team_name,
        "year": year,
        "record": record,
        "home_away": home_away,
        "recent_form": form,
        "stats": stats,
        "top_players": top_players,
        "opponent_splits": opponent_splits,
        "season_history": season_history,
    }
