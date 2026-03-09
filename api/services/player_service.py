"""Player search, profile, and game log services."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from api.data_loader import DataCache


def search_players(query: str, limit: int = 20) -> list[dict]:
    return DataCache.get().search_players(query, limit)


def list_players(year: int = 2025) -> list[dict]:
    """Return all players who played in the given year with season averages."""
    cache = DataCache.get()
    pg = cache.player_games
    mask = pg["year"] == year
    season = pg.loc[mask]

    if season.empty:
        return []

    stat_cols = ["GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO"]
    grouped = season.groupby(["player_id", "player", "team"], observed=True)

    agg = {col: "mean" for col in stat_cols}
    agg["match_id"] = "count"
    stats = grouped.agg(agg).reset_index()
    stats.rename(columns={"match_id": "games"}, inplace=True)

    for col in stat_cols:
        stats[col] = stats[col].round(2)

    results = []
    for _, row in stats.iterrows():
        results.append({
            "player_id": row["player_id"],
            "name": row["player"],
            "team": row["team"],
            "games": int(row["games"]),
            "avg_goals": round(float(row["GL"]), 2),
            "avg_disposals": round(float(row["DI"]), 1),
            "avg_marks": round(float(row["MK"]), 1),
            "avg_tackles": round(float(row["TK"]), 1),
            "avg_kicks": round(float(row["KI"]), 1),
            "avg_handballs": round(float(row["HB"]), 1),
            "avg_hitouts": round(float(row["HO"]), 1),
        })

    results.sort(key=lambda x: -x["games"])
    return results


def _r(v, d=2):
    """Round a float safely."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)


def get_player_profile(player_id: str) -> dict | None:
    cache = DataCache.get()
    pg = cache.player_games
    mask = pg["player_id"] == player_id
    player_data = pg.loc[mask].sort_values("date").copy()

    if player_data.empty:
        return None

    latest = player_data.iloc[-1]
    name = latest["player"]
    team = latest["team"]

    CORE_STATS = ["GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO"]
    EXTENDED_STATS = ["GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO",
                      "CP", "UP", "IF", "CL", "CG", "FF", "FA", "CM",
                      "one_pct", "RB", "MI", "BO", "GA"]

    # ── Season averages ──────────────────────────────────────────────
    seasons = []
    for year, grp in player_data.groupby("year", observed=True):
        entry = {"year": int(year), "games": len(grp)}
        for col in EXTENDED_STATS:
            if col in grp.columns:
                entry[col] = _r(grp[col].mean())
        seasons.append(entry)

    # ── Career averages ──────────────────────────────────────────────
    career = {"games": len(player_data)}
    for col in EXTENDED_STATS:
        if col in player_data.columns:
            career[f"avg_{col.lower()}"] = _r(player_data[col].mean())
            career[f"std_{col.lower()}"] = _r(player_data[col].std())

    # ── Career highs ─────────────────────────────────────────────────
    career_highs = {}
    for col in CORE_STATS:
        if col not in player_data.columns:
            continue
        idx = player_data[col].idxmax()
        if pd.isna(idx):
            continue
        row = player_data.loc[idx]
        career_highs[col.lower()] = {
            "value": int(row[col]),
            "opponent": row["opponent"],
            "venue": row["venue"],
            "date": str(row["date"])[:10],
            "year": int(row["year"]),
            "round": int(row["round_number"]),
        }

    # ── Recent form (last 10 games) ─────────────────────────────────
    recent = player_data.tail(10)
    form = []
    for _, row in recent.iterrows():
        entry = {
            "date": str(row["date"])[:10],
            "year": int(row["year"]),
            "round": int(row["round_number"]),
            "opponent": row["opponent"],
            "venue": row["venue"],
            "is_home": bool(row["is_home"]),
        }
        for col in CORE_STATS:
            entry[col] = int(row[col])
        form.append(entry)

    # ── Home vs Away splits ──────────────────────────────────────────
    home_away = {}
    for label, grp in [("home", player_data[player_data["is_home"] == True]),
                        ("away", player_data[player_data["is_home"] == False])]:
        if grp.empty:
            continue
        entry = {"games": len(grp)}
        for col in CORE_STATS:
            if col in grp.columns:
                entry[f"avg_{col.lower()}"] = _r(grp[col].mean())
        home_away[label] = entry

    # ── Opponent splits (min 2 games) ────────────────────────────────
    opponent_splits = []
    for opp, grp in player_data.groupby("opponent", observed=True):
        if len(grp) < 2:
            continue
        entry = {"opponent": str(opp), "games": len(grp)}
        for col in CORE_STATS:
            if col in grp.columns:
                entry[f"avg_{col.lower()}"] = _r(grp[col].mean())
        opponent_splits.append(entry)
    opponent_splits.sort(key=lambda x: -x["games"])

    # ── Venue splits ─────────────────────────────────────────────────
    venue_splits = []
    for venue, grp in player_data.groupby("venue", observed=True):
        entry = {"venue": str(venue), "games": len(grp)}
        for col in CORE_STATS:
            if col in grp.columns:
                entry[f"avg_{col.lower()}"] = _r(grp[col].mean())
        venue_splits.append(entry)
    venue_splits.sort(key=lambda v: -v["games"])

    # ── Streaks ──────────────────────────────────────────────────────
    streaks = _compute_streaks(player_data)

    # ── Consistency (last 2 seasons) ─────────────────────────────────
    recent_years = sorted(player_data["year"].unique())[-2:]
    recent_data = player_data[player_data["year"].isin(recent_years)]
    consistency = {}
    for col in ["GL", "DI", "MK", "TK"]:
        if col not in recent_data.columns or recent_data[col].empty:
            continue
        vals = recent_data[col].dropna()
        if len(vals) < 3:
            continue
        consistency[col.lower()] = {
            "avg": _r(vals.mean()),
            "std": _r(vals.std()),
            "min": int(vals.min()),
            "max": int(vals.max()),
            "median": _r(vals.median()),
            "p25": _r(vals.quantile(0.25)),
            "p75": _r(vals.quantile(0.75)),
            "ceiling": int(vals.quantile(0.9)),
            "floor": int(vals.quantile(0.1)),
            "games": len(vals),
        }

    # ── Scoring by quarter ───────────────────────────────────────────
    quarter_scoring = None
    q_cols = ["q1_goals", "q2_goals", "q3_goals", "q4_goals"]
    if all(c in player_data.columns for c in q_cols):
        valid = player_data.dropna(subset=q_cols)
        if len(valid) >= 5:
            quarter_scoring = {
                "q1": _r(valid["q1_goals"].mean()),
                "q2": _r(valid["q2_goals"].mean()),
                "q3": _r(valid["q3_goals"].mean()),
                "q4": _r(valid["q4_goals"].mean()),
                "games": len(valid),
            }

    # ── FootyWire advanced stats ──────────────────────────────────────
    footywire_stats = _get_footywire_stats(name, cache)

    # ── Predictions vs actuals ───────────────────────────────────────
    pred_vs_actual = _compute_predictions_vs_actuals(player_id, cache)

    return {
        "player_id": player_id,
        "name": name,
        "team": team,
        "total_games": len(player_data),
        "career_goals": int(player_data["GL"].sum()),
        "career_goal_avg": _r(player_data["GL"].mean(), 3),
        "career": career,
        "career_highs": career_highs,
        "seasons": seasons,
        "recent_form": form,
        "home_away": home_away,
        "opponent_splits": opponent_splits,
        "venue_splits": venue_splits,
        "streaks": streaks,
        "consistency": consistency,
        "quarter_scoring": quarter_scoring,
        "predictions_vs_actuals": pred_vs_actual,
        "footywire": footywire_stats,
    }


def _compute_streaks(player_data: pd.DataFrame) -> dict:
    """Compute current and longest streaks for key thresholds."""
    thresholds = [
        ("goals_1plus", "GL", 1),
        ("goals_2plus", "GL", 2),
        ("goals_3plus", "GL", 3),
        ("disp_20plus", "DI", 20),
        ("disp_25plus", "DI", 25),
        ("disp_30plus", "DI", 30),
        ("marks_5plus", "MK", 5),
        ("tackles_4plus", "TK", 4),
    ]

    result = {}
    for key, col, threshold in thresholds:
        if col not in player_data.columns:
            continue
        hits = (player_data[col] >= threshold).values
        if len(hits) == 0:
            continue

        # Current streak (from most recent game backwards)
        current = 0
        for h in reversed(hits):
            if h:
                current += 1
            else:
                break

        # Longest streak
        longest = 0
        run = 0
        for h in hits:
            if h:
                run += 1
                longest = max(longest, run)
            else:
                run = 0

        # Hit rate
        total = len(hits)
        hit_count = int(hits.sum())

        result[key] = {
            "current": current,
            "longest": longest,
            "hit_rate": _r(hit_count / total * 100, 1) if total > 0 else 0,
            "hits": hit_count,
            "total": total,
        }

    return result


def get_player_games(player_id: str, year: int | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    cache = DataCache.get()
    pg = cache.player_games
    mask = pg["player_id"] == player_id
    if year is not None:
        mask = mask & (pg["year"] == year)

    data = pg.loc[mask].sort_values("date", ascending=False)
    page = data.iloc[offset:offset + limit]

    stat_cols = ["GL", "BH", "DI", "MK", "KI", "HB", "TK", "HO",
                 "CP", "UP", "IF", "CL", "CG", "FF", "FA", "CM",
                 "one_pct", "RB", "MI", "BO", "GA"]
    results = []
    for _, row in page.iterrows():
        entry = {
            "match_id": int(row["match_id"]),
            "date": str(row["date"])[:10],
            "year": int(row["year"]),
            "round_number": int(row["round_number"]),
            "team": row["team"],
            "opponent": row["opponent"],
            "venue": row["venue"],
            "is_home": bool(row["is_home"]) if "is_home" in row.index else None,
        }
        for col in stat_cols:
            if col in row.index and pd.notna(row[col]):
                entry[col] = int(row[col])
        results.append(entry)

    return results


def get_player_predictions(player_id: str, year: int = 2025) -> list[dict]:
    cache = DataCache.get()
    store = cache.sequential_store or cache.store
    if store is None:
        return []

    name = cache.get_player_name(player_id)

    preds = store.load_predictions(year=year)
    if preds.empty:
        return []

    matched = preds.loc[preds["player"] == name].copy()
    team_hint = None
    parts = player_id.split("_", 1)
    if len(parts) == 2:
        team_hint = parts[1]

    if team_hint and not matched.empty and "team" in matched.columns:
        strict = matched.loc[matched["team"] == team_hint].copy()
        if not strict.empty:
            matched = strict
        elif matched["team"].nunique(dropna=True) > 1:
            return []

    if matched.empty:
        return []

    results = []
    for _, row in matched.iterrows():
        entry = {"player": row["player"], "team": row.get("team", "")}
        for col in ["opponent", "predicted_goals", "predicted_disposals", "predicted_marks",
                     "p_scorer", "p_2plus_goals", "p_3plus_goals",
                     "p_20plus_disp", "p_25plus_disp", "p_30plus_disp"]:
            if col in row.index and pd.notna(row[col]):
                entry[col] = row[col]
        if "round" in row.index and pd.notna(row["round"]):
            entry["round_number"] = int(row["round"])
        if "match_id" in row.index and pd.notna(row["match_id"]):
            entry["match_id"] = int(row["match_id"])
        if "venue" in row.index and pd.notna(row["venue"]):
            entry["venue"] = row["venue"]
        if "player_role" in row.index and pd.notna(row["player_role"]):
            entry["player_role"] = row["player_role"]
        results.append(entry)

    return results


def _get_footywire_stats(player_name: str, cache: DataCache) -> dict | None:
    """Get per-season FootyWire advanced stats for a player."""
    fw = cache.footywire
    if fw.empty:
        return None

    mask = fw["player"] == player_name
    player_fw = fw.loc[mask]
    if player_fw.empty:
        return None

    # Need match_id → year mapping
    matches = cache.matches
    if matches.empty:
        return None

    year_map = dict(zip(matches["match_id"], matches["year"]))
    player_fw = player_fw.copy()
    player_fw["year"] = player_fw["match_id"].map(year_map)
    player_fw = player_fw.dropna(subset=["year"])

    fw_cols = ["ED", "DE_pct", "CCL", "SCL", "TO", "MG", "SI", "ITC", "T5", "TOG_pct"]

    seasons = []
    for year, grp in player_fw.groupby("year"):
        entry = {"year": int(year), "games": len(grp)}
        for col in fw_cols:
            if col in grp.columns:
                vals = grp[col].dropna()
                if not vals.empty:
                    entry[f"avg_{col.lower()}"] = round(float(vals.mean()), 1)
        seasons.append(entry)

    career = {"games": len(player_fw)}
    for col in fw_cols:
        if col in player_fw.columns:
            vals = player_fw[col].dropna()
            if not vals.empty:
                career[f"avg_{col.lower()}"] = round(float(vals.mean()), 1)

    return {"seasons": seasons, "career": career}


def _compute_predictions_vs_actuals(player_id: str, cache: DataCache) -> dict:
    """Load sequential predictions and outcomes for a player across all years."""
    store = cache.sequential_store or cache.store
    if store is None:
        return {"records": [], "mae": {}}

    name = cache.get_player_name(player_id)
    parts = player_id.split("_", 1)
    team_hint = parts[1] if len(parts) == 2 else None

    all_records = []
    for year in range(2015, 2027):
        try:
            preds = store.load_predictions(year=year)
            outcomes = store.load_outcomes(year=year)
        except Exception:
            continue

        if preds.empty or outcomes.empty:
            continue

        pred_mask = preds["player"] == name
        if team_hint and "team" in preds.columns:
            pred_mask = pred_mask & (preds["team"] == team_hint)
        player_preds = preds.loc[pred_mask]

        if player_preds.empty:
            continue

        merge_cols = ["player", "team", "match_id"]
        merged = player_preds.merge(outcomes, on=merge_cols, how="inner")

        for _, row in merged.iterrows():
            record = {
                "year": year,
                "round": int(row["round"]) if "round" in row.index and pd.notna(row["round"]) else None,
                "match_id": int(row["match_id"]),
                "opponent": row.get("opponent", ""),
                "venue": row.get("venue", ""),
            }
            for pred_col, actual_col in [
                ("predicted_goals", "actual_goals"),
                ("predicted_disposals", "actual_disposals"),
                ("predicted_marks", "actual_marks"),
            ]:
                if pred_col in row.index and pd.notna(row[pred_col]):
                    record[pred_col] = round(float(row[pred_col]), 2)
                if actual_col in row.index and pd.notna(row[actual_col]):
                    record[actual_col] = int(row[actual_col])
            all_records.append(record)

    mae = {}
    if all_records:
        records_df = pd.DataFrame(all_records)
        for metric in ["goals", "disposals", "marks"]:
            pred_col = f"predicted_{metric}"
            actual_col = f"actual_{metric}"
            if pred_col in records_df.columns and actual_col in records_df.columns:
                valid = records_df.dropna(subset=[pred_col, actual_col])
                if not valid.empty:
                    mae[metric] = round(
                        float((valid[pred_col] - valid[actual_col]).abs().mean()), 3
                    )

    return {"records": all_records, "mae": mae}
