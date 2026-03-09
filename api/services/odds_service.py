"""Odds comparison services — model vs market."""
from __future__ import annotations

import pandas as pd

from api.data_loader import DataCache


def _get_store(cache: DataCache):
    return cache.sequential_store or cache.store


def _match_prediction_row(preds: pd.DataFrame, player: str, match_id: int | None = None):
    matched = preds.loc[preds["player"] == player]
    if matched.empty:
        return None

    if match_id is not None and "match_id" in matched.columns:
        exact = matched.loc[matched["match_id"] == match_id]
        if not exact.empty:
            return exact.iloc[0]

    if "team" in matched.columns and matched["team"].nunique(dropna=True) > 1:
        return None

    return matched.iloc[0]


def get_game_odds(year: int, round_num: int) -> list[dict]:
    cache = DataCache.get()
    matches = cache.matches
    odds = cache.odds

    if matches.empty:
        return []

    mask = (matches["year"] == year) & (matches["round_number"] == round_num)
    round_matches = matches.loc[mask]

    # Load game predictions
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
        }

        # Model probabilities
        if not game_preds.empty:
            gp = pd.DataFrame()
            if "match_id" in game_preds.columns:
                gp = game_preds.loc[game_preds["match_id"] == m["match_id"]]
            if gp.empty and "home_team" in game_preds.columns and "away_team" in game_preds.columns:
                gp = game_preds.loc[
                    (game_preds["home_team"] == m["home_team"])
                    & (game_preds["away_team"] == m["away_team"])
                ]
            if len(gp) > 0 and "home_win_prob" in gp.columns:
                hwp = float(gp.iloc[0]["home_win_prob"])
                entry["model_home_prob"] = round(hwp, 4)
                entry["model_away_prob"] = round(1 - hwp, 4)

        # Market odds
        if not odds.empty and "match_id" in odds.columns:
            om = odds.loc[odds["match_id"] == m["match_id"]]
            if len(om) > 0:
                row = om.iloc[0]
                for col, key in [
                    ("home_implied_prob", "market_home_prob"),
                    ("away_implied_prob", "market_away_prob"),
                ]:
                    if col in row.index and pd.notna(row[col]):
                        entry[key] = round(float(row[col]), 4)

        # Compute edges
        if entry.get("model_home_prob") is not None and entry.get("market_home_prob") is not None:
            entry["edge_home"] = round(entry["model_home_prob"] - entry["market_home_prob"], 4)
        if entry.get("model_away_prob") is not None and entry.get("market_away_prob") is not None:
            entry["edge_away"] = round(entry["model_away_prob"] - entry["market_away_prob"], 4)

        results.append(entry)

    return results


def get_player_odds(year: int, round_num: int) -> list[dict]:
    cache = DataCache.get()
    player_odds = cache.player_odds
    matches = cache.matches

    if player_odds.empty or matches.empty:
        return []

    # Get match_ids for this round
    mask = (matches["year"] == year) & (matches["round_number"] == round_num)
    match_ids = matches.loc[mask, "match_id"].tolist()

    if not match_ids:
        return []

    round_odds = player_odds.loc[player_odds["match_id"].isin(match_ids)]
    if round_odds.empty:
        return []

    # Load model predictions for comparison
    preds = pd.DataFrame()
    store = _get_store(cache)
    if store is not None:
        preds = store.load_predictions(year=year, round_num=round_num)

    results = []
    for _, row in round_odds.iterrows():
        player = row["player"]
        pred_row = None
        if not preds.empty:
            pred_row = _match_prediction_row(
                preds, player=player, match_id=int(row["match_id"]) if pd.notna(row.get("match_id")) else None
            )

        # Disposal odds
        if pd.notna(row.get("market_disposal_line")):
            entry = {
                "player": player,
                "team": "",
                "market_type": "disposals",
                "market_line": float(row["market_disposal_line"]),
                "market_price": float(row["market_disposal_over_price"]) if pd.notna(row.get("market_disposal_over_price")) else None,
                "market_implied_prob": float(row["market_disposal_implied_over"]) if pd.notna(row.get("market_disposal_implied_over")) else None,
            }

            # Match with model prediction
            if pred_row is not None:
                entry["team"] = pred_row.get("team", "")
                line = row["market_disposal_line"]
                # Find closest threshold
                for thresh in [10, 15, 20, 25, 30]:
                    if abs(thresh - line) <= 2.5:
                        col = f"p_{thresh}plus_disp"
                        if col in pred_row.index and pd.notna(pred_row[col]):
                            entry["model_prob"] = round(float(pred_row[col]), 4)
                            if entry.get("market_implied_prob") is not None:
                                entry["edge"] = round(entry["model_prob"] - entry["market_implied_prob"], 4)
                            break
            results.append(entry)

        # Goal scorer odds
        if pd.notna(row.get("market_fgs_implied_prob")):
            entry = {
                "player": player,
                "team": "",
                "market_type": "first_goal",
                "market_price": float(row["market_fgs_price"]) if pd.notna(row.get("market_fgs_price")) else None,
                "market_implied_prob": float(row["market_fgs_implied_prob"]),
            }
            if pred_row is not None:
                entry["team"] = pred_row.get("team", "")
            results.append(entry)

        # 2+ goals odds
        if pd.notna(row.get("market_2goals_implied_prob")):
            entry = {
                "player": player,
                "team": "",
                "market_type": "2plus_goals",
                "market_price": float(row["market_2goals_price"]) if pd.notna(row.get("market_2goals_price")) else None,
                "market_implied_prob": float(row["market_2goals_implied_prob"]),
            }
            if pred_row is not None:
                entry["team"] = pred_row.get("team", "")
                if "p_2plus_goals" in pred_row.index and pd.notna(pred_row["p_2plus_goals"]):
                    entry["model_prob"] = round(float(pred_row["p_2plus_goals"]), 4)
                    if entry.get("market_implied_prob") is not None:
                        entry["edge"] = round(entry["model_prob"] - entry["market_implied_prob"], 4)
            results.append(entry)

    return results
