"""Generate game-level predictions (win prob, margin) for upcoming rounds.

Builds synthetic team_match rows from fixture CSVs using the same match_id
scheme as _ensure_fixture_match_ids (negative IDs), so they align with
player predictions. Runs AFLGameWinnerModel and saves to sequential store.
"""
import sys
import pandas as pd
import numpy as np

import config
from store import LearningStore
from model import AFLGameWinnerModel

YEAR = 2026
RUN_ID = "live_2026"
ROUNDS = range(0, 5)


def get_fixture_match_ids(fixture_path):
    """Replicate _ensure_fixture_match_ids logic to get the same negative IDs."""
    fx = pd.read_csv(fixture_path)
    team = fx["team"].astype(str).str.strip()
    opp = fx["opponent"].astype(str).str.strip()
    t1 = np.where(team <= opp, team, opp)
    t2 = np.where(team <= opp, opp, team)
    date_key = pd.to_datetime(fx["date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    venue_key = fx["venue"].astype(str).str.strip().str.lower()

    key_df = pd.DataFrame({"_date_key": date_key, "_venue_key": venue_key, "_t1": t1, "_t2": t2})
    uniq = key_df.drop_duplicates().sort_values(["_date_key", "_venue_key", "_t1", "_t2"]).reset_index(drop=True)
    uniq["match_id"] = -(np.arange(len(uniq)) + 1)
    key_to_id = uniq.set_index(["_date_key", "_venue_key", "_t1", "_t2"])["match_id"]

    fx["match_id"] = key_df.set_index(["_date_key", "_venue_key", "_t1", "_t2"]).index.map(key_to_id).values
    fx["match_id"] = fx["match_id"].astype(int)
    return fx


def build_synthetic_team_rows(fixtures, round_num, year, team_match_df):
    """Create synthetic team_match rows using fixture match_ids."""
    rows = []
    for _, row in fixtures.iterrows():
        team = row["team"]
        team_hist = team_match_df[team_match_df["team"] == team].sort_values("date").tail(5)
        if team_hist.empty:
            continue
        rows.append({
            "match_id": int(row["match_id"]),
            "team": team,
            "opponent": row["opponent"],
            "venue": row["venue"],
            "date": pd.to_datetime(row["date"]),
            "year": year,
            "round_number": round_num,
            "is_home": bool(row["is_home"]),
            "is_finals": False,
            "score": team_hist["score"].mean(),
            "opp_score": team_hist["opp_score"].mean(),
            "margin": team_hist["margin"].mean(),
            "rest_days": 7,
            "attendance": 40000,
        })
    return pd.DataFrame(rows)


def main():
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if not tm_path.exists():
        print("No team_matches.parquet found.")
        sys.exit(1)

    team_match_df = pd.read_parquet(tm_path)
    team_match_df["date"] = pd.to_datetime(team_match_df["date"])
    print(f"Loaded {len(team_match_df)} team_match rows")

    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=RUN_ID)

    for rnd in ROUNDS:
        fixture_path = config.FIXTURES_DIR / f"round_{rnd}_{YEAR}.csv"
        if not fixture_path.exists():
            print(f"R{rnd:02d}: No fixture file, skipping")
            continue

        # Get fixtures with correct match_ids (same as player predictions)
        fixtures = get_fixture_match_ids(fixture_path)

        # Build synthetic team rows
        synthetic = build_synthetic_team_rows(fixtures, rnd, YEAR, team_match_df)
        if synthetic.empty:
            print(f"R{rnd:02d}: No synthetic rows generated")
            continue

        # Combine historical + synthetic
        combined_tm = pd.concat([team_match_df, synthetic], ignore_index=True)
        combined_tm["date"] = pd.to_datetime(combined_tm["date"])

        # Load player predictions for aggregated features
        player_preds = store.load_predictions(year=YEAR, round_num=rnd, run_id=RUN_ID)
        pp_for_gw = pd.DataFrame()
        if player_preds is not None and not player_preds.empty:
            need = ["match_id", "team", "predicted_goals", "predicted_disposals", "predicted_marks"]
            have = [c for c in need if c in player_preds.columns]
            if len(have) >= 3:
                pp_for_gw = player_preds[have].copy()

        # Train game winner model on historical data
        winner_model = AFLGameWinnerModel()
        try:
            winner_model.train_backtest(
                team_match_df,
                player_predictions_df=pp_for_gw if not pp_for_gw.empty else None,
            )

            round_match_ids = synthetic["match_id"].unique()
            combined_sorted = combined_tm.sort_values("date").reset_index(drop=True)
            elo_df = winner_model.elo_system.compute_all(combined_sorted)

            game_df, _ = winner_model.build_game_features(
                combined_sorted, elo_df,
                player_predictions_df=pp_for_gw if not pp_for_gw.empty else None,
            )
            game_df = game_df[game_df["match_id"].isin(round_match_ids)]

            if game_df.empty:
                print(f"R{rnd:02d}: No game features built")
                continue

            for c in winner_model.feature_cols:
                if c not in game_df.columns:
                    game_df[c] = 0

            X = game_df[winner_model.feature_cols].fillna(0)
            hybrid_prob = winner_model.classifier.predict_proba(X)[:, 1]
            pred_margin = winner_model.margin_regressor.predict(X) if winner_model.margin_regressor else np.zeros(len(game_df))

            result = pd.DataFrame({
                "match_id": game_df["match_id"].values,
                "home_team": game_df["team"].values,
                "away_team": game_df["opponent"].values,
                "venue": game_df["venue"].values,
                "home_win_prob": np.round(hybrid_prob, 4),
                "away_win_prob": np.round(1 - hybrid_prob, 4),
                "predicted_margin": np.round(pred_margin, 1),
                "predicted_winner": np.where(
                    hybrid_prob > 0.5,
                    game_df["team"].values,
                    game_df["opponent"].values,
                ),
            })

            store.save_game_predictions(YEAR, rnd, result)
            print(f"R{rnd:02d}: Saved {len(result)} game predictions")
            print(result[["match_id", "home_team", "away_team", "home_win_prob", "predicted_margin", "predicted_winner"]].to_string(index=False))
            print()

        except Exception as e:
            print(f"R{rnd:02d}: Failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
