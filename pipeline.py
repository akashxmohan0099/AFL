"""
AFL Player Goal/Behind Prediction Pipeline — CLI Orchestrator
==============================================================

Usage:
    python pipeline.py --scrape --start 2015 --end 2025   # One-time historical scrape
    python pipeline.py --update                            # Scrape current season + rebuild + predict
    python pipeline.py --update --round 5                  # Predict specific round
    python pipeline.py --train                             # Retrain models only
    python pipeline.py --predict --round 5                 # Predict only (no scrape/retrain)
    python pipeline.py --evaluate                          # Eval on validation set
    python pipeline.py --clean                             # Clean raw data only
    python pipeline.py --features                          # Build features only
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import config
from clean import build_player_matches, load_player_stats
from features import build_features
from model import AFLScoringModel


def cmd_scrape(args):
    """Scrape historical data from AFL Tables."""
    from scraper import scrape_seasons
    start = args.start or config.HISTORICAL_START_YEAR
    end = args.end or config.HISTORICAL_END_YEAR
    print(f"Scraping seasons {start}-{end}...")
    scrape_seasons(start, end, str(config.DATA_DIR))
    print("Scraping complete.")


def cmd_clean(args):
    """Clean and normalize raw data into player_matches.parquet."""
    print("Cleaning raw data...")
    df = build_player_matches(save=True)
    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def cmd_features(args, cleaned_df=None):
    """Build feature matrix from cleaned data."""
    print("Building features...")
    df = build_features(df=cleaned_df, save=True)
    print(f"Feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def cmd_train(args, feature_df=None):
    """Train the prediction models."""
    if feature_df is None:
        feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
        if not feat_path.exists():
            print("No feature matrix found. Run --features first.")
            return None
        feature_df = pd.read_parquet(feat_path)

    print("Training models...")
    model = AFLScoringModel()
    metrics = model.train(feature_df)
    model.save()
    print(f"\nTraining complete. Goals MAE: {metrics['goals_mae']:.4f}")
    return model


def cmd_predict(args, model=None, feature_df=None):
    """Generate predictions for a specific round."""
    round_num = args.round
    year = args.year or config.CURRENT_SEASON_YEAR

    if model is None:
        model = AFLScoringModel()
        try:
            model.load()
        except FileNotFoundError:
            print("No trained model found. Run --train first.")
            return

    # Load fixture file for the upcoming round
    fixture_path = config.FIXTURES_DIR / f"round_{round_num}_{year}.csv"

    if fixture_path.exists():
        fixtures = pd.read_csv(fixture_path)
        print(f"Loaded fixtures from {fixture_path}")
        predictions = _predict_from_fixtures(model, fixtures, year, round_num)
    else:
        # If no fixture file, predict from the most recent data we have
        # (useful for evaluating past rounds)
        if feature_df is None:
            feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
            if not feat_path.exists():
                print("No feature matrix found. Run --features first.")
                return
            feature_df = pd.read_parquet(feat_path)

        # Filter to the specific round
        mask = (feature_df["year"] == year) & (feature_df["round_number"] == round_num)
        round_df = feature_df[mask]

        if round_df.empty:
            print(f"No data found for Round {round_num}, {year}.")
            print(f"Place a fixture CSV at: {fixture_path}")
            print(f"Format: team,opponent,venue,date,is_home")
            return

        print(f"Predicting Round {round_num}, {year} ({len(round_df)} player rows)...")
        predictions = model.predict(round_df, model.feature_cols)

    # Save predictions
    config.ensure_dirs()
    out_path = config.PREDICTIONS_DIR / f"round_{round_num}_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")

    # Display top predicted scorers
    print(f"\n{'='*70}")
    print(f"TOP PREDICTED SCORERS — Round {round_num}, {year}")
    print(f"{'='*70}")
    top = predictions.head(20)
    for _, row in top.iterrows():
        print(
            f"  {row['player']:30s} {row['team']:15s} vs {row['opponent']:15s} "
            f"GL={row['predicted_goals']:.2f} BH={row['predicted_behinds']:.2f} "
            f"Score={row['predicted_score']:.1f}"
        )

    return predictions


def _predict_from_fixtures(model, fixtures, year, round_num):
    """Build features for upcoming fixtures and generate predictions.

    Fixture CSV columns: team, opponent, venue, date, is_home
    Optionally: players (comma-separated player names per team)
    """
    # Load latest player data to get most recent stats
    try:
        stats = load_player_stats()
    except FileNotFoundError:
        print("No player stats found. Run --scrape first.")
        return pd.DataFrame()

    all_predictions = []

    for _, fixture in fixtures.iterrows():
        team = fixture["team"]
        opponent = fixture["opponent"]
        venue = fixture["venue"]

        # Get lineup: use provided players or infer from recent matches
        if "players" in fixture and pd.notna(fixture.get("players")):
            players = [p.strip() for p in str(fixture["players"]).split(",")]
        else:
            # Use most recent team sheet
            recent = stats[stats["team"] == team].sort_values("date_iso")
            last_match = recent["match_id"].iloc[-1] if len(recent) > 0 else None
            if last_match:
                players = recent[recent["match_id"] == last_match]["player"].tolist()
            else:
                continue

        # For each player, build their feature row from history
        for player in players:
            player_history = stats[
                (stats["player"] == player) & (stats["team"] == team)
            ].sort_values("date_iso")

            if len(player_history) < 1:
                continue

            # Use the last row's features as a template, then override
            # venue/opponent specific features
            last_row = player_history.iloc[-1:].copy()
            last_row["venue"] = venue
            last_row["opponent"] = opponent
            last_row["round"] = str(round_num)
            last_row["round_number"] = round_num
            last_row["year"] = year
            last_row["is_home"] = fixture.get("is_home", 1)

            all_predictions.append(last_row)

    if not all_predictions:
        return pd.DataFrame()

    pred_df = pd.concat(all_predictions, ignore_index=True)

    # Build features for these rows
    pred_df = build_features(pred_df, save=False)

    # Predict
    return model.predict(pred_df, model.feature_cols)


def cmd_evaluate(args):
    """Evaluate model on validation set with detailed breakdown."""
    model = AFLScoringModel()
    try:
        model.load()
    except FileNotFoundError:
        print("No trained model found. Run --train first.")
        return

    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    if not feat_path.exists():
        print("No feature matrix found. Run --features first.")
        return

    feature_df = pd.read_parquet(feat_path)
    val_df = feature_df[feature_df["year"] == config.VALIDATION_YEAR]

    if val_df.empty:
        print(f"No validation data for year {config.VALIDATION_YEAR}.")
        return

    print(f"Evaluating on {len(val_df)} validation rows (year={config.VALIDATION_YEAR})...")
    model.evaluate_detailed(val_df, model.feature_cols)


def cmd_update(args):
    """Full update cycle: scrape current season → clean → features → train → predict."""
    year = args.year or config.CURRENT_SEASON_YEAR

    # 1. Scrape current season
    print(f"\n{'='*60}")
    print(f"STEP 1: Scraping {year} season...")
    print(f"{'='*60}")
    from scraper import scrape_seasons
    scrape_seasons(year, year, str(config.DATA_DIR))

    # 2. Clean
    print(f"\n{'='*60}")
    print("STEP 2: Cleaning data...")
    print(f"{'='*60}")
    cleaned = cmd_clean(args)

    # 3. Features
    print(f"\n{'='*60}")
    print("STEP 3: Building features...")
    print(f"{'='*60}")
    feat_df = cmd_features(args, cleaned_df=cleaned)

    # 4. Train
    print(f"\n{'='*60}")
    print("STEP 4: Training models...")
    print(f"{'='*60}")
    model = cmd_train(args, feature_df=feat_df)

    # 5. Predict
    if args.round:
        print(f"\n{'='*60}")
        print(f"STEP 5: Predicting Round {args.round}...")
        print(f"{'='*60}")
        cmd_predict(args, model=model, feature_df=feat_df)


def main():
    parser = argparse.ArgumentParser(
        description="AFL Player Goal/Behind Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --scrape --start 2015 --end 2025
  python pipeline.py --update --round 5
  python pipeline.py --train
  python pipeline.py --predict --round 5
  python pipeline.py --evaluate
        """,
    )

    parser.add_argument("--scrape", action="store_true",
                        help="Scrape historical data from AFL Tables")
    parser.add_argument("--update", action="store_true",
                        help="Full update: scrape current season, rebuild, predict")
    parser.add_argument("--clean", action="store_true",
                        help="Clean and normalize raw data")
    parser.add_argument("--features", action="store_true",
                        help="Build feature matrix from cleaned data")
    parser.add_argument("--train", action="store_true",
                        help="Train prediction models")
    parser.add_argument("--predict", action="store_true",
                        help="Generate predictions for a round")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model on validation set")

    parser.add_argument("--start", type=int, help="Start year for scraping")
    parser.add_argument("--end", type=int, help="End year for scraping")
    parser.add_argument("--round", type=int, help="Round number for prediction")
    parser.add_argument("--year", type=int, help="Season year (default: current)")

    args = parser.parse_args()

    # No flags → show help
    if not any([args.scrape, args.update, args.clean, args.features,
                args.train, args.predict, args.evaluate]):
        parser.print_help()
        return

    config.ensure_dirs()

    if args.scrape:
        cmd_scrape(args)

    if args.update:
        cmd_update(args)
        return  # update runs everything

    if args.clean:
        cleaned = cmd_clean(args)

    if args.features:
        cmd_features(args)

    if args.train:
        cmd_train(args)

    if args.predict:
        if not args.round:
            print("Error: --predict requires --round N")
            return
        cmd_predict(args)

    if args.evaluate:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
