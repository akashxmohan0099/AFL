"""
Seed Supabase with data from local parquet files.

Usage:
  pip install supabase pandas pyarrow
  export SUPABASE_URL=https://your-project.supabase.co
  export SUPABASE_SERVICE_KEY=your-service-role-key
  python supabase/seed.py [--table TABLE_NAME]

Uses the service role key (not anon key) to bypass RLS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from supabase import create_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE = 500  # rows per upsert


def clean_for_json(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts safe for JSON/Supabase."""
    df = df.copy()
    # Convert timestamps to ISO strings
    for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    # Convert categories to strings
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)
    # Convert float columns that are actually integers (e.g. attendance 83493.0 → 83493)
    for col in df.columns:
        if df[col].dtype in ("float64", "float32"):
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                df[col] = df[col].astype("Int64")  # nullable int

    # Replace NaN/inf with None
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    # Convert numpy types to python types
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                if np.isnan(v) or np.isinf(v):
                    row[k] = None
                elif v == int(v):
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            elif isinstance(v, np.bool_):
                row[k] = bool(v)
            elif pd.isna(v):
                row[k] = None
    return records


def upsert_batch(table: str, records: list[dict], conflict_cols: str | None = None):
    """Upsert records in batches."""
    total = len(records)
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        q = supabase.table(table).upsert(batch)
        q.execute()
        print(f"  {table}: {min(i + BATCH_SIZE, total)}/{total}")


def seed_matches():
    print("\n=== matches ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")
    # Lowercase column names to match schema
    df.columns = [c.lower() for c in df.columns]
    records = clean_for_json(df)
    upsert_batch("matches", records)
    print(f"  Done: {len(records)} rows")


def seed_player_games():
    print("\n=== player_games ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "player_games.parquet")
    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]
    # Rename conflicting columns
    df = df.rename(columns={"if": "if"})
    # Drop rate columns (not stored in DB, can be derived)
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    df = df.drop(columns=rate_cols, errors="ignore")
    records = clean_for_json(df)
    upsert_batch("player_games", records)
    print(f"  Done: {len(records)} rows")


def seed_team_matches():
    print("\n=== team_matches ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "team_matches.parquet")
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"if": "if"})
    records = clean_for_json(df)
    upsert_batch("team_matches", records)
    print(f"  Done: {len(records)} rows")


def seed_odds():
    print("\n=== odds ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "odds.parquet")
    df.columns = [c.lower() for c in df.columns]
    records = clean_for_json(df)
    upsert_batch("odds", records)
    print(f"  Done: {len(records)} rows")


def seed_player_odds():
    print("\n=== player_odds ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "player_odds.parquet")
    df.columns = [c.lower() for c in df.columns]
    records = clean_for_json(df)
    upsert_batch("player_odds", records)
    print(f"  Done: {len(records)} rows")


def seed_umpires():
    print("\n=== umpires ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "umpires.parquet")
    df.columns = [c.lower() for c in df.columns]
    # Filter to only match_ids that exist in matches table
    matches_df = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")
    valid_ids = set(matches_df["match_id"].values)
    before = len(df)
    df = df[df["match_id"].isin(valid_ids)]
    if len(df) < before:
        print(f"  Filtered {before - len(df)} orphaned rows (match_id not in matches)")
    records = clean_for_json(df)
    upsert_batch("umpires", records)
    print(f"  Done: {len(records)} rows")


def seed_coaches():
    print("\n=== coaches ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "coaches.parquet")
    df.columns = [c.lower() for c in df.columns]
    # Filter to only match_ids that exist in matches table
    matches_df = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")
    valid_ids = set(matches_df["match_id"].values)
    before = len(df)
    df = df[df["match_id"].isin(valid_ids)]
    if len(df) < before:
        print(f"  Filtered {before - len(df)} orphaned rows (match_id not in matches)")
    records = clean_for_json(df)
    upsert_batch("coaches", records)
    print(f"  Done: {len(records)} rows")


def seed_weather():
    print("\n=== weather ===")
    df = pd.read_parquet(config.BASE_STORE_DIR / "weather.parquet")
    df.columns = [c.lower() for c in df.columns]
    # Drop the raw hourly blob
    df = df.drop(columns=["_raw_hourly"], errors="ignore")
    records = clean_for_json(df)
    upsert_batch("weather", records)
    print(f"  Done: {len(records)} rows")


def _load_sequential(subdir: str) -> pd.DataFrame:
    """Load all parquets from the latest run of each year in sequential store."""
    base = config.SEQUENTIAL_DIR / subdir
    if not base.exists():
        return pd.DataFrame()

    frames = []
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir():
            continue
        runs = sorted(year_dir.iterdir())
        if not runs:
            continue
        latest_run = runs[-1]
        for pq in sorted(latest_run.glob("*.parquet")):
            df = pd.read_parquet(pq)
            # Extract round number from filename (R01.parquet → 1)
            rname = pq.stem  # e.g. "R01"
            rnum = int(rname.replace("R", ""))
            df["round_number"] = rnum
            df["year"] = int(year_dir.name)
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def seed_predictions():
    print("\n=== predictions ===")
    df = _load_sequential("predictions")
    if df.empty:
        print("  No predictions found")
        return
    # Drop the original 'round' column (superseded by round_number from _load_sequential)
    df = df.drop(columns=["round"], errors="ignore")
    df.columns = [c.lower() for c in df.columns]
    # Drop any columns not in the schema
    schema_cols = {
        "year", "round_number", "match_id", "player", "team", "opponent", "venue",
        "player_role", "career_goal_avg",
        "predicted_goals", "predicted_behinds", "predicted_score",
        "lambda_goals", "lambda_behinds", "p_scorer", "p_scorer_raw",
        "p_1plus_goals", "p_2plus_goals", "p_3plus_goals",
        "p_1plus_goals_raw", "p_2plus_goals_raw", "p_3plus_goals_raw",
        "p_goals_0", "p_goals_1", "p_goals_2", "p_goals_3",
        "p_goals_4", "p_goals_5", "p_goals_6", "p_goals_7plus",
        "p_behinds_0", "p_behinds_1", "p_behinds_2", "p_behinds_3", "p_behinds_4plus",
        "conf_lower_gl", "conf_upper_gl",
        "predicted_disposals", "lambda_disposals",
        "p_10plus_disp", "p_15plus_disp", "p_20plus_disp", "p_25plus_disp", "p_30plus_disp",
        "conf_lower_di", "conf_upper_di",
        "predicted_marks", "lambda_marks", "p_mark_taker",
        "p_2plus_mk", "p_3plus_mk", "p_4plus_mk", "p_5plus_mk",
        "p_6plus_mk", "p_7plus_mk", "p_8plus_mk", "p_9plus_mk", "p_10plus_mk",
        "conf_lower_mk", "conf_upper_mk",
    }
    df = df[[c for c in df.columns if c in schema_cols]]
    # Deduplicate on primary key
    df = df.drop_duplicates(subset=["year", "round_number", "player", "team"], keep="last")
    records = clean_for_json(df)
    upsert_batch("predictions", records)
    print(f"  Done: {len(records)} rows")


def seed_outcomes():
    print("\n=== outcomes ===")
    df = _load_sequential("outcomes")
    if df.empty:
        print("  No outcomes found")
        return
    df.columns = [c.lower() for c in df.columns]
    # Keep only schema columns
    schema_cols = {"year", "round_number", "match_id", "player", "team",
                   "actual_goals", "actual_behinds", "actual_disposals", "actual_marks"}
    df = df[[c for c in df.columns if c in schema_cols]]
    # Deduplicate on primary key
    df = df.drop_duplicates(subset=["year", "round_number", "player", "team"], keep="last")
    records = clean_for_json(df)
    upsert_batch("outcomes", records)
    print(f"  Done: {len(records)} rows")


def seed_game_predictions():
    print("\n=== game_predictions ===")
    df = _load_sequential("game_predictions")
    if df.empty:
        print("  No game predictions found")
        return
    df.columns = [c.lower() for c in df.columns]
    records = clean_for_json(df)
    upsert_batch("game_predictions", records)
    print(f"  Done: {len(records)} rows")


def seed_fixtures():
    print("\n=== fixtures ===")
    frames = []
    for csv in sorted(config.FIXTURES_DIR.glob("*.csv")):
        # Parse round number from filename: round_1_2026.csv
        parts = csv.stem.split("_")
        rnum = int(parts[1])
        year = int(parts[2])
        df = pd.read_csv(csv)
        df["round_number"] = rnum
        df["year"] = year
        frames.append(df)

    if not frames:
        print("  No fixtures found")
        return

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["is_home"] = df["is_home"].astype(bool)
    records = clean_for_json(df)
    upsert_batch("fixtures", records)
    print(f"  Done: {len(records)} rows")


def seed_experiments():
    print("\n=== experiments ===")
    exp_dir = config.EXPERIMENTS_DIR
    if not exp_dir.exists():
        print("  No experiments dir")
        return

    records = []
    for f in sorted(exp_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        records.append({
            "filename": f.name,
            "label": data.get("label", f.stem),
            "data": data,
        })

    if records:
        upsert_batch("experiments", records)
    print(f"  Done: {len(records)} rows")


def seed_news():
    print("\n=== news ===")
    # Injuries
    inj_files = sorted(Path(config.NEWS_DIR / "injuries").glob("*.json"))
    if inj_files:
        latest = inj_files[-1]
        with open(latest) as fp:
            injuries = json.load(fp)
        records = []
        # Handle both formats: list of dicts or {"teams": {team: [players]}}
        if isinstance(injuries, list):
            for p in injuries:
                records.append({
                    "team": p.get("team", ""),
                    "player": p.get("player", ""),
                    "injury": p.get("injury", ""),
                    "severity": p.get("severity", 0),
                    "severity_label": p.get("severity_label", ""),
                    "estimated_return": p.get("estimated_return", ""),
                })
        else:
            for team, players in injuries.get("teams", {}).items():
                for p in players:
                    records.append({
                        "team": team,
                        "player": p.get("player", ""),
                        "injury": p.get("injury", ""),
                        "severity": p.get("severity", 0),
                        "severity_label": p.get("severity_label", ""),
                        "estimated_return": p.get("estimated_return", ""),
                    })
        if records:
            # Clear and re-insert
            supabase.table("news_injuries").delete().neq("id", -1).execute()
            upsert_batch("news_injuries", records)
        print(f"  Injuries: {len(records)} rows from {latest.name}")

    # Intel
    intel_path = config.NEWS_DIR / "intel" / "latest.json"
    if intel_path.exists():
        with open(intel_path) as fp:
            intel = json.load(fp)
        # Clear and re-insert
        supabase.table("news_intel").delete().neq("id", -1).execute()
        supabase.table("news_intel").insert({"data": intel}).execute()
        print(f"  Intel: loaded latest.json")

    # Team lists
    tl_dir = config.NEWS_DIR / "team_lists"
    if tl_dir.exists():
        for f in sorted(tl_dir.glob("*.json")):
            # Parse: round_1_2026.json
            parts = f.stem.split("_")
            rnum = int(parts[1])
            year = int(parts[2])
            with open(f) as fp:
                data = json.load(fp)
            supabase.table("news_team_lists").upsert({
                "year": year,
                "round_number": rnum,
                "data": data,
            }).execute()
        print(f"  Team lists: {len(list(tl_dir.glob('*.json')))} files")


def seed_simulations():
    """Seed Monte Carlo simulation results from pipeline output."""
    print("\n=== mc_simulations ===")
    sim_dir = Path(config.SEQUENTIAL_DIR) / "simulations"
    if not sim_dir.exists():
        print("  No simulations directory found")
        return

    frames = []
    for f in sorted(sim_dir.glob("*_mc.parquet")):
        # Parse: 2026_R01_mc.parquet
        parts = f.stem.split("_")
        year = int(parts[0])
        rnd = int(parts[1].replace("R", ""))
        df = pd.read_parquet(f)
        df["year"] = year
        df["round_number"] = rnd
        frames.append(df)

    if not frames:
        print("  No MC simulation files found")
        return

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]

    # Select the columns we need for the web
    keep_cols = [
        "year", "round_number", "match_id", "player", "team", "opponent",
        "predicted_goals", "predicted_disposals",
        "mc_p_1plus_goals", "mc_p_2plus_goals", "mc_p_3plus_goals", "mc_p_4plus_goals",
        "mc_p_10plus_disp", "mc_p_15plus_disp", "mc_p_20plus_disp",
        "mc_p_25plus_disp", "mc_p_30plus_disp",
        "direct_p_1plus_goals", "direct_p_2plus_goals", "direct_p_3plus_goals",
        "direct_p_20plus_disp", "direct_p_25plus_disp", "direct_p_30plus_disp",
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    records = clean_for_json(df)
    upsert_batch("mc_simulations", records)
    print(f"  Done: {len(records)} rows")


ALL_SEEDERS = {
    "matches": seed_matches,
    "player_games": seed_player_games,
    "team_matches": seed_team_matches,
    "predictions": seed_predictions,
    "outcomes": seed_outcomes,
    "game_predictions": seed_game_predictions,
    "fixtures": seed_fixtures,
    "odds": seed_odds,
    "player_odds": seed_player_odds,
    "umpires": seed_umpires,
    "coaches": seed_coaches,
    "weather": seed_weather,
    "experiments": seed_experiments,
    "news": seed_news,
    "mc_simulations": seed_simulations,
}

# Order matters — matches first (FK references)
SEED_ORDER = [
    "matches", "player_games", "team_matches", "weather", "odds", "umpires",
    "coaches", "player_odds", "predictions", "outcomes", "game_predictions",
    "fixtures", "experiments", "news", "mc_simulations",
]


def main():
    parser = argparse.ArgumentParser(description="Seed Supabase with AFL data")
    parser.add_argument("--table", "-t", help="Seed only this table")
    args = parser.parse_args()

    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"Project root: {PROJECT_ROOT}")

    if args.table:
        if args.table not in ALL_SEEDERS:
            print(f"Unknown table: {args.table}. Choose from: {', '.join(ALL_SEEDERS)}")
            sys.exit(1)
        ALL_SEEDERS[args.table]()
    else:
        for name in SEED_ORDER:
            ALL_SEEDERS[name]()

    print("\n=== Seed complete ===")


if __name__ == "__main__":
    main()
