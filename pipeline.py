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
    python pipeline.py --backtest --year 2024              # Walk-forward backtest
    python pipeline.py --diagnose --year 2024              # Diagnostic report on backtest
    python pipeline.py --sequential --year 2025 --reset-calibration
    python pipeline.py --scrape-live --year 2026             # Scrape FootyWire live stats
    python pipeline.py --daily --year 2026                   # Daily: scrape + clean + features
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from sklearn.metrics import roc_auc_score

import config
from clean import build_player_games
from features import build_features
from model import AFLScoringModel, AFLDisposalModel, AFLMarksModel, AFLGameWinnerModel, EloSystem
from store import LearningStore


def _augment_with_dl_features(df, feature_cols):
    """Augment feature matrix with embedding + GRU form features if available.

    Returns (augmented_df, extended_feature_cols).
    Non-destructive: if no embeddings/form features exist, returns originals unchanged.
    """
    added = []

    # Entity embeddings (Phase 2)
    if getattr(config, "EMBEDDING_ENABLED", False):
        try:
            from embeddings import augment_features as emb_augment
            df = emb_augment(df)
            emb_cols = [c for c in df.columns if c.startswith("emb_")]
            new_emb = [c for c in emb_cols if c not in feature_cols]
            added.extend(new_emb)
        except Exception as e:
            print(f"  (Embedding augmentation skipped: {e})")

    # GRU form embeddings (Phase 4)
    if getattr(config, "SEQUENCE_ENABLED", False):
        try:
            from sequence_model import augment_features as seq_augment
            df = seq_augment(df)
            form_cols = [c for c in df.columns if c.startswith("form_emb_")]
            new_form = [c for c in form_cols if c not in feature_cols]
            added.extend(new_form)
        except Exception as e:
            print(f"  (Sequence augmentation skipped: {e})")

    if added:
        feature_cols = feature_cols + added
        print(f"  DL features added: {len(added)} ({len(feature_cols)} total features)")

    return df, feature_cols


def _load_rosters(year):
    """Load team rosters JSON for a given year. Returns dict or None."""
    import json
    roster_path = config.FIXTURES_DIR / f"rosters_{year}.json"
    if not roster_path.exists():
        return None
    with open(roster_path) as f:
        rosters = json.load(f)
    print(f"  Loaded rosters: {sum(len(v) for v in rosters.values())} players, {len(rosters)} teams")
    return rosters


def _load_team_lists(year, round_num, player_names_by_team):
    """Load scraped team selections for a given round.

    Reads the JSON produced by news.scrape_team_selections() and resolves
    abbreviated names (e.g. "Wicks, S") to full names (e.g. "Wicks, Sam")
    using the known player names from player_games.parquet.

    Args:
        year: Season year
        round_num: Round number
        player_names_by_team: dict of {team: [full_name, ...]} from player_games

    Returns:
        dict of {team: [player_full_name, ...]} for the playing squad
        (selected + interchange, excluding emergencies), or None if not available.
    """
    import json

    team_list_path = config.NEWS_DIR / "team_lists" / f"round_{round_num}_{year}.json"
    if not team_list_path.exists():
        return None

    with open(team_list_path) as f:
        data = json.load(f)

    teams = data.get("teams", [])
    if not teams:
        return None

    # Build a cross-team index for fallback matching (traded players)
    all_players = []
    for names in player_names_by_team.values():
        all_players.extend(names)
    all_players = list(set(all_players))

    result = {}
    total_matched = 0
    total_players = 0

    for entry in teams:
        team = entry.get("team", "")
        # Map common alternative names
        if team == "Kangaroos":
            team = "North Melbourne"
        # Playing squad = selected + interchange (NOT emergencies), deduplicated
        squad_abbrev_raw = entry.get("selected", []) + entry.get("interchange", [])
        seen = set()
        squad_abbrev = []
        for p in squad_abbrev_raw:
            if p not in seen:
                seen.add(p)
                squad_abbrev.append(p)
        if not squad_abbrev:
            continue

        # Build lookup from abbreviated -> full name using known players for this team
        known = player_names_by_team.get(team, [])
        matched = _match_abbreviated_names(squad_abbrev, known, all_players_fallback=all_players)

        result[team] = matched
        total_matched += len(matched)
        total_players += len(squad_abbrev)

    if result:
        print(f"  Loaded team lists (R{round_num}): {total_matched}/{total_players} players matched across {len(result)} teams")
    return result if result else None


def _match_abbreviated_names(abbrev_names, full_names, all_players_fallback=None):
    """Match abbreviated names like 'Wicks, S' to full names like 'Wicks, Sam'.

    Handles several FootyWire name format quirks:
      - Abbreviated first names: "Wicks, S" → "Wicks, Sam"
      - Hyphenated surname abbreviations: "D-Uniacke, L" → "Davies-Uniacke, Luke"
      - Multi-word surname prefixes: "Chee, C Ah" → "Ah Chee, Callum"
      - Traded players: falls back to cross-team lookup

    Args:
        abbrev_names: list of "Last, F" or "Last, First" strings from team lists
        full_names: list of "Last, First" strings from player_games.parquet (team-specific)
        all_players_fallback: list of all player names across all teams (for traded players)

    Returns:
        list of matched full names (preserving order of abbrev_names)
    """
    import re

    def _build_index(names):
        """Build (last_lower, first_char_lower) -> [full_name, ...] index."""
        idx = {}
        for fn in names:
            if "," not in fn:
                continue
            last, first = fn.split(",", 1)
            last = last.strip().lower()
            first = first.strip()
            first_char = first[0].lower() if first else ""
            key = (last, first_char)
            if key not in idx:
                idx[key] = []
            idx[key].append(fn)
        return idx

    def _build_hyphen_index(names):
        """Build index for matching abbreviated hyphenated surnames.

        E.g. "D-Uniacke" should match "Davies-Uniacke".
        Key: (first_char_of_first_part, second_part_lower, first_char_of_first_name)
        """
        idx = {}
        for fn in names:
            if "," not in fn:
                continue
            last, first = fn.split(",", 1)
            last = last.strip()
            first = first.strip()
            first_char = first[0].lower() if first else ""
            if "-" in last:
                parts = last.split("-", 1)
                abbr_char = parts[0][0].lower()
                suffix = parts[1].lower()
                key = (abbr_char, suffix, first_char)
                if key not in idx:
                    idx[key] = []
                idx[key].append(fn)
        return idx

    def _normalize_surname(name):
        """Normalize multi-word surname prefixes.

        'Achkar, H El' → 'El Achkar, H', 'Chee, C Ah' → 'Ah Chee, C'
        """
        if "," not in name:
            return name
        last, first = name.split(",", 1)
        last = last.strip()
        first = first.strip()
        # Check if first name ends with a surname prefix (e.g. "H El", "C Ah")
        prefix_match = re.match(
            r"^(.+?)\s+(El|Ah|De|Di|Le|La|Van|Von|Mac|Mc|O')$",
            first, re.IGNORECASE,
        )
        if prefix_match:
            actual_first = prefix_match.group(1)
            prefix = prefix_match.group(2)
            return f"{prefix} {last}, {actual_first}"
        return name

    index = _build_index(full_names)
    fallback_index = _build_index(all_players_fallback) if all_players_fallback else {}
    hyphen_index = _build_hyphen_index(full_names)
    fallback_hyphen_index = _build_hyphen_index(all_players_fallback) if all_players_fallback else {}

    def _lookup(last, first_char, indexes, hyphen_indexes):
        """Try all matching strategies in order."""
        key = (last, first_char)
        for idx in indexes:
            cands = idx.get(key, [])
            if cands:
                return cands

        # Hyphenated abbreviation: "d-uniacke" → match "davies-uniacke"
        if "-" in last:
            parts = last.split("-", 1)
            abbr_char = parts[0].lower()
            suffix = parts[1].lower()
            if len(abbr_char) == 1:
                hkey = (abbr_char, suffix, first_char)
                for hidx in hyphen_indexes:
                    cands = hidx.get(hkey, [])
                    if cands:
                        return cands

        return []

    matched = []
    for abbr in abbrev_names:
        if "," not in abbr:
            parts = abbr.rsplit(" ", 1)
            if len(parts) == 2:
                abbr = f"{parts[1]}, {parts[0]}"
            else:
                continue

        # Normalize multi-word surnames (e.g. "Chee, C Ah" → "Ah Chee, C")
        abbr = _normalize_surname(abbr)

        last, first = abbr.split(",", 1)
        last = last.strip().lower()
        first = first.strip()
        first_char = first[0].lower() if first else ""

        # Try team-specific first, then cross-team fallback
        candidates = _lookup(
            last, first_char,
            [index, fallback_index],
            [hyphen_index, fallback_hyphen_index],
        )

        if len(candidates) == 1:
            matched.append(candidates[0])
        elif len(candidates) > 1:
            # Multiple matches — try full first-name prefix match
            best = None
            for c in candidates:
                c_first = c.split(",", 1)[1].strip()
                if c_first.lower().startswith(first.lower()):
                    best = c
                    break
            matched.append(best if best else candidates[0])
        # else: no match — true rookie not yet in player_games

    return matched


def _new_run_id(prefix=None):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if prefix:
        return f"{prefix}_{ts}"
    return ts


def _set_global_seed():
    np.random.seed(config.RANDOM_SEED)


def _build_player_predictions_for_winner_features(
    feature_df,
    feature_cols,
    scoring_model=None,
    disposal_model=None,
    marks_model=None,
):
    """Build player-level predictions for winner-model team aggregates.

    Each model must use its own fitted scaler. Reusing the scoring scaler for
    disposals/marks silently distorts the Poisson component and weakens the
    downstream winner features.
    """
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()

    from model import _prepare_features

    pred_df = feature_df[["match_id", "team"]].copy()
    added_cols = []

    if scoring_model is not None and getattr(scoring_model, "scaler", None) is not None:
        X_raw, _, X_scaled = _prepare_features(
            feature_df, feature_cols, scaler=scoring_model.scaler
        )
        if X_scaled is not None and X_scaled.shape[0] == X_raw.shape[0]:
            pred_goals, _, _ = scoring_model._ensemble_predict(X_raw, X_scaled, "goals")
            pred_df["predicted_goals"] = pred_goals
            added_cols.append("predicted_goals")

    if disposal_model is not None and getattr(disposal_model, "scaler", None) is not None:
        X_raw, _, X_scaled = _prepare_features(
            feature_df, feature_cols, scaler=disposal_model.scaler
        )
        if X_scaled is not None and X_scaled.shape[0] == X_raw.shape[0]:
            pred_df["predicted_disposals"] = disposal_model._predict_raw(X_raw, X_scaled)
            added_cols.append("predicted_disposals")

    if marks_model is not None and getattr(marks_model, "scaler", None) is not None:
        X_raw, _, X_scaled = _prepare_features(
            feature_df, feature_cols, scaler=marks_model.scaler
        )
        if X_scaled is not None and X_scaled.shape[0] == X_raw.shape[0]:
            pred_df["predicted_marks"] = marks_model._predict_raw(X_raw, X_scaled)
            added_cols.append("predicted_marks")

    if not added_cols:
        return pd.DataFrame()

    return pred_df


def _ensure_fixture_match_ids(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Ensure fixture rows have a stable per-match match_id.

    Fixture CSVs typically contain *two* rows per match (home and away). Many
    feature joins (opponent/team context) require both teams share the same
    match_id, so we assign a deterministic synthetic id per unique match key
    when match_id is missing.

    Match key = (date, venue, sorted(team, opponent)).
    """
    fx = fixtures.copy()

    # Normalize/ensure required columns exist.
    for col in ["team", "opponent", "venue", "date"]:
        if col not in fx.columns:
            raise ValueError(f"Fixture CSV missing required column: {col}")

    if "match_id" not in fx.columns:
        fx["match_id"] = np.nan

    fx["match_id"] = pd.to_numeric(fx["match_id"], errors="coerce")

    date_key = pd.to_datetime(fx["date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    venue_key = fx["venue"].astype(str).str.strip().str.lower()
    team = fx["team"].astype(str).str.strip()
    opp = fx["opponent"].astype(str).str.strip()

    # Symmetric team pairing so home/away rows map to the same match.
    t1 = np.where(team <= opp, team, opp)
    t2 = np.where(team <= opp, opp, team)

    key_df = pd.DataFrame(
        {"_date_key": date_key, "_venue_key": venue_key, "_t1": t1, "_t2": t2},
        index=fx.index,
    )

    # If an explicit match_id exists for a key, propagate it to all rows in that key.
    if fx["match_id"].notna().any():
        tmp = pd.concat([fx[["match_id"]], key_df], axis=1)
        nunique = tmp.groupby(["_date_key", "_venue_key", "_t1", "_t2"], observed=True)["match_id"].nunique(dropna=True)
        bad = nunique[nunique > 1]
        if not bad.empty:
            print("  Warning: multiple match_id values found for the same fixture match key; using the first non-null.")

        # Map: key -> first non-null match_id
        first_non_null = (
            tmp.groupby(["_date_key", "_venue_key", "_t1", "_t2"], observed=True)["match_id"]
            .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
            .reset_index()
        )
        key_to_existing = first_non_null.set_index(["_date_key", "_venue_key", "_t1", "_t2"])["match_id"]

        existing_filled = key_df.set_index(["_date_key", "_venue_key", "_t1", "_t2"]).index.map(key_to_existing)
        fx["match_id"] = fx["match_id"].fillna(pd.Series(existing_filled, index=fx.index))

    # Assign deterministic synthetic ids for any remaining missing match_ids.
    missing_mask = fx["match_id"].isna()
    if missing_mask.any():
        uniq = key_df[missing_mask].drop_duplicates().sort_values(["_date_key", "_venue_key", "_t1", "_t2"]).reset_index(drop=True)
        # Negative ids to avoid collisions with real match ids.
        uniq["_synthetic_match_id"] = -(np.arange(len(uniq)) + 1)
        key_to_synth = uniq.set_index(["_date_key", "_venue_key", "_t1", "_t2"])["_synthetic_match_id"]

        synth_ids = key_df[missing_mask].set_index(["_date_key", "_venue_key", "_t1", "_t2"]).index.map(key_to_synth)
        fx.loc[missing_mask, "match_id"] = pd.Series(synth_ids.values, index=fx.index[missing_mask])

    fx["match_id"] = fx["match_id"].astype(int)
    return fx


def cmd_scrape(args):
    """Scrape historical data from AFL Tables."""
    from scraper import scrape_seasons
    start = args.start or config.HISTORICAL_START_YEAR
    end = args.end or config.HISTORICAL_END_YEAR
    print(f"Scraping seasons {start}-{end}...")
    scrape_seasons(start, end, str(config.DATA_DIR))
    print("Scraping complete.")


def cmd_scrape_profiles(args):
    """Scrape player profile pages for physical attributes and career splits."""
    from scraper import scrape_player_profiles

    # Load recent player list from player_details
    pd_dir = config.DATA_DIR / "player_details"
    files = sorted(pd_dir.glob("player_details_*.csv"))
    if not files:
        print("No player_details files found. Run --scrape first.")
        return

    import pandas as _pd
    from clean import normalize_player_name

    # Get unique players from recent seasons
    recent_files = files[-3:]  # last 3 seasons
    dfs = [_pd.read_csv(f, low_memory=False) for f in recent_files]
    details = _pd.concat(dfs, ignore_index=True)
    # Exclude coach rows
    details = details[details["jumper"].astype(str).str.strip() != "C"]
    details["player"] = details["player"].apply(normalize_player_name)
    players = sorted(details["player"].unique())

    print(f"Found {len(players)} unique players from recent seasons")
    print(f"Output directory: {config.PLAYER_PROFILES_DIR}")

    result = scrape_player_profiles(players, str(config.PLAYER_PROFILES_DIR))
    print(f"\nProfile scraping complete:")
    print(f"  Profiles: {result['profiles']}")
    print(f"  Vs opponent records: {result['vs_opponent']}")
    print(f"  Vs venue records: {result['vs_venue']}")
    if result['failed']:
        print(f"  Failed: {len(result['failed'])} players")


def cmd_scrape_footywire(args):
    """Scrape FootyWire advanced stats (ED, DE%, CCL, SCL, TO, MG, TOG%)."""
    from scrape_footywire import scrape_all_seasons

    start = getattr(args, "start", None) or config.HISTORICAL_START_YEAR
    end = getattr(args, "end", None) or config.CURRENT_SEASON_YEAR

    print(f"Scraping FootyWire advanced stats for {start}-{end}...")
    result = scrape_all_seasons(start, end, str(config.FOOTYWIRE_DIR))

    if result is not None:
        print(f"\nFootyWire scraping complete: {len(result)} total player-match rows")
        # Build combined parquet
        from clean import build_footywire_parquet
        build_footywire_parquet()
    else:
        print("No data collected")


def cmd_scrape_live(args):
    """Scrape FootyWire basic match stats for the current season (incremental)."""
    from scrape_footywire_live import scrape_season

    year = getattr(args, "year", None) or config.CURRENT_SEASON_YEAR
    print(f"Scraping FootyWire live stats for {year} (incremental)...")
    result = scrape_season(year, incremental=True)
    if result:
        print(f"Live stats saved to {result}")
    else:
        print("No new matches to scrape.")


def cmd_scrape_news(args):
    """Scrape team selections, injury list, news articles, and process intel."""
    from news import scrape_team_selections, scrape_injury_list, scrape_afl_news
    from news import build_team_changes_parquet, build_injuries_parquet

    year = getattr(args, "year", None) or config.CURRENT_SEASON_YEAR

    print(f"Scraping team selections for {year}...")
    result = scrape_team_selections(year)
    if result:
        print(f"  Round {result['round_number']}: {len(result['teams'])} teams")
    else:
        print("  No team selections found")

    print("Scraping injury list...")
    injuries = scrape_injury_list()
    print(f"  {len(injuries)} injury records")

    print("Building parquets...")
    build_team_changes_parquet(year)
    build_injuries_parquet()

    print("Scraping news articles (RSS)...")
    articles = scrape_afl_news()
    print(f"  {len(articles)} articles")

    print("Processing intel signals...")
    from news_intel import process_articles, rebuild_latest
    process_articles(days=2)
    rebuild_latest()
    print("News & intel scraping complete.")


def cmd_daily(args):
    """Daily pipeline: scrape live data + clean + features + sequential learn + predict next.

    Full cycle after each round:
      1. Scrape new match results from FootyWire (incremental)
      2. Rebuild clean data & features
      3. For each newly completed round: train on all prior data, record outcomes, learn
      4. Predict the next upcoming round (from fixtures)
    """
    import json
    import time
    from scrape_footywire_live import scrape_season
    from features import add_dynamic_sample_weights
    from model import CalibratedPredictor, _prepare_features
    from analysis import generate_round_analysis

    year = getattr(args, "year", None) or config.CURRENT_SEASON_YEAR
    print(f"{'='*70}")
    print(f"  LIVE LEARNING PIPELINE — {year}")
    print(f"{'='*70}")

    # ── Step 1: Scrape new match results ──
    print("\n[1/7] Scraping new match results from FootyWire...")
    result = scrape_season(year, incremental=True)
    if result is None:
        print("  No new data scraped.")
    else:
        print(f"  Saved to {result}")

    # ── Step 2: Rebuild clean data & features ──
    print("\n[2/7] Rebuilding clean data...")
    cleaned = build_player_games(save=True)
    print(f"  Cleaned: {cleaned.shape[0]:,} rows")

    features_path = config.DATA_DIR / "features" / "feature_matrix.parquet"
    if features_path.exists():
        features_path.unlink()

    print("\n[3/7] Rebuilding features...")
    feature_df = build_features()
    print(f"  Features: {feature_df.shape[0]:,} rows, {feature_df.shape[1]} columns")

    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    # ── Step 2.5: Update weather forecasts for upcoming matches ──
    print("\n  Step 2.5: Fetching weather forecasts...")
    from weather import fetch_forecast_for_fixtures
    try:
        forecast_df = fetch_forecast_for_fixtures(year)
        if forecast_df is not None and not forecast_df.empty:
            print(f"    Fetched forecasts for {len(forecast_df)} matches")
    except Exception as e:
        print(f"    Weather forecast fetch failed: {e}")

    # ── Step 2.6: Fetch team news, articles & intel ──
    print("\n  Step 2.6: Fetching team news & intel...")
    try:
        from news import scrape_team_selections, scrape_injury_list, scrape_afl_news
        from news import build_team_changes_parquet, build_injuries_parquet
        sel = scrape_team_selections(year)
        if sel:
            print(f"    Team selections: Round {sel['round_number']}, {len(sel['teams'])} teams")
        inj = scrape_injury_list()
        print(f"    Injuries: {len(inj)} records")
        build_team_changes_parquet(year)
        build_injuries_parquet()
        articles = scrape_afl_news()
        print(f"    Articles: {len(articles)} scraped")
        from news_intel import process_articles, rebuild_latest
        process_articles(days=2)
        rebuild_latest()
    except Exception as e:
        print(f"    News/intel fetch failed: {e}")

    # ── Step 3: Determine which rounds need learning ──
    # Find completed rounds in feature matrix for this year
    season_df = feature_df[feature_df["year"] == year]
    completed_rounds = sorted(season_df["round_number"].dropna().unique())

    # Find which rounds already have outcomes saved (already learned)
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR)
    # Try to find or create a persistent run for this year's live learning
    live_run_id = f"live_{year}"
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=live_run_id)

    already_learned = set()
    for rnd in completed_rounds:
        outcomes = store.load_outcomes(year=year)
        if not outcomes.empty:
            # Check if this round's match_ids are in outcomes
            round_mids = set(season_df[season_df["round_number"] == rnd]["match_id"].unique())
            outcome_mids = set(outcomes["match_id"].unique())
            if round_mids.issubset(outcome_mids):
                already_learned.add(int(rnd))

    new_rounds = [r for r in completed_rounds if int(r) not in already_learned]

    if not new_rounds:
        print(f"\n[4/7] No new completed rounds to learn from.")
    else:
        print(f"\n[4/7] Learning from {len(new_rounds)} new round(s): {[int(r) for r in new_rounds]}")

        # Load team-match data for game winner model
        tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
        has_team_data = False
        team_match_df = pd.DataFrame()
        if tm_path.exists():
            team_match_df = pd.read_parquet(tm_path)
            team_match_df["date"] = pd.to_datetime(team_match_df["date"])
            has_team_data = True

        # Load or create isotonic calibrator
        use_isotonic = getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic"
        isotonic_calibrator = store.load_isotonic_calibrator() or CalibratedPredictor()
        _iso_accum = store.load_isotonic_accum() if use_isotonic else {}

        for rnd in new_rounds:
            rnd_int = int(rnd)
            rnd_start = time.time()

            # Split: train on everything before this round
            train_mask = (
                (feature_df["year"] < year)
                | ((feature_df["year"] == year) & (feature_df["round_number"] < rnd))
            )
            train_df = feature_df[train_mask].copy()
            test_mask = (feature_df["year"] == year) & (feature_df["round_number"] == rnd)
            test_df = feature_df[test_mask].copy()

            if len(train_df) < 50 or test_df.empty:
                print(f"  R{rnd_int:02d}: Skipping (insufficient data)")
                continue

            train_df = add_dynamic_sample_weights(train_df, year, rnd)

            # Train all models
            scoring_model = AFLScoringModel()
            scoring_model.train_backtest(train_df, feature_cols)

            disposal_model = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION)
            disposal_model.train_backtest(train_df, feature_cols)

            marks_model = AFLMarksModel(distribution=config.MARKS_DISTRIBUTION)
            marks_model.train_backtest(train_df, feature_cols)

            player_preds_for_gw = _build_player_predictions_for_winner_features(
                train_df,
                feature_cols,
                scoring_model=scoring_model,
                disposal_model=disposal_model,
                marks_model=marks_model,
            )

            # Game winner model
            game_preds = pd.DataFrame()
            if has_team_data:
                winner_model = AFLGameWinnerModel()
                tm_train = team_match_df[
                    (team_match_df["year"] < year)
                    | ((team_match_df["year"] == year) & (team_match_df["round_number"] < rnd))
                ].copy()
                tm_round = team_match_df[
                    (team_match_df["year"] == year) & (team_match_df["round_number"] == rnd)
                ].copy()

                if len(tm_train) >= 20 and not tm_round.empty:
                    try:
                        winner_model.train_backtest(
                            tm_train,
                            player_predictions_df=player_preds_for_gw if not player_preds_for_gw.empty else None,
                        )
                        _test_pp = _build_player_predictions_for_winner_features(
                            test_df,
                            feature_cols,
                            scoring_model=scoring_model,
                            disposal_model=disposal_model,
                            marks_model=marks_model,
                        )
                        all_pp = pd.concat([player_preds_for_gw, _test_pp], ignore_index=True) \
                            if not _test_pp.empty else player_preds_for_gw
                        game_preds = _predict_games_for_round(
                            winner_model, tm_train, tm_round,
                            player_predictions_df=all_pp if not all_pp.empty else None,
                            store=store,
                        )
                    except Exception as e:
                        print(f"  R{rnd_int:02d}: Game winner failed: {e}")

            # Predict
            pred_store = None if use_isotonic else store
            scoring_raw = scoring_model.predict_distributions(test_df, store=pred_store, feature_cols=feature_cols)
            disposal_raw = disposal_model.predict_distributions(test_df, store=pred_store, feature_cols=feature_cols)
            marks_raw = marks_model.predict_distributions(test_df, store=pred_store, feature_cols=feature_cols)

            merged_raw = _merge_predictions(scoring_raw, disposal_raw)
            merged_raw = _merge_predictions(merged_raw, marks_raw)

            merged_preds = merged_raw
            if use_isotonic:
                merged_preds = _apply_isotonic_calibration_to_predictions(merged_raw, isotonic_calibrator)

            outcomes = _build_outcomes(test_df)
            diagnostics = _build_diagnostics(merged_preds, outcomes, test_df=test_df)

            # Save
            store.save_predictions(year, rnd_int, merged_preds)
            store.save_outcomes(year, rnd_int, outcomes)
            store.save_diagnostics(year, rnd_int, diagnostics)
            if not game_preds.empty:
                store.save_game_predictions(year, rnd_int, game_preds)

            # Learn: update calibration
            _update_sequential_calibration(store, merged_raw, test_df)
            store.compute_calibration_adjustments()

            # Isotonic accumulation (key-based alignment to avoid row-order mismatch)
            if use_isotonic:
                iso_skip = getattr(config, "ISOTONIC_SKIP_TARGETS", set())
                _actual_cols2 = {}
                if "GL" in test_df.columns:
                    _actual_cols2["actual_goals"] = test_df["GL"]
                if "DI" in test_df.columns:
                    _actual_cols2["actual_disp"] = test_df["DI"]
                if "MK" in test_df.columns:
                    _actual_cols2["actual_marks"] = test_df["MK"]

                _iso_join2 = test_df[["player", "team", "match_id"]].copy()
                for col, vals in _actual_cols2.items():
                    _iso_join2[col] = vals.values
                _iso_merged2 = merged_raw.merge(
                    _iso_join2, on=["player", "team", "match_id"], how="inner"
                )

                actual_goals = _iso_merged2["actual_goals"].values if "actual_goals" in _iso_merged2.columns else None
                actual_disp = _iso_merged2["actual_disp"].values if "actual_disp" in _iso_merged2.columns else None
                actual_marks = _iso_merged2["actual_marks"].values if "actual_marks" in _iso_merged2.columns else None

                if actual_goals is not None and "1plus_goals" not in iso_skip:
                    p1_col = next((c for c in ["p_1plus_goals_raw", "p_scorer_raw", "p_scorer"]
                                   if c in _iso_merged2.columns), None)
                    if p1_col:
                        _iso_accum.setdefault("1plus_goals", {"preds": [], "actuals": []})
                        _iso_accum["1plus_goals"]["preds"].extend(_iso_merged2[p1_col].values.astype(float).tolist())
                        _iso_accum["1plus_goals"]["actuals"].extend((actual_goals >= 1).astype(int).tolist())

                if actual_goals is not None:
                    for threshold, name in [(2, "2plus_goals"), (3, "3plus_goals")]:
                        if name in iso_skip:
                            continue
                        raw_col = f"p_{threshold}plus_goals_raw"
                        p_col = raw_col if raw_col in _iso_merged2.columns else f"p_{threshold}plus_goals"
                        if p_col in _iso_merged2.columns:
                            _iso_accum.setdefault(name, {"preds": [], "actuals": []})
                            _iso_accum[name]["preds"].extend(_iso_merged2[p_col].values.astype(float).tolist())
                            _iso_accum[name]["actuals"].extend((actual_goals >= threshold).astype(int).tolist())

                if actual_disp is not None:
                    for t in config.DISPOSAL_THRESHOLDS:
                        p_col = f"p_{t}plus_disp"
                        if p_col in _iso_merged2.columns:
                            _iso_accum.setdefault(f"{t}plus_disp", {"preds": [], "actuals": []})
                            _iso_accum[f"{t}plus_disp"]["preds"].extend(_iso_merged2[p_col].values.astype(float).tolist())
                            _iso_accum[f"{t}plus_disp"]["actuals"].extend((actual_disp >= t).astype(int).tolist())

                if actual_marks is not None:
                    for t in config.MARKS_THRESHOLDS:
                        p_col = f"p_{t}plus_mk"
                        if p_col in _iso_merged2.columns:
                            _iso_accum.setdefault(f"{t}plus_mk", {"preds": [], "actuals": []})
                            _iso_accum[f"{t}plus_mk"]["preds"].extend(_iso_merged2[p_col].values.astype(float).tolist())
                            _iso_accum[f"{t}plus_mk"]["actuals"].extend((actual_marks >= t).astype(int).tolist())

                # Refit isotonic calibrators
                isotonic_min = getattr(config, "ISOTONIC_MIN_SAMPLES", 100)
                for tgt, data in _iso_accum.items():
                    if len(data["preds"]) >= isotonic_min:
                        isotonic_calibrator.fit(tgt, np.array(data["preds"]), np.array(data["actuals"]))
                store.save_isotonic_calibrator(isotonic_calibrator)
                store.save_isotonic_accum(_iso_accum)

            # Report
            elapsed = time.time() - rnd_start
            try:
                tm_round_data = team_match_df[
                    (team_match_df["year"] == year) & (team_match_df["round_number"] == rnd)
                ] if has_team_data else pd.DataFrame()
                analysis = generate_round_analysis(
                    year, rnd_int, merged_preds, outcomes,
                    game_preds, tm_round_data, test_df, store
                )
                store.save_analysis(year, rnd_int, analysis)
                summary = analysis.get("summary", {})
                mae = summary.get("goals_mae", float("nan"))
                tm = summary.get("threshold_metrics", {})
                br1 = tm.get("1plus_goals", {}).get("brier_score")
                mae_str = f"MAE={mae:.3f}" if mae and not np.isnan(mae) else "MAE=N/A"
                br1_str = f"Br1+={br1:.3f}" if br1 else "Br1+=N/A"
                print(f"  R{rnd_int:02d}  n={len(test_df):<4d} {mae_str}  {br1_str}  ({elapsed:.1f}s)  ✓ learned")
            except Exception as e:
                print(f"  R{rnd_int:02d}  n={len(test_df):<4d} ({elapsed:.1f}s)  ✓ learned  (analysis: {e})")

    # ── Step 4: Predict rounds with unplayed games ──
    # Handles both in-progress rounds (some games played, some remaining)
    # and the next fully-unplayed round.
    all_fixture_rounds = sorted([
        int(f.stem.split("_")[1])
        for f in config.FIXTURES_DIR.glob(f"round_*_{year}.csv")
    ])

    # Load played game keys from matches.parquet to detect in-progress rounds
    played_keys: set = set()
    matches_path = config.BASE_STORE_DIR / "matches.parquet"
    if matches_path.exists():
        try:
            all_matches = pd.read_parquet(matches_path)
            sm = all_matches[all_matches["year"] == year] if not all_matches.empty else pd.DataFrame()
            for _, m in sm.iterrows():
                if pd.notna(m.get("home_score")) and pd.notna(m.get("away_score")):
                    played_keys.add((str(m.get("home_team", "")), str(m.get("away_team", "")), int(m["round_number"])))
        except Exception as e:
            print(f"  Warning: Could not load match results for round detection: {e}")

    # Find rounds to predict: any round with unplayed games, stop after first fully-unplayed
    rounds_to_predict = []
    for r in all_fixture_rounds:
        fix_path = config.FIXTURES_DIR / f"round_{r}_{year}.csv"
        if not fix_path.exists():
            continue
        fix_df = pd.read_csv(fix_path)
        home_rows = fix_df[fix_df["is_home"] == 1]
        has_unplayed = any(
            (str(row["team"]), str(row["opponent"]), r) not in played_keys
            for _, row in home_rows.iterrows()
        )
        is_fully_unplayed = not any(
            (str(row["team"]), str(row["opponent"]), r) in played_keys
            for _, row in home_rows.iterrows()
        )
        if has_unplayed:
            rounds_to_predict.append(r)
            if is_fully_unplayed:
                break  # Stop after the first fully-unplayed round

    if rounds_to_predict:
        print(f"\n[5/7] Predicting round(s) with unplayed games: {rounds_to_predict}...")
        class PredArgs:
            pass
        for r_pred in rounds_to_predict:
            print(f"  Predicting Round {r_pred}...")
            pred_args = PredArgs()
            pred_args.round = r_pred
            pred_args.year = year
            preds = cmd_predict(pred_args)
            # Also save to the sequential store so the API can read them
            if preds is not None and not preds.empty:
                store.save_predictions(year, r_pred, preds)
                print(f"  Saved Round {r_pred} predictions to sequential store ({live_run_id})")
    else:
        print(f"\n[5/7] No upcoming rounds to predict (all fixtures played or no fixture files).")

    # ── Step 6: Game winner predictions for upcoming rounds ──
    if rounds_to_predict:
        print(f"\n[6/7] Generating game winner predictions for upcoming rounds...")
        tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
        if tm_path.exists():
            team_match_df = pd.read_parquet(tm_path)
            team_match_df["date"] = pd.to_datetime(team_match_df["date"])

            # Train game winner model on all available data
            winner_model = AFLGameWinnerModel()
            try:
                # Train player models for winner features
                all_train = feature_df[
                    (feature_df["year"] < year)
                    | ((feature_df["year"] == year)
                       & (feature_df["round_number"] < min(rounds_to_predict)))
                ].copy()
                all_train = add_dynamic_sample_weights(all_train, year, min(rounds_to_predict))

                scoring_model = AFLScoringModel()
                scoring_model.train_backtest(all_train, feature_cols)
                disposal_model = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION)
                disposal_model.train_backtest(all_train, feature_cols)
                marks_model = AFLMarksModel(distribution=config.MARKS_DISTRIBUTION)
                marks_model.train_backtest(all_train, feature_cols)

                player_preds_for_gw = _build_player_predictions_for_winner_features(
                    all_train, feature_cols,
                    scoring_model=scoring_model,
                    disposal_model=disposal_model,
                    marks_model=marks_model,
                )

                tm_train = team_match_df[
                    (team_match_df["year"] < year)
                    | ((team_match_df["year"] == year)
                       & (team_match_df["round_number"] < min(rounds_to_predict)))
                ].copy()

                winner_model.train_backtest(
                    tm_train,
                    player_predictions_df=player_preds_for_gw if not player_preds_for_gw.empty else None,
                )

                for r_pred in rounds_to_predict:
                    fix_path = config.FIXTURES_DIR / f"round_{r_pred}_{year}.csv"
                    if not fix_path.exists():
                        continue
                    fix_df = pd.read_csv(fix_path)
                    home_rows = fix_df[fix_df["is_home"] == 1]

                    # Build synthetic team_match rows from fixtures
                    synthetic_rows = []
                    for _, row in fix_df.iterrows():
                        # Generate a synthetic match_id from fixture info
                        date_str = str(row.get("date", "2026-01-01"))
                        date_val = pd.to_datetime(date_str, errors="coerce")
                        match_id = int(f"{year}{r_pred:02d}{len(synthetic_rows):02d}")

                        synthetic_rows.append({
                            "match_id": match_id,
                            "date": date_val,
                            "year": year,
                            "round_number": r_pred,
                            "venue": row.get("venue", ""),
                            "team": row["team"],
                            "opponent": row["opponent"],
                            "is_home": bool(row["is_home"]),
                            "is_finals": False,
                            "score": np.nan,
                            "opp_score": np.nan,
                            "margin": np.nan,
                            "GL": np.nan, "BH": np.nan, "DI": np.nan,
                            "IF": np.nan, "CL": np.nan, "CP": np.nan,
                            "TK": np.nan, "RB": np.nan, "MK": np.nan,
                            "result": np.nan,
                            "rest_days": np.nan,
                            "attendance": np.nan,
                        })

                    if synthetic_rows:
                        tm_round = pd.DataFrame(synthetic_rows)
                        # Ensure home rows share match_id
                        for _, hr in home_rows.iterrows():
                            h_team = hr["team"]
                            a_team = hr["opponent"]
                            h_mask = (tm_round["team"] == h_team) & (tm_round["opponent"] == a_team) & (tm_round["is_home"] == True)
                            a_mask = (tm_round["team"] == a_team) & (tm_round["opponent"] == h_team) & (tm_round["is_home"] == False)
                            if h_mask.any() and a_mask.any():
                                shared_id = tm_round.loc[h_mask, "match_id"].iloc[0]
                                tm_round.loc[a_mask, "match_id"] = shared_id

                        game_preds = _predict_games_for_round(
                            winner_model, tm_train, tm_round,
                            player_predictions_df=player_preds_for_gw if not player_preds_for_gw.empty else None,
                            store=store,
                        )
                        if not game_preds.empty:
                            store.save_game_predictions(year, r_pred, game_preds)
                            print(f"  R{r_pred}: {len(game_preds)} game predictions saved")
                            for _, gp in game_preds.iterrows():
                                winner = gp.get("predicted_winner", "?")
                                prob = gp.get("home_win_prob", 0.5)
                                print(f"    {gp.get('home_team','?')} vs {gp.get('away_team','?')}: "
                                      f"{winner} ({max(prob, 1-prob):.1%})")
                        else:
                            print(f"  R{r_pred}: No game predictions generated")
            except Exception as e:
                print(f"  Game winner prediction failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  No team_matches.parquet — skipping game winner predictions")
    else:
        print(f"\n[6/7] No upcoming rounds — skipping game winner predictions.")

    # ── Step 7: Monte Carlo simulation ready ──
    # The API runs Monte Carlo simulations on-demand per match request
    # using the stored predictions. No pre-computation needed.
    print(f"\n[7/7] Monte Carlo simulations ready (API runs on-demand from stored predictions)")

    print(f"\n{'='*70}")
    print(f"  LIVE LEARNING COMPLETE — {year}")
    print(f"  Sequential store: {config.SEQUENTIAL_DIR}")
    print(f"{'='*70}")


def cmd_clean(args):
    """Clean and normalize raw data into player_matches.parquet."""
    print("Cleaning raw data...")
    df = build_player_games(save=True)
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

    # Augment with DL features if available
    import json
    fc_path = config.FEATURES_DIR / "feature_columns.json"
    if fc_path.exists():
        with open(fc_path) as f:
            fc = json.load(f)
        feature_df, _ = _augment_with_dl_features(feature_df, fc)

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

    # Load disposal model (optional — predictions work without it)
    disp_model = None
    try:
        disp_model = AFLDisposalModel()
        disp_model.load()
    except (FileNotFoundError, Exception):
        print("  (Disposal model not available — skipping disposal thresholds)")

    # Load marks model (optional)
    marks_model = None
    try:
        marks_model = AFLMarksModel()
        marks_model.load()
    except (FileNotFoundError, Exception):
        print("  (Marks model not available — skipping marks thresholds)")

    # Load fixture file for the upcoming round
    fixture_path = config.FIXTURES_DIR / f"round_{round_num}_{year}.csv"

    rosters = _load_rosters(year)

    # Calibration source: prefer sequential calibrator (most recent), fallback to learning store.
    cal_store = None
    if getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic":
        try:
            cal_store = LearningStore(base_dir=config.SEQUENTIAL_DIR)
            if cal_store.load_isotonic_calibrator() is None:
                cal_store = LearningStore()
        except Exception:
            cal_store = None

    if fixture_path.exists():
        fixtures = pd.read_csv(fixture_path)
        print(f"Loaded fixtures from {fixture_path}")
        predictions = _predict_from_fixtures(
            model, fixtures, year, round_num, disp_model=disp_model,
            marks_model=marks_model, rosters=rosters, store=cal_store
        )
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
        predictions = model.predict_distributions(
            round_df, store=cal_store, feature_cols=model.feature_cols
        )

        # Merge disposal predictions if available
        if disp_model is not None:
            try:
                disp_preds = disp_model.predict_distributions(
                    round_df, store=cal_store, feature_cols=disp_model.feature_cols
                )
                predictions = _merge_predictions(predictions, disp_preds)
            except Exception as e:
                print(f"  Warning: Disposal predictions failed: {e}")

        # Merge marks predictions if available
        if marks_model is not None:
            try:
                marks_preds = marks_model.predict_distributions(
                    round_df, store=cal_store, feature_cols=marks_model.feature_cols
                )
                predictions = _merge_predictions(predictions, marks_preds)
            except Exception as e:
                print(f"  Warning: Marks predictions failed: {e}")

    # Derive P(2+) and P(3+) goals from PMF columns (fallback only).
    if "p_goals_0" in predictions.columns and "p_goals_1" in predictions.columns:
        if "p_2plus_goals" not in predictions.columns:
            predictions["p_2plus_goals"] = (
                1 - predictions["p_goals_0"] - predictions["p_goals_1"]
            ).clip(0, 1)
        if "p_goals_2" in predictions.columns and "p_3plus_goals" not in predictions.columns:
            predictions["p_3plus_goals"] = (
                1 - predictions["p_goals_0"]
                - predictions["p_goals_1"]
                - predictions["p_goals_2"]
            ).clip(0, 1)

    # Save predictions
    config.ensure_dirs()
    out_dir = config.PREDICTIONS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"round_{round_num}_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")

    # Save threshold CSV (year-scoped)
    _save_threshold_csv(predictions, round_num, out_dir)

    # Backward-compatible: also write legacy top-level outputs for the current season.
    if int(year) == int(config.CURRENT_SEASON_YEAR):
        legacy_path = config.PREDICTIONS_DIR / f"round_{round_num}_predictions.csv"
        predictions.to_csv(legacy_path, index=False)
        print(f"  (Legacy) Predictions saved to {legacy_path}")
        _save_threshold_csv(predictions, round_num, config.PREDICTIONS_DIR)

    # Save to LearningStore
    store = LearningStore(run_id=getattr(args, "run_id", None))
    store.save_predictions(year, round_num, predictions)
    print(f"  Predictions also saved to LearningStore")

    # Display top predicted scorers
    print(f"\n{'='*70}")
    print(f"TOP PREDICTED SCORERS — Round {round_num}, {year}")
    print(f"{'='*70}")
    top = predictions.head(20)
    for _, row in top.iterrows():
        p_sc = row.get("p_scorer", 0)
        print(
            f"  {row['player']:30s} {row['team']:15s} vs {row['opponent']:15s} "
            f"P={p_sc:.2f} GL={row['predicted_goals']:.2f} BH={row['predicted_behinds']:.2f} "
            f"Score={row['predicted_score']:.1f}"
        )

    # Full team sheet display grouped by team
    if "team" in predictions.columns and "p_scorer" in predictions.columns:
        print(f"\n{'='*70}")
        print(f"TEAM SHEETS — Round {round_num}, {year}")
        print(f"{'='*70}")
        for team_name, team_df in predictions.groupby("team", observed=True):
            team_sorted = team_df.sort_values("p_scorer", ascending=False)
            opp = team_sorted["opponent"].iloc[0] if "opponent" in team_sorted.columns else "?"
            print(f"\n  {team_name} vs {opp}")
            print(f"  {'Player':30s}  {'P(scorer)':>9s}  {'Pred GL':>7s}  {'Pred BH':>7s}  {'Score':>6s}")
            for _, row in team_sorted.iterrows():
                print(
                    f"  {row['player']:30s}  {row['p_scorer']:9.3f}  "
                    f"{row['predicted_goals']:7.2f}  {row['predicted_behinds']:7.2f}  "
                    f"{row['predicted_score']:6.1f}"
                )

    # Threshold probability table
    _display_threshold_probabilities(predictions, round_num, year)

    return predictions


def _predict_from_fixtures(model, fixtures, year, round_num, disp_model=None, marks_model=None, rosters=None, store=None):
    """Build features for upcoming fixtures and generate predictions.

    Fixture CSV columns: team, opponent, venue, date, is_home
    Optionally: players (comma-separated player names per team)

    Player resolution tiers:
      0. CSV 'players' column (per-row team sheet overrides)
      1. Scraped team lists (selected + interchange = playing 22-23)
      2. Roster JSON (full squad from rosters_{year}.json)
      3. Last historical match lineup (fallback)
    """
    # Use cleaned player games so build_features has required columns
    pg_path = config.BASE_STORE_DIR / "player_games.parquet"
    if not pg_path.exists():
        print("No player_games.parquet found. Run --clean first.")
        return pd.DataFrame()
    stats = pd.read_parquet(pg_path)
    if stats.empty:
        print("player_games.parquet is empty. Run --clean first.")
        return pd.DataFrame()

    date_col = "date" if "date" in stats.columns else None
    if date_col is None:
        print("player_games.parquet missing required 'date' column.")
        return pd.DataFrame()
    stats[date_col] = pd.to_datetime(stats[date_col], errors="coerce")
    stats = stats.sort_values(date_col)

    # Exclude DNP rows from lineup/history lookups.
    if "did_not_play" in stats.columns:
        stats_real = stats[~stats["did_not_play"].fillna(False).astype(bool)].copy()
    else:
        stats_real = stats

    # Ensure home/away fixture rows share a stable match_id so opponent/team joins work.
    fixtures = _ensure_fixture_match_ids(fixtures)
    fixture_match_ids = set(fixtures["match_id"].astype(int).unique().tolist())

    # Load scraped team lists (selected + interchange = actual playing squad).
    # Build per-team player name index for abbreviated name resolution.
    all_teams = fixtures["team"].unique().tolist()
    player_names_by_team = {}
    for t in all_teams:
        names = stats_real[stats_real["team"] == t]["player"].unique().tolist()
        player_names_by_team[t] = names
    team_lists = _load_team_lists(year, round_num, player_names_by_team)

    # Build synthetic "future match" rows (one per player per fixture team),
    # then run build_features on history + synthetic rows so rolling features are non-empty.
    synthetic_rows = []

    for i, fixture in fixtures.iterrows():
        team = fixture["team"]
        opponent = fixture["opponent"]
        venue = fixture["venue"]
        fixture_date = pd.to_datetime(fixture.get("date"), errors="coerce")
        synthetic_match_id = int(fixture["match_id"])

        # Get lineup: Tier 0 → CSV players, Tier 1 → scraped team list,
        #             Tier 2 → roster JSON, Tier 3 → last match
        if "players" in fixture and pd.notna(fixture.get("players")):
            players = [p.strip() for p in str(fixture["players"]).split(",")]
        elif team_lists and team in team_lists:
            players = team_lists[team]
        elif rosters and team in rosters:
            players = rosters[team]
        else:
            recent = stats_real[stats_real["team"] == team].sort_values(date_col)
            last_match = recent["match_id"].iloc[-1] if len(recent) > 0 else None
            if last_match:
                players = recent[recent["match_id"] == last_match]["player"].tolist()
            else:
                continue

        # For each player, create a synthetic future row using their last match as a schema template.
        for player in players:
            # Try current team first
            player_history = stats_real[
                (stats_real["player"] == player) & (stats_real["team"] == team)
            ].sort_values(date_col)

            if len(player_history) < 1:
                # Transferred player — use history from previous team(s)
                player_history = stats_real[
                    stats_real["player"] == player
                ].sort_values(date_col)
                if len(player_history) < 1:
                    continue  # True rookie, no AFL history

            # Use last match as template for required columns (player_id, jumper, etc).
            last_row = player_history.iloc[-1:].copy()
            last_date = pd.to_datetime(last_row["date"].iloc[0], errors="coerce")

            # Pre-game career counters should advance beyond the last played match.
            # In player_games.parquet they represent "pre this match", so for a future
            # match we approximate by adding the last match contribution.
            try:
                prev_games_pre = float(last_row["career_games_pre"].iloc[0])
                prev_goals_pre = float(last_row["career_goals_pre"].iloc[0])
                last_gl = float(last_row["GL"].iloc[0]) if "GL" in last_row.columns else 0.0
                games_pre = prev_games_pre + 1.0
                goals_pre = prev_goals_pre + (0.0 if np.isnan(last_gl) else last_gl)
                last_row["career_games_pre"] = games_pre
                last_row["career_goals_pre"] = goals_pre
                last_row["career_goal_avg_pre"] = (goals_pre / games_pre) if games_pre > 0 else last_row["career_goal_avg_pre"]
            except Exception:
                pass

            last_row["match_id"] = synthetic_match_id
            if pd.notna(fixture_date):
                last_row["date"] = fixture_date
                # Approximate age at fixture date.
                if "age_years" in last_row.columns and pd.notna(last_date):
                    delta_days = (fixture_date - last_date).days
                    if np.isfinite(delta_days):
                        last_row["age_years"] = float(last_row["age_years"].iloc[0]) + (delta_days / 365.25)

            last_row["team"] = team  # Correct team for transfers
            last_row["venue"] = venue
            last_row["opponent"] = opponent
            last_row["round_number"] = int(round_num)
            last_row["year"] = int(year)
            last_row["is_home"] = int(fixture.get("is_home", 1))

            # Upcoming fixtures are not finals by default.
            if "is_finals" in last_row.columns:
                last_row["is_finals"] = 0
            if "round_label" in last_row.columns:
                last_row["round_label"] = f"R{int(round_num)}"

            # Ensure rule-regime columns reflect the fixture year (history rows already correct).
            if "season_era" in last_row.columns:
                last_row["season_era"] = int(config.ERA_MAP.get(int(year), config.CURRENT_PREDICTION_ERA))
            if "is_covid_season" in last_row.columns:
                last_row["is_covid_season"] = int(int(year) == int(config.COVID_SEASON_YEAR))
            if "quarter_length_ratio" in last_row.columns:
                last_row["quarter_length_ratio"] = float(
                    config.COVID_QUARTER_LENGTH_RATIO if int(year) == int(config.COVID_SEASON_YEAR) else 1.0
                )

            # This row represents a future match, so blank current-match stat columns
            # to avoid contaminating any aggregations that do not use shift(1).
            stat_cols = [
                "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF", "CL", "CG",
                "FF", "FA", "BR", "CP", "UP", "CM", "MI", "one_pct", "BO", "GA",
                "pct_played",
                "q1_goals", "q1_behinds", "q2_goals", "q2_behinds",
                "q3_goals", "q3_behinds", "q4_goals", "q4_behinds",
            ]
            rate_cols = [f"{c}_rate" for c in [
                "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF", "CL", "CG",
                "FF", "FA", "BR", "CP", "UP", "CM", "MI", "one_pct", "BO", "GA",
            ]]
            for c in stat_cols + rate_cols:
                if c in last_row.columns:
                    last_row[c] = np.nan

            # Fixture rows are assumed to play (not DNP).
            if "did_not_play" in last_row.columns:
                last_row["did_not_play"] = False

            synthetic_rows.append(last_row)

    if rosters:
        if "is_home" in fixtures.columns:
            n_fixture_teams = fixtures[fixtures["is_home"] == 1]["team"].nunique()
        else:
            n_fixture_teams = fixtures["team"].nunique()
        print(f"  Roster: {len(synthetic_rows)} players with history "
              f"(from {n_fixture_teams} matches)")

    if not synthetic_rows:
        return pd.DataFrame()

    future_df = pd.concat(synthetic_rows, ignore_index=True)
    if "date" in future_df.columns:
        future_df["date"] = pd.to_datetime(future_df["date"], errors="coerce")

    # Build features on (history + synthetic future rows), then keep only fixtures.
    full_df = pd.concat([stats_real, future_df], ignore_index=True)
    full_df = build_features(full_df, save=False)
    pred_df = full_df[
        (full_df["match_id"].isin(fixture_match_ids))
        & (full_df["year"] == int(year))
        & (full_df["round_number"] == int(round_num))
    ].copy()

    # Predict with full distributions
    predictions = model.predict_distributions(
        pred_df, store=store, feature_cols=model.feature_cols
    )

    # Merge disposal predictions if available
    if disp_model is not None:
        try:
            disp_preds = disp_model.predict_distributions(
                pred_df, store=store, feature_cols=disp_model.feature_cols
            )
            predictions = _merge_predictions(predictions, disp_preds)
        except Exception as e:
            print(f"  Warning: Disposal predictions failed: {e}")

    # Merge marks predictions if available
    if marks_model is not None:
        try:
            marks_preds = marks_model.predict_distributions(
                pred_df, store=store, feature_cols=marks_model.feature_cols
            )
            predictions = _merge_predictions(predictions, marks_preds)
        except Exception as e:
            print(f"  Warning: Marks predictions failed: {e}")

    return predictions


# ------------------------------------------------------------------
# Threshold probability helpers
# ------------------------------------------------------------------

def _fmt_prob(p):
    """Format a probability for threshold display."""
    if pd.isna(p) or p < 0.01:
        return "  -  "
    return f"{p:5.2f}"


def _display_threshold_probabilities(preds, round_num, year):
    """Display per-player threshold probabilities grouped by match."""
    has_goals = "p_2plus_goals" in preds.columns
    has_disp = "p_15plus_disp" in preds.columns

    if not has_goals and not has_disp:
        return

    print(f"\n{'='*70}")
    print(f"PLAYER THRESHOLD PROBABILITIES — Round {round_num}, {year}")
    print(f"{'='*70}")

    # Group by match using sorted team pairs (works for both fixture and
    # historical paths regardless of match_id values)
    preds = preds.copy()
    preds["_match_key"] = preds.apply(
        lambda r: tuple(sorted([str(r["team"]), str(r["opponent"])])), axis=1
    )

    for _, match_df in preds.groupby("_match_key", sort=False):
        match_sorted = match_df.sort_values("p_scorer", ascending=False)

        # Match header
        teams = match_sorted[["team", "opponent"]].drop_duplicates()
        team1 = teams.iloc[0]["team"]
        team2 = teams.iloc[0]["opponent"]
        venue = match_sorted["venue"].iloc[0] if "venue" in match_sorted.columns else ""
        venue_str = f"  ({venue})" if venue else ""
        print(f"\n  {team1} vs {team2}{venue_str}")

        # Column header
        header = f"  {'Player':30s}"
        if has_goals:
            header += f"  {'1+GL':>5s}  {'2+GL':>5s}  {'3+GL':>5s}"
        if has_disp:
            header += f"  {'15+DI':>5s}  {'20+DI':>5s}  {'25+DI':>5s}  {'30+DI':>5s}"
        print(header)

        for _, row in match_sorted.iterrows():
            line = f"  {row['player']:30s}"
            if has_goals:
                line += f"  {_fmt_prob(row.get('p_scorer', 0))}"
                line += f"  {_fmt_prob(row.get('p_2plus_goals', 0))}"
                line += f"  {_fmt_prob(row.get('p_3plus_goals', 0))}"
            if has_disp:
                line += f"  {_fmt_prob(row.get('p_15plus_disp', 0))}"
                line += f"  {_fmt_prob(row.get('p_20plus_disp', 0))}"
                line += f"  {_fmt_prob(row.get('p_25plus_disp', 0))}"
                line += f"  {_fmt_prob(row.get('p_30plus_disp', 0))}"
            print(line)


def _save_threshold_csv(preds, round_num, out_dir):
    """Save a threshold-focused CSV with goal and disposal probabilities."""
    cols = ["player", "team", "opponent", "venue"]
    # Goal thresholds
    for c in ["p_scorer", "p_2plus_goals", "p_3plus_goals"]:
        if c in preds.columns:
            cols.append(c)
    # Disposal thresholds (skip 10+ — nearly everyone hits it)
    for t in [15, 20, 25, 30]:
        col = f"p_{t}plus_disp"
        if col in preds.columns:
            cols.append(col)
    # Point estimates
    for c in ["predicted_goals", "predicted_disposals"]:
        if c in preds.columns:
            cols.append(c)

    available_cols = [c for c in cols if c in preds.columns]
    threshold_df = preds[available_cols].copy()

    # Rename p_scorer to p_1plus_goals for clarity in the CSV
    if "p_scorer" in threshold_df.columns:
        threshold_df = threshold_df.rename(columns={"p_scorer": "p_1plus_goals"})

    sort_col = "p_1plus_goals" if "p_1plus_goals" in threshold_df.columns else "player"
    threshold_df = threshold_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    out_path = out_dir / f"round_{round_num}_thresholds.csv"
    threshold_df.to_csv(out_path, index=False)
    print(f"  Threshold probabilities saved to {out_path}")


def _update_calibration_for_round(store, pred_goals, actual_goals, scorer_prob):
    """Update calibration state with data from a single backtest round."""
    # Bucket predictions into probability bins and track hit rates
    buckets = np.arange(0.05, 1.0, 0.1).round(2)
    rows = []

    # 1+ goals calibration
    for bucket in buckets:
        lo = bucket - 0.05
        hi = bucket + 0.05
        mask = (scorer_prob >= lo) & (scorer_prob < hi)
        if mask.sum() > 0:
            rows.append({
                "target": "1plus_goals",
                "probability_bucket": float(bucket),
                "predicted": int(mask.sum()),
                "occurred": int((actual_goals[mask] >= 1).sum()),
            })

    # 2+ goals calibration (use Poisson CDF on pred_goals)
    from scipy.stats import poisson as poisson_dist
    p_2plus = np.array([1 - poisson_dist.cdf(1, max(mu, 0.01)) for mu in pred_goals])
    for bucket in buckets:
        lo = bucket - 0.05
        hi = bucket + 0.05
        mask = (p_2plus >= lo) & (p_2plus < hi)
        if mask.sum() > 0:
            rows.append({
                "target": "2plus_goals",
                "probability_bucket": float(bucket),
                "predicted": int(mask.sum()),
                "occurred": int((actual_goals[mask] >= 2).sum()),
            })

    if rows:
        cal_df = pd.DataFrame(rows)
        store.update_calibration(cal_df)


def cmd_backtest(args):
    """Walk-forward backtesting: for each round in the target season,
    train on all prior data, predict that round, and record accuracy."""
    import json
    from analysis import _compute_threshold_metrics

    year = args.year
    if not year:
        print("Error: --backtest requires --year YYYY")
        return

    run_id = getattr(args, "run_id", None) or _new_run_id(prefix=f"backtest_{year}")

    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    if not feat_path.exists():
        print("No feature matrix found. Run --features first.")
        return

    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    if not feat_cols_path.exists():
        print("No feature_columns.json found. Run --features first.")
        return

    feature_df = pd.read_parquet(feat_path)
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    # Augment with DL features if available
    feature_df, feature_cols = _augment_with_dl_features(feature_df, feature_cols)

    # Check we have enough training history
    min_train_year = year - config.BACKTEST_TRAIN_MIN_YEARS
    available_years = sorted(feature_df["year"].unique())
    if not any(y <= min_train_year for y in available_years):
        print(f"Not enough history for backtesting year {year}.")
        print(f"  Need data from at least {min_train_year}, have: {available_years}")
        return

    # Get rounds in the target season
    season_df = feature_df[feature_df["year"] == year].copy()
    if season_df.empty:
        print(f"No data for season {year}.")
        return

    rounds = sorted(season_df["round_number"].dropna().unique())
    print(f"\nWalk-forward backtest for {year}")
    print(f"  Rounds to test: {len(rounds)}")
    print(f"  Training data starts: {min(available_years)}")
    print(f"  Run ID: {run_id}")
    print()

    results = []
    player_rows = []
    store = LearningStore(run_id=run_id)
    all_p_1plus = []
    all_p_2plus = []
    all_p_3plus = []
    all_actual_goals = []

    for rnd in rounds:
        rnd_int = int(rnd)

        # Train set: all data before this round in this year + all prior years
        train_mask = (
            (feature_df["year"] < year)
            | ((feature_df["year"] == year) & (feature_df["round_number"] < rnd))
        )
        train_df = feature_df[train_mask].copy()

        # Test set: this round only
        test_mask = (feature_df["year"] == year) & (feature_df["round_number"] == rnd)
        test_df = feature_df[test_mask].copy()

        if len(train_df) < 20 or test_df.empty:
            continue

        # Train model
        model = AFLScoringModel()
        model.train_backtest(train_df, feature_cols)

        # Predict
        from model import _prepare_features
        X_test_raw, X_test_clean, X_test_scaled = _prepare_features(
            test_df, feature_cols, scaler=model.scaler
        )
        pred_goals, scorer_prob, lambda_if_scorer = model._ensemble_predict(
            X_test_raw, X_test_scaled, "goals"
        )
        pred_behinds, _, _ = model._ensemble_predict(X_test_raw, X_test_scaled, "behinds")

        actual_goals = test_df["GL"].values
        actual_behinds = test_df["BH"].values
        baseline = test_df["career_goal_avg_pre"].fillna(0).values

        mae = float(np.mean(np.abs(actual_goals - pred_goals)))
        baseline_mae = float(np.mean(np.abs(actual_goals - baseline)))
        improvement = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else 0.0
        mean_pred = float(pred_goals.mean())
        mean_actual = float(actual_goals.mean())

        # Ranking metrics: scorer AUC and precision@20
        actual_scored = (actual_goals >= 1).astype(int)
        try:
            scorer_auc = roc_auc_score(actual_scored, scorer_prob)
        except ValueError:
            scorer_auc = float("nan")

        # Precision@20: of top 20 by P(scorer), how many actually scored?
        if len(scorer_prob) >= 20:
            top20_idx = np.argsort(scorer_prob)[::-1][:20]
            precision_at_20 = float(actual_scored[top20_idx].mean())
        else:
            precision_at_20 = float("nan")

        # Probabilistic thresholds for calibration diagnostics
        lam = np.clip(lambda_if_scorer, 0.01, None)
        p_1plus = np.clip(scorer_prob, 0, 1)
        p_2plus = np.clip(scorer_prob * (1 - poisson_dist.cdf(1, lam)), 0, 1)
        p_3plus = np.clip(scorer_prob * (1 - poisson_dist.cdf(2, lam)), 0, 1)
        all_p_1plus.append(p_1plus)
        all_p_2plus.append(p_2plus)
        all_p_3plus.append(p_3plus)
        all_actual_goals.append(actual_goals)

        # Collect player-level predictions
        n_train = len(train_df)
        error = pred_goals - actual_goals
        abs_error = np.abs(error)
        baseline_abs_error = np.abs(baseline - actual_goals)
        beat_baseline = (abs_error < baseline_abs_error).astype(int)

        rnd_players = pd.DataFrame({
            "player": test_df["player"].values,
            "team": test_df["team"].values,
            "opponent": test_df["opponent"].values,
            "venue": test_df["venue"].values,
            "round": rnd_int,
            "year": year,
            "is_home": test_df["is_home"].values if "is_home" in test_df.columns else 0,
            "player_role": test_df["player_role"].values if "player_role" in test_df.columns else "general",
            "p_scorer": np.round(scorer_prob, 4),
            "predicted_goals": np.round(pred_goals, 4),
            "predicted_behinds": np.round(pred_behinds, 4),
            "actual_goals": actual_goals,
            "actual_behinds": actual_behinds,
            "career_goal_avg": baseline,
            "error": np.round(error, 4),
            "abs_error": np.round(abs_error, 4),
            "baseline_abs_error": np.round(baseline_abs_error, 4),
            "beat_baseline": beat_baseline,
            "n_train": n_train,
        })
        player_rows.append(rnd_players)

        # --- LearningStore: save predictions, outcomes, diagnostics ---
        pred_record = pd.DataFrame({
            "player": test_df["player"].values,
            "team": test_df["team"].values,
            "opponent": test_df["opponent"].values,
            "p_scorer": np.round(scorer_prob, 4),
            "predicted_goals": np.round(pred_goals, 4),
            "predicted_behinds": np.round(pred_behinds, 4),
        })
        store.save_predictions(year, rnd_int, pred_record)

        outcome_record = pd.DataFrame({
            "player": test_df["player"].values,
            "team": test_df["team"].values,
            "actual_goals": actual_goals,
            "actual_behinds": actual_behinds,
        })
        store.save_outcomes(year, rnd_int, outcome_record)

        diag_record = pd.DataFrame({
            "player": test_df["player"].values,
            "team": test_df["team"].values,
            "error": np.round(error, 4),
            "abs_error": np.round(abs_error, 4),
            "beat_baseline": beat_baseline,
            "mae": mae,
            "baseline_mae": baseline_mae,
        })
        store.save_diagnostics(year, rnd_int, diag_record)

        # Update calibration: bucket predictions and track hit rates
        _update_calibration_for_round(store, pred_goals, actual_goals, scorer_prob)

        results.append({
            "round": rnd_int,
            "n_players": len(test_df),
            "n_train": len(train_df),
            "mae": round(mae, 4),
            "baseline_mae": round(baseline_mae, 4),
            "improvement_pct": round(improvement, 1),
            "mean_predicted": round(mean_pred, 3),
            "mean_actual": round(mean_actual, 3),
            "scorer_auc": round(scorer_auc, 4) if not np.isnan(scorer_auc) else None,
            "precision_at_20": round(precision_at_20, 4) if not np.isnan(precision_at_20) else None,
        })

        auc_str = f"AUC={scorer_auc:.3f}" if not np.isnan(scorer_auc) else "AUC=N/A"
        print(f"  Round {rnd_int:3d}  n={len(test_df):4d}  "
              f"MAE={mae:.4f}  baseline={baseline_mae:.4f}  "
              f"improv={improvement:+.1f}%  {auc_str}  "
              f"pred={mean_pred:.3f}  actual={mean_actual:.3f}")

    if not results:
        print("No rounds had enough data for backtesting.")
        return

    # Summary
    results_df = pd.DataFrame(results)

    overall_mae = results_df["mae"].mean()
    overall_baseline = results_df["baseline_mae"].mean()
    overall_improv = ((overall_baseline - overall_mae) / overall_baseline * 100) if overall_baseline > 0 else 0.0

    # Ranking summary
    auc_values = results_df["scorer_auc"].dropna()
    p20_values = results_df["precision_at_20"].dropna()
    overall_auc = auc_values.mean() if len(auc_values) > 0 else float("nan")
    overall_p20 = p20_values.mean() if len(p20_values) > 0 else float("nan")

    print(f"\n{'='*70}")
    print(f"BACKTEST SUMMARY — {year}")
    print(f"{'='*70}")
    print(f"  Rounds tested:     {len(results)}")
    print(f"  Overall MAE:       {overall_mae:.4f}")
    print(f"  Baseline MAE:      {overall_baseline:.4f}")
    print(f"  Improvement:       {overall_improv:+.1f}%")
    print(f"  Mean predicted GL: {results_df['mean_predicted'].mean():.3f}")
    print(f"  Mean actual GL:    {results_df['mean_actual'].mean():.3f}")
    if not np.isnan(overall_auc):
        print(f"  Scorer AUC:        {overall_auc:.4f}")
    if not np.isnan(overall_p20):
        print(f"  Precision@20:      {overall_p20:.4f}")

    # Proper calibration diagnostics
    print(f"\n  Calibration diagnostics (bucketed reliability):")
    if all_actual_goals:
        actual = np.concatenate(all_actual_goals)
        threshold_payload = [
            ("1plus_goals", np.concatenate(all_p_1plus), (actual >= 1).astype(int)),
            ("2plus_goals", np.concatenate(all_p_2plus), (actual >= 2).astype(int)),
            ("3plus_goals", np.concatenate(all_p_3plus), (actual >= 3).astype(int)),
        ]
        for name, preds, y in threshold_payload:
            metrics = _compute_threshold_metrics(preds, y)
            if not metrics:
                continue
            print(
                f"    {name:12s}  Brier={metrics['brier_score']:.4f}  "
                f"LogLoss={metrics['log_loss']:.4f}  BaseRate={metrics['base_rate']:.4f}  n={metrics['n']}"
            )
            for b in metrics.get("calibration_curve", []):
                print(
                    f"      [{b['bin_lower']:.2f}, {b['bin_upper']:.2f}]  "
                    f"pred={b['predicted_mean']:.3f}  obs={b['observed_mean']:.3f}  n={b['count']}"
                )

    # Save round-level results
    config.ensure_dirs()
    out_path = config.BACKTEST_DIR / f"backtest_{year}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    # Save player-level predictions
    if player_rows:
        players_df = pd.concat(player_rows, ignore_index=True)
        players_path = config.BACKTEST_DIR / f"backtest_{year}_players.csv"
        players_df.to_csv(players_path, index=False)
        print(f"  Player-level predictions saved to {players_path}")

    # Print learning store summary
    summary = store.get_learning_summary(year)
    print(f"\n  LearningStore: {summary['prediction_rounds']} prediction rounds, "
          f"{summary['outcome_rounds']} outcome rounds, "
          f"{summary['diagnostic_rounds']} diagnostic rounds saved")
    print(f"  LearningStore run_id: {summary.get('run_id')}")

    # Save feature importances from the final model
    if model is not None and model.goals_gbt is not None and hasattr(model.goals_gbt, "feature_importances_"):
        importances = model.goals_gbt.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp_path = config.BACKTEST_DIR / f"backtest_{year}_importances.csv"
        imp_df.to_csv(imp_path, index=False)
        print(f"  Feature importances saved to {imp_path}")

    # Save experiment results as JSON (for A/B comparison)
    experiment_name = getattr(args, "save_experiment", None)
    if experiment_name and all_actual_goals:
        actual = np.concatenate(all_actual_goals)
        threshold_results = {}
        for name, preds, y in [
            ("1plus_goals", np.concatenate(all_p_1plus), (actual >= 1).astype(int)),
            ("2plus_goals", np.concatenate(all_p_2plus), (actual >= 2).astype(int)),
            ("3plus_goals", np.concatenate(all_p_3plus), (actual >= 3).astype(int)),
        ]:
            metrics = _compute_threshold_metrics(preds, y)
            if metrics:
                # Add hit_rate at 50% and 70% confidence thresholds
                p_arr = np.asarray(preds)
                y_arr = np.asarray(y)
                for pct, threshold in [("p50", 0.5), ("p70", 0.7)]:
                    confident = p_arr >= threshold
                    n_conf = int(confident.sum())
                    if n_conf > 0:
                        hit_rate = round(float(y_arr[confident].mean()), 4)
                    else:
                        hit_rate = None
                    metrics[f"hit_rate_{pct}"] = hit_rate
                    metrics[f"n_confident_{pct}"] = n_conf
                threshold_results[name] = metrics

        # Compute ECE from 1plus_goals calibration curve
        cal_ece = None
        if "1plus_goals" in threshold_results:
            cal_curve = threshold_results["1plus_goals"].get("calibration_curve", [])
            if cal_curve:
                total_n = sum(b["count"] for b in cal_curve)
                if total_n > 0:
                    cal_ece = round(sum(
                        abs(b["predicted_mean"] - b["observed_mean"]) * b["count"]
                        for b in cal_curve
                    ) / total_n, 4)

        experiment_data = {
            "label": experiment_name,
            "season": year,
            "run_id": run_id,
            "n_predictions": int(len(actual)),
            "n_rounds": len(results),
            "thresholds": threshold_results,
            "scorer_auc": {
                "overall": round(float(overall_auc), 4) if not np.isnan(overall_auc) else None,
            },
            "mae": {
                "goals": round(float(overall_mae), 4),
                "baseline_goals": round(float(overall_baseline), 4),
                "improvement_pct": round(float(overall_improv), 1),
            },
            "calibration_ece": cal_ece,
        }

        experiments_dir = config.EXPERIMENTS_DIR
        experiments_dir.mkdir(parents=True, exist_ok=True)
        exp_path = experiments_dir / f"{experiment_name}.json"
        with open(exp_path, "w") as f:
            json.dump(experiment_data, f, indent=2)
        print(f"\n  Experiment saved to {exp_path}")


def cmd_diagnose(args):
    """Read player-level backtest CSV and print a full diagnostic report."""
    year = args.year
    if not year:
        print("Error: --diagnose requires --year YYYY")
        return

    players_path = config.BACKTEST_DIR / f"backtest_{year}_players.csv"
    if not players_path.exists():
        print(f"No player-level backtest data for {year}.")
        print(f"Run: python pipeline.py --backtest --year {year}")
        return

    df = pd.read_csv(players_path)
    imp_path = config.BACKTEST_DIR / f"backtest_{year}_importances.csv"
    imp_df = pd.read_csv(imp_path) if imp_path.exists() else None

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT — {year} Backtest")
    print(f"{'='*70}")

    _diagnose_overall(df)
    _diagnose_round_trend(df)
    _diagnose_by_role(df)
    _diagnose_by_actual_goals(df)
    _diagnose_home_away(df)
    _diagnose_over_predictions(df)
    _diagnose_under_predictions(df)
    _diagnose_player_consistency(df)
    _diagnose_feature_importance(imp_df)
    _diagnose_ranking(df)
    _diagnose_refinement_suggestions(df, imp_df)


# ------------------------------------------------------------------
# Diagnose helpers
# ------------------------------------------------------------------

def _diagnose_overall(df):
    """Section 1: Overall Accuracy."""
    print(f"\n--- Section 1: Overall Accuracy ---")
    n = len(df)
    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    baseline_mae = df["baseline_abs_error"].mean()
    improvement = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else 0.0
    beat_pct = df["beat_baseline"].mean() * 100
    mean_pred = df["predicted_goals"].mean()
    mean_actual = df["actual_goals"].mean()
    cal_ratio = mean_pred / mean_actual if mean_actual > 0 else float("inf")

    print(f"  Total predictions:    {n}")
    print(f"  MAE:                  {mae:.4f}")
    print(f"  RMSE:                 {rmse:.4f}")
    print(f"  Baseline MAE:         {baseline_mae:.4f}")
    print(f"  Improvement:          {improvement:+.1f}%")
    print(f"  % beating baseline:   {beat_pct:.1f}%")
    print(f"  Mean predicted:       {mean_pred:.3f}")
    print(f"  Mean actual:          {mean_actual:.3f}")
    print(f"  Calibration ratio:    {cal_ratio:.3f} (1.0 = perfect)")


def _diagnose_round_trend(df):
    """Section 2: Round-by-Round Trend."""
    print(f"\n--- Section 2: Round-by-Round Trend ---")
    rnd_stats = df.groupby("round").agg(
        n=("abs_error", "count"),
        mae=("abs_error", "mean"),
        baseline_mae=("baseline_abs_error", "mean"),
        n_train=("n_train", "first"),
    ).reset_index()

    print(f"  {'Round':>5s}  {'n':>5s}  {'MAE':>7s}  {'Baseline':>8s}  {'n_train':>7s}")
    for _, row in rnd_stats.iterrows():
        print(f"  {int(row['round']):5d}  {int(row['n']):5d}  {row['mae']:.4f}  "
              f"{row['baseline_mae']:.4f}  {int(row['n_train']):7d}")

    # Trend assessment: correlate round number with MAE
    if len(rnd_stats) >= 4:
        rounds_arr = rnd_stats["round"].values.astype(float)
        mae_arr = rnd_stats["mae"].values
        corr = np.corrcoef(rounds_arr, mae_arr)[0, 1]
        if corr < -0.3:
            print(f"  Trend: IMPROVING over the season (r={corr:.2f}) — model learns from in-season data")
        elif corr > 0.3:
            print(f"  Trend: WORSENING over the season (r={corr:.2f}) — possible overfitting or drift")
        else:
            print(f"  Trend: STABLE across rounds (r={corr:.2f})")


def _diagnose_by_role(df):
    """Section 3: Accuracy by Player Role."""
    print(f"\n--- Section 3: Accuracy by Player Role ---")
    roles = df["player_role"].unique()
    if len(roles) == 1 and roles[0] == "general":
        print("  All players classified as 'general' — need more match history for role classification.")
        print(f"  {'Role':>15s}  {'n':>6s}  {'MAE':>7s}  {'BasMAE':>7s}  {'Improv':>7s}")
        n = len(df)
        mae = df["abs_error"].mean()
        bl = df["baseline_abs_error"].mean()
        imp = ((bl - mae) / bl * 100) if bl > 0 else 0.0
        print(f"  {'general':>15s}  {n:6d}  {mae:.4f}  {bl:.4f}  {imp:+.1f}%")
        return

    print(f"  {'Role':>15s}  {'n':>6s}  {'MAE':>7s}  {'BasMAE':>7s}  {'Improv':>7s}")
    for role in sorted(df["player_role"].unique()):
        sub = df[df["player_role"] == role]
        n = len(sub)
        if n < 10:
            continue
        mae = sub["abs_error"].mean()
        bl = sub["baseline_abs_error"].mean()
        imp = ((bl - mae) / bl * 100) if bl > 0 else 0.0
        print(f"  {role:>15s}  {n:6d}  {mae:.4f}  {bl:.4f}  {imp:+.1f}%")


def _diagnose_by_actual_goals(df):
    """Section 4: Accuracy by Actual Goals Scored."""
    print(f"\n--- Section 4: Accuracy by Actual Goals Scored ---")
    buckets = [
        (0, "0 goals", df["actual_goals"] == 0),
        (1, "1 goal", df["actual_goals"] == 1),
        (2, "2 goals", df["actual_goals"] == 2),
        (3, "3+ goals", df["actual_goals"] >= 3),
    ]

    total = len(df)
    print(f"  {'Bucket':>10s}  {'n':>6s}  {'%':>6s}  {'MAE':>7s}  {'AvgPred':>7s}  {'AvgAct':>7s}  {'Bias':>7s}")
    for _, label, mask in buckets:
        sub = df[mask]
        n = len(sub)
        if n < 5:
            continue
        pct = n / total * 100
        mae = sub["abs_error"].mean()
        avg_pred = sub["predicted_goals"].mean()
        avg_act = sub["actual_goals"].mean()
        bias = avg_pred - avg_act
        print(f"  {label:>10s}  {n:6d}  {pct:5.1f}%  {mae:.4f}  {avg_pred:.3f}  {avg_act:.3f}  {bias:+.3f}")

    zero_pct = (df["actual_goals"] == 0).mean() * 100
    print(f"\n  Zero-inflation: {zero_pct:.1f}% of all player-rounds scored 0 goals")


def _diagnose_home_away(df):
    """Section 5: Home vs Away."""
    print(f"\n--- Section 5: Home vs Away ---")
    if "is_home" not in df.columns:
        print("  No is_home column available.")
        return

    for label, val in [("Home", 1), ("Away", 0)]:
        sub = df[df["is_home"] == val]
        if len(sub) < 10:
            continue
        mae = sub["abs_error"].mean()
        bl = sub["baseline_abs_error"].mean()
        avg_pred = sub["predicted_goals"].mean()
        avg_act = sub["actual_goals"].mean()
        print(f"  {label:>6s}  n={len(sub):5d}  MAE={mae:.4f}  baseline={bl:.4f}  "
              f"pred={avg_pred:.3f}  actual={avg_act:.3f}")


def _diagnose_over_predictions(df):
    """Section 6: Top 10 Over-Predictions."""
    print(f"\n--- Section 6: Top 10 Over-Predictions ---")
    top = df.nlargest(10, "error")
    print(f"  {'Player':30s}  {'Rnd':>3s}  {'Pred':>5s}  {'Actual':>6s}  {'Error':>6s}  {'Team':15s}  {'vs':15s}")
    for _, row in top.iterrows():
        print(f"  {row['player']:30s}  {int(row['round']):3d}  {row['predicted_goals']:5.2f}  "
              f"{int(row['actual_goals']):6d}  {row['error']:+5.2f}  {row['team']:15s}  {row['opponent']:15s}")


def _diagnose_under_predictions(df):
    """Section 7: Top 10 Under-Predictions."""
    print(f"\n--- Section 7: Top 10 Under-Predictions ---")
    top = df.nsmallest(10, "error")
    print(f"  {'Player':30s}  {'Rnd':>3s}  {'Pred':>5s}  {'Actual':>6s}  {'Error':>6s}  {'Team':15s}  {'vs':15s}")
    for _, row in top.iterrows():
        print(f"  {row['player']:30s}  {int(row['round']):3d}  {row['predicted_goals']:5.2f}  "
              f"{int(row['actual_goals']):6d}  {row['error']:+5.2f}  {row['team']:15s}  {row['opponent']:15s}")


def _diagnose_player_consistency(df):
    """Section 8: Player Consistency — who does the model nail vs miss."""
    print(f"\n--- Section 8: Player Consistency ---")
    player_stats = df.groupby("player").agg(
        n=("abs_error", "count"),
        mae=("abs_error", "mean"),
        mean_error=("error", "mean"),
        baseline_mae=("baseline_abs_error", "mean"),
    ).reset_index()
    player_stats = player_stats[player_stats["n"] >= 2]

    if player_stats.empty:
        print("  Not enough multi-appearance players.")
        return

    player_stats["improvement"] = (
        (player_stats["baseline_mae"] - player_stats["mae"]) / player_stats["baseline_mae"] * 100
    ).fillna(0)

    # Best predicted players (lowest MAE, min 3 appearances)
    frequent = player_stats[player_stats["n"] >= 3].copy()
    if len(frequent) >= 5:
        print(f"\n  Best predicted (lowest MAE, 3+ appearances):")
        print(f"  {'Player':30s}  {'n':>3s}  {'MAE':>7s}  {'Bias':>7s}")
        for _, row in frequent.nsmallest(5, "mae").iterrows():
            print(f"  {row['player']:30s}  {int(row['n']):3d}  {row['mae']:.4f}  {row['mean_error']:+.3f}")

        print(f"\n  Worst predicted (highest MAE, 3+ appearances):")
        print(f"  {'Player':30s}  {'n':>3s}  {'MAE':>7s}  {'Bias':>7s}")
        for _, row in frequent.nlargest(5, "mae").iterrows():
            print(f"  {row['player']:30s}  {int(row['n']):3d}  {row['mae']:.4f}  {row['mean_error']:+.3f}")

        # Systematic bias: players consistently over- or under-predicted
        biased = frequent[frequent["mean_error"].abs() > 0.3]
        if not biased.empty:
            print(f"\n  Systematic bias (|mean error| > 0.3):")
            print(f"  {'Player':30s}  {'n':>3s}  {'Bias':>7s}  {'Direction':>12s}")
            for _, row in biased.reindex(biased["mean_error"].abs().sort_values(ascending=False).index).head(10).iterrows():
                direction = "over-pred" if row["mean_error"] > 0 else "under-pred"
                print(f"  {row['player']:30s}  {int(row['n']):3d}  {row['mean_error']:+.3f}  {direction:>12s}")


def _diagnose_feature_importance(imp_df):
    """Section 9: Feature Importance."""
    print(f"\n--- Section 9: Feature Importance ---")
    if imp_df is None or imp_df.empty:
        print("  No feature importance data available.")
        return

    # Top 20 features
    top = imp_df.head(20)
    print(f"  Top 20 GBT features:")
    print(f"  {'#':>3s}  {'Feature':45s}  {'Importance':>10s}")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {i:3d}  {row['feature']:45s}  {row['importance']:.4f}")

    # Category-level aggregation
    categories = {
        "career": ["career_", "age_", "is_first_year"],
        "rolling_form": ["player_gl_avg_", "player_bh_avg_", "player_di_avg_",
                         "player_mk_avg_", "player_tk_avg_", "player_if50_avg_",
                         "player_cl_avg_", "player_ho_avg_", "player_ga_avg_",
                         "player_mi_avg_", "player_cm_avg_", "player_cp_avg_",
                         "player_ff_avg_", "player_rb_avg_", "player_one_pct_avg_",
                         "player_accuracy_", "player_gl_streak", "player_gl_trend",
                         "season_goals_total", "days_since_last"],
        "venue": ["venue_", "player_gl_at_venue", "player_bh_at_venue", "player_gl_venue_diff"],
        "opponent": ["opp_", "player_vs_opp"],
        "team": ["team_", "player_goal_share"],
        "scoring_pattern": ["player_q", "player_late_scorer", "player_multi_goal"],
        "role": ["role_", "forward_score"],
        "teammate": ["teammate_"],
        "matchup": ["opp_key_defenders", "opp_defender_strength"],
        "interaction": ["interact_"],
    }

    print(f"\n  Category-level importance:")
    cat_totals = {}
    total_imp = imp_df["importance"].sum()
    for cat_name, prefixes in categories.items():
        cat_imp = imp_df[imp_df["feature"].apply(
            lambda f: any(f.startswith(p) or p in f for p in prefixes)
        )]["importance"].sum()
        cat_totals[cat_name] = cat_imp

    # Add "other" for unmatched
    matched = sum(cat_totals.values())
    cat_totals["other"] = total_imp - matched

    for cat, imp in sorted(cat_totals.items(), key=lambda x: -x[1]):
        pct = imp / total_imp * 100 if total_imp > 0 else 0
        if pct < 0.5:
            continue
        print(f"  {cat:20s}  {pct:5.1f}%")


def _diagnose_ranking(df):
    """Section 10: Ranking Performance — P(scorer) evaluation."""
    print(f"\n--- Section 10: Ranking Performance ---")
    if "p_scorer" not in df.columns:
        print("  No p_scorer column — skipping ranking diagnosis.")
        return

    actual_scored = (df["actual_goals"] >= 1).astype(int)

    # Overall scorer AUC
    try:
        overall_auc = roc_auc_score(actual_scored, df["p_scorer"])
        print(f"  Overall Scorer AUC: {overall_auc:.4f}")
    except ValueError:
        print("  Overall Scorer AUC: N/A (single class)")
        overall_auc = None

    # Per-round AUC trend
    print(f"\n  Per-round AUC:")
    print(f"  {'Round':>5s}  {'AUC':>7s}  {'P@10':>6s}  {'P@20':>6s}")
    for rnd in sorted(df["round"].unique()):
        rnd_df = df[df["round"] == rnd]
        rnd_scored = (rnd_df["actual_goals"] >= 1).astype(int)
        try:
            rnd_auc = roc_auc_score(rnd_scored, rnd_df["p_scorer"])
        except ValueError:
            rnd_auc = float("nan")

        # Precision@10 and @20 per round
        rnd_sorted = rnd_df.sort_values("p_scorer", ascending=False)
        p_at_10 = (rnd_sorted.head(10)["actual_goals"] >= 1).mean() if len(rnd_sorted) >= 10 else float("nan")
        p_at_20 = (rnd_sorted.head(20)["actual_goals"] >= 1).mean() if len(rnd_sorted) >= 20 else float("nan")

        auc_s = f"{rnd_auc:.4f}" if not np.isnan(rnd_auc) else "  N/A "
        p10_s = f"{p_at_10:.3f}" if not np.isnan(p_at_10) else " N/A "
        p20_s = f"{p_at_20:.3f}" if not np.isnan(p_at_20) else " N/A "
        print(f"  {int(rnd):5d}  {auc_s}  {p10_s}  {p20_s}")

    # Top false positives: high P(scorer) but didn't score
    non_scorers = df[df["actual_goals"] == 0].sort_values("p_scorer", ascending=False)
    if len(non_scorers) >= 5:
        print(f"\n  Top 10 False Positives (high P(scorer), scored 0):")
        print(f"  {'Player':30s}  {'Rnd':>3s}  {'P(scorer)':>9s}  {'Team':15s}  {'vs':15s}")
        for _, row in non_scorers.head(10).iterrows():
            print(f"  {row['player']:30s}  {int(row['round']):3d}  {row['p_scorer']:9.3f}  "
                  f"{row['team']:15s}  {row['opponent']:15s}")

    # Top misses: scored but low P(scorer)
    scorers = df[df["actual_goals"] >= 1].sort_values("p_scorer", ascending=True)
    if len(scorers) >= 5:
        print(f"\n  Top 10 Misses (scored but low P(scorer)):")
        print(f"  {'Player':30s}  {'Rnd':>3s}  {'P(scorer)':>9s}  {'Actual GL':>9s}  {'Team':15s}")
        for _, row in scorers.head(10).iterrows():
            print(f"  {row['player']:30s}  {int(row['round']):3d}  {row['p_scorer']:9.3f}  "
                  f"{int(row['actual_goals']):9d}  {row['team']:15s}")


def _diagnose_refinement_suggestions(df, imp_df):
    """Section 10: Auto-generated refinement suggestions."""
    print(f"\n--- Section 10: Refinement Suggestions ---")
    suggestions = []

    # 1. Calibration bias
    mean_pred = df["predicted_goals"].mean()
    mean_actual = df["actual_goals"].mean()
    cal_ratio = mean_pred / mean_actual if mean_actual > 0 else 1.0
    if cal_ratio < 0.85:
        suggestions.append(
            f"CALIBRATION: Model under-predicts (ratio={cal_ratio:.3f}). "
            f"Consider reducing Poisson alpha or increasing GBT weight."
        )
    elif cal_ratio > 1.15:
        suggestions.append(
            f"CALIBRATION: Model over-predicts (ratio={cal_ratio:.3f}). "
            f"Consider increasing Poisson alpha or reducing GBT weight."
        )

    # 2. Zero-inflation handling
    zero_mask = df["actual_goals"] == 0
    if zero_mask.any():
        zero_pred_mean = df.loc[zero_mask, "predicted_goals"].mean()
        if zero_pred_mean > 0.4:
            suggestions.append(
                f"ZERO-INFLATION: Mean prediction for 0-goal players is {zero_pred_mean:.3f} (>0.4). "
                f"Consider a two-stage model (classify scorer/non-scorer first, then predict amount)."
            )

    # 3. High-scorer gap
    high_mask = df["actual_goals"] >= 3
    if high_mask.sum() >= 5:
        high_pred_mean = df.loc[high_mask, "predicted_goals"].mean()
        if high_pred_mean < 2.0:
            suggestions.append(
                f"HIGH-SCORER GAP: Mean prediction for 3+ goal games is {high_pred_mean:.3f} (<2.0). "
                f"Model misses big games — boost key_forward features or add forward-specific interactions."
            )

    # 4. Home/away split
    if "is_home" in df.columns:
        home_mae = df.loc[df["is_home"] == 1, "abs_error"].mean() if (df["is_home"] == 1).any() else 0
        away_mae = df.loc[df["is_home"] == 0, "abs_error"].mean() if (df["is_home"] == 0).any() else 0
        if abs(home_mae - away_mae) > 0.1:
            worse = "away" if away_mae > home_mae else "home"
            suggestions.append(
                f"HOME/AWAY: MAE gap = {abs(home_mae - away_mae):.3f} ({worse} is worse). "
                f"Consider adding home-ground advantage or travel-distance features."
            )

    # 5. No learning effect (MAE increases across rounds)
    rnd_stats = df.groupby("round")["abs_error"].mean()
    if len(rnd_stats) >= 4:
        rounds_arr = rnd_stats.index.values.astype(float)
        mae_arr = rnd_stats.values
        corr = np.corrcoef(rounds_arr, mae_arr)[0, 1]
        if corr > 0.3:
            suggestions.append(
                f"NO LEARNING: MAE increases across rounds (r={corr:.2f}). "
                f"Model may be overfitting or drifting. Try tuning rolling window sizes or adding regularization."
            )

    # 6. Feature concentration
    if imp_df is not None and len(imp_df) >= 5:
        total_imp = imp_df["importance"].sum()
        top5_imp = imp_df.head(5)["importance"].sum()
        top5_pct = top5_imp / total_imp * 100 if total_imp > 0 else 0
        if top5_pct > 50:
            top5_names = ", ".join(imp_df.head(5)["feature"].tolist())
            suggestions.append(
                f"FEATURE CONCENTRATION: Top 5 features account for {top5_pct:.0f}% of importance ({top5_names}). "
                f"Consider more regularization (increase min_samples_leaf) or feature selection."
            )

    # 7. Sparse data
    n = len(df)
    if n < 200:
        suggestions.append(
            f"SPARSE DATA: Only {n} predictions. Consider scraping more seasons "
            f"or including more players to improve model reliability."
        )

    if not suggestions:
        print("  No major issues detected. Model looks well-calibrated!")
    else:
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")

    print()


# ------------------------------------------------------------------
# Sequential Learning
# ------------------------------------------------------------------

def _merge_predictions(scoring_preds, disposal_preds):
    """Join scoring and disposal predictions on (player, team, match_id)."""
    if disposal_preds.empty:
        return scoring_preds

    join_cols = ["player", "team", "match_id"]
    # Only keep disposal-specific columns (avoid duplicating shared cols)
    disp_cols = [c for c in disposal_preds.columns
                 if c not in scoring_preds.columns or c in join_cols]
    merged = scoring_preds.merge(
        disposal_preds[disp_cols], on=join_cols, how="left"
    )
    return merged


def _apply_isotonic_calibration_to_predictions(preds: pd.DataFrame, calibrator) -> pd.DataFrame:
    """Apply isotonic calibrators to threshold probability columns.

    Expects a CalibratedPredictor-compatible object with .transform(target, preds).
    This is used in sequential mode so we can:
      1) fit isotonic on raw model outputs, and
      2) save/evaluate calibrated probabilities without "calibrating calibrated" values.
    """
    if preds is None or preds.empty or calibrator is None:
        return preds

    out = preds.copy()
    skip_targets = getattr(config, "ISOTONIC_SKIP_TARGETS", set())

    def _cal(col, tgt):
        if col not in out.columns:
            return
        try:
            out[col] = np.round(np.clip(calibrator.transform(tgt, out[col].values.astype(float)), 0, 1), 4)
        except Exception:
            pass

    # Goals
    for tgt in ["1plus_goals", "2plus_goals", "3plus_goals"]:
        if tgt not in skip_targets:
            _cal(f"p_{tgt}", tgt)

    # Disposals
    for t in getattr(config, "DISPOSAL_THRESHOLDS", []):
        tgt = f"{t}plus_disp"
        if tgt not in skip_targets:
            _cal(f"p_{t}plus_disp", tgt)

    # Marks
    for t in getattr(config, "MARKS_THRESHOLDS", []):
        tgt = f"{t}plus_mk"
        if tgt not in skip_targets:
            _cal(f"p_{t}plus_mk", tgt)

    # Game winner (optional)
    _cal("home_win_prob", "game_winner")

    # Independent calibrations can break cross-threshold ordering; enforce monotonicity.
    def _mono(cols):
        present = [c for c in cols if c in out.columns]
        if len(present) < 2:
            return
        arr = out[present].to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.minimum.accumulate(arr, axis=1)
        out[present] = np.round(arr, 4)

    _mono(["p_1plus_goals", "p_2plus_goals", "p_3plus_goals"])
    _mono([f"p_{t}plus_disp" for t in getattr(config, "DISPOSAL_THRESHOLDS", [])])
    _mono([f"p_{t}plus_mk" for t in getattr(config, "MARKS_THRESHOLDS", [])])

    # Keep p_scorer aligned to the (possibly calibrated + monotonicity-enforced) 1+ probability.
    if "p_1plus_goals" in out.columns:
        out["p_scorer"] = out["p_1plus_goals"].values

    return out


def _build_outcomes(test_df, tm_round=None):
    """Extract actual goals, behinds, disposals from test feature DataFrame."""
    result = pd.DataFrame({
        "player": test_df["player"].values,
        "team": test_df["team"].values,
        "match_id": test_df["match_id"].values,
        "actual_goals": test_df["GL"].values,
        "actual_behinds": test_df["BH"].values,
    })
    if "DI" in test_df.columns:
        result["actual_disposals"] = test_df["DI"].values
    if "MK" in test_df.columns:
        result["actual_marks"] = test_df["MK"].values
    return result


def _build_diagnostics(predictions, outcomes, test_df=None):
    """Per-player error analysis from merged predictions and outcomes."""
    join_cols = ["player", "team", "match_id"]
    merged = predictions.merge(outcomes, on=join_cols, how="inner")

    if merged.empty:
        return pd.DataFrame()

    diag = pd.DataFrame({
        "player": merged["player"].values,
        "team": merged["team"].values,
        "match_id": merged["match_id"].values,
    })

    if "predicted_goals" in merged.columns and "actual_goals" in merged.columns:
        error = merged["predicted_goals"].values - merged["actual_goals"].values
        diag["goal_error"] = np.round(error, 4)
        diag["goal_abs_error"] = np.round(np.abs(error), 4)

    if "predicted_behinds" in merged.columns and "actual_behinds" in merged.columns:
        bh_error = merged["predicted_behinds"].values - merged["actual_behinds"].values
        diag["behind_error"] = np.round(bh_error, 4)

    if "predicted_disposals" in merged.columns and "actual_disposals" in merged.columns:
        di_error = merged["predicted_disposals"].values - merged["actual_disposals"].values
        diag["disposal_error"] = np.round(di_error, 4)

    if "predicted_marks" in merged.columns and "actual_marks" in merged.columns:
        mk_error = merged["predicted_marks"].values - merged["actual_marks"].values
        diag["marks_error"] = np.round(mk_error, 4)

    # Miss classification
    if test_df is not None and "predicted_goals" in merged.columns:
        from analysis import classify_prediction_misses
        miss_df = classify_prediction_misses(merged, test_df)
        if not miss_df.empty:
            miss_lookup = {(str(r["player"]), str(r["team"])): r["miss_type"]
                           for _, r in miss_df.iterrows()}
            diag["miss_type"] = [
                miss_lookup.get((str(p), str(t)), "")
                for p, t in zip(diag["player"], diag["team"])
            ]
        else:
            diag["miss_type"] = ""
    else:
        diag["miss_type"] = ""

    return diag


def _update_sequential_calibration(store, preds, test_df):
    """Bucket predictions and update calibration state for goals and disposals."""
    actual_goals = test_df["GL"].values
    buckets = np.arange(0.05, 1.0, 0.1).round(2)
    rows = []

    # 1+ goals calibration (from p_scorer)
    scorer_prob = None
    if "p_1plus_goals_raw" in preds.columns:
        scorer_prob = preds["p_1plus_goals_raw"].values.astype(float)
    elif "p_scorer_raw" in preds.columns:
        scorer_prob = preds["p_scorer_raw"].values.astype(float)
    elif "p_scorer" in preds.columns:
        scorer_prob = preds["p_scorer"].values.astype(float)

    if scorer_prob is not None:
        for bucket in buckets:
            lo, hi = bucket - 0.05, bucket + 0.05
            mask = (scorer_prob >= lo) & (scorer_prob < hi)
            if mask.sum() > 0:
                rows.append({
                    "target": "1plus_goals",
                    "probability_bucket": float(bucket),
                    "predicted": int(mask.sum()),
                    "occurred": int((actual_goals[mask] >= 1).sum()),
                })

    # 2+ and 3+ goals calibration from predicted threshold probabilities
    for threshold, target_name in [(2, "2plus_goals"), (3, "3plus_goals")]:
        p_col = f"p_{threshold}plus_goals_raw"
        if p_col in preds.columns:
            p_exceed = preds[p_col].values.astype(float)
        else:
            p_col = f"p_{threshold}plus_goals"
            if p_col in preds.columns:
                p_exceed = preds[p_col].values.astype(float)
            elif "p_goals_0" in preds.columns and "p_goals_1" in preds.columns:
                # Fallback: derive from PMF columns.
                p0 = preds["p_goals_0"].values.astype(float)
                p1 = preds["p_goals_1"].values.astype(float)
                if threshold == 2:
                    p_exceed = 1.0 - p0 - p1
                else:
                    p2 = preds["p_goals_2"].values.astype(float) if "p_goals_2" in preds.columns else 0.0
                    p_exceed = 1.0 - p0 - p1 - p2
                p_exceed = np.clip(p_exceed, 0.0, 1.0)
            else:
                continue

            for bucket in buckets:
                lo, hi = bucket - 0.05, bucket + 0.05
                mask = (p_exceed >= lo) & (p_exceed < hi)
                if mask.sum() > 0:
                    rows.append({
                        "target": target_name,
                        "probability_bucket": float(bucket),
                        "predicted": int(mask.sum()),
                        "occurred": int((actual_goals[mask] >= threshold).sum()),
                    })

    # Disposal calibration — use predicted probabilities from the DataFrame
    # (distribution-agnostic: works for Poisson, Gaussian, NegBin)
    if "DI" in test_df.columns:
        actual_disp = test_df["DI"].values
        for threshold in config.DISPOSAL_THRESHOLDS:
            target_name = f"{threshold}plus_disp"
            p_col = f"p_{threshold}plus_disp"
            if p_col not in preds.columns:
                continue
            p_exceed = preds[p_col].values.astype(float)
            for bucket in buckets:
                lo, hi = bucket - 0.05, bucket + 0.05
                mask = (p_exceed >= lo) & (p_exceed < hi)
                if mask.sum() > 0:
                    rows.append({
                        "target": target_name,
                        "probability_bucket": float(bucket),
                        "predicted": int(mask.sum()),
                        "occurred": int((actual_disp[mask] >= threshold).sum()),
                    })

    if rows:
        cal_df = pd.DataFrame(rows)
        store.update_calibration(cal_df)


def _predict_games_for_round(winner_model, tm_train, tm_round,
                             player_predictions_df=None, store=None):
    """Build game features and predict for a specific round's matches."""
    # Combine training data with round data for feature computation
    combined = pd.concat([tm_train, tm_round], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)

    try:
        result = winner_model.predict_with_margin(
            combined, player_predictions_df=player_predictions_df,
            store=store,
        )
        # Filter to only the round's matches
        round_match_ids = tm_round["match_id"].unique()
        result = result[result["match_id"].isin(round_match_ids)]
        return result
    except Exception as e:
        print(f"  Warning: Game winner prediction build failed: {e}")
        return pd.DataFrame()


def cmd_sequential(args, disposal_distribution=None):
    """Sequential learning: process a season round-by-round, learning from
    each round's outcomes before predicting the next."""
    import json
    import time

    year = args.year or config.SEQUENTIAL_YEAR
    run_id = getattr(args, "run_id", None) or _new_run_id(prefix=f"sequential_{year}")

    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    if not feat_path.exists():
        print("No feature matrix found. Run --features first.")
        return

    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    if not feat_cols_path.exists():
        print("No feature_columns.json found. Run --features first.")
        return

    feature_df = pd.read_parquet(feat_path)
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    # Augment with DL features if available
    feature_df, feature_cols = _augment_with_dl_features(feature_df, feature_cols)

    # Load team-match data for game winner model
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if tm_path.exists():
        team_match_df = pd.read_parquet(tm_path)
        team_match_df["date"] = pd.to_datetime(team_match_df["date"])
        has_team_data = True
    else:
        print("  Warning: No team_matches.parquet — game predictions disabled")
        team_match_df = pd.DataFrame()
        has_team_data = False

    # Sequential store — output goes to data/sequential/
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=run_id)

    # Get rounds in the target season
    season_df = feature_df[feature_df["year"] == year].copy()
    if season_df.empty:
        print(f"No data for season {year}.")
        return

    rounds = sorted(season_df["round_number"].dropna().unique())
    print(f"\nSequential learning for {year}")
    print(f"  Rounds: {len(rounds)}")
    print(f"  Players per round: ~{len(season_df) // max(len(rounds), 1)}")
    print(f"  Run ID: {run_id}")
    if getattr(args, "reset_calibration", False):
        reset_done = store.reset_calibration(run_id=run_id, latest=False)
        msg = "cleared" if reset_done else "already empty"
        print(f"  Calibration state reset: {msg}")
    else:
        seeded = store.seed_calibration_from_latest()
        if seeded:
            print("  Calibration state seeded from latest prior run")
        else:
            print("  Calibration state starts fresh for this run")
    print()

    total_start = time.time()
    round_results = []

    from analysis import generate_round_analysis
    from model import CalibratedPredictor

    # Isotonic calibration state
    use_isotonic = getattr(config, "CALIBRATION_METHOD", "bucket") == "isotonic"
    isotonic_min = getattr(config, "ISOTONIC_MIN_SAMPLES", 100)
    isotonic_interval = getattr(config, "ISOTONIC_REFIT_INTERVAL", 5)
    isotonic_calibrator = store.load_isotonic_calibrator() or CalibratedPredictor()
    # Load accumulated isotonic data from prior year runs (cross-year calibration)
    _iso_accum = store.load_isotonic_accum()
    if _iso_accum:
        n_prior = sum(len(v["preds"]) for v in _iso_accum.values())
        print(f"  Isotonic accumulation loaded: {n_prior:,} samples across {len(_iso_accum)} targets")
        # Immediately refit calibrator on loaded data so round 1 benefits
        for tgt, data in _iso_accum.items():
            if len(data["preds"]) >= isotonic_min:
                isotonic_calibrator.fit(
                    tgt,
                    np.array(data["preds"]),
                    np.array(data["actuals"]),
                )
        store.save_isotonic_calibrator(isotonic_calibrator)
    else:
        print("  Isotonic accumulation starts fresh")

    for rnd in rounds:
        rnd_int = int(rnd)
        rnd_start = time.time()

        # 1. SPLIT
        train_mask = (
            (feature_df["year"] < year)
            | ((feature_df["year"] == year) & (feature_df["round_number"] < rnd))
        )
        train_df = feature_df[train_mask].copy()
        test_mask = (feature_df["year"] == year) & (feature_df["round_number"] == rnd)
        test_df = feature_df[test_mask].copy()

        if len(train_df) < 50 or test_df.empty:
            continue

        # Recompute sample weights dynamically for this round
        from features import add_dynamic_sample_weights
        train_df = add_dynamic_sample_weights(train_df, year, rnd)

        # 2. TRAIN
        scoring_model = AFLScoringModel()
        scoring_model.train_backtest(train_df, feature_cols)

        disposal_model = AFLDisposalModel(
            distribution=disposal_distribution or config.DISPOSAL_DISTRIBUTION
        )
        disposal_model.train_backtest(train_df, feature_cols)

        marks_model = AFLMarksModel(
            distribution=config.MARKS_DISTRIBUTION
        )
        marks_model.train_backtest(train_df, feature_cols)

        # Build player-level predictions on training data for game winner features
        player_preds_for_gw = _build_player_predictions_for_winner_features(
            train_df,
            feature_cols,
            scoring_model=scoring_model,
            disposal_model=disposal_model,
            marks_model=marks_model,
        )

        # Game winner model
        game_preds = pd.DataFrame()
        if has_team_data:
            winner_model = AFLGameWinnerModel()
            tm_train = team_match_df[
                (team_match_df["year"] < year)
                | ((team_match_df["year"] == year) & (team_match_df["round_number"] < rnd))
            ].copy()
            tm_round = team_match_df[
                (team_match_df["year"] == year) & (team_match_df["round_number"] == rnd)
            ].copy()

            if len(tm_train) >= 20 and not tm_round.empty:
                try:
                    winner_model.train_backtest(
                        tm_train,
                        player_predictions_df=player_preds_for_gw if not player_preds_for_gw.empty else None,
                    )
                    # Also generate player preds for the test round for prediction
                    _test_pp = _build_player_predictions_for_winner_features(
                        test_df,
                        feature_cols,
                        scoring_model=scoring_model,
                        disposal_model=disposal_model,
                        marks_model=marks_model,
                    )
                    # Combine train + test player preds for predict_with_margin
                    all_pp = pd.concat([player_preds_for_gw, _test_pp], ignore_index=True) \
                        if not _test_pp.empty else player_preds_for_gw
                    game_preds = _predict_games_for_round(
                        winner_model, tm_train, tm_round,
                        player_predictions_df=all_pp if not all_pp.empty else None,
                        store=store,
                    )
                except Exception as e:
                    print(f"  Warning: Game winner model failed for R{rnd_int}: {e}")

        # 3. PREDICT
        # In isotonic mode we predict *raw* probabilities and apply calibration externally,
        # so isotonic fitting always sees uncalibrated model outputs.
        pred_store = None if use_isotonic else store
        scoring_raw = scoring_model.predict_distributions(test_df, store=pred_store, feature_cols=feature_cols)
        disposal_raw = disposal_model.predict_distributions(test_df, store=pred_store, feature_cols=feature_cols)
        marks_raw = marks_model.predict_distributions(test_df, store=pred_store, feature_cols=feature_cols)

        merged_raw = _merge_predictions(scoring_raw, disposal_raw)
        merged_raw = _merge_predictions(merged_raw, marks_raw)

        # 4. CALIBRATE + SAVE (save calibrated, but learn from raw)
        merged_preds = merged_raw
        if use_isotonic:
            merged_preds = _apply_isotonic_calibration_to_predictions(merged_raw, isotonic_calibrator)

        outcomes = _build_outcomes(test_df)
        diagnostics = _build_diagnostics(merged_preds, outcomes, test_df=test_df)

        store.save_predictions(year, rnd_int, merged_preds)
        store.save_outcomes(year, rnd_int, outcomes)
        store.save_diagnostics(year, rnd_int, diagnostics)

        if not game_preds.empty:
            store.save_game_predictions(year, rnd_int, game_preds)

        # 5. LEARN
        _update_sequential_calibration(store, merged_raw, test_df)
        store.compute_calibration_adjustments()

        # 5b. Isotonic calibration: accumulate data and refit periodically
        if use_isotonic:
            iso_skip = getattr(config, "ISOTONIC_SKIP_TARGETS", set())

            # Key-based alignment: merge predictions with actuals to avoid
            # row-order misalignment between merged_raw and test_df.
            _actual_cols = {}
            if "GL" in test_df.columns:
                _actual_cols["actual_goals"] = test_df["GL"]
            if "DI" in test_df.columns:
                _actual_cols["actual_disp"] = test_df["DI"]
            if "MK" in test_df.columns:
                _actual_cols["actual_marks"] = test_df["MK"]

            _iso_join = test_df[["player", "team", "match_id"]].copy()
            for col, vals in _actual_cols.items():
                _iso_join[col] = vals.values
            _iso_merged = merged_raw.merge(
                _iso_join, on=["player", "team", "match_id"], how="inner"
            )

            actual_goals = _iso_merged["actual_goals"].values if "actual_goals" in _iso_merged.columns else None
            actual_disp = _iso_merged["actual_disp"].values if "actual_disp" in _iso_merged.columns else None
            actual_marks = _iso_merged["actual_marks"].values if "actual_marks" in _iso_merged.columns else None

            # Accumulate goal thresholds (skip if in ISOTONIC_SKIP_TARGETS)
            if actual_goals is not None and "1plus_goals" not in iso_skip:
                p1_col = None
                if "p_1plus_goals_raw" in _iso_merged.columns:
                    p1_col = "p_1plus_goals_raw"
                elif "p_scorer_raw" in _iso_merged.columns:
                    p1_col = "p_scorer_raw"
                elif "p_scorer" in _iso_merged.columns:
                    p1_col = "p_scorer"

                if p1_col is not None:
                    _iso_accum.setdefault("1plus_goals", {"preds": [], "actuals": []})
                    _iso_accum["1plus_goals"]["preds"].extend(_iso_merged[p1_col].values.astype(float).tolist())
                    _iso_accum["1plus_goals"]["actuals"].extend((actual_goals >= 1).astype(int).tolist())

            # 2+/3+ goals: fit on raw predicted threshold probabilities when available.
            if actual_goals is not None:
                for threshold, name in [(2, "2plus_goals"), (3, "3plus_goals")]:
                    if name in iso_skip:
                        continue
                    raw_col = f"p_{threshold}plus_goals_raw"
                    if raw_col in _iso_merged.columns:
                        p_exceed = _iso_merged[raw_col].values.astype(float)
                    else:
                        p_col = f"p_{threshold}plus_goals"
                        if p_col in _iso_merged.columns:
                            p_exceed = _iso_merged[p_col].values.astype(float)
                        else:
                            continue
                    _iso_accum.setdefault(name, {"preds": [], "actuals": []})
                    _iso_accum[name]["preds"].extend(p_exceed.tolist())
                    _iso_accum[name]["actuals"].extend((actual_goals >= threshold).astype(int).tolist())

            # Accumulate disposal thresholds
            if actual_disp is not None:
                for t in config.DISPOSAL_THRESHOLDS:
                    p_col = f"p_{t}plus_disp"
                    if p_col in _iso_merged.columns:
                        name = f"{t}plus_disp"
                        _iso_accum.setdefault(name, {"preds": [], "actuals": []})
                        _iso_accum[name]["preds"].extend(_iso_merged[p_col].values.astype(float).tolist())
                        _iso_accum[name]["actuals"].extend((actual_disp >= t).astype(int).tolist())

            # Accumulate marks thresholds
            if actual_marks is not None:
                for t in config.MARKS_THRESHOLDS:
                    p_col = f"p_{t}plus_mk"
                    if p_col in _iso_merged.columns:
                        name = f"{t}plus_mk"
                        _iso_accum.setdefault(name, {"preds": [], "actuals": []})
                        _iso_accum[name]["preds"].extend(_iso_merged[p_col].values.astype(float).tolist())
                        _iso_accum[name]["actuals"].extend((actual_marks >= t).astype(int).tolist())

            # Accumulate game winner calibration data (skip if in ISOTONIC_SKIP_TARGETS)
            if not game_preds.empty and "home_win_prob" in game_preds.columns and has_team_data and "game_winner" not in iso_skip:
                home_round = tm_round[tm_round["is_home"]].copy()
                if not home_round.empty:
                    home_actuals_map = home_round.set_index("match_id")["margin"].to_dict()
                    for _, gp_row in game_preds.iterrows():
                        mid = gp_row["match_id"]
                        if mid in home_actuals_map:
                            actual_win = 1 if home_actuals_map[mid] > 0 else 0
                            _iso_accum.setdefault("game_winner", {"preds": [], "actuals": []})
                            _iso_accum["game_winner"]["preds"].append(float(gp_row["home_win_prob"]))
                            _iso_accum["game_winner"]["actuals"].append(actual_win)

            # Refit isotonic calibrators periodically
            rounds_done = len(round_results) + 1
            if rounds_done % isotonic_interval == 0:
                for tgt, data in _iso_accum.items():
                    if len(data["preds"]) >= isotonic_min:
                        isotonic_calibrator.fit(
                            tgt,
                            np.array(data["preds"]),
                            np.array(data["actuals"]),
                        )
                store.save_isotonic_calibrator(isotonic_calibrator)
                store.save_isotonic_accum(_iso_accum)

        # 6. ANALYZE
        game_actuals = tm_round if has_team_data else pd.DataFrame()
        try:
            analysis = generate_round_analysis(
                year, rnd_int, merged_preds, outcomes,
                game_preds, game_actuals, test_df, store
            )
            store.save_analysis(year, rnd_int, analysis)
        except Exception as e:
            print(f"  Warning: Analysis failed for R{rnd_int}: {e}")
            analysis = {}

        # 7. PRINT
        elapsed = time.time() - rnd_start
        summary = analysis.get("summary", {})
        mae = summary.get("goals_mae", float("nan"))
        auc = summary.get("scorer_auc", float("nan"))
        tm = summary.get("threshold_metrics", {})
        br1 = tm.get("1plus_goals", {}).get("brier_score")
        br2 = tm.get("2plus_goals", {}).get("brier_score")

        mae_str = f"MAE={mae:.3f}" if mae is not None and not (isinstance(mae, float) and np.isnan(mae)) else "MAE=N/A"
        br1_str = f"Br1+={br1:.3f}" if br1 is not None else "Br1+=N/A"
        br2_str = f"Br2+={br2:.3f}" if br2 is not None else "Br2+=N/A"
        auc_str = f"AUC={auc:.3f}" if auc is not None and not (isinstance(auc, float) and np.isnan(auc)) else "AUC=N/A"

        n_players = len(test_df)
        print(f"  R{rnd_int:02d}  n={n_players:<4d} {mae_str}  {br1_str}  {br2_str}  {auc_str}  ({elapsed:.1f}s)")

        round_results.append({
            "round": rnd_int,
            "n_players": n_players,
            "mae": mae,
            "auc": auc,
            "brier_1plus": br1,
            "brier_2plus": br2,
            "elapsed": round(elapsed, 1),
        })

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL LEARNING SUMMARY — {year}")
    print(f"{'='*70}")

    if round_results:
        maes = [r["mae"] for r in round_results if r["mae"] is not None]
        if maes:
            overall_mae = np.mean(maes)
            n_rounds = len(maes)
            half = n_rounds // 2

            first_half_mae = np.mean(maes[:half]) if half > 0 else float("nan")
            second_half_mae = np.mean(maes[half:]) if half > 0 else float("nan")

            print(f"  Rounds processed:  {n_rounds}")
            print(f"  Overall MAE:       {overall_mae:.4f}")
            print(f"  First-half MAE:    {first_half_mae:.4f}")
            print(f"  Second-half MAE:   {second_half_mae:.4f}")

            if first_half_mae > 0:
                learning_effect = (first_half_mae - second_half_mae) / first_half_mae * 100
                print(f"  Learning effect:   {learning_effect:+.1f}%")

        # Calibration summary
        cal = store.get_calibration_state(year=year)
        active = cal[cal["calibration_adj"] != 0]
        print(f"\n  Active calibration buckets: {len(active)}")
        if len(active) > 0:
            print(f"  Mean adjustment: {active['calibration_adj'].mean():.4f}")

        aucs = [r["auc"] for r in round_results if r["auc"] is not None]
        if aucs:
            print(f"  Mean Scorer AUC:   {np.mean(aucs):.4f}")

    # Save final isotonic accumulation for next year's run
    if use_isotonic and _iso_accum:
        store.save_isotonic_accum(_iso_accum)
        store.save_isotonic_calibrator(isotonic_calibrator)
        n_total = sum(len(v["preds"]) for v in _iso_accum.values())
        print(f"  Isotonic accumulation saved: {n_total:,} samples across {len(_iso_accum)} targets")

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(f"  Output: {config.SEQUENTIAL_DIR}")


def cmd_train_disposals(args, feature_df=None):
    """Train the disposal prediction model."""
    if feature_df is None:
        feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
        if not feat_path.exists():
            print("No feature matrix found. Run --features first.")
            return None
        feature_df = pd.read_parquet(feat_path)

    # Augment with DL features if available
    import json
    fc_path = config.FEATURES_DIR / "feature_columns.json"
    if fc_path.exists():
        with open(fc_path) as f:
            fc = json.load(f)
        feature_df, _ = _augment_with_dl_features(feature_df, fc)

    print(f"Training disposal model ({config.DISPOSAL_DISTRIBUTION} distribution)...")
    model = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION)
    metrics = model.train(feature_df)
    model.save()
    print(f"\nDisposal training complete. MAE: {metrics['disp_mae']:.4f}")
    return model


def cmd_train_marks(args, feature_df=None):
    """Train the marks prediction model."""
    if feature_df is None:
        feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
        if not feat_path.exists():
            print("No feature matrix found. Run --features first.")
            return None
        feature_df = pd.read_parquet(feat_path)

    # Augment with DL features if available
    import json
    fc_path = config.FEATURES_DIR / "feature_columns.json"
    if fc_path.exists():
        with open(fc_path) as f:
            fc = json.load(f)
        feature_df, _ = _augment_with_dl_features(feature_df, fc)

    print(f"Training marks model ({config.MARKS_DISTRIBUTION} distribution)...")
    model = AFLMarksModel(distribution=config.MARKS_DISTRIBUTION)
    metrics = model.train(feature_df)
    model.save()
    print(f"\nMarks training complete. MAE: {metrics['marks_mae']:.4f}")
    return model


def cmd_train_embeddings(args, feature_df=None):
    """Train entity embedding model and augment feature matrix."""
    if feature_df is None:
        feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
        if not feat_path.exists():
            print("No feature matrix found. Run --features first.")
            return
        feature_df = pd.read_parquet(feat_path)

    from embeddings import train_embedding_model, extract_embeddings
    from embeddings import save_embeddings, augment_features as emb_augment

    import json
    fc_path = config.FEATURES_DIR / "feature_columns.json"
    if not fc_path.exists():
        print("No feature_columns.json found. Run --features first.")
        return
    with open(fc_path) as f:
        feature_cols = json.load(f)

    print("Training entity embedding model...")
    model, vocabs = train_embedding_model(feature_df, feature_cols)
    emb_dfs = extract_embeddings(model, vocabs)
    save_embeddings(model, vocabs, emb_dfs)

    # Test augmentation
    df_aug = emb_augment(feature_df)
    emb_cols = [c for c in df_aug.columns if c.startswith("emb_")]
    print(f"Entity embeddings trained — {len(emb_cols)} embedding features available")


def cmd_train_sequence(args, feature_df=None):
    """Train GRU sequence model and extract form embeddings."""
    if feature_df is None:
        feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
        if not feat_path.exists():
            print("No feature matrix found. Run --features first.")
            return
        feature_df = pd.read_parquet(feat_path)

    from sequence_model import (build_sequences, train_form_model,
                                extract_form_embeddings, save_form_model)

    print("Building player sequences...")
    seq_data = build_sequences(feature_df)

    print("Training GRU form model...")
    model = train_form_model(seq_data)

    print("Extracting form embeddings...")
    form_df = extract_form_embeddings(model, seq_data)
    save_form_model(model, seq_data, form_df)

    emb_cols = [c for c in form_df.columns if c.startswith("form_emb_")]
    print(f"GRU form model trained — {len(emb_cols)} form features available")


def cmd_train_winner(args):
    """Train the game winner prediction model."""
    team_match_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if not team_match_path.exists():
        print("No team_matches.parquet found. Run --clean first.")
        return None

    team_match_df = pd.read_parquet(team_match_path)
    print(f"Training game winner model on {len(team_match_df)} team-match rows...")

    player_predictions_df = pd.DataFrame()
    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    if feat_path.exists() and feat_cols_path.exists():
        import json

        feature_df = pd.read_parquet(feat_path)
        with open(feat_cols_path) as f:
            feature_cols = json.load(f)
        feature_df, feature_cols = _augment_with_dl_features(feature_df, feature_cols)

        scoring_model = None
        disposal_model = None
        marks_model = None

        try:
            scoring_model = AFLScoringModel()
            scoring_model.load()
        except FileNotFoundError:
            pass

        try:
            disposal_model = AFLDisposalModel(distribution=config.DISPOSAL_DISTRIBUTION)
            disposal_model.load()
        except FileNotFoundError:
            pass

        try:
            marks_model = AFLMarksModel(distribution=config.MARKS_DISTRIBUTION)
            marks_model.load()
        except FileNotFoundError:
            pass

        player_predictions_df = _build_player_predictions_for_winner_features(
            feature_df,
            feature_cols,
            scoring_model=scoring_model,
            disposal_model=disposal_model,
            marks_model=marks_model,
        )
        if not player_predictions_df.empty:
            print(
                f"  Using aggregated player predictions for winner training "
                f"({len(player_predictions_df)} player rows)"
            )
        else:
            print("  Warning: winner training falling back to team-only features")

    model = AFLGameWinnerModel()
    metrics = model.train(
        team_match_df,
        player_predictions_df=player_predictions_df if not player_predictions_df.empty else None,
    )
    model.save()
    return model


def cmd_backtest_winner(args):
    """Walk-forward backtest for the game winner model.

    For each round in --year, train on all prior data, predict that round,
    and accumulate hybrid/residual/market metrics.
    """
    year = args.year
    if not year:
        print("Error: --backtest-winner requires --year YYYY")
        return

    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if not tm_path.exists():
        print("No team_matches.parquet found. Run --clean first.")
        return

    team_match_df = pd.read_parquet(tm_path)
    team_match_df["date"] = pd.to_datetime(team_match_df["date"])

    feature_df = pd.DataFrame()
    feature_cols = []
    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    if feat_path.exists() and feat_cols_path.exists():
        import json

        feature_df = pd.read_parquet(feat_path)
        with open(feat_cols_path) as f:
            feature_cols = json.load(f)
        feature_df, feature_cols = _augment_with_dl_features(feature_df, feature_cols)
        print(f"  Loaded feature matrix for winner backtest ({len(feature_cols)} features)")
    else:
        print("  Warning: no feature matrix available — winner backtest will use team-only features")

    odds_map = {}
    odds_col = "market_home_implied_prob"
    odds_path = config.BASE_STORE_DIR / "odds.parquet"
    if odds_path.exists():
        try:
            odds_df = pd.read_parquet(odds_path)
            if odds_col in odds_df.columns and "match_id" in odds_df.columns:
                odds_map = (
                    odds_df[["match_id", odds_col]]
                    .dropna(subset=[odds_col])
                    .drop_duplicates(subset=["match_id"], keep="last")
                    .set_index("match_id")[odds_col]
                    .to_dict()
                )
                print(f"  Loaded market odds for {len(odds_map)} matches")
            else:
                print("  Warning: odds.parquet missing market_home_implied_prob column")
        except Exception as e:
            print(f"  Warning: failed loading odds.parquet: {e}")

    season_df = team_match_df[team_match_df["year"] == year]
    if season_df.empty:
        print(f"No data for season {year}.")
        return

    rounds = sorted(season_df["round_number"].dropna().unique())
    print(f"\nGame Winner Walk-Forward Backtest for {year}")
    print(f"  Rounds to test: {len(rounds)}")
    print()

    def _safe_auc(y_true, y_prob):
        y = np.asarray(y_true, dtype=int)
        p = np.asarray(y_prob, dtype=float)
        if len(y) == 0 or len(np.unique(y)) < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y, p))
        except ValueError:
            return float("nan")

    def _safe_brier(y_true, y_prob):
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_prob, dtype=float)
        if len(y) == 0:
            return float("nan")
        return float(np.mean((p - y) ** 2))

    results = []
    all_actuals = []
    hybrid_probs = []
    residual_probs = []
    market_probs = []
    market_actuals = []
    total_games = 0
    hybrid_correct = 0
    residual_correct = 0
    market_correct = 0
    market_total = 0

    for rnd in rounds:
        rnd_int = int(rnd)

        # Train on all data before this round
        train_mask = (
            (team_match_df["year"] < year)
            | ((team_match_df["year"] == year) & (team_match_df["round_number"] < rnd))
        )
        tm_train = team_match_df[train_mask].copy()
        tm_round = team_match_df[
            (team_match_df["year"] == year) & (team_match_df["round_number"] == rnd)
        ].copy()

        if len(tm_train) < 50 or tm_round.empty:
            continue

        player_preds_train = pd.DataFrame()
        player_preds_test = pd.DataFrame()
        if not feature_df.empty:
            train_df = feature_df[
                (feature_df["year"] < year)
                | ((feature_df["year"] == year) & (feature_df["round_number"] < rnd))
            ].copy()
            test_df = feature_df[
                (feature_df["year"] == year) & (feature_df["round_number"] == rnd)
            ].copy()

            if len(train_df) >= 20 and not test_df.empty:
                scoring_model = AFLScoringModel()
                scoring_model.train_backtest(train_df, feature_cols)

                disposal_model = AFLDisposalModel(
                    distribution=config.DISPOSAL_DISTRIBUTION
                )
                disposal_model.train_backtest(train_df, feature_cols)

                marks_model = AFLMarksModel(
                    distribution=config.MARKS_DISTRIBUTION
                )
                marks_model.train_backtest(train_df, feature_cols)

                player_preds_train = _build_player_predictions_for_winner_features(
                    train_df,
                    feature_cols,
                    scoring_model=scoring_model,
                    disposal_model=disposal_model,
                    marks_model=marks_model,
                )
                player_preds_test = _build_player_predictions_for_winner_features(
                    test_df,
                    feature_cols,
                    scoring_model=scoring_model,
                    disposal_model=disposal_model,
                    marks_model=marks_model,
                )

        model = AFLGameWinnerModel()
        model.train_backtest(
            tm_train,
            player_predictions_df=player_preds_train if not player_preds_train.empty else None,
        )

        # Predict this round
        all_pp = (
            pd.concat([player_preds_train, player_preds_test], ignore_index=True)
            if not player_preds_test.empty else player_preds_train
        )
        game_preds = _predict_games_for_round(
            model,
            tm_train,
            tm_round,
            player_predictions_df=all_pp if not all_pp.empty else None,
        )
        if game_preds.empty:
            continue

        # Get actuals: home team margin > 0 means home win
        home_round = tm_round[tm_round["is_home"]].copy()
        actuals = home_round.set_index("match_id")["margin"].to_dict()

        round_total = 0
        round_hybrid_correct = 0
        round_residual_correct = 0
        round_market_correct = 0
        round_market_total = 0
        round_actuals = []
        round_hybrid_probs = []
        round_residual_probs = []
        round_market_probs = []
        round_market_actuals = []

        for _, row in game_preds.iterrows():
            mid = row["match_id"]
            if mid not in actuals:
                continue

            actual_win = 1 if actuals[mid] > 0 else 0
            hybrid_p = float(row.get("hybrid_prob_home", row["home_win_prob"]))
            residual_p = float(row.get("residual_prob_home", row["home_win_prob"]))
            hybrid_win = 1 if hybrid_p > 0.5 else 0
            residual_win = 1 if residual_p > 0.5 else 0

            round_total += 1
            total_games += 1
            round_hybrid_correct += int(hybrid_win == actual_win)
            round_residual_correct += int(residual_win == actual_win)
            hybrid_correct += int(hybrid_win == actual_win)
            residual_correct += int(residual_win == actual_win)

            all_actuals.append(actual_win)
            hybrid_probs.append(hybrid_p)
            residual_probs.append(residual_p)
            round_actuals.append(actual_win)
            round_hybrid_probs.append(hybrid_p)
            round_residual_probs.append(residual_p)

            mkt_available = bool(row.get("market_prior_available", 0))
            mkt_p = row.get("market_prior_prob_home", np.nan)
            if (not mkt_available or not pd.notna(mkt_p)) and (mid in odds_map):
                mkt_p = odds_map.get(mid)
                mkt_available = mkt_p is not None and pd.notna(mkt_p)
            if mkt_available and pd.notna(mkt_p):
                mkt_p = float(mkt_p)
                mkt_win = 1 if mkt_p > 0.5 else 0
                market_correct += int(mkt_win == actual_win)
                market_total += 1
                round_market_correct += int(mkt_win == actual_win)
                round_market_total += 1
                market_probs.append(mkt_p)
                market_actuals.append(actual_win)
                round_market_probs.append(mkt_p)
                round_market_actuals.append(actual_win)

        if round_total > 0:
            round_hybrid_acc = round_hybrid_correct / round_total
            round_residual_acc = round_residual_correct / round_total
            round_market_acc = (
                round_market_correct / round_market_total
                if round_market_total > 0 else np.nan
            )
            round_actuals_arr = np.array(round_actuals, dtype=float)
            round_hybrid_brier = float(np.mean((np.array(round_hybrid_probs) - round_actuals_arr) ** 2))
            round_residual_brier = float(np.mean((np.array(round_residual_probs) - round_actuals_arr) ** 2))
            round_market_brier = (
                float(np.mean((np.array(round_market_probs) - np.array(round_market_actuals, dtype=float)) ** 2))
                if round_market_total > 0 else np.nan
            )
            results.append({
                "round": rnd_int,
                "n_games": round_total,
                "hybrid_accuracy": round_hybrid_acc,
                "residual_accuracy": round_residual_acc,
                "market_accuracy": round_market_acc,
                "hybrid_brier": round_hybrid_brier,
                "residual_brier": round_residual_brier,
                "market_brier": round_market_brier,
                "market_n": round_market_total,
            })
            mkt_str = (
                f"  market_acc={round_market_acc:.3f} ({round_market_total}/{round_total})"
                if round_market_total > 0 else "  market_acc=N/A"
            )
            print(
                f"  Round {rnd_int:3d}  n={round_total:2d}  "
                f"hybrid_acc={round_hybrid_acc:.3f}  residual_acc={round_residual_acc:.3f}{mkt_str}"
            )

    if not results:
        print("No rounds had enough data for backtesting.")
        return

    # Summary
    results_df = pd.DataFrame(results)
    macro_hybrid_acc = float(results_df["hybrid_accuracy"].mean())
    macro_residual_acc = float(results_df["residual_accuracy"].mean())
    macro_market_acc = (
        float(results_df["market_accuracy"].dropna().mean())
        if results_df["market_accuracy"].notna().any() else float("nan")
    )
    overall_hybrid_acc = float(hybrid_correct / total_games) if total_games > 0 else float("nan")
    overall_residual_acc = float(residual_correct / total_games) if total_games > 0 else float("nan")
    market_acc = float(market_correct / market_total) if market_total > 0 else float("nan")
    market_coverage = float(market_total / total_games) if total_games > 0 else 0.0

    hybrid_auc = _safe_auc(all_actuals, hybrid_probs)
    residual_auc = _safe_auc(all_actuals, residual_probs)
    market_auc = _safe_auc(market_actuals, market_probs)

    hybrid_brier = _safe_brier(all_actuals, hybrid_probs)
    residual_brier = _safe_brier(all_actuals, residual_probs)
    market_brier = _safe_brier(market_actuals, market_probs)

    delta_acc_vs_residual = (
        overall_hybrid_acc - overall_residual_acc
        if not (np.isnan(overall_hybrid_acc) or np.isnan(overall_residual_acc))
        else float("nan")
    )
    delta_auc_vs_residual = (
        hybrid_auc - residual_auc
        if not (np.isnan(hybrid_auc) or np.isnan(residual_auc))
        else float("nan")
    )
    delta_brier_vs_residual = (
        hybrid_brier - residual_brier
        if not (np.isnan(hybrid_brier) or np.isnan(residual_brier))
        else float("nan")
    )
    delta_acc_vs_market = (
        overall_hybrid_acc - market_acc
        if not (np.isnan(overall_hybrid_acc) or np.isnan(market_acc))
        else float("nan")
    )
    delta_auc_vs_market = (
        hybrid_auc - market_auc
        if not (np.isnan(hybrid_auc) or np.isnan(market_auc))
        else float("nan")
    )
    delta_brier_vs_market = (
        hybrid_brier - market_brier
        if not (np.isnan(hybrid_brier) or np.isnan(market_brier))
        else float("nan")
    )

    print(f"\n{'='*60}")
    print(f"GAME WINNER BACKTEST SUMMARY — {year}")
    print(f"{'='*60}")
    print(f"  Rounds tested:       {len(results)}")
    print(f"  Total games:         {total_games}")
    print(f"  Hybrid weighted acc: {overall_hybrid_acc:.3f}")
    print(f"  Legacy weighted acc: {overall_residual_acc:.3f}")
    if not np.isnan(market_acc):
        print(
            f"  Market weighted acc: {market_acc:.3f} "
            f"({market_total}/{total_games}, {market_coverage*100:.1f}% coverage)"
        )
    print(f"  Hybrid macro acc:    {macro_hybrid_acc:.3f}")
    print(f"  Legacy macro acc:    {macro_residual_acc:.3f}")
    if not np.isnan(macro_market_acc):
        print(f"  Market macro acc:    {macro_market_acc:.3f}")
    if not np.isnan(hybrid_auc):
        print(f"  Hybrid AUC:          {hybrid_auc:.4f}")
    if not np.isnan(residual_auc):
        print(f"  Legacy AUC:          {residual_auc:.4f}")
    if not np.isnan(market_auc):
        print(f"  Market AUC:          {market_auc:.4f}")
    print(f"  Hybrid Brier:        {hybrid_brier:.4f}")
    print(f"  Legacy Brier:        {residual_brier:.4f}")
    if not np.isnan(market_brier):
        print(f"  Market Brier:        {market_brier:.4f}")
    print("\n  Hybrid deltas (better: +acc/+AUC, -Brier):")
    print(
        f"    vs Legacy:  acc={delta_acc_vs_residual:+.3f}  "
        f"auc={delta_auc_vs_residual:+.4f}  brier={delta_brier_vs_residual:+.4f}"
    )
    if not np.isnan(delta_acc_vs_market):
        print(
            f"    vs Market:  acc={delta_acc_vs_market:+.3f}  "
            f"auc={delta_auc_vs_market:+.4f}  brier={delta_brier_vs_market:+.4f}"
        )

    min_cov = float(getattr(config, "WINNER_MIN_MARKET_COVERAGE", 0.30))
    if market_coverage < min_cov:
        print(
            f"\n  Warning: market coverage {market_coverage*100:.1f}% "
            f"is below configured minimum {min_cov*100:.1f}%"
        )

    # Save results
    config.ensure_dirs()
    out_path = config.BACKTEST_DIR / f"backtest_winner_{year}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")


def cmd_tune(args):
    """Hyperparameter tuning using Optuna with walk-forward cross-validation.

    Supports --tune-target {scoring, disposals, winner}.
    Saves best params to data/tuning/best_params_{target}.json for review.
    Does NOT auto-apply — user reviews and updates config.py manually.
    """
    import json

    try:
        import optuna
    except ImportError:
        print("Error: optuna not installed. Run: pip install optuna")
        return

    target = getattr(args, "tune_target", "scoring")
    n_trials = getattr(config, "TUNE_N_TRIALS", 50)
    n_folds = getattr(config, "TUNE_WALK_FORWARD_FOLDS", 3)

    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    if not feat_path.exists() or not feat_cols_path.exists():
        print("No feature matrix found. Run --features first.")
        return

    feature_df = pd.read_parquet(feat_path)
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    years = sorted(feature_df["year"].unique())
    if len(years) < n_folds + 2:
        print(f"Not enough years for {n_folds}-fold walk-forward CV. Have {len(years)} years.")
        return

    # Walk-forward folds: train on years[:split], validate on years[split]
    fold_splits = []
    for i in range(n_folds):
        val_year = years[-(n_folds - i)]
        fold_splits.append(val_year)

    print(f"\nHyperparameter tuning for '{target}'")
    print(f"  Trials: {n_trials}")
    print(f"  Walk-forward validation years: {fold_splits}")
    print()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if target == "scoring":
        def objective(trial):
            gbt_params = {
                "max_iter": trial.suggest_int("max_iter", *config.TUNE_GBT_RANGES["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *config.TUNE_GBT_RANGES["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *config.TUNE_GBT_RANGES["learning_rate"], log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", *config.TUNE_GBT_RANGES["min_samples_leaf"]),
                "random_state": config.RANDOM_SEED,
            }
            poisson_alpha = trial.suggest_float("poisson_alpha", *config.TUNE_POISSON_ALPHA_RANGE, log=True)
            w_poi = trial.suggest_float("w_poisson", *config.TUNE_ENSEMBLE_POISSON_WEIGHT_RANGE)

            maes = []
            for val_year in fold_splits:
                train_df = feature_df[feature_df["year"] < val_year].copy()
                val_df = feature_df[feature_df["year"] == val_year].copy()
                if val_df.empty or len(train_df) < 100:
                    continue

                model = AFLScoringModel(
                    gbt_params=gbt_params,
                    poisson_params={"alpha": poisson_alpha, "max_iter": 1000},
                    ensemble_weights={"poisson": w_poi, "gbt": 1 - w_poi},
                )
                model.train_backtest(train_df, feature_cols)
                from model import _prepare_features
                X_raw, X_clean, X_scaled = _prepare_features(
                    val_df, feature_cols, scaler=model.scaler
                )
                pred_goals, _, _ = model._ensemble_predict(X_raw, X_scaled, "goals", df=val_df)
                mae = float(np.mean(np.abs(val_df["GL"].values - pred_goals)))
                maes.append(mae)

            return np.mean(maes) if maes else float("inf")

    elif target == "disposals":
        def objective(trial):
            gbt_params = {
                "max_iter": trial.suggest_int("max_iter", *config.TUNE_GBT_RANGES["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *config.TUNE_GBT_RANGES["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *config.TUNE_GBT_RANGES["learning_rate"], log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", *config.TUNE_GBT_RANGES["min_samples_leaf"]),
                "random_state": config.RANDOM_SEED,
            }
            poisson_alpha = trial.suggest_float("poisson_alpha", *config.TUNE_POISSON_ALPHA_RANGE, log=True)

            maes = []
            for val_year in fold_splits:
                train_df = feature_df[feature_df["year"] < val_year].copy()
                val_df = feature_df[feature_df["year"] == val_year].copy()
                if val_df.empty or len(train_df) < 100:
                    continue

                model = AFLDisposalModel(
                    distribution=config.DISPOSAL_DISTRIBUTION,
                    gbt_params=gbt_params,
                    poisson_params={"alpha": poisson_alpha, "max_iter": 1000},
                )
                model.train_backtest(train_df, feature_cols)
                from model import _prepare_features
                X_raw, X_clean, X_scaled = _prepare_features(
                    val_df, feature_cols, scaler=model.scaler
                )
                pred_disp = model._predict_raw(X_clean, X_scaled, df=val_df)
                mae = float(np.mean(np.abs(val_df["DI"].values - pred_disp)))
                maes.append(mae)

            return np.mean(maes) if maes else float("inf")

    elif target == "winner":
        tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
        if not tm_path.exists():
            print("No team_matches.parquet found. Run --clean first.")
            return
        team_match_df = pd.read_parquet(tm_path)
        team_match_df["date"] = pd.to_datetime(team_match_df["date"])

        def objective(trial):
            k_factor = trial.suggest_float("k_factor", *config.TUNE_ELO_RANGES["k_factor"])
            home_adv = trial.suggest_float("home_advantage", *config.TUNE_ELO_RANGES["home_advantage"])
            regression = trial.suggest_float("season_regression", *config.TUNE_ELO_RANGES["season_regression"])
            gw_params = {
                "max_iter": trial.suggest_int("max_iter", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
                "random_state": config.RANDOM_SEED,
            }

            aucs = []
            for val_year in fold_splits:
                tm_train = team_match_df[team_match_df["year"] < val_year].copy()
                tm_val = team_match_df[team_match_df["year"] == val_year].copy()
                if tm_val.empty or len(tm_train) < 100:
                    continue

                model = AFLGameWinnerModel(
                    gbt_params=gw_params,
                    elo_params={
                        "k_factor": k_factor,
                        "home_advantage": home_adv,
                        "season_regression": regression,
                    },
                )
                model.train_backtest(tm_train)
                # Evaluate on validation
                elo_df = model.elo_system.compute_all(
                    pd.concat([tm_train, tm_val], ignore_index=True).sort_values("date")
                )
                game_df, feats = model.build_game_features(
                    pd.concat([tm_train, tm_val], ignore_index=True).sort_values("date"),
                    elo_df,
                )
                val_games = game_df[game_df["year"] == val_year]
                if val_games.empty:
                    continue
                X_val = val_games[feats].fillna(0)
                y_val = val_games["home_win"].values
                pred_prob = model.classifier.predict_proba(X_val)[:, 1]
                try:
                    auc = roc_auc_score(y_val, pred_prob)
                    aucs.append(auc)
                except ValueError:
                    pass

            return -np.mean(aucs) if aucs else 0.0  # negate for minimize

    else:
        print(f"Unknown tune target: {target}. Use scoring, disposals, or winner.")
        return

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"TUNING RESULTS — {target}")
    print(f"{'='*60}")
    print(f"  Best trial: #{best.number}")
    print(f"  Best value: {best.value:.6f}")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save results
    config.ensure_dirs()
    out_path = config.TUNING_DIR / f"best_params_{target}.json"
    result = {
        "target": target,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "fold_years": fold_splits,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Review and update config.py manually to apply.")


def cmd_sequential_report(args):
    """Generate post-season report from sequential learning data."""
    import json

    year = args.year
    if not year:
        print("Error: --sequential-report requires --year YYYY")
        return

    from analysis import generate_season_report

    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=getattr(args, "run_id", None))
    report = generate_season_report(store, year)

    if "error" in report:
        print(f"Error: {report['error']}")
        return

    # Save JSON
    report_path = config.SEQUENTIAL_DIR / f"season_report_{year}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Season report saved to {report_path}")
    summary = store.get_learning_summary(year)
    if summary.get("run_id"):
        print(f"Run ID: {summary['run_id']}")

    # Print formatted summary
    print(f"\n{'='*70}")
    print(f"SEASON REPORT — {year}")
    print(f"{'='*70}")
    print(f"  Rounds analyzed: {report.get('rounds_analyzed', 0)}")

    # --- Threshold evaluation (PRIMARY) ---
    te = report.get("threshold_evaluation", {})
    if te:
        print(f"\n  Probability Evaluation (PRIMARY):")
        # Display names for nicer output
        display_names = {
            "1plus_goals": "Goals 1+", "2plus_goals": "Goals 2+", "3plus_goals": "Goals 3+",
            "10plus_disp": "Disp 10+", "15plus_disp": "Disp 15+", "20plus_disp": "Disp 20+",
            "25plus_disp": "Disp 25+", "30plus_disp": "Disp 30+",
        }
        print(f"    {'Threshold':<14s} {'Brier':>8s} {'LogLoss':>8s} {'BaseRate':>9s} {'n':>7s}")
        for name in ["1plus_goals", "2plus_goals", "3plus_goals",
                      "10plus_disp", "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp"]:
            m = te.get(name)
            if m:
                dname = display_names.get(name, name)
                print(f"    {dname:<14s} {m['brier_score']:8.4f} {m['log_loss']:8.4f} {m['base_rate']:9.4f} {m['n']:7d}")

        # Calibration curves
        print(f"\n  Calibration Curves:")
        for name in ["1plus_goals", "2plus_goals", "3plus_goals",
                      "10plus_disp", "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp"]:
            m = te.get(name)
            if m and m.get("calibration_curve"):
                dname = display_names.get(name, name)
                print(f"    {dname} (Brier={m['brier_score']:.4f}):")
                print(f"      {'Predicted':>12s} {'Observed':>10s} {'n':>6s} {'Gap':>7s}")
                for b in m["calibration_curve"]:
                    gap = b["observed_mean"] - b["predicted_mean"]
                    print(f"      {b['bin_lower']:.2f}-{b['bin_upper']:.2f}    {b['predicted_mean']:6.2f}    {b['observed_mean']:6.2f}  {b['count']:5d}   {gap:+.2f}")

    # --- Learning curve ---
    lc = report.get("learning_curve", {})
    if lc.get("first_half_mae") is not None:
        print(f"\n  Learning Curve:")
        print(f"    First-half MAE:  {lc['first_half_mae']}")
        print(f"    Second-half MAE: {lc['second_half_mae']}")
        print(f"    Learning effect: {lc.get('learning_effect_pct', 'N/A')}%")
        if lc.get("first_half_brier_1plus") is not None:
            print(f"    First-half Brier (1+):  {lc['first_half_brier_1plus']}")
            print(f"    Second-half Brier (1+): {lc['second_half_brier_1plus']}")

    # --- MAE (SECONDARY) ---
    lc_rounds = lc.get("rounds", [])
    if lc_rounds:
        maes = [r["goals_mae"] for r in lc_rounds if r.get("goals_mae") is not None]
        if maes:
            print(f"\n  Goals MAE (SECONDARY):")
            print(f"    Overall:  {np.mean(maes):.4f}")

    # Calibration
    cc = report.get("calibration_curve", {})
    if cc.get("mean_absolute_calibration_error") is not None:
        print(f"\n  Legacy Calibration:")
        print(f"    Mean abs error:  {cc['mean_absolute_calibration_error']}")

    # Miss distribution
    md = report.get("miss_type_distribution", {})
    if md.get("total_significant_misses", 0) > 0:
        print(f"\n  Miss Classification ({md['total_significant_misses']} total):")
        for mt, pct in md.get("percentages", {}).items():
            count = md.get("counts", {}).get(mt, 0)
            print(f"    {mt:20s}  {count:4d}  ({pct}%)")

    # Weather
    ws = report.get("weather_summary", {})
    if ws.get("total_wet_matches", 0) > 0:
        print(f"\n  Weather Impact:")
        print(f"    Wet matches: {ws['total_wet_matches']}/{ws['total_matches']} ({ws.get('wet_match_pct', 0)}%)")
        if ws.get("avg_wet_mae") is not None:
            print(f"    Wet MAE:  {ws['avg_wet_mae']}")
            print(f"    Dry MAE:  {ws.get('avg_dry_mae', 'N/A')}")

    # Game winner
    gw = report.get("game_winner_accuracy", {})
    if gw.get("total_games", 0) > 0:
        print(f"\n  Game Winner Predictions:")
        print(f"    Accuracy: {gw['correct_predictions']}/{gw['total_games']} ({gw.get('accuracy_pct', 0)}%)")
        if gw.get("margin_mae") is not None:
            print(f"    Margin MAE: {gw['margin_mae']}")

    # Player leaderboard
    pl = report.get("player_leaderboard", {})
    if pl.get("best"):
        print(f"\n  Best Predicted Players:")
        for p in pl["best"][:5]:
            print(f"    {p['player']:30s} {p['team']:15s}  MAE={p['mae']}  n={p['appearances']}")
    if pl.get("worst"):
        print(f"\n  Worst Predicted Players:")
        for p in pl["worst"][:5]:
            print(f"    {p['player']:30s} {p['team']:15s}  MAE={p['mae']}  n={p['appearances']}")

    # Archetype accuracy
    aa = report.get("archetype_accuracy", {})
    if aa.get("per_archetype"):
        print(f"\n  Per-Archetype Accuracy:")
        for a in aa["per_archetype"]:
            auc_s = f"AUC={a['scorer_auc']}" if a.get("scorer_auc") is not None else "AUC=N/A"
            print(f"    Archetype {a['archetype']}:  MAE={a.get('goals_mae', 'N/A')}  {auc_s}  n={a['n_predictions']}")

    # Streak summary
    ss = report.get("streak_summary", {})
    if ss.get("longest_hot_streaks"):
        print(f"\n  Top Scoring Streaks:")
        for h in ss["longest_hot_streaks"][:5]:
            print(f"    {h['player']:30s} {h['team']:15s}  {h.get('streak', 0)} rounds (R{h.get('round', '?')})")

    print()
    return report


def _compute_hit_rates(store, year):
    """Compute hit-rate metrics (accuracy/precision/recall at P>=0.50) per threshold."""
    from analysis import _extract_threshold_data, _load_season_merged_predictions_outcomes

    merged = _load_season_merged_predictions_outcomes(store, year)
    if merged.empty:
        return {}

    threshold_data = _extract_threshold_data(merged)
    results = {}
    for name, (pred_probs, actual_binary) in threshold_data.items():
        n = len(pred_probs)
        if n < 10:
            continue
        predicted_yes = (pred_probs >= 0.50).astype(int)
        actual = actual_binary.astype(int)
        correct = (predicted_yes == actual).sum()
        tp = ((predicted_yes == 1) & (actual == 1)).sum()
        fp = ((predicted_yes == 1) & (actual == 0)).sum()
        fn = ((predicted_yes == 0) & (actual == 1)).sum()
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        results[name] = {
            "accuracy": float(correct / n),
            "precision": precision,
            "recall": recall,
            "n": int(n),
            "n_predicted_yes": int(predicted_yes.sum()),
            "n_actual_yes": int(actual.sum()),
        }
    return results


def _print_year_summary(year, report, hit_rates):
    """Print abbreviated per-year summary after backtest."""
    display_names = {
        "1plus_goals": "Goals 1+", "2plus_goals": "Goals 2+", "3plus_goals": "Goals 3+",
        "10plus_disp": "Disp 10+", "15plus_disp": "Disp 15+", "20plus_disp": "Disp 20+",
        "25plus_disp": "Disp 25+", "30plus_disp": "Disp 30+",
    }
    threshold_order = [
        "1plus_goals", "2plus_goals", "3plus_goals",
        "10plus_disp", "15plus_disp", "20plus_disp", "25plus_disp", "30plus_disp",
    ]

    print(f"\n{'='*70}")
    print(f"SEASON REPORT — {year}")
    print(f"{'='*70}")

    # Threshold Brier table
    te = report.get("threshold_evaluation", {})
    if te:
        print(f"\n  Probability Evaluation:")
        print(f"    {'Threshold':<14s} {'Brier':>8s} {'LogLoss':>8s} {'BaseRate':>9s} {'n':>7s}")
        for name in threshold_order:
            m = te.get(name)
            if m:
                dname = display_names.get(name, name)
                print(f"    {dname:<14s} {m['brier_score']:8.4f} {m['log_loss']:8.4f} {m['base_rate']:9.4f} {m['n']:7d}")

    # Hit rate table
    if hit_rates:
        print(f"\n  Hit Rates (P >= 0.50):")
        print(f"    {'Threshold':<14s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'PredY':>7s} {'ActY':>7s} {'n':>7s}")
        for name in threshold_order:
            hr = hit_rates.get(name)
            if hr:
                dname = display_names.get(name, name)
                print(f"    {dname:<14s} {hr['accuracy']:7.3f} {hr['precision']:7.3f} {hr['recall']:7.3f} "
                      f"{hr['n_predicted_yes']:7d} {hr['n_actual_yes']:7d} {hr['n']:7d}")

    # Game winner
    gw = report.get("game_winner_accuracy", {})
    if gw.get("total_games", 0) > 0:
        print(f"\n  Game Winner: {gw['correct_predictions']}/{gw['total_games']} "
              f"({gw.get('accuracy_pct', 0)}%)"
              f"  Margin MAE: {gw.get('margin_mae', 'N/A')}")

    # Learning effect
    lc = report.get("learning_curve", {})
    if lc.get("learning_effect_pct") is not None:
        print(f"  Learning effect: {lc['learning_effect_pct']:+.1f}%"
              f"  (1st half MAE: {lc['first_half_mae']}, 2nd half MAE: {lc['second_half_mae']})")

    print()


def _print_cross_season_comparison(year_reports):
    """Print side-by-side comparison table across all years."""
    if len(year_reports) < 2:
        return

    years = [yr["year"] for yr in year_reports]
    col_w = 12

    def _header():
        cols = "".join(f"{yr:>{col_w}}" for yr in years)
        return f"    {'':20s}{cols}"

    def _row(label, extractor, fmt=".4f"):
        vals = []
        for yr in year_reports:
            try:
                v = extractor(yr)
            except (KeyError, TypeError):
                v = None
            if v is None:
                vals.append(f"{'N/A':>{col_w}}")
            else:
                formatted = f"{v:{fmt}}"
                vals.append(f"{formatted:>{col_w}}")
        return f"    {label:20s}{''.join(vals)}"

    print(f"{'='*70}")
    print(f"CROSS-SEASON COMPARISON")
    print(f"{'='*70}")
    print(_header())
    print(f"    {'─'*20}{'─'*col_w*len(years)}")

    # Brier Scores
    print(f"\n  Brier Scores (lower = better):")
    for name, dname in [("1plus_goals", "Goals 1+"), ("2plus_goals", "Goals 2+"),
                        ("3plus_goals", "Goals 3+"), ("20plus_disp", "Disp 20+")]:
        print(_row(dname, lambda r, n=name: r["report"]["threshold_evaluation"].get(n, {}).get("brier_score")))

    # Hit Rates
    print(f"\n  Hit Rate @ P>=0.50 (higher = better):")
    for name, dname in [("1plus_goals", "Goals 1+"), ("2plus_goals", "Goals 2+"),
                        ("3plus_goals", "Goals 3+"), ("20plus_disp", "Disp 20+")]:
        print(_row(dname, lambda r, n=name: r["hit_rates"].get(n, {}).get("accuracy"), fmt=".3f"))

    # Learning Trajectory
    print(f"\n  Learning Trajectory:")
    print(_row("Overall MAE", lambda r: r["overall_mae"]))
    print(_row("1st Half MAE", lambda r: r["report"]["learning_curve"].get("first_half_mae")))
    print(_row("2nd Half MAE", lambda r: r["report"]["learning_curve"].get("second_half_mae")))
    print(_row("Learning %", lambda r: r["report"]["learning_curve"].get("learning_effect_pct"), fmt="+.1f"))

    # Game Winner
    print(f"\n  Game Winner:")
    print(_row("Accuracy %", lambda r: r["report"]["game_winner_accuracy"].get("accuracy_pct"), fmt=".1f"))
    print(_row("Margin MAE", lambda r: r["report"]["game_winner_accuracy"].get("margin_mae")))

    # Year-over-year trend for Brier 1+ goals
    print(f"\n  Year-over-Year Brier (Goals 1+):")
    for i in range(1, len(year_reports)):
        prev = year_reports[i - 1]
        curr = year_reports[i]
        try:
            prev_b = prev["report"]["threshold_evaluation"]["1plus_goals"]["brier_score"]
            curr_b = curr["report"]["threshold_evaluation"]["1plus_goals"]["brier_score"]
            delta = curr_b - prev_b
            direction = "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged"
            print(f"    {prev['year']} → {curr['year']}: {delta:+.4f} ({direction})")
        except (KeyError, TypeError):
            print(f"    {prev['year']} → {curr['year']}: N/A")

    print()


def cmd_sequential_year_range(args, disposal_distribution=None):
    """Run sequential learning across multiple seasons with full reports."""
    import json
    import time

    year_range = args.year_range
    if not year_range:
        print("Error: --year-range requires format YYYY-YYYY (e.g., 2023-2025)")
        return

    try:
        start_year, end_year = [int(y) for y in year_range.split("-")]
    except ValueError:
        print("Error: --year-range format must be YYYY-YYYY (e.g., 2023-2025)")
        return

    from analysis import generate_season_report

    years = list(range(start_year, end_year + 1))
    est_rounds = len(years) * 28
    est_min = est_rounds * 8 / 60

    print(f"\nMulti-season sequential learning: {years}")
    print(f"  ~{est_rounds} rounds x ~8s = ~{est_min:.0f} min estimated")
    print(f"{'='*70}")

    year_reports = []
    total_start = time.time()

    for yr in years:
        print(f"\n{'='*70}")
        print(f"SEASON {yr}")
        print(f"{'='*70}")

        # Run the walk-forward backtest for this year
        class YearArgs:
            year = yr
        cmd_sequential(YearArgs(), disposal_distribution=disposal_distribution)

        # Generate the full season report
        store = LearningStore(base_dir=config.SEQUENTIAL_DIR)
        report = generate_season_report(store, yr)

        if "error" in report:
            print(f"  Warning: Report generation failed — {report['error']}")
            continue

        # Compute hit rates
        hit_rates = _compute_hit_rates(store, yr)

        # Compute overall MAE from learning curve
        lc_rounds = report.get("learning_curve", {}).get("rounds", [])
        maes = [r["goals_mae"] for r in lc_rounds if r.get("goals_mae") is not None]
        overall_mae = float(np.mean(maes)) if maes else None

        # Save JSON report
        report_path = config.SEQUENTIAL_DIR / f"season_report_{yr}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved: {report_path}")

        # Print per-year summary
        _print_year_summary(yr, report, hit_rates)

        # Accumulate for cross-season comparison
        year_reports.append({
            "year": yr,
            "report": report,
            "hit_rates": hit_rates,
            "overall_mae": overall_mae,
        })

    # Cross-season comparison
    if len(year_reports) >= 2:
        _print_cross_season_comparison(year_reports)

    total_elapsed = time.time() - total_start
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Output: {config.SEQUENTIAL_DIR}")


def cmd_simulate(args):
    """Run Monte Carlo simulation for a round using trained models."""
    import json
    import time

    year = args.year or config.SEQUENTIAL_YEAR
    rnd = args.round
    n_sims = getattr(args, "n_sims", 10000)

    if not rnd:
        print("Error: --simulate requires --round N")
        return

    print(f"\nMonte Carlo Simulation — {year} Round {rnd} ({n_sims:,} sims)")
    print("=" * 60)

    # Load feature matrix
    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    if not feat_path.exists():
        print("No feature matrix found. Run --features first.")
        return

    feature_df = pd.read_parquet(feat_path)
    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    # Filter to target round
    test_df = feature_df[
        (feature_df["year"] == year)
        & (feature_df["round_number"] == rnd)
    ].copy()
    train_df = feature_df[
        (feature_df["year"] < year)
        | ((feature_df["year"] == year) & (feature_df["round_number"] < rnd))
    ].copy()

    if test_df.empty:
        print(f"No data for {year} R{rnd}")
        return

    print(f"  Players: {len(test_df)}")

    # Train models (backtest-style)
    scoring_model = AFLScoringModel()
    scoring_model.train_backtest(train_df, feature_cols)

    disposal_model = AFLDisposalModel(
        distribution=config.DISPOSAL_DISTRIBUTION
    )
    disposal_model.train_backtest(train_df, feature_cols)

    # Predict distributions
    scoring_preds = scoring_model.predict_distributions(test_df, feature_cols=feature_cols)
    disposal_preds = disposal_model.predict_distributions(test_df, feature_cols=feature_cols)
    merged_preds = _merge_predictions(scoring_preds, disposal_preds)

    # Propagate is_home from test_df
    if "is_home" in test_df.columns:
        merged_preds["is_home"] = test_df["is_home"].values

    # Game winner predictions
    game_preds = pd.DataFrame()
    tm_path = config.BASE_STORE_DIR / "team_matches.parquet"
    if tm_path.exists():
        team_match_df = pd.read_parquet(tm_path)
        team_match_df["date"] = pd.to_datetime(team_match_df["date"])
        tm_train = team_match_df[
            (team_match_df["year"] < year)
            | ((team_match_df["year"] == year) & (team_match_df["round_number"] < rnd))
        ].copy()
        tm_round = team_match_df[
            (team_match_df["year"] == year) & (team_match_df["round_number"] == rnd)
        ].copy()

        if len(tm_train) >= 20 and not tm_round.empty:
            winner_model = AFLGameWinnerModel()
            try:
                train_pp = _build_player_predictions_for_winner_features(
                    train_df,
                    feature_cols,
                    scoring_model=scoring_model,
                    disposal_model=disposal_model,
                )
                test_pp = _build_player_predictions_for_winner_features(
                    test_df,
                    feature_cols,
                    scoring_model=scoring_model,
                    disposal_model=disposal_model,
                )
                all_pp = (
                    pd.concat([train_pp, test_pp], ignore_index=True)
                    if not test_pp.empty else train_pp
                )
                winner_model.train_backtest(
                    tm_train,
                    player_predictions_df=train_pp if not train_pp.empty else None,
                )
                game_preds = _predict_games_for_round(
                    winner_model,
                    tm_train,
                    tm_round,
                    player_predictions_df=all_pp if not all_pp.empty else None,
                )
            except Exception as e:
                print(f"  Warning: Game winner model failed: {e}")

    # Create simulator and estimate correlations
    from model import MonteCarloSimulator
    simulator = MonteCarloSimulator(
        scoring_model=scoring_model,
        disposal_model=disposal_model,
    )
    simulator.estimate_correlation_factors(train_df, team_match_df if tm_path.exists() else None)

    # Run simulation
    t0 = time.time()
    mc_results = simulator.simulate_round(
        merged_preds, game_preds_df=game_preds, n_sims=n_sims
    )
    elapsed = time.time() - t0
    print(f"\n  Simulation completed in {elapsed:.1f}s ({n_sims:,} sims x {len(test_df)} players)")

    # Save results
    out_dir = Path(config.SEQUENTIAL_DIR) / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}_R{rnd:02d}_simulated.csv"
    mc_results.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    # Print comparison table
    _print_mc_comparison(mc_results)


def _print_mc_comparison(mc_df, n_show=15):
    """Print comparison table: direct model vs Monte Carlo probabilities."""
    # Sort by direct p_1plus descending, show top N
    df = mc_df.sort_values("direct_p_1plus_goals", ascending=False).head(n_show)

    print(f"\n{'=' * 95}")
    print(f"  Direct Model vs Monte Carlo — Top {n_show} by P(1+ Goals)")
    print(f"{'=' * 95}")
    print(f"{'Player':<22} {'Team':<12} {'Direct':>8} {'MC':>8} {'Diff':>7}  "
          f"{'D_20+d':>7} {'MC_20+d':>7} {'Diff':>7}")
    print("-" * 95)

    for _, row in df.iterrows():
        d1 = row["direct_p_1plus_goals"]
        m1 = row["mc_p_1plus_goals"]
        diff1 = m1 - d1
        d20 = row.get("direct_p_20plus_disp", float("nan"))
        m20 = row.get("mc_p_20plus_disp", float("nan"))
        diff20 = m20 - d20 if not (np.isnan(d20) or np.isnan(m20)) else float("nan")

        d1_s = f"{d1:.1%}" if not np.isnan(d1) else "N/A"
        m1_s = f"{m1:.1%}"
        diff1_s = f"{diff1:+.1%}" if not np.isnan(diff1) else "N/A"
        d20_s = f"{d20:.1%}" if not np.isnan(d20) else "N/A"
        m20_s = f"{m20:.1%}"
        diff20_s = f"{diff20:+.1%}" if not np.isnan(diff20) else "N/A"

        print(f"{str(row['player']):<22} {str(row['team']):<12} "
              f"{d1_s:>8} {m1_s:>8} {diff1_s:>7}  "
              f"{d20_s:>7} {m20_s:>7} {diff20_s:>7}")


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
                        help="Train scoring prediction models")
    parser.add_argument("--train-disposals", action="store_true",
                        dest="train_disposals",
                        help="Train disposal prediction model")
    parser.add_argument("--train-marks", action="store_true",
                        dest="train_marks",
                        help="Train marks prediction model")
    parser.add_argument("--train-winner", action="store_true",
                        dest="train_winner",
                        help="Train game winner prediction model")
    parser.add_argument("--train-embeddings", action="store_true",
                        dest="train_embeddings",
                        help="Train entity embedding model (Phase 2 — requires PyTorch)")
    parser.add_argument("--train-sequence", action="store_true",
                        dest="train_sequence",
                        help="Train GRU sequence form model (Phase 4 — requires PyTorch)")
    parser.add_argument("--predict", action="store_true",
                        help="Generate predictions for a round")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model on validation set")
    parser.add_argument("--backtest", action="store_true",
                        help="Walk-forward backtest on a season (requires --year)")
    parser.add_argument("--backtest-winner", action="store_true",
                        dest="backtest_winner",
                        help="Walk-forward backtest for game winner model (requires --year)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Diagnostic report on backtest results (requires --year)")
    parser.add_argument("--tune", action="store_true",
                        help="Hyperparameter tuning using Optuna with walk-forward CV")
    parser.add_argument("--tune-target", type=str, dest="tune_target",
                        default="scoring", choices=["scoring", "disposals", "winner"],
                        help="Target model for tuning (default: scoring)")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential learning: round-by-round with calibration feedback (requires --year)")
    parser.add_argument("--sequential-report", action="store_true",
                        dest="sequential_report",
                        help="Generate post-season report from sequential learning data (requires --year)")
    parser.add_argument("--year-range", type=str, dest="year_range",
                        help="Run sequential learning across multiple seasons (e.g., 2023-2025)")
    parser.add_argument("--run-id", type=str, dest="run_id",
                        help="Optional run identifier for LearningStore outputs")
    parser.add_argument("--reset-calibration", action="store_true",
                        help="Reset calibration state for the active run before sequential processing")

    parser.add_argument("--scrape-profiles", action="store_true",
                        dest="scrape_profiles",
                        help="Scrape player profile pages (height/weight/DOB + career splits)")
    parser.add_argument("--scrape-footywire", action="store_true",
                        dest="scrape_footywire",
                        help="Scrape FootyWire advanced stats (ED, DE%%, CCL, SCL, TO, MG, TOG%%)")
    parser.add_argument("--scrape-live", action="store_true",
                        dest="scrape_live",
                        help="Scrape FootyWire basic match stats for current season (incremental)")
    parser.add_argument("--scrape-news", action="store_true",
                        dest="scrape_news",
                        help="Scrape team selections + injury list for upcoming round")
    parser.add_argument("--daily", action="store_true",
                        help="Daily pipeline: scrape live + clean + features (for cron)")
    parser.add_argument("--multi", action="store_true",
                        help="Run correlated multi-bet analysis (requires --round)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run Monte Carlo simulation after prediction (requires --round or --sequential)")
    parser.add_argument("--n-sims", type=int, dest="n_sims", default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--save-experiment", type=str, dest="save_experiment",
                        help="Save backtest results as named experiment JSON (e.g. 'baseline_pre')")

    parser.add_argument("--start", type=int, help="Start year for scraping")
    parser.add_argument("--end", type=int, help="End year for scraping")
    parser.add_argument("--round", type=int, help="Round number for prediction")
    parser.add_argument("--year", type=int, help="Season year (default: current)")
    parser.add_argument("--player", type=str,
                        help="Player search term (e.g., \"Cripps\", \"Patrick Cripps\")")
    parser.add_argument("--player-detail", action="store_true",
                        dest="player_detail",
                        help="Show extended profile with all stats")
    parser.add_argument("--model", type=str, default="scoring",
                        choices=list(config.MODEL_TARGETS.keys()),
                        help="Model target (default: scoring)")

    args = parser.parse_args()
    _set_global_seed()

    # No flags → show help
    if not any([args.scrape, args.update, args.clean, args.features,
                args.train, args.train_disposals, args.train_marks, args.train_winner,
                args.train_embeddings, args.train_sequence,
                args.predict, args.evaluate, args.backtest,
                args.backtest_winner, args.tune,
                args.diagnose, args.sequential, args.sequential_report,
                args.year_range, args.player, args.scrape_profiles,
                args.scrape_footywire, args.scrape_live, args.scrape_news,
                args.daily, args.simulate, args.multi]):
        parser.print_help()
        return

    config.ensure_dirs()

    if args.scrape:
        cmd_scrape(args)

    if args.scrape_profiles:
        cmd_scrape_profiles(args)

    if args.scrape_footywire:
        cmd_scrape_footywire(args)

    if args.scrape_live:
        cmd_scrape_live(args)

    if args.scrape_news:
        cmd_scrape_news(args)

    if args.daily:
        cmd_daily(args)
        return  # daily runs everything

    if args.update:
        cmd_update(args)
        return  # update runs everything

    if args.clean:
        cleaned = cmd_clean(args)

    if args.features:
        cmd_features(args)

    if args.train:
        cmd_train(args)

    if args.train_disposals:
        cmd_train_disposals(args)

    if args.train_marks:
        cmd_train_marks(args)

    if args.train_embeddings:
        cmd_train_embeddings(args)

    if args.train_sequence:
        cmd_train_sequence(args)

    if args.train_winner:
        cmd_train_winner(args)

    if args.predict:
        if args.round is None:
            print("Error: --predict requires --round N")
            return
        cmd_predict(args)

    if args.evaluate:
        cmd_evaluate(args)

    if args.backtest:
        cmd_backtest(args)

    if args.backtest_winner:
        cmd_backtest_winner(args)

    if args.tune:
        cmd_tune(args)

    if args.diagnose:
        cmd_diagnose(args)

    if args.sequential:
        cmd_sequential(args)

    if args.sequential_report:
        cmd_sequential_report(args)

    if args.player:
        from player import cmd_player
        cmd_player(args)

    if args.simulate:
        cmd_simulate(args)

    if args.multi:
        from multi import cmd_multi
        cmd_multi(args)

    if args.year_range:
        cmd_sequential_year_range(args)


if __name__ == "__main__":
    main()
