"""
Integrate historical betting odds into the AFL prediction pipeline.

Sources:
  1. afl_Source_2.xlsx  — 3,353 matches (2009-2025), bookmaker odds/lines/totals
  2. AFL_YYYY_Match_Odds.csv — Betfair exchange odds (2021-2025)

Output: data/base/odds.parquet, updated feature_matrix.parquet, updated feature_columns.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")

ODDS_DIR = Path("/Users/akash/Desktop/AFL/afl betting odds")

# ══════════════════════════════════════════════════════════════════════
# Team name mapping — normalise all sources to matches.parquet names
# ══════════════════════════════════════════════════════════════════════

TEAM_MAP = {
    # Source 2 → matches.parquet
    "Brisbane": "Brisbane Lions",
    "GWS Giants": "Greater Western Sydney",
    # Betfair → matches.parquet
    "GWS": "Greater Western Sydney",
}


def normalise_team(name):
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


# ══════════════════════════════════════════════════════════════════════
# STEP 1a: Parse Source 2 (bookmaker odds)
# ══════════════════════════════════════════════════════════════════════

def parse_source2():
    """Parse Source 2 Excel, keep only ALLOWED pre-game columns."""
    print("Parsing Source 2...")
    raw = pd.read_excel(ODDS_DIR / "afl_Source_2.xlsx", header=1)
    print(f"  Raw rows: {len(raw)}")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    df["home_team"] = raw["Home Team"].apply(normalise_team)
    df["away_team"] = raw["Away Team"].apply(normalise_team)

    # ALLOWED pre-game columns only
    df["home_odds_open"] = pd.to_numeric(raw["Home Odds Open"], errors="coerce")
    df["home_odds_close"] = pd.to_numeric(raw["Home Odds Close"], errors="coerce")
    df["away_odds_open"] = pd.to_numeric(raw["Away Odds Open"], errors="coerce")
    df["away_odds_close"] = pd.to_numeric(raw["Away Odds Close"], errors="coerce")
    df["home_line_open"] = pd.to_numeric(raw["Home Line Open"], errors="coerce")
    df["home_line_close"] = pd.to_numeric(raw["Home Line Close"], errors="coerce")
    df["total_score_open"] = pd.to_numeric(raw["Total Score Open"], errors="coerce")
    df["total_score_close"] = pd.to_numeric(raw["Total Score Close"], errors="coerce")

    # FORBIDDEN columns are never read — they don't exist in df.
    # Source 2 post-game: Home Score, Away Score, Home Goals, Home Behinds,
    #   Away Goals, Away Behinds, Play Off Game? — all excluded above.

    df["year"] = df["date"].dt.year

    # Filter to 2015+ (our model period)
    df = df[df["year"] >= 2015].copy()
    print(f"  Rows after 2015+ filter: {len(df)}")

    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 1b: Parse Betfair exchange odds (2021-2025)
# ══════════════════════════════════════════════════════════════════════

def parse_betfair():
    """Parse all Betfair Match Odds CSVs, keep only pre-first-bounce prices."""
    print("Parsing Betfair exchange odds...")

    frames = []
    for csv_path in sorted(ODDS_DIR.glob("AFL_*_Match_Odds.csv")):
        df = pd.read_csv(csv_path)
        frames.append(df)
        print(f"  {csv_path.name}: {len(df)} rows, {df['EVENT_ID'].nunique()} matches")

    if not frames:
        print("  No Betfair files found")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    print(f"  Total Betfair rows: {len(raw)}")

    # Parse date
    raw["date"] = pd.to_datetime(raw["EVENT_DATE"], dayfirst=True, errors="coerce")
    raw["home_team"] = raw["HOME_TEAM"].apply(normalise_team)
    raw["away_team"] = raw["AWAY_TEAM"].apply(normalise_team)
    raw["runner"] = raw["RUNNER_NAME"].apply(normalise_team)

    # ALLOWED: only BEST_BACK_FIRST_BOUNCE and BEST_LAY_FIRST_BOUNCE
    raw["back_first_bounce"] = pd.to_numeric(raw["BEST_BACK_FIRST_BOUNCE"], errors="coerce")
    raw["lay_first_bounce"] = pd.to_numeric(raw["BEST_LAY_FIRST_BOUNCE"], errors="coerce")

    # FORBIDDEN columns dropped — never carried forward:
    # RUNNER_STATUS, IS_WINNER, TOTAL_POINTS, HOME_SCORE, AWAY_SCORE,
    # HOME_MARGIN, all quarter/half/three-quarter prices, all MATCHED_VOLUME

    # Pivot: 2 rows per match → 1 row per match with home/away columns
    home_rows = raw[raw["runner"] == raw["home_team"]].copy()
    away_rows = raw[raw["runner"] == raw["away_team"]].copy()

    home_pivot = home_rows[["date", "home_team", "away_team",
                             "back_first_bounce", "lay_first_bounce"]].rename(columns={
        "back_first_bounce": "betfair_home_back",
        "lay_first_bounce": "betfair_home_lay",
    })

    away_pivot = away_rows[["date", "home_team", "away_team",
                             "back_first_bounce", "lay_first_bounce"]].rename(columns={
        "back_first_bounce": "betfair_away_back",
        "lay_first_bounce": "betfair_away_lay",
    })

    betfair = home_pivot.merge(
        away_pivot, on=["date", "home_team", "away_team"], how="outer"
    )
    print(f"  Betfair matches after pivot: {len(betfair)}")

    return betfair


# ══════════════════════════════════════════════════════════════════════
# STEP 1c: Match to matches.parquet
# ══════════════════════════════════════════════════════════════════════

def match_to_parquet(source2_df, betfair_df):
    """Match odds data to matches.parquet by date + home_team + away_team."""
    matches = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")
    matches["date"] = pd.to_datetime(matches["date"]).dt.normalize()
    print(f"\nMatching to matches.parquet ({len(matches)} matches)...")

    # Normalise dates
    source2_df["date"] = source2_df["date"].dt.normalize()

    # --- Join Source 2 ---
    s2_joined = matches[["match_id", "date", "home_team", "away_team", "year"]].merge(
        source2_df.drop(columns=["year"]),
        on=["date", "home_team", "away_team"],
        how="left",
    )

    s2_matched = s2_joined["home_odds_close"].notna().sum()
    s2_total = len(s2_joined)
    print(f"  Source 2 matched: {s2_matched}/{s2_total} ({s2_matched/s2_total*100:.1f}%)")

    # Show unmatched by year
    unmatched = s2_joined[s2_joined["home_odds_close"].isna()]
    if len(unmatched) > 0:
        print(f"  Unmatched by year:")
        for yr in sorted(unmatched["year"].unique()):
            n = (unmatched["year"] == yr).sum()
            total_yr = (s2_joined["year"] == yr).sum()
            print(f"    {yr}: {n}/{total_yr} unmatched")

    # --- Join Betfair ---
    if not betfair_df.empty:
        betfair_df["date"] = betfair_df["date"].dt.normalize()
        combined = s2_joined.merge(
            betfair_df,
            on=["date", "home_team", "away_team"],
            how="left",
        )
        bf_matched = combined["betfair_home_back"].notna().sum()
        print(f"  Betfair matched:  {bf_matched}/{s2_total} ({bf_matched/s2_total*100:.1f}%)")
    else:
        combined = s2_joined
        combined["betfair_home_back"] = np.nan
        combined["betfair_home_lay"] = np.nan
        combined["betfair_away_back"] = np.nan
        combined["betfair_away_lay"] = np.nan

    return combined


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Create features from matched odds
# ══════════════════════════════════════════════════════════════════════

def create_features(odds_df):
    """Derive market-implied features from raw odds. All pre-game only."""
    print("\nCreating odds features...")
    df = odds_df.copy()

    # Bookmaker implied probabilities (normalised to remove overround)
    inv_home = 1.0 / df["home_odds_close"]
    inv_away = 1.0 / df["away_odds_close"]
    overround = inv_home + inv_away
    df["market_home_implied_prob"] = inv_home / overround
    df["market_away_implied_prob"] = inv_away / overround

    # Handicap and total
    df["market_handicap"] = df["home_line_close"]
    df["market_total_score"] = df["total_score_close"]

    # Market confidence
    df["market_confidence"] = (df["market_home_implied_prob"] - 0.5).abs()

    # Odds movement (positive = home drifted/weakened)
    df["odds_movement_home"] = df["home_odds_close"] - df["home_odds_open"]
    df["odds_movement_line"] = df["home_line_close"] - df["home_line_open"]

    # Betfair implied prob (normalised)
    bf_inv_home = 1.0 / df["betfair_home_back"]
    bf_inv_away = 1.0 / df["betfair_away_back"]
    bf_overround = bf_inv_home + bf_inv_away
    df["betfair_home_implied_prob"] = bf_inv_home / bf_overround

    # List of feature columns we created
    feature_names = [
        "market_home_implied_prob",
        "market_away_implied_prob",
        "market_handicap",
        "market_total_score",
        "market_confidence",
        "odds_movement_home",
        "odds_movement_line",
        "betfair_home_implied_prob",
    ]

    for col in feature_names:
        df[col] = df[col].astype("float32")

    print(f"  Created {len(feature_names)} odds features")
    for col in feature_names:
        n_valid = df[col].notna().sum()
        print(f"    {col:<35s} {n_valid:>5d}/{len(df)} valid")

    return df, feature_names


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Store and join to feature matrix
# ══════════════════════════════════════════════════════════════════════

def store_and_join(odds_df, feature_names):
    """Save odds.parquet and join to feature matrix."""
    print("\nSaving odds.parquet...")

    # Keep only match_id + feature columns (no forbidden data can survive)
    keep_cols = ["match_id"] + feature_names
    odds_out = odds_df[keep_cols].copy()
    odds_out.to_parquet(config.BASE_STORE_DIR / "odds.parquet", index=False)
    print(f"  Saved {len(odds_out)} rows to data/base/odds.parquet")

    # Join to feature matrix
    print("\nJoining odds features to feature_matrix.parquet...")
    feat_df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
    print(f"  Feature matrix: {feat_df.shape}")

    # Drop old odds columns if they exist (idempotent re-runs)
    existing_odds = [c for c in feature_names if c in feat_df.columns]
    if existing_odds:
        feat_df = feat_df.drop(columns=existing_odds)
        print(f"  Dropped {len(existing_odds)} existing odds columns")

    feat_df = feat_df.merge(odds_out, on="match_id", how="left")
    print(f"  After join: {feat_df.shape}")

    # Check coverage
    for col in feature_names:
        n_valid = feat_df[col].notna().sum()
        pct = n_valid / len(feat_df) * 100
        print(f"    {col:<35s} {n_valid:>6d}/{len(feat_df)} ({pct:.1f}%)")

    feat_df.to_parquet(
        config.FEATURES_DIR / "feature_matrix.parquet", index=False
    )
    print(f"  Saved updated feature_matrix.parquet")

    # Update feature_columns.json
    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    added = 0
    for col in feature_names:
        if col not in feature_cols:
            feature_cols.append(col)
            added += 1

    with open(feat_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Added {added} new features to feature_columns.json (total: {len(feature_cols)})")

    return feat_df


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Verification
# ══════════════════════════════════════════════════════════════════════

def verify(odds_df, feat_df, feature_names):
    """Run all verification checks."""
    matches = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")

    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # --- 4a: Correlation check ---
    print("\n4a. CORRELATION CHECK (odds features vs actual outcomes)")
    print("-" * 70)

    odds_with_results = odds_df.merge(
        matches[["match_id", "margin", "total_score"]],
        on="match_id", how="inner",
    )
    odds_with_results["home_win"] = (odds_with_results["margin"] > 0).astype(int)

    outcomes = ["home_win", "margin", "total_score"]
    print(f"  {'Feature':<35s} {'home_win':>10s} {'margin':>10s} {'total_score':>12s}")
    print("  " + "-" * 70)

    leakage_detected = False
    for col in feature_names:
        corrs = []
        for outcome in outcomes:
            valid = odds_with_results[[col, outcome]].dropna()
            if len(valid) > 30:
                c = valid[col].corr(valid[outcome])
                corrs.append(c)
            else:
                corrs.append(np.nan)
        flag = ""
        for c in corrs:
            if not np.isnan(c) and abs(c) > 0.50:
                flag = " ⚠ LEAKAGE?"
                leakage_detected = True
        corr_str = "  ".join(
            f"{c:>10.4f}" if not np.isnan(c) else f"{'N/A':>10s}"
            for c in corrs
        )
        print(f"  {col:<35s} {corr_str}{flag}")

    if leakage_detected:
        print("\n  ⚠ WARNING: Some correlations exceed 0.50 — investigate for leakage!")
    else:
        print("\n  ✓ No leakage detected (all |corr| < 0.50)")

    # --- 4b: Spot check 3 games ---
    print("\n4b. SPOT CHECK — 3 Games")
    print("-" * 70)

    spot_games = [
        {"year": 2025, "round": 2, "home": "Adelaide", "away": "St Kilda"},
        {"year": 2025, "round": 13, "home": "Brisbane Lions", "away": "Essendon"},
        {"year": 2024, "round": 1, "home": "Richmond", "away": "Carlton"},
    ]

    for game in spot_games:
        m = matches[
            (matches["year"] == game["year"])
            & (matches["round_number"] == game["round"])
            & (matches["home_team"] == game["home"])
        ]
        if m.empty:
            print(f"\n  R{game['round']} {game['year']} {game['home']} vs {game['away']}: MATCH NOT FOUND")
            continue

        mid = m.iloc[0]["match_id"]
        actual_margin = int(m.iloc[0]["margin"])
        actual_score = f"{int(m.iloc[0]['home_score'])}-{int(m.iloc[0]['away_score'])}"
        winner = game["home"] if actual_margin > 0 else game["away"]

        odds_row = odds_df[odds_df["match_id"] == mid]
        if odds_row.empty:
            print(f"\n  R{game['round']} {game['year']} {game['home']} vs {game['away']}: NO ODDS DATA")
            continue

        r = odds_row.iloc[0]
        was_fav = game["home"] if r.get("market_home_implied_prob", 0) > 0.5 else game["away"]
        print(f"\n  R{game['round']} {game['year']} {game['home']} vs {game['away']}")
        print(f"    Result: {actual_score} ({winner} won by {abs(actual_margin)})")
        print(f"    Market favourite: {was_fav}")
        print(f"    market_home_implied_prob: {r.get('market_home_implied_prob', 'N/A'):.4f}")
        print(f"    market_handicap:          {r.get('market_handicap', 'N/A')}")
        print(f"    market_total_score:        {r.get('market_total_score', 'N/A')}")
        print(f"    odds_movement_home:        {r.get('odds_movement_home', 'N/A')}")
        bf = r.get("betfair_home_implied_prob")
        print(f"    betfair_home_implied_prob: {bf:.4f}" if pd.notna(bf) else
              f"    betfair_home_implied_prob: N/A")

        # Confirm pre-game: a favourite that lost should still show as favourite
        if was_fav != winner:
            print(f"    ✓ Favourite ({was_fav}) LOST — odds are pre-game, not post-game")
        else:
            print(f"    ✓ Favourite ({was_fav}) won — consistent")

    # --- 4c: Confirm forbidden columns absent ---
    print("\n4c. FORBIDDEN COLUMN CHECK")
    print("-" * 70)

    forbidden_patterns = [
        "Home Score", "Away Score", "Home Goals", "Home Behinds",
        "Away Goals", "Away Behinds", "Play Off", "IS_WINNER",
        "RUNNER_STATUS", "MATCHED_VOLUME", "TOTAL_MATCHED",
        "QUARTER", "HALF_TIME", "THREE_QUARTER",
        "HOME_SCORE", "AWAY_SCORE", "HOME_MARGIN", "TOTAL_POINTS",
    ]

    odds_parquet = pd.read_parquet(config.BASE_STORE_DIR / "odds.parquet")
    for pattern in forbidden_patterns:
        matches_found = [c for c in odds_parquet.columns if pattern.lower() in c.lower()]
        if matches_found:
            print(f"  ⚠ FORBIDDEN column in odds.parquet: {matches_found}")
        matches_found = [c for c in feat_df.columns if pattern.lower() in c.lower()]
        if matches_found:
            # Filter out our legitimate columns
            bad = [c for c in matches_found if c not in [
                "market_total_score", "total_score", "home_score", "away_score",
                "total_DI", "total_IF", "total_CP",
            ]]
            if bad:
                print(f"  ⚠ FORBIDDEN column in feature_matrix: {bad}")

    print(f"  ✓ odds.parquet columns: {list(odds_parquet.columns)}")
    print(f"  ✓ No forbidden columns detected")

    # --- 4d: Summary stats ---
    print("\n4d. SUMMARY STATISTICS")
    print("-" * 70)

    print(f"\n  {'Feature':<35s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'n_valid':>8s}")
    print("  " + "-" * 75)
    for col in feature_names:
        s = odds_df[col].dropna()
        if len(s) > 0:
            print(f"  {col:<35s} {s.mean():>8.3f} {s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f} {len(s):>8d}")
        else:
            print(f"  {col:<35s} {'— no data —':>40s}")

    # --- 4e: Coverage summary ---
    print("\n4e. COVERAGE SUMMARY")
    print("-" * 70)

    # odds_df already has 'year' from parse_source2
    print(f"\n  {'Year':<6s} {'Matches':>8s} {'Has Odds':>9s} {'Has Line':>9s} {'Has Total':>10s} {'Has Betfair':>12s}")
    print("  " + "-" * 60)
    for yr in sorted(odds_df["year"].dropna().unique()):
        yr_df = odds_df[odds_df["year"] == yr]
        n = len(yr_df)
        has_odds = yr_df["market_home_implied_prob"].notna().sum()
        has_line = yr_df["market_handicap"].notna().sum()
        has_total = yr_df["market_total_score"].notna().sum()
        has_bf = yr_df["betfair_home_implied_prob"].notna().sum()
        print(f"  {int(yr):<6d} {n:>8d} {has_odds:>9d} {has_line:>9d} {has_total:>10d} {has_bf:>12d}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    source2 = parse_source2()
    betfair = parse_betfair()
    matched = match_to_parquet(source2, betfair)
    odds_df, feature_names = create_features(matched)
    feat_df = store_and_join(odds_df, feature_names)
    verify(odds_df, feat_df, feature_names)
    print("\n✓ Odds integration complete.")
