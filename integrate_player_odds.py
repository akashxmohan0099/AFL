"""
Integrate Betfair player-level market odds into the AFL prediction pipeline.

Sources:
  AFL_YYYY_All_Markets.csv (2021-2025) — Player Disposals, First Goalscorer,
  2 Goals Or More, 3 Goals Or More markets.

Output: data/base/player_odds.parquet, updated feature_matrix.parquet,
        updated feature_columns.json

ALLOWED columns: BEST_BACK_PRICE_60_MIN_PRIOR, BEST_LAY_PRICE_60_MIN_PRIOR only.
FORBIDDEN: IS_WINNER, RUNNER_STATUS, HOME_SCORE, AWAY_SCORE, HOME_MARGIN,
           TOTAL_POINTS, all MATCHED_VOLUME_*, TOTAL_MATCHED_VOLUME.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")

ODDS_DIR = Path("/Users/akash/Desktop/AFL/afl betting odds")

# ══════════════════════════════════════════════════════════════════════
# Team name mapping — normalise Betfair names to matches.parquet names
# ══════════════════════════════════════════════════════════════════════

TEAM_MAP = {
    "Brisbane": "Brisbane Lions",
    "GWS Giants": "Greater Western Sydney",
    "GWS": "Greater Western Sydney",
}


def normalise_team(name):
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


# ══════════════════════════════════════════════════════════════════════
# Player name matching: Betfair → pipeline
# ══════════════════════════════════════════════════════════════════════

# Betfair uses "First Last" (e.g. "Patrick Cripps")
# Pipeline uses "last, first" (e.g. "cripps, patrick")

# Hyphenated name variants in Betfair data — normalize both forms
_HYPHEN_NAMES = {
    "neal bullen": "neal-bullen",
    "horne francis": "horne-francis",
    "byrne jones": "byrne-jones",
    "coleman jones": "coleman-jones",
    "collier dawkins": "collier-dawkins",
    "davies uniacke": "davies-uniacke",
    "day wicks": "day-wicks",
    "eggmolesse smith": "eggmolesse-smith",
    "ellis yolmen": "ellis-yolmen",
    "grainger barras": "grainger-barras",
    "horlin smith": "horlin-smith",
    "hoskin elliott": "hoskin-elliott",
    "mcdonald tipungwuti": "mcdonald-tipungwuti",
    "moniz wakefield": "moniz-wakefield",
    "petrevski seton": "petrevski-seton",
    "powell pepper": "powell-pepper",
    "ugle hagan": "ugle-hagan",
    "vickers willis": "vickers-willis",
    "wanganeen milera": "wanganeen-milera",
    "zerk thatcher": "zerk-thatcher",
}


# Multi-part surname prefixes: "Jordan De Goey" → "De Goey, Jordan"
_MULTI_PART_PREFIXES = {"de", "van", "mc", "ah"}

# Explicit Betfair-to-pipeline name corrections (typos, abbreviations)
_NAME_CORRECTIONS = {
    "ashcroft, will": "Ashcroft, Will",  # typo "WIll" → "Will"
    "berrry, jarrod": "Berry, Jarrod",
    "bryne-jones, darcy": "Byrne-Jones, Darcy",
    "duursma, xaiver": "Duursma, Xavier",
    "mcclugagge, hugh": "McCluggage, Hugh",
    "fritsch, bailey": "Fritsch, Bayley",
    "raynor, cam": "Rayner, Cam",
    "rosas, malcom": "Rosas, Malcolm",
    "willams, bailey": "Williams, Bailey",
    "elliot, jamie": "Elliott, Jamie",
    "simpkin, jye": "Simpkin, Jy",
    "dambrosio, massimo": "D'Ambrosio, Massimo",
}


def betfair_name_to_pipeline(name):
    """Convert Betfair 'First Last' → pipeline 'Last, First' format.

    Pipeline stores names in original case: "Cripps, Patrick", "Neal-Bullen, Alex".
    Handles hyphenated names, multi-part surnames (De Goey, Ah Chee, Van Rooyen),
    Jnr/Jr suffixes, and parenthesized team disambiguators.

    Examples:
        "Patrick Cripps"       → "Cripps, Patrick"
        "Jason Horne Francis"  → "Horne-Francis, Jason"
        "Jason Horne-Francis"  → "Horne-Francis, Jason"
        "Alex Neal Bullen"     → "Neal-Bullen, Alex"
        "Jordan De Goey"       → "De Goey, Jordan"
        "Brendon Ah Chee"      → "Ah Chee, Brendon"
        "Jacob Van Rooyen"     → "Van Rooyen, Jacob"
    """
    name = str(name).strip()
    name = " ".join(name.split())  # normalise whitespace

    # Strip parenthesized team disambiguators: "(ESS)" / "(WC)" etc.
    name = re.sub(r"\s*\([A-Za-z]+\)\s*,?\s*", " ", name).strip()
    # Also handle non-breaking space variants
    name = name.replace("\xa0", " ").strip()
    name = " ".join(name.split())

    # Strip "Jnr" / "Jr" suffixes
    name = re.sub(r"\s+(?:Jnr|Jr|Snr|Sr)\.?\s*$", "", name, flags=re.IGNORECASE).strip()

    parts = name.split()
    if len(parts) < 2:
        return name

    # Try to match known hyphenated surnames (check 2-part and 3-part combos)
    lower_parts = [p.lower() for p in parts]
    for n_surname_parts in [3, 2]:
        if len(parts) > n_surname_parts:
            candidate = " ".join(lower_parts[-n_surname_parts:])
            if candidate in _HYPHEN_NAMES:
                first = " ".join(parts[:-n_surname_parts])
                last = "-".join(w.capitalize() for w in _HYPHEN_NAMES[candidate].split("-"))
                return f"{last}, {first}"

    # If the last part already has a hyphen (e.g. "Jason Horne-Francis"), keep it
    if "-" in parts[-1]:
        first = " ".join(parts[:-1])
        last = parts[-1]
        return f"{last}, {first}"

    # Check if penultimate + last form a known hyphenated name
    if len(parts) >= 3:
        two_part = f"{lower_parts[-2]} {lower_parts[-1]}"
        if two_part in _HYPHEN_NAMES:
            first = " ".join(parts[:-2])
            last = "-".join(w.capitalize() for w in _HYPHEN_NAMES[two_part].split("-"))
            return f"{last}, {first}"

    # Handle multi-part surname prefixes: "De", "Van", "Ah", "Mc"
    # e.g. "Jordan De Goey" → first="Jordan", last="De Goey"
    if len(parts) >= 3:
        for prefix_idx in range(1, len(parts) - 1):
            if lower_parts[prefix_idx] in _MULTI_PART_PREFIXES:
                first = " ".join(parts[:prefix_idx])
                last = " ".join(parts[prefix_idx:])
                result = f"{last}, {first}"
                # Check for explicit corrections
                corrected = _NAME_CORRECTIONS.get(result.lower())
                if corrected:
                    return corrected
                return result

    # Standard case: "First Last" → "Last, First"
    first = " ".join(parts[:-1])
    last = parts[-1]
    result = f"{last}, {first}"

    # Apply explicit corrections
    corrected = _NAME_CORRECTIONS.get(result.lower())
    if corrected:
        return corrected

    return result


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Parse All Markets CSVs
# ══════════════════════════════════════════════════════════════════════

def parse_player_disposals(raw):
    """Extract Player Disposals markets: line and over/under prices."""
    mask = raw["MARKET_NAME"].str.startswith("Player Disposals", na=False)
    disp = raw[mask].copy()
    if disp.empty:
        return pd.DataFrame()

    # Extract player name from MARKET_NAME: "Player Disposals - Patrick Cripps"
    # Some have line in name: "Player Disposals - Andrew Brayshaw 27.5" — strip it
    disp["betfair_player"] = disp["MARKET_NAME"].str.replace(
        r"^Player Disposals\s*-\s*", "", regex=True
    ).str.replace(r"\s+\d+\.5\s*$", "", regex=True).str.strip()

    # Extract disposal line from RUNNER_NAME: "Over 24.5 Disposals" → 24.5
    disp["disposal_line"] = disp["RUNNER_NAME"].str.extract(
        r"(\d+\.5)", expand=False
    ).astype(float)

    # Determine over/under
    disp["is_over"] = disp["RUNNER_NAME"].str.contains("Over", case=False, na=False)

    # ALLOWED price column only
    disp["price_60"] = pd.to_numeric(
        disp["BEST_BACK_PRICE_60_MIN_PRIOR"], errors="coerce"
    )

    # Pivot: one row per (match context, player) with over and under prices
    over = disp[disp["is_over"]].copy()
    under = disp[~disp["is_over"]].copy()

    key_cols = ["EVENT_DATE", "HOME_TEAM", "AWAY_TEAM", "betfair_player"]

    over_agg = over.groupby(key_cols, observed=True).agg(
        disposal_line=("disposal_line", "first"),
        over_price=("price_60", "first"),
    ).reset_index()

    under_agg = under.groupby(key_cols, observed=True).agg(
        under_price=("price_60", "first"),
    ).reset_index()

    result = over_agg.merge(under_agg, on=key_cols, how="left")
    print(f"  Player Disposals: {len(result)} player-match rows")
    return result


def parse_first_goalscorer(raw):
    """Extract First Goalscorer market: player and FGS price."""
    mask = raw["MARKET_NAME"] == "First Goalscorer"
    fgs = raw[mask].copy()
    if fgs.empty:
        return pd.DataFrame()

    # Exclude "Any Other Player" and "No Goalscorer" runners
    fgs = fgs[~fgs["RUNNER_NAME"].isin(["Any Other Player", "No Goalscorer"])].copy()

    fgs["betfair_player"] = fgs["RUNNER_NAME"].str.strip()
    fgs["fgs_price"] = pd.to_numeric(
        fgs["BEST_BACK_PRICE_60_MIN_PRIOR"], errors="coerce"
    )

    key_cols = ["EVENT_DATE", "HOME_TEAM", "AWAY_TEAM", "betfair_player"]
    result = fgs.groupby(key_cols, observed=True).agg(
        fgs_price=("fgs_price", "first"),
    ).reset_index()
    print(f"  First Goalscorer: {len(result)} player-match rows")
    return result


def parse_goals_market(raw, n_goals):
    """Extract N Goals Or More market (2 or 3)."""
    market_name = f"{n_goals} Goals Or More"
    mask = raw["MARKET_NAME"] == market_name
    goals = raw[mask].copy()
    if goals.empty:
        return pd.DataFrame()

    goals["betfair_player"] = goals["RUNNER_NAME"].str.strip()
    price_col = f"goals{n_goals}_price"
    goals[price_col] = pd.to_numeric(
        goals["BEST_BACK_PRICE_60_MIN_PRIOR"], errors="coerce"
    )

    key_cols = ["EVENT_DATE", "HOME_TEAM", "AWAY_TEAM", "betfair_player"]
    result = goals.groupby(key_cols, observed=True).agg(
        **{price_col: (price_col, "first")},
    ).reset_index()
    print(f"  {market_name}: {len(result)} player-match rows")
    return result


def parse_all_markets():
    """Parse all Betfair All Markets CSVs, extract player-level markets."""
    print("Parsing Betfair player markets...")

    frames = []
    for csv_path in sorted(ODDS_DIR.glob("AFL_*_All_Markets.csv")):
        df = pd.read_csv(csv_path, low_memory=False)
        frames.append(df)
        year = csv_path.stem.split("_")[1]
        print(f"  {csv_path.name}: {len(df)} rows ({year})")

    if not frames:
        print("  No All Markets files found")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded: {len(raw)}")

    disp = parse_player_disposals(raw)
    fgs = parse_first_goalscorer(raw)
    g2 = parse_goals_market(raw, 2)
    g3 = parse_goals_market(raw, 3)

    return disp, fgs, g2, g3


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Match to pipeline data
# ══════════════════════════════════════════════════════════════════════

def match_to_pipeline(disp, fgs, g2, g3):
    """Match Betfair player markets to pipeline player_games data."""
    print("\nMatching to pipeline data...")

    # Load matches.parquet for match_id lookup
    matches = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")
    matches["date"] = pd.to_datetime(matches["date"]).dt.normalize()

    # Load player_games for player name matching
    pg = pd.read_parquet(config.BASE_STORE_DIR / "player_games.parquet")
    pg["date"] = pd.to_datetime(pg["date"]).dt.normalize()

    # Build match lookup: (date, home_team, away_team) → match_id
    match_lookup = matches[["match_id", "date", "home_team", "away_team"]].copy()

    # Build player lookup: (match_id, player) for validation
    player_lookup = pg[["match_id", "player", "team"]].copy()
    player_set = set(pg["player"].unique())

    def _prepare_market_df(market_df):
        """Normalize date/team/player in a market DataFrame."""
        if market_df.empty:
            return market_df
        mdf = market_df.copy()
        mdf["date"] = pd.to_datetime(mdf["EVENT_DATE"], dayfirst=True).dt.normalize()
        mdf["home_team"] = mdf["HOME_TEAM"].apply(normalise_team)
        mdf["away_team"] = mdf["AWAY_TEAM"].apply(normalise_team)
        mdf["player"] = mdf["betfair_player"].apply(betfair_name_to_pipeline)
        # Join to get match_id
        mdf = mdf.merge(
            match_lookup, on=["date", "home_team", "away_team"], how="inner"
        )
        return mdf

    # Process each market type
    disp_matched = _prepare_market_df(disp)
    fgs_matched = _prepare_market_df(fgs)
    g2_matched = _prepare_market_df(g2)
    g3_matched = _prepare_market_df(g3)

    # Report match rates
    for name, orig, matched in [
        ("Disposals", disp, disp_matched),
        ("FGS", fgs, fgs_matched),
        ("2 Goals", g2, g2_matched),
        ("3 Goals", g3, g3_matched),
    ]:
        if not orig.empty:
            rate = len(matched) / len(orig) * 100
            print(f"  {name}: {len(matched)}/{len(orig)} rows matched to matches ({rate:.1f}%)")

    # Check player name match rate
    if not disp_matched.empty:
        matched_players = disp_matched["player"].isin(player_set)
        match_rate = matched_players.mean() * 100
        n_unmatched = (~matched_players).sum()
        print(f"  Disposal player name match rate: {match_rate:.1f}% ({n_unmatched} unmatched)")
        if n_unmatched > 0 and n_unmatched <= 20:
            unmatched = disp_matched.loc[~matched_players, "player"].unique()
            for p in sorted(unmatched)[:10]:
                print(f"    Unmatched: '{p}'")

    # Combine into single DataFrame keyed by (match_id, player)
    # Start with disposal features
    if not disp_matched.empty:
        combined = disp_matched[["match_id", "player",
                                  "disposal_line", "over_price", "under_price"]].copy()
    else:
        combined = pd.DataFrame(columns=["match_id", "player"])

    # Merge FGS
    if not fgs_matched.empty:
        fgs_cols = fgs_matched[["match_id", "player", "fgs_price"]].copy()
        if combined.empty:
            combined = fgs_cols
        else:
            combined = combined.merge(fgs_cols, on=["match_id", "player"], how="outer")

    # Merge 2 Goals
    if not g2_matched.empty:
        g2_cols = g2_matched[["match_id", "player", "goals2_price"]].copy()
        if combined.empty:
            combined = g2_cols
        else:
            combined = combined.merge(g2_cols, on=["match_id", "player"], how="outer")

    # Merge 3 Goals
    if not g3_matched.empty:
        g3_cols = g3_matched[["match_id", "player", "goals3_price"]].copy()
        if combined.empty:
            combined = g3_cols
        else:
            combined = combined.merge(g3_cols, on=["match_id", "player"], how="outer")

    if combined.empty:
        print("  WARNING: No player market data matched")
        return combined

    # Deduplicate: keep first row per (match_id, player)
    combined = combined.drop_duplicates(subset=["match_id", "player"], keep="first")

    print(f"\n  Combined: {len(combined)} unique (match_id, player) rows")
    print(f"  Columns: {list(combined.columns)}")

    return combined


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Create features
# ══════════════════════════════════════════════════════════════════════

PLAYER_ODDS_FEATURES = [
    "market_disposal_line",
    "market_disposal_over_price",
    "market_disposal_implied_over",
    "market_fgs_price",
    "market_fgs_implied_prob",
    "market_2goals_price",
    "market_2goals_implied_prob",
    "market_3goals_price",
]


def create_features(combined_df):
    """Derive player-level market features from matched odds data."""
    print("\nCreating player market features...")
    df = combined_df.copy()

    # Disposal features
    df["market_disposal_line"] = df.get("disposal_line", pd.Series(dtype=float))
    df["market_disposal_over_price"] = df.get("over_price", pd.Series(dtype=float))

    # Implied probability for disposal over (normalise overround)
    if "over_price" in df.columns and "under_price" in df.columns:
        inv_over = 1.0 / df["over_price"]
        inv_under = 1.0 / df["under_price"]
        overround = inv_over + inv_under
        df["market_disposal_implied_over"] = inv_over / overround
    else:
        df["market_disposal_implied_over"] = np.nan

    # First Goalscorer features
    df["market_fgs_price"] = df.get("fgs_price", pd.Series(dtype=float))
    # FGS is multi-runner so raw 1/price is the implied prob (no overround removal)
    df["market_fgs_implied_prob"] = 1.0 / df["market_fgs_price"]

    # 2 Goals Or More
    df["market_2goals_price"] = df.get("goals2_price", pd.Series(dtype=float))
    df["market_2goals_implied_prob"] = 1.0 / df["market_2goals_price"]

    # 3 Goals Or More
    df["market_3goals_price"] = df.get("goals3_price", pd.Series(dtype=float))

    # Cast to float32
    for col in PLAYER_ODDS_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # Keep only match_id, player, and feature columns
    keep_cols = ["match_id", "player"] + PLAYER_ODDS_FEATURES
    out = df[[c for c in keep_cols if c in df.columns]].copy()

    print(f"  Created {len(PLAYER_ODDS_FEATURES)} player market features")
    for col in PLAYER_ODDS_FEATURES:
        if col in out.columns:
            n_valid = out[col].notna().sum()
            print(f"    {col:<35s} {n_valid:>6d}/{len(out)} valid")

    return out


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Store and join to feature matrix
# ══════════════════════════════════════════════════════════════════════

def store_and_join(player_odds_df):
    """Save player_odds.parquet and join to feature matrix."""
    print("\nSaving player_odds.parquet...")

    player_odds_df.to_parquet(
        config.BASE_STORE_DIR / "player_odds.parquet", index=False
    )
    print(f"  Saved {len(player_odds_df)} rows to data/base/player_odds.parquet")

    # Join to feature matrix
    print("\nJoining player market features to feature_matrix.parquet...")
    feat_path = config.FEATURES_DIR / "feature_matrix.parquet"
    feat_df = pd.read_parquet(feat_path)
    print(f"  Feature matrix: {feat_df.shape}")

    # Drop old player odds columns if they exist (idempotent re-runs)
    existing = [c for c in PLAYER_ODDS_FEATURES if c in feat_df.columns]
    if existing:
        feat_df = feat_df.drop(columns=existing)
        print(f"  Dropped {len(existing)} existing player odds columns")

    # Left join on (match_id, player)
    feat_df = feat_df.merge(player_odds_df, on=["match_id", "player"], how="left")
    print(f"  After join: {feat_df.shape}")

    # Coverage report
    for col in PLAYER_ODDS_FEATURES:
        if col in feat_df.columns:
            n_valid = feat_df[col].notna().sum()
            pct = n_valid / len(feat_df) * 100
            print(f"    {col:<35s} {n_valid:>6d}/{len(feat_df)} ({pct:.1f}%)")

    # Downcast to float32
    for col in PLAYER_ODDS_FEATURES:
        if col in feat_df.columns and feat_df[col].dtype == np.float64:
            feat_df[col] = feat_df[col].astype(np.float32)

    feat_df.to_parquet(feat_path, index=False)
    print(f"  Saved updated feature_matrix.parquet")

    # Update feature_columns.json
    feat_cols_path = config.FEATURES_DIR / "feature_columns.json"
    with open(feat_cols_path) as f:
        feature_cols = json.load(f)

    added = 0
    for col in PLAYER_ODDS_FEATURES:
        if col not in feature_cols:
            feature_cols.append(col)
            added += 1

    with open(feat_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Added {added} new features to feature_columns.json (total: {len(feature_cols)})")

    return feat_df


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Verification
# ══════════════════════════════════════════════════════════════════════

def verify(player_odds_df, feat_df):
    """Run verification checks on player market features."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # 5a: Forbidden column check
    print("\n5a. FORBIDDEN COLUMN CHECK")
    print("-" * 70)
    forbidden_patterns = [
        "IS_WINNER", "RUNNER_STATUS", "MATCHED_VOLUME", "TOTAL_MATCHED",
        "HOME_SCORE", "AWAY_SCORE", "HOME_MARGIN", "TOTAL_POINTS",
    ]
    odds_parquet = pd.read_parquet(config.BASE_STORE_DIR / "player_odds.parquet")
    clean = True
    for pattern in forbidden_patterns:
        bad = [c for c in odds_parquet.columns if pattern.lower() in c.lower()]
        if bad:
            print(f"  FORBIDDEN column in player_odds.parquet: {bad}")
            clean = False
    if clean:
        print(f"  player_odds.parquet columns: {list(odds_parquet.columns)}")
        print(f"  No forbidden columns detected")

    # 5b: Spot check — Patrick Cripps, Round 1, 2024
    print("\n5b. SPOT CHECK")
    print("-" * 70)
    matches = pd.read_parquet(config.BASE_STORE_DIR / "matches.parquet")

    # Find a Cripps match in 2024
    cripps_name = "Cripps, Patrick"
    cripps_rows = player_odds_df[player_odds_df["player"] == cripps_name]
    if not cripps_rows.empty:
        sample = cripps_rows.head(3)
        for _, r in sample.iterrows():
            mid = r["match_id"]
            m = matches[matches["match_id"] == mid]
            if not m.empty:
                m_row = m.iloc[0]
                print(f"  match_id={mid}: {m_row['home_team']} vs {m_row['away_team']} "
                      f"(Rd {m_row['round_number']}, {m_row['year']})")
            disp_line = r.get("market_disposal_line")
            over_price = r.get("market_disposal_over_price")
            fgs = r.get("market_fgs_price")
            print(f"    disposal_line={disp_line}, over_price={over_price}, fgs_price={fgs}")
    else:
        print(f"  No data found for '{cripps_name}'")

    # 5c: Name match rate
    print("\n5c. NAME MATCH SUMMARY")
    print("-" * 70)
    pg = pd.read_parquet(config.BASE_STORE_DIR / "player_games.parquet")
    pipeline_players = set(pg["player"].unique())
    odds_players = set(player_odds_df["player"].unique())
    matched = odds_players & pipeline_players
    unmatched = odds_players - pipeline_players
    rate = len(matched) / len(odds_players) * 100 if odds_players else 0
    print(f"  Betfair unique players: {len(odds_players)}")
    print(f"  Matched to pipeline:    {len(matched)} ({rate:.1f}%)")
    print(f"  Unmatched:              {len(unmatched)}")
    if unmatched and len(unmatched) <= 30:
        for p in sorted(unmatched):
            print(f"    '{p}'")

    # 5d: Coverage by year
    print("\n5d. COVERAGE BY YEAR")
    print("-" * 70)
    merged = player_odds_df.merge(
        matches[["match_id", "year"]], on="match_id", how="left"
    )
    print(f"  {'Year':<6s} {'Rows':>6s} {'HasDisp':>8s} {'HasFGS':>8s} "
          f"{'Has2G':>8s} {'Has3G':>8s}")
    for yr in sorted(merged["year"].dropna().unique()):
        yr_df = merged[merged["year"] == yr]
        n = len(yr_df)
        has_disp = yr_df["market_disposal_line"].notna().sum() if "market_disposal_line" in yr_df else 0
        has_fgs = yr_df["market_fgs_price"].notna().sum() if "market_fgs_price" in yr_df else 0
        has_2g = yr_df["market_2goals_price"].notna().sum() if "market_2goals_price" in yr_df else 0
        has_3g = yr_df["market_3goals_price"].notna().sum() if "market_3goals_price" in yr_df else 0
        print(f"  {int(yr):<6d} {n:>6d} {has_disp:>8d} {has_fgs:>8d} {has_2g:>8d} {has_3g:>8d}")

    # 5e: Summary statistics
    print("\n5e. SUMMARY STATISTICS")
    print("-" * 70)
    print(f"  {'Feature':<35s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'n_valid':>8s}")
    print("  " + "-" * 75)
    for col in PLAYER_ODDS_FEATURES:
        if col in player_odds_df.columns:
            s = player_odds_df[col].dropna()
            if len(s) > 0:
                print(f"  {col:<35s} {s.mean():>8.3f} {s.std():>8.3f} "
                      f"{s.min():>8.3f} {s.max():>8.3f} {len(s):>8d}")
            else:
                print(f"  {col:<35s} {'— no data —':>40s}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    disp, fgs, g2, g3 = parse_all_markets()
    combined = match_to_pipeline(disp, fgs, g2, g3)
    if combined.empty:
        print("\nNo data to process.")
    else:
        player_odds = create_features(combined)
        feat_df = store_and_join(player_odds)
        verify(player_odds, feat_df)
        print("\nPlayer market odds integration complete.")
