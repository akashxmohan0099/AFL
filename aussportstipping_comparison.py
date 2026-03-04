#!/usr/bin/env python3
"""
AusSportsTipping Comparison Analytics
=====================================
Derives all key metrics that AusSportsTipping offers, using our existing
pipeline data (team_matches, matches, odds parquets + EloSystem).

Usage:
    python aussportstipping_comparison.py [--year 2025] [--last-n 6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import VENUE_NAME_MAP, TEAM_NAME_MAP
from model import EloSystem


# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("data/base")
TEAM_MATCHES = BASE / "team_matches.parquet"
MATCHES = BASE / "matches.parquet"
ODDS = BASE / "odds.parquet"
OUTPUT_JSON = Path("data/experiments/aussportstipping_comparison.json")


def load_data():
    """Load all required parquet files."""
    tm = pd.read_parquet(TEAM_MATCHES)
    matches = pd.read_parquet(MATCHES)
    odds = None
    if ODDS.exists():
        odds = pd.read_parquet(ODDS)
    return tm, matches, odds


def compute_elo(tm: pd.DataFrame) -> pd.DataFrame:
    """Run EloSystem on team_matches, return elo df."""
    elo = EloSystem()
    elo_df = elo.compute_all(tm)
    return elo_df


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Elo Ratings (Team Rankings)
# ═══════════════════════════════════════════════════════════════════════════

def elo_ratings(tm: pd.DataFrame, elo_df: pd.DataFrame, year: int) -> dict:
    """Current Elo rankings for all teams."""
    elo_sys = EloSystem()
    # Get the latest elo_pre for each team in the target year
    df = elo_df.merge(tm[["match_id", "team", "date", "year"]], on=["match_id", "team"])
    df = df[df["year"] <= year].sort_values("date")
    latest = df.groupby("team", observed=True).last().reset_index()
    latest = latest.sort_values("elo_pre", ascending=False)

    print("\n" + "=" * 60)
    print(" 1. ELO RATINGS — Team Rankings")
    print("=" * 60)
    print(f"{'Rank':<5} {'Team':<25} {'Elo':>8} {'Win Exp vs Avg':>14}")
    print("-" * 55)
    results = []
    for i, (_, row) in enumerate(latest.iterrows(), 1):
        wp = elo_sys.expected_win_prob(row["elo_pre"], 1500, is_home=False)
        print(f"{i:<5} {row['team']:<25} {row['elo_pre']:>8.1f} {wp:>13.1%}")
        results.append({
            "rank": i, "team": row["team"],
            "elo": round(float(row["elo_pre"]), 1),
            "win_prob_vs_avg": round(float(wp), 4),
        })
    return {"elo_rankings": results}


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Line/Spread Coverage (ATS Record)
# ═══════════════════════════════════════════════════════════════════════════

def ats_record(tm: pd.DataFrame, matches: pd.DataFrame, odds: pd.DataFrame,
               year: int) -> dict:
    """Against-the-spread record per team."""
    if odds is None or "market_handicap" not in odds.columns:
        print("\n[SKIPPED] ATS Record — no handicap data in odds.parquet")
        return {"ats_record": "no_data"}

    # market_handicap is home line; positive = home is underdog
    home = tm[tm["is_home"] & (tm["year"] == year)][["match_id", "team", "margin"]].copy()
    away = tm[~tm["is_home"] & (tm["year"] == year)][["match_id", "team", "margin"]].copy()

    # Merge odds
    home = home.merge(odds[["match_id", "market_handicap"]], on="match_id", how="inner")
    away = away.merge(odds[["match_id", "market_handicap"]], on="match_id", how="inner")

    # Home covers if margin > -handicap (handicap is from home perspective)
    home["covers"] = home["margin"] + home["market_handicap"] > 0
    home["push"] = (home["margin"] + home["market_handicap"]) == 0
    # Away covers if -margin > handicap → margin < -handicap → margin + handicap < 0
    away["covers"] = away["margin"] - away["market_handicap"] > 0  # away margin = -home margin adjusted
    # Actually: away margin is stored as (away_score - opp_score) so it's already from away perspective
    # For away: they cover if their margin > line (away line = -home handicap)
    away["covers"] = away["margin"] > away["market_handicap"]
    away["push"] = away["margin"] == away["market_handicap"]

    combined = pd.concat([home[["team", "covers", "push"]], away[["team", "covers", "push"]]])
    stats = combined.groupby("team", observed=True).agg(
        games=("covers", "count"),
        covers=("covers", "sum"),
        pushes=("push", "sum"),
    ).reset_index()
    stats["fails"] = stats["games"] - stats["covers"] - stats["pushes"]
    stats["cover_pct"] = stats["covers"] / stats["games"]
    stats = stats.sort_values("cover_pct", ascending=False)

    print("\n" + "=" * 60)
    print(f" 2. ATS RECORD — Against the Spread ({year})")
    print("=" * 60)
    print(f"{'Team':<25} {'W':>4} {'L':>4} {'P':>4} {'Cover%':>8}")
    print("-" * 48)
    results = []
    for _, row in stats.iterrows():
        print(f"{row['team']:<25} {int(row['covers']):>4} {int(row['fails']):>4} "
              f"{int(row['pushes']):>4} {row['cover_pct']:>7.1%}")
        results.append({
            "team": row["team"], "covers": int(row["covers"]),
            "fails": int(row["fails"]), "pushes": int(row["pushes"]),
            "cover_pct": round(float(row["cover_pct"]), 4),
        })
    return {"ats_record": results}


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Predictability Index
# ═══════════════════════════════════════════════════════════════════════════

def predictability_index(tm: pd.DataFrame, elo_df: pd.DataFrame,
                         year: int) -> dict:
    """Score volatility relative to Elo expectation per team."""
    df = elo_df.merge(tm[["match_id", "team", "margin", "year"]], on=["match_id", "team"])
    df = df[df["year"] == year]

    # Expected margin ≈ elo_diff / 33 (rough Elo-to-margin conversion)
    df["expected_margin"] = df["elo_diff"] / 33.0
    df["residual"] = df["margin"] - df["expected_margin"]

    stats = df.groupby("team", observed=True).agg(
        games=("residual", "count"),
        residual_std=("residual", "std"),
        residual_mean=("residual", "mean"),
        avg_margin=("margin", "mean"),
    ).reset_index()
    stats["predictability"] = 1.0 / (1.0 + stats["residual_std"] / 30.0)
    stats = stats.sort_values("predictability", ascending=False)

    print("\n" + "=" * 60)
    print(f" 3. PREDICTABILITY INDEX ({year})")
    print("=" * 60)
    print(f"{'Team':<25} {'Index':>8} {'Resid Std':>10} {'Avg Margin':>11}")
    print("-" * 57)
    results = []
    for _, row in stats.iterrows():
        print(f"{row['team']:<25} {row['predictability']:>8.3f} "
              f"{row['residual_std']:>10.1f} {row['avg_margin']:>+11.1f}")
        results.append({
            "team": row["team"],
            "predictability": round(float(row["predictability"]), 4),
            "residual_std": round(float(row["residual_std"]), 2),
            "avg_margin": round(float(row["avg_margin"]), 2),
        })
    return {"predictability_index": results}


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Score Statistics
# ═══════════════════════════════════════════════════════════════════════════

def score_statistics(tm: pd.DataFrame, year: int) -> dict:
    """Points for/against/differential per team."""
    df = tm[tm["year"] == year]

    stats = df.groupby("team", observed=True).agg(
        games=("score", "count"),
        pts_for=("score", "mean"),
        pts_against=("opp_score", "mean"),
        pts_for_total=("score", "sum"),
        pts_against_total=("opp_score", "sum"),
        margin_avg=("margin", "mean"),
        margin_std=("margin", "std"),
        wins=("result", lambda x: (x == "W").sum()),
    ).reset_index()
    stats["win_pct"] = stats["wins"] / stats["games"]
    stats["pts_diff_total"] = stats["pts_for_total"] - stats["pts_against_total"]
    stats = stats.sort_values("pts_diff_total", ascending=False)

    print("\n" + "=" * 60)
    print(f" 4. SCORE STATISTICS ({year})")
    print("=" * 60)
    print(f"{'Team':<25} {'PF':>6} {'PA':>6} {'Diff':>6} {'W%':>6} {'Games':>6}")
    print("-" * 58)
    results = []
    for _, row in stats.iterrows():
        print(f"{row['team']:<25} {row['pts_for']:>6.1f} {row['pts_against']:>6.1f} "
              f"{row['margin_avg']:>+6.1f} {row['win_pct']:>5.1%} {int(row['games']):>6}")
        results.append({
            "team": row["team"],
            "pts_for_avg": round(float(row["pts_for"]), 2),
            "pts_against_avg": round(float(row["pts_against"]), 2),
            "margin_avg": round(float(row["margin_avg"]), 2),
            "win_pct": round(float(row["win_pct"]), 4),
            "games": int(row["games"]),
        })
    return {"score_statistics": results}


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Stadium Analysis
# ═══════════════════════════════════════════════════════════════════════════

def stadium_analysis(tm: pd.DataFrame, year: int) -> dict:
    """Team performance by venue."""
    df = tm[(tm["year"] >= year - 2) & (tm["year"] <= year)]  # 3-year window

    stats = df.groupby(["team", "venue"], observed=True).agg(
        games=("margin", "count"),
        wins=("result", lambda x: (x == "W").sum()),
        avg_margin=("margin", "mean"),
        avg_score=("score", "mean"),
    ).reset_index()
    stats = stats[stats["games"] >= 3]  # Minimum sample
    stats["win_pct"] = stats["wins"] / stats["games"]
    stats = stats.sort_values(["team", "win_pct"], ascending=[True, False])

    print("\n" + "=" * 60)
    print(f" 5. STADIUM ANALYSIS ({year - 2}-{year}, min 3 games)")
    print("=" * 60)
    print(f"{'Team':<22} {'Venue':<22} {'W-L':>6} {'W%':>6} {'Margin':>8}")
    print("-" * 68)
    results = []
    for _, row in stats.iterrows():
        losses = int(row["games"] - row["wins"])
        print(f"{row['team']:<22} {str(row['venue']):<22} "
              f"{int(row['wins'])}-{losses:>2} {row['win_pct']:>5.1%} {row['avg_margin']:>+8.1f}")
        results.append({
            "team": row["team"], "venue": str(row["venue"]),
            "games": int(row["games"]), "wins": int(row["wins"]),
            "win_pct": round(float(row["win_pct"]), 4),
            "avg_margin": round(float(row["avg_margin"]), 2),
        })
    return {"stadium_analysis": results}


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Winning Margins
# ═══════════════════════════════════════════════════════════════════════════

def winning_margins(matches: pd.DataFrame, year: int) -> dict:
    """Distribution of winning margins by team and overall."""
    df = matches[matches["year"] == year].copy()
    df["abs_margin"] = df["margin"].abs()

    # Overall distribution
    bins = [0, 6, 12, 24, 48, 100, 999]
    labels = ["1-6 (nail-biter)", "7-12 (close)", "13-24 (comfortable)",
              "25-48 (dominant)", "49-100 (blowout)", "100+ (historic)"]
    df["margin_bucket"] = pd.cut(df["abs_margin"], bins=bins, labels=labels, right=True)

    dist = df["margin_bucket"].value_counts().sort_index()
    total = len(df)

    print("\n" + "=" * 60)
    print(f" 6. WINNING MARGINS ({year})")
    print("=" * 60)
    print(f"{'Bucket':<25} {'Count':>6} {'Pct':>8}")
    print("-" * 42)
    results_dist = []
    for bucket, count in dist.items():
        pct = count / total
        print(f"{str(bucket):<25} {count:>6} {pct:>7.1%}")
        results_dist.append({
            "bucket": str(bucket), "count": int(count),
            "pct": round(float(pct), 4),
        })

    # Per-team average winning margin (when winning)
    # Build team-level view
    tm_year = pd.DataFrame()
    winners = []
    for _, row in df.iterrows():
        if row["margin"] > 0:
            winners.append({"team": row["home_team"], "win_margin": row["margin"]})
        elif row["margin"] < 0:
            winners.append({"team": row["away_team"], "win_margin": -row["margin"]})
    if winners:
        wdf = pd.DataFrame(winners)
        team_margins = wdf.groupby("team", observed=True).agg(
            avg_win_margin=("win_margin", "mean"),
            max_win_margin=("win_margin", "max"),
            wins=("win_margin", "count"),
        ).sort_values("avg_win_margin", ascending=False).reset_index()

        print(f"\n{'Team':<25} {'Avg Win':>8} {'Max Win':>8} {'Wins':>6}")
        print("-" * 50)
        results_team = []
        for _, row in team_margins.iterrows():
            print(f"{row['team']:<25} {row['avg_win_margin']:>8.1f} "
                  f"{int(row['max_win_margin']):>8} {int(row['wins']):>6}")
            results_team.append({
                "team": row["team"],
                "avg_win_margin": round(float(row["avg_win_margin"]), 2),
                "max_win_margin": int(row["max_win_margin"]),
                "wins": int(row["wins"]),
            })
    else:
        results_team = []

    return {"winning_margins": {"distribution": results_dist, "by_team": results_team}}


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Home-Field Advantage
# ═══════════════════════════════════════════════════════════════════════════

def home_field_advantage(tm: pd.DataFrame, year: int) -> dict:
    """Home vs away performance splits + multi-year trend."""
    # Per-team home/away split for target year
    df = tm[tm["year"] == year]

    splits = []
    for team in sorted(df["team"].unique()):
        tdf = df[df["team"] == team]
        home = tdf[tdf["is_home"]]
        away = tdf[~tdf["is_home"]]
        splits.append({
            "team": team,
            "home_w": int((home["result"] == "W").sum()),
            "home_l": int((home["result"] == "L").sum()),
            "home_margin": float(home["margin"].mean()) if len(home) > 0 else 0.0,
            "away_w": int((away["result"] == "W").sum()),
            "away_l": int((away["result"] == "L").sum()),
            "away_margin": float(away["margin"].mean()) if len(away) > 0 else 0.0,
        })

    print("\n" + "=" * 60)
    print(f" 7. HOME-FIELD ADVANTAGE ({year})")
    print("=" * 60)
    print(f"{'Team':<22} {'Home W-L':>9} {'H Margin':>9} {'Away W-L':>9} {'A Margin':>9}")
    print("-" * 62)
    for s in splits:
        diff = s["home_margin"] - s["away_margin"]
        print(f"{s['team']:<22} {s['home_w']}-{s['home_l']:>2}{s['home_margin']:>+9.1f} "
              f"{s['away_w']}-{s['away_l']:>2}{s['away_margin']:>+9.1f}")

    # League-wide trend
    print(f"\n{'Year':<8} {'Home Win%':>10} {'Avg Home Margin':>16}")
    print("-" * 37)
    trends = []
    for y in range(max(2015, year - 5), year + 1):
        ydf = tm[(tm["year"] == y) & tm["is_home"]]
        if len(ydf) == 0:
            continue
        hw_pct = (ydf["result"] == "W").mean()
        hm = ydf["margin"].mean()
        print(f"{y:<8} {hw_pct:>9.1%} {hm:>+16.1f}")
        trends.append({"year": y, "home_win_pct": round(float(hw_pct), 4),
                        "home_margin_avg": round(float(hm), 2)})

    return {"home_field_advantage": {"splits": splits, "trends": trends}}


# ═══════════════════════════════════════════════════════════════════════════
# Section 8: Betting Form (Unit Profit)
# ═══════════════════════════════════════════════════════════════════════════

def betting_form(tm: pd.DataFrame, matches: pd.DataFrame, odds: pd.DataFrame,
                 year: int) -> dict:
    """Simulate $1 flat bets on each team, compute ROI."""
    if odds is None:
        print("\n[SKIPPED] Betting Form — no odds data")
        return {"betting_form": "no_data"}

    # We need implied probability → decimal odds: decimal = 1 / implied_prob
    mo = matches[matches["year"] == year][["match_id", "home_team", "away_team",
                                            "home_score", "away_score"]].copy()
    mo = mo.merge(odds[["match_id", "market_home_implied_prob", "market_away_implied_prob"]],
                  on="match_id", how="inner")
    mo["home_win"] = mo["home_score"] > mo["away_score"]
    mo["away_win"] = mo["away_score"] > mo["home_score"]
    mo["home_odds"] = 1.0 / mo["market_home_implied_prob"]
    mo["away_odds"] = 1.0 / mo["market_away_implied_prob"]

    # Compute profit per team
    profits = {}
    for _, row in mo.iterrows():
        ht, at = row["home_team"], row["away_team"]
        # Home team bet
        if ht not in profits:
            profits[ht] = {"bets": 0, "profit": 0.0}
        profits[ht]["bets"] += 1
        if row["home_win"]:
            profits[ht]["profit"] += row["home_odds"] - 1  # win: pay odds - stake
        else:
            profits[ht]["profit"] -= 1  # lose stake

        # Away team bet
        if at not in profits:
            profits[at] = {"bets": 0, "profit": 0.0}
        profits[at]["bets"] += 1
        if row["away_win"]:
            profits[at]["profit"] += row["away_odds"] - 1
        else:
            profits[at]["profit"] -= 1

    results = []
    for team in sorted(profits, key=lambda t: profits[t]["profit"], reverse=True):
        p = profits[team]
        roi = p["profit"] / p["bets"] if p["bets"] > 0 else 0
        results.append({
            "team": team, "bets": p["bets"],
            "profit": round(p["profit"], 2),
            "roi": round(roi, 4),
        })

    print("\n" + "=" * 60)
    print(f" 8. BETTING FORM — $1 Flat Bet ROI ({year})")
    print("=" * 60)
    print(f"{'Team':<25} {'Bets':>5} {'Profit':>8} {'ROI':>8}")
    print("-" * 49)
    for r in results:
        print(f"{r['team']:<25} {r['bets']:>5} {r['profit']:>+8.2f} {r['roi']:>+7.1%}")

    return {"betting_form": results}


# ═══════════════════════════════════════════════════════════════════════════
# Section 9: Form Guide
# ═══════════════════════════════════════════════════════════════════════════

def form_guide(tm: pd.DataFrame, elo_df: pd.DataFrame, year: int,
               last_n: int = 6) -> dict:
    """Recent W/L/D string with Elo-weighted opponent quality."""
    df = tm[tm["year"] == year].sort_values("date")
    df = df.merge(elo_df[["match_id", "team", "opp_elo_pre"]],
                  on=["match_id", "team"], how="left")

    print("\n" + "=" * 60)
    print(f" 9. FORM GUIDE — Last {last_n} matches ({year})")
    print("=" * 60)
    print(f"{'Team':<22} {'Form':<10} {'Opp Elo Avg':>12} {'Margin Avg':>11}")
    print("-" * 58)

    results = []
    for team in sorted(df["team"].unique()):
        tdf = df[df["team"] == team].tail(last_n)
        form_str = "".join(tdf["result"].astype(str).tolist())
        opp_elo = tdf["opp_elo_pre"].mean() if "opp_elo_pre" in tdf.columns else np.nan
        margin = tdf["margin"].mean()
        wins = (tdf["result"] == "W").sum()
        print(f"{team:<22} {form_str:<10} {opp_elo:>12.1f} {margin:>+11.1f}")
        results.append({
            "team": team, "form": form_str,
            "wins_last_n": int(wins), "last_n": last_n,
            "opp_elo_avg": round(float(opp_elo), 1) if not np.isnan(opp_elo) else None,
            "margin_avg": round(float(margin), 2),
        })

    return {"form_guide": results}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AusSportsTipping Comparison Analytics")
    parser.add_argument("--year", type=int, default=2025,
                        help="Season to analyse (default: 2025)")
    parser.add_argument("--last-n", type=int, default=6,
                        help="Number of recent matches for form guide (default: 6)")
    args = parser.parse_args()

    year = args.year
    last_n = args.last_n

    print(f"\n{'#' * 60}")
    print(f"  AusSportsTipping Comparison Analytics — {year}")
    print(f"{'#' * 60}")

    # Load data
    tm, matches, odds = load_data()
    print(f"\nLoaded: {len(tm)} team-matches, {len(matches)} matches"
          + (f", {len(odds)} odds rows" if odds is not None else ", no odds"))

    # Compute Elo ratings
    elo_df = compute_elo(tm)
    print(f"Computed Elo for {elo_df['team'].nunique()} teams")

    # Run all 9 sections
    output = {"year": year, "last_n": last_n}
    output.update(elo_ratings(tm, elo_df, year))
    output.update(ats_record(tm, matches, odds, year))
    output.update(predictability_index(tm, elo_df, year))
    output.update(score_statistics(tm, year))
    output.update(stadium_analysis(tm, year))
    output.update(winning_margins(matches, year))
    output.update(home_field_advantage(tm, year))
    output.update(betting_form(tm, matches, odds, year))
    output.update(form_guide(tm, elo_df, year, last_n))

    # Save JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved JSON summary → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
