"""
AFL Multi-Bet Correlation Engine
=================================
Monte Carlo simulation for computing correlated joint probabilities
across same-game multi-bet legs.

Key insight: bookmakers price multi legs assuming independence.
But within the same game, outcomes are correlated through shared
game state (team total, margin, pace). We model this by:
  1. Sampling match-level outcomes (margin, total score)
  2. Sampling player-level stats conditioned on team totals
  3. Computing joint hit rates from simulation traces

Usage:
    from multi import MultiEngine

    engine = MultiEngine(n_sims=10000)
    traces = engine.simulate_match(match_players_df, home_team, away_team,
                                    predicted_margin, predicted_total, home_win_prob)
    joint_prob = engine.compute_joint_prob(legs, traces)
    best = engine.find_best_multis(traces, candidate_legs)
"""

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config
from store import LearningStore

# ---------------------------------------------------------------------------
# Configuration defaults (pulled from config.py)
# ---------------------------------------------------------------------------

N_SIMS = getattr(config, "MULTI_N_SIMS", 10_000)
MAX_LEGS = getattr(config, "MULTI_MAX_LEGS", 4)
MIN_EDGE = getattr(config, "MULTI_MIN_EDGE", 0.03)
MIN_LEG_PROB = getattr(config, "MULTI_MIN_LEG_PROB", 0.30)

# Market-type overround margins (from config, with defaults from bet365 analysis)
_OVERROUND = getattr(config, "BOOK_OVERROUND", {})

def _book_implied(model_prob: float, market_type: str) -> float:
    """Convert model probability to book-implied probability using market-type overround.

    The bookmaker inflates implied probabilities by the overround.
    For a fair probability p, the book prices it as p * (1 + overround).
    We invert: book_implied = model_prob * (1 + overround), capped at 1.0.
    """
    overround = _OVERROUND.get(market_type, 0.08)  # default 8% if unknown
    return float(min(model_prob * (1.0 + overround), 1.0))

# Empirical AFL ratios (points per scoring shot, disposals per point, etc.)
_POINTS_PER_GOAL = 6.0       # approximate — each scoring shot is ~4.5 points
_DISP_PER_POINT = 4.5        # empirical team disposals per team score point
_MARKS_PER_POINT = 1.1       # empirical team marks per team score point
_DISP_CV = 0.30              # coefficient of variation for disposal draws
_NEGBIN_R_DEFAULT = 6.35     # NegBin dispersion for marks (from config/model)
_MARGIN_STD = 40.0           # std of margin distribution
_TOTAL_STD = 30.0            # std of total score distribution


# ---------------------------------------------------------------------------
# Helper: build leg label
# ---------------------------------------------------------------------------

def _player_short(name):
    """'Cripps, Patrick' -> 'P. Cripps'"""
    if ", " in name:
        last, first = name.split(", ", 1)
        return f"{first[0]}. {last}"
    return name


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class MultiEngine:
    """Correlated Monte Carlo engine for same-game multi-bet pricing.

    Simulates match-level outcomes (margin, total) and conditions
    player-level stats on the simulated team totals. This captures
    within-game correlation that independence-based pricing ignores.
    """

    def __init__(self, n_sims=None, seed=42):
        """Initialise engine.

        Args:
            n_sims: Number of Monte Carlo simulations per match.
            seed:   Random seed for reproducibility.
        """
        self.n_sims = n_sims or N_SIMS
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Match simulation
    # ------------------------------------------------------------------

    def simulate_match(
        self,
        match_players_df,
        home_team,
        away_team,
        predicted_margin,
        predicted_total,
        home_win_prob,
    ):
        """Simulate a single match and return per-simulation traces.

        Args:
            match_players_df: DataFrame with one row per player in this match.
                Required columns: player, team, is_home,
                                  lambda_goals, p_scorer,
                                  lambda_disp, lambda_marks, p_mark_taker
            home_team:        Home team name.
            away_team:        Away team name.
            predicted_margin: Predicted home margin (positive = home favoured).
            predicted_total:  Predicted combined score.
            home_win_prob:    P(home win).

        Returns:
            dict with keys:
                margin      — (n_sims,) float32 home margin draws
                total       — (n_sims,) float32 total score draws
                home_score  — (n_sims,) float32
                away_score  — (n_sims,) float32
                home_team   — str
                away_team   — str
                home_win_prob — float
                players     — list of dicts, one per player:
                    {player, team, is_home, idx,
                     goals: (n_sims,), disp: (n_sims,), marks: (n_sims,)}
        """
        n = self.n_sims
        rng = self.rng

        # ── Match-level draws ───────────────────────────────────────
        margin = rng.normal(predicted_margin, _MARGIN_STD, size=n).astype(np.float32)
        total = rng.normal(predicted_total, _TOTAL_STD, size=n).astype(np.float32)

        home_score = np.clip((total + margin) / 2.0, 10.0, 200.0)
        away_score = np.clip((total - margin) / 2.0, 10.0, 200.0)

        # Team-level derived quantities (vectorised over sims)
        home_goals_team = home_score / _POINTS_PER_GOAL       # (n,)
        away_goals_team = away_score / _POINTS_PER_GOAL
        home_disp_team = home_score * _DISP_PER_POINT          # (n,)
        away_disp_team = away_score * _DISP_PER_POINT
        home_marks_team = home_score * _MARKS_PER_POINT        # (n,)
        away_marks_team = away_score * _MARKS_PER_POINT

        # ── Per-team lambda sums (for share allocation) ─────────────
        df = match_players_df
        home_mask = df["is_home"].values.astype(bool)
        away_mask = ~home_mask

        # Extract player parameters
        lam_goals_all = df["lambda_goals"].values.astype(np.float64)
        p_scorer_all = df["p_scorer"].values.astype(np.float64)
        lam_disp_all = df["lambda_disp"].values.astype(np.float64)
        lam_marks_all = df["lambda_marks"].values.astype(np.float64)
        p_mark_taker_all = df["p_mark_taker"].values.astype(np.float64)

        # Team lambda sums for share calculation
        home_lam_goals_sum = max(lam_goals_all[home_mask].sum(), 0.01)
        away_lam_goals_sum = max(lam_goals_all[away_mask].sum(), 0.01)
        home_lam_disp_sum = max(lam_disp_all[home_mask].sum(), 0.01)
        away_lam_disp_sum = max(lam_disp_all[away_mask].sum(), 0.01)
        home_lam_marks_sum = max(lam_marks_all[home_mask].sum(), 0.01)
        away_lam_marks_sum = max(lam_marks_all[away_mask].sum(), 0.01)

        # ── Player-level simulation ─────────────────────────────────
        players = []
        n_players = len(df)
        player_names = df["player"].values
        player_teams = df["team"].values
        is_home_arr = home_mask

        for i in range(n_players):
            is_home = bool(is_home_arr[i])

            # --- Goals ---
            # Share of team goals proportional to lambda_goals
            if is_home:
                share_goals = lam_goals_all[i] / home_lam_goals_sum
                team_goals_sim = home_goals_team  # (n,)
            else:
                share_goals = lam_goals_all[i] / away_lam_goals_sum
                team_goals_sim = away_goals_team

            player_mu_goals = share_goals * team_goals_sim  # (n,) expected goals per sim
            # Zero-inflation: player only scores if they are a scorer in this sim
            p_sc = np.clip(p_scorer_all[i], 0.0, 1.0)
            scorer_mask = rng.random(n) < p_sc  # (n,) bool
            # Draw from Poisson conditioned on team total
            # Use ceiling of player_mu_goals as Poisson lambda (clipped > 0)
            raw_goals = rng.poisson(np.maximum(player_mu_goals, 0.001))  # (n,)
            sim_goals = np.where(scorer_mask, np.maximum(raw_goals, 1), 0).astype(np.int16)

            # --- Disposals ---
            if is_home:
                share_disp = lam_disp_all[i] / home_lam_disp_sum
                team_disp_sim = home_disp_team
            else:
                share_disp = lam_disp_all[i] / away_lam_disp_sum
                team_disp_sim = away_disp_team

            player_mu_disp = share_disp * team_disp_sim  # (n,)
            player_std_disp = player_mu_disp * _DISP_CV
            sim_disp = np.maximum(
                rng.normal(player_mu_disp, np.maximum(player_std_disp, 0.5)),
                0.0,
            ).astype(np.float32)

            # --- Marks ---
            if is_home:
                share_marks = lam_marks_all[i] / home_lam_marks_sum
                team_marks_sim = home_marks_team
            else:
                share_marks = lam_marks_all[i] / away_lam_marks_sum
                team_marks_sim = away_marks_team

            player_mu_marks = share_marks * team_marks_sim  # (n,)
            # NegBin: parameterised by (r, p) where mean = r*(1-p)/p
            r = _NEGBIN_R_DEFAULT
            # For each sim, p_param = r / (r + mu), then draw NegBin(r, p_param)
            mu_mk = np.maximum(player_mu_marks, 0.01)
            p_param = r / (r + mu_mk)  # (n,)
            sim_marks = rng.negative_binomial(r, p_param).astype(np.int16)

            players.append({
                "player": str(player_names[i]),
                "team": str(player_teams[i]),
                "is_home": is_home,
                "idx": i,
                "goals": sim_goals,
                "disp": sim_disp,
                "marks": sim_marks,
            })

        return {
            "margin": margin,
            "total": total,
            "home_score": home_score,
            "away_score": away_score,
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": home_win_prob,
            "players": players,
        }

    # ------------------------------------------------------------------
    # 2. Joint probability computation
    # ------------------------------------------------------------------

    def compute_joint_prob(self, legs, traces):
        """Compute joint probability that ALL legs hit from simulation traces.

        Args:
            legs: list of leg dicts. Each leg has:
                type: str — one of:
                    "goals_over"      — player goals >= threshold
                    "disp_over"       — player disposals >= threshold
                    "marks_over"      — player marks >= threshold
                    "match_winner"    — home or away win
                    "match_total_over" — combined score >= threshold
                    "team_total_over" — specific team score >= threshold
                    "margin_over"     — abs(margin) >= threshold
                threshold: float
                player_idx: int (index into traces["players"], for player legs)
                team: str (for team_total_over or match_winner)
                side: str "home" or "away" (for match_winner)
            traces: dict returned by simulate_match()

        Returns:
            float — joint probability (mean of AND across all legs)
        """
        n = self.n_sims
        joint_mask = np.ones(n, dtype=bool)

        for leg in legs:
            leg_type = leg["type"]
            threshold = leg.get("threshold", 0)

            if leg_type == "goals_over":
                p_data = traces["players"][leg["player_idx"]]["goals"]
                joint_mask &= (p_data >= threshold)

            elif leg_type == "disp_over":
                p_data = traces["players"][leg["player_idx"]]["disp"]
                joint_mask &= (p_data >= threshold)

            elif leg_type == "marks_over":
                p_data = traces["players"][leg["player_idx"]]["marks"]
                joint_mask &= (p_data >= threshold)

            elif leg_type == "match_winner":
                if leg.get("side") == "home":
                    joint_mask &= (traces["margin"] > 0)
                else:
                    joint_mask &= (traces["margin"] < 0)

            elif leg_type == "match_total_over":
                combined = traces["home_score"] + traces["away_score"]
                joint_mask &= (combined >= threshold)

            elif leg_type == "team_total_over":
                team = leg["team"]
                if team == traces["home_team"]:
                    joint_mask &= (traces["home_score"] >= threshold)
                elif team == traces["away_team"]:
                    joint_mask &= (traces["away_score"] >= threshold)
                else:
                    # Team not in this match — leg cannot hit
                    joint_mask[:] = False

            elif leg_type == "margin_over":
                joint_mask &= (np.abs(traces["margin"]) >= threshold)

            else:
                raise ValueError(f"Unknown leg type: {leg_type}")

        return float(joint_mask.mean())

    # ------------------------------------------------------------------
    # 3. Find best multi-bet combinations
    # ------------------------------------------------------------------

    def find_best_multis(
        self,
        traces,
        candidate_legs,
        max_legs=None,
        min_edge=None,
        top_n=20,
    ):
        """Find multi-bet combinations with positive edge over independence.

        Args:
            traces:         dict returned by simulate_match()
            candidate_legs: list of leg dicts, each with:
                - All fields needed by compute_joint_prob()
                - book_implied_prob: float (bookmaker's implied probability)
                - label: str (human-readable label)
                - solo_prob: float (model's marginal probability)
            max_legs:       Maximum legs per combo (default: config MULTI_MAX_LEGS)
            min_edge:       Minimum edge to keep (default: config MULTI_MIN_EDGE)
            top_n:          Return top N combos sorted by edge.

        Returns:
            list of dicts, each with:
                legs:             list of leg dicts
                n_legs:           int
                joint_prob:       float (correlated MC probability)
                indep_prob:       float (product of marginal model probs)
                book_implied:     float (product of book implied probs)
                edge_vs_book:     float (joint_prob - book_implied)
                edge_vs_indep:    float (joint_prob - indep_prob)
                correlation_lift: float (joint_prob / indep_prob)
        """
        max_legs = max_legs or MAX_LEGS
        min_edge = min_edge or MIN_EDGE

        # Pre-filter: only legs with reasonable probability and positive single-leg edge
        filtered = []
        for leg in candidate_legs:
            solo = leg.get("solo_prob", 0.0)
            book = leg.get("book_implied_prob", 1.0)
            if solo >= MIN_LEG_PROB and solo > book:
                filtered.append(leg)

        if len(filtered) < 2:
            return []

        # Cap candidates to avoid combinatorial explosion
        # Sort by edge descending, keep top 15
        filtered.sort(key=lambda x: x.get("solo_prob", 0) - x.get("book_implied_prob", 1), reverse=True)
        filtered = filtered[:15]

        results = []

        for n_legs in range(2, min(max_legs + 1, len(filtered) + 1)):
            for combo in itertools.combinations(filtered, n_legs):
                # Skip combos with duplicate players (same player, different stat)
                # unless they are team/match-level legs
                player_counts = defaultdict(int)
                skip = False
                for leg in combo:
                    p = leg.get("player", "")
                    if p:
                        player_counts[p] += 1
                        if player_counts[p] > 2:
                            skip = True
                            break
                if skip:
                    continue

                # Compute joint probability via correlated simulation
                joint_prob = self.compute_joint_prob(list(combo), traces)

                # Independence assumption (product of marginals)
                indep_prob = 1.0
                for leg in combo:
                    indep_prob *= leg.get("solo_prob", 0.5)

                # Book implied (product of book probs)
                book_implied = 1.0
                for leg in combo:
                    book_implied *= leg.get("book_implied_prob", leg.get("solo_prob", 0.5))

                edge_vs_book = joint_prob - book_implied

                if edge_vs_book < min_edge:
                    continue

                correlation_lift = joint_prob / indep_prob if indep_prob > 0 else 1.0

                results.append({
                    "legs": [_leg_summary(leg) for leg in combo],
                    "n_legs": n_legs,
                    "joint_prob": round(joint_prob, 4),
                    "indep_prob": round(indep_prob, 4),
                    "book_implied": round(book_implied, 4),
                    "edge_vs_book": round(edge_vs_book, 4),
                    "edge_vs_indep": round(joint_prob - indep_prob, 4),
                    "correlation_lift": round(correlation_lift, 3),
                })

        # Sort by edge descending
        results.sort(key=lambda x: x["edge_vs_book"], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # 4. Cross-game multi computation
    # ------------------------------------------------------------------

    def compute_cross_game_prob(self, legs_by_match, traces_by_match):
        """Compute joint probability for legs spanning multiple matches.

        Cross-game legs are independent (different matches), so we multiply
        within-game joint probabilities.

        Args:
            legs_by_match: dict mapping match_id -> list of legs for that match
            traces_by_match: dict mapping match_id -> traces from simulate_match()

        Returns:
            float — joint probability across all matches
        """
        cross_prob = 1.0
        for match_id, legs in legs_by_match.items():
            if match_id not in traces_by_match:
                return 0.0
            within_prob = self.compute_joint_prob(legs, traces_by_match[match_id])
            cross_prob *= within_prob
        return cross_prob


# ---------------------------------------------------------------------------
# Leg summary helper
# ---------------------------------------------------------------------------

def _leg_summary(leg):
    """Extract a clean summary dict from a candidate leg."""
    return {
        "player": leg.get("player", ""),
        "team": leg.get("team", ""),
        "type": leg.get("type", ""),
        "threshold": leg.get("threshold", 0),
        "label": leg.get("label", ""),
        "solo_prob": round(leg.get("solo_prob", 0.0), 4),
        "book_implied_prob": round(leg.get("book_implied_prob", 0.0), 4),
    }


# ---------------------------------------------------------------------------
# Build candidate legs from prediction DataFrames
# ---------------------------------------------------------------------------

def build_candidate_legs_from_predictions(
    preds_df, game_preds_df=None, match_id=None,
):
    """Build candidate leg list from pipeline prediction DataFrames.

    Args:
        preds_df:      Player-level predictions (from cmd_predict or sequential store).
                       Expected columns: player, team, match_id, is_home,
                           p_scorer / p_1plus_goals, p_2plus_goals, p_3plus_goals,
                           predicted_goals, lambda_goals,
                           p_15plus_disp, p_20plus_disp, p_25plus_disp, p_30plus_disp,
                           predicted_disposals, lambda_disposals,
                           predicted_marks, lambda_marks (optional)
        game_preds_df: Game-level predictions (optional).
                       Expected columns: match_id, home_team, away_team,
                           home_win_prob, predicted_margin.
        match_id:      Filter to a single match (optional).

    Returns:
        (candidate_legs, match_info) where:
            candidate_legs: list of leg dicts ready for find_best_multis()
            match_info: dict with match-level parameters for simulate_match()
    """
    df = preds_df.copy()
    if match_id is not None:
        df = df[df["match_id"] == match_id]
    if df.empty:
        return [], {}

    mid = df["match_id"].iloc[0]

    # Determine home/away teams
    if "is_home" in df.columns:
        home_rows = df[df["is_home"] == True]
        away_rows = df[df["is_home"] == False]
    else:
        teams = df["team"].unique()
        home_rows = df[df["team"] == teams[0]] if len(teams) > 0 else df
        away_rows = df[df["team"] == teams[1]] if len(teams) > 1 else pd.DataFrame()

    home_team = str(home_rows["team"].iloc[0]) if len(home_rows) > 0 else ""
    away_team = str(away_rows["team"].iloc[0]) if len(away_rows) > 0 else ""

    # Game-level info
    predicted_margin = 0.0
    predicted_total = 160.0
    home_win_prob = 0.5

    if game_preds_df is not None and len(game_preds_df) > 0:
        gp = game_preds_df[game_preds_df["match_id"] == mid]
        # Fallback: match by team names when match_ids differ (e.g. placeholder vs real)
        if len(gp) == 0 and "home_team" in game_preds_df.columns:
            teams = {home_team, away_team}
            gp = game_preds_df[
                (game_preds_df["home_team"].isin(teams))
                & (game_preds_df["away_team"].isin(teams))
            ]
        if len(gp) > 0:
            row = gp.iloc[0]
            predicted_margin = float(row.get("predicted_margin", 0.0))
            home_win_prob = float(row.get("home_win_prob", 0.5))
            # Correct home/away assignment from game predictions (authoritative)
            if "home_team" in row.index:
                gp_home = str(row["home_team"])
                gp_away = str(row["away_team"])
                if gp_home != home_team and gp_home == away_team:
                    home_team = gp_home
                    away_team = gp_away

    # Estimate total from player predicted_score sums
    if "predicted_score" in df.columns:
        predicted_total = float(df["predicted_score"].sum())
    elif "predicted_goals" in df.columns:
        predicted_total = float(df["predicted_goals"].sum()) * _POINTS_PER_GOAL * 1.8

    match_info = {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_margin": predicted_margin,
        "predicted_total": predicted_total,
        "home_win_prob": home_win_prob,
    }

    # Build match_players_df for simulate_match
    if "is_home" in df.columns:
        is_home_vals = df["is_home"].values
    else:
        is_home_vals = (df["team"] == home_team).values
    players_df = pd.DataFrame({
        "player": df["player"].values,
        "team": df["team"].values,
        "is_home": is_home_vals,
        "lambda_goals": df.get(
            "lambda_goals", df.get("predicted_goals", pd.Series([0.5] * len(df)))
        ).values.astype(float),
        "p_scorer": df.get(
            "p_scorer", df.get("p_1plus_goals", pd.Series([0.3] * len(df)))
        ).values.astype(float),
        "lambda_disp": df.get(
            "lambda_disposals", df.get("predicted_disposals", pd.Series([15.0] * len(df)))
        ).values.astype(float),
        "lambda_marks": df.get(
            "lambda_marks", df.get("predicted_marks", pd.Series([3.0] * len(df)))
        ).values.astype(float),
        "p_mark_taker": df.get(
            "p_mark_taker", pd.Series([0.5] * len(df))
        ).values.astype(float),
    })
    match_info["players_df"] = players_df

    # ── Build candidate legs ────────────────────────────────────────
    candidates = []

    # Player-level goal legs
    goal_legs_spec = [
        ("1plus_goals", 1, "p_1plus_goals", "p_scorer"),
        ("2plus_goals", 2, "p_2plus_goals", None),
        ("3plus_goals", 3, "p_3plus_goals", None),
    ]
    for leg_name, threshold, col1, col2 in goal_legs_spec:
        prob_col = col1 if col1 in df.columns else col2
        if prob_col is None or prob_col not in df.columns:
            continue
        for idx, (_, row) in enumerate(df.iterrows()):
            prob = float(row.get(prob_col, 0.0))
            if pd.isna(prob) or prob < 0.10:
                continue
            # Find player index in players_df
            player_idx = int(np.where(players_df["player"].values == row["player"])[0][0])
            candidates.append({
                "type": "goals_over",
                "threshold": threshold,
                "player_idx": player_idx,
                "player": str(row["player"]),
                "team": str(row["team"]),
                "solo_prob": float(np.clip(prob, 0, 1)),
                "book_implied_prob": _book_implied(float(np.clip(prob, 0, 1)), "player_goals"),
                "label": f"{_player_short(str(row['player']))} {threshold}+ goals",
            })

    # Player-level disposal legs
    disp_legs_spec = [
        (15, "p_15plus_disp"),
        (20, "p_20plus_disp"),
        (25, "p_25plus_disp"),
        (30, "p_30plus_disp"),
    ]
    for threshold, prob_col in disp_legs_spec:
        if prob_col not in df.columns:
            continue
        for _, row in df.iterrows():
            prob = float(row.get(prob_col, 0.0))
            if pd.isna(prob) or prob < 0.10:
                continue
            player_idx = int(np.where(players_df["player"].values == row["player"])[0][0])
            candidates.append({
                "type": "disp_over",
                "threshold": threshold,
                "player_idx": player_idx,
                "player": str(row["player"]),
                "team": str(row["team"]),
                "solo_prob": float(np.clip(prob, 0, 1)),
                "book_implied_prob": _book_implied(float(np.clip(prob, 0, 1)), "player_disposals"),
                "label": f"{_player_short(str(row['player']))} {threshold}+ disposals",
            })

    # Player-level marks legs (if available)
    marks_legs_spec = [
        (3, "p_3plus_mk"),
        (4, "p_4plus_mk"),
        (5, "p_5plus_mk"),
        (6, "p_6plus_mk"),
    ]
    for threshold, prob_col in marks_legs_spec:
        if prob_col not in df.columns:
            continue
        for _, row in df.iterrows():
            prob = float(row.get(prob_col, 0.0))
            if pd.isna(prob) or prob < 0.10:
                continue
            player_idx = int(np.where(players_df["player"].values == row["player"])[0][0])
            candidates.append({
                "type": "marks_over",
                "threshold": threshold,
                "player_idx": player_idx,
                "player": str(row["player"]),
                "team": str(row["team"]),
                "solo_prob": float(np.clip(prob, 0, 1)),
                "book_implied_prob": _book_implied(float(np.clip(prob, 0, 1)), "player_marks"),
                "label": f"{_player_short(str(row['player']))} {threshold}+ marks",
            })

    # Match winner legs
    candidates.append({
        "type": "match_winner",
        "side": "home",
        "team": home_team,
        "player": "",
        "player_idx": -1,
        "threshold": 0,
        "solo_prob": float(np.clip(home_win_prob, 0, 1)),
        "book_implied_prob": _book_implied(float(np.clip(home_win_prob, 0, 1)), "h2h"),
        "label": f"{home_team} to win",
    })
    candidates.append({
        "type": "match_winner",
        "side": "away",
        "team": away_team,
        "player": "",
        "player_idx": -1,
        "threshold": 0,
        "solo_prob": float(np.clip(1.0 - home_win_prob, 0, 1)),
        "book_implied_prob": _book_implied(float(np.clip(1.0 - home_win_prob, 0, 1)), "h2h"),
        "label": f"{away_team} to win",
    })

    # Team total legs
    for team, pred_score in [(home_team, predicted_total / 2 + predicted_margin / 2),
                              (away_team, predicted_total / 2 - predicted_margin / 2)]:
        for threshold in [60, 70, 80, 90, 100, 110]:
            from scipy.stats import norm
            prob = float(1.0 - norm.cdf(threshold, loc=pred_score, scale=25.0))
            if prob < 0.10:
                continue
            candidates.append({
                "type": "team_total_over",
                "threshold": threshold,
                "team": team,
                "player": "",
                "player_idx": -1,
                "solo_prob": float(np.clip(prob, 0, 1)),
                "book_implied_prob": _book_implied(float(np.clip(prob, 0, 1)), "totals"),
                "label": f"{team} {threshold}+ points",
            })

    # Match total legs
    for threshold in [130, 140, 150, 160, 170, 180]:
        from scipy.stats import norm
        prob = float(1.0 - norm.cdf(threshold, loc=predicted_total, scale=30.0))
        if prob < 0.10:
            continue
        candidates.append({
            "type": "match_total_over",
            "threshold": threshold,
            "player": "",
            "player_idx": -1,
            "team": "",
            "solo_prob": float(np.clip(prob, 0, 1)),
            "book_implied_prob": _book_implied(float(np.clip(prob, 0, 1)), "totals"),
            "label": f"Match total {threshold}+ points",
        })

    return candidates, match_info


def overlay_real_odds(candidates, player_odds_df, match_teams=None):
    """Replace placeholder book_implied_prob with actual Betfair odds where available.

    Args:
        candidates: list of candidate leg dicts (mutated in place)
        player_odds_df: DataFrame from player_odds.parquet with columns:
            player, team, market_type, line, implied_prob
        match_teams: optional list of team names to filter odds

    Returns:
        int — number of candidates updated with real odds
    """
    if player_odds_df is None or player_odds_df.empty:
        return 0

    df = player_odds_df
    if match_teams:
        df = df[df["team"].isin(match_teams)]
    if df.empty:
        return 0

    updated = 0
    for leg in candidates:
        player = leg.get("player", "")
        if not player:
            continue

        leg_type = leg["type"]
        threshold = leg.get("threshold", 0)

        # Map leg type + threshold to Betfair market type and line
        market_type = None
        line = None
        if leg_type == "goals_over" and threshold == 1:
            market_type = "FGS"  # First Goal Scorer (proxy for anytime)
        elif leg_type == "disp_over":
            market_type = "Disposals"
            line = threshold
        elif leg_type == "goals_over" and threshold >= 2:
            market_type = f"{threshold} Goals"

        if market_type is None:
            continue

        # Find matching odds row
        mask = df["player"].str.contains(player.split(",")[0].strip(), case=False, na=False)
        if "market_type" in df.columns:
            mask = mask & (df["market_type"] == market_type)
        if line is not None and "line" in df.columns:
            mask = mask & (df["line"] == line)

        matches = df[mask]
        if matches.empty:
            continue

        row = matches.iloc[0]
        if "implied_prob" in row.index and pd.notna(row["implied_prob"]):
            leg["book_implied_prob"] = float(row["implied_prob"])
            leg["odds_source"] = "betfair"
            updated += 1

    return updated


# ---------------------------------------------------------------------------
# Run multi analysis for a round
# ---------------------------------------------------------------------------

def run_multi_for_round(year, round_num, run_id=None, n_sims=None):
    """Load predictions and generate correlated multi-bet suggestions for a round.

    Args:
        year:      Season year.
        round_num: Round number.
        run_id:    Optional run ID for sequential store.
        n_sims:    Number of simulations.

    Returns:
        dict with:
            matches: list of match result dicts
            summary: aggregate stats
    """
    n_sims = n_sims or N_SIMS
    engine = MultiEngine(n_sims=n_sims)

    # Try sequential store first, then predictions directory
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=run_id)

    preds = pd.DataFrame()
    game_preds = pd.DataFrame()

    # Load from sequential store
    try:
        preds = store.load_predictions(year=year, round_num=round_num)
    except Exception:
        pass

    if preds.empty:
        # Fallback: load from predictions CSV
        pred_path = config.PREDICTIONS_DIR / str(year) / f"round_{round_num}_predictions.csv"
        if pred_path.exists():
            preds = pd.read_csv(pred_path)

    if preds.empty:
        print(f"No predictions found for {year} R{round_num}")
        return {"matches": [], "summary": {}}

    # Load game predictions
    try:
        game_preds = store.load_game_predictions(year=year, round_num=round_num)
    except Exception:
        pass

    print(f"\nMulti-Bet Correlation Engine — {year} Round {round_num}")
    print(f"  Simulations: {n_sims:,}")
    print(f"  Players: {len(preds)}")
    print("=" * 70)

    match_ids = preds["match_id"].unique()
    all_results = []

    for mid in match_ids:
        match_preds = preds[preds["match_id"] == mid]
        candidates, match_info = build_candidate_legs_from_predictions(
            match_preds, game_preds_df=game_preds, match_id=mid
        )

        if not candidates or "players_df" not in match_info:
            continue

        # Run simulation
        traces = engine.simulate_match(
            match_info["players_df"],
            match_info["home_team"],
            match_info["away_team"],
            match_info["predicted_margin"],
            match_info["predicted_total"],
            match_info["home_win_prob"],
        )

        # Find best multis
        best = engine.find_best_multis(traces, candidates)

        match_label = f"{match_info['home_team']} vs {match_info['away_team']}"
        print(f"\n  {match_label}")
        print(f"  Predicted: margin {match_info['predicted_margin']:+.0f}, "
              f"total {match_info['predicted_total']:.0f}")
        print(f"  Candidate legs: {len(candidates)}, Combos with edge: {len(best)}")

        if best:
            for i, combo in enumerate(best[:5]):
                legs_str = " + ".join(l["label"] for l in combo["legs"])
                print(f"    #{i+1}: {legs_str}")
                print(f"         Joint: {combo['joint_prob']:.1%}  "
                      f"Indep: {combo['indep_prob']:.1%}  "
                      f"Book: {combo['book_implied']:.1%}  "
                      f"Edge: {combo['edge_vs_book']:+.1%}  "
                      f"Lift: {combo['correlation_lift']:.2f}x")

        all_results.append({
            "match_id": int(mid),
            "match_label": match_label,
            "home_team": match_info["home_team"],
            "away_team": match_info["away_team"],
            "predicted_margin": match_info["predicted_margin"],
            "predicted_total": match_info["predicted_total"],
            "n_candidates": len(candidates),
            "best_combos": best,
        })

    # Summary
    total_combos = sum(len(r["best_combos"]) for r in all_results)
    matches_with_edge = sum(1 for r in all_results if r["best_combos"])
    avg_edge = 0.0
    if total_combos > 0:
        all_edges = [c["edge_vs_book"] for r in all_results for c in r["best_combos"]]
        avg_edge = float(np.mean(all_edges))

    summary = {
        "year": year,
        "round": round_num,
        "n_matches": len(match_ids),
        "matches_with_edge": matches_with_edge,
        "total_combos": total_combos,
        "avg_edge": round(avg_edge, 4),
        "n_sims": n_sims,
    }

    print(f"\n{'=' * 70}")
    print(f"  Summary: {total_combos} combos with edge across "
          f"{matches_with_edge}/{len(match_ids)} matches")
    if total_combos > 0:
        print(f"  Average edge: {avg_edge:+.1%}")
    print(f"{'=' * 70}")

    # Save results
    _save_multi_results(all_results, summary, year, round_num)

    return {"matches": all_results, "summary": summary}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_multi_results(matches, summary, year, round_num):
    """Save multi-bet results to JSON."""
    multi_dir = getattr(config, "MULTI_DIR", config.DATA_DIR / "multi")
    multi_dir.mkdir(parents=True, exist_ok=True)

    out_path = multi_dir / f"multi_{year}_R{round_num:02d}.json"

    output = {
        "summary": summary,
        "matches": matches,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_multis(year, run_id=None, n_sims=None):
    """Backtest correlated multi-bet engine across a full season.

    Loads sequential predictions round by round, simulates matches,
    generates candidate legs, and tracks hit rates.

    Args:
        year:   Season to backtest.
        run_id: Sequential run ID.
        n_sims: Simulations per match.

    Returns:
        dict with aggregate backtest results.
    """
    n_sims = n_sims or 5000  # fewer sims for backtest speed
    engine = MultiEngine(n_sims=n_sims)
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=run_id)

    # Find available runs
    runs = store.list_runs(year=year, subdir="predictions")
    if not runs:
        print(f"No sequential runs found for {year}")
        return {}

    run_id = run_id or runs[-1]
    store = LearningStore(base_dir=config.SEQUENTIAL_DIR, run_id=run_id)

    round_tuples = store.list_rounds(subdir="predictions", year=year)
    if not round_tuples:
        print(f"No prediction rounds for {year}")
        return {}

    print(f"\nMulti-Bet Backtest — {year} (run: {run_id})")
    print(f"  Rounds: {len(round_tuples)}, Sims per match: {n_sims:,}")
    print("=" * 70)

    # Accumulators
    total_combos = 0
    total_hits = 0
    edge_bins = defaultdict(lambda: {"n": 0, "hits": 0})
    round_results = {}

    for yr, rnd in sorted(round_tuples):
        try:
            preds = store.load_predictions(year=yr, round_num=rnd)
            outcomes = store.load_outcomes(year=yr, round_num=rnd)
        except Exception:
            continue

        if preds.empty or outcomes.empty:
            continue

        # Merge predictions with outcomes for hit verification
        merged = preds.merge(outcomes, on=["player", "team", "match_id"], how="inner",
                             suffixes=("", "_actual"))

        game_preds = pd.DataFrame()
        try:
            game_preds = store.load_game_predictions(year=yr, round_num=rnd)
        except Exception:
            pass

        round_combos = 0
        round_hits = 0

        for mid in merged["match_id"].unique():
            match_df = merged[merged["match_id"] == mid]
            candidates, match_info = build_candidate_legs_from_predictions(
                match_df, game_preds_df=game_preds, match_id=mid
            )

            if not candidates or "players_df" not in match_info:
                continue

            traces = engine.simulate_match(
                match_info["players_df"],
                match_info["home_team"],
                match_info["away_team"],
                match_info["predicted_margin"],
                match_info["predicted_total"],
                match_info["home_win_prob"],
            )

            best = engine.find_best_multis(traces, candidates, min_edge=0.02)

            for combo in best:
                # Check if combo actually hit using outcome data
                hit = _check_combo_hit(combo, match_df)
                if hit is not None:
                    total_combos += 1
                    round_combos += 1
                    if hit:
                        total_hits += 1
                        round_hits += 1

                    # Bin by edge
                    edge = combo["edge_vs_book"]
                    if edge >= 0.10:
                        bin_key = "10%+"
                    elif edge >= 0.05:
                        bin_key = "5-10%"
                    else:
                        bin_key = "2-5%"
                    edge_bins[bin_key]["n"] += 1
                    if hit:
                        edge_bins[bin_key]["hits"] += 1

        round_results[rnd] = {"combos": round_combos, "hits": round_hits}
        if round_combos > 0:
            print(f"  R{rnd:02d}: {round_hits}/{round_combos} hit "
                  f"({round_hits / round_combos:.0%})")

    # Final report
    print(f"\n{'=' * 70}")
    print(f"  BACKTEST RESULTS — {year}")
    print(f"{'=' * 70}")
    overall_rate = total_hits / total_combos if total_combos > 0 else 0
    print(f"  Total combos: {total_combos}")
    print(f"  Total hits:   {total_hits} ({overall_rate:.1%})")
    print()
    print(f"  {'Edge Bin':<12} {'Combos':>8} {'Hits':>8} {'Hit Rate':>10}")
    print(f"  {'-'*10:<12} {'-'*8:>8} {'-'*8:>8} {'-'*10:>10}")
    for bin_key in ["2-5%", "5-10%", "10%+"]:
        b = edge_bins[bin_key]
        hr = b["hits"] / b["n"] if b["n"] > 0 else 0
        print(f"  {bin_key:<12} {b['n']:>8} {b['hits']:>8} {hr:>9.1%}")
    print(f"{'=' * 70}")

    return {
        "year": year,
        "total_combos": total_combos,
        "total_hits": total_hits,
        "hit_rate": round(overall_rate, 4),
        "edge_bins": dict(edge_bins),
        "rounds": round_results,
    }


def _check_combo_hit(combo, match_df):
    """Check if a multi-bet combo hit using actual outcomes.

    Returns True/False/None (None if actuals not available).
    """
    for leg in combo["legs"]:
        leg_type = leg["type"]
        player = leg.get("player", "")

        if leg_type == "goals_over":
            row = match_df[match_df["player"] == player]
            if row.empty:
                return None
            actual = row.iloc[0].get("actual_goals", np.nan)
            if pd.isna(actual):
                return None
            if actual < leg["threshold"]:
                return False

        elif leg_type == "disp_over":
            row = match_df[match_df["player"] == player]
            if row.empty:
                return None
            actual = row.iloc[0].get("actual_disposals", np.nan)
            if pd.isna(actual):
                return None
            if actual < leg["threshold"]:
                return False

        elif leg_type == "marks_over":
            row = match_df[match_df["player"] == player]
            if row.empty:
                return None
            actual = row.iloc[0].get("actual_marks", np.nan)
            if pd.isna(actual):
                return None
            if actual < leg["threshold"]:
                return False

        elif leg_type in ("match_winner", "match_total_over", "team_total_over", "margin_over"):
            # These require team_matches data which we don't have here;
            # skip for now (return None to exclude from hit tracking)
            return None

    return True


# ---------------------------------------------------------------------------
# CLI entry point (called from pipeline.py)
# ---------------------------------------------------------------------------

def cmd_multi(args):
    """CLI handler for --multi flag."""
    year = args.year or config.CURRENT_SEASON_YEAR
    round_num = getattr(args, "round", None)
    run_id = getattr(args, "run_id", None)
    n_sims = getattr(args, "n_sims", N_SIMS)

    if round_num is None:
        print("Error: --multi requires --round N")
        return

    return run_multi_for_round(year, round_num, run_id=run_id, n_sims=n_sims)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AFL Multi-Bet Correlation Engine")
    parser.add_argument("--year", type=int, default=config.CURRENT_SEASON_YEAR)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--n-sims", type=int, default=N_SIMS)
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest across full season")
    args = parser.parse_args()

    if args.backtest:
        backtest_multis(args.year, run_id=args.run_id, n_sims=args.n_sims)
    else:
        run_multi_for_round(args.year, args.round, run_id=args.run_id, n_sims=args.n_sims)
