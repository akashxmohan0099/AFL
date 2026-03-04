# AFL Player Prediction Pipeline

A machine-learning pipeline that predicts individual player scoring (goals, behinds), disposal counts, and marks for Australian Football League matches. It scrapes historical data from AFL Tables and FootyWire, integrates weather and betting odds, engineers 252 features, trains multi-stage ensemble models, and generates per-player probability distributions for upcoming rounds.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Sources & APIs](#data-sources--apis)
3. [Raw Data Points](#raw-data-points)
4. [Pipeline Stages](#pipeline-stages)
5. [Feature Engineering (252 Features)](#feature-engineering-252-features)
6. [Models](#models)
7. [Predictions & Output](#predictions--output)
8. [Backtesting & Learning](#backtesting--learning)
9. [Analysis & Experiments](#analysis--experiments)
10. [Configuration Reference](#configuration-reference)
11. [File Map](#file-map)

---

## Architecture Overview

```
AFL Tables (web scrape)    FootyWire (scrape)    Open-Meteo API    Betting Odds (XLSX/CSV)
        │                        │                     │                     │
        ▼                        ▼                     ▼                     ▼
   Raw CSVs ──────────────────► clean.py ◄─────────────┴─────────────────────┘
   (player_stats, player_details,       │
    scoring, matches, umpires)          ▼
                                 player_games.parquet  (101K rows, 71 cols)
                                 matches.parquet       (2,258 rows, 38 cols)
                                 team_matches.parquet  (4,516 rows, 24 cols)
                                 umpires.parquet / coaches.parquet
                                 player_profiles.parquet
                                 footywire_advanced.parquet (93K rows)
                                        │
                                        ▼
                                  features.py (252 features)
                                        │
                              ┌─────────┼──────────┬──────────┐
                              ▼         ▼          ▼          ▼
                        Scoring     Disposal     Marks    Game Winner
                         Model       Model      Model      Model
                              │         │          │          │
                              ▼         ▼          ▼          ▼
                              Predictions per player per round
                                        │
                                        ▼
                              LearningStore (round-by-round)
                                        │
                                        ▼
                              Analysis & Calibration
```

---

## Data Sources & APIs

### 1. AFL Tables (Primary — Web Scrape)

The scraper (`scraper.py`) fetches season-by-season HTML pages from **afltables.com** and extracts data into year-specific CSVs:

| Category | Output Directory | Content |
|---|---|---|
| Player stats | `data/player_stats/` | Per-game box scores (kicks, marks, handballs, goals, etc.) |
| Player details | `data/player_details/` | Age, career games, career goals |
| Scoring events | `data/scoring/` | Quarter-by-quarter goal/behind breakdowns |
| Match metadata | `data/matches/` | Date, venue, scores, attendance, round info |
| Umpires | `data/umpires/` | Umpire assignments per match (extracted from match pages, zero extra requests) |
| Player profiles | (via `--scrape-profiles`) | Height, weight, DOB + career splits |

**Years covered:** 2015–2025 (configurable via `--start` / `--end`)

### 2. FootyWire Advanced Stats (Web Scrape)

Scraped by `scrape_footywire.py` into `data/footywire/advanced_stats_{year}.csv` (2015–2025, 101K total rows).

**Stats collected:** Effective Disposals (ED), Disposal Efficiency (DE%), Centre Clearances (CCL), Stoppage Clearances (SCL), Turnovers (TO), Metres Gained (MG), Score Involvements (SI), Intercepts (ITC), Tackles Inside 50 (T5), Time On Ground % (TOG%).

### 3. Open-Meteo Historical Weather API

| Field | Value |
|---|---|
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Authentication | None (free tier) |
| Rate limit | 0.5s between calls |
| Caching | File-based JSON cache in `data/base/weather_cache/` |

**Hourly variables fetched:**
- `temperature_2m`, `apparent_temperature`, `precipitation`, `rain`
- `wind_speed_10m`, `wind_gusts_10m`, `wind_direction_10m`, `relative_humidity_2m`, `dew_point_2m`
- `cloud_cover`, `surface_pressure`

**25 venues mapped** with lat/lon coordinates.

### 4. Betting Odds (Three Sources)

**Bookmaker (Source 2):** `afl_Source_2.xlsx` — 3,353 matches (2009–2025)
- Pre-game only: `home_odds_open`, `home_odds_close`, `away_odds_open`, `away_odds_close`, `home_line_open`, `home_line_close`, `total_score_open`, `total_score_close`

**Betfair Match Odds:** `AFL_YYYY_Match_Odds.csv` (2021–2025)
- Pre-game only: `BEST_BACK_FIRST_BOUNCE`, `BEST_LAY_FIRST_BOUNCE`

**Betfair Player Markets:** Disposals, First Goal Scorer, 2/3+ Goals (23.6K rows, 91.8% name match rate)
- Processed by `integrate_player_odds.py` → `data/base/player_odds.parquet`

---

## Raw Data Points

### Player Box-Score Stats (22 columns)

| Abbreviation | Stat | Abbreviation | Stat |
|---|---|---|---|
| KI | Kicks | CP | Contested Possessions |
| MK | Marks | UP | Uncontested Possessions |
| HB | Handballs | CM | Contested Marks |
| DI | Disposals | MI | Marks Inside 50 |
| GL | Goals | one_pct | One-Percenters |
| BH | Behinds | BO | Brownlow Votes |
| HO | Hitouts | GA | Goals Against |
| TK | Tackles | FF | Free Kicks For |
| RB | Rebounds | FA | Free Kicks Against |
| IF | Inside 50s | BR | Bounces |
| CL | Clearances | CG | Contested Gains |

Each stat also has a **rate-normalized** version (`*_rate`) — stat divided by `(pct_played / 90)` to account for partial-game time.

### Player Metadata
- `age_years` (float), `career_games_pre`, `career_goals_pre`, `career_goal_avg_pre` — all pre-game values to avoid leakage

### Match Metadata
- `match_id` (int64), `date`, `year`, `round_number` (1–28), `round_label` ("1"–"25" or EF/SF/PF/GF)
- `venue`, `is_home`, `is_finals`, `opponent`, `team`
- `home_score`, `away_score`, `margin`, `total_score`, `attendance`
- `game_time_minutes`, `home_rushed_behinds`, `away_rushed_behinds`

### Quarter-Level Scoring
- `q1_goals`, `q1_behinds`, `q2_goals`, `q2_behinds`, `q3_goals`, `q3_behinds`, `q4_goals`, `q4_behinds`

### Team-Match Aggregates
- `result` (W/L/D), `score`, `opp_score`, `margin`, `rest_days`
- Team totals: GL, BH, DI, IF, CL, CP, TK, RB, MK

---

## Pipeline Stages

All stages are invoked via `pipeline.py`:

```
python pipeline.py --scrape [--start 2015] [--end 2025]   # Fetch raw CSVs from AFL Tables
python pipeline.py --scrape-profiles                        # Fetch player height/weight/DOB + career splits
python pipeline.py --scrape-footywire                       # Fetch FootyWire advanced stats
python pipeline.py --clean                                  # Normalize → parquets
python pipeline.py --features                               # Engineer 252 features
python pipeline.py --train                                  # Train scoring model (goals + behinds)
python pipeline.py --train-disposals                        # Train disposal model
python pipeline.py --train-winner                           # Train game-winner model
python pipeline.py --predict --round N [--year YYYY]        # Predict a specific round
python pipeline.py --evaluate                               # Evaluate on validation year (2024)
python pipeline.py --backtest [--year YYYY]                 # Walk-forward per-round backtest
python pipeline.py --backtest-winner [--year YYYY]          # Walk-forward game winner backtest
python pipeline.py --diagnose [--year YYYY]                 # Breakdown of backtest results
python pipeline.py --sequential [--year YYYY]               # Calibration-aware sequential learning
python pipeline.py --tune                                   # Optuna hyperparameter tuning
python pipeline.py --save-experiment NAME                    # Save experiment results to JSON
python pipeline.py --update                                 # Scrape + rebuild + predict (current season)
python pipeline.py --reset-calibration                      # Clear calibration state
```

### Stage Details

| Stage | Input | Output | Description |
|---|---|---|---|
| **Scrape** | AFL Tables HTML | Year CSVs in `data/player_stats/`, `data/scoring/`, etc. | Fetches historical data per season |
| **Scrape Profiles** | AFL Tables player pages | `player_profiles.parquet`, `career_splits_*.parquet` | Height, weight, DOB, career splits |
| **Scrape FootyWire** | FootyWire HTML | `data/footywire/advanced_stats_{year}.csv` | Advanced stats (ED, DE%, TOG%, etc.) |
| **Clean** | Raw CSVs | `player_games.parquet` (101K rows), `matches.parquet`, `team_matches.parquet`, `umpires.parquet`, `coaches.parquet`, `footywire_advanced.parquet` | Joins, normalizes, computes rate columns |
| **Features** | Cleaned parquets | `feature_matrix.parquet` (252 features) + `feature_columns.json` | Rolling averages, archetypes, opponent profiles, venue, weather, odds, umpires, coaches |
| **Train** | Feature matrix | Model `.pkl` files + metadata JSONs | Fits scoring/disposal/marks/winner ensembles |
| **Predict** | Trained models + feature matrix | `round_N_predictions.csv`, `round_N_thresholds.csv` | Per-player distributions for an upcoming round |
| **Backtest** | Feature matrix | LearningStore entries per round | Walk-forward: train on all prior data, predict each round, save outcomes |
| **Tune** | Feature matrix | `data/tuning/best_params_*.json` | Optuna walk-forward hyperparameter optimization |

---

## Feature Engineering (252 Features)

Stages A–W, built in `features.py`. After pruning redundant features, the final matrix has **252 features**.

### A. Career & Age (5)
`age_years`, `age_squared`, `career_games_pre`, `career_goals_pre`, `career_goal_avg_capped`

### B. Recency-Weighted Rolling Averages (~74)

**Raw stat rolling means** over 3/5/10-game windows:
`player_gl_avg_3`, `player_gl_avg_5`, `player_gl_avg_10`, `player_di_avg_3`, `player_mk_avg_3`, `player_tk_avg_3`, `player_if50_avg_3`, `player_cl_avg_3`, `player_ho_avg_3`, `player_ga_avg_3`, `player_mi_avg_3`, `player_cm_avg_3`, `player_cp_avg_3`, `player_ff_avg_3`, `player_rb_avg_3`, `player_one_pct_avg_3`, etc.

**Rate-normalized rolling** (3/5/10 windows):
`player_gl_rate_avg_3`, `player_bh_rate_avg_3`, `player_di_rate_avg_3`, `player_mk_rate_avg_3`, etc.

**Exponentially weighted means** (span=5):
`player_gl_ewm_5`, `player_bh_ewm_5`, `player_di_ewm_5`, `player_mk_ewm_5`, plus rate versions

**Accuracy** (GL / (GL + BH)):
`player_accuracy_3`, `player_accuracy_5`, `player_accuracy_10`

**Streaks & form:**
`player_gl_streak`, `player_gl_streak_weighted` (decay=0.85), `player_gl_cold_streak`, `player_form_ratio`, `player_is_hot`, `player_is_cold`, `player_streak_just_broke`

**Volatility & trend:**
`player_gl_volatility_5`, `player_gl_trend_5`, `player_di_volatility_5`, `player_di_trend_5`

**Season cumulative:**
`season_goals_total`, `season_disposals_total`, `season_goals_rate_avg`, `season_disposals_rate_avg`

**Other:**
`days_since_last_match`, `is_returning_from_break`, `player_ki_hb_ratio_3`, `player_ki_hb_ratio_5`

### C. Venue Features (4)
`player_gl_at_venue_avg`, `player_bh_at_venue_avg`, `player_gl_venue_diff`, `venue_avg_goals_per_team`

### D. Opponent Defense (13)
`opp_goals_conceded_avg_5`, `opp_goals_conceded_avg_10`, `player_vs_opp_gl_avg`, `player_vs_opp_games`, `player_vs_opp_gl_diff`, `opp_disp_conceded_avg_5`, `opp_disp_conceded_avg_10`, `opp_contested_poss_diff_5`, `opp_key_defenders_count`, `opp_defender_strength_score`

### E. Defender Matchup Features (~6)
Player-vs-opponent historical performance splits.

### F. Team Context (8)
`team_goals_avg_5`, `team_goals_avg_10`, `team_if_avg_5`, `team_cl_avg_5`, `team_clearance_dominance_5`, `team_mid_quality_score`, `player_goal_share_5`, `team_win_pct_5`, `team_margin_avg_5`

### G. Scoring Patterns (6)
`player_q1_gl_pct`, `player_q2_gl_pct`, `player_q3_gl_pct`, `player_q4_gl_pct`, `player_late_scorer_pct`, `player_multi_goal_rate`

### H. Role Classification (2)
`forward_score` — derived from goals/inside-50s ratio
`player_role` — categorical: ruck, key_forward, small_forward, key_defender, midfielder, general

### I. Teammate Features (3)
`teammate_enabler_count`, `teammate_scoring_avg`, `interact_team_form_share`

### J. Interaction Terms (~10)
`interact_player_vs_opp_defense`, `interact_form_vs_defense`, `interact_home_scoring`, `interact_venue_boost`, `interact_hot_vs_weak_defense`, `interact_streak_forward`, `interact_disp_vs_contested`, `interact_disp_pace`, `interact_disp_vs_cp_diff`, `interact_mid_supply_forward`

### K. GMM Archetype Features (8)

Uses a **Gaussian Mixture Model (6 clusters)** fitted on player stat profiles to classify players into archetypes: Forward, Midfielder, Ruck, Defender, Tagger, Utility.

**Soft assignments:** `archetype_prob_0` through `archetype_prob_5`
**Archetype concession profiles:** `opp_arch_gl_conceded_avg_5`, `opp_arch_disp_conceded_avg_5`
**Disposal ceiling:** `archetype_di_ceiling_ratio`, `archetype_di_ceiling_5`

### L. Weather Features (~19)
`temperature_avg`, `apparent_temperature_avg`, `temperature_range`, `rain_total`, `wind_speed_avg`, `wind_speed_max`, `wind_gusts_max`, `wind_severity`, `wind_direction_variability`, `humidity_avg`, `dew_point_avg`, `cloud_cover_avg`, `feels_like_delta`, `weather_difficulty_score`, `slippery_conditions`

### M. Game Environment (~10)
`game_pace_proxy`, `expected_margin_diff`, `expected_margin_abs`, `ground_length`, `ground_width`, `ground_area`, `ground_shape_ratio`, `is_night_game`, `is_twilight_game`

### N. Umpire Features (~5)
Umpire scoring tendencies, free kick rates based on umpire assignment history (lookback 20 matches, min 10 games).

### O. Coach Features (~5)
Coach historical scoring rates, head-to-head matchup tendencies (min 10 games, min 3 H2H matches).

### P. Player Physical Features (~6)
Height, weight, BMI, age interactions with archetype (requires archetype from Stage K). Fallbacks: 186cm height, 86kg weight, 25 years age.

### Q. Career Split Features (~4)
Home/away, day/night career performance splits (min 3 games per split). Disabled by default (`CAREER_SPLIT_FEATURES_ENABLED=False`) — snapshot career splits are leaky for historical rows.

### R. Team Venue Features (~4)
Team-level venue scoring history over 5-year lookback (min 3 games). Fallback: 80 points.

### S. Player Market Odds Features (~8)
Implied probabilities, market total score, odds movement, market confidence, player-level implied goals/disposals.

### T. Venue Elevation Features (~2)
Ground elevation, altitude-adjusted scoring.

### U. Disposal-Specific Interaction Features (~8)
TOG rolling averages, disposal share, UP rolling, weather × disposal interactions, opponent tackles, rest day effects. Key feature: `player_disp_share_10` (#1 most important, 0.074 permutation importance).

### V. FootyWire Advanced Stats — DISABLED
24 features from ED, DE%, CCL, SCL, TO, MG, TOG%. **Reverted after A/B test showed BSS regression at all thresholds.**

### W. CBA Features — Stub
Placeholder for DFS Australia CBA data (future).

---

## Models

### 1. AFLScoringModel (Goals + Behinds)

A **two-stage ensemble** that addresses zero-inflation — roughly 68% of player-matches result in 0 goals.

**Stage 1 — Scorer Classifier:**
- Binary: P(player scores >= 1 goal)
- Algorithm: `GradientBoostingClassifier` (full train) or `HistGradientBoostingClassifier` (backtest)
- Output: `p_scorer`

**Stage 2 — Goals Regressor (trained on scorers only, GL >= 1):**

| Component | Algorithm | Weight |
|---|---|---|
| Poisson | `PoissonRegressor` (alpha=0.014, max_iter=1000) | 80% |
| GBT | `GradientBoostingRegressor` (n_estimators=300, max_depth=4, lr=0.05) | 20% |

**Behinds Regressor:** Same Poisson + GBT ensemble, trained on all players (behinds are more diffuse).

**Confidence intervals:** 80% intervals from a zero-inflated Poisson mixture using `p_scorer` and predicted lambda.

**Training split:** 2015–2023 train, 2024 validation. Baseline: `career_goal_avg_pre`.

### 2. AFLDisposalModel

Single-stage regressor for disposal count predictions.

| Component | Algorithm | Weight |
|---|---|---|
| Poisson | `PoissonRegressor` (alpha=0.015) | 80% |
| GBT | `HistGradientBoostingRegressor` (max_iter=217, max_depth=5, lr=0.052) | 20% |

**Threshold probabilities:** P(X >= k) for k in {10, 15, 20, 25, 30} disposals.

**Distribution:** Gaussian (default, best performer) — uses predicted mean + std from residuals.

**Upper-tail corrections** for 25+ and 30+ thresholds: STD multiplier 1.2, skew alpha 2.0, 30+ probability scaled 1.3x with cap at 0.45.

**Isotonic calibration** is the key lever — walk-forward isotonic post-processing yields BSS 30.9% at 15+, 30.4% at 20+, 13.7% at 25+/30+.

### 3. AFLMarksModel

Single-stage regressor for marks predictions. Same Poisson + GBT architecture.

**Threshold probabilities:** P(X >= k) for k in {2, 3, 4, 5, 6, 7, 8, 9, 10} marks.

### 4. EloSystem

Team-strength ratings used as input to the game-winner model.

| Parameter | Value |
|---|---|
| Initial rating | 1500 |
| K-factor | 12.3 |
| Home advantage | +16.2 points |
| Season regression | 0.285x toward 1500 |
| Margin scaling | FiveThirtyEight formula |

### 5. AFLGameWinnerModel

Predicts match winner using team-level features.

- **Algorithm:** `HistGradientBoostingClassifier` (binary: home win or not)
- **Features:** Elo ratings, aggregated player predictions, team stats (rest days, venue, is_home)
- **Hybrid mode:** Logit-space blend of market prior + ML residual (controlled by `WINNER_HYBRID_*` config)
- **Training:** One row per team per match

---

## Predictions & Output

For each player in a given round, the pipeline produces:

### Scoring Predictions
| Column | Description |
|---|---|
| `predicted_goals` | Expected goals (ensemble mean) |
| `predicted_behinds` | Expected behinds (ensemble mean) |
| `predicted_score` | `goals * 6 + behinds` |
| `p_scorer` | P(player kicks >= 1 goal) |
| `conf_lower_gl` | 80% confidence interval lower bound |
| `conf_upper_gl` | 80% confidence interval upper bound |
| `top_features` | Top 3 features driving this prediction |

### Disposal Thresholds
| Column | Description |
|---|---|
| `predicted_disposals` | Expected disposal count |
| `p_10plus_disp` | P(disposals >= 10) |
| `p_15plus_disp` | P(disposals >= 15) |
| `p_20plus_disp` | P(disposals >= 20) |
| `p_25plus_disp` | P(disposals >= 25) |
| `p_30plus_disp` | P(disposals >= 30) |

### Marks Thresholds
| Column | Description |
|---|---|
| `predicted_marks` | Expected marks count |
| `p_2plus_mk` through `p_10plus_mk` | P(marks >= k) for k in {2..10} |

### Game Winner
| Column | Description |
|---|---|
| `home_win_prob` | P(home team wins) |
| `predicted_winner` | Predicted winning team |

Output files: `data/predictions/round_N_predictions.csv` and `round_N_thresholds.csv`

---

## Backtesting & Learning

### Walk-Forward Backtest (`--backtest`)
For each round in a season:
1. Train on all data prior to that round
2. Predict that round
3. Compare predictions to actual outcomes
4. Save everything to the LearningStore

### LearningStore (`store.py`)
Persistent, run-versioned, append-only storage in `data/sequential/`:

| Subdirectory | Contents |
|---|---|
| `predictions/` | Per-round predicted values (run-versioned) |
| `outcomes/` | Per-round actual results |
| `diagnostics/` | Per-round error metrics |
| `calibration/` | Calibration state (predicted vs actual bucket probabilities) |
| `archetypes/` | GMM cluster assignments over time |
| `concessions/` | Opponent concession profiles per round |
| `analysis/` | Round-level JSON analysis summaries |
| `game_predictions/` | Game winner predictions |

### Calibration System
- Walk-forward isotonic calibration (the key BSS lever)
- Needs ~1500+ samples to stabilize; applying too early hurts BSS
- Monitors P(X >= k) for all goal, disposal, and marks thresholds
- Min 100 samples before fitting isotonic calibrator
- Refits every 5 rounds in sequential mode

### Sequential Learning (`--sequential`)
Calibration-aware version of backtest that updates calibration state between rounds, simulating live deployment.

### Experiment Comparison (`compare_experiments.py`)
Loads experiment JSONs from `data/experiments/` and prints BSS comparison table across all disposal thresholds.

---

## Analysis & Experiments

### Round Analysis (`analysis.py`)
Generates per-round JSON summaries covering:
- Summary statistics (MAE, Brier, log-loss)
- Calibration curves (predicted vs actual probability)
- Hot/cold player identification
- Biggest misses and hits
- Weather impact analysis
- Team-level breakdowns
- Model improvement tracking
- Streak analysis
- Miss classification (why the model was wrong)
- Archetype drift (cluster changes over time)
- Concession drift (opponent defensive profile changes)

### Experiment Scripts

| Script | Purpose |
|---|---|
| `experiment_ensemble.py` | Compare single-model HistGBT vs multi-model ensemble. Measures Brier score, AUC, MAE with paired t-tests. |
| `experiment_ensemble_fast.py` | Lightweight version of ensemble comparison (HistGBT only) |
| `experiment_disposal_dist.py` | Compare Poisson vs Gaussian vs Negative Binomial for disposal threshold probabilities. |
| `experiment_teammate_abc.py` | Analyze teammate assist/block/create contributions to scoring. |
| `compare_experiments.py` | Load and compare experiment JSONs side-by-side with BSS at all thresholds. |

### Analysis Utilities

| Script | Purpose |
|---|---|
| `build_baseline.py` | Compile baseline v2/v3 summary reports from backtest results into JSON |
| `build_baseline_v21.py` | Compile baseline v2.1 (with odds integration) report |
| `feature_importance_analysis.py` | Compute and rank GBT feature importances per model |
| `investigate_hitrate.py` | Deep-dive into scorer classifier accuracy by role, team, venue |
| `disposal_distribution_analysis.py` | Empirical vs predicted disposal distribution comparison |
| `disposal_distribution_comparison.py` | Side-by-side distribution analysis across methods |
| `study_early_ladder_momentum.py` | Correlate early-season ladder position with player performance |
| `integrate_odds.py` | Merge bookmaker and Betfair match odds into the feature pipeline |
| `integrate_player_odds.py` | Parse Betfair player-level markets (disposals, FGS, 2/3 goals) |

### Validation (`validate.py`)

Multi-stage validation:

| Stage | Checks |
|---|---|
| `validate_cleaned()` | Required columns present, no duplicate (player, team, match_id), non-negative GL/BH, valid dates, min 2 players per match |
| `validate_features()` | All feature columns exist, no inf values, no all-NaN columns, non-negative targets |
| `validate_predictions()` | Required output columns present, non-negative predictions, p_scorer in [0, 1] |
| `validate_umpires()` | Umpire data integrity checks |

---

## Configuration Reference

All tunable parameters live in `config.py`.

### Era Weights
```
2015–2019  →  0.4x
2020–2022  →  0.7x
2023–2024  →  0.9x
2025–2026  →  1.0x
```

### Rule Eras
```
Era 1: 2015                  Era 5: 2021–2022
Era 2: 2016–2018             Era 6: 2023–2025
Era 3: 2019                  Era 7: 2026 (no centre bounces, 5-man interchange,
Era 4: 2020 (COVID — 80%)            top-10 wildcard finals)
```

### Feature Engineering
| Parameter | Value |
|---|---|
| `ROLLING_WINDOWS` | [3, 5, 10] |
| `VENUE_LOOKBACK_YEARS` | 3 |
| `MATCHUP_MIN_MATCHES` | 3 |
| `N_ARCHETYPES` | 6 |
| `RECENCY_DECAY_HALF_LIFE` | 365 days |
| `UMPIRE_LOOKBACK_MATCHES` | 20 |
| `COACH_MIN_GAMES` | 10 |
| `TEAM_VENUE_LOOKBACK_YEARS` | 5 |

### Model Hyperparameters
| Parameter | Value |
|---|---|
| `ENSEMBLE_WEIGHTS` | Poisson 80%, GBT 20% |
| `DISPOSAL_DISTRIBUTION` | Gaussian |
| `CALIBRATION_METHOD` | Isotonic |
| `ISOTONIC_MIN_SAMPLES` | 100 |
| `ISOTONIC_REFIT_INTERVAL` | 5 rounds |

### Training
| Parameter | Value |
|---|---|
| `HISTORICAL_START_YEAR` | 2015 |
| `HISTORICAL_END_YEAR` | 2026 |
| `CURRENT_SEASON_YEAR` | 2026 |
| `VALIDATION_YEAR` | 2024 |
| `MIN_PLAYER_MATCHES` | 5 |
| `BACKTEST_TRAIN_MIN_YEARS` | 3 |

### Thresholds
**Goals:** 1+, 2+, 3+
**Disposals:** 10+, 15+, 20+, 25+, 30+
**Marks:** 2+, 3+, 4+, 5+, 6+, 7+, 8+, 9+, 10+

---

## File Map

```
AFL/
├── pipeline.py                         # CLI orchestrator (all stages)
├── clean.py                            # Data loading, cleaning, rate normalization
├── features.py                         # 252-feature engineering pipeline (stages A-W)
├── model.py                            # AFLScoringModel, AFLDisposalModel, AFLMarksModel,
│                                       #   EloSystem, AFLGameWinnerModel, CalibratedPredictor
├── config.py                           # All tunable parameters and paths
├── store.py                            # LearningStore — persistent round-by-round records
├── validate.py                         # Data validation (cleaned, features, predictions, umpires)
├── analysis.py                         # Round-by-round analysis engine
├── weather.py                          # Open-Meteo API integration (25 venues, wind direction)
├── scraper.py                          # AFL Tables scraper (matches, stats, umpires, profiles)
├── scrape_footywire.py                 # FootyWire advanced stats scraper
├── player.py                           # Player-level utilities
├── integrate_odds.py                   # Bookmaker + Betfair match odds merging
├── integrate_player_odds.py            # Betfair player-level market odds (91.8% name match)
├── compare_experiments.py              # Experiment A/B comparison tables
│
├── Experiment & analysis scripts:
│   ├── experiment_ensemble.py          # Single vs multi-model ensemble comparison
│   ├── experiment_ensemble_fast.py     # Fast ensemble comparison (HistGBT only)
│   ├── experiment_disposal_dist.py     # Poisson vs Gaussian vs NegBin experiment
│   ├── experiment_teammate_abc.py      # Teammate assist/block/create analysis
│   ├── build_baseline.py              # Baseline report builder (v2/v3)
│   ├── build_baseline_v21.py          # Baseline v2.1 with odds
│   ├── feature_importance_analysis.py # Feature ranking
│   ├── investigate_hitrate.py         # Scorer classifier deep-dive
│   ├── disposal_distribution_analysis.py
│   ├── disposal_distribution_comparison.py
│   └── study_early_ladder_momentum.py
│
├── app/
│   ├── ABOUT.md                        # Detailed pipeline documentation
│   ├── generate_pdf.py                 # PDF report generation
│   ├── rosters_2026.json               # Current season rosters
│   └── round_*_2026.csv                # Per-round fixture/prediction CSVs
│
├── data/
│   ├── base/                           # Cleaned parquets (gitignored, regenerable)
│   ├── features/
│   │   ├── feature_matrix.parquet      # 252-feature matrix
│   │   └── feature_columns.json        # Feature name list
│   ├── models/                         # Trained model weights (.pkl, gitignored) + metadata JSONs
│   ├── predictions/                    # Per-round prediction CSVs (curated snapshots only)
│   ├── experiments/                    # Experiment result JSONs (0–4)
│   ├── fixtures/                       # Round fixtures + rosters
│   ├── footywire/                      # FootyWire advanced stats CSVs (2015–2025)
│   ├── umpires/                        # Umpire assignment CSVs (2015–2025)
│   ├── tuning/                         # Optuna best params JSONs
│   ├── sequential/                     # Sequential learning output (run-versioned, gitignored)
│   └── backtest/                       # Backtest output (gitignored)
│
├── baseline_v3.2.json                  # Current baseline summary
└── momentum_study_2015_2025.json       # Momentum study results
```
