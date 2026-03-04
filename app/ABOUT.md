# AFL Player Prediction Pipeline

A machine-learning pipeline that predicts individual player scoring (goals, behinds) and disposal counts for Australian Football League matches. It scrapes historical data from AFL Tables, engineers 182 features, trains a multi-stage ensemble model, and generates per-player probability distributions for upcoming rounds.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Sources & APIs](#data-sources--apis)
3. [Raw Data Points](#raw-data-points)
4. [Pipeline Stages](#pipeline-stages)
5. [Feature Engineering (182 Features)](#feature-engineering-182-features)
6. [Models](#models)
7. [Predictions & Output](#predictions--output)
8. [Backtesting & Learning](#backtesting--learning)
9. [Analysis & Experiments](#analysis--experiments)
10. [Configuration Reference](#configuration-reference)
11. [File Map](#file-map)

---

## Architecture Overview

```
AFL Tables (web scrape)          Open-Meteo API          Betting Odds (XLSX/CSV)
        │                              │                          │
        ▼                              ▼                          ▼
   Raw CSVs ──────────────────► clean.py ◄────────────────────────┘
   (player_stats, player_details,       │
    scoring, matches per year)          ▼
                                 player_games.parquet  (101K rows, 71 cols)
                                 matches.parquet       (2,258 rows, 38 cols)
                                 team_matches.parquet  (4,516 rows, 24 cols)
                                        │
                                        ▼
                                  features.py
                                        │
                                        ▼
                                 feature_matrix.parquet (182 features, float32)
                                        │
                              ┌─────────┼─────────┐
                              ▼         ▼         ▼
                        Scoring     Disposal   Game Winner
                         Model       Model       Model
                              │         │         │
                              ▼         ▼         ▼
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

The scraper fetches season-by-season HTML pages from **afltables.com** and extracts four categories of data into year-specific CSVs:

| Category | Output Directory | Content |
|---|---|---|
| Player stats | `data/player_stats/` | Per-game box scores (kicks, marks, handballs, goals, etc.) |
| Player details | `data/player_details/` | Age, career games, career goals |
| Scoring events | `data/scoring/` | Quarter-by-quarter goal/behind breakdowns |
| Match metadata | `data/matches/` | Date, venue, scores, attendance, round info |

**Years covered:** 2015–2025 (configurable via `--start` / `--end`)

### 2. Open-Meteo Historical Weather API

| Field | Value |
|---|---|
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Authentication | None (free tier) |
| Rate limit | 0.5s between calls |
| Caching | File-based JSON cache in `data/base/weather_cache/` |

**Hourly variables fetched:**
- `temperature_2m`, `apparent_temperature`, `precipitation`, `rain`
- `wind_speed_10m`, `wind_gusts_10m`, `relative_humidity_2m`, `dew_point_2m`
- `cloud_cover`, `surface_pressure`

**25 venues mapped** with lat/lon coordinates — MCG, Docklands, Adelaide Oval, Perth Stadium, Gabba, Carrara, SCG, Kardinia Park, Sydney Showground, Subiaco, York Park, Bellerive Oval, Manuka Oval, Marrara Oval, Eureka Stadium, Cazaly's Stadium, Traeger Park, Norwood Oval, Stadium Australia, Jiangwan Stadium (Shanghai), Summit Sports Park, Barossa Oval, Hands Oval, Riverway Stadium, Wellington (NZ).

### 3. Betting Odds (Two Sources)

**Bookmaker (Source 2):** `afl_Source_2.xlsx` — 3,353 matches (2009–2025)
- Pre-game only: `home_odds_open`, `home_odds_close`, `away_odds_open`, `away_odds_close`, `home_line_open`, `home_line_close`, `total_score_open`, `total_score_close`

**Betfair Exchange:** `AFL_YYYY_Match_Odds.csv` (2021–2025)
- Pre-game only: `BEST_BACK_FIRST_BOUNCE`, `BEST_LAY_FIRST_BOUNCE`
- All post-game and in-play data explicitly excluded

**Derived odds features:**
`market_home_implied_prob`, `market_away_implied_prob`, `market_handicap`, `market_total_score`, `market_confidence`, `odds_movement_home`, `odds_movement_line`, `betfair_home_implied_prob`

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
python pipeline.py --clean                                  # Normalize → parquets
python pipeline.py --features                               # Engineer 182 features
python pipeline.py --train                                  # Train scoring model (goals + behinds)
python pipeline.py --train-disposals                        # Train disposal model
python pipeline.py --train-winner                           # Train game-winner model
python pipeline.py --predict --round N [--year YYYY]        # Predict a specific round
python pipeline.py --evaluate                               # Evaluate on validation year (2024)
python pipeline.py --backtest [--year YYYY]                 # Walk-forward per-round backtest
python pipeline.py --diagnose [--year YYYY]                 # Breakdown of backtest results
python pipeline.py --sequential [--year YYYY]               # Calibration-aware sequential learning
python pipeline.py --update                                 # Scrape + rebuild + predict (current season)
python pipeline.py --reset-calibration                      # Clear calibration state
```

### Stage Details

| Stage | Input | Output | Description |
|---|---|---|---|
| **Scrape** | AFL Tables HTML | Year CSVs in `data/player_stats/`, `data/scoring/`, etc. | Fetches historical data per season |
| **Clean** | Raw CSVs | `player_games.parquet` (101K rows, 71 cols), `matches.parquet` (2,258 rows), `team_matches.parquet` (4,516 rows) | Joins, normalizes, computes rate columns, fixes finals labeling |
| **Features** | Cleaned parquets | `feature_matrix.parquet` (182 features) + `feature_columns.json` | Rolling averages, archetypes, opponent profiles, venue, weather, odds |
| **Train** | Feature matrix | `goals_*.pkl`, `behinds_*.pkl`, `scorer_clf.pkl`, `scaler.pkl` | Fits two-stage scoring ensemble |
| **Predict** | Trained models + feature matrix | `round_N_predictions.csv`, `round_N_thresholds.csv` | Per-player distributions for an upcoming round |
| **Backtest** | Feature matrix | LearningStore entries per round | Walk-forward: train on all prior data, predict each round, save outcomes |

---

## Feature Engineering (182 Features)

After building and pruning (22 redundant features removed, including `is_covid_season` and `quarter_length_ratio` which had r=1.0 with `era_4`), the final matrix has **182 features**.

### A. Career & Age (5)
`age_years`, `age_squared`, `career_games_pre`, `career_goals_pre`, `career_goal_avg_capped`

### B. Recency-Weighted Rolling Averages (~74)

**Raw stat rolling means** over 3/5/10-game windows:
`player_gl_avg_3`, `player_gl_avg_5`, `player_gl_avg_10`, `player_di_avg_3`, `player_mk_avg_3`, `player_tk_avg_3`, `player_if50_avg_3`, `player_cl_avg_3`, `player_ho_avg_3`, `player_ga_avg_3`, `player_mi_avg_3`, `player_cm_avg_3`, `player_cp_avg_3`, `player_ff_avg_3`, `player_rb_avg_3`, `player_one_pct_avg_3`, etc.

**Rate-normalized rolling** (3/5/10 windows):
`player_gl_rate_avg_3`, `player_bh_rate_avg_3`, `player_di_rate_avg_3`, `player_mk_rate_avg_3`, `player_tk_rate_avg_3`, `player_if50_rate_avg_3`, `player_cp_rate_avg_3`, etc.

**Exponentially weighted means** (span=5):
`player_gl_ewm_5`, `player_bh_ewm_5`, `player_mi_ewm_5`, `player_if50_ewm_5`, `player_di_ewm_5`, `player_mk_ewm_5`, plus rate versions

**Accuracy** (GL / (GL + BH)):
`player_accuracy_3`, `player_accuracy_5`, `player_accuracy_10`

**Streaks & form:**
`player_gl_streak`, `player_gl_streak_weighted` (decay=0.85), `player_gl_cold_streak`, `player_form_ratio`, `player_is_hot`, `player_is_cold`, `player_streak_just_broke`

**Volatility & trend:**
`player_gl_volatility_5`, `player_gl_trend_5`, `player_di_volatility_5`, `player_di_trend_5`, `player_gl_rate_volatility_5`, `player_gl_rate_trend_5`, `player_di_rate_volatility_5`, `player_di_rate_trend_5`

**Season cumulative:**
`season_goals_total`, `season_disposals_total`, `season_goals_rate_avg`, `season_disposals_rate_avg`

**Other:**
`days_since_last_match`, `is_returning_from_break`, `player_ki_hb_ratio_3`, `player_ki_hb_ratio_5`

### C. Venue Features (4)
`player_gl_at_venue_avg`, `player_bh_at_venue_avg`, `player_gl_venue_diff`, `venue_avg_goals_per_team`

### D. Opponent Defense (13)
`opp_goals_conceded_avg_5`, `opp_goals_conceded_avg_10`, `player_vs_opp_gl_avg`, `player_vs_opp_games`, `player_vs_opp_gl_diff`, `opp_disp_conceded_avg_5`, `opp_disp_conceded_avg_10`, `opp_contested_poss_diff_5`, `opp_key_defenders_count`, `opp_defender_strength_score`

### E. Team Context (8)
`team_goals_avg_5`, `team_goals_avg_10`, `team_if_avg_5`, `team_cl_avg_5`, `team_clearance_dominance_5`, `team_mid_quality_score`, `player_goal_share_5`, `team_win_pct_5`, `team_margin_avg_5`

### F. Scoring Patterns (6)
`player_q1_gl_pct`, `player_q2_gl_pct`, `player_q3_gl_pct`, `player_q4_gl_pct`, `player_late_scorer_pct`, `player_multi_goal_rate`

### G. Role Classification (2)
`forward_score` — derived from goals/inside-50s ratio
`player_role` — categorical: ruck, key_forward, small_forward, key_defender, midfielder, general

### H. Teammate Features (3)
`teammate_enabler_count`, `teammate_scoring_avg`, `interact_team_form_share`

### I. Interaction Terms (8)
`interact_player_vs_opp_defense`, `interact_form_vs_defense`, `interact_home_scoring`, `interact_venue_boost`, `interact_hot_vs_weak_defense`, `interact_streak_forward`, `interact_disp_vs_contested`, `interact_disp_pace`, `interact_disp_vs_cp_diff`, `interact_mid_supply_forward`

### J. Archetype Features (8)

Uses a **Gaussian Mixture Model (6 clusters)** fitted on player stat profiles to classify players into archetypes: Forward, Midfielder, Ruck, Defender, Tagger, Utility.

**Soft assignments:** `archetype_prob_0` through `archetype_prob_5`

**Archetype concession profiles:** `opp_arch_gl_conceded_avg_5`, `opp_arch_disp_conceded_avg_5` — how many goals/disposals the opponent concedes to players of this archetype

**Disposal ceiling:** `archetype_di_ceiling_ratio`, `archetype_di_ceiling_5`

### K. Game Environment (10)
`game_pace_proxy`, `expected_margin_diff`, `expected_margin_abs`, `ground_length`, `ground_width`, `ground_area`, `ground_shape_ratio`, `is_night_game`, `is_twilight_game`

### L. Era / Rule Regime (4)
`era_2`, `era_3`, `era_4`, `era_6` — one-hot for rule-change eras (6 total: 2015, 2016–18, 2019, 2020-COVID, 2021–22, 2023–25)

### M. Weather Features (19)
`temperature_avg`, `apparent_temperature_avg`, `temperature_range`, `rain_total`, `wind_speed_avg`, `wind_speed_max`, `wind_gusts_max`, `wind_severity`, `humidity_avg`, `dew_point_avg`, `cloud_cover_avg`, `feels_like_delta`, `weather_difficulty_score`, `slippery_conditions`

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
| Poisson | `PoissonRegressor` (alpha=0.5, max_iter=1000) | 40% |
| GBT | `GradientBoostingRegressor` (n_estimators=300, max_depth=4, lr=0.05, min_samples_leaf=20, subsample=0.8) | 60% |

When `MULTI_MODEL_ENSEMBLE=True`, the GBT slot is replaced by an average of 3 `HistGradientBoostingRegressor` variants:
- Base: max_depth=3, max_iter=100
- Deep: max_depth=6, max_iter=60
- Wide: max_depth=2, max_iter=200

**Behinds Regressor:** Same Poisson + GBT ensemble, trained on all players (behinds are more diffuse).

**Confidence intervals:** 80% intervals from a zero-inflated Poisson mixture using `p_scorer` and predicted lambda.

**Training split:** 2015–2023 train, 2024 validation. Baseline: `career_goal_avg_pre`.

### 2. AFLDisposalModel

Single-stage regressor for disposal count predictions.

| Component | Algorithm | Weight |
|---|---|---|
| Poisson | `PoissonRegressor` (alpha=0.5) | 40% |
| GBT | `GradientBoostingRegressor` (same params) | 60% |

**Threshold probabilities:** P(X >= k) for k in {10, 15, 20, 25, 30} disposals.

**Distribution options** (`config.DISPOSAL_DISTRIBUTION`):
- **Gaussian** (default, best performer) — uses predicted mean + std from residuals
- **Poisson** — classic count model
- **Negative Binomial** — overdispersed count model

**Upper-tail corrections** for 25+ and 30+ thresholds: STD multiplier 1.2, skew alpha 2.0, 30+ probability scaled 1.3x with cap at 0.45.

### 3. EloSystem

Team-strength ratings used as input to the game-winner model.

| Parameter | Value |
|---|---|
| Initial rating | 1500 |
| K-factor | 30 |
| Home advantage | +30 points |
| Season regression | 0.5x toward 1500 |
| Margin scaling | FiveThirtyEight formula (larger upsets move ratings more) |

### 4. AFLGameWinnerModel

Predicts match winner using team-level features.

- **Algorithm:** `HistGradientBoostingClassifier` (binary: home win or not)
- **54 features:** Elo ratings, aggregated player predictions (avg p_scorer, avg predicted goals per team), team stats (rest days, venue, is_home)
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
Persistent, run-versioned, append-only storage in `data/learning/`:

| Subdirectory | Contents |
|---|---|
| `predictions/` | Per-round predicted values |
| `outcomes/` | Per-round actual results |
| `diagnostics/` | Per-round error metrics |
| `calibration/` | Calibration state (predicted vs actual bucket probabilities) |
| `archetypes/` | GMM cluster assignments over time |
| `concessions/` | Opponent concession profiles per round |
| `analysis/` | Round-level JSON analysis summaries |
| `game_predictions/` | Game winner predictions |

### Calibration System
- Monitors P(X >= k) for all goal and disposal thresholds
- Computes per-bucket multiplicative corrections (0.5x–2.0x) applied to the Poisson lambda
- Min 30 samples required per bucket before adjustments kick in
- Max adjustment: 0.3 (i.e., 0.7x–1.3x range)

### Sequential Learning (`--sequential`)
Calibration-aware version of backtest that updates calibration state between rounds, simulating live deployment.

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
| `experiment_ensemble.py` | Compare single-model HistGBT vs multi-model ensemble (HistGBT + GBT + ExtraTrees). Measures Brier score, AUC, MAE with paired t-tests. |
| `experiment_ensemble_fast.py` | Lightweight version of ensemble comparison (HistGBT only) |
| `experiment_disposal_dist.py` | Compare Poisson vs Gaussian vs Negative Binomial for disposal threshold probabilities. Measures Brier score, log-loss, hit rates per threshold. |
| `experiment_teammate_abc.py` | Analyze teammate assist/block/create contributions to scoring. Identifies top enablers per player. |

### Analysis Utilities

| Script | Purpose |
|---|---|
| `build_baseline.py` | Compile baseline v2/v3 summary reports from backtest results into JSON |
| `build_baseline_v21.py` | Compile baseline v2.1 (with odds integration) report |
| `feature_importance_analysis.py` | Compute and rank GBT feature importances per model |
| `investigate_hitrate.py` | Deep-dive into scorer classifier accuracy by role, team, venue |
| `disposal_distribution_analysis.py` | Empirical vs predicted disposal distribution comparison |
| `disposal_distribution_comparison.py` | Side-by-side distribution analysis across methods |
| `study_early_ladder_momentum.py` | Correlate early-season ladder position with individual player performance |
| `integrate_odds.py` | Merge bookmaker and Betfair odds into the feature pipeline |

### Validation (`validate.py`)

Three-stage validation:

| Stage | Checks |
|---|---|
| `validate_cleaned()` | Required columns present, no duplicate (player, team, match_id), non-negative GL/BH, valid dates, min 2 players per match |
| `validate_features()` | All feature columns exist, no inf values, no all-NaN columns, non-negative targets |
| `validate_predictions()` | Required output columns present, non-negative predictions, p_scorer in [0, 1] |

---

## Configuration Reference

All tunable parameters live in `config.py`.

### Era Weights
```
2015–2019  →  0.4x
2020–2022  →  0.7x
2023–2024  →  0.9x
2025       →  1.0x
```

### Rule Eras
```
Era 1: 2015          Era 4: 2020 (COVID — 80% quarter length)
Era 2: 2016–2018     Era 5: 2021–2022
Era 3: 2019          Era 6: 2023–2025
```

### Feature Engineering
| Parameter | Value |
|---|---|
| `ROLLING_WINDOWS` | [3, 5, 10] |
| `VENUE_LOOKBACK_YEARS` | 3 |
| `MATCHUP_MIN_MATCHES` | 3 |
| `N_ARCHETYPES` | 6 |
| `RECENCY_DECAY_HALF_LIFE` | 365 days |

### Model Hyperparameters
| Parameter | Value |
|---|---|
| `ENSEMBLE_WEIGHTS` | Poisson 40%, GBT 60% |
| `GBT n_estimators` | 300 (full) / 100 (backtest) |
| `GBT max_depth` | 4 (full) / 3 (backtest) |
| `GBT learning_rate` | 0.05 |
| `GBT min_samples_leaf` | 20 (full) / 25 (backtest) |
| `GBT subsample` | 0.8 |
| `Poisson alpha` | 0.5 |
| `MULTI_MODEL_ENSEMBLE` | True |
| `DISPOSAL_DISTRIBUTION` | Gaussian |

### Streaks & Form
| Parameter | Value |
|---|---|
| `STREAK_DECAY` | 0.85 |
| `HOT_THRESHOLD` | 1.5 |
| `COLD_THRESHOLD` | 0.5 |
| `STREAK_BROKE_MIN` | 3 |

### Training
| Parameter | Value |
|---|---|
| `HISTORICAL_START_YEAR` | 2015 |
| `HISTORICAL_END_YEAR` | 2025 |
| `CURRENT_SEASON_YEAR` | 2025 |
| `VALIDATION_YEAR` | 2024 |
| `MIN_PLAYER_MATCHES` | 5 |
| `BACKTEST_TRAIN_MIN_YEARS` | 3 |

### Thresholds
**Goals:** 1+, 2+, 3+
**Disposals:** 10+, 15+, 20+, 25+, 30+

### Calibration
| Parameter | Value |
|---|---|
| `CALIBRATION_N_BUCKETS` | 10 |
| `CALIBRATION_MIN_BUCKET_SIZE` | 5 |
| `CALIBRATION_MIN_SAMPLES` | 30 |
| `CALIBRATION_MAX_ADJUSTMENT` | 0.3 |

---

## File Map

```
AFL/
├── pipeline.py                         # CLI orchestrator (all stages)
├── clean.py                            # Data loading, cleaning, rate normalization
├── features.py                         # 182-feature engineering pipeline
├── model.py                            # AFLScoringModel, AFLDisposalModel, EloSystem, AFLGameWinnerModel
├── config.py                           # All tunable parameters and paths
├── store.py                            # LearningStore — persistent round-by-round records
├── validate.py                         # Three-stage data validation
├── analysis.py                         # Round-by-round analysis engine
├── weather.py                          # Open-Meteo API integration (25 venues)
├── integrate_odds.py                   # Bookmaker + Betfair odds merging
├── build_baseline.py                   # Baseline report builder (v2/v3)
├── build_baseline_v21.py               # Baseline v2.1 with odds
├── experiment_ensemble.py              # Single vs multi-model ensemble comparison
├── experiment_ensemble_fast.py         # Fast ensemble comparison
├── experiment_disposal_dist.py         # Poisson vs Gaussian vs NegBin experiment
├── experiment_teammate_abc.py          # Teammate contribution analysis
├── feature_importance_analysis.py      # Feature ranking
├── investigate_hitrate.py              # Scorer classifier deep-dive
├── disposal_distribution_analysis.py   # Empirical distribution analysis
├── disposal_distribution_comparison.py # Distribution method comparison
├── study_early_ladder_momentum.py      # Ladder position vs performance study
├── .gitignore
├── README.md
├── app/
│   └── ABOUT.md                        # This file
├── data/
│   ├── base/                           # Cleaned parquets (gitignored, regenerable)
│   ├── features/
│   │   ├── feature_matrix.parquet      # 182-feature matrix
│   │   └── feature_columns.json        # Feature name list
│   ├── cleaned/
│   │   └── player_matches.parquet      # Cleaned player-match data
│   ├── models/
│   │   ├── *.pkl                       # Trained model weights (gitignored)
│   │   ├── model_metadata.json         # Scoring model metadata
│   │   ├── disp_model_metadata.json    # Disposal model metadata
│   │   └── game_winner_metadata.json   # Game winner model metadata
│   ├── predictions/
│   │   ├── round_N_predictions.csv     # Per-player predictions
│   │   └── round_N_thresholds.csv      # Disposal threshold probabilities
│   ├── experiments/                    # Experiment result CSVs
│   ├── learning/                       # LearningStore (gitignored)
│   ├── backtest/                       # Backtest output (gitignored)
│   ├── sequential/                     # Sequential learning output (gitignored)
│   ├── venue_dimensions.json           # Ground length/width for all venues
│   ├── player_stats/                   # Raw scraped CSVs (gitignored)
│   ├── player_details/                 # Raw scraped CSVs (gitignored)
│   ├── scoring/                        # Raw scraped CSVs (gitignored)
│   └── matches/                        # Raw scraped CSVs (gitignored)
├── baseline_v2.json                    # Baseline v2 summary
├── baseline_v2.1_with_odds.json        # Baseline v2.1 summary
├── baseline_v3.json                    # Baseline v3 summary
├── baseline_v3.1.json                  # Baseline v3.1 summary
└── momentum_study_2015_2025.json       # Momentum study results
```
