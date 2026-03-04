# AFL Player Prediction Pipeline

A machine-learning pipeline that predicts individual player scoring (goals, behinds), disposal counts, and marks for Australian Football League matches. It scrapes historical data from AFL Tables and FootyWire, integrates weather and betting odds, engineers 252 features, trains multi-stage ensemble models, and generates per-player probability distributions for upcoming rounds.

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

## Data Sources

| Source | Type | Content |
|---|---|---|
| **AFL Tables** | Web scrape | Player stats, details, scoring, matches, umpires (2015–2025) |
| **FootyWire** | Web scrape | Advanced stats: ED, DE%, CCL, SCL, TO, MG, SI, ITC, T5, TOG% (2015–2025) |
| **Open-Meteo** | API | Historical weather for 25 venues (temp, rain, wind, humidity, pressure) |
| **Bookmaker** | XLSX | Match odds 2009–2025 (open/close home/away/line/total) |
| **Betfair Exchange** | CSV | Match odds 2021–2025 (back/lay at first bounce) |
| **Betfair Player** | CSV | Player markets: disposals, first goal scorer, 2/3+ goals (23.6K rows) |

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
```

## Feature Engineering (252 Features)

Stages A–W, built in `features.py`:

| Stage | Category | Count | Key Features |
|---|---|---|---|
| A | Career & Age | 5 | `age_years`, `career_games_pre`, `career_goal_avg_capped` |
| B | Rolling Averages | ~74 | 3/5/10-game windows for raw + rate stats, EWM, streaks, volatility, trend |
| C | Venue | 4 | `player_gl_at_venue_avg`, `venue_avg_goals_per_team` |
| D | Opponent Defense | 13 | `opp_goals_conceded_avg_5/10`, `opp_defender_strength_score` |
| E | Defender Matchups | ~6 | Player-vs-opponent historical stats |
| F | Team Context | 8 | `team_goals_avg_5`, `team_win_pct_5`, `player_goal_share_5` |
| G | Scoring Patterns | 6 | Quarter-level goal distribution, multi-goal rate |
| H | Role Classification | 2 | `forward_score`, `player_role` |
| I | Teammate Features | 3 | `teammate_enabler_count`, `teammate_scoring_avg` |
| J | Interaction Terms | ~10 | Form × defense, home × scoring, hot × weak defense |
| K | GMM Archetypes | 8 | 6-cluster soft assignments, archetype concession profiles |
| L | Weather | ~19 | Temperature, rain, wind severity, difficulty score, slippery conditions |
| M | Sample Weights | — | Era-based recency weights (not a feature, used in training) |
| N | Umpire | ~5 | Umpire scoring tendencies, free kick rates |
| O | Coach | ~5 | Coach historical scoring rates, H2H matchups |
| P | Player Physical | ~6 | Height, weight, BMI, age interactions with archetype |
| Q | Career Splits | ~4 | Home/away, day/night career performance splits |
| R | Team Venue | ~4 | Team-level venue scoring history |
| S | Market Odds | ~8 | Implied probabilities, market total score, odds movement |
| T | Venue Elevation | ~2 | Ground elevation, altitude scoring adjustment |
| U | Disposal Interactions | ~8 | TOG rolling, disposal share, weather × disposal interactions |
| V | FootyWire Advanced | — | **DISABLED** (A/B test showed BSS regression) |
| W | CBA Features | — | Stub for DFS Australia data (future) |

## Models

### 1. AFLScoringModel (Goals + Behinds)

Two-stage ensemble addressing zero-inflation (~68% of player-matches = 0 goals):

- **Stage 1:** Binary scorer classifier — P(player scores >= 1 goal)
- **Stage 2:** Poisson (80%) + GBT (20%) ensemble on scorers only (GL >= 1)
- **Behinds:** Same ensemble, trained on all players

### 2. AFLDisposalModel

Single-stage Poisson + GBT regressor. Gaussian distribution for threshold probabilities (P(X >= k) for k in {10, 15, 20, 25, 30}). Isotonic calibration is the key lever (BSS 30.9% at 15+).

### 3. AFLMarksModel

Single-stage regressor for marks predictions. Threshold probabilities for k in {2, 3, 4, 5, 6, 7, 8, 9, 10}.

### 4. AFLGameWinnerModel

- **Algorithm:** HistGradientBoostingClassifier (binary: home win)
- **Hybrid mode:** Logit-space blend of market prior + ML residual
- **Features:** Elo ratings, aggregated player predictions, team stats

### 5. EloSystem

Team-strength ratings (K=12.3, home advantage=16.2, season regression=0.285).

## Predictions & Output

For each player in a given round:

| Category | Columns |
|---|---|
| Scoring | `predicted_goals`, `predicted_behinds`, `predicted_score`, `p_scorer`, `conf_lower_gl`, `conf_upper_gl` |
| Disposals | `predicted_disposals`, `p_10plus_disp` through `p_30plus_disp` |
| Marks | `predicted_marks`, `p_2plus_mk` through `p_10plus_mk` |
| Game Winner | `home_win_prob`, `predicted_winner` |

Output: `data/predictions/round_N_predictions.csv` and `round_N_thresholds.csv`

## Backtesting & Learning

### Walk-Forward Backtest (`--backtest`)
For each round: train on all prior data, predict, compare to actuals, save to LearningStore.

### LearningStore (`store.py`)
Run-versioned, append-only storage in `data/sequential/predictions/{year}/run_{id}/`.

### Sequential Learning (`--sequential`)
Calibration-aware backtest with walk-forward isotonic calibration (~1500+ samples needed to stabilize).

### Experiment Comparison (`compare_experiments.py`)
Loads experiment JSONs from `data/experiments/` and prints BSS comparison across disposal thresholds.

## Configuration

All parameters in `config.py`. Key settings:

| Parameter | Value |
|---|---|
| Historical range | 2015–2026 |
| Current season | 2026 |
| Validation year | 2024 |
| Rolling windows | [3, 5, 10] |
| GMM archetypes | 6 |
| Ensemble weights | Poisson 80%, GBT 20% |
| Disposal distribution | Gaussian |
| Calibration | Isotonic (min 100 samples, refit every 5 rounds) |
| Era 7 (2026) | No centre bounces, 5-man interchange, top-10 wildcard finals |

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
├── weather.py                          # Open-Meteo API integration (25 venues)
├── scraper.py                          # AFL Tables scraper (matches, stats, umpires, profiles)
├── scrape_footywire.py                 # FootyWire advanced stats scraper
├── player.py                           # Player-level utilities
├── integrate_odds.py                   # Bookmaker + Betfair match odds merging
├── integrate_player_odds.py            # Betfair player-level market odds (91.8% name match)
├── compare_experiments.py              # Experiment A/B comparison tables
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
│   ├── models/                         # Trained model weights + metadata JSONs
│   ├── predictions/                    # Per-round prediction CSVs
│   ├── experiments/                    # Experiment result JSONs (0–4)
│   ├── fixtures/                       # Round fixtures + rosters
│   ├── footywire/                      # FootyWire advanced stats CSVs (2015–2025)
│   ├── umpires/                        # Umpire assignment CSVs (2015–2025)
│   ├── tuning/                         # Optuna best params JSONs
│   ├── sequential/                     # Sequential learning output (gitignored)
│   └── backtest/                       # Backtest output (gitignored)
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
├── baseline_v3.2.json                  # Current baseline summary
└── momentum_study_2015_2025.json       # Momentum study results
```

## Repo Hygiene

- `data/predictions/` keeps curated benchmark snapshots only (rounds 1, 2, 10, 13, 22)
- `data/tuning/` keeps canonical `best_params_*.json` files only
- `app/*.pdf` and `Fixtures/` are gitignored
- LearningStore uses run-versioned directories — always use latest run for metrics

## ML Pipeline Artifact Notes

LearningStore artifacts are written to run-versioned directories:
- `predictions/outcomes/diagnostics/game_predictions/analysis`: stored under `.../{year}/run_{id}/...`
- Calibration state: `calibration/run_{id}/calibration_state.parquet`
- Use `--run-id <id>` to force a specific run id
- Sequential mode keeps calibration by default; use `--reset-calibration` to clear
