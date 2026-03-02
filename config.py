"""
AFL Prediction Pipeline — Configuration
========================================
All tunable parameters in one place. Import from here, never hardcode
magic numbers in feature engineering or model code.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BASE_STORE_DIR = DATA_DIR / "base"         # Layer 1-3: typed parquets
CLEANED_DIR = DATA_DIR / "cleaned"          # Legacy (kept for migration)
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
FIXTURES_DIR = DATA_DIR / "fixtures"
LEARNING_DIR = DATA_DIR / "learning"        # Layer 4: persistent learning store

# Scraper output subdirectories (matches scraper.py layout)
PLAYER_STATS_DIR = DATA_DIR / "player_stats"
PLAYER_DETAILS_DIR = DATA_DIR / "player_details"
SCORING_DIR = DATA_DIR / "scoring"
MATCHES_DIR = DATA_DIR / "matches"

# ---------------------------------------------------------------------------
# Era weights for recency — recent seasons matter more
# ---------------------------------------------------------------------------
ERA_WEIGHTS = {
    (2015, 2019): 0.4,
    (2020, 2022): 0.7,
    (2023, 2024): 0.9,
    (2025, 2025): 1.0,
}
RECENCY_DECAY_HALF_LIFE = 365  # days — used in exponential decay

# ---------------------------------------------------------------------------
# Season era / rule regime mapping
# ---------------------------------------------------------------------------
ERA_MAP = {
    2015: 1,                    # 120 interchange cap, substitute rule
    2016: 2, 2017: 2, 2018: 2, # 90 interchange cap, no substitute
    2019: 3,                    # 6-6-6 starting positions introduced, major tactical shift
    2020: 4,                    # COVID — 16-min quarters, hubs, compressed schedule
    2021: 5, 2022: 5,          # 75 interchange cap, medical substitute, normal quarters
    2023: 6, 2024: 6, 2025: 6, # General substitute, stand rule matured
}
COVID_SEASON_YEAR = 2020
COVID_QUARTER_LENGTH_RATIO = 0.8  # 16/20 minutes
CURRENT_PREDICTION_ERA = 6  # era for upcoming/future predictions

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [3, 5, 10]
VENUE_LOOKBACK_YEARS = 3
MATCHUP_MIN_MATCHES = 3
ENABLER_COUNT = 3  # top-N teammate enablers per player
N_ARCHETYPES = 6    # GMM archetype clusters (Forward, Midfielder, Ruck, Defender, Tagger, Utility)

# ---------------------------------------------------------------------------
# Role classification thresholds (from last 10 matches averages)
# Lowered from original to classify more players (previously all got "general")
# ---------------------------------------------------------------------------
RUCK_HO_THRESHOLD = 10
KEY_FORWARD_GL_THRESHOLD = 0.7
KEY_FORWARD_MI_THRESHOLD = 1.0
SMALL_FORWARD_GL_THRESHOLD = 0.3
SMALL_FORWARD_IF_THRESHOLD = 2
KEY_DEFENDER_RB_THRESHOLD = 2.5
KEY_DEFENDER_ONE_PCT_THRESHOLD = 2.5
MIDFIELDER_DI_THRESHOLD = 18
MIDFIELDER_CL_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
ENSEMBLE_WEIGHTS = {"poisson": 0.4, "gbt": 0.6}
DISPOSAL_DISTRIBUTION = "gaussian"  # 'poisson', 'gaussian', or 'negbin' — Gaussian wins at all thresholds (2025 backtest)

# Upper-tail correction for Gaussian disposal probabilities.
# Applied only to thresholds in DISPOSAL_UPPER_TAIL_THRESHOLDS.
DISPOSAL_UPPER_TAIL_ENABLED = True
DISPOSAL_UPPER_TAIL_THRESHOLDS = [25, 30]
DISPOSAL_UPPER_TAIL_STD_MULTIPLIER = 1.2
DISPOSAL_UPPER_TAIL_SKEW_ALPHA = 2.0
# Optional probability shaping for 30+ only:
# - scale lifts underpredicted bulk probabilities
# - cap prevents extreme overconfident tail probabilities
DISPOSAL_30PLUS_PROB_SCALE = 1.3
DISPOSAL_30PLUS_PROB_CAP = 0.45

GBT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}

# Lighter params for backtest (faster iteration, similar diagnostic value)
GBT_PARAMS_BACKTEST = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 25,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}

# HistGradientBoosting params for backtest (10-50x faster than standard GBT)
HIST_GBT_PARAMS_BACKTEST = {
    "max_iter": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 25,
    "random_state": RANDOM_SEED,
}

# Multi-model ensemble: average predictions from HistGBT variants with different
# hyperparameters. All use histogram-based splitting for speed. Diversity comes
# from different depth/learning-rate/iteration tradeoffs.
MULTI_MODEL_ENSEMBLE = True  # False = single HistGBT (original), True = 3-model average

# HistGBT-deep: deeper trees, fewer iterations, higher learning rate
HIST_GBT_DEEP_PARAMS_BACKTEST = {
    "max_iter": 60,
    "max_depth": 6,
    "learning_rate": 0.08,
    "min_samples_leaf": 40,
    "random_state": RANDOM_SEED,
}

# HistGBT-wide: shallower trees, more iterations, lower learning rate
HIST_GBT_WIDE_PARAMS_BACKTEST = {
    "max_iter": 200,
    "max_depth": 2,
    "learning_rate": 0.03,
    "min_samples_leaf": 15,
    "random_state": RANDOM_SEED,
}

POISSON_PARAMS = {
    "alpha": 0.5,        # reduced from 1.0 — was over-regularized, pulling predictions toward zero
    "max_iter": 1000,
}

# ---------------------------------------------------------------------------
# Streak feature parameters
# ---------------------------------------------------------------------------
STREAK_DECAY = 0.85          # Exponential decay factor for weighted streaks
HOT_THRESHOLD = 1.5          # form_ratio above this = hot
COLD_THRESHOLD = 0.5         # form_ratio below this = cold
STREAK_BROKE_MIN = 3         # Minimum streak length to trigger "just broke"

# ---------------------------------------------------------------------------
# Miss classification thresholds
# ---------------------------------------------------------------------------
MISS_ERROR_THRESHOLD = 1.5          # abs goal error >= this triggers classification
MISS_OPPORTUNITY_PCT_PLAYED = 60    # pct_played < this → opportunity miss
MISS_TEAM_ENV_DEVIATION = 0.25      # team scored 25%+ fewer/more goals than avg → team env
MISS_MATCHUP_CONCESSION_DEV = 0.3   # opponent conceded 30%+ differently from profile → matchup

# ---------------------------------------------------------------------------
# Probability calibration evaluation
# ---------------------------------------------------------------------------
GOAL_THRESHOLDS = {"1plus_goals": 1, "2plus_goals": 2, "3plus_goals": 3}
DISPOSAL_THRESHOLDS_EVAL = {"10plus_disp": 10, "15plus_disp": 15, "20plus_disp": 20, "25plus_disp": 25, "30plus_disp": 30}
CALIBRATION_N_BUCKETS = 10
CALIBRATION_MIN_BUCKET_SIZE = 5

# ---------------------------------------------------------------------------
# Model targets (future: add "disposals" entry)
# ---------------------------------------------------------------------------
MODEL_TARGETS = {
    "scoring": {
        "targets": ["GL", "BH"],
        "primary_target": "GL",
        "description": "Goals and behinds prediction",
    },
    "disposals": {
        "targets": ["DI"],
        "primary_target": "DI",
        "description": "Disposals prediction",
        "thresholds": [20, 25, 30],
    },
}

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
VALIDATION_YEAR = 2024
MIN_PLAYER_MATCHES = 5
HISTORICAL_START_YEAR = 2015
HISTORICAL_END_YEAR = 2025
CURRENT_SEASON_YEAR = 2025

# Backtesting
BACKTEST_TRAIN_MIN_YEARS = 3
BACKTEST_DIR = DATA_DIR / "backtest"

# Sequential learning
SEQUENTIAL_DIR = DATA_DIR / "sequential"
SEQUENTIAL_YEAR = 2025
CALIBRATION_MIN_SAMPLES = 30       # min predictions per bucket before applying adjustment
CALIBRATION_MAX_ADJUSTMENT = 0.3   # cap adjustment to prevent wild swings
DISPOSAL_THRESHOLDS = [10, 15, 20, 25, 30]
GOAL_DISTRIBUTION_MAX_K = 7       # compute P(0), P(1), ..., P(6), P(7+)

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required data directories."""
    for d in [DATA_DIR, RAW_DIR, BASE_STORE_DIR, CLEANED_DIR, FEATURES_DIR,
              MODELS_DIR, PREDICTIONS_DIR, FIXTURES_DIR, BACKTEST_DIR,
              LEARNING_DIR, SEQUENTIAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # Sequential learning subdirectories
    for subdir in ["predictions", "outcomes", "diagnostics", "analysis",
                   "game_predictions", "calibration", "archetypes", "concessions"]:
        (SEQUENTIAL_DIR / subdir).mkdir(parents=True, exist_ok=True)
