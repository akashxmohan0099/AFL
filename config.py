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
UMPIRES_DIR = DATA_DIR / "umpires"
COACHES_DIR = DATA_DIR / "coaches"
PLAYER_PROFILES_DIR = DATA_DIR / "player_profiles"
FOOTYWIRE_DIR = DATA_DIR / "footywire"

# ---------------------------------------------------------------------------
# Era weights for recency — recent seasons matter more
# ---------------------------------------------------------------------------
ERA_WEIGHTS = {
    (2015, 2019): 0.4,
    (2020, 2022): 0.7,
    (2023, 2024): 0.9,
    (2025, 2026): 1.0,
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
    2026: 7,                    # No centre bounces, 5-man interchange, top-10 wildcard finals
}
COVID_SEASON_YEAR = 2020
COVID_QUARTER_LENGTH_RATIO = 0.8  # 16/20 minutes
CURRENT_PREDICTION_ERA = 7  # era for upcoming/future predictions

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOWS = [3, 5, 10]
VENUE_LOOKBACK_YEARS = 3
MATCHUP_MIN_MATCHES = 3
ENABLER_COUNT = 3  # top-N teammate enablers per player
N_ARCHETYPES = 6    # GMM archetype clusters (Forward, Midfielder, Ruck, Defender, Tagger, Utility)

# Umpire features
UMPIRE_LOOKBACK_MATCHES = 20
UMPIRE_MIN_GAMES = 10

# Coach features
COACH_MIN_GAMES = 10
COACH_H2H_MIN_MATCHES = 3

# Player physical attributes
PLAYER_PROFILE_HEIGHT_FALLBACK = 186.0   # median AFL height cm
PLAYER_PROFILE_WEIGHT_FALLBACK = 86.0    # median AFL weight kg
PLAYER_PROFILE_AGE_FALLBACK = 25.0

# Career splits
CAREER_SPLIT_MIN_GAMES = 3
CAREER_SPLIT_FEATURES_ENABLED = False  # snapshot career splits are leaky for historical rows unless explicitly approved

# Career disposal cold-start fallback (used when no prior match history exists)
CAREER_DISP_AVG_FALLBACK = 15.0

# Team venue records
TEAM_VENUE_LOOKBACK_YEARS = 5
TEAM_VENUE_MIN_GAMES = 3
TEAM_VENUE_SCORE_FALLBACK = 80.0

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
ENSEMBLE_WEIGHTS = {"poisson": 0.8, "gbt": 0.2}
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

# HistGradientBoosting params for backtest (scoring model)
HIST_GBT_PARAMS_BACKTEST = {
    "max_iter": 250,
    "max_depth": 5,
    "learning_rate": 0.15,
    "min_samples_leaf": 20,
    "random_state": RANDOM_SEED,
}

POISSON_PARAMS = {
    "alpha": 0.014,
    "max_iter": 1000,
}

# Disposal-specific params (separate from scoring)
DISPOSAL_GBT_PARAMS_BACKTEST = {
    "max_iter": 217,
    "max_depth": 5,
    "learning_rate": 0.052,
    "min_samples_leaf": 10,
    "random_state": RANDOM_SEED,
}
DISPOSAL_POISSON_PARAMS = {
    "alpha": 0.015,
    "max_iter": 1000,
}

# Marks-specific params (separate from scoring/disposals)
MARKS_DISTRIBUTION = "gaussian"
MARKS_GBT_PARAMS_BACKTEST = {
    "max_iter": 217,
    "max_depth": 5,
    "learning_rate": 0.052,
    "min_samples_leaf": 10,
    "random_state": RANDOM_SEED,
}
MARKS_POISSON_PARAMS = {
    "alpha": 0.015,
    "max_iter": 1000,
}
MARKS_THRESHOLDS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# ---------------------------------------------------------------------------
# Market / odds features
# ---------------------------------------------------------------------------
MARKET_FEATURES_ENABLED = True
ODDS_PARQUET_PATH = BASE_STORE_DIR / "odds.parquet"
MARKET_POISSON_BLEND = 0.10   # blend market-implied baseline into Poisson component
# Market player-feature ablation mode:
#   "full"            -> full chain (environment + player implied goals/disposals)
#   "env_only"        -> keep only market_expected_match_goals + market_expected_team_goals
#   "player_goal_only"-> keep only market_implied_player_goals (intermediates hidden)
#   "v31_legacy"      -> revert to v3.1 market feature behavior
MARKET_PLAYER_FEATURE_CONFIG = "v31_legacy"
MARKET_POINTS_PER_GOAL_FALLBACK = 6.2
MARKET_POINTS_PER_DISPOSAL_FALLBACK = 0.38

# ---------------------------------------------------------------------------
# Elo rating system parameters
# ---------------------------------------------------------------------------
ELO_K_FACTOR = 12.3
ELO_HOME_ADVANTAGE = 16.2
ELO_SEASON_REGRESSION = 0.285

# ---------------------------------------------------------------------------
# Game winner model parameters
# ---------------------------------------------------------------------------
GAME_WINNER_PARAMS = {
    "max_iter": 64,
    "max_depth": 4,
    "learning_rate": 0.032,
    "min_samples_leaf": 13,
    "random_state": RANDOM_SEED,
}
GAME_WINNER_PARAMS_BACKTEST = {
    "max_iter": 64,
    "max_depth": 4,
    "learning_rate": 0.032,
    "min_samples_leaf": 13,
    "random_state": RANDOM_SEED,
}

# Hybrid winner mode: market prior + residual ML logit
WINNER_HYBRID_ENABLED = True
WINNER_HYBRID_ALPHA = 0.1
WINNER_HYBRID_BETA = 0.7
WINNER_HYBRID_BIAS = -0.02
WINNER_MARKET_EPS = 1e-6
WINNER_MIN_MARKET_COVERAGE = 0.30

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
MARKS_THRESHOLDS_EVAL = {"2plus_mk": 2, "3plus_mk": 3, "4plus_mk": 4, "5plus_mk": 5, "6plus_mk": 6, "7plus_mk": 7, "8plus_mk": 8, "9plus_mk": 9, "10plus_mk": 10}
CALIBRATION_METHOD = "isotonic"    # isotonic regression calibration (bucket params kept for analysis)
ISOTONIC_MIN_SAMPLES = 100         # min predictions before fitting isotonic calibrator
ISOTONIC_REFIT_INTERVAL = 5        # refit every N rounds in sequential mode
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
    "marks": {
        "targets": ["MK"],
        "primary_target": "MK",
        "description": "Marks prediction",
        "thresholds": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
}

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
VALIDATION_YEAR = 2024
MIN_PLAYER_MATCHES = 5
HISTORICAL_START_YEAR = 2015
HISTORICAL_END_YEAR = 2026
CURRENT_SEASON_YEAR = 2026

# Backtesting
BACKTEST_TRAIN_MIN_YEARS = 3
BACKTEST_DIR = DATA_DIR / "backtest"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# Sequential learning
SEQUENTIAL_DIR = DATA_DIR / "sequential"
SEQUENTIAL_YEAR = 2025
CALIBRATION_MIN_SAMPLES = 30       # min predictions per bucket before applying adjustment
CALIBRATION_MAX_ADJUSTMENT = 0.3   # cap adjustment to prevent wild swings
DISPOSAL_THRESHOLDS = [10, 15, 20, 25, 30]
GOAL_DISTRIBUTION_MAX_K = 7       # compute P(0), P(1), ..., P(6), P(7+)

# ---------------------------------------------------------------------------
# Hyperparameter tuning (Optuna)
# ---------------------------------------------------------------------------
TUNING_DIR = DATA_DIR / "tuning"

TUNE_N_TRIALS = 50
TUNE_WALK_FORWARD_FOLDS = 3  # number of walk-forward validation folds

# Parameter search ranges (for Optuna suggest_*)
TUNE_GBT_RANGES = {
    "n_estimators": (100, 500),
    "max_depth": (2, 6),
    "learning_rate": (0.01, 0.15),
    "min_samples_leaf": (10, 50),
    "subsample": (0.6, 1.0),
}
TUNE_POISSON_ALPHA_RANGE = (0.01, 2.0)
TUNE_ENSEMBLE_POISSON_WEIGHT_RANGE = (0.2, 0.8)
TUNE_ELO_RANGES = {
    "k_factor": (10, 60),
    "home_advantage": (10, 50),
    "season_regression": (0.2, 0.8),
}

# ---------------------------------------------------------------------------
# Name matching maps (AFLTables profile pages → pipeline names)
# ---------------------------------------------------------------------------
VENUE_NAME_MAP = {
    "M.C.G.": "M.C.G.",
    "MCG": "M.C.G.",
    "Melbourne Cricket Ground": "M.C.G.",
    "Docklands": "Docklands",
    "Marvel Stadium": "Docklands",
    "Etihad Stadium": "Docklands",
    "S.C.G.": "S.C.G.",
    "Sydney Cricket Ground": "S.C.G.",
    "Gabba": "Gabba",
    "Brisbane Cricket Ground": "Gabba",
    "Kardinia Park": "Kardinia Park",
    "GMHBA Stadium": "Kardinia Park",
    "Simonds Stadium": "Kardinia Park",
    "Skilled Stadium": "Kardinia Park",
    "Subiaco": "Subiaco",
    "Football Park": "Football Park",
    "AAMI Stadium": "Football Park",
    "Adelaide Oval": "Adelaide Oval",
    "Optus Stadium": "Perth Stadium",
    "Perth Stadium": "Perth Stadium",
    "York Park": "York Park",
    "Aurora Stadium": "York Park",
    "University of Tasmania Stadium": "York Park",
    "Cazalys Stadium": "Cazalys Stadium",
    "TIO Stadium": "TIO Stadium",
    "Marrara Oval": "TIO Stadium",
    "Manuka Oval": "Manuka Oval",
    "Bellerive Oval": "Bellerive Oval",
    "Blundstone Arena": "Bellerive Oval",
    "Carrara": "Carrara",
    "Metricon Stadium": "Carrara",
    "Heritage Bank Stadium": "Carrara",
    "Giants Stadium": "Giants Stadium",
    "Sydney Showground": "Giants Stadium",
    "ENGIE Stadium": "Giants Stadium",
    "Showground Stadium": "Giants Stadium",
    "Jiangwan Stadium": "Jiangwan Stadium",
    "Adelaide Arena at Jiangwan Stadium": "Jiangwan Stadium",
    "Stadium Australia": "Stadium Australia",
    "ANZ Stadium": "Stadium Australia",
    "Accor Stadium": "Stadium Australia",
    "Traeger Park": "Traeger Park",
    "Eureka Stadium": "Eureka Stadium",
    "Mars Stadium": "Eureka Stadium",
    "Riverway Stadium": "Riverway Stadium",
    "Norwood Oval": "Norwood Oval",
}

TEAM_NAME_MAP = {
    # Profile page names → pipeline canonical names
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Brisbane Bears": "Brisbane Bears",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "GWS Giants": "Greater Western Sydney",
    "Greater Western Sydney": "Greater Western Sydney",
    "GWS": "Greater Western Sydney",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
    "Footscray": "Western Bulldogs",
}

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create all required data directories."""
    for d in [DATA_DIR, RAW_DIR, BASE_STORE_DIR, CLEANED_DIR, FEATURES_DIR,
              MODELS_DIR, PREDICTIONS_DIR, FIXTURES_DIR, BACKTEST_DIR,
              LEARNING_DIR, SEQUENTIAL_DIR, TUNING_DIR, EXPERIMENTS_DIR,
              UMPIRES_DIR, COACHES_DIR, PLAYER_PROFILES_DIR, FOOTYWIRE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # Sequential learning subdirectories
    for subdir in ["predictions", "outcomes", "diagnostics", "analysis",
                   "game_predictions", "calibration", "archetypes", "concessions"]:
        (SEQUENTIAL_DIR / subdir).mkdir(parents=True, exist_ok=True)
