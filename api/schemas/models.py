"""Pydantic response models."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class PlayerSearchResult(BaseModel):
    player_id: str
    name: str
    team: str
    total_games: int
    score: float


class SeasonAvg(BaseModel):
    year: int
    games: int
    GL: float
    BH: float
    DI: float
    MK: float
    KI: float
    HB: float
    TK: float
    HO: float


class PlayerProfile(BaseModel):
    player_id: str
    name: str
    team: str
    total_games: int
    career_goals: int
    career_goal_avg: float
    seasons: List[SeasonAvg]
    recent_form: List[dict]


class GameLogEntry(BaseModel):
    match_id: int
    date: str
    year: int
    round_number: int
    team: str
    opponent: str
    venue: str
    GL: int
    BH: int
    DI: int
    MK: int
    KI: int
    HB: int
    TK: int
    HO: int


class PlayerPrediction(BaseModel):
    player: str
    player_id: Optional[str] = None
    team: str
    opponent: str
    predicted_goals: Optional[float] = None
    predicted_disposals: Optional[float] = None
    predicted_marks: Optional[float] = None
    p_scorer: Optional[float] = None
    p_2plus_goals: Optional[float] = None
    p_3plus_goals: Optional[float] = None
    p_20plus_disp: Optional[float] = None
    p_25plus_disp: Optional[float] = None
    p_30plus_disp: Optional[float] = None


class RoundInfo(BaseModel):
    year: int
    round_number: int


class MatchPrediction(BaseModel):
    match_id: Optional[int] = None
    home_team: str
    away_team: str
    venue: Optional[str] = None
    date: Optional[str] = None
    home_win_prob: Optional[float] = None
    away_win_prob: Optional[float] = None
    predicted_winner: Optional[str] = None
    actual_winner: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    correct: Optional[bool] = None


class MatchDetail(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    venue: Optional[str] = None
    date: Optional[str] = None
    year: int
    round_number: int
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_win_prob: Optional[float] = None
    away_win_prob: Optional[float] = None
    home_players: List[PlayerPrediction]
    away_players: List[PlayerPrediction]


class OddsComparison(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    model_home_prob: Optional[float] = None
    market_home_prob: Optional[float] = None
    model_away_prob: Optional[float] = None
    market_away_prob: Optional[float] = None
    edge_home: Optional[float] = None
    edge_away: Optional[float] = None


class PlayerOddsComparison(BaseModel):
    player: str
    team: str
    market_type: str
    market_line: Optional[float] = None
    market_price: Optional[float] = None
    market_implied_prob: Optional[float] = None
    model_prob: Optional[float] = None
    edge: Optional[float] = None


class ThresholdMetric(BaseModel):
    label: str
    n: int
    base_rate: Optional[float] = None
    brier_score: Optional[float] = None
    bss: Optional[float] = None
    accuracy: Optional[float] = None
    hit_rate_p60: Optional[float] = None
    hit_rate_p70: Optional[float] = None
    hit_rate_p80: Optional[float] = None
    calibration_ece: Optional[float] = None
    calibration_curve: Optional[List[dict]] = None


class ExperimentSummary(BaseModel):
    filename: str
    label: str
    mae_goals: Optional[float] = None
    mae_disposals: Optional[float] = None
    mae_marks: Optional[float] = None
    brier_1plus: Optional[float] = None
    bss_1plus: Optional[float] = None
    brier_2plus: Optional[float] = None
    bss_20plus_disp: Optional[float] = None
    bss_25plus_disp: Optional[float] = None
    bss_5plus_mk: Optional[float] = None
    game_winner_accuracy: Optional[float] = None
