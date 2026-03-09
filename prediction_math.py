"""Shared helpers for prediction probability coherence and auditing."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _clip_probs(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _goal_pmf_columns(df: pd.DataFrame) -> list[str]:
    cols = [f"p_goals_{k}" for k in range(config.GOAL_DISTRIBUTION_MAX_K)]
    cols.append(f"p_goals_{config.GOAL_DISTRIBUTION_MAX_K}plus")
    return [col for col in cols if col in df.columns]


def _goal_threshold_source(df: pd.DataFrame) -> str | None:
    if "p_1plus_goals" in df.columns:
        return "p_1plus_goals"
    if "p_scorer" in df.columns:
        return "p_scorer"
    return None


def reconcile_goal_distribution(df: pd.DataFrame, round_dp: int = 4) -> None:
    """Make goal PMF columns consistent with exported 1+/2+/3+ thresholds."""
    if df is None or df.empty:
        return

    p1_col = _goal_threshold_source(df)
    if p1_col is None or "p_2plus_goals" not in df.columns or "p_3plus_goals" not in df.columns:
        return

    pmf_cols = _goal_pmf_columns(df)
    expected_cols = config.GOAL_DISTRIBUTION_MAX_K + 1
    if len(pmf_cols) != expected_cols:
        return

    pmf = _clip_probs(df[pmf_cols].to_numpy(dtype=float))
    p1 = _clip_probs(df[p1_col].to_numpy(dtype=float))
    p2 = np.minimum(_clip_probs(df["p_2plus_goals"].to_numpy(dtype=float)), p1)
    p3 = np.minimum(_clip_probs(df["p_3plus_goals"].to_numpy(dtype=float)), p2)

    exact_zero = 1.0 - p1
    exact_one = p1 - p2
    exact_two = p2 - p3
    tail_mass = p3

    tail = pmf[:, 3:].copy()
    tail_sum = tail.sum(axis=1)
    rebuilt = np.zeros_like(pmf)
    rebuilt[:, 0] = exact_zero
    rebuilt[:, 1] = exact_one
    rebuilt[:, 2] = exact_two

    positive_tail = tail_sum > 0
    if positive_tail.any():
        rebuilt[positive_tail, 3:] = (
            tail[positive_tail]
            * (tail_mass[positive_tail, None] / tail_sum[positive_tail, None])
        )

    zero_tail = ~positive_tail
    if zero_tail.any():
        rebuilt[zero_tail, 3] = tail_mass[zero_tail]

    rebuilt = np.clip(rebuilt, 0.0, 1.0)
    rebuilt[:, -1] += 1.0 - rebuilt.sum(axis=1)
    rebuilt = np.clip(rebuilt, 0.0, 1.0)
    rebuilt[:, -1] += 1.0 - rebuilt.sum(axis=1)

    df[pmf_cols] = np.round(rebuilt, round_dp)
    if "p_scorer" in df.columns:
        df["p_scorer"] = np.round(p1, round_dp)
    if "p_1plus_goals" in df.columns:
        df["p_1plus_goals"] = np.round(p1, round_dp)
    df["p_2plus_goals"] = np.round(p2, round_dp)
    df["p_3plus_goals"] = np.round(p3, round_dp)


def audit_prediction_frame(df: pd.DataFrame) -> dict:
    """Summarize probability sanity checks for a prediction DataFrame."""
    if df is None or df.empty:
        return {
            "rows": 0,
            "probability_columns": 0,
            "out_of_bounds": 0,
            "goal_threshold_monotonic_violations": 0,
            "disposal_threshold_monotonic_violations": 0,
            "marks_threshold_monotonic_violations": 0,
            "goal_pmf_sum_max_abs_error": 0.0,
            "goal_zero_consistency_max_abs_error": 0.0,
            "goal_2plus_consistency_max_abs_error": 0.0,
            "goal_3plus_consistency_max_abs_error": 0.0,
        }

    prob_cols = [c for c in df.columns if c.startswith("p_")]
    out_of_bounds = 0
    for col in prob_cols:
        out_of_bounds += int(((df[col] < -1e-9) | (df[col] > 1 + 1e-9)).sum())

    goal_monotonic = 0
    p1_col = _goal_threshold_source(df)
    if p1_col and "p_2plus_goals" in df.columns and "p_3plus_goals" in df.columns:
        goal_monotonic = int(
            ((df[p1_col] < df["p_2plus_goals"]) | (df["p_2plus_goals"] < df["p_3plus_goals"])).sum()
        )

    def _count_monotonic(cols: list[str]) -> int:
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            return 0
        arr = df[present].to_numpy(dtype=float)
        return int((arr[:, :-1] < arr[:, 1:]).any(axis=1).sum())

    pmf_sum_err = 0.0
    zero_consistency_err = 0.0
    two_plus_err = 0.0
    three_plus_err = 0.0
    pmf_cols = _goal_pmf_columns(df)
    if len(pmf_cols) == config.GOAL_DISTRIBUTION_MAX_K + 1:
        pmf = df[pmf_cols].to_numpy(dtype=float)
        pmf_sum_err = float(np.abs(pmf.sum(axis=1) - 1.0).max())
        if p1_col:
            zero_consistency_err = float(np.abs(df["p_goals_0"] - (1.0 - df[p1_col])).max())
        if "p_2plus_goals" in df.columns and "p_goals_1" in df.columns:
            two_plus_err = float(
                np.abs(df["p_2plus_goals"] - (1.0 - df["p_goals_0"] - df["p_goals_1"])).max()
            )
        if "p_3plus_goals" in df.columns and "p_goals_2" in df.columns:
            three_plus_err = float(
                np.abs(
                    df["p_3plus_goals"]
                    - (1.0 - df["p_goals_0"] - df["p_goals_1"] - df["p_goals_2"])
                ).max()
            )

    return {
        "rows": int(len(df)),
        "probability_columns": int(len(prob_cols)),
        "out_of_bounds": int(out_of_bounds),
        "goal_threshold_monotonic_violations": int(goal_monotonic),
        "disposal_threshold_monotonic_violations": _count_monotonic(
            [f"p_{t}plus_disp" for t in getattr(config, "DISPOSAL_THRESHOLDS", [])]
        ),
        "marks_threshold_monotonic_violations": _count_monotonic(
            [f"p_{t}plus_mk" for t in getattr(config, "MARKS_THRESHOLDS", [])]
        ),
        "goal_pmf_sum_max_abs_error": round(pmf_sum_err, 6),
        "goal_zero_consistency_max_abs_error": round(zero_consistency_err, 6),
        "goal_2plus_consistency_max_abs_error": round(two_plus_err, 6),
        "goal_3plus_consistency_max_abs_error": round(three_plus_err, 6),
    }

