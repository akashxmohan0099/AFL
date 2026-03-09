"""Metrics and experiment comparison services."""

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from api.data_loader import DataCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_experiments import extract_metrics


def _get_store(cache: DataCache):
    return cache.sequential_store or cache.store


def get_experiments() -> list[dict]:
    cache = DataCache.get()
    results = []
    for exp in cache.experiments:
        m = extract_metrics(exp)
        results.append({
            "filename": exp.get("_filename", ""),
            "label": m["label"],
            "mae_goals": m.get("mae"),
            "mae_disposals": m.get("disp_mae"),
            "mae_marks": m.get("marks_mae"),
            "brier_1plus": m.get("brier_1plus"),
            "bss_1plus": _bss_from_brier(m.get("brier_1plus"), _get_base_rate(exp, "1plus_goals")),
            "brier_2plus": m.get("brier_2plus"),
            "bss_20plus_disp": m.get("disp_bss_20"),
            "bss_25plus_disp": m.get("disp_bss_25"),
            "bss_5plus_mk": m.get("marks_bss_5"),
            "game_winner_accuracy": exp.get("game_winner", {}).get("accuracy"),
        })
    return results


def get_backtest_metrics(year: int) -> dict:
    cache = DataCache.get()
    store = _get_store(cache)
    if store is None:
        return {"error": "No learning store"}

    round_pairs = store.list_rounds(subdir="predictions", year=year)
    if not round_pairs:
        return {"year": year, "error": "No data", "rounds": 0}

    try:
        from metrics import compute_all_brier
        merged_rounds = []
        rounds_with_data = 0

        for _, round_num in round_pairs:
            preds = store.load_predictions(year=year, round_num=round_num)
            outcomes = store.load_outcomes(year=year, round_num=round_num)
            if preds.empty or outcomes.empty:
                continue

            merge_cols = ["player", "team"]
            if "match_id" in preds.columns and "match_id" in outcomes.columns:
                merge_cols.append("match_id")
            elif "round" in preds.columns and "round" in outcomes.columns:
                merge_cols.append("round")

            merged = preds.merge(outcomes, on=merge_cols, how="inner", suffixes=("", "_actual"))
            if merged.empty:
                continue

            merged_rounds.append(merged)
            rounds_with_data += 1

        if not merged_rounds:
            return {"year": year, "error": "No data", "rounds": 0}

        merged = pd.concat(merged_rounds, ignore_index=True)

        # Rename outcome columns to expected format
        rename_map = {}
        if "actual_goals" not in merged.columns and "GL" in merged.columns:
            rename_map["GL"] = "actual_goals"
        if "actual_disposals" not in merged.columns and "DI" in merged.columns:
            rename_map["DI"] = "actual_disposals"
        if "actual_marks" not in merged.columns and "MK" in merged.columns:
            rename_map["MK"] = "actual_marks"
        if rename_map:
            merged = merged.rename(columns=rename_map)

        if "actual_goals" not in merged.columns:
            merged = merged.drop(
                columns=["p_scorer", "p_1plus_goals", "p_2plus_goals", "p_3plus_goals", "predicted_goals"],
                errors="ignore",
            )
        if "actual_disposals" not in merged.columns:
            merged = merged.drop(
                columns=["predicted_disposals"] + [
                    col for col in merged.columns if col.startswith("p_") and col.endswith("plus_disp")
                ],
                errors="ignore",
            )
        if "actual_marks" not in merged.columns:
            merged = merged.drop(
                columns=["predicted_marks"] + [
                    col for col in merged.columns if col.startswith("p_") and col.endswith("plus_mk")
                ],
                errors="ignore",
            )

        result = compute_all_brier(merged)
        result["year"] = year
        result["n_predictions"] = len(merged)
        result["rounds"] = rounds_with_data
        return result
    except Exception as e:
        return {"year": year, "error": str(e)}


def get_calibration_data() -> dict:
    """Extract calibration curve data from the latest experiment that has it."""
    cache = DataCache.get()
    if not cache.experiments:
        return {}
    # Find latest experiment with calibration data
    exp = None
    for e in reversed(cache.experiments):
        if any(s in e for s in ("thresholds", "disposal_thresholds", "marks_thresholds")):
            exp = e
            break
    if exp is None:
        return {}
    result = {}
    for section, prefix in [
        ("thresholds", "goals"),
        ("disposal_thresholds", "disposals"),
        ("marks_thresholds", "marks"),
    ]:
        for key, data in exp.get(section, {}).items():
            curve = data.get("calibration_curve", [])
            ece = data.get("calibration_ece")
            bss = data.get("bss")
            n = data.get("n")
            if curve:
                result[key] = {
                    "curve": curve,
                    "ece": ece,
                    "bss": bss,
                    "n": n,
                    "category": prefix,
                }
    return result


def _bss_from_brier(brier_score, base_rate):
    if brier_score is None or base_rate is None or base_rate <= 0:
        return None
    naive = base_rate * (1 - base_rate)
    if naive == 0:
        return 0.0
    return round((1 - brier_score / naive) * 100, 1)


def _get_base_rate(exp, key):
    return exp.get("thresholds", {}).get(key, {}).get("base_rate")


def get_multi_backtest(year: int) -> dict:
    """Load multi-bet backtest results from experiment JSON."""
    cache = DataCache.get()
    for exp in cache.experiments:
        if exp.get("label") == f"multi_backtest_{year}":
            return exp
    return {"error": f"No multi-bet backtest found for {year}"}


def get_runtime_metrics(app: Any) -> dict:
    runtime_metrics = getattr(app.state, "runtime_metrics", None)
    settings = getattr(app.state, "settings", None)
    cache = DataCache.get()
    snapshot = runtime_metrics.snapshot() if runtime_metrics is not None else {}
    snapshot["cache_loaded"] = getattr(cache, "is_loaded", False)
    snapshot["auth_enabled"] = bool(getattr(settings, "auth_enabled", False))
    snapshot["rate_limit_enabled"] = bool(getattr(settings, "rate_limit_enabled", False))
    return snapshot
