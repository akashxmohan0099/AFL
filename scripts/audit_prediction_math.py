"""Local audit for prediction coherence and season-level threshold statistics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from analysis import _extract_threshold_data, _load_season_merged_predictions_outcomes
from metrics import compute_threshold_metrics
from prediction_math import audit_prediction_frame
from store import LearningStore


def _find_latest_prediction_file() -> Path | None:
    files = sorted((ROOT / "data/predictions").glob("round_*_predictions.csv"))
    if not files:
        return None

    def _round_key(path: Path) -> tuple[int, float]:
        stem = path.stem
        try:
            round_num = int(stem.split("_")[1])
        except Exception:
            round_num = -1
        return round_num, path.stat().st_mtime

    return max(files, key=_round_key)


def _print_prediction_audit(path: Path) -> None:
    df = pd.read_csv(path)
    audit = audit_prediction_frame(df)
    print(f"\nLatest prediction file: {path.relative_to(ROOT)}")
    print(f"  Rows: {audit['rows']}")
    print(f"  Probability columns: {audit['probability_columns']}")
    print(f"  Out of bounds: {audit['out_of_bounds']}")
    print(f"  Goal monotonic violations: {audit['goal_threshold_monotonic_violations']}")
    print(f"  Disposal monotonic violations: {audit['disposal_threshold_monotonic_violations']}")
    print(f"  Marks monotonic violations: {audit['marks_threshold_monotonic_violations']}")
    print(f"  Goal PMF max abs(sum-1): {audit['goal_pmf_sum_max_abs_error']:.6f}")
    print(f"  Goal P(0) vs scorer max abs err: {audit['goal_zero_consistency_max_abs_error']:.6f}")
    print(f"  Goal 2+ vs PMF max abs err: {audit['goal_2plus_consistency_max_abs_error']:.6f}")
    print(f"  Goal 3+ vs PMF max abs err: {audit['goal_3plus_consistency_max_abs_error']:.6f}")


def _print_threshold_report(store: LearningStore, year: int) -> None:
    merged = _load_season_merged_predictions_outcomes(store, year)
    if merged.empty:
        print(f"\nSeason {year}: no merged predictions/outcomes available")
        return

    print(f"\nSeason {year} threshold metrics")
    print(f"{'Threshold':<14} {'n':>7} {'Brier':>8} {'BSS':>8} {'LogLoss':>8} {'ECE':>8}")

    threshold_data = _extract_threshold_data(merged)
    for name, (preds, actuals) in threshold_data.items():
        metrics = compute_threshold_metrics(
            preds,
            actuals,
            label=name,
            n_bins=getattr(config, "CALIBRATION_N_BUCKETS", 10),
            min_bucket_size=getattr(config, "CALIBRATION_MIN_BUCKET_SIZE", 1),
            min_n=10,
        )
        if metrics is None:
            continue
        print(
            f"{name:<14} {metrics['n']:7d} {metrics['brier_score']:8.4f} "
            f"{metrics['bss']:8.4f} {metrics['log_loss']:8.4f} {metrics['calibration_ece']:8.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local prediction math and season threshold metrics.")
    parser.add_argument(
        "--year",
        type=int,
        default=getattr(config, "CURRENT_SEASON_YEAR", 2026),
        help="Season year for backtest metric audit.",
    )
    args = parser.parse_args()

    latest_prediction = _find_latest_prediction_file()
    if latest_prediction is not None:
        _print_prediction_audit(latest_prediction)
    else:
        print("No top-level prediction CSVs found under data/predictions")

    store = LearningStore(base_dir=config.SEQUENTIAL_DIR)
    _print_threshold_report(store, int(args.year))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
