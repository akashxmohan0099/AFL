"""Backfill saved prediction exports to keep goal PMFs consistent with thresholds."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction_math import audit_prediction_frame, reconcile_goal_distribution


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save_frame(path: Path, df: pd.DataFrame) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _iter_prediction_files(root: Path):
    if root.is_file():
        yield root
        return
    for pattern in ("round_*_predictions.csv", "*.parquet"):
        for path in sorted(root.glob(pattern)):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile saved prediction probability exports.")
    parser.add_argument(
        "target",
        nargs="?",
        default=str(ROOT / "data/predictions"),
        help="Prediction file or directory to reconcile",
    )
    args = parser.parse_args()

    target = Path(args.target)
    updated = 0
    checked = 0
    for path in _iter_prediction_files(target):
        if path.name.startswith("."):
            continue
        try:
            df = _load_frame(path)
        except Exception:
            continue
        checked += 1
        before = audit_prediction_frame(df)
        reconcile_goal_distribution(df, round_dp=4)
        after = audit_prediction_frame(df)
        if after != before:
            _save_frame(path, df)
            updated += 1
            print(
                f"UPDATED {path.relative_to(ROOT)} "
                f"(goal zero err {before['goal_zero_consistency_max_abs_error']:.6f} -> "
                f"{after['goal_zero_consistency_max_abs_error']:.6f})"
            )

    print(f"Checked {checked} files, updated {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
