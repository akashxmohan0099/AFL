"""Validate checked-in data artifacts used by the API and reports."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate import (
    ValidationError,
    validate_cleaned,
    validate_coaches,
    validate_features,
    validate_predictions,
    validate_temporal_integrity,
    validate_umpires,
)


def _load_json(path: Path):
    with open(path) as fp:
        return json.load(fp)


def _validate_if_present(path: Path, label: str, fn) -> None:
    if not path.exists():
        print(f"SKIP {label}: {path.relative_to(ROOT)} not found")
        return
    print(f"VALIDATE {label}: {path.relative_to(ROOT)}")
    fn(pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path))


def _latest_prediction_file(pred_dir: Path) -> Path | None:
    files = list(pred_dir.glob("round_*_predictions.csv"))
    if not files:
        return None

    def _round_key(path: Path) -> int:
        match = re.search(r"round_(\d+)_predictions\.csv$", path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=lambda path: (_round_key(path), path.stat().st_mtime))


def main() -> int:
    try:
        _validate_if_present(ROOT / "data/base/player_games.parquet", "cleaned player games", validate_cleaned)
        _validate_if_present(ROOT / "data/base/umpires.parquet", "umpires", validate_umpires)
        _validate_if_present(ROOT / "data/base/coaches.parquet", "coaches", validate_coaches)

        feature_matrix_path = ROOT / "data/features/feature_matrix.parquet"
        feature_columns_path = ROOT / "data/features/feature_columns.json"
        if feature_matrix_path.exists() and feature_columns_path.exists():
            print(f"VALIDATE features: {feature_matrix_path.relative_to(ROOT)}")
            feature_df = pd.read_parquet(feature_matrix_path)
            feature_cols = _load_json(feature_columns_path)
            validate_features(feature_df, feature_cols)
            validate_temporal_integrity(feature_df, feature_cols)
        else:
            print("SKIP features: feature matrix or feature_columns.json not found")

        latest_prediction = _latest_prediction_file(ROOT / "data/predictions")
        if latest_prediction is not None:
            print(f"VALIDATE predictions: {latest_prediction.relative_to(ROOT)}")
            validate_predictions(pd.read_csv(latest_prediction))
        else:
            print("SKIP predictions: no round_*_predictions.csv files found")
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
