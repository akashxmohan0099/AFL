"""
AFL Prediction Pipeline — Learning Store
==========================================
Persistent storage for sequential/backtest learning artifacts.

Run-partitioned layout (append-only):
  data/learning/
    predictions/<year>/run_<run_id>/RNN.parquet
    outcomes/<year>/run_<run_id>/RNN.parquet
    diagnostics/<year>/run_<run_id>/RNN.parquet
    game_predictions/<year>/run_<run_id>/RNN.parquet
    analysis/<year>/run_<run_id>/RNN.json
    calibration/run_<run_id>/calibration_state.parquet
    archetypes/*.parquet
    concessions/*.parquet

Legacy flat files (e.g. predictions/YYYY_RNN.parquet) are still readable.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

import config


class LearningStore:
    """Manages persistent learning records across rounds.

    Notes:
      - New writes are run-versioned and append-only.
      - `run_id` is sticky within a store instance for consistent writes.
      - Loaders default to the latest run for a given year/subdir.
      - Legacy flat file layout remains readable for backward compatibility.
    """

    SUBDIRS = [
        "predictions", "outcomes", "diagnostics",
        "calibration", "archetypes", "concessions",
        "analysis", "game_predictions",
    ]

    def __init__(self, base_dir=None, run_id=None):
        self.base_dir = Path(base_dir or config.LEARNING_DIR)
        self.run_id = self._normalise_run_id(run_id)
        self._active_run_id = self.run_id
        self._ensure_dirs()

    @staticmethod
    def _generate_run_id():
        """Generate a unique run id (UTC timestamp with microseconds)."""
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    @staticmethod
    def _normalise_run_id(run_id):
        if run_id is None:
            return None
        rid = str(run_id).strip()
        if rid.startswith("run_"):
            rid = rid[4:]
        return rid or None

    def _ensure_dirs(self):
        """Create directory structure."""
        for subdir in self.SUBDIRS:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _legacy_round_path(self, subdir, year, round_num):
        return self.base_dir / subdir / f"{year}_R{round_num:02d}.parquet"

    def _run_dir(self, subdir, year, run_id):
        rid = self._normalise_run_id(run_id)
        return self.base_dir / subdir / f"{int(year)}" / f"run_{rid}"

    @staticmethod
    def _path_mtime_ns(path):
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return -1

    @classmethod
    def _sort_paths_by_mtime(cls, paths):
        return sorted(paths, key=lambda p: (cls._path_mtime_ns(p), p.name))

    def _latest_run(self, subdir="predictions", year=None):
        runs = self.list_runs(year=year, subdir=subdir)
        return runs[-1] if runs else None

    def list_runs(self, year=None, subdir="predictions"):
        """List known run ids for a subdir (optionally filtered by year)."""
        root = self.base_dir / subdir
        if not root.exists():
            return []

        run_mtimes = {}

        if year is not None:
            y_dir = root / f"{int(year)}"
            if y_dir.exists():
                for d in self._sort_paths_by_mtime(p for p in y_dir.glob("run_*") if p.is_dir()):
                    rid = d.name.replace("run_", "", 1)
                    run_mtimes[rid] = max(run_mtimes.get(rid, -1), self._path_mtime_ns(d))
        else:
            for y_dir in root.iterdir():
                if not y_dir.is_dir() or not y_dir.name.isdigit():
                    continue
                for d in self._sort_paths_by_mtime(p for p in y_dir.glob("run_*") if p.is_dir()):
                    rid = d.name.replace("run_", "", 1)
                    run_mtimes[rid] = max(run_mtimes.get(rid, -1), self._path_mtime_ns(d))

        ordered = sorted(run_mtimes.items(), key=lambda item: (item[1], item[0]))
        return [rid for rid, _ in ordered]

    def _resolve_write_run_id(self, run_id=None):
        rid = self._normalise_run_id(run_id) or self._active_run_id
        if rid is None:
            rid = self._generate_run_id()
        self._active_run_id = rid
        return rid

    def _resolve_read_run_id(self, subdir, year, run_id=None, latest=True):
        """Resolve run id for reads; prefer active run if available."""
        rid = self._normalise_run_id(run_id)
        if rid is not None:
            self._active_run_id = rid
            return rid

        if self.run_id is not None:
            cfg_dir = self._run_dir(subdir, year, self.run_id)
            if cfg_dir.exists():
                self._active_run_id = self.run_id
                return self.run_id

        if latest:
            rid = self._latest_run(subdir=subdir, year=year)
            if rid is not None:
                self._active_run_id = rid
                return rid

        if self._active_run_id is not None:
            active_dir = self._run_dir(subdir, year, self._active_run_id)
            if active_dir.exists():
                return self._active_run_id

        return None

    def _round_path(self, subdir, year, round_num, run_id=None, for_write=False, latest=True):
        year = int(year)
        round_num = int(round_num)

        if for_write:
            rid = self._resolve_write_run_id(run_id)
            return self._run_dir(subdir, year, rid) / f"R{round_num:02d}.parquet"

        rid = self._resolve_read_run_id(subdir, year, run_id=run_id, latest=latest)
        if rid is not None:
            return self._run_dir(subdir, year, rid) / f"R{round_num:02d}.parquet"

        return self._legacy_round_path(subdir, year, round_num)

    def _analysis_path(self, year, round_num, run_id=None, for_write=False, latest=True):
        year = int(year)
        round_num = int(round_num)

        if for_write:
            rid = self._resolve_write_run_id(run_id)
            return self._run_dir("analysis", year, rid) / f"R{round_num:02d}.json"

        rid = self._resolve_read_run_id("analysis", year, run_id=run_id, latest=latest)
        if rid is not None:
            return self._run_dir("analysis", year, rid) / f"R{round_num:02d}.json"

        return self.base_dir / "analysis" / f"{year}_R{round_num:02d}.json"

    def _legacy_calibration_path(self):
        return self.base_dir / "calibration" / "calibration_state.parquet"

    def _latest_calibration_run(self):
        cal_dir = self.base_dir / "calibration"
        if not cal_dir.exists():
            return None
        run_dirs = self._sort_paths_by_mtime(d for d in cal_dir.glob("run_*") if d.is_dir())
        if not run_dirs:
            return None
        return run_dirs[-1].name.replace("run_", "", 1)

    def _calibration_path(self, run_id=None, for_write=False, latest=True, year=None):
        """Resolve calibration state path, preferring run-specific state."""
        if for_write:
            rid = self._resolve_write_run_id(run_id)
            return self.base_dir / "calibration" / f"run_{rid}" / "calibration_state.parquet"

        rid = self._normalise_run_id(run_id)
        if rid is not None:
            self._active_run_id = rid
            return self.base_dir / "calibration" / f"run_{rid}" / "calibration_state.parquet"

        if year is not None:
            rid = self._resolve_read_run_id("predictions", year, run_id=None, latest=latest)
            if rid is not None:
                self._active_run_id = rid
                return self.base_dir / "calibration" / f"run_{rid}" / "calibration_state.parquet"

        if self.run_id is not None:
            cfg_path = self.base_dir / "calibration" / f"run_{self.run_id}" / "calibration_state.parquet"
            if cfg_path.exists():
                self._active_run_id = self.run_id
                return cfg_path

        if latest:
            rid = self._latest_calibration_run()
            if rid is not None:
                self._active_run_id = rid
                return self.base_dir / "calibration" / f"run_{rid}" / "calibration_state.parquet"

        if self._active_run_id is not None:
            return self.base_dir / "calibration" / f"run_{self._active_run_id}" / "calibration_state.parquet"

        return self._legacy_calibration_path()

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def save_predictions(self, year, round_num, df, run_id=None):
        """Save predictions for a round."""
        path = self._round_path("predictions", year, round_num, run_id=run_id, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def load_predictions(self, year=None, round_num=None, run_id=None, latest=True):
        """Load predictions. If year/round given, load that specific round.
        If only year given, load all rounds for that year.
        If neither, load all predictions."""
        return self._load_records("predictions", year, round_num, run_id=run_id, latest=latest)

    def has_predictions(self, year, round_num, run_id=None, latest=True):
        """Check if predictions exist for a specific round."""
        return self._round_path(
            "predictions", year, round_num, run_id=run_id, latest=latest
        ).exists()

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def save_outcomes(self, year, round_num, df, run_id=None):
        """Save actual outcomes for a round."""
        path = self._round_path("outcomes", year, round_num, run_id=run_id, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def load_outcomes(self, year=None, round_num=None, run_id=None, latest=True):
        """Load outcomes."""
        return self._load_records("outcomes", year, round_num, run_id=run_id, latest=latest)

    def has_outcomes(self, year, round_num, run_id=None, latest=True):
        """Check if outcomes exist for a specific round."""
        return self._round_path(
            "outcomes", year, round_num, run_id=run_id, latest=latest
        ).exists()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def save_diagnostics(self, year, round_num, df, run_id=None):
        """Save diagnostic records for a round."""
        path = self._round_path("diagnostics", year, round_num, run_id=run_id, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def load_diagnostics(self, year=None, round_num=None, run_id=None, latest=True):
        """Load diagnostics."""
        return self._load_records("diagnostics", year, round_num, run_id=run_id, latest=latest)

    def load_all_diagnostics(self):
        """Load all diagnostic records across all rounds."""
        return self._load_records("diagnostics", latest=False)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def get_calibration_state(self, year=None, run_id=None, latest=True):
        """Load the current calibration state."""
        path = self._calibration_path(run_id=run_id, for_write=False, latest=latest, year=year)
        if path.exists():
            return pd.read_parquet(path)
        return self._empty_calibration()

    def update_calibration(self, new_data, run_id=None):
        """Update calibration state with new round data.

        new_data should have columns:
          target, probability_bucket, predicted, occurred
        """
        current = self.get_calibration_state(run_id=run_id)

        # Merge: accumulate counts
        for _, row in new_data.iterrows():
            mask = (
                (current["target"] == row["target"])
                & (current["probability_bucket"] == row["probability_bucket"])
            )
            if mask.any():
                current.loc[mask, "n_predictions"] += row["predicted"]
                current.loc[mask, "n_occurred"] += row["occurred"]
            else:
                new_row = pd.DataFrame([{
                    "target": row["target"],
                    "probability_bucket": row["probability_bucket"],
                    "n_predictions": row["predicted"],
                    "n_occurred": row["occurred"],
                    "observed_rate": 0.0,
                    "calibration_adj": 0.0,
                }])
                current = pd.concat([current, new_row], ignore_index=True)

        # Recompute observed rates
        mask = current["n_predictions"] > 0
        current.loc[mask, "observed_rate"] = (
            current.loc[mask, "n_occurred"] / current.loc[mask, "n_predictions"]
        )

        path = self._calibration_path(run_id=run_id, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(path, index=False)
        return current

    def _empty_calibration(self):
        """Create empty calibration DataFrame with standard buckets."""
        targets = [
            "1plus_goals", "2plus_goals", "3plus_goals",
            "10plus_disp", "15plus_disp",
            "20plus_disp", "25plus_disp", "30plus_disp",
        ]
        buckets = np.arange(0.05, 1.0, 0.1).round(2)
        rows = []
        for target in targets:
            for bucket in buckets:
                rows.append({
                    "target": target,
                    "probability_bucket": float(bucket),
                    "n_predictions": 0,
                    "n_occurred": 0,
                    "observed_rate": 0.0,
                    "calibration_adj": 0.0,
                })
        return pd.DataFrame(rows)

    def compute_calibration_adjustments(self, run_id=None):
        """Compute calibration adjustments from accumulated prediction data.

        For each (target, bucket) with enough samples, compute:
          calibration_adj = observed_rate - bucket_midpoint
        Capped to ±CALIBRATION_MAX_ADJUSTMENT.
        """
        current = self.get_calibration_state(run_id=run_id)
        min_samples = config.CALIBRATION_MIN_SAMPLES
        max_adj = config.CALIBRATION_MAX_ADJUSTMENT

        for idx, row in current.iterrows():
            if row["n_predictions"] >= min_samples:
                observed = row["observed_rate"]
                bucket_mid = row["probability_bucket"]
                adj = np.clip(observed - bucket_mid, -max_adj, max_adj)
                current.at[idx, "calibration_adj"] = float(adj)
            else:
                current.at[idx, "calibration_adj"] = 0.0

        path = self._calibration_path(run_id=run_id, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(path, index=False)
        return current

    def get_calibration_adjustment(self, target, predicted_prob, run_id=None):
        """Look up the calibration adjustment for a specific prediction.

        Finds the nearest bucket for the given probability and returns
        the stored calibration_adj (0.0 if insufficient data).
        """
        cal = self.get_calibration_state(run_id=run_id)
        target_cal = cal[cal["target"] == target]
        if target_cal.empty:
            return 0.0

        # Find nearest bucket
        buckets = target_cal["probability_bucket"].values
        idx = np.argmin(np.abs(buckets - predicted_prob))
        return float(target_cal.iloc[idx]["calibration_adj"])

    def get_lambda_calibration(self, target, raw_lambda, run_id=None):
        """Adjust Poisson lambda before computing PMFs using calibration data.

        For goals: checks 1plus_goals, 2plus_goals, 3plus_goals calibration.
        For disposals: checks 10plus_disp, 15plus_disp, 20plus_disp, etc.

        Computes what P(X>=threshold) would be at raw_lambda via Poisson CDF,
        looks up calibration adjustments, and applies average adjustment as
        multiplicative correction to lambda.
        """
        raw_lambda = max(raw_lambda, 0.001)

        if target == "goals":
            thresholds = [
                ("1plus_goals", 1),
                ("2plus_goals", 2),
                ("3plus_goals", 3),
            ]
        elif target == "disposals":
            thresholds = [
                ("10plus_disp", 10),
                ("15plus_disp", 15),
                ("20plus_disp", 20),
                ("25plus_disp", 25),
                ("30plus_disp", 30),
            ]
        else:
            return raw_lambda

        adjustments = []
        for cal_target, k in thresholds:
            # P(X >= k) at raw_lambda
            p_exceed = 1.0 - poisson_dist.cdf(k - 1, raw_lambda)
            adj = self.get_calibration_adjustment(cal_target, p_exceed, run_id=run_id)
            if adj != 0.0:
                adjustments.append(adj)

        if not adjustments:
            return raw_lambda

        # Average adjustment as multiplicative correction
        avg_adj = np.mean(adjustments)
        # Convert additive probability adjustment to lambda multiplier
        # If we're underpredicting (positive adj), increase lambda
        multiplier = 1.0 + avg_adj
        multiplier = max(multiplier, 0.5)  # don't reduce lambda by more than half
        multiplier = min(multiplier, 2.0)  # don't more than double lambda

        return max(raw_lambda * multiplier, 0.001)

    def reset_calibration(self, year=None, run_id=None, latest=True):
        """Delete calibration state for a selected run (if present)."""
        path = self._calibration_path(run_id=run_id, for_write=False, latest=latest, year=year)
        if path.exists():
            path.unlink()
            return True
        return False

    def seed_calibration_from_latest(self):
        """Initialize current run calibration from the latest prior run, if any.

        Returns True when state is copied, else False.
        """
        target = self._calibration_path(for_write=True)
        if target.exists():
            return False

        source = None
        latest_rid = self._latest_calibration_run()
        if latest_rid is not None and latest_rid != self._active_run_id:
            candidate = self.base_dir / "calibration" / f"run_{latest_rid}" / "calibration_state.parquet"
            if candidate.exists():
                source = candidate

        legacy = self._legacy_calibration_path()
        if source is None and legacy.exists():
            source = legacy

        if source is None:
            return False

        target.parent.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(source).to_parquet(target, index=False)
        return True

    # ------------------------------------------------------------------
    # Isotonic Calibration
    # ------------------------------------------------------------------

    def save_isotonic_calibrator(self, calibrator, run_id=None):
        """Save a CalibratedPredictor to pickle in the calibration directory."""
        import pickle
        rid = self._resolve_write_run_id(run_id)
        cal_dir = self.base_dir / "calibration" / f"run_{rid}"
        cal_dir.mkdir(parents=True, exist_ok=True)
        path = cal_dir / "isotonic_calibrator.pkl"
        with open(path, "wb") as f:
            pickle.dump(calibrator, f)
        return path

    def load_isotonic_calibrator(self, run_id=None):
        """Load the latest CalibratedPredictor from the calibration directory.

        Returns None if no calibrator exists.
        """
        import pickle

        # Try active run first, then latest
        rid = self._normalise_run_id(run_id) or self._active_run_id
        if rid is not None:
            path = self.base_dir / "calibration" / f"run_{rid}" / "isotonic_calibrator.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    return None

        # Try latest calibration run
        latest_rid = self._latest_calibration_run()
        if latest_rid is not None:
            path = self.base_dir / "calibration" / f"run_{latest_rid}" / "isotonic_calibrator.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    return None

        return None

    def save_isotonic_accum(self, accum, run_id=None):
        """Save isotonic accumulation data (preds/actuals per target) to disk.

        Saved to a GLOBAL location (not run-specific) so cross-year sequential
        runs can accumulate isotonic training data across all years.
        """
        import pickle
        cal_dir = self.base_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        path = cal_dir / "isotonic_accum.pkl"
        with open(path, "wb") as f:
            pickle.dump(accum, f)
        return path

    def load_isotonic_accum(self, run_id=None):
        """Load isotonic accumulation data from the global calibration directory.

        Returns empty dict if no accumulation data exists.
        """
        import pickle
        path = self.base_dir / "calibration" / "isotonic_accum.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Archetypes
    # ------------------------------------------------------------------

    def save_archetypes(self, df, append_history=True):
        """Save current archetype assignments.

        If append_history=True, also appends to the history file.
        """
        latest_path = self.base_dir / "archetypes" / "archetypes_latest.parquet"
        df.to_parquet(latest_path, index=False)

        if append_history:
            history_path = self.base_dir / "archetypes" / "archetypes_history.parquet"
            if history_path.exists():
                existing = pd.read_parquet(history_path)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(history_path, index=False)
            else:
                df.to_parquet(history_path, index=False)

        return latest_path

    def load_archetypes(self):
        """Load latest archetype assignments."""
        path = self.base_dir / "archetypes" / "archetypes_latest.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def load_archetype_history(self):
        """Load full archetype history."""
        path = self.base_dir / "archetypes" / "archetypes_history.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Concession profiles
    # ------------------------------------------------------------------

    def save_concessions(self, df):
        """Save opponent concession profiles by archetype."""
        path = self.base_dir / "concessions" / "concessions_latest.parquet"
        df.to_parquet(path, index=False)
        return path

    def load_concessions(self):
        """Load latest concession profiles."""
        path = self.base_dir / "concessions" / "concessions_latest.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def save_analysis(self, year, round_num, analysis_dict, run_id=None):
        """Save round analysis as JSON."""
        path = self._analysis_path(year, round_num, run_id=run_id, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(analysis_dict, f, indent=2, default=str)
        return path

    def load_analysis(self, year, round_num, run_id=None, latest=True):
        """Load round analysis JSON."""
        path = self._analysis_path(year, round_num, run_id=run_id, latest=latest)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    # ------------------------------------------------------------------
    # Game predictions
    # ------------------------------------------------------------------

    def save_game_predictions(self, year, round_num, df, run_id=None):
        """Save game-level predictions for a round."""
        path = self._round_path("game_predictions", year, round_num, run_id=run_id, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def load_game_predictions(self, year=None, round_num=None, run_id=None, latest=True):
        """Load game predictions."""
        return self._load_records("game_predictions", year, round_num, run_id=run_id, latest=latest)

    # ------------------------------------------------------------------
    # Summary & utility
    # ------------------------------------------------------------------

    def get_learning_summary(self, year=None, up_to_round=None, run_id=None, latest=True):
        """Return a summary of accumulated learning up to a given point.

        Returns dict with counts of predictions, outcomes, diagnostics
        processed, plus current calibration state.
        """
        def _count_rounds(subdir):
            rounds = self.list_rounds(
                subdir=subdir, year=year, run_id=run_id, latest=latest
            )
            if up_to_round is not None:
                rounds = [r for r in rounds if r[1] <= int(up_to_round)]
            return len(rounds)

        return {
            "run_id": self._active_run_id,
            "prediction_rounds": _count_rounds("predictions"),
            "outcome_rounds": _count_rounds("outcomes"),
            "diagnostic_rounds": _count_rounds("diagnostics"),
            "has_archetypes": (self.base_dir / "archetypes" / "archetypes_latest.parquet").exists(),
            "has_concessions": (self.base_dir / "concessions" / "concessions_latest.parquet").exists(),
            "has_calibration": self._calibration_path(
                run_id=run_id, for_write=False, latest=latest, year=year
            ).exists(),
        }

    def list_rounds(self, subdir="predictions", year=None, run_id=None, latest=True):
        """List all (year, round) tuples that have records."""
        files = self._list_record_files(
            subdir, year=year, run_id=run_id, latest=latest, include_legacy=True
        )
        rounds = [r for r in (self._parse_year_round(f) for f in files) if r is not None]
        rounds = sorted(set(rounds))
        return rounds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_year_round(self, path):
        """Parse (year, round) from either legacy or run-versioned path."""
        stem = path.stem
        if "_R" in stem:
            parts = stem.split("_R")
            if len(parts) == 2 and parts[0].isdigit():
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    return None

        if stem.startswith("R") and stem[1:].isdigit():
            try:
                y_name = path.parent.parent.name
                if y_name.isdigit():
                    return int(y_name), int(stem[1:])
            except Exception:
                return None
        return None

    def _list_record_files(self, subdir, year=None, run_id=None, latest=True, include_legacy=True):
        """List record files under both run-versioned and legacy layouts."""
        d = self.base_dir / subdir
        files = []

        if year is not None:
            year = int(year)
            if run_id is not None:
                rid = self._normalise_run_id(run_id)
                run_dir = self._run_dir(subdir, year, rid)
                files.extend(sorted(run_dir.glob("R*.parquet")))
            else:
                if latest:
                    rid = self._resolve_read_run_id(subdir, year, run_id=None, latest=True)
                    if rid is not None:
                        run_dir = self._run_dir(subdir, year, rid)
                        files.extend(sorted(run_dir.glob("R*.parquet")))
                else:
                    year_dir = d / f"{year}"
                    if year_dir.exists():
                        for run_dir in self._sort_paths_by_mtime(
                            p for p in year_dir.glob("run_*") if p.is_dir()
                        ):
                            files.extend(sorted(run_dir.glob("R*.parquet")))

            if include_legacy:
                files.extend(sorted(d.glob(f"{year}_R*.parquet")))
            return files

        # year=None -> all years
        if run_id is not None:
            rid = self._normalise_run_id(run_id)
            for y_dir in sorted([p for p in d.glob("*") if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
                run_dir = y_dir / f"run_{rid}"
                files.extend(sorted(run_dir.glob("R*.parquet")))
        elif latest:
            year_dirs = sorted(
                [p for p in d.glob("*") if p.is_dir() and p.name.isdigit()],
                key=lambda p: int(p.name),
            )
            for y_dir in year_dirs:
                y = int(y_dir.name)
                rid = self._resolve_read_run_id(subdir, y, run_id=None, latest=True)
                if rid is not None:
                    run_dir = self._run_dir(subdir, y, rid)
                    files.extend(sorted(run_dir.glob("R*.parquet")))
        else:
            files.extend(sorted(d.glob("**/R*.parquet")))

        if include_legacy:
            files.extend(sorted(d.glob("*_R*.parquet")))

        return files

    def _load_records(self, subdir, year=None, round_num=None, run_id=None, latest=True):
        """Generic loader for round-partitioned parquet files."""
        if year is not None and round_num is not None:
            path = self._round_path(
                subdir, year, round_num, run_id=run_id, for_write=False, latest=latest
            )
            if path.exists() and path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.DataFrame()

        files = self._list_record_files(
            subdir, year=year, run_id=run_id, latest=latest, include_legacy=True
        )

        if not files:
            return pd.DataFrame()

        if latest:
            files = self._dedupe_latest_files(files)

        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def _dedupe_latest_files(self, files):
        """Prefer the first file for a (year, round) pair when mixing layouts."""
        deduped = []
        seen = set()

        for path in files:
            key = self._parse_year_round(path)
            if key is None:
                key = ("path", str(path))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)

        return deduped

    def __repr__(self):
        summary = self.get_learning_summary()
        return (
            f"LearningStore(base_dir={self.base_dir}, run_id={summary.get('run_id')})\n"
            f"  Predictions: {summary['prediction_rounds']} rounds\n"
            f"  Outcomes:    {summary['outcome_rounds']} rounds\n"
            f"  Diagnostics: {summary['diagnostic_rounds']} rounds\n"
            f"  Archetypes:  {'yes' if summary['has_archetypes'] else 'no'}\n"
            f"  Concessions: {'yes' if summary['has_concessions'] else 'no'}\n"
            f"  Calibration: {'yes' if summary['has_calibration'] else 'no'}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = LearningStore()
    print(store)
