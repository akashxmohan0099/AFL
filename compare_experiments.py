"""
Compare A/B experiment results for the AFL prediction pipeline.

Loads experiment JSON files and prints a comparison table showing
incremental improvements from each data source integration.

Usage:
    python compare_experiments.py
    python compare_experiments.py data/experiments/  # custom directory
"""

import json
import sys
from pathlib import Path


try:
    import config
    EXPERIMENTS_DIR = config.EXPERIMENTS_DIR
except ImportError:
    EXPERIMENTS_DIR = Path("data/experiments")


def load_experiments(exp_dir=None):
    """Load all experiment JSON files from the experiments directory."""
    d = Path(exp_dir) if exp_dir else EXPERIMENTS_DIR
    if not d.exists():
        print(f"No experiments directory found at {d}")
        return []

    experiments = []
    for f in sorted(d.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        data["_filename"] = f.name
        experiments.append(data)

    return experiments


def extract_metrics(exp):
    """Extract key metrics from an experiment result dict."""
    thresholds = exp.get("thresholds", {})
    mae_info = exp.get("mae", {})
    scorer = exp.get("scorer_auc", {})
    cal_ece = exp.get("calibration_ece")

    # Disposal metrics — check both thresholds dict and legacy disposal_thresholds dict
    disp_thresholds = exp.get("disposal_thresholds", {})
    disp_mae = mae_info.get("disposals")

    # Marks metrics
    marks_thresholds = exp.get("marks_thresholds", {})
    marks_mae = mae_info.get("marks")

    def _get_disp(key, field):
        """Get disposal metric from thresholds or disposal_thresholds."""
        val = thresholds.get(key, {}).get(field)
        if val is None:
            val = disp_thresholds.get(key, {}).get(field)
        return val

    def _get_marks(key, field):
        """Get marks metric from thresholds or marks_thresholds."""
        val = thresholds.get(key, {}).get(field)
        if val is None:
            val = marks_thresholds.get(key, {}).get(field)
        return val

    # Compute BSS from Brier if not stored
    def _get_bss(key, getter=None):
        getter = getter or _get_disp
        bss = getter(key, "brier_skill_score")
        if bss is not None:
            return bss
        brier = getter(key, "brier_score")
        base_rate = getter(key, "base_rate")
        if brier is not None and base_rate is not None and base_rate > 0:
            naive = base_rate * (1 - base_rate)
            return round((1 - brier / naive) * 100, 1) if naive > 0 else None
        return None

    return {
        "label": exp.get("label", exp.get("_filename", "?")),
        "mae": mae_info.get("goals"),
        "brier_1plus": thresholds.get("1plus_goals", {}).get("brier_score"),
        "brier_2plus": thresholds.get("2plus_goals", {}).get("brier_score"),
        "brier_3plus": thresholds.get("3plus_goals", {}).get("brier_score"),
        "auc": scorer.get("overall"),
        "p20_1plus": thresholds.get("1plus_goals", {}).get("hit_rate_p50"),
        "ece": cal_ece,
        "improvement_pct": mae_info.get("improvement_pct"),
        # Disposal metrics
        "disp_mae": disp_mae,
        "disp_brier_15": _get_disp("15plus_disp", "brier_score"),
        "disp_brier_20": _get_disp("20plus_disp", "brier_score"),
        "disp_brier_25": _get_disp("25plus_disp", "brier_score"),
        "disp_brier_30": _get_disp("30plus_disp", "brier_score"),
        "disp_bss_15": _get_bss("15plus_disp"),
        "disp_bss_20": _get_bss("20plus_disp"),
        "disp_bss_25": _get_bss("25plus_disp"),
        "disp_bss_30": _get_bss("30plus_disp"),
        # Marks metrics
        "marks_mae": marks_mae,
        "marks_brier_3": _get_marks("3plus_mk", "brier_score"),
        "marks_brier_5": _get_marks("5plus_mk", "brier_score"),
        "marks_brier_7": _get_marks("7plus_mk", "brier_score"),
        "marks_bss_3": _get_bss("3plus_mk", _get_marks),
        "marks_bss_5": _get_bss("5plus_mk", _get_marks),
        "marks_bss_7": _get_bss("7plus_mk", _get_marks),
    }


def print_comparison(experiments):
    """Print formatted comparison table."""
    if not experiments:
        print("No experiments to compare.")
        return

    rows = [extract_metrics(e) for e in experiments]

    # Check if any experiment has disposal metrics
    has_disp = any(r.get("disp_mae") is not None or r.get("disp_brier_20") is not None
                   for r in rows)

    # Header
    print()
    print(f"{'#':<3s} {'Experiment':<40s} {'MAE':>7s} {'Brier1+':>8s} "
          f"{'Brier2+':>8s} {'AUC':>7s} {'P@50':>7s} {'ECE':>7s}")
    print("-" * 95)

    baseline_row = rows[0] if rows else None

    for i, row in enumerate(rows):
        label = row["label"][:39]
        mae = f"{row['mae']:.4f}" if row["mae"] is not None else "  N/A"
        b1 = f"{row['brier_1plus']:.4f}" if row["brier_1plus"] is not None else "  N/A"
        b2 = f"{row['brier_2plus']:.4f}" if row["brier_2plus"] is not None else "  N/A"
        auc = f"{row['auc']:.4f}" if row["auc"] is not None else "  N/A"
        p20 = f"{row['p20_1plus']:.4f}" if row["p20_1plus"] is not None else "  N/A"
        ece = f"{row['ece']:.4f}" if row["ece"] is not None else "  N/A"
        print(f"{i:<3d} {label:<40s} {mae:>7s} {b1:>8s} {b2:>8s} {auc:>7s} {p20:>7s} {ece:>7s}")

    # Disposal metrics table (if available)
    if has_disp:
        print()
        print(f"{'#':<3s} {'Experiment':<40s} {'DI MAE':>7s} {'B15+':>8s} "
              f"{'B20+':>8s} {'B25+':>8s} {'B30+':>8s} {'BSS15':>7s} {'BSS20':>7s} {'BSS25':>7s} {'BSS30':>7s}")
        print("-" * 125)
        for i, row in enumerate(rows):
            label = row["label"][:39]
            dmae = f"{row['disp_mae']:.4f}" if row.get("disp_mae") is not None else "  N/A"
            db15 = f"{row['disp_brier_15']:.4f}" if row.get("disp_brier_15") is not None else "  N/A"
            db20 = f"{row['disp_brier_20']:.4f}" if row.get("disp_brier_20") is not None else "  N/A"
            db25 = f"{row['disp_brier_25']:.4f}" if row.get("disp_brier_25") is not None else "  N/A"
            db30 = f"{row['disp_brier_30']:.4f}" if row.get("disp_brier_30") is not None else "  N/A"
            bss15 = f"{row['disp_bss_15']:.1f}%" if row.get("disp_bss_15") is not None else "  N/A"
            bss20 = f"{row['disp_bss_20']:.1f}%" if row.get("disp_bss_20") is not None else "  N/A"
            bss25 = f"{row['disp_bss_25']:.1f}%" if row.get("disp_bss_25") is not None else "  N/A"
            bss30 = f"{row['disp_bss_30']:.1f}%" if row.get("disp_bss_30") is not None else "  N/A"
            print(f"{i:<3d} {label:<40s} {dmae:>7s} {db15:>8s} {db20:>8s} {db25:>8s} {db30:>8s} {bss15:>7s} {bss20:>7s} {bss25:>7s} {bss30:>7s}")

    # Marks metrics table (if available)
    has_marks = any(r.get("marks_mae") is not None or r.get("marks_brier_3") is not None
                    for r in rows)
    if has_marks:
        print()
        print(f"{'#':<3s} {'Experiment':<40s} {'MK MAE':>7s} {'B3+':>8s} "
              f"{'B5+':>8s} {'B7+':>8s} {'BSS3':>7s} {'BSS5':>7s} {'BSS7':>7s}")
        print("-" * 105)
        for i, row in enumerate(rows):
            label = row["label"][:39]
            mmae = f"{row['marks_mae']:.4f}" if row.get("marks_mae") is not None else "  N/A"
            mb3 = f"{row['marks_brier_3']:.4f}" if row.get("marks_brier_3") is not None else "  N/A"
            mb5 = f"{row['marks_brier_5']:.4f}" if row.get("marks_brier_5") is not None else "  N/A"
            mb7 = f"{row['marks_brier_7']:.4f}" if row.get("marks_brier_7") is not None else "  N/A"
            bss3 = f"{row['marks_bss_3']:.1f}%" if row.get("marks_bss_3") is not None else "  N/A"
            bss5 = f"{row['marks_bss_5']:.1f}%" if row.get("marks_bss_5") is not None else "  N/A"
            bss7 = f"{row['marks_bss_7']:.1f}%" if row.get("marks_bss_7") is not None else "  N/A"
            print(f"{i:<3d} {label:<40s} {mmae:>7s} {mb3:>8s} {mb5:>8s} {mb7:>8s} {bss3:>7s} {bss5:>7s} {bss7:>7s}")

    # Cumulative delta from baseline
    if baseline_row and len(rows) > 1:
        last = rows[-1]
        print()
        print("Cumulative delta (last vs baseline):")

        for key, label, lower_better in [
            ("mae", "MAE", True),
            ("brier_1plus", "Brier 1+", True),
            ("brier_2plus", "Brier 2+", True),
            ("auc", "AUC", False),
            ("ece", "ECE", True),
        ]:
            base_val = baseline_row.get(key)
            last_val = last.get(key)
            if base_val is not None and last_val is not None:
                delta = last_val - base_val
                pct = delta / base_val * 100 if base_val != 0 else 0
                direction = "improvement" if (delta < 0) == lower_better else "regression"
                print(f"  {label:<12s}: {delta:+.4f} ({abs(pct):.1f}% {direction})")

    # Step-by-step deltas
    if len(rows) > 1:
        print()
        print("Step-by-step deltas (Brier 1+):")
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]
            prev_b1 = prev.get("brier_1plus")
            curr_b1 = curr.get("brier_1plus")
            if prev_b1 is not None and curr_b1 is not None:
                delta = curr_b1 - prev_b1
                label = curr["label"][:40]
                verdict = "KEEP" if delta <= 0.001 else "REVERT"
                print(f"  Step {i}: {delta:+.4f}  {label}  [{verdict}]")


if __name__ == "__main__":
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else None
    experiments = load_experiments(exp_dir)
    if not experiments:
        # Also try loading baseline files
        baseline = Path("baseline_v3.2.json")
        if baseline.exists():
            with open(baseline) as f:
                experiments.append(json.load(f))

    print_comparison(experiments)
