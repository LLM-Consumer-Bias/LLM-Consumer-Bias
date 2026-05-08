"""Phase 6 Step 2: Run QC across all data directories, produce summary table.

Output: results/qc_summary.csv with columns:
    model, temperature, condition, total, valid, invalid, invalid_rate,
    error_rate, v4_pass_rate, rep_invalid_rate

Usage:
    python scripts/05_analysis/quality_control_all.py
"""

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "04_parse"))
sys.path.insert(0, str(ROOT / "scripts" / "05_analysis"))

from quality_control import analyze_responses
from preprocessing import parse_directory_name

RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"


def main():
    print("=" * 60)
    print("Phase 6 Step 2: Quality Control Summary")
    print("=" * 60)

    rows = []
    raw_dirs = sorted(RAW_DIR.iterdir()) if RAW_DIR.exists() else []

    for d in raw_dirs:
        if not d.is_dir():
            continue
        jsonl_file = d / "responses.jsonl"
        if not jsonl_file.exists():
            continue

        dir_meta = parse_directory_name(d.name)
        print(f"\n  Analyzing {d.name}...")

        metrics = analyze_responses(jsonl_file)

        # REP-form invalid rate
        rep_invalid = sum(v for k, v in metrics["form_invalid"].items() if "REP" in k)
        rep_total = sum(v for k, v in metrics["form_total"].items() if "REP" in k)
        rep_invalid_rate = rep_invalid / max(rep_total, 1)

        row = {
            "model": dir_meta["model"],
            "temperature": dir_meta["temperature"],
            "condition": dir_meta["condition"],
            "directory": d.name,
            "total": metrics["total"],
            "valid": metrics["valid"],
            "invalid": metrics["invalid"],
            "errors": metrics["errors"],
            "invalid_rate": metrics["invalid_rate"],
            "error_rate": metrics["error_rate"],
            "v4_total": metrics["v4_total"],
            "v4_pass_rate": metrics["v4_pass_rate"],
            "rep_total": rep_total,
            "rep_invalid": rep_invalid,
            "rep_invalid_rate": rep_invalid_rate,
        }
        rows.append(row)

        # Print alerts
        status = "OK"
        alerts = []
        if metrics["invalid_rate"] > 0.05:
            alerts.append(f"INVALID={metrics['invalid_rate']:.1%}")
        if metrics["v4_pass_rate"] < 0.80 and metrics["v4_total"] > 0:
            alerts.append(f"V4={metrics['v4_pass_rate']:.1%}")
        if rep_invalid_rate > 0.15 and rep_total > 0:
            alerts.append(f"REP={rep_invalid_rate:.1%}")
        if alerts:
            status = "ALERT: " + ", ".join(alerts)

        print(f"    Total={metrics['total']:,}  Valid={metrics['valid']:,}  "
              f"Invalid={metrics['invalid_rate']:.2%}  "
              f"V4={metrics['v4_pass_rate']:.1%}  [{status}]")

    if not rows:
        print("\nNo data directories found!")
        return

    # Save summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "qc_summary.csv", index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved: results/qc_summary.csv ({len(df)} directories)")
    print(f"{'=' * 60}")

    # Print summary table
    print("\n" + df.to_string(index=False))

    # Overall statistics
    print(f"\n--- Overall ---")
    print(f"Total records: {df['total'].sum():,}")
    print(f"Valid: {df['valid'].sum():,}")
    print(f"Invalid: {df['invalid'].sum():,} ({df['invalid'].sum() / max(df['total'].sum(), 1):.2%})")
    print(f"Errors: {df['errors'].sum():,}")
    print(f"Mean V4 pass rate: {df['v4_pass_rate'].mean():.1%}")


if __name__ == "__main__":
    main()
