"""Quality control: compute invalid rate, error rate, V4 pass rate per model.

Usage:
    python quality_control.py --dir data/raw/phi4_t1.0
"""

import argparse
import pathlib
import sys
from collections import Counter

import jsonlines

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "04_parse"))

from parse_responses import parse_response


def analyze_responses(responses_file: pathlib.Path) -> dict:
    """Analyze a JSONL file and return QC metrics."""
    total = 0
    errors = 0
    invalid = 0
    valid = 0
    v4_total = 0
    v4_chose_a = 0

    form_invalid = Counter()
    form_total = Counter()

    with jsonlines.open(responses_file) as reader:
        for record in reader:
            total += 1

            if record.get("status") == "error":
                errors += 1
                continue

            form_id = record.get("form_id", "")
            response_text = record.get("response") or ""

            # Parse
            choice = parse_response(response_text, form_id)

            form_total[form_id] += 1
            if choice == "invalid":
                invalid += 1
                form_invalid[form_id] += 1
            else:
                valid += 1

            # V4 dominance check
            version = record.get("version", "")
            if version == "V4":
                v4_total += 1
                if choice == "A":
                    v4_chose_a += 1

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "errors": errors,
        "invalid_rate": invalid / max(total - errors, 1),
        "error_rate": errors / max(total, 1),
        "v4_total": v4_total,
        "v4_pass_rate": v4_chose_a / max(v4_total, 1),
        "form_invalid": dict(form_invalid),
        "form_total": dict(form_total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to model output directory")
    args = parser.parse_args()

    responses_file = pathlib.Path(args.dir) / "responses.jsonl"
    if not responses_file.exists():
        print(f"No responses.jsonl in {args.dir}")
        return 1

    metrics = analyze_responses(responses_file)

    print(f"QC Report: {args.dir}")
    print(f"  Total: {metrics['total']}")
    print(f"  Valid: {metrics['valid']}")
    print(f"  Invalid: {metrics['invalid']} ({metrics['invalid_rate']:.1%})")
    print(f"  Errors: {metrics['errors']} ({metrics['error_rate']:.1%})")
    print(f"  V4 pass rate: {metrics['v4_pass_rate']:.1%} ({metrics['v4_total']} V4 tasks)")

    # Per-form invalid rates
    rep_invalid = sum(v for k, v in metrics["form_invalid"].items() if "REP" in k)
    rep_total = sum(v for k, v in metrics["form_total"].items() if "REP" in k)
    if rep_total > 0:
        rep_rate = rep_invalid / rep_total
        print(f"  REP-form invalid rate: {rep_rate:.1%}")
        if rep_rate > 0.15:
            print("  WARNING: REP invalid rate > 15% — consider excluding REP-forms")

    # Alerts
    if metrics["invalid_rate"] > 0.05:
        print("  ALERT: Invalid rate > 5%!")
    if metrics["v4_pass_rate"] < 0.80 and metrics["v4_total"] > 0:
        print("  ALERT: V4 pass rate < 80%!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
