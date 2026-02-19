#!/usr/bin/env python3
"""Compare experiment results across configs.

Finds the latest result CSV for each experiment config and prints a
summary table. By default compares all exp1 top-k configs.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --results-dir results --pattern "exp1_topk_*"
    python scripts/compare_results.py --files results/exp1_topk_1_evaluation_2.csv results/exp1_topk_5_evaluation_1.csv
"""

import csv
import statistics
import argparse
import re
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def avg(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", "nan", None)]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(path: Path) -> dict:
    rows = load_csv(path)
    ans   = [r for r in rows if r["answerable"] == "True"]
    unans = [r for r in rows if r["answerable"] == "False"]

    gen_lats = [float(r["generation_latency_ms"]) for r in rows
                if r.get("generation_latency_ms") not in ("", None)]
    hallu = sum(1 for r in rows if r.get("has_hallucination") == "True")
    recall_zero = sum(1 for r in ans if float(r["retrieval_recall"]) == 0)

    return {
        "n":              len(rows),
        "n_ans":          len(ans),
        "n_unans":        len(unans),
        "recall_ans":     avg(ans,   "retrieval_recall"),
        "miss_pct":       recall_zero / len(ans) if ans else float("nan"),
        "correctness_ans": avg(ans,  "answer_correctness"),
        "hallu_rate":     hallu / len(rows),
        "abstention":     avg(unans, "abstention_score"),
        "retr_lat":       avg(rows,  "retrieval_latency_ms"),
        "gen_lat_median": statistics.median(gen_lats) if gen_lats else float("nan"),
        "total_lat":      avg(rows,  "total_latency_ms"),
    }


def find_latest(results_dir: Path, pattern: str) -> list[Path]:
    """For each unique config name matching pattern, return the highest-indexed CSV."""
    escaped = re.escape(pattern).replace(r"\*", ".*")
    regex = re.compile(rf"^({escaped})_evaluation_(\d+)\.csv$")
    latest: dict[str, tuple[int, Path]] = {}
    for f in results_dir.glob("*.csv"):
        m = regex.match(f.name)
        if m:
            config, idx = m.group(1), int(m.group(2))
            if config not in latest or idx > latest[config][0]:
                latest[config] = (idx, f)
    def sort_key(item):
        name = item[0]
        nums = [int(x) for x in re.findall(r"\d+", name)]
        return (re.sub(r"\d+", "", name), nums)

    return [path for _, (_, path) in sorted(latest.items(), key=sort_key)]


def print_table(rows: list[tuple[str, dict]]) -> None:
    col_w = max(len(label) for label, _ in rows)

    header = (
        f"{'Config':<{col_w}}  {'N':>4}  {'Recall':>6}  {'Miss%':>5}  "
        f"{'Correct':>7}  {'Hallu%':>6}  {'Abstain':>7}  "
        f"{'RetrMs':>6}  {'GenMs(med)':>10}  {'TotalMs':>7}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for label, s in rows:
        miss = f"{s['miss_pct']*100:.1f}%"
        hallu = f"{s['hallu_rate']*100:.1f}%"
        print(
            f"{label:<{col_w}}  {s['n']:>4}  {s['recall_ans']:>6.3f}  {miss:>5}  "
            f"{s['correctness_ans']:>7.3f}  {hallu:>6}  {s['abstention']:>7.3f}  "
            f"{s['retr_lat']:>6.0f}  {s['gen_lat_median']:>10.0f}  {s['total_lat']:>7.0f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare experiment result CSVs")
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing result CSVs (default: results)"
    )
    parser.add_argument(
        "--pattern", default="exp1_topk_*",
        help="Glob pattern for config names to include (default: exp1_topk_*)"
    )
    parser.add_argument(
        "--files", nargs="+",
        help="Explicit CSV file paths (overrides --pattern)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        results_dir = project_root / args.results_dir
        paths = find_latest(results_dir, args.pattern)

    if not paths:
        print(f"No result files found (pattern={args.pattern!r} in {args.results_dir})")
        return

    rows = []
    for path in paths:
        # Derive a short label from the filename: strip _evaluation_N.csv suffix
        label = re.sub(r"_evaluation_\d+$", "", path.stem)
        rows.append((label, summarize(path)))

    print_table(rows)


if __name__ == "__main__":
    main()
