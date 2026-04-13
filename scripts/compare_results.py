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
    recall_zero = sum(1 for r in ans if float(r["retrieval_recall"]) == 0)

    # Detect schema version by column names
    is_v2 = "factual_correctness" in rows[0] if rows else False

    summary = {
        "n":              len(rows),
        "n_ans":          len(ans),
        "n_unans":        len(unans),
        "recall_ans":     avg(ans,   "retrieval_recall"),
        "miss_pct":       recall_zero / len(ans) if ans else float("nan"),
        "abstention":     avg(unans, "abstention_score"),
        "retr_lat":       avg(rows,  "retrieval_latency_ms"),
        "gen_lat_median": statistics.median(gen_lats) if gen_lats else float("nan"),
        "total_lat":      avg(rows,  "total_latency_ms"),
        "schema":         2 if is_v2 else 1,
    }

    if is_v2:
        summary["correctness_ans"] = avg(ans, "factual_correctness")
        summary["faithfulness"]    = avg(ans, "faithfulness")
        summary["ctx_precision"]   = avg(ans, "context_precision")
        summary["hallu_rate"]      = None
    else:
        summary["correctness_ans"] = avg(ans, "answer_correctness")
        hallu = sum(1 for r in rows if r.get("has_hallucination") == "True")
        summary["hallu_rate"]      = hallu / len(rows) if rows else float("nan")
        summary["faithfulness"]    = None
        summary["ctx_precision"]   = None

    return summary


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

    # Detect if any rows use v2 schema
    any_v2 = any(s["schema"] == 2 for _, s in rows)
    any_v1 = any(s["schema"] == 1 for _, s in rows)

    if any_v2 and not any_v1:
        # Pure v2: show ragas metrics
        header = (
            f"{'Config':<{col_w}}  {'N':>4}  {'Recall':>6}  {'Miss%':>5}  "
            f"{'Correct':>7}  {'Faith':>6}  {'CtxPrec':>7}  {'Abstain':>7}  "
            f"{'RetrMs':>6}  {'GenMs(med)':>10}  {'TotalMs':>7}"
        )
        sep = "-" * len(header)
        print(header)
        print(sep)
        for label, s in rows:
            miss = f"{s['miss_pct']*100:.1f}%"
            faith = f"{s['faithfulness']:.3f}" if s["faithfulness"] is not None else "   -  "
            ctx_p = f"{s['ctx_precision']:.3f}" if s["ctx_precision"] is not None else "     - "
            print(
                f"{label:<{col_w}}  {s['n']:>4}  {s['recall_ans']:>6.3f}  {miss:>5}  "
                f"{s['correctness_ans']:>7.3f}  {faith:>6}  {ctx_p:>7}  {s['abstention']:>7.3f}  "
                f"{s['retr_lat']:>6.0f}  {s['gen_lat_median']:>10.0f}  {s['total_lat']:>7.0f}"
            )
    elif any_v1 and not any_v2:
        # Pure v1: show legacy metrics
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
    else:
        # Mixed v1+v2: show all columns, dashes for unavailable
        header = (
            f"{'Config':<{col_w}}  {'N':>4}  {'Recall':>6}  {'Miss%':>5}  "
            f"{'Correct':>7}  {'Hallu%':>6}  {'Faith':>6}  {'CtxPrec':>7}  {'Abstain':>7}  "
            f"{'RetrMs':>6}  {'GenMs(med)':>10}  {'TotalMs':>7}"
        )
        sep = "-" * len(header)
        print(header)
        print(sep)
        for label, s in rows:
            miss = f"{s['miss_pct']*100:.1f}%"
            hallu = f"{s['hallu_rate']*100:.1f}%" if s["hallu_rate"] is not None else "    - "
            faith = f"{s['faithfulness']:.3f}" if s["faithfulness"] is not None else "   -  "
            ctx_p = f"{s['ctx_precision']:.3f}" if s["ctx_precision"] is not None else "     - "
            print(
                f"{label:<{col_w}}  {s['n']:>4}  {s['recall_ans']:>6.3f}  {miss:>5}  "
                f"{s['correctness_ans']:>7.3f}  {hallu:>6}  {faith:>6}  {ctx_p:>7}  {s['abstention']:>7.3f}  "
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
