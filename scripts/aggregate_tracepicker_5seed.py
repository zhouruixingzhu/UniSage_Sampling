"""Aggregate per-seed v0.2 TracePicker `eval/trace_eval_summary.csv` files
into a mean ± std summary across N seeds (default N=5, Rev C-Q4).

This mirrors `scripts/aggregate_sifter_5seed.py` exactly so TracePicker and Sifter
5-seed tables are produced under the same convention:

- Each per-seed summary already contains `<metric>_mean`, `<metric>_std`,
  `<metric>_n` columns where the within-seed mean/std is over cases.
- This aggregator treats each seed's `<metric>_mean` as one observation and
  computes the across-seed mean / std (ddof=1) / min / max for every metric,
  keyed by (dataset, ranking, budget).

Run-dir naming for TracePicker primary scope (no fingerprint suffix in the
hand-named seeds):
  tracepicker_v02_{dataset_key}_primary_portable_seed{N}
where `dataset_key` is `hip_tt` (combined hipster+train_ticket primary) or
`aiops21`.

Usage:
  python scripts/aggregate_tracepicker_5seed.py --auto \
      --datasets hip_tt aiops21 \
      --root  results/v02_sampling_runs \
      --out-root results/v02_sampling_runs
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


KEY_COLS = ["dataset", "ranking", "budget"]


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_seed_from_dirname(name: str) -> int:
    # Matches both `..._seed-1_...` (Sifter convention) and
    # `..._seed1` / `..._seed1_jsd` (TracePicker hand-named).
    m = re.search(r"_seed-?(\d+)(?:[_/]|$)", name)
    if not m:
        raise ValueError(f"could not parse seed from dir name: {name}")
    return int(m.group(1))


def discover_runs(root: Path, dataset_key: str, scope: str) -> List[Path]:
    """Find seed1..seedN dirs matching the dataset key + scope (portable backend)."""
    pattern = re.compile(
        rf"^tracepicker_v02_{re.escape(dataset_key)}_{re.escape(scope)}_portable_seed(\d+)$"
    )
    matches: List[tuple[int, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            matches.append((int(m.group(1)), child))
    matches.sort(key=lambda kv: kv[0])
    return [p for _, p in matches]


def load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "eval" / "trace_eval_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    df = pd.read_csv(path)
    df["__seed__"] = parse_seed_from_dirname(run_dir.name)
    df["__run_dir__"] = run_dir.name
    return df


def aggregate(runs: Sequence[Path]) -> tuple[pd.DataFrame, dict]:
    if len(runs) < 2:
        raise SystemExit(f"need at least 2 runs to aggregate, got {len(runs)}")

    per_seed = [load_summary(r) for r in runs]
    seeds = [int(df["__seed__"].iloc[0]) for df in per_seed]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds detected: {seeds}")

    key_sets = [tuple(sorted(map(tuple, df[KEY_COLS].values.tolist()))) for df in per_seed]
    if len(set(key_sets)) != 1:
        raise SystemExit("per-seed (dataset,ranking,budget) keys differ across runs")

    stacked = pd.concat(per_seed, ignore_index=True)
    mean_cols = [c for c in stacked.columns if c.endswith("_mean")]
    if not mean_cols:
        raise SystemExit("no `*_mean` columns found in summaries")
    base_metrics = [c[: -len("_mean")] for c in mean_cols]

    grouped = stacked.groupby(KEY_COLS, sort=False)

    rows: List[dict] = []
    for key, group in grouped:
        row = dict(zip(KEY_COLS, key))
        row["n_seeds"] = len(group)
        if "cases" in group.columns:
            unique_cases = sorted(group["cases"].unique().tolist())
            row["cases"] = unique_cases[0] if len(unique_cases) == 1 else f"mixed:{unique_cases}"

        for base in base_metrics:
            vals = group[f"{base}_mean"].astype(float).to_numpy()
            row[f"{base}_seedmean"] = float(np.mean(vals))
            row[f"{base}_seedstd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{base}_seedmin"] = float(np.min(vals))
            row[f"{base}_seedmax"] = float(np.max(vals))

        rows.append(row)

    agg = pd.DataFrame(rows)
    agg = agg.sort_values(KEY_COLS).reset_index(drop=True)

    provenance = {
        "schema_version": "tracepicker_v02_5seed_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_seeds": int(len(runs)),
        "seeds": [int(s) for s in seeds],
        "base_metrics": base_metrics,
        "key_cols": KEY_COLS,
        "ddof": 1,
        "runs": [
            {
                "seed": int(parse_seed_from_dirname(r.name)),
                "run_dir": str(r),
                "summary_path": str(r / "eval" / "trace_eval_summary.csv"),
                "summary_sha256": sha256_file(r / "eval" / "trace_eval_summary.csv"),
            }
            for r in runs
        ],
    }
    return agg, provenance


HEADLINE_METRICS = [
    "A_exposed_recall",
    "A_exposed_prop",
    "struct_unseen_recall",
    "struct_rare_le_0p1_recall",
    "lat_tail_q99_recall",
    "pattern_coverage",
    "pattern_entropy",
    "pattern_jsd",
]


def render_markdown(agg: pd.DataFrame, provenance: dict) -> str:
    lines = []
    lines.append("# TracePicker v0.2 — 5-seed mean ± std (Rev C-Q4)")
    lines.append("")
    lines.append(f"- Generated: {provenance['generated_at']}")
    lines.append(f"- Seeds: {provenance['seeds']}")
    lines.append(f"- Aggregation convention: each seed's per-case `<metric>_mean` "
                 f"treated as one observation; std uses ddof=1.")
    lines.append(f"- Source runs ({provenance['n_seeds']}):")
    for r in provenance["runs"]:
        lines.append(f"  - seed {r['seed']}: `{Path(r['run_dir']).name}`")
    lines.append("")
    lines.append("## Headline metrics (mean ± std across 5 seeds)")
    lines.append("")
    for metric in HEADLINE_METRICS:
        mean_col = f"{metric}_seedmean"
        std_col = f"{metric}_seedstd"
        if mean_col not in agg.columns:
            continue
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| dataset | ranking | budget | mean | std | min | max |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in agg.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['ranking']} | {row['budget']:.3f} "
                f"| {row[mean_col]:.4f} | {row[std_col]:.4f} "
                f"| {row[f'{metric}_seedmin']:.4f} | {row[f'{metric}_seedmax']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, agg: pd.DataFrame, provenance: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trace_eval_5seed.csv"
    agg.to_csv(csv_path, index=False)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out_dir / "summary.md").write_text(render_markdown(agg, provenance))
    print(f"[ok] wrote {csv_path}  rows={len(agg)}  cols={len(agg.columns)}")
    print(f"[ok] wrote {out_dir / 'provenance.json'}")
    print(f"[ok] wrote {out_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, default=None,
                        help="Explicit list of run directories to aggregate.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (required when --runs is given).")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover seed1..N dirs in --root by (dataset, scope).")
    parser.add_argument("--root", type=Path, default=Path("results/v02_sampling_runs"))
    parser.add_argument("--datasets", nargs="+", default=["hip_tt", "aiops21"],
                        help="Dataset keys (as encoded in run dir names) to aggregate.")
    parser.add_argument("--scope", default="primary")
    parser.add_argument("--out-root", type=Path, default=Path("results/v02_sampling_runs"))
    args = parser.parse_args()

    if args.runs:
        if args.out is None:
            raise SystemExit("--out is required when --runs is provided")
        agg, prov = aggregate(args.runs)
        write_outputs(args.out, agg, prov)
        return

    if not args.auto:
        raise SystemExit("either pass --runs ... --out ... or use --auto")

    for dataset_key in args.datasets:
        runs = discover_runs(args.root, dataset_key, args.scope)
        print(f"[discover] {dataset_key}/{args.scope}: {len(runs)} runs")
        for r in runs:
            print(f"  - {r.name}")
        if len(runs) < 2:
            print(f"  [skip] need >=2 runs, got {len(runs)}")
            continue
        out_dir = args.out_root / (
            f"tracepicker_v02_{len(runs)}seed_aggregate_{dataset_key}_{args.scope}_portable"
        )
        agg, prov = aggregate(runs)
        write_outputs(out_dir, agg, prov)


if __name__ == "__main__":
    main()
