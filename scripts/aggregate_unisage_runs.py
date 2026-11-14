"""Aggregate UniSage-v1 budget-8 reps 1-5 into mean / std / min / max artifacts.

UniSage-v1 is deterministic by design: rep1..rep5 produce bit-equal trace and
log eval summaries under the locked config (UNISAGE_EDGE_LAT=raw,
UNISAGE_RCA_ATTR=target, UNISAGE_PROPAGATE=1, UNISAGE_PROPAGATE_GUARD=1,
UNISAGE_TRACE_COMPOSITION=dynamic_product_pattern_guard,
UNISAGE_LOG_COMPOSITION=product, 8-budget grid). The aggregate is still
produced so the artifact set is symmetric with sifter / gleaner / tracepicker /
trastrainer 5-seed aggregates and so the repeated-run protocol is verifiable
from on-disk evidence (every `*_repstd` column should be 0.0).

Usage (auto-discover):
  python scripts/aggregate_unisage_runs.py --auto

Or explicit:
  python scripts/aggregate_unisage_runs.py \
      --kind trace \
      --runs results/v02_sampling_runs/unisage_v1_rep{1,2,3,4,5}_budget8_hip_tt_primary \
      --out  results/v02_sampling_runs/unisage_v1_budget8_5rep_aggregate_hip_tt_primary
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


KEY_COLS = ["dataset", "ranking", "budget"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_rep_from_dirname(name: str) -> int:
    m = re.search(r"_rep(\d+)_budget8(?:$|_)", name)
    if not m:
        raise ValueError(f"could not parse rep from dir name: {name}")
    return int(m.group(1))


def trace_summary_path(run_dir: Path) -> Path:
    return run_dir / "eval" / "trace_eval_summary.csv"


def log_summary_path(run_dir: Path) -> Path:
    return run_dir / "log_eval_clean" / "log_eval_summary.csv"


def load_summary(run_dir: Path, kind: str) -> pd.DataFrame:
    if kind == "trace":
        path = trace_summary_path(run_dir)
    elif kind == "log":
        path = log_summary_path(run_dir)
    else:
        raise ValueError(f"unknown kind: {kind}")
    if not path.exists():
        raise FileNotFoundError(f"missing summary: {path}")
    df = pd.read_csv(path)
    df["__rep__"] = parse_rep_from_dirname(run_dir.name)
    df["__run_dir__"] = run_dir.name
    return df


def aggregate(runs: Sequence[Path], kind: str) -> tuple[pd.DataFrame, dict]:
    if len(runs) < 2:
        raise SystemExit(f"need at least 2 runs to aggregate, got {len(runs)}")

    per_rep = [load_summary(r, kind) for r in runs]
    reps = [int(df["__rep__"].iloc[0]) for df in per_rep]
    if len(set(reps)) != len(reps):
        raise SystemExit(f"duplicate reps detected: {reps}")

    key_sets = [tuple(sorted(map(tuple, df[KEY_COLS].values.tolist()))) for df in per_rep]
    if len(set(key_sets)) != 1:
        raise SystemExit("per-rep (dataset,ranking,budget) keys differ across runs")

    stacked = pd.concat(per_rep, ignore_index=True)
    mean_cols = [c for c in stacked.columns if c.endswith("_mean")]
    if not mean_cols:
        raise SystemExit("no `*_mean` columns found in summaries")
    base_metrics = [c[: -len("_mean")] for c in mean_cols]

    grouped = stacked.groupby(KEY_COLS, sort=False)

    rows: List[dict] = []
    for key, group in grouped:
        row = dict(zip(KEY_COLS, key))
        row["n_reps"] = len(group)
        if "cases" in group.columns:
            unique_cases = sorted(group["cases"].unique().tolist())
            row["cases"] = unique_cases[0] if len(unique_cases) == 1 else f"mixed:{unique_cases}"

        for base in base_metrics:
            vals = group[f"{base}_mean"].astype(float).to_numpy()
            row[f"{base}_repmean"] = float(np.mean(vals))
            row[f"{base}_repstd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{base}_repmin"] = float(np.min(vals))
            row[f"{base}_repmax"] = float(np.max(vals))

        rows.append(row)

    agg = pd.DataFrame(rows)
    agg = agg.sort_values(KEY_COLS).reset_index(drop=True)

    sha_field = "summary_sha256"
    src_paths = []
    for r in runs:
        p = trace_summary_path(r) if kind == "trace" else log_summary_path(r)
        src_paths.append({
            "rep": int(parse_rep_from_dirname(r.name)),
            "run_dir": str(r),
            "summary_path": str(p),
            sha_field: sha256_file(p),
        })

    provenance = {
        "schema_version": "unisage_v1_5rep_v1",
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_reps": int(len(runs)),
        "reps": [int(r) for r in reps],
        "base_metrics": base_metrics,
        "key_cols": KEY_COLS,
        "ddof": 1,
        "config": {
            "UNISAGE_EDGE_LAT": "raw",
            "UNISAGE_RCA_ATTR": "target",
            "UNISAGE_PROPAGATE": "1",
            "UNISAGE_PROPAGATE_GUARD": "1",
            "UNISAGE_TRACE_COMPOSITION": "dynamic_product_pattern_guard",
            "UNISAGE_LOG_COMPOSITION": "product",
            "budgets": [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20],
            "code_lock": "f899b19",
        },
        "deterministic": True,
        "runs": src_paths,
    }
    return agg, provenance


TRACE_HEADLINE_METRICS = [
    "A_exposed_recall",
    "A_exposed_prop",
    "pattern_coverage",
    "pattern_entropy",
    "pattern_jsd",
]

LOG_HEADLINE_METRICS = [
    "linked_A_exposed_recall",
    "log_manifested_recall",
    "log_template_unseen_recall",
    "log_template_rare_le_0p1_recall",
    "template_coverage",
    "template_jsd",
]


def render_markdown(agg: pd.DataFrame, provenance: dict) -> str:
    kind = provenance["kind"]
    headline = TRACE_HEADLINE_METRICS if kind == "trace" else LOG_HEADLINE_METRICS
    title_kind = "trace" if kind == "trace" else "log"
    lines = []
    lines.append(f"# UniSage-v1 budget-8 — 5-rep deterministic {title_kind} aggregate")
    lines.append("")
    lines.append(f"- Generated: {provenance['generated_at']}")
    lines.append(f"- Reps: {provenance['reps']}")
    lines.append(f"- Code lock: `{provenance['config']['code_lock']}`")
    lines.append(f"- Deterministic protocol — every `_repstd` cell below is expected to be 0.0")
    lines.append(f"- Source runs ({provenance['n_reps']}):")
    for r in provenance["runs"]:
        lines.append(f"  - rep {r['rep']}: `{Path(r['run_dir']).name}` (sha256 `{r['summary_sha256'][:12]}…`)")
    lines.append("")
    lines.append("## Headline metrics (mean / std / min / max across 5 reps)")
    lines.append("")
    for metric in headline:
        mean_col = f"{metric}_repmean"
        std_col = f"{metric}_repstd"
        if mean_col not in agg.columns:
            continue
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| dataset | ranking | budget | mean | std | min | max |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in agg.iterrows():
            mean_val = row[mean_col]
            std_val = row[std_col]
            min_val = row[f"{metric}_repmin"]
            max_val = row[f"{metric}_repmax"]
            mean_s = "nan" if pd.isna(mean_val) else f"{mean_val:.4f}"
            std_s = "nan" if pd.isna(std_val) else f"{std_val:.4f}"
            min_s = "nan" if pd.isna(min_val) else f"{min_val:.4f}"
            max_s = "nan" if pd.isna(max_val) else f"{max_val:.4f}"
            lines.append(
                f"| {row['dataset']} | {row['ranking']} | {row['budget']:.3f} "
                f"| {mean_s} | {std_s} | {min_s} | {max_s} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, agg: pd.DataFrame, provenance: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "trace_eval_5rep.csv" if provenance["kind"] == "trace" else "log_eval_5rep.csv"
    csv_path = out_dir / csv_name
    agg.to_csv(csv_path, index=False)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out_dir / "summary.md").write_text(render_markdown(agg, provenance))
    print(f"[ok] wrote {csv_path}  rows={len(agg)}  cols={len(agg.columns)}")
    print(f"[ok] wrote {out_dir / 'provenance.json'}")
    print(f"[ok] wrote {out_dir / 'summary.md'}")


def discover_trace_runs(root: Path, scope: str) -> List[Path]:
    pattern = re.compile(rf"^unisage_v1_rep(\d+)_budget8_{re.escape(scope)}$")
    matches: List[tuple[int, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            matches.append((int(m.group(1)), child))
    matches.sort(key=lambda kv: kv[0])
    return [p for _, p in matches]


def discover_log_runs(root: Path) -> List[Path]:
    pattern = re.compile(r"^unisage_v1_rep(\d+)_budget8$")
    matches: List[tuple[int, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            matches.append((int(m.group(1)), child))
    matches.sort(key=lambda kv: kv[0])
    return [p for _, p in matches]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--kind", choices=["trace", "log"], default=None)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("results/v02_sampling_runs"))
    parser.add_argument("--out-root", type=Path, default=Path("results/v02_sampling_runs"))
    args = parser.parse_args()

    if args.runs:
        if args.out is None or args.kind is None:
            raise SystemExit("--out and --kind are required when --runs is provided")
        agg, prov = aggregate(args.runs, args.kind)
        write_outputs(args.out, agg, prov)
        return

    if not args.auto:
        raise SystemExit("either pass --runs ... --out ... --kind ... or use --auto")

    for scope in ["hip_tt_primary", "aiops21_primary", "aiops21_excluded"]:
        runs = discover_trace_runs(args.root, scope)
        print(f"[discover trace/{scope}] {len(runs)} runs")
        for r in runs:
            print(f"  - {r.name}")
        if len(runs) < 2:
            print(f"  [skip] need >=2 runs, got {len(runs)}")
            continue
        out_dir = args.out_root / f"unisage_v1_budget8_{len(runs)}rep_aggregate_{scope}"
        agg, prov = aggregate(runs, "trace")
        write_outputs(out_dir, agg, prov)

    log_runs = discover_log_runs(args.root)
    print(f"[discover log] {len(log_runs)} runs")
    for r in log_runs:
        print(f"  - {r.name}")
    if len(log_runs) >= 2:
        out_dir = args.out_root / f"unisage_v1_budget8_{len(log_runs)}rep_aggregate_log_hip_tt_primary"
        agg, prov = aggregate(log_runs, "log")
        write_outputs(out_dir, agg, prov)


if __name__ == "__main__":
    main()
