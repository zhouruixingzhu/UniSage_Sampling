"""Run v0.2 random-uniform trace retention against frozen manifests.

Unlike `scripts/evaluate_trace_sampling.py`'s deterministic hash `uniform` reference, this
script produces a true seed-varying random-retention baseline: for each
(seed, case), assign every canonical TraceID a reproducible pseudo-random score
and evaluate top-k retention at the standard budget grid.

Outputs mirror the other v0.2 baseline run directories:

  results/v02_sampling_runs/uniform_random_v02_<dataset_key>_primary_seed-<N>/
    eval/trace_eval_per_case.csv
    eval/trace_eval_summary.csv
    eval/trace_eval_summary.md
    eval/trace_eval_provenance.json
    eval/trace_eval_skipped_cases.csv
    selected_cases.csv
    run_provenance.json
    run_summary.md

Example:
  python scripts/run_uniform_random.py --dataset-key hipster_train_ticket --seed 1
  python scripts/run_uniform_random.py --dataset-key aiops21 --seed 1
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

import sampling_labels as lp
from evaluate_trace_sampling import BUDGETS, build_metric_row, render_summary, summarize


PRIMARY_SCOPE = "primary_trace_label_scope"


def dataset_filter(dataset_key: str) -> set[str]:
    if dataset_key in {"hip_tt", "hipster_train_ticket"}:
        return {"hipster", "train_ticket"}
    if dataset_key == "aiops21":
        return {"aiops21"}
    raise ValueError(f"unknown dataset key: {dataset_key}")


def uniform_random_score(seed: int, dataset: str, batch: str, case: str, trace_id: object) -> float:
    material = f"{seed}\0{dataset}\0{batch}\0{case}\0{trace_id}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def evaluate_manifest(entry, budgets: Iterable[float], seed: int) -> list[dict]:
    base = pd.read_csv(entry.manifest_path)
    base["TraceID"] = base["TraceID"].astype(str)
    ranked = base.copy()
    ranked["_uniform_random_score"] = ranked["TraceID"].map(
        lambda trace_id: uniform_random_score(seed, entry.dataset, str(entry.batch), str(entry.case), trace_id)
    )
    ranked = ranked.sort_values(["_uniform_random_score", "TraceID"], ascending=[False, True]).reset_index(drop=True)

    rows = []
    n_ranked = len(ranked)
    for budget in budgets:
        k = min(n_ranked, max(1, int(round(len(base) * budget)))) if n_ranked else 0
        kept = ranked.head(k).drop(columns=["_uniform_random_score"])
        rows.append(build_metric_row(base, kept, "uniform_random", budget, n_ranked))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-key", choices=["hipster_train_ticket", "hip_tt", "aiops21"], required=True)
    parser.add_argument("--scope", default="primary")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest-index", type=Path, default=Path("results/label_protocol_v02/trace_manifest_index.csv"))
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--out-root", type=Path, default=Path("results/v02_sampling_runs"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.scope != "primary":
        raise SystemExit("random-uniform runner currently supports primary scope only")

    datasets = dataset_filter(args.dataset_key)
    dataset_key = "hipster_train_ticket" if args.dataset_key == "hip_tt" else args.dataset_key
    run_id = args.run_id or f"uniform_random_v02_{dataset_key}_{args.scope}_seed-{args.seed}"
    out_dir = args.out_root / run_id
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.manifest_index)
    index = index[
        index["dataset"].isin(datasets)
        & (index["trace_label_scope"] == PRIMARY_SCOPE)
    ].copy()
    index = index.sort_values(["dataset", "batch", "case"]).reset_index(drop=True)
    if index.empty:
        raise SystemExit(f"no primary cases found for dataset key {args.dataset_key}")

    rows: list[dict] = []
    for entry in index.itertuples(index=False):
        for row in evaluate_manifest(entry, args.budgets, args.seed):
            row.update(
                {
                    "dataset": entry.dataset,
                    "batch": entry.batch,
                    "case": entry.case,
                    "trace_label_scope": entry.trace_label_scope,
                }
            )
            rows.append(row)

    per_case = pd.DataFrame(rows)
    summary = summarize(per_case)
    selected = index[["dataset", "batch", "case", "trace_label_scope", "n_traces", "manifest_path", "manifest_sha256"]].copy()
    skipped = pd.DataFrame(columns=["dataset", "batch", "case", "reason"])

    per_case.to_csv(eval_dir / "trace_eval_per_case.csv", index=False)
    summary.to_csv(eval_dir / "trace_eval_summary.csv", index=False)
    skipped.to_csv(eval_dir / "trace_eval_skipped_cases.csv", index=False)
    selected.to_csv(out_dir / "selected_cases.csv", index=False)

    provenance = {
        "protocol_version": lp.PROTOCOL_VERSION,
        "mode": "version_stamped_uniform_random_trace",
        "dataset_key": dataset_key,
        "scope": args.scope,
        "seed": args.seed,
        "manifest_index": args.manifest_index.as_posix(),
        "rankings": ["uniform_random"],
        "budgets": args.budgets,
        "n_cases": int(len(index)),
        "skipped_cases": 0,
        "randomization": "sha256(seed,dataset,batch,case,TraceID) pseudo-random score; top-k without replacement",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (eval_dir / "trace_eval_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (eval_dir / "trace_eval_summary.md").write_text(render_summary(summary, provenance["mode"]))

    lines = [
        f"# {run_id}",
        "",
        f"- Dataset key: `{dataset_key}`",
        f"- Scope: `{args.scope}`",
        f"- Seed: `{args.seed}`",
        f"- Cases: {len(index)}",
        f"- Budgets: {args.budgets}",
        f"- Output: `{out_dir}`",
        "",
        "Randomization: reproducible SHA-256 pseudo-random score per TraceID; top-k without replacement.",
        "",
    ]
    (out_dir / "run_summary.md").write_text("\n".join(lines))

    print(render_summary(summary, provenance["mode"]))
    print(f"Wrote random-uniform run to {out_dir}; cases={len(index)}")


if __name__ == "__main__":
    main()
