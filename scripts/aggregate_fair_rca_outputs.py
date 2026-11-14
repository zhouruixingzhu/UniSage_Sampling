"""Aggregate per-run fair-RCA outputs across available sampler seeds.

Each evaluator harness is run once per seed/scope because the existing CLIs
accept one run directory per sampler. This script merges those per-run outputs
into the root output directory and adds a 100%-budget collapse check.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


EVALUATORS = {
    "unisage_rcl": {
        "per_case": "fair_rca_per_case.csv",
        "rank": "rank",
        "prefix": "fair_rca",
    },
    "nezha": {
        "per_case": "nezha_rca_per_case.csv",
        "rank": "rank_ac",
        "prefix": "nezha_rca",
    },
    "microrca": {
        "per_case": "microrca_rca_per_case.csv",
        "rank": "rank",
        "prefix": "microrca_rca",
    },
    "tracerca": {
        "per_case": "tracerca_rca_per_case.csv",
        "rank": "rank",
        "prefix": "tracerca_rca",
    },
}


def infer_scope_seed(run_dir: Path) -> tuple[str, int | None]:
    name = run_dir.name
    if "hip_tt" in name or "hipster_train_ticket" in name:
        scope = "hip_tt_primary"
    elif "aiops21" in name:
        scope = "aiops21_primary"
    else:
        scope = "unknown"

    m = re.search(r"(?:seed|rep)-?(\d+)", name)
    seed = int(m.group(1)) if m else None
    return scope, seed


def run_dirs(root: Path) -> list[Path]:
    candidates = []
    for parent in (root / "per_run", root):
        if parent.is_dir():
            candidates.extend(
                p for p in parent.iterdir()
                if p.is_dir() and not p.name.startswith("_") and "smoke" not in p.name
            )
    return sorted(set(candidates))


def reciprocal_rank(rank: pd.Series) -> pd.Series:
    r = pd.to_numeric(rank, errors="coerce").fillna(0)
    return np.where(r > 0, 1.0 / r, 0.0)


def load_per_case(root: Path, evaluator: str) -> tuple[pd.DataFrame, list[str]]:
    spec = EVALUATORS[evaluator]
    frames = []
    missing = []
    for d in run_dirs(root):
        path = d / spec["per_case"]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        scope, seed = infer_scope_seed(d)
        df.insert(0, "run_id", d.name)
        df.insert(1, "scope", scope)
        df.insert(2, "seed", seed)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), missing
    return pd.concat(frames, ignore_index=True), missing


def aggregate(per_case: pd.DataFrame, evaluator: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = EVALUATORS[evaluator]
    rank_col = spec["rank"]
    df = per_case.copy()
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce").fillna(0)
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["_rr"] = reciprocal_rank(df[rank_col])
    group_cols = ["scope", "dataset", "sampler", "retention", "budget"]
    if "mode" in df.columns:
        group_cols.append("mode")
    seed_group_cols = ["run_id", "seed"] + group_cols

    seed_summary = (
        df.groupby(seed_group_cols, dropna=False)
        .agg(
            n_cases=("case", "count"),
            n_kept_mean=("n_kept", "mean"),
            ac1=(rank_col, lambda s: (s == 1).mean()),
            ac3=(rank_col, lambda s: s.between(1, 3).mean()),
            mrr=("_rr", "mean"),
        )
        .reset_index()
    )

    mean_std = (
        seed_summary.groupby(group_cols, dropna=False)
        .agg(
            n_seed=("seed", lambda s: s.dropna().nunique()),
            n_runs=("run_id", "nunique"),
            n_cases_mean=("n_cases", "mean"),
            n_kept_mean=("n_kept_mean", "mean"),
            n_kept_std=("n_kept_mean", "std"),
            ac1_mean=("ac1", "mean"),
            ac1_std=("ac1", "std"),
            ac3_mean=("ac3", "mean"),
            ac3_std=("ac3", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_std=("mrr", "std"),
        )
        .reset_index()
    )
    return seed_summary, mean_std


def collapse_check(per_case: pd.DataFrame, evaluator: str) -> pd.DataFrame:
    spec = EVALUATORS[evaluator]
    rank_col = spec["rank"]
    df = per_case.copy()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df = df[df["budget"] >= 1.0].copy()
    if df.empty:
        return pd.DataFrame()

    group_cols = ["run_id", "scope", "seed", "dataset", "batch", "case", "retention"]
    if "mode" in df.columns:
        group_cols.append("mode")

    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        ranks = sorted(set(pd.to_numeric(group[rank_col], errors="coerce").fillna(0).astype(int)))
        nkept = sorted(set(pd.to_numeric(group["n_kept"], errors="coerce").fillna(-1).astype(int)))
        top_col = "top1" if "top1" in group.columns else None
        tops = sorted(set(group[top_col].dropna().astype(str))) if top_col else []
        rows.append(
            {
                **key_map,
                "n_samplers": group["sampler"].nunique(),
                "n_kept_values": ";".join(map(str, nkept)),
                "n_n_kept_values": len(nkept),
                "rank_values": ";".join(map(str, ranks)),
                "n_rank_values": len(ranks),
                "top1_values": ";".join(tops),
                "n_top1_values": len(tops),
                "ok": len(nkept) <= 1 and len(ranks) <= 1 and (not top_col or len(tops) <= 1),
            }
        )
    return pd.DataFrame(rows)


def status_markdown(
    evaluator: str,
    per_case: pd.DataFrame,
    seed_summary: pd.DataFrame,
    mean_std: pd.DataFrame,
    collapse: pd.DataFrame,
) -> str:
    lines = [f"# Fair RCA status: {evaluator}\n"]
    if per_case.empty:
        lines.append("No per-case rows were found.\n")
        return "\n".join(lines)

    lines.append(
        f"Per-case rows: {len(per_case)} | datasets: "
        f"{', '.join(sorted(map(str, per_case['dataset'].unique())))} | "
        f"samplers: {', '.join(sorted(map(str, per_case['sampler'].unique())))}\n"
    )

    lines.append("\n## Available Seeds\n")
    coverage = (
        per_case.groupby(["scope", "dataset", "sampler"], dropna=False)["seed"]
        .apply(lambda s: ",".join(map(str, sorted(set(int(x) for x in s.dropna())))))
        .reset_index(name="available_seeds")
    )
    expected = {1, 2, 3, 4, 5}
    coverage["missing_1_to_5"] = coverage["available_seeds"].apply(
        lambda x: ",".join(map(str, sorted(expected - set(int(v) for v in x.split(",") if v))))
    )
    lines.append("```\n" + coverage.to_string(index=False) + "\n```\n")

    lines.append("\n## 100 Percent Collapse Check\n")
    if collapse.empty:
        lines.append("No budget=1.0 rows were found.\n")
    else:
        n_bad = int((~collapse["ok"]).sum())
        lines.append(f"Rows checked: {len(collapse)} | failures: {n_bad}\n")
        by_run = (
            collapse.groupby(["scope", "run_id"], dropna=False)["ok"]
            .agg(rows="count", failures=lambda s: int((~s).sum()))
            .reset_index()
        )
        lines.append("```\n" + by_run.to_string(index=False) + "\n```\n")
        if n_bad:
            bad = collapse[~collapse["ok"]].head(20)
            lines.append("\nFirst failures:\n")
            lines.append("```\n" + bad.to_string(index=False) + "\n```\n")

    lines.append("\n## Output Tables\n")
    prefix = EVALUATORS[evaluator]["prefix"]
    lines.extend(
        [
            f"- `{prefix}_per_case_all.csv`",
            f"- `{prefix}_seed_summary.csv`",
            f"- `{prefix}_mean_std_summary.csv`",
            f"- `{prefix}_collapse_check.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--evaluator", choices=sorted(EVALUATORS), required=True)
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    per_case, _missing = load_per_case(root, args.evaluator)
    prefix = EVALUATORS[args.evaluator]["prefix"]
    if per_case.empty:
        (root / "STATUS.md").write_text(status_markdown(args.evaluator, per_case, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
        print(f"No rows found under {root}")
        return

    seed_summary, mean_std = aggregate(per_case, args.evaluator)
    collapse = collapse_check(per_case, args.evaluator)

    per_case.to_csv(root / f"{prefix}_per_case_all.csv", index=False)
    seed_summary.to_csv(root / f"{prefix}_seed_summary.csv", index=False)
    mean_std.to_csv(root / f"{prefix}_mean_std_summary.csv", index=False)
    collapse.to_csv(root / f"{prefix}_collapse_check.csv", index=False)
    (root / "STATUS.md").write_text(
        status_markdown(args.evaluator, per_case, seed_summary, mean_std, collapse)
    )
    print(f"Wrote aggregate outputs under {root}")
    print(mean_std.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
