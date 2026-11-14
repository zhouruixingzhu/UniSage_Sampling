"""Rerun route-specific log sampling from existing rankings and trace samples.

This is the fast counterpart to `scripts/rerun_log_sampling_route.py`. It avoids
rerunning AD/RCL when the goal is only to repair route-contaminated log outputs:
per-case `final_ranking.csv` is reused, logs are rescored, and optional
cross-modal propagation is applied from a known pre-propagation trace run.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

import UniSage
from sampling import (
    apply_trace_composition_strategy,
    cross_modal_propagation,
    sort_log_sampling_frame,
)


def case_dir(data_root: Path, dataset: str, batch: str, case: str) -> Path:
    if dataset == "aiops21":
        return data_root / "aiops21" / "all" / case
    return data_root / dataset / "batches" / batch / case


def artifact_stem(dataset: str, batch: str, case: str) -> str:
    return f"{dataset}__{batch}__{case}"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_system_anomalous(route_root: Path, stem: str, ranking: pd.DataFrame) -> bool:
    provenance_path = route_root / "log_provenance_clean" / f"{stem}.log_sampling_provenance.json"
    if provenance_path.exists():
        try:
            return bool(json.loads(provenance_path.read_text()).get("system_anomalous", False))
        except Exception:
            pass
    return not ranking.empty


def read_trace_sample(trace_index: pd.DataFrame, dataset: str, batch: str, case: str) -> pd.DataFrame:
    matched = trace_index[
        (trace_index["dataset"].astype(str) == dataset)
        & (trace_index["batch"].astype(str) == batch)
        & (trace_index["case"].astype(str) == case)
        & (trace_index["status"].astype(str) == "ok")
    ]
    if matched.empty:
        raise RuntimeError(f"missing trace sample for {dataset}/{batch}/{case}")
    sample_path = Path(str(matched.iloc[0]["sample_path"]))
    if not sample_path.exists():
        raise RuntimeError(f"missing trace sample file: {sample_path}")
    return pd.read_csv(sample_path)


def trace_sampling_fell_back(trace_df: pd.DataFrame) -> bool:
    if "SamplingFallback" not in trace_df.columns:
        return False
    fallback = trace_df["SamplingFallback"].dropna().astype(str)
    return fallback.str.len().gt(0).any()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/v02_sampling_runs"))
    parser.add_argument("--system-source-route", required=True,
                        help="route containing matching archived log provenance under output-root")
    parser.add_argument("--trace-source-route", required=True,
                        help="matching trace run containing sample_index.csv under output-root")
    parser.add_argument("--selected-cases", type=Path, default=None,
                        help="trace run's selected_cases.csv; defaults to <output-root>/<route>_hip_tt_primary/selected_cases.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    route_root = args.output_root / args.route
    selected_path = args.selected_cases or args.output_root / f"{args.route}_hip_tt_primary" / "selected_cases.csv"
    trace_index_path = args.output_root / args.trace_source_route / "sample_index.csv"
    system_source_root = args.output_root / args.system_source_route
    if not selected_path.exists():
        raise SystemExit(f"Missing selected cases: {selected_path}")
    if not trace_index_path.exists():
        raise SystemExit(f"Missing trace index: {trace_index_path}")

    samples_dir = route_root / "log_samples_clean"
    provenance_dir = route_root / "log_provenance_clean"
    logs_dir = route_root / "logs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(selected_path)
    if args.limit:
        selected = selected.head(args.limit)
    trace_index = pd.read_csv(trace_index_path)
    propagate = os.environ.get("UNISAGE_PROPAGATE", "1") != "0"

    rows = []
    started = time.time()
    for i, entry in enumerate(selected.itertuples(index=False), start=1):
        dataset = str(entry.dataset)
        batch = str(entry.batch)
        case = str(entry.case)
        cdir = case_dir(args.data_root, dataset, batch, case)
        stem = artifact_stem(dataset, batch, case)
        print(f"[{args.route} fast log rerun {i}/{len(selected)}] {dataset}/{batch}/{case}", flush=True)
        row = {
            "dataset": dataset,
            "batch": batch,
            "case": case,
            "case_dir": cdir.as_posix(),
            "sample_path": (samples_dir / f"{stem}.log_sampling.csv").as_posix(),
            "provenance_path": (provenance_dir / f"{stem}.log_sampling_provenance.json").as_posix(),
            "diagnostics_path": (provenance_dir / f"{stem}.cross_modal_propagation_diagnostics.json").as_posix(),
            "status": "ok",
            "error": "",
            "elapsed_sec": 0.0,
        }
        t0 = time.time()
        try:
            ranking_path = cdir / "final_ranking.csv"
            ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame(columns=["service_name", "CombinedScore"])
            system_anomalous = read_system_anomalous(system_source_root, stem, ranking)
            log_df = UniSage.log_sampling(system_anomalous, cdir, ranking, "hipster")
            if log_df is None or log_df.empty:
                raise RuntimeError("log_sampling returned no rows")

            diag_dst = Path(row["diagnostics_path"])
            trace_df = read_trace_sample(trace_index, dataset, batch, case) if propagate else pd.DataFrame()
            should_propagate = propagate and system_anomalous and not trace_sampling_fell_back(trace_df)
            if should_propagate:
                trace_df, log_df, diag = cross_modal_propagation(
                    trace_df,
                    log_df,
                    w_t=UniSage.PROP_W_T,
                    w_l=UniSage.PROP_W_L,
                    return_diagnostics=True,
                )
                trace_df = apply_trace_composition_strategy(trace_df)
                log_df = sort_log_sampling_frame(log_df)
                log_df.to_csv(cdir / "log_sampling_result.csv", index=False)
                diag_dst.write_text(json.dumps(diag, indent=2) + "\n")
            elif propagate and diag_dst.exists():
                diag_dst.unlink()
                row["diagnostics_path"] = ""
            elif diag_dst.exists():
                diag_dst.unlink()
                row["diagnostics_path"] = ""
            else:
                row["diagnostics_path"] = ""

            if not copy_if_exists(cdir / "log_sampling_result.csv", Path(row["sample_path"])):
                raise RuntimeError(f"missing {cdir / 'log_sampling_result.csv'}")
            if not copy_if_exists(cdir / "log_sampling_provenance.json", Path(row["provenance_path"])):
                row["provenance_path"] = ""
            if propagate and not diag_dst.exists():
                row["diagnostics_path"] = ""
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [error] {row['error']}", flush=True)
        row["elapsed_sec"] = round(time.time() - t0, 3)
        rows.append(row)
        pd.DataFrame(rows).to_csv(route_root / "log_sample_index_clean.csv", index=False)

    sample_index = pd.DataFrame(rows)
    sample_index.to_csv(route_root / "log_sample_index_clean.csv", index=False)
    summary = {
        "route": args.route,
        "selected_cases": int(len(selected)),
        "status_counts": sample_index["status"].value_counts(dropna=False).to_dict(),
        "elapsed_sec": round(time.time() - started, 3),
        "propagation_enabled_env": os.environ.get("UNISAGE_PROPAGATE", "1"),
        "propagation_guard_env": os.environ.get("UNISAGE_PROPAGATE_GUARD", "1"),
        "system_source_route": args.system_source_route,
        "trace_source_route": args.trace_source_route,
        "log_samples_clean": samples_dir.as_posix(),
        "log_provenance_clean": provenance_dir.as_posix(),
        "log_sample_index_clean": (route_root / "log_sample_index_clean.csv").as_posix(),
    }
    (route_root / "log_rerun_clean_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
