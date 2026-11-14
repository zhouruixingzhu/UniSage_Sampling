"""Sequentially rerun per-case log sampling for one route.

This helper exists because UniSage writes `log_sampling_result.csv` into the
shared case directory as a side effect. Running multiple route agents in
parallel can therefore contaminate copied log artifacts. This script runs one
route at a time and copies the log output immediately after each case.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from aiomultiprocess import Pool

import UniSage


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


async def main_async(args: argparse.Namespace) -> None:
    route_root = args.output_root / args.route
    selected_path = args.selected_cases or args.output_root / f"{args.route}_hip_tt_primary" / "selected_cases.csv"
    if not selected_path.exists():
        raise SystemExit(f"Missing selected cases: {selected_path}")

    samples_dir = route_root / "log_samples_clean"
    provenance_dir = route_root / "log_provenance_clean"
    logs_dir = route_root / "logs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(selected_path)
    if args.limit:
        selected = selected.head(args.limit)

    index_path = route_root / "log_sample_index_clean.csv"
    rows = []
    done_keys = set()
    if args.resume and index_path.exists():
        prior = pd.read_csv(index_path)
        if not prior.empty:
            ok_prior = prior[prior["status"] == "ok"].copy()
            for entry in ok_prior.itertuples(index=False):
                key = (str(entry.dataset), str(entry.batch), str(entry.case))
                done_keys.add(key)
                rows.append({k: getattr(entry, k) for k in ok_prior.columns})
            print(f"[resume] skipping {len(done_keys)} cases already marked ok", flush=True)
    started = time.time()
    async with Pool() as pool:
        for i, entry in enumerate(selected.itertuples(index=False), start=1):
            dataset = str(entry.dataset)
            batch = str(entry.batch)
            case = str(entry.case)
            if (dataset, batch, case) in done_keys:
                continue
            cdir = case_dir(args.data_root, dataset, batch, case)
            stem = artifact_stem(dataset, batch, case)
            print(f"[{args.route} log rerun {i}/{len(selected)}] {dataset}/{batch}/{case}", flush=True)
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
                case_started = time.time()
                result = await UniSage.evaluate(cdir, pool, "hipster")
                if not result or result[4] is None or result[4].empty:
                    raise RuntimeError("UniSage.evaluate returned no log sampling frame")
                sample_src = cdir / "log_sampling_result.csv"
                prov_src = cdir / "log_sampling_provenance.json"
                diag_src = cdir / "cross_modal_propagation_diagnostics.json"
                diag_dst = Path(row["diagnostics_path"])
                if not copy_if_exists(sample_src, Path(row["sample_path"])):
                    raise RuntimeError(f"missing {sample_src}")
                if not copy_if_exists(prov_src, Path(row["provenance_path"])):
                    row["provenance_path"] = ""
                diag_is_current = diag_src.exists() and diag_src.stat().st_mtime >= case_started - 1.0
                if diag_is_current and not copy_if_exists(diag_src, diag_dst):
                    row["diagnostics_path"] = ""
                elif not diag_is_current:
                    if diag_dst.exists():
                        diag_dst.unlink()
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
        "log_samples_clean": samples_dir.as_posix(),
        "log_provenance_clean": provenance_dir.as_posix(),
        "log_sample_index_clean": (route_root / "log_sample_index_clean.csv").as_posix(),
    }
    (route_root / "log_rerun_clean_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/v02_sampling_runs"))
    parser.add_argument("--selected-cases", type=Path, default=None,
                        help="trace run's selected_cases.csv; defaults to <output-root>/<route>_hip_tt_primary/selected_cases.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already marked ok in log_sample_index_clean.csv",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
