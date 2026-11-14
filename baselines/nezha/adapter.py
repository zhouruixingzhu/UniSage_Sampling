"""UniSage <-> Nezha data bridging.

Two responsibilities:

1. **Case index** -- map every UniSage Hipster/Train-Ticket case_key onto its
   matching Nezha fault (same dataset, lineage-verified by `inject_timestamp +
   inject_pod + inject_type`) and the corresponding Nezha date / abnormal-window
   filename (`<HH>_<MM>` = inject_time + 2 min). Produces
   `results/nezha/nezha_case_index.csv`. AIOps21 is out of scope for this index
   -- it doesn't share Nezha's data lineage and is handled separately (Phase 4).

2. **Scratch rca_data materializer** -- given (case, retained_TraceID_set,
   fidelity_mode, optional log subset path), build a temp directory laid out
   exactly like Nezha's `rca_data` but with the **abnormal-window traceid file
   replaced** by a filtered copy. Nezha's `data_integrate` iterates over the
   rows of the `<HH>_<MM>_traceid.csv` -- restricting that file IS the sampling
   hook for traces.

   - **Mode A (trace-isolated):** only `traceid/<HH>_<MM>_traceid.csv` is
     rewritten. `trace/`, `log/`, `metric/` symlink directly to the full Nezha
     dirs. The apples-to-apples row vs trace-only RCA evaluators -- sampler
     differences come purely from which TraceIDs are visible.

   - **Mode B (trace-derived multimodal, symmetric):** in addition,
     `trace/<HH>_<MM>_trace.csv` is filtered to retained TraceIDs and
     `log/<HH>_<MM>_log.csv` is filtered to spans of retained traces (Nezha
     joins logs to traces via SpanID; restricting by retained-trace SpanIDs is
     the same rule for every sampler). Metrics stay full -- UniSage has no
     metric sampler; per-window metric files already scope the time range.
     Symmetric and fair: tests "does the trace subset preserve downstream
     multimodal RCA, given a uniform rule for carrying logs forward?"

   - **Mode C (end-to-end multimodal, asymmetric by sampler):** for samplers
     that produce a separate log subset (UniSage's
     `data/<ds>/batches/<batch>/<case>/log_sampling_result.csv`), use THAT
     subset (top-N by `LogSamplingProb` at the same budget as traces) as the
     log file, after recovering SpanID from the log body text and reformatting
     to Nezha's `TimeUnixNano, SpanID, Log` schema. Baseline samplers (which
     have no separate log output) fall back to Mode-B trace-derived logs. This
     is each sampler's actual end-to-end multimodal output -- not a controlled
     comparison, but the deployment-realistic view. Report alongside B with the
     mechanism made explicit in the paper.

The Nezha algorithm itself is invoked verbatim via the subprocess wrapper
`baselines/nezha/nezha_single_case.py`, which runs in the isolated `nezha`
conda env (py3.11 + pandas 1.5.3 + drain3 0.9.10; see
`baselines/nezha/PROVENANCE.md`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEZHA_DIR = PROJECT_ROOT / "baselines" / "nezha"
NEZHA_RUNNER = NEZHA_DIR / "nezha_single_case.py"


def resolve_nezha_python(python_exe: str | None = None) -> str:
    """Locate the dedicated nezha environment; never use a driver-env fallback."""
    configured = python_exe or os.environ.get("NEZHA_PYTHON")
    if configured:
        expanded = os.path.expanduser(configured)
        found = shutil.which(expanded)
        if found:
            # Keep the environment's path (resolving symlinks can select its base env).
            return str(Path(found).absolute())
        raise ValueError(
            f"nezha interpreter is not an executable file: {configured}. "
            "Use --nezha-python /path/to/nezha/bin/python or set NEZHA_PYTHON."
        )

    # Preserve the original installation when present; also discover other conda roots.
    roots = [Path("/opt/anaconda3")]
    for variable in ("CONDA_EXE", "CONDA_PREFIX"):
        value = os.environ.get(variable)
        if value:
            path = Path(value).expanduser()
            roots.append(path.parent.parent if variable == "CONDA_EXE" else path)
    roots.extend([Path(sys.prefix), Path(sys.prefix).parent.parent])
    conda_exe = shutil.which("conda")
    if conda_exe:
        roots.append(Path(conda_exe).parent.parent)
    roots.extend(Path.home() / folder for folder in (
        "anaconda3", "miniconda3", "miniforge3", "mambaforge", ".conda"
    ))
    roots.extend(Path("/opt") / folder for folder in ("miniconda3", "miniforge3"))
    for root in dict.fromkeys(roots):
        candidates = [root / "envs" / "nezha" / "bin" / "python"]
        if root.name == "nezha" and root.parent.name == "envs":
            candidates.insert(0, root / "bin" / "python")
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.absolute())
    raise ValueError(
        "Could not find the isolated conda environment 'nezha'. Create it using "
        "baselines/nezha/PROVENANCE.md, then use --nezha-python "
        "/path/to/nezha/bin/python or set NEZHA_PYTHON. "
        "The current driver interpreter is not used as a fallback."
    )


# Anchors for normal_time on each Nezha date -- copied verbatim from upstream
# `main.py` so our standalone "via-adapter" run reproduces upstream AS@k.
NORMAL_TIMES = {
    "2022-08-22": "2022-08-22 03:51",
    "2022-08-23": "2022-08-23 17:00",
    "2023-01-29": "2023-01-29 08:50",
    "2023-01-30": "2023-01-30 11:39",
}

# UniSage dataset name -> Nezha namespace (`--ns` arg)
NS_OF_DATASET = {"hipster": "hipster", "train_ticket": "ts"}


# --------------------------------------------------------------------------- #
# Case index
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CaseEntry:
    dataset: str           # "hipster" | "train_ticket"
    batch: str             # "batch1" | "batch2"
    case: str              # UniSage case_key, e.g. "03_53"
    ns: str                # Nezha namespace: "hipster" | "ts"
    nz_date: str           # e.g. "2022-08-22"
    nz_window: str         # abnormal window stem, e.g. "03_55" (inject + 2 min)
    inject_time: str       # raw inject_time string from Nezha fault entry
    inject_pod: str        # ground-truth root-cause pod
    inject_type: str       # e.g. "cpu_contention"
    inject_timestamp: str  # epoch seconds, the lineage key

    @property
    def normal_time(self) -> str:
        return NORMAL_TIMES[self.nz_date]


def _compute_window(inject_time: str) -> str:
    date, hm = inject_time.split(" ")
    hh, mm = hm.split(":")[0], hm.split(":")[1]
    mm_i = int(mm) + 2
    hh_i = int(hh)
    if mm_i >= 60:
        hh_i += 1
        mm_i -= 60
    return f"{hh_i:02d}_{mm_i:02d}"


def _scan_nezha_index(nezha_rca_dir: Path) -> dict[tuple, dict]:
    """Build (inject_timestamp, inject_pod, inject_type) -> Nezha metadata."""
    index: dict[tuple, dict] = {}
    for date_dir in sorted(nezha_rca_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        fl_path = date_dir / f"{date_dir.name}-fault_list.json"
        if not fl_path.exists():
            continue
        fl = json.loads(fl_path.read_text())
        for _hour, faults in fl.items():
            for f in faults:
                k = (f["inject_timestamp"], f["inject_pod"], f["inject_type"])
                index[k] = dict(nz_date=date_dir.name, fault=f)
    return index


def build_case_index(unisage_data_root: Path, nezha_rca_dir: Path,
                     datasets: Iterable[str] = ("hipster", "train_ticket")) -> pd.DataFrame:
    """Discover every UniSage case (hipster, train_ticket) that has a Nezha twin."""
    nz_idx = _scan_nezha_index(nezha_rca_dir)
    rows: list[dict] = []
    for ds in datasets:
        for batch_dir in sorted((unisage_data_root / ds / "batches").iterdir()):
            label_path = batch_dir / "label.json"
            if not label_path.exists():
                continue
            labels = json.loads(label_path.read_text())
            for case_key, meta in labels.items():
                key = (meta.get("inject_timestamp"), meta.get("inject_pod"),
                       meta.get("inject_type"))
                if key not in nz_idx:
                    continue
                nz = nz_idx[key]
                rows.append(dict(
                    dataset=ds, batch=batch_dir.name, case=case_key,
                    ns=NS_OF_DATASET[ds],
                    nz_date=nz["nz_date"],
                    nz_window=_compute_window(nz["fault"]["inject_time"]),
                    inject_time=nz["fault"]["inject_time"],
                    inject_pod=nz["fault"]["inject_pod"],
                    inject_type=nz["fault"]["inject_type"],
                    inject_timestamp=str(nz["fault"]["inject_timestamp"]),
                ))
    return pd.DataFrame(rows)


def write_case_index(unisage_data_root: Path, nezha_rca_dir: Path,
                     out_path: Path) -> pd.DataFrame:
    df = build_case_index(unisage_data_root, nezha_rca_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Scratch rca_data materializer
# --------------------------------------------------------------------------- #

def _symlink_or_copy_tree(src: Path, dst: Path) -> None:
    """Symlink a directory tree if possible, fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst, dirs_exist_ok=True)


# Regex for recovering SpanID from UniSage's parsed log Body text. Both Hipster
# and Train-Ticket logs prefix every message with "TraceID: <32hex> SpanID: <16hex>".
SPANID_FROM_BODY = r"SpanID:\s*([0-9a-f]{16})"


def unisage_log_subset_to_nezha(
    unisage_log_sampling_csv: Path,
    abnormal_minute_window: tuple[pd.Timestamp, pd.Timestamp],
    log_budget: float,
) -> pd.DataFrame:
    """Convert a UniSage `log_sampling_result.csv` to Nezha's expected log
    schema, restricted to the abnormal-window minute and top-budget by
    `LogSamplingProb`.

    Returns a DataFrame with columns `TimeUnixNano, SpanID, Log`. Rows whose
    `Body` lacks a recoverable SpanID are dropped (Nezha indexes logs by
    SpanID, so unattributed lines would never be queried).
    """
    df = pd.read_csv(unisage_log_sampling_csv)
    if df.empty:
        return pd.DataFrame(columns=["TimeUnixNano", "SpanID", "Log"])

    # Scope to the abnormal-window minute (matches Nezha's per-minute log file).
    ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    lo, hi = abnormal_minute_window
    df = df.loc[(ts >= lo) & (ts < hi)].copy()
    df["_ts"] = ts.loc[df.index]
    if df.empty:
        return pd.DataFrame(columns=["TimeUnixNano", "SpanID", "Log"])

    # Top-N by LogSamplingProb at the same budget as traces. UniSage's
    # combined log score is the final selection signal.
    if "LogSamplingProb" in df.columns:
        n_target = max(1, int(round(len(df) * log_budget)))
        df = df.nlargest(n_target, "LogSamplingProb")

    span = df["Body"].astype(str).str.extract(SPANID_FROM_BODY, expand=False)
    df = df.assign(SpanID=span).dropna(subset=["SpanID"])
    if df.empty:
        return pd.DataFrame(columns=["TimeUnixNano", "SpanID", "Log"])

    # Nezha reads Log as a JSON-ish blob and re-parses with its own drain3; the
    # raw Body works the same way and is what UniSage's parsed file kept.
    return pd.DataFrame({
        "TimeUnixNano": df["_ts"].astype("int64"),
        "SpanID": df["SpanID"].astype(str),
        "Log": df["Body"].astype(str),
    })


def abnormal_minute_window(entry: "CaseEntry") -> tuple[pd.Timestamp, pd.Timestamp]:
    """The single minute Nezha reads for the abnormal window of this fault."""
    date_str = entry.nz_date
    hh, mm = entry.nz_window.split("_")
    start = pd.Timestamp(f"{date_str} {hh}:{mm}:00", tz="UTC")
    return start, start + pd.Timedelta(minutes=1)


def materialize_case(entry: CaseEntry, nezha_rca_dir: Path,
                     retained_ids: set[str] | None,
                     mode: str, scratch_root: Path,
                     unisage_log_sampling_csv: Path | None = None,
                     log_budget: float | None = None,
                     log_policy: str = "native") -> Path:
    """Build a scratch rca_data dir for one (case, retention) cell.

    `retained_ids=None` means full-data (no sampling) -- use this for the
    standalone reproduction-through-adapter correctness check.

    For Mode C with `log_policy="native"`, pass `unisage_log_sampling_csv`
    (UniSage's per-case `log_sampling_result.csv`) and `log_budget` (same as
    the trace budget for the cell). `log_policy="rca_support"` intentionally
    keeps raw logs linked to retained trace spans, preserving repeated support
    statistics for Nezha-style event graphs. When `unisage_log_sampling_csv` is
    None in native Mode C, the function falls back to Mode-B logic so baseline
    samplers coexist in the same harness call.

    Returns the path to the scratch rca_data root, which the caller passes to
    `nezha_single_case.py --rca-path`.
    """
    if mode not in {"A", "B", "C"}:
        raise ValueError(f"unknown fidelity mode: {mode!r} (expected 'A' / 'B' / 'C')")
    if log_policy not in {"native", "rca_support"}:
        raise ValueError(f"unknown log_policy: {log_policy!r} (expected 'native' / 'rca_support')")
    scratch_root = scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    # Resolve nezha_rca_dir to an ABSOLUTE path: os.symlink stores the src
    # verbatim and relative paths there resolve from the symlink's location,
    # not from cwd. Without resolving, every symlink in scratch breaks the
    # moment Nezha tries to follow it from inside scratch_root.
    nezha_rca_dir = nezha_rca_dir.resolve()
    date = entry.nz_date
    window = entry.nz_window
    src_date_dir = nezha_rca_dir / date
    dst_date_dir = scratch_root / date
    dst_date_dir.mkdir(parents=True, exist_ok=True)

    # Symlink fault_list.json (unused by our runner, but Nezha may glob it).
    src_fl = src_date_dir / f"{date}-fault_list.json"
    dst_fl = dst_date_dir / f"{date}-fault_list.json"
    if src_fl.exists() and not dst_fl.exists():
        try:
            os.symlink(src_fl, dst_fl)
        except OSError:
            shutil.copy2(src_fl, dst_fl)

    # Subdirs: trace, traceid, log, metric.
    # Default: symlink the whole src subdir; then for the abnormal window we
    # break the symlink and write the filtered file.
    for sub in ("trace", "traceid", "log", "metric"):
        dst_sub = dst_date_dir / sub
        src_sub = src_date_dir / sub
        if not src_sub.exists():
            continue
        if dst_sub.exists() or dst_sub.is_symlink():
            # already materialized (re-use scratch root); skip.
            continue
        try:
            os.symlink(src_sub, dst_sub, target_is_directory=True)
        except OSError:
            shutil.copytree(src_sub, dst_sub)

    if retained_ids is None:
        return scratch_root  # full-data passthrough

    # Replace the abnormal-window files for the retained-ID subset. Because
    # the per-subdir entry above is a symlink to the source dir, we cannot
    # write inside it without affecting the source. Convert that one subdir
    # to a writable directory of symlinked siblings + the rewritten file.

    def _materialize_subdir(sub: str, target_filename: str) -> Path:
        """Replace `dst_date_dir/<sub>` symlink with a writable dir whose
        siblings are symlinks back to the source, then return the writable
        path where the target file should be created."""
        dst_sub = dst_date_dir / sub
        src_sub = src_date_dir / sub
        if dst_sub.is_symlink():
            dst_sub.unlink()
            dst_sub.mkdir()
            for child in src_sub.iterdir():
                if child.name == target_filename:
                    continue  # we will write our own
                try:
                    os.symlink(child, dst_sub / child.name)
                except OSError:
                    if child.is_dir():
                        shutil.copytree(child, dst_sub / child.name)
                    else:
                        shutil.copy2(child, dst_sub / child.name)
        return dst_sub / target_filename

    # --- traceid (always filtered) -----------------------------------------
    tid_target = _materialize_subdir("traceid", f"{window}_traceid.csv")
    src_tid = src_date_dir / "traceid" / f"{window}_traceid.csv"
    if src_tid.exists():
        # Single-column, no header (Nezha's format).
        raw_ids = pd.read_csv(src_tid, header=None, dtype=str)[0].astype(str)
        keep_mask = raw_ids.isin(retained_ids)
        raw_ids.loc[keep_mask].to_csv(tid_target, header=False, index=False)
    else:
        # Defensive: write the retained set even if Nezha didn't ship that minute.
        pd.Series(sorted(retained_ids), dtype=str).to_csv(
            tid_target, header=False, index=False)

    if mode == "A":
        return scratch_root

    # --- Mode B / C: also filter trace + log ------------------------------
    # Trace: same rule in both modes -- rows whose TraceID is in retained.
    trc_target = _materialize_subdir("trace", f"{window}_trace.csv")
    src_trc = src_date_dir / "trace" / f"{window}_trace.csv"
    if src_trc.exists():
        df = pd.read_csv(src_trc)
        df["TraceID"] = df["TraceID"].astype(str)
        kept_spans = df.loc[df["TraceID"].isin(retained_ids)]
        kept_spans.to_csv(trc_target, index=False)
        kept_span_ids = set(kept_spans["SpanID"].astype(str))
    else:
        kept_span_ids = set()

    log_target = _materialize_subdir("log", f"{window}_log.csv")
    src_log = src_date_dir / "log" / f"{window}_log.csv"

    # Decide log source by mode + whether the caller supplied a separate log
    # subset path (Mode C, UniSage). When in Mode C but the sampler has no
    # log subset, fall back to Mode-B logic so baselines work in the same call.
    use_unisage_log = (
        mode == "C"
        and log_policy == "native"
        and unisage_log_sampling_csv is not None
        and Path(unisage_log_sampling_csv).exists()
    )
    if use_unisage_log:
        budget = float(log_budget) if log_budget is not None else 1.0
        window_range = abnormal_minute_window(entry)
        nzlog = unisage_log_subset_to_nezha(
            Path(unisage_log_sampling_csv), window_range, budget)
        # Always emit, even if empty: Nezha's pd.read_csv(... usecols=[...])
        # needs the header columns to exist.
        if nzlog.empty:
            pd.DataFrame(columns=["TimeUnixNano", "SpanID", "Log"]).to_csv(
                log_target, index=False)
        else:
            nzlog.to_csv(log_target, index=False)
    elif src_log.exists():
        # Mode B (and Mode C fallback for baselines): logs whose SpanID
        # belongs to a retained-trace span -- "logs naturally carried with
        # retained traces".
        try:
            df = pd.read_csv(src_log, dtype={"SpanID": str})
        except pd.errors.ParserError as exc:
            print(
                f"[nezha_adapter] warning: skipping malformed log rows in {src_log}: {exc}",
                file=sys.stderr,
            )
            df = pd.read_csv(src_log, dtype={"SpanID": str}, on_bad_lines="skip")
        df["SpanID"] = df["SpanID"].astype(str)
        df.loc[df["SpanID"].isin(kept_span_ids)].to_csv(log_target, index=False)

    return scratch_root


# --------------------------------------------------------------------------- #
# Subprocess invocation
# --------------------------------------------------------------------------- #

def run_nezha_subprocess(entry: CaseEntry, rca_path: Path,
                         construct_path: Path,
                         nezha_python: str | None = None,
                         timeout_sec: int = 900) -> dict:
    """Invoke `nezha_single_case.py` in the isolated `nezha` env and parse JSON.

    cwd is fixed to NEZHA_DIR so `alarm.py`'s bare relative `metric_threshold/`
    reference resolves; consequently `rca_path` and `construct_path` MUST be
    absolute, or they'll be resolved relative to NEZHA_DIR (a different place
    than wherever the caller intended).
    """
    nezha_python = resolve_nezha_python(nezha_python)
    rca_path = Path(rca_path).resolve()
    construct_path = Path(construct_path).resolve()
    cmd = [
        nezha_python, str(NEZHA_RUNNER),
        "--ns", entry.ns,
        "--normal-time", entry.normal_time,
        "--inject-time", entry.inject_time,
        "--construct-path", str(construct_path),
        "--rca-path", str(rca_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_sec, cwd=NEZHA_DIR)
    if proc.returncode != 0:
        return {"error": f"exit={proc.returncode}", "stderr_tail": proc.stderr[-2000:]}
    # The runner prints exactly one JSON line as stdout; Nezha's logger writes
    # progress to its file logger, not stdout, so this is robust.
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # pragma: no cover
        return {"error": f"json_parse: {exc}", "stdout_tail": proc.stdout[-2000:]}


# --------------------------------------------------------------------------- #
# CLI: build index / smoke
# --------------------------------------------------------------------------- #

def _cmd_build_index(args: argparse.Namespace) -> None:
    df = write_case_index(
        Path(args.unisage_data_root),
        Path(args.nezha_rca_dir),
        Path(args.out),
    )
    print(f"wrote {len(df)} rows to {args.out}")
    print(df.groupby(["dataset", "batch"]).size().to_string())


def _cmd_smoke(args: argparse.Namespace) -> None:
    """Standalone correctness check: run Nezha on ONE adapter-materialized case
    (full data, no sampling) and verify it produces a non-empty pod ranking.

    Since UniSage's data IS Nezha's data, a Mode-A run with retained_ids=None
    must reproduce what `python main.py --ns ... --level service` would emit
    for that single fault. We don't recompute AS@k here -- the harness will."""
    nezha_python = resolve_nezha_python(args.nezha_python)
    df = build_case_index(Path(args.unisage_data_root), Path(args.nezha_rca_dir))
    row = df[(df["dataset"] == args.dataset) & (df["case"] == args.case)].iloc[0]
    entry = CaseEntry(
        dataset=row["dataset"], batch=row["batch"], case=row["case"], ns=row["ns"],
        nz_date=row["nz_date"], nz_window=row["nz_window"],
        inject_time=row["inject_time"], inject_pod=row["inject_pod"],
        inject_type=row["inject_type"], inject_timestamp=str(row["inject_timestamp"]),
    )
    scratch = Path(args.scratch) / f"smoke_{entry.dataset}_{entry.case}"
    if scratch.exists():
        shutil.rmtree(scratch)
    materialize_case(entry, Path(args.nezha_rca_dir), retained_ids=None,
                     mode="A", scratch_root=scratch)
    construct = Path(args.nezha_construct_dir)
    out = run_nezha_subprocess(entry, scratch, construct, nezha_python=nezha_python)
    print(json.dumps({
        "case": f"{entry.dataset}/{entry.batch}/{entry.case}",
        "ns": entry.ns, "nz_date": entry.nz_date, "nz_window": entry.nz_window,
        "gt_pod": entry.inject_pod, "inject_type": entry.inject_type,
        "n_results": out.get("n_results"),
        "top5_pods": [r["pod"] for r in (out.get("ranked") or [])[:5]],
        "error": out.get("error"),
        "stderr_tail": out.get("stderr_tail"),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="UniSage <-> Nezha bridge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("build-index", help="materialize nezha_case_index.csv")
    p_idx.add_argument("--unisage-data-root", default="data")
    p_idx.add_argument("--nezha-rca-dir", default="baselines/nezha/rca_data")
    p_idx.add_argument("--out", default="results/nezha/nezha_case_index.csv")
    p_idx.set_defaults(func=_cmd_build_index)

    p_smk = sub.add_parser("smoke", help="run Nezha via the adapter on one full-data case")
    p_smk.add_argument("--dataset", required=True, choices=("hipster", "train_ticket"))
    p_smk.add_argument("--case", required=True, help="UniSage case_key, e.g. 03_53")
    p_smk.add_argument("--unisage-data-root", default="data")
    p_smk.add_argument("--nezha-rca-dir", default="baselines/nezha/rca_data")
    p_smk.add_argument("--nezha-construct-dir", default="baselines/nezha/construct_data")
    p_smk.add_argument("--scratch", default="results/nezha/_scratch")
    p_smk.add_argument("--nezha-python", default=None,
                       help="isolated nezha interpreter (or NEZHA_PYTHON; auto-discover named conda env)")
    p_smk.set_defaults(func=_cmd_smoke)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
