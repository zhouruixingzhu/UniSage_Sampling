"""UniSage <-> TraceRCA data bridge.

TraceRCA (IWQoS'21) is a **trace-only** root-cause localizer: it turns each trace
into a set of caller->callee invocations, flags anomalous invocations with a
per-edge k-sigma test against a fault-free reference, then ranks root-cause
services by association-rule mining over the anomalous traces.

This adapter:
  * ``build_invocations`` -- turn a normalized span DataFrame (TraceID, SpanID,
    ParentID, PodName, timing) into TraceRCA's invocation table
    (trace_id, source, target, latency) via a parent->child span join. One code
    path serves Hipster / Train Ticket (Nezha-lineage) and AIOps21 because
    ``AD.trace.trace_processing.normalize_trace_columns`` unifies the schema.
  * ``materialize_case`` -- write the (possibly sampled) abnormal invocation
    table and the FULL fault-free historical invocation table into a scratch
    dir consumed by ``baselines/tracerca/tracerca_single_case.py``.
  * ``run_tracerca_subprocess`` -- run that single-case runner in the dedicated
    ``tracerca`` conda env and parse its ranked-service JSON.

The sampling hook is the abnormal invocation table: only retained TraceIDs feed
it. The fault-free reference stays full (UniSage samples traces, not the
fault-free baseline) -- the same convention used by the fair-RCA / MicroRCA /
Nezha harnesses.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACERCA_DIR = PROJECT_ROOT / "baselines" / "tracerca"
TRACERCA_RUNNER = TRACERCA_DIR / "tracerca_single_case.py"


def resolve_tracerca_python(python_exe: str | None = None) -> str:
    """Locate the dedicated tracerca environment; never use a driver-env fallback."""
    configured = python_exe or os.environ.get("TRACERCA_PYTHON")
    if configured:
        expanded = os.path.expanduser(configured)
        found = shutil.which(expanded)
        if found:
            # Keep the environment's path (resolving symlinks can select its base env).
            return str(Path(found).absolute())
        raise ValueError(
            f"tracerca interpreter is not an executable file: {configured}. "
            "Use --python /path/to/tracerca/bin/python or set TRACERCA_PYTHON."
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
        candidates = [root / "envs" / "tracerca" / "bin" / "python"]
        if root.name == "tracerca" and root.parent.name == "envs":
            candidates.insert(0, root / "bin" / "python")
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.absolute())
    raise ValueError(
        "Could not find the isolated conda environment 'tracerca'. Create it using "
        "baselines/tracerca/PROVENANCE.md, then use --python "
        "/path/to/tracerca/bin/python or set TRACERCA_PYTHON. "
        "The current driver interpreter is not used as a fallback."
    )


INVO_COLUMNS = ["trace_id", "source", "target", "latency"]


# --- shared with microrca_adapter (copied to avoid coupling to nezha rca_data) ---
def service_base(name: object) -> str:
    """Strip a k8s replicaset+pod suffix to the logical service name."""
    s = str(name or "").strip()
    s = s.removesuffix("_metric").removesuffix("_metric.csv").removesuffix(".csv")
    m = re.match(r"(.+)-[a-z0-9]{9,}-[a-z0-9]{5}$", s)
    return m.group(1) if m else s


def first_service_rank(ranked: list[str], gt: str) -> int:
    """1-indexed rank of a ground-truth pod/service in a service-level list."""
    if not gt:
        return 0
    gt_service = service_base(gt)
    for i, name in enumerate(ranked, 1):
        svc = service_base(name)
        if svc == gt_service or name == gt or str(name).startswith(gt) or gt.startswith(str(name)):
            return i
    return 0


def default_forbidden(dataset: str) -> frozenset:
    """Entry services to drop from itemsets.

    Upstream hardcodes ``{"gateway"}`` for Train Ticket (the Istio gateway is the
    ubiquitous entry node and never an injected service). Hipster's analogous
    entry ``frontend`` CAN be the injected service, so it must NOT be forbidden;
    AIOps21 has no single safe entry to drop. See PROVENANCE.md.
    """
    if dataset == "train_ticket":
        # service_base("ts-gateway-service-<rs>-<pod>") == "ts-gateway-service".
        # Upstream forbids "gateway" under its simple_name scheme; ours keeps the
        # ts-*-service form, so forbid that.
        return frozenset({"ts-gateway-service"})
    return frozenset()


def fault_window_spans(spans: pd.DataFrame, dataset: str, case: str,
                       window_min: int = 3) -> pd.DataFrame:
    """Restrict abnormal spans to the fault window for Hipster / Train Ticket.

    The case name encodes the injection minute (``HH_MM``) and is the *robust*
    window anchor: the ``inject_timestamp`` field in some Train Ticket cases is
    8h off (a UTC/local-time inconsistency in the label files), whereas the case
    name always matches the ``inject_time`` string and the per-minute trace
    files. Span ``StartTimeUnixNano`` is stored as local-wall-clock-as-unix, so
    interpreting it as UTC recovers the local minute-of-day.

    Window = ``[case minute, case minute + window_min)``. This concentrates the
    fault signal that a 9-minute Hipster window otherwise dilutes; Train Ticket
    cases already ship only ~3 minute files, so it is a near no-op there.

    AIOps21 case names are not ``HH_MM`` and its ``abnormal/`` dir is already the
    fault window, so spans are returned unchanged for any non-HH_MM case.
    """
    if dataset not in ("hipster", "train_ticket") or spans.empty:
        return spans
    if "StartTimeUnixNano" not in spans.columns:
        return spans
    parts = str(case).split("_")
    if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return spans
    start_mod = int(parts[0]) * 60 + int(parts[1])
    dt = pd.to_datetime(pd.to_numeric(spans["StartTimeUnixNano"], errors="coerce"), unit="ns")
    mod = dt.dt.hour * 60 + dt.dt.minute
    return spans[(mod >= start_mod) & (mod < start_mod + window_min)]


def build_invocations(spans: pd.DataFrame) -> pd.DataFrame:
    """Parent->child span join -> TraceRCA invocation table.

    Each child span becomes one invocation ``(source=parent service,
    target=child service)`` with ``latency`` = the child span's response time.
    Self-edges and parent-less (root) spans are dropped, mirroring upstream
    ``run_invo_encoding.py`` (which keeps only ``source != target`` edges).
    """
    empty = pd.DataFrame(columns=INVO_COLUMNS)
    if spans is None or spans.empty:
        return empty
    required = {"TraceID", "SpanID", "ParentID", "PodName"}
    if not required.issubset(spans.columns):
        return empty

    base = spans.copy()
    base["TraceID"] = base["TraceID"].astype(str)
    base["SpanID"] = base["SpanID"].astype(str)
    base["ParentID"] = base["ParentID"].astype(str)

    parents = base[["TraceID", "SpanID", "PodName"]].rename(
        columns={"SpanID": "_ParentSpanID", "PodName": "_ParentPod"}
    )
    child = base.merge(
        parents,
        left_on=["TraceID", "ParentID"],
        right_on=["TraceID", "_ParentSpanID"],
        how="left",
    ).dropna(subset=["_ParentPod"])
    if child.empty:
        return empty

    child["source"] = child["_ParentPod"].map(service_base)
    child["target"] = child["PodName"].map(service_base)
    child = child[(child["source"] != "") & (child["target"] != "") & (child["source"] != child["target"])]
    if child.empty:
        return empty

    if {"StartTimeUnixNano", "EndTimeUnixNano"}.issubset(child.columns):
        start = pd.to_numeric(child["StartTimeUnixNano"], errors="coerce")
        end = pd.to_numeric(child["EndTimeUnixNano"], errors="coerce")
        latency = (end - start).clip(lower=0) / 1e6  # ns -> ms
    elif "Duration" in child.columns:
        latency = pd.to_numeric(child["Duration"], errors="coerce")
    else:
        latency = pd.Series(0.0, index=child.index)

    out = pd.DataFrame({
        "trace_id": child["TraceID"].astype(str),
        "source": child["source"].astype(str),
        "target": child["target"].astype(str),
        "latency": latency.astype(float),
    })
    out = out.dropna(subset=["latency"]).reset_index(drop=True)
    return out


def materialize_case(
    abnormal_invo: pd.DataFrame,
    historical_invo: pd.DataFrame,
    scratch_root: Path,
    meta: Optional[dict] = None,
) -> Path:
    scratch_root.mkdir(parents=True, exist_ok=True)
    abnormal_invo.to_pickle(scratch_root / "invo_abnormal.pkl")
    historical_invo.to_pickle(scratch_root / "invo_historical.pkl")
    payload = dict(meta or {})
    payload.update(
        n_abnormal_invocations=int(len(abnormal_invo)),
        n_abnormal_traces=int(abnormal_invo["trace_id"].nunique()) if len(abnormal_invo) else 0,
        n_historical_invocations=int(len(historical_invo)),
    )
    (scratch_root / "case_meta.json").write_text(json.dumps(payload, indent=2))
    return scratch_root


def run_tracerca_subprocess(
    scratch: Path,
    python_exe: Optional[str] = None,
    sigma: float = 1.0,
    fisher: float = 3.0,
    min_support: float = 0.1,
    k: int = 100,
    features: Iterable[str] = ("latency",),
    forbidden: Iterable[str] = (),
    timeout_sec: int = 300,
) -> dict:
    exe = resolve_tracerca_python(python_exe)
    cmd = [
        exe,
        str(TRACERCA_RUNNER),
        "--case-dir", str(scratch),
        "--sigma", str(sigma),
        "--fisher", str(fisher),
        "--min-support", str(min_support),
        "--k", str(k),
        "--features", ",".join(features),
        "--forbidden", ",".join(forbidden),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"ranked": [], "error": f"timeout after {timeout_sec}s", "stderr_tail": str(exc)}

    stdout = proc.stdout.strip()
    if proc.returncode != 0:
        return {
            "ranked": [],
            "error": f"returncode={proc.returncode}",
            "stdout_tail": stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }
    try:
        return json.loads(stdout.splitlines()[-1])
    except Exception as exc:
        return {
            "ranked": [],
            "error": f"json parse failed: {exc}",
            "stdout_tail": stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }


# ----------------------------- smoke CLI --------------------------------------
def _smoke() -> None:
    import argparse

    sys.path.insert(0, str(PROJECT_ROOT))
    from AD.trace.trace_processing import normalize_trace_columns
    from eval_fair_rca_on_sampled_traces import GroundTruth, load_spans

    ap = argparse.ArgumentParser(description="Full-data adapter fidelity smoke for one case.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--python", default=None,
                    help="isolated tracerca interpreter (or TRACERCA_PYTHON; auto-discover named conda env)")
    ap.add_argument("--out", default="results/tracerca/_smoke")
    args = ap.parse_args()
    try:
        args.python = resolve_tracerca_python(args.python)
    except ValueError as exc:
        ap.error(str(exc))

    data_root = Path(args.data_root)
    if args.dataset in ("hipster", "train_ticket"):
        case_dir = data_root / "data" / args.dataset / "batches" / args.batch / args.case
    else:
        case_dir = data_root / "data" / args.dataset / "all" / args.case
    gt = GroundTruth(data_root).inject_service(args.dataset, args.case, case_dir)

    abn = load_spans(case_dir, args.dataset, normal=False)
    nrm = load_spans(case_dir, args.dataset, normal=True)
    abnormal_invo = build_invocations(abn)
    historical_invo = build_invocations(nrm)

    scratch = Path(args.out) / f"{args.dataset}__{args.batch}__{args.case}"
    materialize_case(abnormal_invo, historical_invo, scratch,
                     meta=dict(dataset=args.dataset, batch=args.batch, case=args.case, gt=gt))
    result = run_tracerca_subprocess(
        scratch, python_exe=args.python, forbidden=default_forbidden(args.dataset),
    )
    ranked = result.get("ranked", [])
    rank = first_service_rank(ranked, gt or "")
    print(json.dumps(dict(
        case=f"{args.dataset}/{args.batch}/{args.case}", gt=gt,
        rank=rank, top5=ranked[:5], diagnostics=result.get("diagnostics"),
        error=result.get("error"),
    ), indent=2))


if __name__ == "__main__":
    _smoke()
