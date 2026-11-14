#!/usr/bin/env python3
"""Aggregate RQ5 runtime measurements from existing v0.2 artefacts.

The script intentionally reads only completed sampling artefacts and writes
under results/rq5_runtime/ by default. It does not run samplers or evaluators.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "results" / "v02_sampling_runs"
OUT_DIR = ROOT / "results" / "rq5_runtime"
BASELINE_RUNTIME_MD = (
    ROOT / "scripts" / "reference_data" / "seed1_runtime_table.md"
)
EXCLUDED_INPUTS: list[str] = []


def display_path(path: Path) -> str:
    """Use repository-relative provenance when possible, absolute paths otherwise."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class RunSpec:
    method: str
    scope: str
    pattern: str
    csv_name: str = "sample_index.csv"
    measured_work: str = "trace sampling"
    caveat: str = ""


RUN_SPECS = [
    RunSpec(
        "Sieve",
        "hip_tt_primary",
        "sieve_v02_hipster_train_ticket_primary_seed-*_budgets-*_fp-a468bf3f6701",
    ),
    RunSpec("Sieve", "aiops21_primary", "sieve_v02_aiops21_primary_seed-*_serial"),
    RunSpec(
        "Sifter",
        "hip_tt_primary",
        "sifter_v02_hipster_train_ticket_primary_seed-*_budgets-*_fp-5d38b557f13b",
    ),
    RunSpec(
        "Sifter",
        "aiops21_primary",
        "sifter_v02_aiops21_primary_seed-*_budgets-*_fp-5d38b557f13b",
    ),
    RunSpec(
        "TraStrainer",
        "hip_tt_primary",
        "trastrainer_v02_hipster_train_ticket_primary_seed-[0-9]",
    ),
    RunSpec(
        "TraStrainer",
        "aiops21_primary",
        "trastrainer_v02_aiops21_primary_seed-[0-9]",
    ),
    RunSpec(
        "TracePicker",
        "hip_tt_primary",
        "tracepicker_v02_hip_tt_primary_portable_seed[0-9]",
    ),
    RunSpec(
        "TracePicker",
        "aiops21_primary",
        "tracepicker_v02_aiops21_primary_portable_seed[0-9]",
    ),
    RunSpec(
        "Gleaner",
        "hip_tt_primary",
        "gleaner_v02_hipster_train_ticket_primary_seed-*_budgets-*_fp-5ec54d83a83d",
        measured_work="trace sampling with log-enriched scoring",
    ),
    RunSpec(
        "Gleaner",
        "aiops21_primary",
        "gleaner_v02_aiops21_primary_seed-*_budgets-*_fp-5ec54d83a83d",
        measured_work="trace sampling with log-enriched scoring",
    ),
    RunSpec(
        "UniSage-v1 trace route",
        "hip_tt_primary",
        "unisage_v1_rep*_budget8_hip_tt_primary",
        measured_work="analysis-guided trace sampling",
    ),
    RunSpec(
        "UniSage-v1 trace route",
        "aiops21_primary",
        "unisage_v1_rep*_budget8_aiops21_primary",
        measured_work="analysis-guided trace sampling",
    ),
    RunSpec(
        "UniSage-v1 trace route",
        "aiops21_excluded",
        "unisage_v1_rep*_budget8_aiops21_excluded",
        measured_work="analysis-guided trace sampling",
    ),
    RunSpec(
        "UniSage-v1 log route",
        "hip_tt_primary",
        "unisage_v1_rep*_budget8",
        csv_name="log_sample_index_clean.csv",
        measured_work="native log sampling route",
        caveat="available only for Hipster+Train Ticket clean log reruns",
    ),
]


def parse_run_id(path: Path) -> str:
    rep = re.search(r"rep(\d+)", path.name)
    seed = re.search(r"seed-?(\d+)", path.name)
    if rep:
        return f"rep{rep.group(1)}"
    if seed:
        return f"seed{seed.group(1)}"
    return path.name


def scope_label(scope: str) -> str:
    return {
        "hip_tt_primary": "Hipster+Train Ticket primary",
        "aiops21_primary": "AIOps21 primary",
        "aiops21_excluded": "AIOps21 excluded",
    }.get(scope, scope)


def read_elapsed_rows(spec: RunSpec, run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / spec.csv_name
    if not csv_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "elapsed_sec" not in (reader.fieldnames or []):
            return []
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue
            elapsed_raw = row.get("elapsed_sec", "")
            try:
                elapsed = float(elapsed_raw)
            except ValueError:
                continue
            if elapsed < 0:
                continue
            dataset = row.get("dataset", "")
            batch = row.get("batch", "")
            case = row.get("case", "")
            rows.append(
                {
                    "method": spec.method,
                    "scope": spec.scope,
                    "dataset": dataset,
                    "batch": batch,
                    "case": case,
                    "run_id": parse_run_id(run_dir),
                    "run_dir": display_path(run_dir),
                    "elapsed_sec": f"{elapsed:.9f}",
                    "measured_work": spec.measured_work,
                    "caveat": spec.caveat,
                }
            )
    return rows


def collect_observations() -> list[dict[str, str]]:
    observations: list[dict[str, str]] = []
    for spec in RUN_SPECS:
        for run_dir in sorted(RUN_ROOT.glob(spec.pattern)):
            if not run_dir.is_dir():
                continue
            rows = read_elapsed_rows(spec, run_dir)
            if is_incomplete_sieve_hip_tt(spec, rows):
                EXCLUDED_INPUTS.append(
                    f"`{display_path(run_dir)}`: excluded from main Sieve "
                    "Hipster+Train Ticket runtime because it contains only "
                    f"{count_unique_cases(rows)} unique cases; the complete "
                    f"seed=1 value is imported from `{display_path(BASELINE_RUNTIME_MD)}`."
                )
                continue
            observations.extend(rows)
    observations.extend(derive_unisage_trace_log_rows())
    return observations


def count_unique_cases(rows: list[dict[str, str]]) -> int:
    return len({(r["dataset"], r["batch"], r["case"]) for r in rows})


def is_incomplete_sieve_hip_tt(spec: RunSpec, rows: list[dict[str, str]]) -> bool:
    return (
        spec.method == "Sieve"
        and spec.scope == "hip_tt_primary"
        and rows
        and count_unique_cases(rows) < 89
    )


def load_csv_by_case(path: Path) -> dict[tuple[str, str, str], float]:
    rows: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue
            try:
                elapsed = float(row.get("elapsed_sec", ""))
            except ValueError:
                continue
            key = (row.get("dataset", ""), row.get("batch", ""), row.get("case", ""))
            rows[key] = elapsed
    return rows


def derive_unisage_trace_log_rows() -> list[dict[str, str]]:
    """Derive a transparent trace+log sum for Hipster+TT where both routes exist."""

    rows: list[dict[str, str]] = []
    for rep in range(1, 6):
        trace_dir = RUN_ROOT / f"unisage_v1_rep{rep}_budget8_hip_tt_primary"
        log_dir = RUN_ROOT / f"unisage_v1_rep{rep}_budget8"
        trace_rows = load_csv_by_case(trace_dir / "sample_index.csv")
        log_rows = load_csv_by_case(log_dir / "log_sample_index_clean.csv")
        for key in sorted(set(trace_rows) & set(log_rows)):
            dataset, batch, case = key
            elapsed = trace_rows[key] + log_rows[key]
            rows.append(
                {
                    "method": "UniSage-v1 trace+log route",
                    "scope": "hip_tt_primary",
                    "dataset": dataset,
                    "batch": batch,
                    "case": case,
                    "run_id": f"rep{rep}",
                    "run_dir": (
                        f"derived:unisage_v1_rep{rep}_budget8_hip_tt_primary"
                        f"+unisage_v1_rep{rep}_budget8"
                    ),
                    "elapsed_sec": f"{elapsed:.9f}",
                    "measured_work": "derived sum of separate trace and log routes",
                    "caveat": "not a fused end-to-end timer; sums two measured routes",
                }
            )
    return rows


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), statistics.stdev(vals)


def summarize(observations: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for obs in observations:
        grouped[(obs["method"], obs["scope"])].append(obs)

    summaries: list[dict[str, object]] = []
    for (method, scope), rows in sorted(grouped.items()):
        values = [float(r["elapsed_sec"]) for r in rows]
        mean, std = mean_std(values)
        run_ids = sorted({r["run_id"] for r in rows}, key=natural_key)
        cases = {(r["dataset"], r["batch"], r["case"]) for r in rows}
        datasets = sorted({r["dataset"] for r in rows})
        caveats = sorted({r["caveat"] for r in rows if r["caveat"]})
        if len(run_ids) == 1:
            caveats.append(f"single measured run ({run_ids[0]}); std is across cases only")
        if method == "Uniform":
            caveats.append("uniform runtime not measured in current artefacts")
        summaries.append(
            {
                "method": method,
                "scope": scope,
                "mean_sec_per_case": mean,
                "std_sec_per_case": std,
                "observations": len(rows),
                "unique_cases": len(cases),
                "runs": ", ".join(run_ids),
                "n_runs": len(run_ids),
                "datasets": ", ".join(datasets),
                "measured_work": rows[0]["measured_work"],
                "caveat": "; ".join(dict.fromkeys(caveats)),
            }
        )
    apply_seed1_reference_fallbacks(summaries)
    return summaries


def parse_runtime_cell(cell: str) -> tuple[float, float, int] | None:
    normalized = cell.replace("\u00b1", "+/-")
    match = re.search(
        r"([0-9.]+)\s*\+/-\s*([0-9.]+)\s*\(n=([0-9]+)\)", normalized
    )
    if not match:
        return None
    return float(match.group(1)), float(match.group(2)), int(match.group(3))


def parse_seed1_runtime_table() -> dict[tuple[str, str], tuple[float, float, int]]:
    values: dict[tuple[str, str], tuple[float, float, int]] = {}
    if not BASELINE_RUNTIME_MD.exists():
        return values
    for line in BASELINE_RUNTIME_MD.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0].lower() in {"method", "---"}:
            continue
        method = {
            "sifter": "Sifter",
            "sieve": "Sieve",
            "trastrainer": "TraStrainer",
            "tracepicker": "TracePicker",
            "gleaner": "Gleaner",
        }.get(cells[0].lower())
        if not method:
            continue
        hip = parse_runtime_cell(cells[1])
        aio = parse_runtime_cell(cells[2])
        if hip:
            values[(method, "hip_tt_primary")] = hip
        if aio:
            values[(method, "aiops21_primary")] = aio
    return values


def apply_seed1_reference_fallbacks(summaries: list[dict[str, object]]) -> None:
    existing = {(s["method"], s["scope"]): s for s in summaries}
    reference = parse_seed1_runtime_table()
    fallback_key = ("Sieve", "hip_tt_primary")
    if fallback_key in existing or fallback_key not in reference:
        return
    mean, std, n = reference[fallback_key]
    summaries.append(
        {
            "method": "Sieve",
            "scope": "hip_tt_primary",
            "mean_sec_per_case": mean,
            "std_sec_per_case": std,
            "observations": n,
            "unique_cases": n,
            "runs": "seed1",
            "n_runs": 1,
            "datasets": "hipster, train_ticket",
            "measured_work": "trace sampling",
            "caveat": (
                "seed=1 aggregate imported from "
                f"`{display_path(BASELINE_RUNTIME_MD)}`; "
                "raw complete per-case sample_index is not present in the "
                "current UniSage tree"
            ),
        }
    )


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def fmt_mean_std(summary: dict[str, object] | None) -> str:
    if summary is None:
        return "not measured"
    mean = float(summary["mean_sec_per_case"])
    std = float(summary["std_sec_per_case"])
    n_obs = int(summary["observations"])
    n_cases = int(summary["unique_cases"])
    n_runs = int(summary["n_runs"])
    return f"{mean:.1f} +/- {std:.1f} (obs={n_obs}, cases={n_cases}, runs={n_runs})"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_seed1_reference() -> str:
    if not BASELINE_RUNTIME_MD.exists():
        return (
            f"Seed=1 recovery runtime reference not found at "
            f"`{display_path(BASELINE_RUNTIME_MD)}`."
        )
    rel = display_path(BASELINE_RUNTIME_MD)
    content = BASELINE_RUNTIME_MD.read_text()
    content = (
        content.replace("\u00b1", "+/-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return f"Source `{rel}`:\n\n{content.strip()}"


def write_runtime_markdown(summaries: list[dict[str, object]]) -> None:
    by_key = {(s["method"], s["scope"]): s for s in summaries}
    method_order = [
        "Uniform",
        "Sieve",
        "Sifter",
        "TraStrainer",
        "TracePicker",
        "Gleaner",
        "UniSage-v1 trace route",
        "UniSage-v1 log route",
        "UniSage-v1 trace+log route",
    ]
    measured_work = {
        s["method"]: str(s["measured_work"])
        for s in summaries
        if s["method"] not in {"UniSage-v1 trace+log route"}
    }
    measured_work["Uniform"] = "not measured"
    measured_work["UniSage-v1 trace+log route"] = (
        "derived sum of separate trace and log routes"
    )

    lines: list[str] = []
    lines.append("# RQ5 Runtime Aggregation")
    lines.append("")
    lines.append(
        "Unit: seconds per case. Mean +/- std is computed over successful "
        "`elapsed_sec` observations from the listed sample indexes; for multi-seed "
        "runs, observations are case x seed. The std therefore captures case-size "
        "variation as well as run-to-run jitter."
    )
    lines.append("")
    lines.append("## Main Section 9.1 Shape")
    lines.append("")
    lines.append(
        "| Method | Hipster+Train Ticket sec/case | AIOps21 primary sec/case | "
        "Measured work | Caveat |"
    )
    lines.append("|---|---:|---:|---|---|")
    for method in method_order:
        hip = by_key.get((method, "hip_tt_primary"))
        aio = by_key.get((method, "aiops21_primary"))
        caveats = []
        for summary in (hip, aio):
            if summary and summary.get("caveat"):
                caveats.append(f"{scope_label(str(summary['scope']))}: {summary['caveat']}")
        if method == "Uniform":
            caveats.append(
                "No standalone uniform wall-clock sampler timing was found in current artefacts."
            )
        lines.append(
            f"| {method} | {fmt_mean_std(hip)} | {fmt_mean_std(aio)} | "
            f"{measured_work.get(method, '')} | {'; '.join(caveats) or '-'} |"
        )

    lines.append("")
    lines.append("## Additional Measured Scope")
    lines.append("")
    lines.append("| Method | Scope | sec/case | Measured work | Caveat |")
    lines.append("|---|---|---:|---|---|")
    for summary in summaries:
        if summary["scope"] == "aiops21_excluded":
            lines.append(
                f"| {summary['method']} | {scope_label(str(summary['scope']))} | "
                f"{fmt_mean_std(summary)} | {summary['measured_work']} | "
                f"{summary['caveat'] or '-'} |"
            )

    lines.append("")
    lines.append("## Detailed Rows")
    lines.append("")
    lines.append(
        "| Method | Scope | Mean sec/case | Std sec/case | Observations | "
        "Unique cases | Runs | Datasets | Measured work | Caveat |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|---|")
    for summary in summaries:
        lines.append(
            f"| {summary['method']} | {scope_label(str(summary['scope']))} | "
            f"{float(summary['mean_sec_per_case']):.3f} | "
            f"{float(summary['std_sec_per_case']):.3f} | "
            f"{summary['observations']} | {summary['unique_cases']} | "
            f"{summary['runs']} | {summary['datasets']} | "
            f"{summary['measured_work']} | {summary['caveat'] or '-'} |"
        )

    lines.append("")
    lines.append("## Seed=1 Recovery Reference")
    lines.append("")
    lines.append(parse_seed1_reference())
    lines.append("")
    if EXCLUDED_INPUTS:
        lines.append("## Excluded Partial Inputs")
        lines.append("")
        lines.extend(f"- {item}" for item in EXCLUDED_INPUTS)
        lines.append("")
    lines.append("## Breakdown Status")
    lines.append("")
    lines.append(
        "No `breakdown_bar.csv` was emitted because current UniSage-v1 artefacts "
        "do not contain recoverable per-stage wall-clock timings for "
        "normalization, AD scoring, RCA/RCL, propagation, trace sampling, and "
        "log sampling. See `breakdown_bar_limitation.md`."
    )
    lines.append("")
    (OUT_DIR / "runtime_table.md").write_text("\n".join(lines))


def write_breakdown_limitation() -> None:
    checked = [
        "unisage_v1_rep1_budget8_hip_tt_primary/run_summary.md",
        "unisage_v1_rep1_budget8_aiops21_primary/run_summary.md",
        "unisage_v1_rep1_budget8_aiops21_excluded/run_summary.md",
        "unisage_v1_rep1_budget8_hip_tt_primary/run_provenance.json",
        "unisage_v1_rep1_budget8/log_sample_index_clean.csv",
        "unisage_v1_rep1_budget8/log_provenance_clean/*.log_sampling_provenance.json",
    ]
    text = """# UniSage-v1 Runtime Breakdown Limitation

Prompt E requested a stacked-bar source CSV with normalization, AD scoring,
RCA/RCL, propagation, trace sampling, and log sampling.

Current artefacts are insufficient to recover those stage timings without
fabrication:

- UniSage-v1 trace `sample_index.csv` files expose only route-level
  `elapsed_sec`, `n_ranked`, and checksum columns.
- UniSage-v1 `run_summary.md` files report only dataset-level total elapsed
  seconds, case counts, and evaluator summaries.
- UniSage-v1 `run_provenance.json` files record configuration and provenance,
  not stage timings.
- `log_sample_index_clean.csv` records route-level elapsed seconds for the
  clean log rerun, but not normalization/AD/RCA/propagation/log-sampling stage
  splits.
- Per-case log provenance and propagation diagnostics record configuration and
  changed-row counts, not wall-clock timings.

Checked sources:
"""
    text += "\n".join(f"- `{display_path(RUN_ROOT / path)}`" for path in checked)
    text += (
        "\n\nTherefore `breakdown_bar.csv` is intentionally not generated. "
        "A future run should instrument these stages at the UniSage-v1 route "
        "level before plotting the stacked runtime bar.\n"
    )
    (OUT_DIR / "breakdown_bar_limitation.md").write_text(text)


def main() -> None:
    global RUN_ROOT, OUT_DIR, BASELINE_RUNTIME_MD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT,
                        help="Directory containing completed sampling run archives")
    parser.add_argument("--output", type=Path, default=OUT_DIR,
                        help="Directory for generated runtime summaries")
    parser.add_argument("--seed1-runtime-table", type=Path, default=BASELINE_RUNTIME_MD,
                        help="Seed-1 runtime reference table (see scripts/reference_data/README.md)")
    args = parser.parse_args()
    RUN_ROOT = args.run_root.resolve()
    OUT_DIR = args.output.resolve()
    BASELINE_RUNTIME_MD = args.seed1_runtime_table.resolve()
    EXCLUDED_INPUTS.clear()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = collect_observations()
    summaries = summarize(observations)
    summaries.sort(
        key=lambda s: (
            [
                "Uniform",
                "Sieve",
                "Sifter",
                "TraStrainer",
                "TracePicker",
                "Gleaner",
                "UniSage-v1 trace route",
                "UniSage-v1 log route",
                "UniSage-v1 trace+log route",
            ].index(s["method"])
            if s["method"]
            in {
                "Uniform",
                "Sieve",
                "Sifter",
                "TraStrainer",
                "TracePicker",
                "Gleaner",
                "UniSage-v1 trace route",
                "UniSage-v1 log route",
                "UniSage-v1 trace+log route",
            }
            else 99,
            str(s["scope"]),
        )
    )

    write_csv(
        OUT_DIR / "runtime_observations.csv",
        observations,
        [
            "method",
            "scope",
            "dataset",
            "batch",
            "case",
            "run_id",
            "run_dir",
            "elapsed_sec",
            "measured_work",
            "caveat",
        ],
    )
    write_csv(
        OUT_DIR / "runtime_summary.csv",
        summaries,
        [
            "method",
            "scope",
            "mean_sec_per_case",
            "std_sec_per_case",
            "observations",
            "unique_cases",
            "runs",
            "n_runs",
            "datasets",
            "measured_work",
            "caveat",
        ],
    )
    write_runtime_markdown(summaries)
    write_breakdown_limitation()
    print(f"Wrote {display_path(OUT_DIR)}")


if __name__ == "__main__":
    main()
