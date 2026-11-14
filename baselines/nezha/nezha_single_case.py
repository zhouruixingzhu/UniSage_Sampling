"""[UniSage] Single-fault Nezha runner — emits ranked-pod JSON to stdout.

Thin alternative to upstream `main.py`: instead of accumulating AS@k across an
entire fault_list, run Nezha on ONE injected fault with caller-supplied paths
to the (possibly scratch / sampled) `rca_data` and full `construct_data`. The
return is the raw ranked-pod list from `pattern_ranker.pattern_ranker` so the
UniSage harness can compute AC@1/AC@3/MRR against the ground-truth pod itself.

Used by `UniSage/nezha_adapter.py` and `UniSage/eval_nezha_rca_on_sampled_traces.py`.
This file must live INSIDE baselines/nezha/ because Nezha's modules pin paths to
`dirname(__file__)` (drain3 template state at `log_template/*.bin`, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make Nezha's own modules importable when invoked as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pattern_ranker import get_pattern, pattern_ranker  # noqa: E402
from main import get_miner  # noqa: E402


def compute_abnormal_time(inject_time_str: str) -> str:
    """Mirror upstream `evaluation_pod`'s inject_time -> abnormal_time logic (+2 min).

    Upstream collapses seconds (`fault["inject_time"].split(":")[0]` + `:MM`), so we
    do the same here for bit-identical results.
    """
    date, hm = inject_time_str.split(" ")
    hh, mm = hm.split(":")[0], hm.split(":")[1]
    mm = int(mm) + 2
    hh = int(hh)
    if mm >= 60:
        hh += 1
        mm -= 60
    return f"{date} {hh:02d}:{mm:02d}"


def run_single(ns: str, normal_time: str, inject_time: str,
               construct_path: str, rca_path: str) -> dict:
    miner = get_miner(ns)
    normal_pattern_list, normal_event_graphs, _ = get_pattern(
        normal_time, ns, construct_path, miner)
    abnormal_time = compute_abnormal_time(inject_time)
    result_list, _ = pattern_ranker(
        normal_pattern_list, normal_event_graphs, abnormal_time, ns, miner,
        rca_path=rca_path,
    )
    return {
        "abnormal_time": abnormal_time,
        "n_results": len(result_list),
        "ranked": [
            {"pod": r.get("pod"), "score": float(r.get("score", 0.0)),
             "depth": int(r.get("deepth", 0)), "events": r.get("events"),
             "resource": r.get("resource")}
            for r in result_list
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True, choices=["hipster", "ts"])
    ap.add_argument("--normal-time", required=True,
                    help="Normal-window string, e.g. '2022-08-22 03:51'")
    ap.add_argument("--inject-time", required=True,
                    help="Fault inject_time, e.g. '2022-08-22 03:53:54'")
    ap.add_argument("--construct-path", required=True,
                    help="Path to construct_data dir (typically baselines/nezha/construct_data)")
    ap.add_argument("--rca-path", required=True,
                    help="Path to rca_data dir (scratch dir with filtered traceid for sampled-subset eval)")
    ap.add_argument("--out-json", default=None,
                    help="Optional JSON output path; otherwise prints to stdout")
    args = ap.parse_args()

    out = run_single(args.ns, args.normal_time, args.inject_time,
                     args.construct_path, args.rca_path)
    payload = json.dumps(out)
    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
