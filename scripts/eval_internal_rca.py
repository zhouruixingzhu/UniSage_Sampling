
import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
"""Measure UniSage's internal RCA (the combined phi_s service ranking) localization
accuracy against ground-truth injected service, for a hipster/train_ticket batch that
has a label.json. Reports AC@1 / AC@3 / MRR and a per-case breakdown.

Run from UniSage/:
    python scripts/eval_internal_rca.py [data/hipster/batches/batch1]

Note: this runs the full evaluate() pipeline per case (which writes final_ranking.csv);
propagation/sampling do not affect final_ranking, so leave them at defaults.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from aiomultiprocess import Pool

import UniSage

BATCH = Path("data/hipster/batches/batch1")


async def run():
    labels = json.load(open(BATCH / "label.json"))
    rows = []
    async with Pool() as pool:
        for case, info in labels.items():
            cdir = BATCH / case
            gt = info.get("inject_pod")
            itype = info.get("inject_type")
            if not cdir.is_dir():
                rows.append((case, itype, gt, "NO_DIR", 0)); continue
            try:
                await UniSage.evaluate(cdir, pool, "hipster")
            except Exception as e:
                rows.append((case, itype, gt, f"ERROR:{type(e).__name__}", 0)); continue
            fr = cdir / "final_ranking.csv"
            if not fr.exists():
                rows.append((case, itype, gt, "NO_RANKING", 0)); continue
            ranked = list(pd.read_csv(fr).get("service_name", []))
            rank = (ranked.index(gt) + 1) if gt in ranked else 0
            top1 = ranked[0] if ranked else None
            rows.append((case, itype, gt, top1, rank))

    n = len(rows)
    ac1 = sum(1 for r in rows if r[4] == 1) / n
    ac3 = sum(1 for r in rows if 1 <= r[4] <= 3) / n
    mrr = sum((1.0 / r[4]) for r in rows if r[4] > 0) / n
    print(f"\n=== Internal RCA (phi_s) localization on {BATCH}  ({n} cases) ===")
    print(f"AC@1={ac1:.3f}   AC@3={ac3:.3f}   MRR={mrr:.3f}")
    print(f"{'case':8} {'inject_type':16} {'rank_gt':>7}  top1_pred  (gt)")
    for case, itype, gt, top1, rank in rows:
        print(f"{case:8} {str(itype):16} {str(rank):>7}  {top1}   (gt={gt})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UniSage's internal RCL on a Hipster/Train Ticket batch.")
    parser.add_argument("batch", nargs="?", type=Path, default=BATCH)
    args = parser.parse_args()
    BATCH = args.batch
    if not (BATCH / "label.json").is_file():
        parser.error(f"Batch label file not found: {BATCH / 'label.json'}")
    asyncio.run(run())
