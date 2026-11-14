
import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
"""Non-invasive single-case smoke test for the `unisage` conda env.

Runs UniSage.evaluate() on one or more cases WITHOUT modifying UniSage.py
(whose main() is hardcoded to the aiops21 batch). Prints per-case status and
any traceback, so we can confirm the fresh environment runs the pipeline
end-to-end before changing any method code.

Run from the UniSage/ directory:
    python scripts/smoke_test.py [case_path] [hipster|aiops21]
"""
import argparse
import asyncio
import sys
import time
import traceback
from pathlib import Path

from aiomultiprocess import Pool

import UniSage  # safe to import: main() is guarded by __name__ == "__main__"

# Case(s) to run, relative to UniSage/. Override via argv:
#   python scripts/smoke_test.py <case_path> <hipster|aiops21>
CASES = [("data/hipster/batches/batch1/03_53", "hipster")]


async def run():
    failed = False
    async with Pool() as pool:
        for base, dtype in CASES:
            print(f"\n===== SMOKE: {base} ({dtype}) =====", flush=True)
            t = time.time()
            try:
                res = await UniSage.evaluate(Path(base), pool, dtype)
                status = "OK" if res else "RAN (no anomaly / empty result)"
                print(f"[{status}] {base} in {time.time() - t:.1f}s", flush=True)
            except Exception:
                failed = True
                traceback.print_exc()
                print(f"[FAILED] {base} after {time.time() - t:.1f}s", flush=True)
    return not failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single-case UniSage smoke test.")
    parser.add_argument("case_path", nargs="?", type=Path, default=Path(CASES[0][0]))
    parser.add_argument("dataset", nargs="?", choices=["hipster", "aiops21"], default="hipster")
    args = parser.parse_args()
    if not args.case_path.is_dir():
        parser.error(f"Case directory not found: {args.case_path}. Download the dataset or specify case_path.")
    CASES = [(args.case_path.as_posix(), args.dataset)]
    t0 = time.time()
    ok = asyncio.run(run())
    print(f"\nsmoke total {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
