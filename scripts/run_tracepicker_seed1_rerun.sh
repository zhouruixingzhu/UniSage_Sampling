#!/usr/bin/env bash
# Rerun TracePicker v0.2 seed=1 for hip+TT primary and aiops21 primary
# with the current 8-budget set (matches seeds 2-5).
# Legacy 5-budget seed1 dirs have been renamed to *_legacy_5budgets/.
set -euo pipefail

# Activate the desired environment before running this script. PYTHON_BIN may
# name a command on PATH or an executable path (relative to the caller's cwd).
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! PYTHON_BIN=$(command -v "$PYTHON_BIN"); then
  echo "Error: Python interpreter not found. Activate your environment or set PYTHON_BIN." >&2
  exit 1
fi
case "$PYTHON_BIN" in
  /*) ;;
  *) PYTHON_BIN="$PWD/$PYTHON_BIN" ;;
esac
if [[ ! -f "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Error: PYTHON_BIN must point to an executable file: $PYTHON_BIN" >&2
  exit 1
fi

# Relative external paths are resolved from the caller's cwd before changing it.
if [[ -z "${TRACEPICKER_ROOT:-}" ]]; then
  echo "Error: Set TRACEPICKER_ROOT to the external TracePicker repository." >&2
  exit 1
fi
if [[ ! -f "$TRACEPICKER_ROOT/core/TracePicker.py" ]]; then
  echo "Error: core/TracePicker.py not found under TRACEPICKER_ROOT: $TRACEPICKER_ROOT" >&2
  exit 1
fi
TRACEPICKER_ROOT=$(cd -- "$TRACEPICKER_ROOT" && pwd -P)
export TRACEPICKER_ROOT

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$REPO_ROOT"

LOG_DIR=logs/tracepicker_5seed_rerun
mkdir -p "$LOG_DIR"

start_all=$(date +%s)

ts=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$ts] >>> hip+TT primary seed=1"
/usr/bin/time -p "$PYTHON_BIN" scripts/run_tracepicker_baseline.py \
  --seed 1 \
  --datasets hipster train_ticket \
  --scope primary \
  --run-id "tracepicker_v02_hip_tt_primary_portable_seed1" \
  > "${LOG_DIR}/hip_tt_seed1.log" 2>&1
ts=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$ts] <<< hip+TT primary seed=1 done"

ts=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$ts] >>> aiops21 primary seed=1"
/usr/bin/time -p "$PYTHON_BIN" scripts/run_tracepicker_baseline.py \
  --seed 1 \
  --datasets aiops21 \
  --scope primary \
  --run-id "tracepicker_v02_aiops21_primary_portable_seed1" \
  > "${LOG_DIR}/aiops21_seed1.log" 2>&1
ts=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$ts] <<< aiops21 primary seed=1 done"

end_all=$(date +%s)
echo "[done] total elapsed seconds: $((end_all - start_all))"
