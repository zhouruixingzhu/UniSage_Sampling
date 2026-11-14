# Experiment utilities

Run these command-line tools from the `UniSage/` repository root as
`python scripts/<tool>.py ...`. Each Python entry point adds the repository root
to `sys.path`, so imports of the core pipeline and baseline packages continue to
work from that invocation.

- `run_*` and `rerun_*`: sampling baselines and targeted experiment runs.
- `eval_*`, `posthoc_*`, and `rca_benefit_curve.py`:
  evaluation and analysis.
- `sampling_labels.py`: shared label definitions, scope rules, and audit CLI.
- `build_trace_manifest.py`: generate frozen trace-label manifests.
- `evaluate_trace_sampling.py`: evaluate trace-sampling outputs against manifests.
- `log_protocol_*`: log-manifest generation and evaluation.
- `aggregate_*` and `build_linked_log_fairness_table.py`: result aggregation.
- `smoke_test.py`: a local end-to-end smoke runner.

Common entry points:

| Script | Purpose |
|---|---|
| `run_sampling_comparison.py` | Run UniSage and evaluate trace retention |
| `run_gleaner_baseline.py`, `run_sieve_baseline.py`, `run_sifter_baseline.py` | Run trace-sampling baselines |
| `run_tracepicker_baseline.py`, `run_trastrainer_baseline.py` | Run trace-sampling baselines |
| `run_uniform_random.py` | Run the seeded Uniform baseline |
| `posthoc_trace_composition.py` | Evaluate composition variants |
| `aggregate_unisage_runs.py` | Aggregate UniSage repetitions |

Protocol versions and archived run-directory names are retained for provenance;
they are independent of the entry-point filenames.

Baseline implementations and their UniSage adapters are grouped under the
existing `baselines/<method>/` packages. Core runtime files (`UniSage.py`,
`sampling.py`, and `utils.py`) remain at the repository root.

The final external RCA baselines are **MicroRCA**, **Nezha**, and **TraceRCA**.
Their sampled-trace evaluators are `eval_microrca_rca_on_sampled_traces.py`,
`eval_nezha_rca_on_sampled_traces.py`, and
`eval_tracerca_rca_on_sampled_traces.py`, respectively. The shared
`eval_fair_rca_on_sampled_traces.py` and `eval_internal_rca.py` evaluate
UniSage's own RCL.

## Environment and paths

Activate the environment from `environment.yml` before running these tools.
The shell wrappers locate the repository themselves, so they can be invoked
from any working directory. They use the active `python`, or the executable
specified by `PYTHON_BIN`; they do not activate a machine-specific Conda path.

```bash
export TRACEPICKER_ROOT=/path/to/TracePicker
bash scripts/run_tracepicker_seed1_rerun.sh
bash scripts/run_tracepicker_5seed_rerun.sh
bash scripts/run_trastrainer_5seed_rerun.sh
```

TracePicker requires its external checkout. The first wrapper runs seed 1;
the two `5seed` wrappers run seeds 2–5 to supplement seed 1. The TraStrainer
wrapper covers Hipster and Train Ticket; use its Python runner directly for
AIOps21. All three wrappers preserve the runners' default budgets.

Nezha and TraceRCA use separate environments described under their baseline
directories. Set `NEZHA_PYTHON` / `TRACERCA_PYTHON`, or pass
`--nezha-python` / `--python`, to select their Python executables. The adapters
can also discover the corresponding named Conda environment. Missing
interpreters produce a setup error before evaluation starts.

## Trace and log artifacts

`run_sampling_comparison.py` archives trace outputs. To reproduce log
retention, run `rerun_log_sampling_route.py` with the same configuration and
the trace run's explicit case list, then evaluate its archived log index:

```bash
export TRACE_RUN=/path/to/archived-trace-run
python scripts/rerun_log_sampling_route.py --route reproduction \
  --selected-cases "$TRACE_RUN/selected_cases.csv"
python scripts/log_protocol_eval.py \
  --sample-index results/v02_sampling_runs/reproduction/log_sample_index_clean.csv \
  --output results/log_reproduction
```

Set `TRACE_RUN` to the actual run directory. Run configurations
sequentially because the pipeline also writes intermediate outputs into shared
case directories. The optional `rerun_log_sampling_route_reuse.py` additionally
requires explicit `--system-source-route` and `--trace-source-route`; only use
it when the shared service rankings and archived sources match the intended
configuration.

## Analysis and checks

- `posthoc_trace_composition.py` includes the final composition ablation;
  retain it when reproducing that table, even though it also has exploratory modes.
- `rca_benefit_curve.py` evaluates the v0.2 analysis-degradation experiment.
- `aggregate_runtime_table.py` reads existing timing artifacts and the bundled
  seed-1 reference under `reference_data/`; it no longer requires the hub repo.
  Use `--run-root`, `--output`, and `--seed1-runtime-table` to override paths.
- `aggregate_fair_rca_outputs.py --evaluator` supports `microrca`, `nezha`,
  `tracerca`, and the internal `unisage_rcl` evaluator.
- `python scripts/smoke_test.py --help` and
  `python scripts/eval_internal_rca.py --help` describe the installation check
  and optional internal-RCL diagnostic.

The retired `coverage_eval.py`, `coverage_decomp.py`, and
`composition_robustness.py` used an older root-service-path proxy. They are
excluded from the public script set; the v0.2 protocol tools define the paper's
trace and log evaluation metrics.

Experiment outputs are not bundled. After downloading the datasets, generate
the required manifests before running samplers or evaluators:

```bash
python scripts/sampling_labels.py
python scripts/build_trace_manifest.py
python scripts/log_protocol_materialize.py
```

These commands create local outputs under the gitignored `results/` directory.
