# UniSage

This repository contains the code for **UniSage: A Unified and Diagnosis-Driven Sampling Framework for Microservice Systems**.

Our paper has been accepted at **ICSE 2027** (IEEE/ACM International Conference on Software Engineering).

```
UniSage is a unified framework for trace and log sampling that reverses this workflow into a post-analysis-aware paradigm.
UniSage first performs lightweight anomaly detection and root cause analysis on the full stream of observability data, then selectively retains a minimal yet informative subset for long-term storage or inspection. 
Our approach emphasizes both anomaly relevance and data diversity, ensuring retained samples are informative for diagnosis while avoiding redundancy.
Furthermore, the sampling process is interpretable, providing clear explanations for data preservation. 
```

## Data Directory Structure

There are three primary datasets you can work with:

1. ​**Hipster (Online Boutique) Dataset**​: Located in the `data/hipster/batches` directory.
2. ​**Train Ticket Dataset**​: Located in the `data/train_ticket/batches` directory.
3. ​**AIOps21 Dataset**​: Real-world observability data with fault injection cases.

### Dataset Download

All three datasets are kept out of git (too large). Download them from the shared Google
Drive folder and extract under `data/`:

**Google Drive:** https://drive.google.com/drive/folders/1vm4tDeXvVgHNKiMr0TT9SeLd3VTfqWAx?usp=drive_link

The final layout should be `data/hipster/batches/`, `data/train_ticket/batches/`, and
`data/aiops21/all/`.

You can modify the dataset directory in the `UniSage.py` file by updating the `root_base_dir` variable to point to your desired dataset.

### Example

To use the `hipster` dataset's first batch, you would set the following in `UniSage.py`:

```python
root_base_dir = Path(r"data/hipster/batches/batch1")
```

Alternatively, for the `train_ticket` dataset's first batch:

```python
root_base_dir = Path(r"data/train_ticket/batches/batch1")
```

Or for the `aiops21` dataset with all cases:

```python
root_base_dir = Path(r"data/aiops21/all")
```

## Running the Code

To run the project, simply execute the following command:

```bash
python UniSage.py
```

This will execute the code and apply UniSage algorithm to the selected dataset.

## Experiment Utilities

Command-line runners, evaluators, protocol tools, rerun helpers, and result
aggregators live in `scripts/`. Run them from the repository root, for example:

```bash
python scripts/run_sampling_comparison.py --help
```

Baseline-specific implementations and UniSage integration adapters live in the
existing `baselines/<method>/` packages. Shared algorithm entry points remain in
the repository root (`UniSage.py`, `sampling.py`, and `utils.py`). See
[`scripts/README.md`](scripts/README.md) for the script groups.

The external RCA baselines used in our evaluation are **MicroRCA**, **Nezha**,
and **TraceRCA**.

## Directory Overview

```text
UniSage/
├── AD/                     # Anomaly detection
│   ├── abstract/           # Detector interfaces
│   ├── log/                # Log anomaly detection
│   ├── metric/             # Metric anomaly detection
│   ├── trace/              # Trace anomaly detection
│   └── utils/              # Shared detector utilities
├── RCL/                    # Root cause localization
│   ├── log_rcl.py
│   ├── metric_rcl.py
│   └── trace_rcl.py
├── baselines/              # Baseline implementations and adapters
├── scripts/                # Experiment runners, evaluators, and aggregators
├── data/                   # Datasets downloaded separately; excluded from git
│   ├── hipster/batches/
│   ├── train_ticket/batches/
│   └── aiops21/all/
├── UniSage.py              # Main pipeline entry point
├── sampling.py             # Sampling strategies and retention
├── utils.py                # Shared pipeline utilities
├── environment.yml         # Conda environment
└── requirements.txt        # Python dependencies
```

## Requirements

Make sure to install the necessary Python dependencies before running the project. You can do so by running:

```bash
pip install -r requirements.txt
```

## Citation

If you use UniSage in your research, please cite our paper:

```bibtex
@inproceedings{UniSage,
  title = {UniSage: A Unified and Diagnosis-Driven Sampling Framework for Microservice Systems},
  author = {Zhu, Zhouruixing and Jiang, Zhihan and Yang, Tianyi and He, Pinjia},
  booktitle = {Proceedings of the 2027 IEEE/ACM International Conference on Software Engineering},
  year = {2027},
}
```
