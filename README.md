# AeroDetect

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

AeroDetect: Benchmarking YOLO and Faster R-CNN for Aerial Object Detection

## Datasets

### Prerequisites

Add your Kaggle API credentials to a `.env` file at the project root: `KAGGLE_KEY=your_kaggle_api_key`

### Download

One or more dataset names can be passed as positional arguments; `all` downloads everything:

```bash
uv run dataset military
uv run dataset military skyfusion
uv run dataset all
```

Run `uv run dataset --help` to see all options.

### Process

`process.py` converts downloaded raw datasets into YOLO-ready layouts:

```text
<dataset_root>/
  data.yaml
  images/train|val|test
  labels/train|val|test
```

One or more dataset names can be passed as positional arguments; `all` processes everything:

```bash
uv run process military
uv run process skyfusion
uv run process military skyfusion
uv run process all
```

Run `uv run process --help` to see all options.

## Training

Training is implemented in [`aerodetect/modeling/train.py`](aerodetect/modeling/train.py) and uses:

- [Hydra](https://hydra.cc/) for configuration management
- [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking

Before training:

1. Configure Weights & Biases for this repository:

   ```bash
   uv run wandb login
   uv run wandb init
   ```

   During `wandb init`, select:
   - entity/team: `aerodetect`
   - project: `detection-yolo`

1. Make sure your dataset is already processed (for example with `uv run process military`).

Important: the training command must specify which Hydra `db` config to use via `+db=<name>`.

Examples:

```bash
uv run ./aerodetect/modeling/train.py +db=military
uv run ./aerodetect/modeling/train.py +db=skyfusion
```

Available `db` configs are in `aerodetect/modeling/conf/db/`.

## Project Organization

```tree
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         aerodetect and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── scripts            <- Utility scripts, including Slurm job launchers
│   └── yolo
│       ├── military.sh   <- Slurm job script for YOLO training (`+db=military`)
│       └── skyfusion.sh  <- Slurm job script for YOLO training (`+db=skyfusion`)
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── aerodetect   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes aerodetect a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py
    │   ├── conf                <- Hydra config
    │   │   ├── config.yaml
    │   │   └── db
    │   │       ├── military.yaml   <- Hydra training profile for military (`+db=military`)
    │   │       └── skyfusion.yaml  <- Hydra training profile for skyfusion (`+db=skyfusion`)
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Slurm Usage (Helper Notes)

This section is a practical reference for running on Slurm (PLGrid-style setup), not a replacement for your cluster's official documentation.

1. Load required modules:

    ```bash
    module load GCCcore/13.3.0
    module load Python/3.12.3
    module load CUDA/12.8.0
    ```

1. Install `uv`:

    ```bash
    pip install uv
    ```

1. Move environment/cache/datasets to `$SCRATCH` (`HOME` has limited storage):

    ```bash
    cd /path/to/aero-detect
    uv cache clean

    mkdir -p $SCRATCH/aero-detect/
    mkdir -p $SCRATCH/aero-detect/data
    mkdir -p $SCRATCH/aero-detect/.cache/uv
    mkdir -p $SCRATCH/aero-detect/.cache/kagglehub

    rm -rf data
    ln -s $SCRATCH/aero-detect/data data
    mkdir -p data/raw data/processed data/interim data/external

    export UV_CACHE_DIR="$SCRATCH/aero-detect/.cache/uv"
    export UV_PROJECT_ENVIRONMENT="$SCRATCH/aero-detect/.venv"
    export KAGGLEHUB_CACHE="$SCRATCH/aero-detect/.cache/kagglehub"
    ```

1. Example interactive Slurm session:

    ```bash
    srun -A plgzzsn2026-gpu-a100 -p plgrid-gpu-a100 -t 02:00:00 -c 4 --gres=gpu:1 --mem=16G --nodes=1 --pty bash
    ```

1. Prepare environment and data:

    ```bash
    uv sync --reinstall
    uv run dataset all
    uv run process all
    ```

1. Configure _Weights & Biases_ for this repository:

    ```bash
    uv run wandb login
    uv run wandb init
    ```

    During `wandb init`, select:
    - entity/team: `aerodetect`
    - project: `detection-yolo`

1. Run training (directly or via job scripts):

    ```bash
    uv run ./aerodetect/modeling/train.py +db=military
    sbatch scripts/yolo/military.sh
    sbatch scripts/yolo/skyfusion.sh
    ```
