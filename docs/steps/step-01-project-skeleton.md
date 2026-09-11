# Step 1 — Project skeleton

## What we built

An installable Python package (`src/medimageforge/`) with:

- `config.py` — loads `configs/default.yaml`, resolves all paths relative to the project root
- `logging_utils.py` — one logging setup used by every future component
- `cli.py` + `__main__.py` — the single entry point: `python -m medimageforge info`
- `tests/test_smoke.py` — verifies the foundation every later step relies on
- `data/` — the CT-ICH dataset, fetched via `kagglehub` (Kaggle mirror of the PhysioNet dataset)

## Why it exists

Without this step, every later script would hardcode paths, configure its own
logging, and be impossible to test or move between machines. The skeleton is the
contract everything else builds on.

## How it works

```text
python -m medimageforge info
        │
        ▼
__main__.py ──► cli.py:main()
                    │ 1. argparse picks the subcommand ("info")
                    │ 2. load_config() reads configs/default.yaml → dict
                    │ 3. configure_logging(level) sets the root logger once
                    ▼
                cmd_info(config)
                    data_path(config, key) → PROJECT_ROOT / config["paths"][key]
                    prints [OK]/[MISSING] per path
```

Key idea: `config.py` finds the project root via `Path(__file__).resolve().parents[2]`.
This works because `pip install -e .` (editable install) keeps `__file__`
pointing at the real source tree — so the package is importable anywhere, yet
paths still resolve to this checkout.

## Concepts learned

- **Why a package, not loose scripts:** loose scripts can only import each other
  if run from the right directory. An installed package (`pip install -e .`) is
  importable from tests, notebooks, scripts, and the CLI — always the same code.
- **Why config lives outside code:** `data_path()` never hardcodes
  `data/Patients_CT`. The same code runs on a laptop, CI, or a server by
  pointing `--config` at a different YAML file.
- **Why structured logging instead of `print`:** every module gets a named
  logger (`get_logger(__name__)`), so a pipeline log shows *which component*
  spoke and *when* — the seed of the audit trail we'll build in Step 13.
- **Why `python -m medimageforge`:** one front door. Config is loaded and
  logging configured in exactly one place, no matter which subcommand runs.

## Data acquisition (this project's deviation)

The README assumes data already in `data/`. We fetched it with:

```python
import kagglehub
path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
```

then copied the extracted folder into `data/`. The Kaggle dataset is a mirror of
the same PhysioNet CT-ICH release — it includes `Patients_CT/`,
`hemorrhage_diagnosis.csv`, `patient_demographics.csv`, and `SHA256SUMS.txt`.

## Verified

- `python -m medimageforge info` → all data paths `[OK]`, `artifacts_dir`
  `[MISSING]` (expected: nothing generated yet)
- `pytest` → 4 passed
