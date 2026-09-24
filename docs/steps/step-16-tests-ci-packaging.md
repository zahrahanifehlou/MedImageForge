# Step 16 — Tests, CI & packaging

Phase 6 opens with the step that decides whether the last fifteen steps are a
**platform** or a **folder on one laptop**. Roadmap contract: *"CI runs green
on a clean checkout; Docker image runs the CLI."*

## What we built

| artifact | purpose |
|---|---|
| `tests/conftest.py` — `needs_data` gating | tests split into "always runnable" vs "requires the real dataset" |
| `.github/workflows/ci.yml` | GitHub Actions: every push/PR installs the package on a clean checkout and runs pytest + a CLI smoke test |
| `Dockerfile` | `python:3.10-slim` image; `ENTRYPOINT ["python","-m","medimageforge"]`, `CMD ["info"]` |
| `.dockerignore` | keeps `data/`, `artifacts/`, `datasets/`, `.secrets/`, `.venv/` out of the image |

## Why this step exists

Every earlier step was verified *here*, on this machine, where `data/` happens
to exist. But `data/` and `artifacts/` are gitignored — **a fresh clone has
neither**. Before this step, a clean checkout ran the suite and got 13 failures
because `test_smoke.py`, `test_explore.py`, `test_imaging.py`, and six more
modules assume `data/Patients_CT/` and `artifacts/manifest.db` exist.

A missing dataset is a **deployment fact, not a test failure** — and conflating
the two trains you to ignore red CI, which is how real failures slip through.

## The `needs_data` marker — how it works

```python
# conftest.py
def dataset_available() -> bool:
    return (REPO_ROOT / "data" / "Patients_CT").is_dir() and \
           (REPO_ROOT / "artifacts" / "manifest.db").is_file()

def pytest_collection_modifyitems(items):
    if dataset_available():
        return
    for item in items:
        if item.get_closest_marker("needs_data"):
            item.add_marker(pytest.mark.skip(reason="..."))
```

Three pieces, each worth understanding:

1. **The marker is declared intent.** `@pytest.mark.needs_data` on a test says
   "this test exercises the real dataset." Registered in `pyproject.toml`
   under `[tool.pytest.ini_options] markers` so it's a first-class name, not a
   typo-prone string.
2. **The hook runs at collection time.** `pytest_collection_modifyitems` sees
   every collected test *before* any run — it stamps `skip` on marked items
   when `dataset_available()` is False. Skipping at collection beats failing
   at runtime: the report says *why* ("dataset absent"), not a stack trace
   from `Path.read_bytes()`.
3. **The check looks for the artifact, not just the data.** A checkout could
   have `data/` but no `manifest.db` (dataset downloaded, pipeline never run).
   Requiring *both* is the honest test of "can this test actually run."

**Design alternative rejected:** auto-skipping by probing paths inside each
test. That hides the intent; the marker makes "this test needs data" a
*declaration* you can grep for and count.

## Results — the same suite, two worlds

| checkout | result |
|---|---|
| this machine (data + artifacts present) | **274 passed, 1 skipped** |
| clean checkout simulation (`data/`, `artifacts/` moved away) | **262 passed, 13 skipped** |
| inside the Docker image (no data baked in) | **262 passed, 13 skipped** |

The identical counts between "simulated clean checkout" and "real Docker
container" is the point: the container *is* a clean checkout.

## The CI workflow

`.github/workflows/ci.yml` — on every push to `main` and every PR:

```
checkout → setup-python 3.10 (pip cache)
        → pip install -r requirements.txt --extra-index-url <pytorch cpu>
        → pip install -e .
        → pytest -q
        → python -m medimageforge info   (smoke test)
```

Why the smoke test exists separately from pytest: it proves the *installed
package* works — entry point resolves, config loads, imports are real — not
just that test files execute. A package can pass tests while being
uninstallable.

## The bug that took the most time: pip index semantics

The first Dockerfile did the "obvious" thing for the CPU-only torch wheel:

```dockerfile
RUN pip install "torch==2.13.0+cpu" --index-url https://download.pytorch.org/whl/cpu
```

It failed with `No matching distribution found for flit_core`. The chain of
causation is worth learning:

1. **`--index-url` replaces PyPI entirely.** The pytorch index only hosts
   torch-family wheels, but pip now resolves *torch's dependencies* there too.
2. The pytorch index's page for `typing-extensions` links to the PyPI wheel —
   but the link name is `typing-extensions` while the wheel's metadata says
   `typing_extensions`. **Modern pip (≥23.x, in the slim image) treats this
   normalized-name mismatch as invalid and discards the wheel.** The host's
   ancient pip 22.2.2 tolerated it — which is why it worked locally and
   failed in Docker.
3. Wheel rejected → pip falls back to the sdist → sdist needs `flit_core` to
   build → `flit_core` isn't on the pytorch index → dead end.

**The fix:** `--extra-index-url` instead of `--index-url`. PyPI stays the
primary index; the pytorch index is *additional*. torch's `+cpu` pin exists
only on the pytorch index so it resolves there; every other package resolves
from PyPI where names are consistent. Same fix applied in `ci.yml` — the
runners have modern pip and would have hit the identical wall.

**The lesson:** `--index-url` means "use *only* this index." It's for private
mirrors that carry everything. For a supplementary wheel source, it's always
`--extra-index-url`.

## The Dockerfile — decisions worth noticing

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .                       # before code: layer caching
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url ...
COPY . .
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["python", "-m", "medimageforge"]
CMD ["info"]
```

- **`requirements.txt` copied before the source tree.** Docker caches layers;
  dependencies change rarely, code changes constantly. This ordering means an
  edit to `cli.py` rebuilds in seconds instead of re-downloading torch.
- **`ENTRYPOINT` + `CMD` split.** `docker run medimageforge qc` runs `qc`;
  `docker run medimageforge` alone defaults to `info` — the safe smoke test.
- **No data baked in.** `.dockerignore` excludes `data/`, `artifacts/`,
  `datasets/`, `.secrets/` — both because medical data doesn't belong in an
  image and because it would balloon the build context by 300+ MB. Mount at
  runtime: `docker run -v "$PWD/data:/app/data" ... medimageforge ingest`.

## Verified

```
docker build -t medimageforge .          → SUCCESS (torch 2.13.0+cpu, all deps)
docker run --rm medimageforge            → info: runs, reports paths MISSING (correct — nothing mounted)
docker run --rm --entrypoint python medimageforge -m pytest tests/ -q
                                         → 262 passed, 13 skipped
local pytest -q                          → 274 passed, 1 skipped
```

## Known limitations (honest)

- CI runs the *synthetic* suite — the 13 `needs_data` tests skip there. They
  only run where the dataset exists. That's inherent to a data-heavy project;
  the marker at least makes the skipped surface explicit and countable.
- The image contains no model weights or releases — it's a code+deps image.
  Running `train` inside requires mounted data.
- No multi-stage build or non-root user yet; a production hardening pass
  would add both.
