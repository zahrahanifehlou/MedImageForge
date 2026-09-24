"""Shared fixtures for the API-layer tests (Steps 14–15).

One synthetic platform serves both test files: test_api checks the server
side, test_api_client checks the client the UI will use — against the same
fixture, so they can never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medimageforge.annotations import load_annotations
from medimageforge.api import create_app
from medimageforge.manifest import ingest

# -- dataset gating ----------------------------------------------------------
# Tests marked needs_data exercise the REAL dataset and the artifacts the
# pipeline produced from it. On a clean checkout (CI) neither exists, and
# they must SKIP — a missing dataset is a deployment fact, not a failure.
REPO_ROOT = Path(__file__).resolve().parent.parent


def dataset_available() -> bool:
    return (REPO_ROOT / "data" / "Patients_CT").is_dir() and (
        REPO_ROOT / "artifacts" / "manifest.db"
    ).is_file()


def pytest_collection_modifyitems(items):
    if dataset_available():
        return
    skip = pytest.mark.skip(
        reason="needs_data: dataset/artifacts absent on this checkout"
    )
    for item in items:
        if item.get_closest_marker("needs_data"):
            item.add_marker(skip)

WINDOWS = ["brain", "bone"]
MASK_SUFFIX = "_HGE_Seg"
PSEUDONYM = "PAT-test0123abcd"
REAL_ID = "049"

LABEL_COLUMNS = [
    "PatientNumber", "SliceNumber", "Intraventricular", "Intraparenchymal",
    "Subarachnoid", "Epidural", "Subdural", "No_Hemorrhage", "Fracture_Yes_No",
]


@pytest.fixture
def app_client(tmp_path):
    """A whole API pointed at a tiny synthetic platform in tmp_path.

    Returns (TestClient, tmp_path). TestClient IS an httpx.Client over the
    app's ASGI transport, so api_client functions accept it directly —
    the UI's contract is tested against the real API, no live server.
    """
    # -- raw zone ---------------------------------------------------------
    data = tmp_path / "data"
    brain = data / "Patients_CT" / REAL_ID / "brain"
    brain.mkdir(parents=True)
    for name in ("1.jpg", "14.jpg", f"14{MASK_SUFFIX}.jpg"):
        (brain / name).write_bytes(name.encode())

    # -- manifest + label store -------------------------------------------
    db = tmp_path / "manifest.db"
    ingest(data, db, WINDOWS, MASK_SUFFIX)
    labels = pd.DataFrame(
        [
            [49, 1, 0, 0, 0, 0, 0, 1, 0],   # negative
            [49, 14, 0, 0, 0, 1, 0, 0, 0],  # epidural, has mask
        ],
        columns=LABEL_COLUMNS,
    )
    csv = tmp_path / "labels.csv"
    labels.to_csv(csv, index=False)
    load_annotations(db, labels, csv, 3)

    # -- pseudonym map (Step 6's table — created by privacy.py, not the
    #    manifest schema, so install it the same way) ---------------------
    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS patients (
               patient_id TEXT PRIMARY KEY,
               pseudonym  TEXT NOT NULL UNIQUE,
               created_at TEXT NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO patients (patient_id, pseudonym, created_at) "
        "VALUES (?, ?, ?)",
        (REAL_ID, PSEUDONYM, "2026-09-18T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    # -- curated zone: two real PNGs (slice 1 has an image but NO mask —
    #    that combination is what lets the overlay-param test reach the
    #    mask check instead of dying at the image check) ---------------
    curated = tmp_path / "curated"
    img_dir = curated / REAL_ID / "brain"
    img_dir.mkdir(parents=True)
    for name in ("1.png", "14.png"):
        Image.fromarray(np.full((8, 8), 128, dtype=np.uint8)).save(
            img_dir / name
        )

    # -- artifacts: a QC report + an audit log ------------------------------
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    # Same schema as qc.write_report: `gate`, not `passed` — a fixture that
    # encodes reality, not our guess about it (see the /qc endpoint fix).
    (artifacts / "qc_report.json").write_text(json.dumps({
        "gate": "PASS", "n_errors": 0, "n_warnings": 1,
        "checks": [{"check": "slice-counts", "result": "WARN",
                    "detail": "084: brain=36, bone=35",
                    "errors": [], "warnings": ["084"]}],
    }))
    from medimageforge.audit import audit_run
    with audit_run(artifacts / "audit.log", command="ingest",
                   argv=["ingest"], config={}):
        pass

    # -- a release ----------------------------------------------------------
    release = tmp_path / "datasets" / "v1.0"
    release.mkdir(parents=True)
    (release / "metadata.json").write_text(json.dumps({
        "version": "v1.0", "created_at": "2026-09-18T00:00:00+00:00",
        "total_patients": 82, "total_images": 5001,
    }))

    config = {
        "paths": {
            "manifest_db": str(db),
            "curated_dir": str(curated),
            "deid_dir": str(tmp_path / "deid"),
            "datasets_dir": str(tmp_path / "datasets"),
            "artifacts_dir": str(artifacts),
            "audit_log": str(artifacts / "audit.log"),
        },
        "dataset": {"windows": WINDOWS},
    }
    return TestClient(create_app(config)), tmp_path
