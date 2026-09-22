"""Step 14 tests: the read-only API.

The property under test is the boundary itself: the API speaks pseudonyms
only. Every response is checked for real-ID leakage, and a real-ID query
must 404 — the protocol cannot express it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from conftest import PSEUDONYM, REAL_ID, WINDOWS
from medimageforge.api import create_app


# ---------------------------------------------------------------------------
# liveness + patients
# ---------------------------------------------------------------------------

def test_health(app_client):
    client, _ = app_client
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["manifest"] is True


def test_patients_lists_pseudonyms_only(app_client):
    client, _ = app_client
    body = client.get("/patients").json()
    assert len(body) == 1
    row = body[0]
    assert row["pseudonym"] == PSEUDONYM
    assert row["brain_slices"] == 2
    assert row["masks"] == 1
    assert row["hemorrhage_positive"] is True
    # The privacy boundary: no real ID anywhere in the payload.
    assert REAL_ID not in json.dumps(body)


def test_patient_detail_and_demographics(app_client):
    client, _ = app_client
    body = client.get(f"/patients/{PSEUDONYM}").json()
    assert body["pseudonym"] == PSEUDONYM
    assert body["slices"] == {"brain": 2}
    assert body["masks"] == 1
    assert body["demographics"] is None  # no deid export in the fixture


def test_patient_demographics_come_from_deid_export(app_client, tmp_path):
    client, _ = app_client
    deid = tmp_path / "deid"
    deid.mkdir()
    pd.DataFrame(
        [[PSEUDONYM, 35.0, "Male"], ["PAT-other9999x", 60.0, "Female"]],
        columns=["patient", "Age (years)", "Gender"],
    ).to_csv(deid / "demographics_pseudonymized.csv", index=False)
    body = client.get(f"/patients/{PSEUDONYM}").json()
    assert body["demographics"]["Age (years)"] == 35.0
    assert REAL_ID not in json.dumps(body)


def test_patient_slices(app_client):
    client, _ = app_client
    body = client.get(f"/patients/{PSEUDONYM}/slices").json()
    assert [s["slice_no"] for s in body] == [1, 14]
    s14 = next(s for s in body if s["slice_no"] == 14)
    assert s14["has_mask"] is True
    assert s14["positive_labels"] == ["epidural"]
    s1 = next(s for s in body if s["slice_no"] == 1)
    assert s1["positive_labels"] == []


# ---------------------------------------------------------------------------
# slices + the privacy boundary
# ---------------------------------------------------------------------------

def test_slice_detail_echoes_pseudonym_not_real_id(app_client):
    client, _ = app_client
    body = client.get(f"/slices/{PSEUDONYM}/14").json()
    assert body["patient_id"] == PSEUDONYM
    assert "epidural" in body["hemorrhage_types"]
    assert REAL_ID not in json.dumps(body)


def test_real_id_query_cannot_be_expressed(app_client):
    """The protocol test: asking for patient '049' 404s — the API does not
    know real IDs exist."""
    client, _ = app_client
    assert client.get(f"/patients/{REAL_ID}").status_code == 404
    assert client.get(f"/slices/{REAL_ID}/14").status_code == 404
    assert client.get(f"/slices/{REAL_ID}/14/image").status_code == 404


def test_unknown_slice_is_404(app_client):
    client, _ = app_client
    assert client.get(f"/slices/{PSEUDONYM}/99").status_code == 404
    assert client.get("/patients/PAT-nope").status_code == 404


def test_slice_image_serves_curated_png(app_client):
    client, _ = app_client
    response = client.get(f"/slices/{PSEUDONYM}/14/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:4] == b"\x89PNG"


def test_slice_image_bad_window_is_404(app_client):
    client, _ = app_client
    assert client.get(
        f"/slices/{PSEUDONYM}/14/image?window=soft"
    ).status_code == 404


def test_overlay_requires_a_mask(app_client):
    client, _ = app_client
    # slice 1 exists but has no mask -> 404 with a clear reason
    response = client.get(f"/slices/{PSEUDONYM}/1/image?overlay=true")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# labels, datasets, QC, audit
# ---------------------------------------------------------------------------

def test_labels_distribution(app_client):
    client, _ = app_client
    body = client.get("/labels/distribution").json()
    epidural = next(r for r in body if r["code"] == "epidural")
    assert epidural["positives"] == 1
    assert epidural["total"] == 2


def test_datasets_lists_releases(app_client):
    client, _ = app_client
    body = client.get("/datasets").json()
    assert len(body) == 1
    assert body[0]["version"] == "v1.0"
    detail = client.get("/datasets/v1.0").json()
    assert detail["total_patients"] == 82


def test_datasets_missing_release_is_404(app_client):
    client, _ = app_client
    assert client.get("/datasets/v9.9").status_code == 404


def test_qc_report_served(app_client):
    client, _ = app_client
    body = client.get("/qc").json()
    assert body["passed"] is True
    assert body["n_warnings"] == 1
    assert body["checks"][0]["check"] == "slice-counts"


def test_qc_missing_report_is_404(tmp_path):
    config = {
        "paths": {
            "manifest_db": str(tmp_path / "manifest.db"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "deid_dir": str(tmp_path / "deid"),
            "datasets_dir": str(tmp_path / "datasets"),
            "curated_dir": str(tmp_path / "curated"),
        },
        "dataset": {"windows": WINDOWS},
    }
    client = TestClient(create_app(config))
    # no manifest -> health still answers (liveness != readiness)
    assert client.get("/health").json()["manifest"] is False
    assert client.get("/qc").status_code == 404


def test_audit_endpoints(app_client):
    client, _ = app_client
    body = client.get("/audit").json()
    assert body["total"] == 1
    assert body["records"][0]["command"] == "ingest"
    verify = client.get("/audit/verify").json()
    assert verify["ok"] is True


def test_manifest_missing_is_503_not_500(tmp_path):
    """A missing manifest is a deployment problem, not a bug — 503 with a
    remediation hint beats a stack trace."""
    config = {
        "paths": {
            "manifest_db": str(tmp_path / "manifest.db"),
            "artifacts_dir": str(tmp_path),
            "deid_dir": str(tmp_path),
            "datasets_dir": str(tmp_path),
            "curated_dir": str(tmp_path),
        },
        "dataset": {"windows": WINDOWS},
    }
    client = TestClient(create_app(config))
    response = client.get("/patients")
    assert response.status_code == 503
    assert "ingest" in response.json()["detail"]
