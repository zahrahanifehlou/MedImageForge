"""Step 15 tests: the typed API client — the ONLY door the UI uses.

These tests prove the roadmap's contract: everything the UI needs is
reachable over the API. They drive api_client against fastapi's TestClient
(itself an httpx.Client on the app's ASGI transport), so no live server is
needed — and the client can never succeed against a private path the API
doesn't expose.
"""

from __future__ import annotations

import pytest

from conftest import PSEUDONYM, REAL_ID
from medimageforge import api_client
from medimageforge.api_client import ApiError


def test_health(app_client):
    client, _ = app_client
    body = api_client.health(client)
    assert body["status"] == "ok"


def test_list_patients(app_client):
    client, _ = app_client
    patients = api_client.list_patients(client)
    assert [p["pseudonym"] for p in patients] == [PSEUDONYM]


def test_patient_detail(app_client):
    client, _ = app_client
    body = api_client.get_patient(client, PSEUDONYM)
    assert body["slices"] == {"brain": 2}
    assert body["masks"] == 1


def test_patient_slices(app_client):
    client, _ = app_client
    slices = api_client.patient_slices(client, PSEUDONYM)
    assert [s["slice_no"] for s in slices] == [1, 14]


def test_get_slice(app_client):
    client, _ = app_client
    body = api_client.get_slice(client, PSEUDONYM, 14)
    assert body["patient_id"] == PSEUDONYM
    assert "epidural" in body["hemorrhage_types"]


def test_slice_image_returns_png_bytes(app_client):
    client, _ = app_client
    data = api_client.slice_image(client, PSEUDONYM, 14)
    assert data[:4] == b"\x89PNG"


def test_label_distribution(app_client):
    client, _ = app_client
    rows = api_client.label_distribution(client)
    epidural = next(r for r in rows if r["code"] == "epidural")
    assert epidural["positives"] == 1


def test_datasets(app_client):
    client, _ = app_client
    datasets = api_client.list_datasets(client)
    assert [d["version"] for d in datasets] == ["v1.0"]
    detail = api_client.get_dataset(client, "v1.0")
    assert detail["total_patients"] == 82


def test_qc_status(app_client):
    client, _ = app_client
    body = api_client.qc_status(client)
    assert body["passed"] is True


def test_audit_endpoints(app_client):
    client, _ = app_client
    body = api_client.audit_records(client)
    assert body["total"] == 1
    verify = api_client.audit_verify(client)
    assert verify["ok"] is True


# ---------------------------------------------------------------------------
# Errors: ApiError carries status + server detail, never a bare stack trace
# ---------------------------------------------------------------------------

def test_not_found_raises_api_error_with_detail(app_client):
    client, _ = app_client
    with pytest.raises(ApiError) as err:
        api_client.get_slice(client, PSEUDONYM, 99)
    assert err.value.status_code == 404
    assert "no annotation" in err.value.detail


def test_real_id_is_not_a_valid_query(app_client):
    """Same boundary as the API tests: the client cannot ask for '049'."""
    client, _ = app_client
    with pytest.raises(ApiError) as err:
        api_client.get_patient(client, REAL_ID)
    assert err.value.status_code == 404


def test_unknown_patient_image_is_404(app_client):
    client, _ = app_client
    with pytest.raises(ApiError):
        api_client.slice_image(client, "PAT-nope", 1)


def test_overlay_param_is_sent(app_client):
    """overlay=true reaches the server (verified by its 404 for a slice
    with no mask — a 200 would mean the flag was dropped)."""
    client, _ = app_client
    with pytest.raises(ApiError) as err:
        api_client.slice_image(client, PSEUDONYM, 1, overlay=True)
    assert err.value.status_code == 404
    assert "no mask" in err.value.detail
