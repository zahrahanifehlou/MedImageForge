"""Step 15: typed client for the read-only API — the ONLY door the UI uses.

Why a separate module:
    "The UI uses only this API" is a testable claim only if the HTTP calls
    live in one place. Every Streamlit page calls these functions; nothing
    in the UI imports manifest/privacy/sqlite3. Tests drive this client
    against fastapi.testclient.TestClient — which IS an httpx.Client over
    the app's ASGI transport — so the contract is exercised against the
    real API with no live server.

    Functions take `client` as a parameter rather than hiding a global:
    production builds httpx.Client(base_url=...), tests inject TestClient.
    Same dependency-injection pattern as create_app(config).
"""

from __future__ import annotations

import httpx


class ApiError(Exception):
    """The API answered, but not with 200.

    Carries the status code and the server's `detail` so the UI can show a
    human reason ("no annotation for patient X slice 99") instead of a
    stack trace. A missing manifest on the server is a 503 with a
    remediation hint — surfacing it beats crashing.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def make_client(base_url: str, timeout: float = 10.0) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)


def _get(client: httpx.Client, path: str, **params):
    response = client.get(path, params={k: v for k, v in params.items() if v is not None})
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text[:200])
        except ValueError:
            detail = response.text[:200]
        raise ApiError(response.status_code, detail)
    return response


# -- typed endpoints ---------------------------------------------------------

def health(client: httpx.Client) -> dict:
    return _get(client, "/health").json()


def list_patients(client: httpx.Client) -> list[dict]:
    return _get(client, "/patients").json()


def get_patient(client: httpx.Client, pseudonym: str) -> dict:
    return _get(client, f"/patients/{pseudonym}").json()


def patient_slices(client: httpx.Client, pseudonym: str) -> list[dict]:
    return _get(client, f"/patients/{pseudonym}/slices").json()


def get_slice(client: httpx.Client, pseudonym: str, slice_no: int) -> dict:
    return _get(client, f"/slices/{pseudonym}/{slice_no}").json()


def slice_image(
    client: httpx.Client, pseudonym: str, slice_no: int,
    window: str = "brain", overlay: bool = False,
) -> bytes:
    """PNG bytes — the client never sees a server path, only pixels."""
    return _get(
        client,
        f"/slices/{pseudonym}/{slice_no}/image",
        window=window,
        overlay="true" if overlay else "false",
    ).content


def label_distribution(client: httpx.Client) -> list[dict]:
    return _get(client, "/labels/distribution").json()


def list_datasets(client: httpx.Client) -> list[dict]:
    return _get(client, "/datasets").json()


def get_dataset(client: httpx.Client, version: str) -> dict:
    return _get(client, f"/datasets/{version}").json()


def qc_status(client: httpx.Client) -> dict:
    return _get(client, "/qc").json()


def audit_records(client: httpx.Client, tail: int = 15) -> dict:
    return _get(client, "/audit", tail=tail).json()


def audit_verify(client: httpx.Client) -> dict:
    return _get(client, "/audit/verify").json()
