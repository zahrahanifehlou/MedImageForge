"""Step 15: Streamlit browser over the API — the reviewer's window.

Why this exists:
    A clinician or reviewer should never need SQL or Python to ask "what's
    in this dataset?" This UI is deliberately a pure API client: it imports
    NOTHING from the pipeline internals — no sqlite, no manifest, no file
    paths. Every datum on screen arrived over HTTP via api_client, which is
    what makes Step 14's contract real: if the API can't serve it, the UI
    can't show it.

Run with:  python -m medimageforge ui
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from medimageforge.api_client import (
    ApiError,
    audit_records,
    audit_verify,
    get_dataset,
    get_patient,
    get_slice,
    health,
    label_distribution,
    list_datasets,
    list_patients,
    make_client,
    patient_slices,
    qc_status,
    slice_image,
)
from medimageforge.config import load_config


def _client() -> object:
    """Build the HTTP client once per session; the URL is user-editable so
    a reviewer can point the UI at a different deployment."""
    config = load_config()
    api = config.get("api", {})
    default = f"http://{api.get('host', '127.0.0.1')}:{api.get('port', 8000)}"
    base_url = st.sidebar.text_input("API base URL", value=default)
    return make_client(base_url)


def _guard(fn, *args, **kwargs):
    """Run one API call; turn ApiError/connection failure into st.error.

    Returns (value, None) on success or (None, message) on failure — the
    caller decides how to degrade. A UI that crashes on a 404 is worse than
    one that explains it.
    """
    try:
        return fn(*args, **kwargs), None
    except ApiError as e:
        return None, f"API {e.status_code}: {e.detail}"
    except Exception as e:  # connection refused, DNS, timeout
        return None, f"cannot reach the API: {e}"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_patients(client) -> None:
    patients, err = _guard(list_patients, client)
    if err:
        st.error(err)
        return
    frame = pd.DataFrame(patients)
    st.subheader(f"{len(frame)} patients")
    st.dataframe(frame, width="stretch", hide_index=True)

    pseudonym = st.selectbox(
        "Patient", frame["pseudonym"],
        format_func=lambda p: p,
    )
    if not pseudonym:
        return

    detail, err = _guard(get_patient, client, pseudonym)
    if err:
        st.error(err)
        return
    cols = st.columns(4)
    cols[0].metric("brain slices", detail["slices"].get("brain", 0))
    cols[1].metric("bone slices", detail["slices"].get("bone", 0))
    cols[2].metric("masks", detail["masks"])
    demo = detail.get("demographics") or {}
    if demo:
        cols[3].metric("age", demo.get("Age (years)", "—"))
        st.caption(
            "demographics (de-identified): "
            + ", ".join(f"{k}={v}" for k, v in demo.items() if v is not None)
        )

    slices, err = _guard(patient_slices, client, pseudonym)
    if err or not slices:
        st.error(err or "no annotations for this patient")
        return
    labels_col, image_col = st.columns([1, 2])
    with labels_col:
        only_positive = st.checkbox("only hemorrhage-positive slices")
        shown = [
            s for s in slices
            if not only_positive
            or any(l != "fracture" for l in s["positive_labels"])
        ]
        slice_no = st.selectbox(
            "Slice", [s["slice_no"] for s in shown],
            format_func=lambda n: _slice_label(shown, n),
        )
    if slice_no is None:
        st.info("no slices match the filter")
        return

    record, err = _guard(get_slice, client, pseudonym, slice_no)
    if err:
        st.error(err)
        return
    with labels_col:
        st.markdown("**Labels**")
        st.json(record["labels"])
        if record["mask_rel_path"]:
            st.caption(f"mask: {record['mask_rel_path']}")

    with image_col:
        overlay = st.checkbox(
            "mask overlay", value=bool(record["mask_rel_path"]),
            disabled=not record["mask_rel_path"],
        )
        window = st.radio("window", ["brain", "bone"], horizontal=True)
        png, err = _guard(
            slice_image, client, pseudonym, slice_no,
            window=window, overlay=overlay and window == "brain",
        )
        if err:
            st.error(err)
        else:
            st.image(png, caption=f"{pseudonym} · {window} · slice {slice_no}",
                     width=420)


def _slice_label(shown: list[dict], n: int) -> str:
    row = next((s for s in shown if s["slice_no"] == n), None)
    if not row:
        return str(n)
    flags = []
    if row["has_mask"]:
        flags.append("mask")
    if row["positive_labels"]:
        flags.append("+".join(row["positive_labels"]))
    return f"{n}  ({', '.join(flags)})" if flags else str(n)


def page_datasets(client) -> None:
    datasets, err = _guard(list_datasets, client)
    if err:
        st.error(err)
        return
    st.subheader(f"{len(datasets)} dataset release(s)")
    st.dataframe(pd.DataFrame(datasets), width="stretch", hide_index=True)

    for row in datasets:
        detail, err = _guard(get_dataset, client, row["version"])
        with st.expander(f"{row['version']} — {row['total_images']} images"):
            if err:
                st.error(err)
                continue
            if detail.get("splits"):
                st.markdown("**splits**")
                st.dataframe(pd.DataFrame(detail["splits"]), hide_index=True)
            if detail.get("selection"):
                st.markdown("**active-learning selection**")
                st.json(detail["selection"])
            if detail.get("dataset_card"):
                st.markdown(detail["dataset_card"])


def page_qc(client) -> None:
    body, err = _guard(qc_status, client)
    if err:
        st.error(err)
        return
    if body["passed"]:
        st.success(f"QC gate: {body['gate']} — {body['n_errors']} errors, "
                   f"{body['n_warnings']} warnings")
    else:
        st.error(f"QC gate: {body['gate']} — {body['n_errors']} errors, "
                 f"{body['n_warnings']} warnings")
    st.dataframe(
        pd.DataFrame(body["checks"]), width="stretch", hide_index=True
    )


def page_audit(client) -> None:
    verify, err = _guard(audit_verify, client)
    if err:
        st.error(err)
    elif verify["ok"]:
        st.success(f"audit chain INTACT — {verify['n_records']} records")
    else:
        st.error(f"audit chain BROKEN at seq {verify['broken_at']}: "
                 f"{verify['detail']}")

    body, err = _guard(audit_records, client, 50)
    if err:
        st.error(err)
        return
    rows = [
        {
            "#": r["seq"],
            "time": r["ts_start"][:19],
            "command": " ".join(r["argv"]),
            "actor": r["actor"],
            "status": r["status"],
            "dur(s)": r.get("duration_s"),
        }
        for r in reversed(body["records"])
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def page_labels(client) -> None:
    rows, err = _guard(label_distribution, client)
    if err:
        st.error(err)
        return
    st.subheader("Label distribution")
    frame = pd.DataFrame(rows)
    st.bar_chart(frame.set_index("display_name")["positives"])
    st.dataframe(frame, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="MedImageForge", layout="wide")
    st.title("MedImageForge")
    st.caption("read-only review UI — every datum on this page came over the "
               "Step-14 API; nothing touches the manifest directly")

    client = _client()
    status, err = _guard(health, client)
    if err:
        st.sidebar.error(f"API unreachable: {err}")
        st.stop()
    st.sidebar.success(f"API v{status['version']} · manifest {'OK' if status['manifest'] else 'MISSING'}")

    page = st.sidebar.radio(
        "page", ["Patients", "Labels", "Datasets", "QC", "Audit"]
    )
    {"Patients": page_patients, "Labels": page_labels,
     "Datasets": page_datasets, "QC": page_qc,
     "Audit": page_audit}[page](client)


main()
