"""Step 14: read-only HTTP API over the manifest and artifacts.

Why this module exists:
    Until now, everything that wanted data touched the SQLite manifest
    directly — fine for a single developer, wrong for a platform. A service
    boundary forces clean access patterns and is the future home of access
    control (Step 17/18): the API is the one place that decides what a
    client may see.

    THE central design rule — the API speaks PSEUDONYMS only.
    The manifest sits inside the controlled zone and keys everything by real
    patient ID. If clients could pass "049" they would become PHI consumers
    and Step 6's boundary would leak through a new door. So every endpoint
    takes and returns pseudonyms; translation happens server-side via the
    patients table. A client literally cannot express a real-ID query.

    Read-only on purpose: there are no POST/PUT endpoints yet. Annotation
    writes will arrive with the reviewer workflow, and they will need
    authentication first — a write API without auth is an incident, not a
    feature.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from medimageforge import __version__
from medimageforge.annotations import get_slice_annotation, label_distribution
from medimageforge.config import data_path
from medimageforge.data import load_pseudonym_map
from medimageforge.manifest import connect


def _pseudonymize_path(path_str: str, real: str, pseudonym: str) -> str:
    """Replace the real patient ID where it appears as a PATH SEGMENT.

    Provenance fields like mask_rel_path carry the real ID inside a larger
    string ('Patients_CT/049/brain/14.png'). Field-level translation misses
    it — the same leak family as Step 9's index.csv containing real IDs.
    Segments must match exactly so '1049' or '0492' are untouched.
    """
    return "/".join(
        pseudonym if segment == real else segment
        for segment in path_str.split("/")
    )


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


def create_app(config: dict) -> FastAPI:
    """Build the API around a resolved config — a factory, not a global,
    so tests can point a whole app at a tmp_path fixture."""
    app = FastAPI(
        title="MedImageForge API",
        version=__version__,
        description=(
            "Read-only access to the MedImageForge manifest, curated zone, "
            "dataset releases, QC status, and audit log. All patient "
            "references are pseudonyms — real IDs never cross this boundary."
        ),
    )
    windows = set(config["dataset"]["windows"])

    def db_path() -> Path:
        p = data_path(config, "manifest_db")
        if not p.is_file():
            raise _unavailable("manifest.db missing — run `ingest` first")
        return p

    def pseudonym_to_real() -> dict[str, str]:
        try:
            return load_pseudonym_map(db_path())
        except RuntimeError as e:
            raise _unavailable(str(e)) from e

    def resolve(pseudonym: str) -> str:
        real = pseudonym_to_real().get(pseudonym)
        if real is None:
            raise _not_found(f"unknown patient pseudonym '{pseudonym}'")
        return real

    # -- liveness ----------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "manifest": data_path(config, "manifest_db").is_file(),
        }

    # -- patients ----------------------------------------------------------
    @app.get("/patients")
    def list_patients() -> list[dict]:
        conn = connect(db_path())
        try:
            rows = conn.execute(
                """
                SELECT p.pseudonym,
                       SUM(CASE WHEN f.window = 'brain' AND f.kind = 'slice'
                                THEN 1 ELSE 0 END) AS brain_slices,
                       SUM(CASE WHEN f.window = 'bone' AND f.kind = 'slice'
                                THEN 1 ELSE 0 END) AS bone_slices,
                       SUM(CASE WHEN f.kind = 'mask' THEN 1 ELSE 0 END) AS masks
                FROM patients p
                LEFT JOIN files f ON f.patient_id = p.patient_id
                GROUP BY p.patient_id ORDER BY p.pseudonym
                """
            ).fetchall()
            positive = {
                r[0]
                for r in conn.execute(
                    """
                    SELECT DISTINCT sa.patient_id
                    FROM slice_annotations sa
                    JOIN slice_labels sl ON sl.annotation_id = sa.id
                    JOIN label_taxonomy lt ON lt.code = sl.label_code
                    WHERE sl.value = 1 AND lt.category = 'hemorrhage_type'
                    """
                )
            }
            patient_ids = {
                row["pseudonym"]: row["patient_id"]
                for row in conn.execute(
                    "SELECT patient_id, pseudonym FROM patients"
                )
            }
        finally:
            conn.close()
        return [
            {
                "pseudonym": row["pseudonym"],
                "brain_slices": row["brain_slices"],
                "bone_slices": row["bone_slices"],
                "masks": row["masks"],
                "hemorrhage_positive": patient_ids.get(row["pseudonym"]) in positive,
            }
            for row in rows
        ]

    @app.get("/patients/{pseudonym}")
    def patient_detail(pseudonym: str) -> dict:
        real = resolve(pseudonym)
        conn = connect(db_path())
        try:
            counts = conn.execute(
                """
                SELECT window, COUNT(*) AS n
                FROM files WHERE patient_id = ? AND kind = 'slice'
                GROUP BY window
                """,
                (real,),
            ).fetchall()
            masks = conn.execute(
                "SELECT COUNT(*) FROM files WHERE patient_id = ? AND kind = 'mask'",
                (real,),
            ).fetchone()[0]
        finally:
            conn.close()

        # Demographics come from the DE-IDENTIFIED export — the API never
        # opens the raw CSV with real patient numbers.
        demographics = None
        deid = data_path(config, "deid_dir") / "demographics_pseudonymized.csv"
        if deid.is_file():
            frame = pd.read_csv(deid)
            match = frame[frame["patient"] == pseudonym]
            if not match.empty:
                demographics = {
                    k: (None if pd.isna(v) else v)
                    for k, v in match.iloc[0].items()
                    if k != "patient"
                }

        return {
            "pseudonym": pseudonym,
            "slices": {r["window"]: r["n"] for r in counts},
            "masks": masks,
            "demographics": demographics,
        }

    @app.get("/patients/{pseudonym}/slices")
    def patient_slices(pseudonym: str) -> list[dict]:
        real = resolve(pseudonym)
        conn = connect(db_path())
        try:
            rows = conn.execute(
                """
                SELECT sa.slice_no, sa.mask_rel_path,
                       GROUP_CONCAT(CASE WHEN sl.value = 1 THEN sl.label_code END)
                           AS positives
                FROM slice_annotations sa
                LEFT JOIN slice_labels sl ON sl.annotation_id = sa.id
                WHERE sa.patient_id = ?
                GROUP BY sa.slice_no ORDER BY sa.slice_no
                """,
                (real,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "slice_no": r["slice_no"],
                "has_mask": r["mask_rel_path"] is not None,
                "positive_labels": sorted(r["positives"].split(","))
                if r["positives"] else [],
            }
            for r in rows
        ]

    # -- one slice ---------------------------------------------------------
    @app.get("/slices/{pseudonym}/{slice_no}")
    def slice_detail(pseudonym: str, slice_no: int) -> dict:
        real = resolve(pseudonym)
        records = get_slice_annotation(db_path(), real, slice_no)
        if not records:
            raise _not_found(
                f"no annotation for patient {pseudonym} slice {slice_no}"
            )
        record = records[0]
        # Never echo the real ID back — including inside paths, where it
        # hides as a segment (the Step 9 index.csv leak family).
        record["patient_id"] = pseudonym
        if record.get("mask_rel_path"):
            record["mask_rel_path"] = _pseudonymize_path(
                record["mask_rel_path"], real, pseudonym
            )
        return record

    @app.get("/slices/{pseudonym}/{slice_no}/image")
    def slice_image(
        pseudonym: str, slice_no: int,
        window: str = Query(default="brain"),
        overlay: bool = Query(default=False),
    ) -> Response:
        if window not in windows:
            raise _not_found(f"unknown window '{window}' — expected {sorted(windows)}")
        real = resolve(pseudonym)
        path = (
            data_path(config, "curated_dir") / real / window / f"{slice_no}.png"
        )
        if not path.is_file():
            raise _not_found(
                f"no curated {window} image for {pseudonym} slice {slice_no}"
            )
        if not overlay:
            return FileResponse(path)

        # Overlay = brain window + curated binary mask, rendered on demand.
        if window != "brain":
            raise _not_found("mask overlays exist for the brain window only")
        from medimageforge.imaging import load_gray, overlay_mask

        records = get_slice_annotation(db_path(), real, slice_no)
        mask_rel = records[0]["mask_rel_path"] if records else None
        if not mask_rel:
            raise _not_found(f"no mask for {pseudonym} slice {slice_no}")
        mask_path = data_path(config, "curated_dir") / mask_rel
        if not mask_path.is_file():
            raise _not_found("mask recorded but missing from the curated zone")

        from PIL import Image
        import numpy as np

        brain = np.array(Image.open(path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        buffer = io.BytesIO()
        Image.fromarray(overlay_mask(brain, mask)).save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    # -- labels ------------------------------------------------------------
    @app.get("/labels/distribution")
    def labels() -> list[dict]:
        try:
            return label_distribution(db_path())
        except Exception as e:  # label tables absent -> explain, not 500
            raise _unavailable(
                f"label store unavailable — run `load-labels` ({e})"
            ) from e

    # -- dataset releases --------------------------------------------------
    @app.get("/datasets")
    def datasets() -> list[dict]:
        root = data_path(config, "datasets_dir")
        out = []
        for meta in sorted(root.glob("*/metadata.json")):
            data = json.loads(meta.read_text())
            out.append(
                {
                    "version": data["version"],
                    "created_at": data.get("created_at"),
                    "total_patients": data.get("total_patients"),
                    "total_images": data.get("total_images"),
                    "derived_from": data.get("derived_from"),
                }
            )
        return out

    @app.get("/datasets/{version}")
    def dataset_detail(version: str) -> dict:
        release = data_path(config, "datasets_dir") / version
        meta = release / "metadata.json"
        if not meta.is_file():
            raise _not_found(f"no release '{version}'")
        data = json.loads(meta.read_text())
        splits = release / "splits.csv"
        if splits.is_file():
            data["splits"] = pd.read_csv(splits).to_dict("records")
        card = release / "dataset_card.md"
        if card.is_file():
            data["dataset_card"] = card.read_text()
        return data

    # -- QC ----------------------------------------------------------------
    @app.get("/qc")
    def qc() -> dict:
        report = data_path(config, "artifacts_dir") / "qc_report.json"
        if not report.is_file():
            raise _not_found("no QC report — run `python -m medimageforge qc`")
        data = json.loads(report.read_text())
        return {
            # the report writes `gate: PASS|FAIL`; `passed` keeps a bool API
            "passed": data.get("gate", "").upper() == "PASS",
            "gate": data.get("gate"),
            "n_errors": data.get("n_errors"),
            "n_warnings": data.get("n_warnings"),
            "checks": [
                {"check": c["check"], "result": c["result"], "detail": c["detail"]}
                for c in data.get("checks", [])
            ],
        }

    # -- audit (Step 13) ---------------------------------------------------
    @app.get("/audit")
    def audit(tail: int = Query(default=15, ge=1, le=200)) -> dict:
        from medimageforge.audit import audit_log_path, read_log

        records = read_log(audit_log_path(config))
        return {
            "total": len(records),
            "records": records[-tail:],
        }

    @app.get("/audit/verify")
    def audit_verify() -> dict:
        from medimageforge.audit import audit_log_path, verify_log

        return verify_log(audit_log_path(config))

    return app
