"""Tests for the Step 6 privacy gate.

Two jobs here:
  1. prove each PHI detector fires on deliberately dirty data (the real
     dataset is clean, so it never exercises a failure);
  2. prove the pseudonymization is actually opaque — including a live
     brute-force attack showing why an unkeyed hash would not be.
"""

import hashlib

import pandas as pd
import pytest
from PIL import Image

from medimageforge.config import data_path, load_config
from medimageforge.manifest import connect
from medimageforge.privacy import (
    ENV_SALT,
    PATIENTS_SCHEMA,
    build_pseudonym_map,
    check_age_rule,
    export_deidentified,
    find_id_leaks,
    load_or_create_salt,
    pseudonymize,
    pseudonymize_frames,
    scan_columns,
    scan_free_text,
    scan_image_metadata,
)

SALT = b"test-salt-not-a-real-secret"


# ---------------------------------------------------------------------------
# pseudonymization
# ---------------------------------------------------------------------------

def test_pseudonym_is_deterministic_and_formatted():
    a = pseudonymize("049", SALT)
    b = pseudonymize("049", SALT)
    assert a == b                      # joins across artifacts must still work
    assert a.startswith("PAT-")
    assert len(a) == len("PAT-") + 12


def test_different_patients_get_different_pseudonyms():
    assert pseudonymize("049", SALT) != pseudonymize("050", SALT)


def test_salt_changes_the_pseudonym():
    """The salt is the key: rotate it and every token changes."""
    assert pseudonymize("049", SALT) != pseudonymize("049", b"a-different-salt")


def test_unkeyed_hash_is_brute_forceable_but_hmac_is_not():
    """WHY we use HMAC and not sha256(patient_id).

    The identifier space is tiny — 3-digit patient numbers. An attacker just
    hashes every candidate and looks up the result. This test performs that
    attack: it succeeds against a plain digest and fails against HMAC.
    """
    target = "049"
    rainbow = {hashlib.sha256(f"{i:03d}".encode()).hexdigest(): f"{i:03d}"
               for i in range(1000)}

    naive = hashlib.sha256(target.encode()).hexdigest()
    assert rainbow[naive] == target          # identity recovered instantly

    keyed = pseudonymize(target, SALT).removeprefix("PAT-")
    assert not any(h.startswith(keyed) for h in rainbow)  # attack fails


def test_salt_file_is_created_private_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_SALT, raising=False)
    salt_file = tmp_path / ".secrets" / "salt"

    first = load_or_create_salt(salt_file)
    assert salt_file.is_file()
    assert oct(salt_file.stat().st_mode)[-3:] == "600"  # owner-only
    assert load_or_create_salt(salt_file) == first      # stable across calls


def test_env_var_overrides_salt_file(tmp_path, monkeypatch):
    salt_file = tmp_path / "salt"
    salt_file.write_text("file-salt")
    monkeypatch.setenv(ENV_SALT, "env-salt")
    assert load_or_create_salt(salt_file) == b"env-salt"


# ---------------------------------------------------------------------------
# PHI detectors — each must fire on dirty data
# ---------------------------------------------------------------------------

def test_image_metadata_scan_flags_exif(tmp_path):
    path = tmp_path / "with_exif.jpg"
    im = Image.new("L", (8, 8))
    exif = im.getexif()
    exif[270] = "PatientName: Jane Doe"   # 270 = ImageDescription
    im.save(path, exif=exif)

    findings = scan_image_metadata([path])
    assert findings and findings[0]["issue"] == "exif-present"


def test_image_metadata_scan_passes_clean_image(tmp_path):
    path = tmp_path / "clean.jpg"
    Image.new("L", (8, 8)).save(path)
    assert scan_image_metadata([path]) == []


def test_forbidden_column_names_are_flagged():
    df = pd.DataFrame(
        {"PatientName": ["x"], "MRN": [1], "DOB": ["2000-01-01"], "SliceNumber": [1]}
    )
    issues = {f["column"] for f in scan_columns(df, "test")}
    assert {"PatientName", "MRN", "DOB"} <= issues
    assert "SliceNumber" not in issues


def test_free_text_identifiers_are_flagged():
    df = pd.DataFrame(
        {
            "note": [
                "contact jane@example.com",
                "scanned on 12/03/2018",
                "call 555-123-4567 now",
                "reviewed by Dr. Smith",
                "Subdural HGE",          # clinical text only — must not flag
            ]
        }
    )
    kinds = {f["issue"] for f in scan_free_text(df, "test")}
    assert "possible-email" in kinds
    assert "possible-date" in kinds
    assert "possible-person_title" in kinds
    clean_rows = {f["row"] for f in scan_free_text(df, "test")}
    assert 4 not in clean_rows


def test_relative_time_is_not_flagged_as_a_date():
    """The real dataset's only note says 'after about two weeks of the
    accident'. A relative interval is not an identifying date."""
    df = pd.DataFrame({"Note": ["CT scan was after about two weeks of the accident"]})
    assert scan_free_text(df, "test") == []


def test_age_over_89_is_flagged():
    df = pd.DataFrame({"Age (years)": [35, 72, 91]})
    findings = check_age_rule(df, "test")
    assert len(findings) == 1
    assert findings[0]["value"] == 91


def test_ages_within_safe_harbor_pass():
    assert check_age_rule(pd.DataFrame({"Age (years)": [0.5, 35, 89]}), "test") == []


# ---------------------------------------------------------------------------
# leak detection — the check that had a false positive
# ---------------------------------------------------------------------------

def _frame(patient_ids, extra):
    frame = pd.DataFrame({"patient": [pseudonymize(p, SALT) for p in patient_ids], **extra})
    return frame, pd.Series([int(p) for p in patient_ids])


def test_coincidental_number_is_not_a_leak():
    """The false positive that made us rewrite this check.

    An age of 49 belonging to some other patient is not a disclosure of
    patient 049 — the value does not align with its own row's identity.
    """
    frame, real_ids = _frame(["049", "050", "130"], {"age": [35, 49, 60]})
    assert find_id_leaks(frame, real_ids) == []


def test_column_reproducing_the_patient_id_is_a_leak():
    """The failure the check must still catch: a column that IS the ID."""
    frame, real_ids = _frame(["049", "050", "130"], {"legacy_id": [49, 50, 130]})
    findings = find_id_leaks(frame, real_ids)
    assert any(f["issue"] == "column-reproduces-patient-id" for f in findings)
    assert findings[0]["fraction"] == 1.0


def test_zero_padded_id_is_also_caught():
    frame, real_ids = _frame(["049", "050"], {"folder": ["049", "050"]})
    assert any(
        f["issue"] == "column-reproduces-patient-id" for f in find_id_leaks(frame, real_ids)
    )


def test_malformed_pseudonym_is_flagged():
    frame = pd.DataFrame({"patient": ["049", "PAT-abc"]})
    findings = find_id_leaks(frame, pd.Series([49, 50]))
    assert any(f["issue"] == "malformed-pseudonym" for f in findings)


def test_missing_pseudonym_column_is_flagged():
    findings = find_id_leaks(pd.DataFrame({"age": [1]}), pd.Series([49]))
    assert any(f["issue"] == "pseudonym-column-missing" for f in findings)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def test_export_drops_the_real_id_column(tmp_path):
    labels = pd.DataFrame({"PatientNumber": [49, 50], "SliceNumber": [1, 2]})
    demographics = pd.DataFrame({"Patient Number": [49, 50], "Age (years)": [35, 40]})
    mapping = {"049": pseudonymize("049", SALT), "050": pseudonymize("050", SALT)}

    items = pseudonymize_frames(labels, demographics, mapping, 3)
    paths = export_deidentified(items, tmp_path / "deid")

    for path in paths:
        out = pd.read_csv(path)
        assert "patient" in out.columns
        assert "PatientNumber" not in out.columns
        assert "Patient Number" not in out.columns
        assert out["patient"].str.startswith("PAT-").all()


# ---------------------------------------------------------------------------
# real dataset
# ---------------------------------------------------------------------------

@pytest.mark.needs_data
def test_real_patients_are_all_mapped(tmp_path):
    """Runs against a COPY of the manifest — never the real one.

    The first version of this test called build_pseudonym_map() directly on
    artifacts/manifest.db with the test salt. That function persists what it
    computes, so simply running pytest overwrote the real pseudonyms and left
    the published release unable to resolve its own image paths. A test must
    not mutate production artifacts.
    """
    import shutil

    config = load_config()
    real_db = data_path(config, "manifest_db")
    if not real_db.is_file():
        pytest.skip("run `python -m medimageforge ingest` first")

    db = tmp_path / "manifest_copy.db"
    shutil.copy(real_db, db)

    mapping = build_pseudonym_map(db, SALT)
    assert len(mapping) == 82
    assert len(set(mapping.values())) == 82  # no collisions


def test_read_pseudonym_map_does_not_write(tmp_path):
    """A reader must never modify what it reads.

    Release verification used to call build_pseudonym_map(), which rebuilt and
    overwrote the mapping — so it verified its own repair and reported PASS on
    corrupted data. read_pseudonym_map() is the read-only path.
    """
    from medimageforge.privacy import read_pseudonym_map

    db = tmp_path / "m.db"
    with connect(db) as conn:
        conn.executescript(PATIENTS_SCHEMA)
        conn.execute(
            "INSERT INTO patients (patient_id, pseudonym, created_at)"
            " VALUES ('049', 'PAT-deadbeef0001', 'now')"
        )
        conn.commit()

    before = db.read_bytes()
    assert read_pseudonym_map(db) == {"049": "PAT-deadbeef0001"}
    assert db.read_bytes() == before        # byte-for-byte unchanged


def test_read_pseudonym_map_is_empty_without_the_table(tmp_path):
    from medimageforge.privacy import read_pseudonym_map

    assert read_pseudonym_map(tmp_path / "missing.db") == {}
