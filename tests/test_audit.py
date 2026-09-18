"""Step 13 tests: audit log, hash chain, change detection, lineage tracing.

The two properties under test are the ones the step exists for:
  1. APPEND-ONLY is enforced by tamper-EVIDENCE — editing, deleting, or
     reordering history must be caught by verify_log, not just by policy.
  2. PRODUCER attribution must reflect what a run actually changed, not
     what it was allowed to touch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medimageforge.audit import (
    AUDIT_IO,
    GENESIS_HASH,
    audit_run,
    config_sha256,
    fingerprint,
    find_producer,
    read_log,
    render_log,
    render_trace,
    resolve_spec,
    trace_artifact,
    verify_log,
)


def _config(tmp_path: Path) -> dict:
    """Minimal config pointing every path the specs use into tmp_path.

    data_path() joins keys to PROJECT_ROOT, so we use absolute-looking
    paths via a trick: the audit helpers are tested through explicit paths,
    and resolve_spec is tested separately with a fake config.
    """
    return {"paths": {"artifacts_dir": "artifacts"}}


def _run(log_path: Path, command: str, tmp_path: Path,
         inputs=(), outputs=(), body=None, config=None) -> dict:
    """Drive one audit_run with a tiny 'command body'."""
    cfg = config or _config(tmp_path)
    with audit_run(
        log_path, command=command, argv=[command], config=cfg,
        inputs=list(inputs), outputs=list(outputs),
    ) as rec:
        if body:
            body()
        rec["exit_code"] = 0
    return read_log(log_path)[-1]


# ---------------------------------------------------------------------------
# The log itself: append-only JSONL with a hash chain
# ---------------------------------------------------------------------------

def test_records_append_with_monotonic_seq(tmp_path):
    log = tmp_path / "audit.log"
    _run(log, "first", tmp_path)
    _run(log, "second", tmp_path)
    _run(log, "third", tmp_path)
    records = read_log(log)
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert [r["command"] for r in records] == ["first", "second", "third"]


def test_genesis_record_chains_from_zero_hash(tmp_path):
    log = tmp_path / "audit.log"
    rec = _run(log, "first", tmp_path)
    assert rec["prev_hash"] == GENESIS_HASH
    assert len(rec["record_hash"]) == 64


def test_each_record_links_to_its_predecessor(tmp_path):
    log = tmp_path / "audit.log"
    _run(log, "a", tmp_path)
    _run(log, "b", tmp_path)
    r1, r2 = read_log(log)
    assert r2["prev_hash"] == r1["record_hash"]


def test_record_carries_execution_context(tmp_path):
    log = tmp_path / "audit.log"
    rec = _run(log, "cmd --flag", tmp_path)
    assert rec["argv"] == ["cmd --flag"]
    assert "@" in rec["actor"]
    assert rec["ts_start"] and rec["ts_end"]
    assert rec["duration_s"] >= 0
    assert rec["code_version"]
    assert rec["config_sha256"]
    assert rec["status"] == "ok"
    assert rec["exit_code"] == 0


# ---------------------------------------------------------------------------
# Tamper-evidence: the point of the hash chain
# ---------------------------------------------------------------------------

def _tamper_line(log_path: Path, line_no: int, mutate) -> None:
    lines = log_path.read_text().splitlines(keepends=True)
    rec = json.loads(lines[line_no])
    lines[line_no] = json.dumps(mutate(rec)) + "\n"
    log_path.write_text("".join(lines))


def test_editing_a_record_is_detected(tmp_path):
    log = tmp_path / "audit.log"
    _run(log, "a", tmp_path)
    _run(log, "b", tmp_path)
    _tamper_line(log, 0, lambda r: {**r, "status": "failed"})
    result = verify_log(log)
    assert not result["ok"]
    assert result["broken_at"] == 1
    assert "edited" in result["detail"]


def test_deleting_a_middle_record_is_detected(tmp_path):
    log = tmp_path / "audit.log"
    for _ in range(3):
        _run(log, "x", tmp_path)
    lines = log.read_text().splitlines(keepends=True)
    log.write_text("".join([lines[0], lines[2]]))
    result = verify_log(log)
    assert not result["ok"]
    # seq 3 now sits at position 2 — sequence gap
    assert result["broken_at"] == 2


def test_reordering_records_is_detected(tmp_path):
    log = tmp_path / "audit.log"
    _run(log, "a", tmp_path)
    _run(log, "b", tmp_path)
    lines = log.read_text().splitlines(keepends=True)
    log.write_text("".join([lines[1], lines[0]]))
    assert not verify_log(log)["ok"]


def test_corrupt_json_line_fails_loudly(tmp_path):
    log = tmp_path / "audit.log"
    _run(log, "a", tmp_path)
    log.write_text(log.read_text() + "{not json\n")
    with pytest.raises(ValueError, match="corrupt at line 2"):
        read_log(log)


def test_verify_empty_log_is_ok(tmp_path):
    assert verify_log(tmp_path / "audit.log")["ok"]


# ---------------------------------------------------------------------------
# Status: failures and crashes are history too
# ---------------------------------------------------------------------------

def test_nonzero_exit_marks_failed(tmp_path):
    log = tmp_path / "audit.log"
    with audit_run(log, command="qc", argv=["qc"], config=_config(tmp_path)) as rec:
        rec["exit_code"] = 1
    assert read_log(log)[-1]["status"] == "failed"


def test_exception_is_recorded_and_reraised(tmp_path):
    log = tmp_path / "audit.log"
    with pytest.raises(RuntimeError, match="boom"):
        with audit_run(log, command="train", argv=["train"],
                       config=_config(tmp_path)):
            raise RuntimeError("boom")
    rec = read_log(log)[-1]
    assert rec["status"] == "error"
    assert "boom" in rec["error"]
    assert rec["record_hash"]  # still sealed into the chain


# ---------------------------------------------------------------------------
# Fingerprints and change detection
# ---------------------------------------------------------------------------

def test_input_fingerprint_is_the_prerun_state(tmp_path):
    """Inputs are hashed BEFORE the command runs — the bytes it read."""
    log = tmp_path / "audit.log"
    src = tmp_path / "input.csv"
    src.write_text("original")
    rec = _run(log, "cmd", tmp_path, inputs=[src],
               body=lambda: src.write_text("mutated during run"))
    assert rec["inputs"][0]["sha256"] != _sha_of(src.read_bytes())
    import hashlib
    assert rec["inputs"][0]["sha256"] == hashlib.sha256(b"original").hexdigest()


def _sha_of(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def test_dir_output_reports_created_children(tmp_path):
    log = tmp_path / "audit.log"
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.txt").write_text("x")
    rec = _run(log, "produce", tmp_path, outputs=[out],
               body=lambda: (out / "run-1" ).mkdir())
    entry = rec["outputs"][0]
    assert entry["kind"] == "dir"
    assert entry["created"] == ["run-1"]
    assert entry["created_count"] == 1
    assert entry["changed"] is True


def test_unchanged_dir_output_marks_not_changed(tmp_path):
    """A read-only run that merely declared a dir output must not look like
    a producer — this is what keeps `release --verify` out of lineage."""
    log = tmp_path / "audit.log"
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("x")
    rec = _run(log, "verify", tmp_path, outputs=[out])
    entry = rec["outputs"][0]
    assert entry["created_count"] == 0
    assert entry["changed"] is False


def test_modified_and_deleted_children_counted(tmp_path):
    import os
    import time
    log = tmp_path / "audit.log"
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep.txt"
    keep.write_text("v1")
    (out / "gone.txt").write_text("bye")

    def body():
        gone = out / "gone.txt"
        gone.unlink()
        time.sleep(0.01)  # mtime_ns granularity guard
        os.utime(keep, ns=(keep.stat().st_atime_ns, keep.stat().st_mtime_ns + 1))

    rec = _run(log, "mutate", tmp_path, outputs=[out], body=body)
    entry = rec["outputs"][0]
    assert entry["deleted_count"] == 1
    assert entry["modified_count"] == 1
    assert entry["changed"] is True


def test_file_output_flags_created_and_changed(tmp_path):
    log = tmp_path / "audit.log"
    out = tmp_path / "report.json"
    rec = _run(log, "write", tmp_path, outputs=[out],
               body=lambda: out.write_text("{}"))
    entry = rec["outputs"][0]
    assert entry["created"] is True
    assert entry["changed"] is True
    # Second run that rewrites identical bytes did NOT produce new content.
    rec2 = _run(log, "write", tmp_path, outputs=[out],
                body=lambda: out.write_text("{}"))
    assert rec2["outputs"][0]["created"] is False
    assert rec2["outputs"][0]["changed"] is False


def test_missing_paths_recorded_as_missing(tmp_path):
    log = tmp_path / "audit.log"
    rec = _run(log, "x", tmp_path,
               inputs=[tmp_path / "nope"], outputs=[tmp_path / "alsono"])
    assert rec["inputs"][0]["kind"] == "missing"
    assert rec["outputs"][0]["kind"] == "missing"


def test_config_hash_changes_with_config(tmp_path):
    cfg = _config(tmp_path)
    assert config_sha256(cfg) != config_sha256({**cfg, "other": 1})
    # Same content, different key order -> same hash (canonical JSON)
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert config_sha256(a) == config_sha256(b)


# ---------------------------------------------------------------------------
# Producer attribution and tracing
# ---------------------------------------------------------------------------

def test_find_producer_prefers_exact_created_match(tmp_path):
    """Two runs declare datasets/ as output; only the one that CREATED
    v1.0/ inside it is the producer of datasets/v1.0."""
    log = tmp_path / "audit.log"
    ds = tmp_path / "datasets"
    ds.mkdir()
    _run(log, "release", tmp_path, outputs=[ds],
         body=lambda: (ds / "v1.0").mkdir())
    _run(log, "active-learning", tmp_path, outputs=[ds],
         body=lambda: (ds / "v1.1").mkdir())
    records = read_log(log)
    producer = find_producer(records, ds / "v1.0")
    assert producer["command"] == "release"
    producer = find_producer(records, ds / "v1.1")
    assert producer["command"] == "active-learning"


def test_read_only_run_is_never_a_producer(tmp_path):
    """`release --verify` declares datasets/ as an output but changes
    nothing — it must not appear as the producer of anything under it."""
    log = tmp_path / "audit.log"
    ds = tmp_path / "datasets"
    ds.mkdir()
    _run(log, "release", tmp_path, outputs=[ds],
         body=lambda: (ds / "v1.0").mkdir())
    _run(log, "release --verify", tmp_path, outputs=[ds])
    records = read_log(log)
    producer = find_producer(records, ds / "v1.0")
    assert producer["command"] == "release"


def test_before_seq_pins_producer_earlier_than_consumer(tmp_path):
    """manifest.db is mutated by run 1 and run 3. A consumer at run 2 must
    trace to run 1, not run 3 — otherwise lineage flows BACKWARD in time."""
    log = tmp_path / "audit.log"
    db = tmp_path / "manifest.db"
    _run(log, "ingest", tmp_path, outputs=[db],
         body=lambda: db.write_text("v1"))
    _run(log, "curate", tmp_path, inputs=[db], outputs=[db],
         body=lambda: db.write_text("v2"))
    _run(log, "load-labels", tmp_path, inputs=[db], outputs=[db],
         body=lambda: db.write_text("v3"))
    records = read_log(log)
    assert find_producer(records, db)["command"] == "load-labels"
    assert find_producer(records, db, before_seq=3)["command"] == "curate"
    assert find_producer(records, db, before_seq=2)["command"] == "ingest"


def test_trace_walks_back_to_raw_root(tmp_path):
    log = tmp_path / "audit.log"
    raw = tmp_path / "data"
    db = tmp_path / "manifest.db"
    out = tmp_path / "out"
    raw.mkdir()
    (raw / "file.csv").write_text("x")
    _run(log, "ingest", tmp_path, inputs=[raw], outputs=[db],
         body=lambda: db.write_text("db"))
    def build():
        out.mkdir(exist_ok=True)
        (out / "model.bin").write_text("m")
    _run(log, "build", tmp_path, inputs=[db], outputs=[out], body=build)
    records = read_log(log)
    tree = trace_artifact(records, out / "model.bin")
    assert tree["producer"]["command"] == "build"
    db_node = tree["inputs"][0]
    assert db_node["producer"]["command"] == "ingest"
    raw_node = db_node["inputs"][0]
    assert raw_node["producer"] is None
    assert "no producing run" in raw_node["note"]


def test_trace_unproduced_artifact_is_honest(tmp_path):
    """An artifact made before the log existed gets 'no producing run' —
    not a guess."""
    records = read_log(tmp_path / "audit.log")
    tree = trace_artifact(records, tmp_path / "ancient.bin")
    assert tree["producer"] is None
    assert "no producing run" in tree["note"]


def test_render_trace_smoke(tmp_path):
    log = tmp_path / "audit.log"
    db = tmp_path / "manifest.db"
    _run(log, "ingest", tmp_path, outputs=[db],
         body=lambda: db.write_text("db"))
    tree = trace_artifact(read_log(log), db)
    text = render_trace(tree)
    assert "ingest" in text and "produced by" in text


def test_render_log_smoke(tmp_path):
    log = tmp_path / "audit.log"
    _run(log, "a", tmp_path)
    _run(log, "b", tmp_path)
    text = render_log(read_log(log), tail=1)
    assert "showing last 1 of 2" in text
    assert " b " in text or "b " in text.splitlines()[-1]


# ---------------------------------------------------------------------------
# Specs and config resolution
# ---------------------------------------------------------------------------

def test_every_command_handler_has_a_spec():
    expected = {
        "info", "explore", "inspect", "ingest", "curate", "privacy",
        "load-labels", "show-slice", "qc", "release", "train", "evaluate",
        "active-learning", "serve",
    }
    assert set(AUDIT_IO) == expected
    # `audit` must NOT audit itself
    assert "audit" not in AUDIT_IO


def test_resolve_spec_key_and_subpath():
    config = {"paths": {"manifest_db": "artifacts/manifest.db",
                        "artifacts_dir": "artifacts"}}
    resolved = resolve_spec(config, ["manifest_db", "artifacts_dir/x.json"])
    assert resolved[0].name == "manifest.db"
    assert resolved[1].name == "x.json"
    assert resolved[1].parent.name == "artifacts"


def test_fingerprint_file_and_missing(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    fp = fingerprint(f)
    assert fp["kind"] == "file" and fp["bytes"] == 3
    assert fp["sha256"] == _sha_of(b"abc")
    assert fingerprint(tmp_path / "nope")["kind"] == "missing"
