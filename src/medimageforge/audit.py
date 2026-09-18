"""Step 13: lineage and audit log.

Why this module exists:
    Everything before this step recorded *data* provenance — file sha256 in
    the manifest, source_sha256 in curation, labels_sha256 in releases. But
    nothing recorded *executions*: who ran `release`, when, with which config
    and code version, reading which inputs, producing which outputs.

    The audit log answers that. It is an append-only JSONL file where each
    line is one pipeline invocation, and every record carries the sha256 of
    the record before it. Appending is a policy — the hash chain is what
    makes tampering with history detectable. Editing or deleting record N
    changes its hash, which breaks the `prev_hash` stored in N+1, which
    breaks every later link. Verification re-walks the chain.

    Two fingerprints per record matter more than they look:
      - inputs are hashed BEFORE the command runs — that is what it read;
      - outputs are hashed AFTER — that is what it produced.
    Hash them in the wrong order and the log describes a state that never
    existed at run time.

    For directory outputs we also snapshot the children before and after and
    record which entries were CREATED. That is what lets `trace` attribute
    datasets/v1.0 to the `release` run and datasets/v1.1 to the
    `active-learning` run, even though both declared the same parent dir.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import os
import platform
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

from medimageforge import __version__
from medimageforge.config import PROJECT_ROOT, data_path

# ---------------------------------------------------------------------------
# Declared inputs/outputs per command.
#
# Entries are either a `paths.<key>` from the config ("manifest_db") or a key
# followed by a sub-path ("artifacts_dir/qc_report.json"). The audit wrapper
# resolves them to absolute paths; a command that produces something outside
# this list still gets logged, just with less precise outputs.
#
# manifest.db appears as BOTH input and output for commands that mutate it —
# that is correct lineage: the run read manifest.db@hashA and left
# manifest.db@hashB.
# ---------------------------------------------------------------------------
AUDIT_IO: dict[str, dict[str, list[str]]] = {
    "info": {"inputs": [], "outputs": []},
    "explore": {
        "inputs": ["data_dir", "labels_csv", "demographics_csv", "checksums_file"],
        "outputs": [],
    },
    "inspect": {"inputs": ["raw_dir"], "outputs": ["artifacts_dir/inspect"]},
    "ingest": {"inputs": ["data_dir"], "outputs": ["manifest_db"]},
    "curate": {
        "inputs": ["manifest_db", "labels_csv", "raw_dir"],
        # manifest.db is also an output: the curation table is written into it.
        "outputs": ["curated_dir", "artifacts_dir/curation_report.json", "manifest_db"],
    },
    "privacy": {
        # salt_file is hashed, not copied — a sha256 of a high-entropy secret
        # is a fingerprint, not the secret. Recording it documents the
        # dependency and would reveal a salt change (which would silently
        # break pseudonym stability across runs).
        "inputs": [
            "manifest_db", "raw_dir", "labels_csv", "demographics_csv", "salt_file",
        ],
        "outputs": ["deid_dir", "artifacts_dir/privacy_report.json", "manifest_db"],
    },
    "load-labels": {
        "inputs": ["manifest_db", "labels_csv"],
        "outputs": ["manifest_db"],
    },
    "show-slice": {"inputs": ["manifest_db"], "outputs": []},
    "qc": {
        "inputs": ["manifest_db", "labels_csv", "demographics_csv"],
        "outputs": ["artifacts_dir/qc_report.json"],
    },
    "release": {
        "inputs": ["manifest_db", "labels_csv", "curated_dir", "salt_file"],
        "outputs": ["datasets_dir"],
    },
    "train": {
        "inputs": ["datasets_dir", "manifest_db", "curated_dir"],
        "outputs": ["runs_dir"],
    },
    "evaluate": {
        "inputs": ["runs_dir", "datasets_dir", "manifest_db", "curated_dir"],
        "outputs": ["eval_dir"],
    },
    "active-learning": {
        "inputs": ["datasets_dir", "manifest_db", "curated_dir", "labels_csv", "salt_file"],
        "outputs": ["artifacts_dir/active", "datasets_dir"],
    },
    # The API server reads everything it serves; producing nothing itself.
    # Auditing it records WHEN the service ran, which is the useful part.
    "serve": {
        "inputs": ["manifest_db", "curated_dir", "datasets_dir", "audit_log", "deid_dir"],
        "outputs": [],
    },
    # `audit` is deliberately absent: the observer does not observe itself.
}

# Cap on created-children recorded per directory output. A first `curate`
# run creates thousands of files; recording all of them would make the log
# bigger than the thing it describes. Prefix matching still attributes
# those files to the right run — `created` exists for exact attribution of
# the few top-level artifacts (a run dir, a release dir) that share a
# parent with siblings from other runs.
MAX_CREATED_ENTRIES = 500


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path, snapshot: dict[str, tuple] | None = None) -> dict:
    """Content fingerprint of a file or directory.

    Files get a full sha256 — they are small and their content IS their
    version. Directories get a cheap fingerprint (file count + total bytes):
    re-hashing hundreds of MB of curated PNGs on every command would be
    wasteful, and content-level integrity for those trees is already
    covered by the manifest and release CHECKSUMS.txt. `snapshot`, when
    passed, is filled with rel_path -> (size, mtime_ns) for every entry so
    the caller can diff before/after states to find what actually changed.
    """
    if not path.exists():
        return {"path": str(path), "kind": "missing"}
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    n_files, total = 0, 0
    for p in sorted(path.rglob("*")):
        if p.is_file():
            st = p.stat()
            n_files += 1
            total += st.st_size
            if snapshot is not None:
                snapshot[str(p.relative_to(path))] = (st.st_size, st.st_mtime_ns)
        elif snapshot is not None:
            snapshot[str(p.relative_to(path))] = (-1, -1)
    return {
        "path": str(path),
        "kind": "dir",
        "n_files": n_files,
        "bytes": total,
    }


def config_sha256(config: dict) -> str:
    """Fingerprint the RESOLVED config dict, not the file on disk.

    The dict is what actually ran (it already has --config resolution
    applied). Two runs with identical settings produce identical hashes even
    if the YAML comments differed — which is the comparison that matters.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def resolve_spec(config: dict, items: list[str]) -> list[Path]:
    """Resolve AUDIT_IO entries ("manifest_db", "artifacts_dir/x.json") to paths."""
    resolved = []
    for item in items:
        head, _, tail = item.partition("/")
        base = data_path(config, head)
        resolved.append(base / tail if tail else base)
    return resolved


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def _record_hash(record: dict) -> str:
    """Hash of this record chained to its predecessor.

    The chain is the tamper-evidence: `prev_hash` is INSIDE the hashed
    content, so rewriting history requires recomputing every later record
    — and `verify_log` walks the whole chain to catch exactly that.
    """
    payload = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(
        record["prev_hash"].encode() + b"\n" + _canonical(payload)
    ).hexdigest()


GENESIS_HASH = "0" * 64


def read_log(log_path: Path) -> list[dict]:
    """Read every record in order. A corrupt JSON line fails loudly."""
    if not log_path.is_file():
        return []
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"audit log corrupt at line {line_no}: {e}"
                    ) from e
    return records


def _append_record(log_path: Path, record: dict) -> None:
    """Append one line. Single write() under O_APPEND is atomic for lines
    smaller than PIPE_BUF; our records are well under that on any platform."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _actor() -> str:
    try:
        user = getpass.getuser()
    except (OSError, KeyError):
        user = os.environ.get("USER", "unknown")
    return f"{user}@{platform.node()}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@contextmanager
def audit_run(
    log_path: Path,
    *,
    command: str,
    argv: list[str],
    config: dict,
    inputs: list[Path] = (),
    outputs: list[Path] = (),
):
    """Wrap one command invocation in an audit record.

    Enter: fingerprint the declared INPUTS (the pre-run state the command
    will read) and snapshot the children of each directory output.
    Exit:  fingerprint the declared OUTPUTS (what now exists), diff the
    children to find what the run created, seal the record into the chain.

    The caller may set rec["exit_code"]; exceptions are recorded as
    status="error" and re-raised — a crashed run is still history.
    """
    prior = read_log(log_path)
    dirs_before: dict[str, dict[str, tuple]] = {}
    files_before: dict[str, str | None] = {}
    record = {
        "seq": (prior[-1]["seq"] + 1) if prior else 1,
        "run_id": uuid.uuid4().hex[:12],
        "command": command,
        "argv": argv,
        "actor": _actor(),
        "ts_start": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "code_version": __version__,
        "git_commit": _git_commit(),
        "config_sha256": config_sha256(config),
        "inputs": [fingerprint(p) for p in inputs],
        "exit_code": None,
        "status": "ok",
        "prev_hash": prior[-1]["record_hash"] if prior else GENESIS_HASH,
    }
    # BEFORE snapshot of declared outputs: what the run is about to touch.
    # This is what lets trace distinguish "produced" from "merely declared"
    # — a read-only mode like `release --verify` changes nothing, and must
    # not be recorded as the producer of the directory it looked at.
    for p in outputs:
        if p.is_dir():
            dirs_before[str(p)] = {}
            fingerprint(p, dirs_before[str(p)])
        elif p.is_file():
            files_before[str(p)] = _sha256_file(p)
        else:
            files_before[str(p)] = None

    try:
        yield record
    except BaseException as e:  # noqa: BLE001 — record, then re-raise
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["ts_end"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        start = _dt.datetime.fromisoformat(record["ts_start"])
        end = _dt.datetime.fromisoformat(record["ts_end"])
        record["duration_s"] = round((end - start).total_seconds(), 3)
        if record["status"] == "ok" and record["exit_code"]:
            record["status"] = "failed"

        out_entries = []
        for p in outputs:
            if p.is_dir():
                after: dict[str, tuple] = {}
                entry = fingerprint(p, after)
                before = dirs_before.get(str(p), {})
                created = sorted(k for k in after if k not in before)
                deleted = [k for k in before if k not in after]
                modified = [
                    k for k in after
                    if k in before and before[k] != after[k]
                ]
                entry["created"] = created[:MAX_CREATED_ENTRIES]
                entry["created_count"] = len(created)
                entry["deleted"] = deleted[:MAX_CREATED_ENTRIES]
                entry["deleted_count"] = len(deleted)
                entry["modified"] = modified[:MAX_CREATED_ENTRIES]
                entry["modified_count"] = len(modified)
                entry["changed"] = bool(created or deleted or modified)
            else:
                entry = fingerprint(p)
                before_sha = files_before.get(str(p))
                if entry["kind"] == "file":
                    entry["created"] = before_sha is None
                    entry["changed"] = before_sha != entry["sha256"]
                else:
                    # existed before but is gone now, or never existed
                    entry["created"] = False
                    entry["changed"] = before_sha is not None
            out_entries.append(entry)
        record["outputs"] = out_entries
        record["record_hash"] = _record_hash(record)
        _append_record(log_path, record)


def verify_log(log_path: Path) -> dict:
    """Re-walk the chain: recompute every record_hash and check every link."""
    records = read_log(log_path)
    prev = GENESIS_HASH
    for i, rec in enumerate(records):
        if rec.get("seq") != i + 1:
            return {
                "ok": False, "n_records": len(records), "broken_at": i + 1,
                "detail": f"sequence gap: expected seq {i + 1}, found {rec.get('seq')}",
            }
        if rec.get("prev_hash") != prev:
            return {
                "ok": False, "n_records": len(records), "broken_at": rec["seq"],
                "detail": f"prev_hash mismatch at seq {rec['seq']} — a record "
                          f"was inserted, removed, or reordered",
            }
        if rec.get("record_hash") != _record_hash(rec):
            return {
                "ok": False, "n_records": len(records), "broken_at": rec["seq"],
                "detail": f"record_hash mismatch at seq {rec['seq']} — the "
                          f"record's contents were edited after it was written",
            }
        prev = rec["record_hash"]
    return {"ok": True, "n_records": len(records), "broken_at": None, "detail": "chain intact"}


# ---------------------------------------------------------------------------
# Lineage tracing
# ---------------------------------------------------------------------------

def _covers(output: dict, target: Path) -> bool:
    """Does this output entry cover `target` (itself or an ancestor dir)?"""
    if output["kind"] == "missing":
        return False
    out = Path(output["path"])
    return target == out or target.is_relative_to(out)


def _output_touched(output: dict, target: Path) -> bool:
    """Did this run change `target` itself — not merely its parent dir?

    Attribution must be target-aware. A run that created datasets/v1.1 did
    change datasets/ but did NOT produce datasets/v1.0; a read-only run
    changed nothing at all. For dirs we check the recorded changed paths in
    BOTH directions: the target may sit inside a created subtree, or a
    changed file may sit inside the target.

    Records written before per-path tracking existed fall back to the
    coarse cover-and-changed test.
    """
    out = Path(output["path"])
    if output["kind"] == "file":
        return target == out and output.get("changed", True)
    lists, truncated = [], False
    for key in ("created", "modified", "deleted"):
        entries = output.get(key)
        count = output.get(f"{key}_count")
        if entries:
            lists.append(entries)
            if count is not None and count > len(entries):
                truncated = True
    if not lists:
        return _covers(output, target) and output.get("changed", True)
    for rel in (r for lst in lists for r in lst):
        p = out / rel
        if target == p or target.is_relative_to(p) or p.is_relative_to(target):
            return True
    # The changed-path list was capped; the target may live in the
    # unrecorded tail — the coarse test is the honest fallback.
    return truncated and _covers(output, target)


def find_producer(records: list[dict], target: Path, before_seq: int = 1 << 30) -> dict | None:
    """Newest run (with seq < before_seq) that actually changed `target`.

    Single pass, newest first — a newer overwrite must beat an older
    create, so passes cannot be split by match type. Read-only runs
    (`release --verify`) and runs that touched a *sibling* under the same
    parent (v1.1 vs v1.0) are both excluded by _output_touched.
    """
    for rec in reversed(records):
        if rec["seq"] >= before_seq:
            continue
        if any(_output_touched(o, target) for o in rec.get("outputs", [])):
            return rec
    return None


def trace_artifact(
    records: list[dict], target: Path, before_seq: int = 1 << 30,
    depth: int = 0, max_depth: int = 10,
) -> dict:
    """Build the ancestry tree of `target` back to the raw zone.

    Each node: {path, producer: record|None, inputs: [child nodes]}.
    A node with no producer is a root — raw data or hand-written config.
    before_seq pins the search to producers that ran BEFORE the consumer;
    without it manifest.db would trace to a later mutation, not the one
    that actually fed this run.
    """
    node = {"path": str(target), "producer": None, "inputs": []}
    if depth >= max_depth:
        node["note"] = "depth limit reached"
        return node
    producer = find_producer(records, target, before_seq)
    if producer is None:
        node["note"] = "no producing run — raw data or created outside the pipeline"
        return node
    node["producer"] = producer
    for inp in producer.get("inputs", []):
        if inp["kind"] == "missing":
            continue
        child = trace_artifact(
            records, Path(inp["path"]), before_seq=producer["seq"],
            depth=depth + 1, max_depth=max_depth,
        )
        node["inputs"].append(child)
    return node


def render_trace(node: dict, prefix: str = "", is_last: bool = True, root: bool = True) -> str:
    """Render a trace_artifact tree as an indented text diagram."""
    lines = []
    if root:
        lines.append(node["path"])
    producer = node.get("producer")
    if producer:
        argv_str = " ".join(producer["argv"][:6])
        lines.append(
            f"{prefix}{'└─' if is_last else '├─'} produced by #{producer['seq']} "
            f"`{argv_str}` — {producer['ts_start'][:19]}Z, "
            f"{producer['actor']}, {producer['status']}, "
            f"cfg {producer['config_sha256'][:8]}"
        )
        children = node.get("inputs", [])
        for i, child in enumerate(children):
            last = i == len(children) - 1
            child_prefix = prefix + ("   " if is_last else "│  ")
            label = Path(child["path"]).name or child["path"]
            if child.get("producer") or child.get("inputs"):
                lines.append(f"{child_prefix}{'└─' if last else '├─'} {label}")
                lines.append(
                    render_trace(child, child_prefix + ("   " if last else "│  "),
                                 is_last=True, root=False)
                )
            else:
                note = child.get("note", "")
                lines.append(
                    f"{child_prefix}{'└─' if last else '├─'} {label}   [{note}]"
                )
    else:
        note = node.get("note", "")
        lines.append(f"{prefix}{'└─' if is_last else '├─'} {note}")
    return "\n".join(lines)


def render_log(records: list[dict], tail: int | None = None) -> str:
    """One line per run: seq, time, command, actor, status, duration."""
    shown = records[-tail:] if tail else records
    lines = [
        f"{'#':>4s}  {'timestamp':19s}  {'command':24s}  {'actor':24s}  "
        f"{'status':6s}  {'dur(s)':>7s}  {'cfg':8s}"
    ]
    for r in shown:
        argv_str = " ".join(a for a in r["argv"] if not a.startswith("/"))[:24]
        lines.append(
            f"{r['seq']:>4d}  {r['ts_start'][:19]:19s}  {argv_str:24s}  "
            f"{r['actor'][:24]:24s}  {r['status']:6s}  "
            f"{r.get('duration_s', 0):7.1f}  {r['config_sha256'][:8]:8s}"
        )
    if tail and len(records) > tail:
        lines.insert(0, f"(showing last {tail} of {len(records)} runs)")
    return "\n".join(lines)


def audit_log_path(config: dict) -> Path:
    """Where the log lives. Tolerant of configs that predate the audit_log key."""
    key = config.get("paths", {}).get("audit_log")
    if key:
        return PROJECT_ROOT / key
    return data_path(config, "artifacts_dir") / "audit.log"
