# Step 13 — Lineage & audit log

*Phase 5 · Services. Previous: [step-12-active-learning.md](step-12-active-learning.md).*

## What we built

An append-only, hash-chained audit log at `artifacts/audit.log`. Every
`python -m medimageforge <command>` invocation now writes one JSONL record:
who ran it (`actor@host`), when, the exact `argv`, the resolved config
fingerprint, code version + git commit, fingerprints of the declared inputs
(before the run) and outputs (after), exit status, and duration.

Three ways to read it:

```bash
python -m medimageforge audit                  # tail of the run history
python -m medimageforge audit --verify         # re-walk the hash chain
python -m medimageforge audit --trace <path>   # full ancestry of an artifact
```

## Why

Until now we had *data* provenance (file `sha256` in the manifest,
`source_sha256` in curation, `labels_sha256` in releases) but no
*execution* provenance. Nothing could answer "which run produced
`datasets/v1.1`, with what config, reading what?" — the core regulated-
MedTech requirement, and incidentally the best debugging tool: a surprising
artifact's first question is always "who made this, from what?"

## How — three design decisions that carry the step

### 1. Append-only is a policy; the hash chain is the enforcement

Nothing stops anyone from `vim artifacts/audit.log`. So each record stores
`prev_hash`, and `record_hash = sha256(prev_hash + canonical_json(record))`.
Editing or deleting record N invalidates its own hash *and* the `prev_hash`
link in N+1 — verification re-walks the whole chain and names the break:

```
=== Audit chain verification ===
Records: 3
Result:  BROKEN — record_hash mismatch at seq 2 — the record's contents
         were edited after it was written
```

This is the same construction as a blockchain, Git's commit chain, and
`SHA256SUMS.txt` files — one idea, three scales. It detects tampering; it
cannot prevent it (an attacker could rewrite *all* records), which is why
production systems also ship logs off-box. For a learning platform,
detection is the lesson.

### 2. Inputs are fingerprinted BEFORE the run; outputs AFTER

An input fingerprint taken after the run describes bytes the command never
read. So `audit_run` (a context manager wrapping dispatch in `main()`)
hashes inputs at `__enter__` and outputs at `__exit__`. The test
`test_input_fingerprint_is_the_prerun_state` mutates a file *inside* the
run and asserts the recorded hash is the original's — proving the ordering,
not just the presence, of the fingerprint.

Config gets the same treatment: `config_sha256` hashes the *resolved*
config dict as canonical JSON — the settings that actually ran, immune to
comment/whitespace changes in the YAML.

### 3. Attribution records what changed, not what was declared

First version of `find_producer` matched "newest run whose declared outputs
cover the path". It immediately mis-attributed `datasets/` to
`release --verify` — a read-only command that declares `datasets_dir` as an
output but changes nothing. Two fixes, each caught by a failing test:

- **Change detection.** `audit_run` snapshots each declared output *before*
  the run (dir children → `rel_path: (size, mtime_ns)`; files → sha256) and
  diffs after: `created`/`modified`/`deleted` paths plus a `changed` flag.
  A verify run shows `changed: false` and can never be a producer.
- **Target-aware matching.** Even a *mutating* run is only the producer of
  the paths it actually touched: `active-learning` created `datasets/v1.1`,
  which changes `datasets/` but does not make it the producer of
  `datasets/v1.0`. Changed paths are checked in both directions — the
  target inside a created subtree (`v1.0/` covers `v1.0/index.csv`), or a
  changed file inside the target (`eval/run-X/evaluation.json` covers
  `eval/run-X`).

And `find_producer` must be a *single* newest-first pass, not "all exact
matches then all prefix matches" — otherwise an older run that created a
file beats a newer run that overwrote it. A test that rewrites
`manifest.db` three times (`ingest` → `curate` → `load-labels`) exposed
exactly that.

## What a real trace looks like

```
$ python -m medimageforge audit --trace artifacts/manifest.db
/home/zahra/MedImageForge/artifacts/manifest.db
└─ produced by #3 `load-labels` — 2026-09-18T13:51:29Z, zahra@z130, ok, cfg 130ce101
   ├─ manifest.db
   │  └─ produced by #2 `ingest` — 2026-09-18T13:47:53Z, zahra@z130, ok, cfg 130ce101
   │     └─ data   [no producing run — raw data or created outside the pipeline]
   └─ hemorrhage_diagnosis.csv   [no producing run — raw data or created outside the pipeline]
```

Two honest details visible here:

- **`before_seq` matters.** `load-labels` both reads and writes
  `manifest.db`. Tracing its input must search producers *earlier* than
  run #3 — otherwise lineage flows backward in time.
- **`datasets/v1.0` traces to "no producing run"** — it was published before
  the log existed. The trace says so rather than guessing. Lineage only
  covers what happened after it was switched on.

## Deliberate limitations (documented, not hidden)

- **Directory outputs get cheap fingerprints** (file count + bytes +
  changed-path lists). Hashing ~330 MB of curated PNGs on every command is
  wasteful; content-level integrity already lives in the manifest and the
  release `CHECKSUMS.txt`.
- **Changed-path lists are capped at 500 entries.** A first `curate` run
  touches ~5000 files; beyond the cap, matching falls back to the coarse
  cover test — noted in the code, correct but less precise.
- **Failures and crashes are recorded** (`status: failed`/`error`,
  exception text included). A crashed run is still history — arguably the
  most important kind.
- **`salt_file` is fingerprinted, not copied.** sha256 of a high-entropy
  secret is a fingerprint, and recording it means a *salt change* — which
  would silently break pseudonym stability — becomes visible in the log.
- **`audit` does not audit itself.** The observer stays out of the
  observed set.

## A small reproducibility observation

`load-labels` counts as a *producer* of `manifest.db` every time because
`annotated_at` timestamps change the file's bytes — an idempotent command
that is not byte-reproducible. The audit log surfaced that incidentally;
it's honest (the DB genuinely differs) but worth remembering when thinking
about "reproducible" pipelines: determinism of *content* and determinism
of *bytes* are different claims.

## Verified

- `pytest` — **242 passed, 1 skipped** (28 new tests)
- Live tamper demo: editing one record's `status` → `BROKEN — record_hash
  mismatch at seq 2`; restore → `INTACT`
- `audit --trace artifacts/manifest.db` produces the three-level chain above
- `release --verify` recorded but correctly *not* attributed as a producer

## Files

- `src/medimageforge/audit.py` — log, chain, fingerprints, trace, renderers
- `src/medimageforge/cli.py` — dispatch table wrapped in `audit_run`;
  `audit` command
- `configs/default.yaml` — `paths.audit_log`
- `tests/test_audit.py` — chain integrity, tamper detection, change
  detection, producer attribution, tracing
