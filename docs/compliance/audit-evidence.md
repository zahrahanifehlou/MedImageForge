# Audit-trail evidence samples

Real outputs captured from the platform's own verification tools.
These are *generated evidence*, not narrative claims — a reviewer can
reproduce each with the command shown.

## 1. Audit chain integrity

```
$ python -m medimageforge audit --verify
Result:  INTACT — chain intact
```

Tamper detection demonstration (Step 13): flipping one record's `status`
field produced `BROKEN — record_hash mismatch at seq 2` — the chain
localizes the exact edited record.

## 2. Command history (every run recorded)

```
$ python -m medimageforge audit
   #  timestamp            command           actor        status  dur(s)  cfg
   1  2026-09-18T13:47:48  release --verify  zahra@z130   ok        4.0  130ce101
   2  2026-09-18T13:47:53  ingest            zahra@z130   ok        0.7  130ce101
   3  2026-09-18T13:51:29  load-labels       zahra@z130   ok        0.1  130ce101
   4  2026-09-24T13:26:25  release --verify  zahra@z130   ok        6.6  21b36d41
```

Each record carries: actor@host, argv, code version + git commit, resolved
config hash, input/output fingerprints, status. A *failed* run is recorded
as `failed` — history includes failures, not just successes.

## 3. Lineage trace — artifact → raw

```
$ python -m medimageforge audit --trace artifacts/manifest.db
=== Lineage of artifacts/manifest.db ===
artifacts/manifest.db
└─ produced by #3 `load-labels` — 2026-09-18T13:51:29Z, zahra@z130, ok
   ├─ manifest.db
   │  └─ produced by #2 `ingest` — 2026-09-18T13:47:53Z, zahra@z130, ok
   │     └─ data   [no producing run — raw data]
   └─ hemorrhage_diagnosis.csv   [raw input]
```

`datasets/v1.0` traces to "no producing run" — published before the log
existed; the trace says so rather than guessing (honest-gap behavior).

## 4. Release verification

```
$ python -m medimageforge release --verify
=== Verifying datasets/v1.0 ===
Release files modified/missing: 0
Referenced images drifted:      0
Real patient IDs leaked:        0
VERIFY: PASS
```

Re-hashes every file in `CHECKSUMS.txt`, every referenced source image,
*and* scans the release for real patient IDs — three integrity properties
in one gate.

## 5. QC gate

```
artifacts/qc_report.json:
  "gate": "PASS", "n_errors": 0, "n_warnings": 4
  warnings: window-count-mismatch on patients 084, ... (documented,
            known dataset characteristic — not a pipeline defect)
```

## 6. What a reviewer can verify independently

| claim | verify with |
|---|---|
| Releases immutable | re-run `release` → refuses overwrite; `release --verify` → PASS |
| Audit untampered | edit one byte of `audit.log` → `audit --verify` reports the seq |
| No real IDs outside | `grep -r "049" datasets/ artifacts/deid/` → no matches |
| Splits are patient-level | `splits.csv` — no patient in two splits; seed reproduces it |
| Model ↔ dataset binding | run record `dataset_version` ↔ `datasets/vX.Y/metadata.json` |
