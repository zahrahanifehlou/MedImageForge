# Step 14 — API service

*Phase 5 · Services. Previous: [step-13-lineage-audit.md](step-13-lineage-audit.md).*

## What we built

A read-only HTTP API (`python -m medimageforge serve`, FastAPI + uvicorn)
over the manifest, curated zone, releases, QC report, and audit log. Every
step so far produced data the API now serves:

```
GET /health                        liveness + manifest presence
GET /patients                    → Step 4's files + Step 6's pseudonyms + Step 7's labels
GET /patients/{pseudonym}        → + Step 6's de-identified demographics
GET /patients/{pseudonym}/slices → Step 7's per-slice labels
GET /slices/{p}/{n}              → Step 7's show-slice, over HTTP
GET /slices/{p}/{n}/image        → Step 5's curated PNGs (+ ?overlay=true → Step 3's masks)
GET /labels/distribution         → Step 7's label store
GET /datasets, /datasets/{v}     → Step 9/12's releases + cards
GET /qc                          → Step 8's gate report
GET /audit, /audit/verify        → Step 13's run log + chain check
```

## Why

Until now every consumer touched `manifest.db` directly — fine for one
developer, wrong for a platform. A service boundary forces clean access
patterns, gives Step 15's UI a stable contract, and is the future home of
access control: *the API is the one place that decides what a client may
see.*

**Read-only on purpose.** No POST/PUT yet. Annotation writes will come with
a reviewer workflow — and they need authentication first. A write API
without auth is an incident, not a feature.

## How — the central design rule

**The API speaks pseudonyms only.** The manifest keys everything by real
patient ID inside the controlled zone. If clients could pass `"049"`, every
client becomes a PHI consumer and Step 6's boundary leaks through a new
door. So:

- Requests take pseudonyms; the server resolves them via the `patients`
  table (`load_pseudonym_map`, the Step 10 read-only reader).
- Responses echo pseudonyms. Asking for `049` returns **404** — the
  protocol *cannot express* a real-ID query.
- Demographics are read from the **de-identified export**
  (`demographics_pseudonymized.csv`), never the raw CSV.
- The service binds **127.0.0.1** — no auth exists yet, so no network
  exposure. That's a deployment posture encoded in config, not a promise.

## Two bugs found — both the privacy-boundary kind

**1. Real IDs leaked inside paths.** `GET /slices/{p}/14` replaced
`patient_id` with the pseudonym — and still leaked `049` inside
`mask_rel_path: "Patients_CT/049/brain/14_HGE_Seg.jpg"`. The test that
catches it is the important part: `assert REAL_ID not in json.dumps(body)`
scans the *whole serialized response*, not the fields I remembered to
translate. Fixed with `_pseudonymize_path`, which rewrites the ID only when
it appears as an exact path segment (so `1049` or `0492` are untouched).

Same bug family as Step 9's `index.csv` leak — **an identifier hidden
inside a larger string**. The lesson generalizes: *don't check the fields
you thought to sanitize; check the bytes that leave the building.*

**2. The `/qc` endpoint read a schema that didn't exist.** I coded
`data["passed"]`; the real report writes `gate: "PASS"`. The live smoke
test caught it — `passed: null` on a report that said PASS. The deeper bug
was in my *test fixture*, which had encoded my assumption instead of the
real schema. Fixed both: the endpoint reads `gate`, and the fixture now
mirrors `qc.write_report`'s actual output. **A fixture must encode
reality, not your guess about it** — the same lesson as Step 8's
random-noise images.

## Verified

- `pytest` — **260 passed, 1 skipped** (18 new API tests + spec update)
- Live: `/patients` → 82 pseudonym rows; `GET /slices/049/14` → **404**;
  `GET /slices/PAT-98c32af848a5/14` → labels with pseudonym echoed;
  `/slices/.../image` → `200 image/png`; `/datasets` → v1.0 + v1.1;
  `/qc` → `gate: PASS`; `/audit/verify` → `chain intact`.
- Error posture checked: missing manifest → **503 with a remediation
  hint**, not a 500 stack trace; missing QC report → **404**.

## Files

- `src/medimageforge/api.py` — `create_app(config)` factory + all routes
- `src/medimageforge/cli.py` — `serve` command (uvicorn)
- `src/medimageforge/audit.py` — `serve` added to the audit IO spec
- `configs/default.yaml` — `api.host` / `api.port`
- `tests/test_api.py` — 18 tests incl. the privacy-boundary suite
- `requirements.txt` — fastapi, uvicorn, httpx (TestClient)

## Not done (deliberately)

- No authentication/authorization — Step 17/18 territory, and the reason
  the bind address stays localhost.
- No pagination on `/patients` — 82 rows don't need it; the API shape
  leaves room (`?offset`/`?limit` later).
- Overlay rendering happens per request — no caching. Fine at this size.
