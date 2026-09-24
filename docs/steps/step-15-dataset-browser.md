# Step 15 — Dataset browser UI

*Phase 5 · Services. Previous: [step-14-api-service.md](step-14-api-service.md).*

## What we built

A Streamlit review UI — `python -m medimageforge ui` — that lets a human
browse the whole platform without touching code or SQL:

| page | shows | data arrives via |
|---|---|---|
| **Patients** | patient table → per-patient metrics → slice picker → CT image + mask overlay + labels | `/patients`, `/patients/{p}`, `/slices/{p}/{n}`, `/image` |
| **Labels** | label distribution chart | `/labels/distribution` |
| **Datasets** | releases, splits, dataset cards | `/datasets`, `/datasets/{v}` |
| **QC** | gate banner + per-check table | `/qc` |
| **Audit** | chain integrity + recent runs | `/audit`, `/audit/verify` |

The sidebar holds the API base URL and a live health indicator — a reviewer
can point the UI at any deployment and immediately see whether it's up.

## Why

Two reasons, per the roadmap: humans must be able to *see* the data
(browse any patient, visually verify labels and masks), and — the subtler
one — **the UI is what makes Step 14's contract real**. "The UI uses only
this API" is a claim you prove by building the UI and finding the gaps.

## How — the architecture that makes the claim testable

```
ui.py            pure Streamlit rendering — imports NOTHING from the
                 pipeline internals (no sqlite, no manifest, no paths)
   │
api_client.py    typed functions, one per endpoint; the ONLY place HTTP
   │             happens in the UI layer
   │
httpx.Client     injected, not global — production gets
                 make_client(base_url), tests inject TestClient
   │
FastAPI app      Step 14 — pseudonyms only, real IDs can't be queried
```

Two deliberate choices:

1. **`api_client` takes the client as a parameter.** `fastapi.testclient.
   TestClient` IS an `httpx.Client` over the app's ASGI transport — so
   `tests/test_api_client.py` drives the real client functions against the
   real API with **zero live server**. The contract is tested, not assumed.
2. **The shared fixture moved to `tests/conftest.py`.** `test_api` checks
   the server side, `test_api_client` checks the client side — against the
   same synthetic platform, so the two sides can never drift apart. This is
   also where Step 14's lesson got applied: the fixture encodes the real
   `gate` schema, not a guess.

Error posture: `ApiError` carries status + server detail ("no mask for
patient X slice 1"), and `_guard` turns both API errors and connection
failures into `st.error` — a UI that explains beats a UI that crashes.

## Verified

- `pytest` — all suite green (60 tests across api + client + audit files)
- **Live end-to-end**: `serve` + `ui` running, Streamlit's `AppTest`
  rendered the real page headlessly — no exceptions, sidebar showed
  `API v0.1.0 · manifest OK`, Patients page showed **82 patients**.
- Deprecation caught by that same smoke run: `use_container_width` →
  `width="stretch"` (5 call sites).

## Files

- `src/medimageforge/ui.py` — the Streamlit app (five pages, pure client)
- `src/medimageforge/api_client.py` — typed API client + `ApiError`
- `src/medimageforge/cli.py` — `ui` command (launches `streamlit run`)
- `src/medimageforge/audit.py` — `ui` added to the audit spec
- `tests/conftest.py` — shared synthetic-platform fixture
- `tests/test_api_client.py` — 14 client-contract tests
- `configs/default.yaml` — `ui.port`
- `requirements.txt` — streamlit

## Not done (deliberately)

- No reviewer actions (annotate/flag) — those are POSTs, and POSTs need
  auth, which is Step 17/18 scope. The UI is read-only like the API.
- No pagination/virtualization — 82 patients, ~30 slices each; the pickers
  are fine at this size.
- Images render per click — no caching layer yet.
