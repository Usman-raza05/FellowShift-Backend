# FellowShift Verification API

## Current architecture and audit findings

The original backend was a single `main.py` module containing FastAPI routes,
provider clients, OCR and image processing, fixed income/marks thresholds, and
an in-memory application dictionary. It initialized both provider clients at
import time, permitted unauthenticated access to the application/admin routes,
accepted unbounded uploads, returned provider exception details, and included
admin endpoints that could mutate the application's status.

The existing top-level provider/model benchmark scripts directly call external
AI APIs and may incur charges. They are exploratory scripts, not the automated
test suite; the `unittest` suite uses in-memory synthetic images and mocked
providers.

The versioned API is stateless for application records. Next.js supplies the
scholarship criteria and remains the source of truth for application state,
selection, and approval. The previous admin routes now return `410 Gone`; the
legacy upload endpoint remains available as a migration bridge, requires the
service token and an idempotency key, accepts optional JSON
scholarship-specific criteria, and no longer persists application data.
Requests without criteria through that legacy endpoint return `NEEDS_REVIEW`.

OCR extracts visible data only. It does not authenticate a document, determine
legal validity, or make an approval decision.

## Endpoint and authentication

- Method: `POST`
- Path: `/api/v1/verifications`
- Production URL: `https://fellowshift-backend.onrender.com/api/v1/verifications`
- Content type: `multipart/form-data`
- Authentication: bearer token in the `Authorization` header, using the
  configured service-token environment variable.
- Idempotency: `Idempotency-Key` header, 8–200 characters.

The service token must be at least 32 characters. Missing/weak configuration
fails closed with HTTP 503; missing or invalid credentials receive HTTP 401.
The token is intended for Next.js server-to-server calls; do not expose it to a
browser or mobile client.

## Multipart request

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `application_data` | JSON string | Yes | `{ "application_id": "...", "application_data": { ... } }`; scalar values only |
| `criteria` | JSON array string | Yes | Actual rules supplied by the scholarship system |
| `required_document_types` | JSON string array | Yes | At least one document type code required for this request |
| `document_types` | JSON string array | No | One type code per `documents` file, in upload order |
| `documents` | Repeated file field | No | JPEG, PNG, or WebP documents |

Example `application_data`:

```json
{
  "application_id": "synthetic-app-123",
  "application_data": {
    "sample_attribute": "synthetic-value"
  }
}
```

Example `criteria`:

```json
[
  {
    "field": "attribute_from_scholarship_config",
    "operator": "eq",
    "value": "value_from_scholarship_config",
    "reason_code": "RULE_CODE_FROM_SCHOLARSHIP_CONFIG",
    "explanation": "Explanation supplied by scholarship configuration."
  }
]
```

The strings above are placeholders, not a valid scholarship rule; replace them
with values from the actual scholarship configuration. The API does not provide
default scholarship rules. Supported operators: `eq`, `ne`, `lt`, `lte`, `gt`, `gte`,
`in`, and `not_in`. Ordering operators require numeric values. `in` and
`not_in` require a non-empty scalar array. Missing application attributes
produce `NEEDS_REVIEW`.

Each document is limited to 10 MiB; at most five documents are accepted, with a
maximum decoded image size of 20 megapixels. The declared MIME type must match
the detected image format. PDF, TIFF, and other document formats are not
supported by this OCR pipeline. Uploads are processed in memory and are not
stored by this service.

## Response

```json
{
  "application_id": "synthetic-app-123",
  "outcome": "ELIGIBLE",
  "reasons": [
    {
      "code": "CRITERIA_MET",
      "explanation": "The supplied scholarship criteria were met; this is not an approval decision."
    }
  ],
  "documents": [
    {
      "document_type": "income_certificate",
      "provider": "MISTRAL",
      "extraction": {
        "is_valid_document": true,
        "document_type": "income_certificate",
        "extracted_full_name": null,
        "extracted_category_caste": null,
        "extracted_annual_income": null,
        "extracted_percentage": null,
        "certificate_number": null,
        "issuing_authority": null,
        "discrepancies_or_issues": []
      }
    }
  ]
}
```

Outcomes are `ELIGIBLE`, `INELIGIBLE`, `NEEDS_REVIEW`, and `FAILED`.
Unreadable or uncertain OCR, a required document that was not uploaded, and
missing eligibility data produce `NEEDS_REVIEW`; they do not imply ineligibility.
An AI timeout/provider failure produces `FAILED` with HTTP 503. `ELIGIBLE` is
only an eligibility screen and never an approval.

Criteria are evaluated against the scalar fields in `application_data`.
When an extracted document field corresponds to a supplied field (for example,
`extracted_annual_income` and `annual_income`), differing values produce
`NEEDS_REVIEW`; document OCR uncertainty is not turned into ineligibility.

## Errors

| HTTP | `error` | Meaning |
|---:|---|---|
| 400 | `EMPTY_DOCUMENT` / `INVALID_DOCUMENT` | Empty or malformed image |
| 401 | `UNAUTHORIZED` | Invalid/missing bearer token |
| 409 | `DUPLICATE_REQUEST` | Idempotency key already claimed |
| 410 | `APPLICATIONS_OWNED_BY_NEXTJS` / `APPLICATION_STATUS_OWNED_BY_NEXTJS` | Legacy admin/status API has been retired |
| 413 | `DOCUMENT_TOO_LARGE`, `TOO_MANY_DOCUMENTS`, `IMAGE_DIMENSIONS_TOO_LARGE`, `APPLICATION_DATA_TOO_LARGE`, or `REQUEST_METADATA_TOO_LARGE` | Upload limit exceeded |
| 415 | `UNSUPPORTED_FILE_TYPE` / `FILE_TYPE_MISMATCH` | Unsupported or mismatched format |
| 422 | `INVALID_REQUEST`, `INVALID_APPLICATION_DATA`, `INVALID_CRITERIA`, or document metadata error | Request shape or criteria invalid |
| 503 | `AUTH_NOT_CONFIGURED`, `INVALID_TIMEOUT_CONFIGURATION`, `AI_PROVIDER_TIMEOUT`, `AI_PROVIDER_FAILURE`, `VERIFICATION_UNAVAILABLE`, or `IDEMPOTENCY_UNAVAILABLE` | Temporary processing/configuration failure |

Error responses use a non-sensitive `detail` object with an `error` code and
safe `message`. Provider exception strings and student data are not returned
or logged.

## Idempotency, timeouts, and retries

The service hashes the idempotency key and stores only that digest and a
timestamp in SQLite; it does not store the submitted data or response. A key
can be claimed once within a 24-hour window. Reuse returns HTTP 409, including
if the original attempt failed. Generate a new key only for a genuinely new
attempt; the caller should save a successful response with its application
record in Next.js.

Each provider call has a bounded HTTP timeout (`AI_PROVIDER_TIMEOUT_SECONDS`),
clamped to leave room for one primary and one fallback attempt within
`AI_REQUEST_TIMEOUT_SECONDS`. Automatic provider retries are disabled; the
single Gemini fallback after Mistral failure is reported as
`GEMINI_FALLBACK`. A timed-out thread may take until the provider's own HTTP
timeout to stop, so the per-provider limit is shorter than the API request
limit.

## Configuration and deployment

Required:

- `FELLOWSHIFT_SERVICE_TOKEN`: random secret, at least 32 characters.
- At least one of `MISTRAL_API_KEY` or `GEMINI_API_KEY`.

Optional:

- `MISTRAL_MODEL` (default `pixtral-12b-2409`)
- `GEMINI_MODEL` (default `gemini-3.5-flash`)
- `AI_PROVIDER_TIMEOUT_SECONDS` (default `25`)
- `AI_REQUEST_TIMEOUT_SECONDS` (default `65`)
- `IDEMPOTENCY_DB_PATH` (default `.data/idempotency.sqlite3`)

For Render, set the start command to
`uvicorn main:app --host 0.0.0.0 --port $PORT`, configure secrets in the
service environment (not in Git), and mount a persistent disk at the configured
SQLite parent directory. Run one service instance: SQLite is a local,
single-instance idempotency store, not a distributed lock. Without persistent
storage, idempotency history is lost on restart/redeploy. Use HTTPS at the
deployment edge, configure reverse-proxy request/rate limits, and monitor
provider health and disk capacity. No database or queue is required for
application records.

Documents are sent to the configured external AI provider for extraction.
Before production use, confirm the provider's data-processing terms, regional
handling, and retention meet FellowShift's privacy and compliance requirements.

## Next.js integration steps

1. Keep all backend and AI provider secrets in Next.js server-only environment
   variables; call the backend from a server action/API route, never directly
   from the browser.
2. Before sending, persist the application and documents in the existing
   Next.js/Supabase workflow. Read files from the existing storage layer and
   send them as repeated `documents` multipart fields; provide matching
   `document_types` in the same order.
3. Construct `application_data` and the scholarship's actual `criteria` from
   trusted server-side scholarship configuration. Do not copy criteria from
   this document's examples.
4. Generate one random idempotency key per verification attempt and preserve it
   across transport retries. Treat HTTP 409 as a duplicate attempt, not a
   successful verification.
5. Persist the returned response and outcome in Next.js. Route `NEEDS_REVIEW`
   to manual review; do not set approved/rejected based solely on this API.
6. Handle HTTP 503 as `FAILED`/retryable with a new key and deliberate retry
   policy; do not convert it to rejection. Set the client timeout above
   `AI_REQUEST_TIMEOUT_SECONDS` plus network overhead.
7. Treat extracted document fields as sensitive. Restrict access and retention
   in Next.js/Supabase according to the platform's privacy policy.
