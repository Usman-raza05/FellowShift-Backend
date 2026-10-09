# AI Test & Benchmark Report

**Project:** ST Scholarship AI Verification Engine  
**Purpose:** Compare AI providers and image-processing approaches for scholarship-document extraction and verification.  
**Last Updated:** 9 October 2026  
**Status:** Development and testing only — not production certified.

## 1. Executive Summary

This report summarizes development-stage testing of the ST Scholarship AI Verification Engine. The objective was to compare AI providers, measure document-processing speed, and evaluate image preprocessing.

In the latest recorded three-document comparison, **Mistral `pixtral-12b-2409` was the fastest**, completing the workload in 15.25 seconds, averaging 5.08 seconds per document. Gemini 3 completed the same workload in 15.88 seconds, averaging 5.29 seconds per document.

Across three recorded benchmark rounds, Mistral was faster than Gemini in every round. Mistral's average total processing time was 20.31 seconds, compared with 26.08 seconds for Gemini, representing approximately 22.1% less elapsed time.

Image preprocessing reduced typical image sizes from approximately 7 MB PNG files at 1792 × 2400 pixels to 0.32–0.40 MB JPEG files at 1400 × 1875 pixels.

These results are preliminary and require further accuracy testing before production deployment.

## 2. Test Scope and Test Data

The workflow tested three types of scholarship-related documents:

- Scheduled Tribe (ST) or caste certificate
- Income certificate
- Academic marksheet

All test documents were fictional demonstration documents marked "SAMPLE — NOT VALID FOR OFFICIAL USE."

| Field | Test Value |
|---|---|
| Student name | Aarav Kumar |
| Category | Scheduled Tribe (ST) |
| Annual income | ₹180,000 |
| Marks | 390 / 500 |
| Percentage | 78% |
| Test certificate ID | `TEST-ST-2026-001` |

These values are synthetic and do not represent real applicants or official scholarship eligibility requirements.

## 3. Recorded Benchmark Results

### 3.1 Latest Three-Document Comparison

| Provider / Model | Total Time | Average per Document |
|---|---:|---:|
| Gemini 3 | 15.88 s | 5.29 s |
| Mistral `pixtral-12b-2409` | **15.25 s** | **5.08 s** |

Mistral completed the workload 0.63 seconds faster than Gemini, representing approximately 4% lower total elapsed time.

### 3.2 Earlier Recorded Comparison

| Provider | Total Time | Average per Document |
|---|---:|---:|
| Gemini | 29.64 s | 9.88 s |
| Mistral | **19.99 s** | **6.66 s** |

Mistral was faster in this earlier run as well. The results came from separate executions and should not be treated as a single controlled experiment.

### 3.3 Three-Round Benchmark Summary

| Round | Gemini | Mistral | Winner |
|---|---:|---:|---|
| 1 | 29.64 s | 19.99 s | Mistral |
| 2 | 15.88 s | 15.25 s | Mistral |
| 3 | 32.73 s | 25.70 s | Mistral |
| **Average total time** | **26.08 s** | **20.31 s** | **Mistral** |

Mistral was faster in all three recorded rounds.

- Average difference: 5.77 seconds per three-document workload.
- Reduction in elapsed time: approximately 22.1%.

These are development observations, not statistically established performance differences.

### 3.4 Per-Document Timing and Timeout Comparison

| Document | Gemini | Mistral |
|---|---:|---:|
| Caste certificate | Timeout at 31.17 s | 4.91 s |
| Income certificate | 18.66 s | 5.67 s |
| Marksheet | 20.47 s | 4.72 s |
| **Successful processing total** | **39.13 s / 2 documents** | **15.30 s / 3 documents** |
| **Average successful processing time** | **19.57 s** | **5.10 s** |

Gemini timed out on the caste certificate, while Mistral successfully processed all three documents. Since the providers completed different numbers of documents, total processing times are not directly comparable.

### 3.5 Other Observed Provider and Model Outcomes

| Provider / Model | Observed Result | Notes |
|---|---|---|
| Gemini 2.5 Flash | HTTP 404 | Model unavailable in the tested environment. |
| Gemini 2.5 Flash-Lite | HTTP 404 | Similar model-availability issue. |
| Gemini 3 | Successful runs with occasional HTTP 503 errors | Intermittent service errors occurred. |
| Groq | Initial write-timeout errors | Initial requests timed out around 30 seconds; a later attempt succeeded. |
| Mistral `pixtral-12b-2409` | Successful extraction runs | Fastest in the latest recorded comparison. |

These observations describe the test environment at the time. Provider availability and model identifiers may change.

## 4. Image Preprocessing

Image preprocessing was introduced to reduce image payload sizes before sending documents to the AI provider.

### 4.1 Before-and-After Comparison

| Property | Before | After |
|---|---|---|
| Image dimensions | 1792 × 2400 pixels | 1400 × 1875 pixels |
| File size | Approximately 7 MB | Approximately 0.32–0.40 MB |
| Format | PNG | JPEG |
| Processing | Original image payload | In-memory resizing and JPEG encoding |

The preprocessing pipeline reduced image payload sizes by approximately 94–95%.

### 4.2 Benefits and Limitations

Image preprocessing provides several potential benefits:

- Reduces image-upload bandwidth requirements.
- Decreases data-transfer overhead.
- May improve request preparation and transmission time.
- Reduces payload-handling requirements.

However, smaller images do not guarantee a proportional reduction in total AI processing time. Model inference, network latency, provider load, and request handling also affect performance.

JPEG compression and resizing may reduce fine details in certificate numbers, stamps, and small printed text. Therefore, preprocessing settings must be evaluated alongside extraction accuracy.

## 5. Application-Level Verification Checks

The application was designed to extract document fields and compare them with information provided in the scholarship application form.

### 5.1 Verification Checks

The workflow includes:

- **Document-type validation:** Checks whether the uploaded document matches the expected type.
- **Required-field validation:** Checks whether necessary fields have been extracted.
- **Student-name consistency:** Compares the extracted name with the application form and supporting documents.
- **Category consistency:** Checks whether the extracted category or caste information matches the declared details.
- **Income consistency:** Compares extracted income with the declared application value.
- **Marks verification:** Compares extracted marks and percentage with submitted academic information.
- **Sample-document detection:** Identifies sample, demo, fictional, or testing-only indicators.

These checks support automated screening and inconsistency detection. They do not independently establish document authenticity or scholarship eligibility.

### 5.2 Development Eligibility Thresholds

The prototype uses the following example thresholds:

| Rule | Example Threshold | Status |
|---|---:|---|
| Annual income | ₹250,000 | Development placeholder |
| Minimum marks | 50% | Development placeholder |

These values are illustrative and must not be treated as official scholarship eligibility requirements. The applicable scheme's current rules must be verified against authoritative sources before deployment.

### 5.3 Verification Modes

The application supports a configurable `VERIFICATION_MODE` setting.

**Test mode (`test`)**

- Intended for clearly identified demonstration documents.
- Supports test-only checks and development workflows.
- Helps validate extraction and rule-processing behavior without implying an official decision.

**Real verification mode (`real`)**

- Performs extraction and rule-based cross-checks for real-document workflows.
- Compares extracted values with application-form information.
- Requires appropriate validation of extracted values and scheme rules.

A successful test-mode result is not an official scholarship approval. Successful AI extraction also does not establish that a certificate is genuine. Real applications require appropriate human review and authoritative verification where necessary.
