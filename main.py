import os
import json
import asyncio
import sqlite3
import time
from typing import Optional
from uuid import uuid4
import logging

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from verification_api.security import require_service_auth
from verification_api.documents import read_and_validate_document
from verification_api.config import get_ai_request_timeout_seconds
from verification_api.providers import optimize_image_for_ai, process_document
from verification_api.schemas import (
    DocumentExtractionResult,
    DocumentVerification,
    EligibilityCriterion,
    VerificationOutcome,
)
from verification_api.service import compare_document_data, evaluate_criteria
from verification_api.idempotency import claim_idempotency_key
from verification_api.diagnostics import (
    log_event,
    log_exception,
    new_correlation_id,
    request_correlation_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FellowShift")

# ==============================================================================
# APP SETUP
# ==============================================================================

app = FastAPI(
    title="FellowShift AI Verification API",
    description=(
        "Stateless scholarship eligibility screening and document extraction."
    ),
    version="2.0.0"
)

VERIFICATION_PATHS = {"/api/v1/verifications", "/api/scholarship/apply"}


@app.middleware("http")
async def log_verification_request(request: Request, call_next):
    if request.url.path not in VERIFICATION_PATHS:
        return await call_next(request)

    correlation_id = new_correlation_id()
    context_token = request_correlation_id.set(correlation_id)
    started_at = time.perf_counter()
    log_event(
        logger,
        logging.INFO,
        "[VERIFICATION]",
        "Request received",
        method=request.method,
    )
    log_event(
        logger,
        logging.INFO,
        "[VERIFICATION]",
        "Request validation started",
    )
    try:
        response = await call_next(request)
    except Exception as error:
        log_exception(
            logger,
            "[VERIFICATION]",
            "Unexpected request failure",
            error,
            error_category="UNHANDLED_REQUEST_EXCEPTION",
        )
        log_event(
            logger,
            logging.ERROR,
            "[VERIFICATION]",
            "Request completed",
            http_status=500,
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        raise
    else:
        log_event(
            logger,
            logging.INFO,
            "[VERIFICATION]",
            "Request completed",
            http_status=response.status_code,
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        return response
    finally:
        request_correlation_id.reset(context_token)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    if request.url.path in VERIFICATION_PATHS:
        log_event(
            logger,
            logging.WARNING,
            "[VERIFICATION]",
            "Request validation failed",
            error_category="INVALID_REQUEST",
            http_status=422,
        )
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "INVALID_REQUEST",
                "message": "The request does not match the endpoint contract.",
            }
        },
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    if request.url.path in VERIFICATION_PATHS and exc.status_code == 422:
        detail = exc.detail
        error_category = (
            detail.get("error", "INVALID_REQUEST")
            if isinstance(detail, dict)
            else "INVALID_REQUEST"
        )
        if not isinstance(error_category, str) or not error_category.isupper():
            error_category = "INVALID_REQUEST"
        log_event(
            logger,
            logging.WARNING,
            "[VERIFICATION]",
            "Request validation failed",
            error_category=error_category,
            http_status=422,
        )
    return await http_exception_handler(request, exc)

# Scholarship criteria evaluation lives in verification_api.service.

# ==============================================================================
# ROOT
# ==============================================================================

@app.get("/")
def read_root():
    logger.info("🚀 FellowShift API Root URL was called!")
    ready = (
        len(os.getenv("FELLOWSHIFT_SERVICE_TOKEN", "")) >= 32
        and bool(os.getenv("MISTRAL_API_KEY") or os.getenv("GEMINI_API_KEY"))
    )
    return {
        "status": "ready" if ready else "configuration_required",
        "service": "FellowShift AI Verification API",
        "api_version": "v1",
    }


# ==============================================================================
# APPLICATION SUBMISSION
# ==============================================================================

@app.post(
    "/api/scholarship/apply",
    status_code=status.HTTP_201_CREATED,
    deprecated=True,
    dependencies=[Depends(require_service_auth)],
)
async def submit_application(
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=200,
    ),
    application_id: Optional[str] = Form(default=None, max_length=128),
    criteria_json: Optional[str] = Form(default=None, max_length=65536),
    student_name: str = Form(..., min_length=1, max_length=200),
    roll_number: str = Form(..., min_length=1, max_length=100),
    st_subcaste: str = Form(..., min_length=1, max_length=120),
    annual_income: float = Form(..., ge=0, allow_inf_nan=False),
    academic_percentage: float = Form(..., ge=0, le=100, allow_inf_nan=False),
    course_name: str = Form(..., min_length=1, max_length=200),
    institute_name: str = Form(..., min_length=1, max_length=200),

    caste_certificate: UploadFile = File(...),
    income_certificate: UploadFile = File(...),
    marksheet: UploadFile = File(...)
):

    """
    Student application submission.

    Pipeline:

    Legacy multipart upload
       ↓
    Runtime image optimization
       ↓
    Parallel Mistral document analysis
       ↓
    Gemini fallback for failed documents
       ↓
    Scholarship-specific criteria (when provided)

    This endpoint does not persist applications or make approval decisions.
    """

    # --------------------------------------------------------------------------
    # APPLICATION ID
    # --------------------------------------------------------------------------

    app_id = application_id or str(uuid4())
    try:
        raw_criteria = json.loads(criteria_json) if criteria_json else []
        if not isinstance(raw_criteria, list):
            raise ValueError
        scholarship_criteria = [
            EligibilityCriterion.model_validate(item)
            for item in raw_criteria
        ]
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_CRITERIA",
                "message": "criteria_json must be a JSON array of valid scholarship criteria.",
            },
        ) from None
    except ValidationError:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_CRITERIA",
                "message": "criteria_json contains an invalid scholarship criterion.",
            },
        ) from None
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_CRITERIA",
                "message": "criteria_json must be a JSON array of valid scholarship criteria.",
            },
        ) from None

    # --------------------------------------------------------------------------
    # READ FILES
    # --------------------------------------------------------------------------

    (
        (caste_original, caste_original_mime),
        (income_original, income_original_mime),
        (marks_original, marks_original_mime),
    ) = await asyncio.gather(
        read_and_validate_document(caste_certificate),
        read_and_validate_document(income_certificate),
        read_and_validate_document(marksheet),
    )

    log_event(
        logger,
        logging.INFO,
        "[IDEMPOTENCY]",
        "Idempotency check started",
    )
    try:
        claimed = await asyncio.to_thread(
            claim_idempotency_key,
            idempotency_key,
        )
    except (OSError, sqlite3.Error) as error:
        log_event(
            logger,
            logging.ERROR,
            "[IDEMPOTENCY]",
            "Idempotency check failed",
            error_type=type(error).__name__,
            error_category="IDEMPOTENCY_UNAVAILABLE",
            http_status=503,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "IDEMPOTENCY_UNAVAILABLE",
                "message": "Request deduplication is temporarily unavailable.",
            },
        ) from None
    if not claimed:
        log_event(
            logger,
            logging.WARNING,
            "[IDEMPOTENCY]",
            "Idempotency check failed",
            error_category="DUPLICATE_REQUEST",
            http_status=409,
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "DUPLICATE_REQUEST",
                "message": "This idempotency key was already used.",
            },
        )
    log_event(
        logger,
        logging.INFO,
        "[IDEMPOTENCY]",
        "Idempotency check succeeded",
    )

    # --------------------------------------------------------------------------
    # OPTIMIZE ALL IMAGES IN PARALLEL
    # --------------------------------------------------------------------------

    try:

        (
            caste_optimized,
            caste_mime
        ), (
            income_optimized,
            income_mime
        ), (
            marks_optimized,
            marks_mime
        ) = await asyncio.gather(

            asyncio.to_thread(
                optimize_image_for_ai,
                caste_original,
                caste_original_mime
            ),

            asyncio.to_thread(
                optimize_image_for_ai,
                income_original,
                income_original_mime
            ),

            asyncio.to_thread(
                optimize_image_for_ai,
                marks_original,
                marks_original_mime
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        log_exception(
            logger,
            "[IMAGE]",
            "Image optimization failed",
            error,
            error_category="IMAGE_PROCESSING_FAILED",
        )

        raise HTTPException(
            status_code=400,
            detail={"error": "IMAGE_PROCESSING_FAILED", "message": "The document image could not be prepared."},
        ) from None

    # --------------------------------------------------------------------------
    # PROCESS ALL THREE DOCUMENTS IN PARALLEL
    # --------------------------------------------------------------------------

    try:
        (
            caste_result,
            income_result,
            marks_result,
        ) = await asyncio.wait_for(
            asyncio.gather(
                process_document(caste_optimized, caste_mime, "Caste Certificate"),
                process_document(income_optimized, income_mime, "Income Certificate"),
                process_document(marks_optimized, marks_mime, "Marksheet"),
            ),
            timeout=get_ai_request_timeout_seconds(),
        )
    except asyncio.TimeoutError:
        log_event(
            logger,
            logging.ERROR,
            "[VERIFICATION]",
            "Request failed",
            error_type="TimeoutError",
            error_category="AI_PROVIDER_TIMEOUT",
            http_status=503,
        )
        return JSONResponse(
            status_code=503,
            content={
                "application_id": app_id,
                "outcome": "FAILED",
                "reasons": [
                    {
                        "code": "AI_PROVIDER_TIMEOUT",
                        "explanation": "Document verification exceeded the configured time limit.",
                    }
                ],
            },
        )
    except RuntimeError as error:
        log_exception(
            logger,
            "[VERIFICATION]",
            "Document verification failed",
            error,
            error_category="AI_PROVIDER_FAILURE",
            http_status=503,
        )
        return JSONResponse(
            status_code=503,
            content={
                "application_id": app_id,
                "outcome": "FAILED",
                "reasons": [
                    {
                        "code": "AI_PROVIDER_FAILURE",
                        "explanation": "Document verification could not be completed.",
                    }
                ],
            },
        )

    caste_doc_ai, caste_provider = caste_result
    income_doc_ai, income_provider = income_result
    marks_doc_ai, marks_provider = marks_result

    if scholarship_criteria:
        outcome, reasons = evaluate_criteria(
            {
                "annual_income": annual_income,
                "academic_percentage": academic_percentage,
                "category": st_subcaste,
                "student_name": student_name,
                "roll_number": roll_number,
                "course_name": course_name,
                "institute_name": institute_name,
            },
            scholarship_criteria,
        )
    else:
        outcome = VerificationOutcome.NEEDS_REVIEW
        reasons = []

    documents_for_review = (
        ("caste_certificate", caste_doc_ai, "ST_CASTE_CERT"),
        ("income_certificate", income_doc_ai, "INCOME_CERT"),
        ("marksheet", marks_doc_ai, "MARKSHEET"),
    )
    uncertain_documents = [
        document_name
        for document_name, document, expected_type in documents_for_review
        if (
            not document.is_valid_document
            or document.discrepancies_or_issues
            or (document.document_type or "").strip().upper() != expected_type
        )
    ]
    document_results = [
        DocumentVerification(
            document_type=document_type,
            provider=provider,
            extraction=document,
        )
        for document_type, document, provider in (
            ("ST_CASTE_CERT", caste_doc_ai, caste_provider),
            ("INCOME_CERT", income_doc_ai, income_provider),
            ("MARKSHEET", marks_doc_ai, marks_provider),
        )
    ]
    data_mismatches = compare_document_data(
        {
            "annual_income": annual_income,
            "academic_percentage": academic_percentage,
            "category": st_subcaste,
            "student_name": student_name,
        },
        document_results,
    )
    if not scholarship_criteria:
        reasons = [
            {
                "code": "MISSING_ELIGIBILITY_CRITERIA",
                "explanation": "No scholarship-specific criteria were supplied.",
            }
        ]
    if uncertain_documents and outcome != VerificationOutcome.INELIGIBLE:
        outcome = VerificationOutcome.NEEDS_REVIEW
        reasons = [
            *data_mismatches,
            {
                "code": "OCR_UNCERTAIN",
                "explanation": "Document readability or extracted content requires manual review.",
            }
        ]
    elif data_mismatches and outcome != VerificationOutcome.INELIGIBLE:
        outcome = VerificationOutcome.NEEDS_REVIEW
        reasons = data_mismatches

    confidence = None
    flags = [reason["explanation"] if isinstance(reason, dict) else reason.explanation for reason in reasons]
    passed_rules = [
        criterion.reason_code
        for criterion in scholarship_criteria
    ] if outcome == VerificationOutcome.ELIGIBLE else []
    score_type = "NOT_SCORED"

    # --------------------------------------------------------------------------
    # No application records or status are persisted by this service.

    # RESPONSE
    # --------------------------------------------------------------------------

    return {

        "message":
            "Screening complete; application approval remains outside this service.",

        "application_id":
            app_id,

        "verification_mode":
            "criteria_driven",

        "test_only":
            False,

        "screening_result": {

            "outcome":
                outcome.value,

            "status":
                outcome.value,

            "verification_mode":
                "criteria_driven",

            "test_only":
                False,

            "score_type":
                score_type,

            "ai_confidence_score":
                confidence,

            "flagged_issues":
                flags,

            "passed_checks":
                passed_rules,

            "providers_used": {

                "caste_certificate":
                    caste_provider,

                "income_certificate":
                    income_provider,

                "marksheet":
                    marks_provider
            },

            "reasons":
                reasons
        }
    }


# ==============================================================================
# ADMIN — ALL APPLICATIONS
# ==============================================================================

@app.get(
    "/api/admin/applications",
    dependencies=[Depends(require_service_auth)],
)
def get_all_applications():
    raise HTTPException(
        status_code=410,
        detail={
            "error": "APPLICATIONS_OWNED_BY_NEXTJS",
            "message": "Application records and statuses are managed by the main application.",
        },
    )


# ==============================================================================
# ADMIN — APPLICATION DETAILS
# ==============================================================================

@app.get(
    "/api/admin/applications/{application_id}",
    dependencies=[Depends(require_service_auth)],
)
def get_application_details(
    application_id: str
):
    raise HTTPException(
        status_code=410,
        detail={
            "error": "APPLICATIONS_OWNED_BY_NEXTJS",
            "message": "Application records and statuses are managed by the main application.",
        },
    )


# ==============================================================================
# ADMIN — OFFICER DECISION
# ==============================================================================

@app.patch(
    "/api/admin/applications/{application_id}/decision",
    dependencies=[Depends(require_service_auth)],
)
def officer_decision(
    application_id: str,
):
    raise HTTPException(
        status_code=410,
        detail={
            "error": "APPLICATION_STATUS_OWNED_BY_NEXTJS",
            "message": "This service does not make or persist application decisions.",
        },
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

from verification_api.routes import router as verification_router

app.include_router(verification_router)
app.state.document_processor = process_document
app.state.image_optimizer = optimize_image_for_ai

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )