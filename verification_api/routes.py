import asyncio
import json
import math
import re
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verification_api.documents import (
    MAX_DOCUMENTS,
    read_and_validate_document,
)
from verification_api.config import get_ai_request_timeout_seconds
from verification_api.idempotency import claim_idempotency_key
from verification_api.schemas import (
    DocumentExtractionResult,
    DocumentVerification,
    EligibilityCriterion,
    ErrorResponse,
    VerificationOutcome,
    VerificationReason,
    VerificationResponse,
)
from verification_api.security import require_service_auth
from verification_api.service import compare_document_data, evaluate_criteria


router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(require_service_auth)],
)


class VerificationSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str = Field(min_length=1, max_length=128)
    application_data: dict[str, Any] = Field(max_length=100)

def _parse_application_data(value: str) -> VerificationSubmission:
    if len(value) > 65536:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "APPLICATION_DATA_TOO_LARGE",
                "message": "application_data must be 64 KiB or smaller.",
            },
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail={"error": "INVALID_APPLICATION_DATA", "message": "application_data must be valid JSON."},
        ) from None
    try:
        submission = VerificationSubmission.model_validate(parsed)
    except ValidationError:
        raise HTTPException(
            status_code=422,
            detail={"error": "INVALID_APPLICATION_DATA", "message": "application_data has an invalid shape."},
        ) from None

    if any(
        not isinstance(item, (str, int, float, bool)) or item is None
        for item in submission.application_data.values()
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_APPLICATION_DATA",
                "message": "Application attributes must be scalar strings, numbers, or booleans.",
            },
        )
    if any(
        isinstance(item, float) and not math.isfinite(item)
        for item in submission.application_data.values()
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_APPLICATION_DATA",
                "message": "Application numeric attributes must be finite.",
            },
        )
    if any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", field)
        for field in submission.application_data
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_APPLICATION_DATA",
                "message": "Application attribute names must be simple identifiers.",
            },
        )
    return submission


def _parse_json_array(value: str, error_code: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail={"error": error_code, "message": "The supplied field must be a JSON array."},
        ) from None
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=422,
            detail={"error": error_code, "message": "The supplied field must be a JSON array."},
        )
    return parsed


def _parse_criteria(value: str) -> list[EligibilityCriterion]:
    items = _parse_json_array(value, "INVALID_CRITERIA")
    try:
        return [EligibilityCriterion.model_validate(item) for item in items]
    except ValidationError:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_CRITERIA",
                "message": "The supplied array contains invalid criteria.",
            },
        ) from None


def _parse_document_types(value: str, error_code: str) -> list[str]:
    items = _parse_json_array(value, error_code)
    document_types: list[str] = []
    for item in items:
        if not isinstance(item, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9 _-]{0,63}",
            item,
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": error_code,
                    "message": "The supplied array contains invalid items.",
                },
            )
        document_types.append(item)
    return document_types


def _response(
    application_id: str,
    outcome: VerificationOutcome,
    reasons: list[VerificationReason],
    documents: list[DocumentVerification] | None = None,
    http_status: int = status.HTTP_200_OK,
) -> JSONResponse:
    response = VerificationResponse(
        application_id=application_id,
        outcome=outcome,
        reasons=reasons,
        documents=documents or [],
    )
    return JSONResponse(
        status_code=http_status,
        content=response.model_dump(mode="json"),
    )


@router.post(
    "/verifications",
    response_model=VerificationResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": VerificationResponse},
    },
)
async def verify_application(
    request: Request,
    application_data: str = Form(...),
    criteria: str = Form(...),
    required_document_types: str = Form(...),
    document_types: str = Form(default="[]"),
    documents: list[UploadFile] = File(default=[]),
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=200,
    ),
) -> JSONResponse:
    if len(criteria) > 65536 or len(required_document_types) > 8192 or len(document_types) > 8192:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "REQUEST_METADATA_TOO_LARGE",
                "message": "Request metadata exceeds the configured size limit.",
            },
        )
    submission = _parse_application_data(application_data)
    parsed_criteria = _parse_criteria(criteria)
    if not parsed_criteria or len(parsed_criteria) > 100:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_CRITERIA",
                "message": "Between 1 and 100 scholarship criteria are required.",
            },
        )

    required_types = _parse_document_types(
        required_document_types,
        "INVALID_REQUIRED_DOCUMENT_TYPES",
    )
    uploaded_types = _parse_document_types(
        document_types,
        "INVALID_DOCUMENT_TYPES",
    )
    required_document_type_values = required_types
    uploaded_document_type_values = uploaded_types
    if not required_document_type_values:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_REQUIRED_DOCUMENT_TYPES",
                "message": "At least one required document type must be supplied.",
            },
        )

    if len(documents) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=413,
            detail={"error": "TOO_MANY_DOCUMENTS", "message": f"A maximum of {MAX_DOCUMENTS} documents is supported."},
        )
    if len(documents) != len(uploaded_document_type_values):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INVALID_DOCUMENT_METADATA",
                "message": "document_types must contain one type for each uploaded document, in upload order.",
            },
        )
    normalized_required_types = [
        _normalize_document_type(document_type)
        for document_type in required_document_type_values
    ]
    if len(set(normalized_required_types)) != len(normalized_required_types):
        raise HTTPException(
            status_code=422,
            detail={"error": "INVALID_REQUIRED_DOCUMENT_TYPES", "message": "Required document types must be unique."},
        )

    checked_documents: list[tuple[str, bytes, str]] = []
    for document_type, upload in zip(uploaded_document_type_values, documents):
        content, mime_type = await read_and_validate_document(upload)
        checked_documents.append((document_type, content, mime_type))

    try:
        claimed = await asyncio.to_thread(claim_idempotency_key, idempotency_key)
    except (OSError, sqlite3.Error):
        raise HTTPException(
            status_code=503,
            detail={"error": "IDEMPOTENCY_UNAVAILABLE", "message": "Request deduplication is temporarily unavailable."},
        ) from None

    if not claimed:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "DUPLICATE_REQUEST",
                "message": "This idempotency key was already used. Submit a new key for a new attempt.",
            },
        )

    uploaded_type_keys = {
        _normalize_document_type(document_type)
        for document_type in uploaded_document_type_values
    }
    missing_types = sorted(
        document_type
        for document_type in required_document_type_values
        if _normalize_document_type(document_type) not in uploaded_type_keys
    )
    if missing_types:
        return _response(
            submission.application_id,
            VerificationOutcome.NEEDS_REVIEW,
            [
                VerificationReason(
                    code="MISSING_REQUIRED_DOCUMENT",
                    explanation="Required document(s) were not supplied: " + ", ".join(missing_types),
                )
            ],
        )

    outcome, reasons = evaluate_criteria(
        submission.application_data,
        parsed_criteria,
    )
    if outcome == VerificationOutcome.INELIGIBLE:
        return _response(submission.application_id, outcome, reasons)
    if outcome == VerificationOutcome.NEEDS_REVIEW:
        return _response(submission.application_id, outcome, reasons)

    processor = getattr(request.app.state, "document_processor", None)
    optimizer = getattr(request.app.state, "image_optimizer", None)
    if processor is None or optimizer is None:
        return _response(
            submission.application_id,
            VerificationOutcome.FAILED,
            [
                VerificationReason(
                    code="VERIFICATION_UNAVAILABLE",
                    explanation="Document verification is not configured.",
                )
            ],
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        timeout_seconds = get_ai_request_timeout_seconds()
    except RuntimeError as error:
        return _response(
            submission.application_id,
            VerificationOutcome.FAILED,
            [
                VerificationReason(
                    code="INVALID_TIMEOUT_CONFIGURATION",
                    explanation=str(error),
                )
            ],
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    async def process_one(
        document_type: str,
        content: bytes,
        mime_type: str,
    ) -> DocumentVerification:
        optimized, optimized_mime = await asyncio.to_thread(
            optimizer,
            content,
            mime_type,
        )
        result, provider = await processor(
            optimized,
            optimized_mime,
            document_type,
        )
        extraction = DocumentExtractionResult.model_validate(
            result.model_dump() if hasattr(result, "model_dump") else result
        )
        return DocumentVerification(
            document_type=document_type,
            provider=provider,
            extraction=extraction,
        )

    try:
        processed_documents = await asyncio.wait_for(
            asyncio.gather(
                *(
                    process_one(document_type, content, mime_type)
                    for document_type, content, mime_type in checked_documents
                )
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return _response(
            submission.application_id,
            VerificationOutcome.FAILED,
            [
                VerificationReason(
                    code="AI_PROVIDER_TIMEOUT",
                    explanation="Document verification exceeded the configured time limit.",
                )
            ],
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception:
        return _response(
            submission.application_id,
            VerificationOutcome.FAILED,
            [
                VerificationReason(
                    code="AI_PROVIDER_FAILURE",
                    explanation="Document verification could not be completed.",
                )
            ],
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    unreadable = [
        document.document_type
        for document in processed_documents
        if not document.extraction.is_valid_document
        or document.extraction.discrepancies_or_issues
        or not _document_type_matches(
            document.extraction.document_type,
            document.document_type,
        )
    ]
    data_mismatches = compare_document_data(
        submission.application_data,
        processed_documents,
    )
    if data_mismatches and not unreadable:
        return _response(
            submission.application_id,
            VerificationOutcome.NEEDS_REVIEW,
            data_mismatches,
            processed_documents,
        )
    if unreadable:
        return _response(
            submission.application_id,
            VerificationOutcome.NEEDS_REVIEW,
            data_mismatches + [
                VerificationReason(
                    code="OCR_UNCERTAIN",
                    explanation=(
                        "Document readability, expected type, or extracted content "
                        "requires manual review for: " + ", ".join(unreadable)
                    ),
                )
            ],
            processed_documents,
        )

    return _response(
        submission.application_id,
        VerificationOutcome.ELIGIBLE,
        [
            VerificationReason(
                code="CRITERIA_MET",
                explanation="The supplied scholarship criteria were met; this is not an approval decision.",
            )
        ],
        processed_documents,
    )


def _document_type_matches(extracted_type: str | None, requested_type: str) -> bool:
    if not extracted_type:
        return False
    return _normalize_document_type(extracted_type) == _normalize_document_type(requested_type)


def _normalize_document_type(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
