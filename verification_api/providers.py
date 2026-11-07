import asyncio
import base64
import json
import logging
import os
import time
from io import BytesIO

from dotenv import load_dotenv
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from mistralai.client import Mistral
from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

from verification_api.config import get_model_name, get_provider_timeout_ms
from verification_api.diagnostics import log_event, log_exception
from verification_api.schemas import DocumentExtractionResult


logger = logging.getLogger(__name__)

load_dotenv()

MAX_IMAGE_WIDTH = 1400
JPEG_QUALITY = 85
MISTRAL_MODEL = get_model_name("MISTRAL_MODEL", "pixtral-12b-2409")
GEMINI_MODEL = get_model_name("GEMINI_MODEL", "gemini-3.5-flash")

_gemini_client: genai.Client | None = None
_mistral_client: Mistral | None = None

DOCUMENT_EXTRACTION_PROMPT = """
You are a document OCR and structured data extraction assistant for scholarship applications.

Analyze the uploaded document image and return accurate structured JSON matching the provided schema.

Instructions:

1. Identify the document type based on its visible content:
   * ST_CASTE_CERT
   * INCOME_CERT
   * MARKSHEET
   * UNKNOWN

2. Extract only information that is actually visible in the document:
   * Student's full name
   * Caste/category/community
   * Annual family income in INR
   * Academic percentage
   * Certificate or registration number
   * Issuing authority, if explicitly shown

3. Set `is_valid_document` to true when the image is readable and its content is recognizable as the expected document category. Do not mark a readable document invalid solely because it is a prototype, test fixture, or digitally generated example.

4. Do not infer that a document is government-issued, authentic, legally valid, or officially verified merely because its layout resembles a certificate.

5. Add `discrepancies_or_issues` only for relevant observations, such as unreadable or missing key information, wrong document category, conflicting values, visible alteration, or information explicitly missing for its intended purpose.

6. Do not add generic warnings repeatedly. Do not invent missing seals, signatures, authorities, or document details.

7. A test/demo disclaimer is contextual information, not evidence of unreadability or fraud. If authenticity is required, report only that authenticity has not been verified.

8. Never fabricate extracted values. Return null for fields that are not present.

9. Return only valid JSON matching the supplied schema. Do not include Markdown fences or additional commentary.
"""


class _GeminiExtractionSchema(BaseModel):
    is_valid_document: bool = False
    document_type: str | None = None
    extracted_full_name: str | None = None
    extracted_category_caste: str | None = None
    extracted_annual_income: float | None = None
    extracted_percentage: float | None = None
    certificate_number: str | None = None
    issuing_authority: str | None = None
    discrepancies_or_issues: list[str] = Field(default_factory=list)


def _get_mistral_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        log_event(
            logger,
            logging.INFO,
            "[MISTRAL]",
            "Provider initialization started",
        )
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            log_event(
                logger,
                logging.ERROR,
                "[MISTRAL]",
                "Provider initialization failed",
                error_type="RuntimeError",
                error_category="MISTRAL_NOT_CONFIGURED",
            )
            raise RuntimeError("Mistral is not configured.")
        try:
            _mistral_client = Mistral(
                api_key=api_key,
                timeout_ms=get_provider_timeout_ms(),
                retry_config=RetryConfig(
                    strategy="none",
                    backoff=BackoffStrategy(
                        initial_interval=0,
                        max_interval=0,
                        exponent=1,
                        max_elapsed_time=0,
                        jitter_ms=0,
                    ),
                    retry_connection_errors=False,
                ),
            )
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "[MISTRAL]",
                "Provider initialization failed",
                error_type=type(error).__name__,
                error_category="MISTRAL_INITIALIZATION_FAILED",
            )
            raise
        log_event(
            logger,
            logging.INFO,
            "[MISTRAL]",
            "Provider initialization succeeded",
        )
    return _mistral_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        log_event(
            logger,
            logging.INFO,
            "[GEMINI]",
            "Provider initialization started",
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            log_event(
                logger,
                logging.ERROR,
                "[GEMINI]",
                "Provider initialization failed",
                error_type="RuntimeError",
                error_category="GEMINI_NOT_CONFIGURED",
            )
            raise RuntimeError("Gemini is not configured.")
        try:
            _gemini_client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=get_provider_timeout_ms(),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "[GEMINI]",
                "Provider initialization failed",
                error_type=type(error).__name__,
                error_category="GEMINI_INITIALIZATION_FAILED",
            )
            raise
        log_event(
            logger,
            logging.INFO,
            "[GEMINI]",
            "Provider initialization succeeded",
        )
    return _gemini_client


def optimize_image_for_ai(
    file_bytes: bytes,
    original_mime_type: str | None,
) -> tuple[bytes, str]:
    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from None
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read uploaded image.",
        ) from None

    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, "white")
        if image.mode == "P":
            image = image.convert("RGBA")
        if image.mode in ("RGBA", "LA"):
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    if image.width > MAX_IMAGE_WIDTH:
        new_height = int(image.height * (MAX_IMAGE_WIDTH / image.width))
        image = image.resize(
            (MAX_IMAGE_WIDTH, new_height),
            Image.Resampling.LANCZOS,
        )

    output = BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
    )
    return output.getvalue(), "image/jpeg"


def _process_with_mistral_sync(
    file_bytes: bytes,
    mime_type: str,
) -> DocumentExtractionResult:
    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{encoded_image}"
    response = _get_mistral_client().chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DOCUMENT_EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": image_data_url},
                ],
            }
        ],
        max_tokens=1000,
        temperature=0,
    )

    if not response.choices:
        raise RuntimeError("Mistral returned no choices.")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Mistral returned an empty response.")

    content = content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "", 1).replace("```", "").strip()
    return DocumentExtractionResult.model_validate(
        json.loads(content),
        extra="ignore",
    )


async def process_document_with_mistral(
    file_bytes: bytes,
    mime_type: str,
) -> DocumentExtractionResult:
    started_at = time.perf_counter()
    log_event(
        logger,
        logging.INFO,
        "[MISTRAL]",
        "Request started",
    )
    try:
        result = await asyncio.to_thread(
            _process_with_mistral_sync,
            file_bytes,
            mime_type,
        )
    except Exception as error:
        log_exception(
            logger,
            "[MISTRAL]",
            "Request failed",
            error,
            error_category="MISTRAL_REQUEST_FAILED",
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        raise RuntimeError("Mistral document processing failed.") from None
    log_event(
        logger,
        logging.INFO,
        "[MISTRAL]",
        "Request completed successfully",
        duration_ms=round((time.perf_counter() - started_at) * 1000),
    )
    return result


def _process_with_gemini_sync(
    file_bytes: bytes,
    mime_type: str,
) -> DocumentExtractionResult:
    response = _get_gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            DOCUMENT_EXTRACTION_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_GeminiExtractionSchema,
            temperature=0,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return DocumentExtractionResult.model_validate_json(
        response.text,
        extra="ignore",
    )


async def process_document_with_gemini(
    file_bytes: bytes,
    mime_type: str,
) -> DocumentExtractionResult:
    started_at = time.perf_counter()
    log_event(
        logger,
        logging.INFO,
        "[GEMINI]",
        "Request started",
    )
    try:
        result = await asyncio.to_thread(
            _process_with_gemini_sync,
            file_bytes,
            mime_type,
        )
    except Exception as error:
        log_exception(
            logger,
            "[GEMINI]",
            "Request failed",
            error,
            error_category="GEMINI_REQUEST_FAILED",
            duration_ms=round((time.perf_counter() - started_at) * 1000),
        )
        raise RuntimeError("Gemini document processing failed.") from None
    log_event(
        logger,
        logging.INFO,
        "[GEMINI]",
        "Request completed successfully",
        duration_ms=round((time.perf_counter() - started_at) * 1000),
    )
    return result


async def process_document(
    file_bytes: bytes,
    mime_type: str,
    document_name: str,
) -> tuple[DocumentExtractionResult, str]:
    try:
        result = await process_document_with_mistral(file_bytes, mime_type)
        return result, "MISTRAL"
    except Exception as error:
        log_event(
            logger,
            logging.WARNING,
            "[GEMINI]",
            "Fallback reason",
            fallback_reason="MISTRAL_REQUEST_FAILED",
            error_type=type(error).__name__,
        )
        log_event(
            logger,
            logging.INFO,
            "[GEMINI]",
            "Fallback started",
            fallback_reason="MISTRAL_REQUEST_FAILED",
        )

    try:
        result = await process_document_with_gemini(file_bytes, mime_type)
        return result, "GEMINI_FALLBACK"
    except Exception as error:
        log_event(
            logger,
            logging.ERROR,
            "[VERIFICATION]",
            "All AI providers failed",
            error_type=type(error).__name__,
            error_category="ALL_PROVIDERS_FAILED",
        )
        raise RuntimeError("AI providers failed to process the document.") from None
