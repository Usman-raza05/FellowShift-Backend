import os
import json
import asyncio
import base64
from io import BytesIO
from typing import List, Optional
from enum import Enum
from datetime import datetime, timezone

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from mistralai.client import Mistral


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

load_dotenv()

# Default is REAL mode for safety.
# If VERIFICATION_MODE is missing from .env, the system will NOT enter test mode.
VERIFICATION_MODE = os.getenv(
    "VERIFICATION_MODE",
    "real"
).strip().lower()

if VERIFICATION_MODE not in {"test", "real"}:
    raise RuntimeError(
        "VERIFICATION_MODE must be 'test' or 'real'"
    )

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set in .env")


# ==============================================================================
# AI CLIENTS
# ==============================================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

mistral_client = Mistral(
    api_key=MISTRAL_API_KEY
)

# Primary vision model
MISTRAL_MODEL = "pixtral-12b-2409"

# Fallback model
GEMINI_MODEL = "gemini-3.5-flash"


# ==============================================================================
# APP SETUP
# ==============================================================================

app = FastAPI(
    title="ST Scholarship AI Verification Engine",
    description=(
        "Backend API for ST Student Scholarship Eligibility, "
        "Document OCR, and Govt Officer Workflows"
    ),
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# MOCK DATABASE
# ==============================================================================

# In production:
# Replace this with PostgreSQL + SQLAlchemy.
db_applications = {}


# ==============================================================================
# CONSTANTS
# ==============================================================================

MAX_IMAGE_WIDTH = 1400
JPEG_QUALITY = 85

MAX_INCOME_LIMIT = 250000.0
MIN_MARKS_LIMIT = 50.0


# ==============================================================================
# TEST DOCUMENT MARKERS
# ==============================================================================

# These markers are expected to appear in the AI-detected
# discrepancy/issue text of clearly labelled fictional/demo documents.
DEMO_MARKERS = (
    "sample",
    "demo",
    "fictional",
    "testing only",
    "for testing",
    "not valid for official use",
    "not an official",
    "test data",
    "software testing",
)


# ==============================================================================
# ENUMS
# ==============================================================================

class ApplicationStatus(str, Enum):
    PENDING_VERIFICATION = "PENDING_VERIFICATION"

    RECOMMENDED_FOR_APPROVAL = (
        "RECOMMENDED_FOR_APPROVAL"
    )

    FLAGGED_FOR_MANUAL_REVIEW = (
        "FLAGGED_FOR_MANUAL_REVIEW"
    )

    APPROVED_BY_OFFICER = (
        "APPROVED_BY_OFFICER"
    )

    REJECTED_BY_OFFICER = (
        "REJECTED_BY_OFFICER"
    )

    # Test-mode-only statuses
    TEST_ONLY_PASSED = "TEST_ONLY_PASSED"
    TEST_ONLY_FAILED = "TEST_ONLY_FAILED"


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class DocumentExtractionResult(BaseModel):
    is_valid_document: bool = False

    document_type: Optional[str] = None

    extracted_full_name: Optional[str] = None

    extracted_category_caste: Optional[str] = None

    extracted_annual_income: Optional[float] = None

    extracted_percentage: Optional[float] = None

    certificate_number: Optional[str] = None

    issuing_authority: Optional[str] = None

    discrepancies_or_issues: List[str] = Field(
        default_factory=list
    )


class OfficerDecisionRequest(BaseModel):
    status: ApplicationStatus
    officer_remarks: str


# ==============================================================================
# AI PROMPT
# ==============================================================================

DOCUMENT_EXTRACTION_PROMPT = """
You are a document OCR and structured data extraction assistant for a scholarship application testing system.

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

3. Set `is_valid_document` to true when the image is readable and its content is recognizable as the expected document category. Do not automatically mark a readable document invalid solely because it is a prototype, test fixture, or digitally generated example.

4. Do not infer that a document is government-issued, authentic, legally valid, or officially verified merely because its layout resembles a certificate.

5. Add `discrepancies_or_issues` only for relevant observations, such as:

   * Unreadable or missing key information
   * Wrong document category
   * Conflicting values
   * Visible signs of alteration
   * Explicitly stated missing information required for the document's intended purpose

6. Do not add generic warnings repeatedly. Do not invent missing seals, signatures, authorities, or document details.

7. A test/demo disclaimer is contextual information, not evidence of unreadability or fraud. Do not classify a document as unreadable merely because it contains that disclaimer. If authenticity is required, report that authenticity has not been verified.

8. Never fabricate extracted values. Return null for fields that are not present.

9. Return only valid JSON matching the supplied schema. Do not include Markdown fences or additional commentary.
"""


# ==============================================================================
# IMAGE OPTIMIZATION
# ==============================================================================

def optimize_image_for_ai(
    file_bytes: bytes,
    original_mime_type: Optional[str]
) -> tuple[bytes, str]:
    """
    Compresses/resizes an uploaded image entirely in memory.

    Original uploaded bytes are never modified.

    Returns:
        optimized_jpeg_bytes, "image/jpeg"
    """

    try:
        image = Image.open(
            BytesIO(file_bytes)
        )

        image.load()

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read uploaded image: {str(e)}"
        )

    # Handle images with transparency.
    if image.mode in ("RGBA", "LA", "P"):

        background = Image.new(
            "RGB",
            image.size,
            "white"
        )

        if image.mode == "P":
            image = image.convert("RGBA")

        if image.mode in ("RGBA", "LA"):

            background.paste(
                image,
                mask=image.getchannel("A")
            )

            image = background

        else:
            image = image.convert("RGB")

    else:
        image = image.convert("RGB")

    # Resize only when image is wider than target width.
    if image.width > MAX_IMAGE_WIDTH:

        new_height = int(
            image.height
            * (MAX_IMAGE_WIDTH / image.width)
        )

        image = image.resize(
            (
                MAX_IMAGE_WIDTH,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

    output = BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True
    )

    optimized_bytes = output.getvalue()

    return optimized_bytes, "image/jpeg"


# ==============================================================================
# MISTRAL PROCESSOR
# ==============================================================================

def _process_with_mistral_sync(
    file_bytes: bytes,
    mime_type: str
) -> DocumentExtractionResult:

    encoded_image = base64.b64encode(
        file_bytes
    ).decode("utf-8")

    image_data_url = (
        f"data:{mime_type};base64,{encoded_image}"
    )

    response = mistral_client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_EXTRACTION_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": image_data_url
                    }
                ]
            }
        ],
        max_tokens=1000,
        temperature=0
    )

    if not response.choices:
        raise RuntimeError(
            "Mistral returned no choices."
        )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Mistral returned an empty response."
        )

    content = content.strip()

    # Mistral may occasionally return markdown fences.
    if content.startswith("```"):

        content = content.replace(
            "```json",
            "",
            1
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

    data = json.loads(content)

    return DocumentExtractionResult.model_validate(
        data
    )


async def process_document_with_mistral(
    file_bytes: bytes,
    mime_type: str
) -> DocumentExtractionResult:

    try:

        return await asyncio.to_thread(
            _process_with_mistral_sync,
            file_bytes,
            mime_type
        )

    except Exception as e:

        raise RuntimeError(
            f"Mistral document processing failed: {str(e)}"
        )


# ==============================================================================
# GEMINI FALLBACK PROCESSOR
# ==============================================================================

def _process_with_gemini_sync(
    file_bytes: bytes,
    mime_type: str
) -> DocumentExtractionResult:

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=file_bytes,
                mime_type=mime_type
            ),
            DOCUMENT_EXTRACTION_PROMPT
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DocumentExtractionResult,
            temperature=0
        )
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return DocumentExtractionResult.model_validate_json(
        response.text
    )


async def process_document_with_gemini(
    file_bytes: bytes,
    mime_type: str
) -> DocumentExtractionResult:

    try:

        return await asyncio.to_thread(
            _process_with_gemini_sync,
            file_bytes,
            mime_type
        )

    except Exception as e:

        raise RuntimeError(
            f"Gemini document processing failed: {str(e)}"
        )


# ==============================================================================
# PRIMARY + FALLBACK PROCESSOR
# ==============================================================================

async def process_document(
    file_bytes: bytes,
    mime_type: str,
    document_name: str
) -> tuple[DocumentExtractionResult, str]:

    """
    Processing strategy:

    1. Mistral is attempted first.
    2. If Mistral fails, Gemini is used as fallback.
    3. Failure of one document does not automatically fail
       the other documents.
    """

    mistral_error = None

    # --------------------------------------------------------------------------
    # PRIMARY: MISTRAL
    # --------------------------------------------------------------------------

    try:

        result = await process_document_with_mistral(
            file_bytes,
            mime_type
        )

        return result, "MISTRAL"

    except Exception as e:

        mistral_error = str(e)

    # --------------------------------------------------------------------------
    # FALLBACK: GEMINI
    # --------------------------------------------------------------------------

    try:

        result = await process_document_with_gemini(
            file_bytes,
            mime_type
        )

        return result, "GEMINI_FALLBACK"

    except Exception as gemini_error:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": (
                    f"AI processing failed for "
                    f"{document_name}."
                ),
                "mistral_error": mistral_error,
                "gemini_error": str(gemini_error)
            }
        )


# ==============================================================================
# TEST MODE HELPERS
# ==============================================================================

def has_demo_marker(
    document: DocumentExtractionResult
) -> bool:
    """
    Checks whether the AI extraction issues contain
    an explicit demo/test marker.
    """

    text = " ".join(
        document.discrepancies_or_issues
    ).lower()

    return any(
        marker in text
        for marker in DEMO_MARKERS
    )


def has_required_extracted_fields(
    document: DocumentExtractionResult,
    expected_type: str
) -> bool:
    """
    Checks document type and required extracted fields.

    This does NOT authenticate the document.
    """

    actual_type = (
        document.document_type or ""
    ).strip().upper()

    if actual_type != expected_type:
        return False

    if not document.extracted_full_name:
        return False

    if expected_type == "ST_CASTE_CERT":

        return bool(
            document.extracted_category_caste
        )

    if expected_type == "INCOME_CERT":

        return (
            document.extracted_annual_income
            is not None
        )

    if expected_type == "MARKSHEET":

        return (
            document.extracted_percentage
            is not None
        )

    return False


# ==============================================================================
# REAL VERIFICATION / ELIGIBILITY ENGINE
# ==============================================================================

def evaluate_eligibility(
    income: float,
    marks_pct: float,
    stated_caste: str,
    caste_doc_analysis: DocumentExtractionResult,
    income_doc_analysis: DocumentExtractionResult,
    marksheet_doc_analysis: DocumentExtractionResult,
    max_income_limit: float = MAX_INCOME_LIMIT,
    min_marks_limit: float = MIN_MARKS_LIMIT
) -> tuple[
    ApplicationStatus,
    float,
    List[str],
    List[str]
]:

    flags = []
    eligibility_reasons = []

    confidence_score = 100.0

    # --------------------------------------------------------------------------
    # 1. ELIGIBILITY RULES
    # --------------------------------------------------------------------------

    if income > max_income_limit:

        flags.append(
            f"Income ₹{income:,.2f} exceeds max threshold "
            f"of ₹{max_income_limit:,.2f}."
        )

    else:

        eligibility_reasons.append(
            "Income within eligible limits."
        )

    if marks_pct < min_marks_limit:

        flags.append(
            f"Marks {marks_pct}% below minimum threshold "
            f"of {min_marks_limit}%."
        )

    else:

        eligibility_reasons.append(
            "Academic score satisfies criteria."
        )

    # --------------------------------------------------------------------------
    # 2. DOCUMENT LEGIBILITY
    # --------------------------------------------------------------------------

    for doc_name, doc_data in [
        ("Caste Cert", caste_doc_analysis),
        ("Income Cert", income_doc_analysis),
        ("Marksheet", marksheet_doc_analysis)
    ]:

        if not doc_data.is_valid_document:

            flags.append(
                f"{doc_name} is unreadable, invalid, "
                "or corrupted."
            )

            confidence_score -= 30.0

        if doc_data.discrepancies_or_issues:

            for issue in (
                doc_data.discrepancies_or_issues
            ):

                flags.append(
                    f"{doc_name} Flag: {issue}"
                )

            confidence_score -= 10.0

    # --------------------------------------------------------------------------
    # 3. CASTE CROSS-VERIFICATION
    # --------------------------------------------------------------------------

    if caste_doc_analysis.extracted_category_caste:

        extracted_caste = (
            caste_doc_analysis
            .extracted_category_caste
            .strip()
            .lower()
        )

        stated_caste_normalized = (
            stated_caste
            .strip()
            .lower()
        )

        if (
            "st" not in extracted_caste
            and stated_caste_normalized
            not in extracted_caste
        ):

            flags.append(
                f"Caste mismatch: Document states "
                f"'{caste_doc_analysis.extracted_category_caste}', "
                f"Form states '{stated_caste}'."
            )

            confidence_score -= 25.0

    # --------------------------------------------------------------------------
    # 4. INCOME CROSS-VERIFICATION
    # --------------------------------------------------------------------------

    if (
        income_doc_analysis
        .extracted_annual_income
        is not None
    ):

        if abs(
            income_doc_analysis.extracted_annual_income
            - income
        ) > 5000:

            flags.append(
                f"Income Mismatch: Form says ₹{income}, "
                f"Certificate says "
                f"₹{income_doc_analysis.extracted_annual_income}."
            )

            confidence_score -= 20.0

    # --------------------------------------------------------------------------
    # 5. FINAL RECOMMENDATION
    # --------------------------------------------------------------------------

    if (
        len(flags) == 0
        and confidence_score >= 85.0
    ):

        recommended_status = (
            ApplicationStatus.RECOMMENDED_FOR_APPROVAL
        )

    else:

        recommended_status = (
            ApplicationStatus.FLAGGED_FOR_MANUAL_REVIEW
        )

    return (
        recommended_status,
        max(0.0, confidence_score),
        flags,
        eligibility_reasons
    )


# ==============================================================================
# TEST MODE EVALUATION
# ==============================================================================

def evaluate_test_mode(
    income: float,
    marks_pct: float,
    stated_caste: str,
    student_name: str,
    caste_doc: DocumentExtractionResult,
    income_doc: DocumentExtractionResult,
    marksheet_doc: DocumentExtractionResult,
) -> tuple[
    ApplicationStatus,
    float,
    List[str],
    List[str]
]:

    """
    Test-mode evaluation.

    IMPORTANT:
    - This does not authenticate government documents.
    - This does not produce a real approval.
    - It only evaluates clearly labelled test/demo fixtures.
    """

    issues: List[str] = []
    passed_checks: List[str] = []

    documents = [
        (
            "Caste certificate",
            caste_doc,
            "ST_CASTE_CERT"
        ),
        (
            "Income certificate",
            income_doc,
            "INCOME_CERT"
        ),
        (
            "Marksheet",
            marksheet_doc,
            "MARKSHEET"
        ),
    ]

    # --------------------------------------------------------------------------
    # 1. TEST DOCUMENT MARKER + REQUIRED FIELDS
    # --------------------------------------------------------------------------

    for name, document, expected_type in documents:

        if not has_demo_marker(document):

            issues.append(
                f"{name}: expected a clearly labelled "
                "test/demo document."
            )

        else:

            passed_checks.append(
                f"{name}: test/demo marker detected."
            )

        if not has_required_extracted_fields(
            document,
            expected_type
        ):

            issues.append(
                f"{name}: required fields could not be "
                "extracted or document type did not match."
            )

        else:

            passed_checks.append(
                f"{name}: document type and required "
                "fields extracted successfully."
            )

    # --------------------------------------------------------------------------
    # 2. STUDENT NAME CROSS-CHECK
    # --------------------------------------------------------------------------

    submitted_name = " ".join(
        student_name.casefold().split()
    )

    for name, document, _ in documents:

        extracted_name = (
            document.extracted_full_name
        )

        if extracted_name:

            normalized_name = " ".join(
                extracted_name.casefold().split()
            )

            if normalized_name != submitted_name:

                issues.append(
                    f"{name}: extracted student name "
                    "does not match the submitted name."
                )

            else:

                passed_checks.append(
                    f"{name}: student name matches."
                )

    # --------------------------------------------------------------------------
    # 3. INCOME CROSS-CHECK
    # --------------------------------------------------------------------------

    extracted_income = (
        income_doc.extracted_annual_income
    )

    if extracted_income is not None:

        if abs(
            extracted_income - income
        ) > 5000:

            issues.append(
                "Income certificate value does not match "
                "the submitted income."
            )

        else:

            passed_checks.append(
                "Income form value matches the test document."
            )

    # --------------------------------------------------------------------------
    # 4. MARKS CROSS-CHECK
    # --------------------------------------------------------------------------

    extracted_marks = (
        marksheet_doc.extracted_percentage
    )

    if extracted_marks is not None:

        if abs(
            extracted_marks - marks_pct
        ) > 1.0:

            issues.append(
                "Marksheet percentage does not match "
                "the submitted academic percentage."
            )

        else:

            passed_checks.append(
                "Marks form value matches the test document."
            )

    # --------------------------------------------------------------------------
    # 5. CASTE CROSS-CHECK
    # --------------------------------------------------------------------------

    extracted_caste = (
        caste_doc.extracted_category_caste
    )

    if extracted_caste:

        normalized_document_caste = (
            extracted_caste.casefold().strip()
        )

        normalized_form_caste = (
            stated_caste.casefold().strip()
        )

        if (
            normalized_document_caste !=
            normalized_form_caste
        ):

            # Allow ST category wording such as
            # "Scheduled Tribe" vs "ST".
            if not (
                "st" in normalized_document_caste
                and "st" in normalized_form_caste
            ):

                issues.append(
                    "Caste/category in the test document "
                    "does not match the submitted form."
                )

            else:

                passed_checks.append(
                    "Caste/category matches the test form."
                )

        else:

            passed_checks.append(
                "Caste/category matches the test form."
            )

    # --------------------------------------------------------------------------
    # 6. TEST ELIGIBILITY THRESHOLDS
    # --------------------------------------------------------------------------

    if income > MAX_INCOME_LIMIT:

        issues.append(
            "Income exceeds the configured test threshold."
        )

    else:

        passed_checks.append(
            "Income eligibility test passed."
        )

    if marks_pct < MIN_MARKS_LIMIT:

        issues.append(
            "Marks are below the configured test threshold."
        )

    else:

        passed_checks.append(
            "Academic eligibility test passed."
        )

    # --------------------------------------------------------------------------
    # 7. FINAL TEST STATUS
    # --------------------------------------------------------------------------

    if issues:

        result_status = (
            ApplicationStatus.TEST_ONLY_FAILED
        )

    else:

        result_status = (
            ApplicationStatus.TEST_ONLY_PASSED
        )

    # --------------------------------------------------------------------------
    # 8. TEST CHECK SCORE
    # --------------------------------------------------------------------------

    # This is NOT an AI confidence score.
    total_checks = (
        len(passed_checks)
        + len(issues)
    )

    score = (
        100.0
        * len(passed_checks)
        / total_checks
        if total_checks
        else 0.0
    )

    return (
        result_status,
        score,
        issues,
        passed_checks
    )


# ==============================================================================
# ROOT
# ==============================================================================

@app.get("/")
def read_root():

    return {
        "message":
            "ST Scholarship AI Verification Backend Running",

        "verification_mode":
            VERIFICATION_MODE,

        "test_only":
            VERIFICATION_MODE == "test",

        "ai_primary":
            "Mistral Pixtral",

        "ai_fallback":
            "Gemini",

        "document_processing":
            "Parallel"
    }


# ==============================================================================
# APPLICATION SUBMISSION
# ==============================================================================

@app.post(
    "/api/scholarship/apply",
    status_code=status.HTTP_201_CREATED
)
async def submit_application(
    student_name: str = Form(...),
    roll_number: str = Form(...),
    st_subcaste: str = Form(...),
    annual_income: float = Form(...),
    academic_percentage: float = Form(...),
    course_name: str = Form(...),
    institute_name: str = Form(...),

    caste_certificate: UploadFile = File(...),
    income_certificate: UploadFile = File(...),
    marksheet: UploadFile = File(...)
):

    """
    Student application submission.

    Pipeline:

    Upload
       ↓
    Runtime image optimization
       ↓
    Parallel Mistral document analysis
       ↓
    Gemini fallback for failed documents
       ↓
    Test OR Real verification engine
       ↓
    Application stored in mock DB
    """

    # --------------------------------------------------------------------------
    # APPLICATION ID
    # --------------------------------------------------------------------------

    app_id = (
        f"ST2026-{len(db_applications) + 1001}"
    )

    # --------------------------------------------------------------------------
    # READ FILES
    # --------------------------------------------------------------------------

    caste_original = (
        await caste_certificate.read()
    )

    income_original = (
        await income_certificate.read()
    )

    marks_original = (
        await marksheet.read()
    )

    # --------------------------------------------------------------------------
    # BASIC FILE CHECK
    # --------------------------------------------------------------------------

    for file_obj, file_bytes in [
        (
            caste_certificate,
            caste_original
        ),
        (
            income_certificate,
            income_original
        ),
        (
            marksheet,
            marks_original
        )
    ]:

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail=f"{file_obj.filename} is empty."
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
                caste_certificate.content_type
            ),

            asyncio.to_thread(
                optimize_image_for_ai,
                income_original,
                income_certificate.content_type
            ),

            asyncio.to_thread(
                optimize_image_for_ai,
                marks_original,
                marksheet.content_type
            )
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Image optimization failed: {str(e)}"
            )
        )

    # --------------------------------------------------------------------------
    # PROCESS ALL THREE DOCUMENTS IN PARALLEL
    # --------------------------------------------------------------------------

    (
        caste_result,
        income_result,
        marks_result
    ) = await asyncio.gather(

        process_document(
            caste_optimized,
            caste_mime,
            "Caste Certificate"
        ),

        process_document(
            income_optimized,
            income_mime,
            "Income Certificate"
        ),

        process_document(
            marks_optimized,
            marks_mime,
            "Marksheet"
        )
    )

    caste_doc_ai, caste_provider = caste_result
    income_doc_ai, income_provider = income_result
    marks_doc_ai, marks_provider = marks_result

    # --------------------------------------------------------------------------
    # VERIFICATION ENGINE
    # --------------------------------------------------------------------------
    #
    # TEST MODE:
    #     evaluate_test_mode()
    #
    # REAL MODE:
    #     evaluate_eligibility()
    #
    # Existing endpoint and form fields remain unchanged.
    # --------------------------------------------------------------------------

    if VERIFICATION_MODE == "test":

        (
            rec_status,
            confidence,
            flags,
            passed_rules
        ) = evaluate_test_mode(

            income=annual_income,

            marks_pct=academic_percentage,

            stated_caste=st_subcaste,

            student_name=student_name,

            caste_doc=caste_doc_ai,

            income_doc=income_doc_ai,

            marksheet_doc=marks_doc_ai
        )

    else:

        (
            rec_status,
            confidence,
            flags,
            passed_rules
        ) = evaluate_eligibility(

            income=annual_income,

            marks_pct=academic_percentage,

            stated_caste=st_subcaste,

            caste_doc_analysis=caste_doc_ai,

            income_doc_analysis=income_doc_ai,

            marksheet_doc_analysis=marks_doc_ai
        )

    # --------------------------------------------------------------------------
    # SCORE TYPE
    # --------------------------------------------------------------------------

    score_type = (
        "TEST_CHECK_SCORE"
        if VERIFICATION_MODE == "test"
        else "RULE_BASED_VERIFICATION_SCORE"
    )

    # --------------------------------------------------------------------------
    # APPLICATION RECORD
    # --------------------------------------------------------------------------

    application_record = {

        "application_id":
            app_id,

        "verification_mode":
            VERIFICATION_MODE,

        "test_only":
            VERIFICATION_MODE == "test",

        "submission_date":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "student_info": {

            "name":
                student_name,

            "roll_number":
                roll_number,

            "subcaste":
                st_subcaste,

            "annual_income":
                annual_income,

            "academic_percentage":
                academic_percentage,

            "course":
                course_name,

            "institute":
                institute_name
        },

        "ai_verification": {

            "confidence_score":
                confidence,

            "score_type":
                score_type,

            "verification_mode":
                VERIFICATION_MODE,

            "test_only":
                VERIFICATION_MODE == "test",

            "recommended_status":
                rec_status,

            "passed_rules":
                passed_rules,

            "flagged_issues":
                flags,

            "providers_used": {

                "caste_certificate":
                    caste_provider,

                "income_certificate":
                    income_provider,

                "marksheet":
                    marks_provider
            },

            "extracted_document_data": {

                "caste_certificate":
                    caste_doc_ai.model_dump(),

                "income_certificate":
                    income_doc_ai.model_dump(),

                "marksheet":
                    marks_doc_ai.model_dump()
            }
        },

        "government_officer_review": {

            "current_status":
                rec_status,

            "reviewed_by":
                None,

            "officer_remarks":
                None,

            "updated_at":
                None
        }
    }

    # --------------------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------------------

    db_applications[
        app_id
    ] = application_record

    # --------------------------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------------------------

    return {

        "message":
            "Application submitted and screened successfully.",

        "application_id":
            app_id,

        "verification_mode":
            VERIFICATION_MODE,

        "test_only":
            VERIFICATION_MODE == "test",

        "screening_result": {

            "status":
                rec_status,

            "verification_mode":
                VERIFICATION_MODE,

            "test_only":
                VERIFICATION_MODE == "test",

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
            }
        }
    }


# ==============================================================================
# ADMIN — ALL APPLICATIONS
# ==============================================================================

@app.get("/api/admin/applications")
def get_all_applications(
    status_filter: Optional[ApplicationStatus] = None,
    min_confidence: Optional[float] = None
):

    results = list(
        db_applications.values()
    )

    if status_filter:

        results = [
            application
            for application in results
            if application[
                "government_officer_review"
            ][
                "current_status"
            ] == status_filter
        ]

    if min_confidence is not None:

        results = [
            application
            for application in results
            if application[
                "ai_verification"
            ][
                "confidence_score"
            ] >= min_confidence
        ]

    return {

        "total_count":
            len(results),

        "applications":
            results
    }


# ==============================================================================
# ADMIN — APPLICATION DETAILS
# ==============================================================================

@app.get(
    "/api/admin/applications/{application_id}"
)
def get_application_details(
    application_id: str
):

    if application_id not in db_applications:

        raise HTTPException(
            status_code=404,
            detail="Application not found."
        )

    return db_applications[
        application_id
    ]


# ==============================================================================
# ADMIN — OFFICER DECISION
# ==============================================================================

@app.patch(
    "/api/admin/applications/{application_id}/decision"
)
def officer_decision(
    application_id: str,
    payload: OfficerDecisionRequest
):

    if application_id not in db_applications:

        raise HTTPException(
            status_code=404,
            detail="Application not found."
        )

    app_record = (
        db_applications[application_id]
    )

    app_record[
        "government_officer_review"
    ][
        "current_status"
    ] = payload.status

    app_record[
        "government_officer_review"
    ][
        "officer_remarks"
    ] = payload.officer_remarks

    app_record[
        "government_officer_review"
    ][
        "reviewed_by"
    ] = "Govt_Officer_ID_101"

    app_record[
        "government_officer_review"
    ][
        "updated_at"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    return {

        "message":
            f"Application {application_id} updated "
            f"to {payload.status.value}",

        "record":
            app_record
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )