from enum import Enum
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid_document: bool = False
    document_type: str | None = None
    extracted_full_name: str | None = None
    extracted_category_caste: str | None = None
    extracted_annual_income: float | None = Field(default=None, ge=0)
    extracted_percentage: float | None = Field(default=None, ge=0, le=100)
    certificate_number: str | None = None
    issuing_authority: str | None = None
    discrepancies_or_issues: list[str] = Field(default_factory=list)


class CriterionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"


class VerificationOutcome(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


class EligibilityCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    operator: CriterionOperator
    value: Any
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    explanation: str = Field(min_length=1, max_length=500)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: Any) -> Any:
        allowed = (str, int, float, bool)
        if isinstance(value, list):
            if not value or any(not isinstance(item, allowed) for item in value):
                raise ValueError("value lists must contain supported scalar values")
            if any(isinstance(item, float) and not math.isfinite(item) for item in value):
                raise ValueError("numeric values must be finite")
        elif not isinstance(value, allowed):
            raise ValueError("value must be a string, number, boolean, or scalar list")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric values must be finite")
        return value

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("explanation must not be blank")
        return value

    @field_validator("value")
    @classmethod
    def validate_operator_value(cls, value: Any, info: Any) -> Any:
        operator = info.data.get("operator")
        numeric_operators = {
            CriterionOperator.LT,
            CriterionOperator.LTE,
            CriterionOperator.GT,
            CriterionOperator.GTE,
        }
        if operator in numeric_operators:
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(item, (int, float)) or isinstance(item, bool)
                   for item in values):
                raise ValueError("ordering operators require numeric values")
        if operator in {CriterionOperator.IN, CriterionOperator.NOT_IN} and not isinstance(value, list):
            raise ValueError("in/not_in operators require a scalar list")
        if operator not in {CriterionOperator.IN, CriterionOperator.NOT_IN} and isinstance(value, list):
            raise ValueError("only in/not_in operators accept a list")
        return value


class VerificationReason(BaseModel):
    code: str
    explanation: str


class DocumentVerification(BaseModel):
    document_type: str
    provider: str
    extraction: DocumentExtractionResult


class VerificationResponse(BaseModel):
    application_id: str = Field(min_length=1, max_length=128)
    outcome: VerificationOutcome
    reasons: list[VerificationReason] = Field(default_factory=list)
    documents: list[DocumentVerification] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    error: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail
