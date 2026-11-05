from typing import Any
import math

from verification_api.schemas import (
    CriterionOperator,
    DocumentVerification,
    EligibilityCriterion,
    VerificationOutcome,
    VerificationReason,
)


def evaluate_criteria(
    application_data: dict[str, Any],
    criteria: list[EligibilityCriterion],
) -> tuple[VerificationOutcome, list[VerificationReason]]:
    missing: list[str] = []
    failed: list[VerificationReason] = []

    for criterion in criteria:
        if criterion.field not in application_data:
            missing.append(criterion.field)
            continue

        actual = application_data[criterion.field]
        expected = criterion.value
        operator = criterion.operator

        try:
            if operator == CriterionOperator.EQ:
                matched = _equal(actual, expected)
            elif operator == CriterionOperator.NE:
                matched = not _equal(actual, expected)
            elif operator == CriterionOperator.LT:
                matched = _numeric(actual) < _numeric(expected)
            elif operator == CriterionOperator.LTE:
                matched = _numeric(actual) <= _numeric(expected)
            elif operator == CriterionOperator.GT:
                matched = _numeric(actual) > _numeric(expected)
            elif operator == CriterionOperator.GTE:
                matched = _numeric(actual) >= _numeric(expected)
            elif operator == CriterionOperator.IN:
                matched = any(_equal(actual, item) for item in expected)
            else:
                matched = all(not _equal(actual, item) for item in expected)
        except (TypeError, ValueError):
            missing.append(criterion.field)
            continue

        if not matched:
            failed.append(
                VerificationReason(
                    code=criterion.reason_code,
                    explanation=criterion.explanation,
                )
            )

    if failed:
        return VerificationOutcome.INELIGIBLE, failed
    if missing:
        return VerificationOutcome.NEEDS_REVIEW, [
            VerificationReason(
                code="MISSING_ELIGIBILITY_DATA",
                explanation=(
                    "Required application data is missing or has an incompatible type: "
                    + ", ".join(sorted(set(missing)))
                ),
            )
        ]
    return VerificationOutcome.ELIGIBLE, []


def compare_document_data(
    application_data: dict[str, Any],
    documents: list[DocumentVerification],
) -> list[VerificationReason]:
    comparisons = (
        ("extracted_full_name", ("student_name", "name"), "student name"),
        ("extracted_category_caste", ("category", "caste", "st_subcaste"), "category"),
        ("extracted_annual_income", ("annual_income",), "annual income"),
        ("extracted_percentage", ("academic_percentage", "percentage"), "academic percentage"),
    )
    mismatches: list[VerificationReason] = []

    for document in documents:
        extracted = document.extraction
        for extraction_field, application_fields, label in comparisons:
            extracted_value = getattr(extracted, extraction_field)
            application_field = next(
                (field for field in application_fields if field in application_data),
                None,
            )
            if extracted_value is None or application_field is None:
                continue
            if not _same_value(extracted_value, application_data[application_field]):
                mismatches.append(
                    VerificationReason(
                        code="DOCUMENT_DATA_MISMATCH",
                        explanation=(
                            f"Extracted {label} in {document.document_type} does not match "
                            "the corresponding application field; manual review is required."
                        ),
                    )
                )
    return mismatches


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            return _numeric(left) == _numeric(right)
        return False
    if isinstance(left, str) and isinstance(right, str):
        return " ".join(left.casefold().split()) == " ".join(right.casefold().split())
    return type(left) is type(right) and left == right


def _numeric(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("numeric comparison requires numeric values")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("numeric comparison requires finite values")
    return number


def _equal(left: Any, right: Any) -> bool:
    numeric_types = (int, float)
    if isinstance(left, numeric_types) and not isinstance(left, bool):
        if isinstance(right, numeric_types) and not isinstance(right, bool):
            return _numeric(left) == _numeric(right)
        return False
    return type(left) is type(right) and left == right
