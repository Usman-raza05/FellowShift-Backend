import asyncio
import json
import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image

import main
from verification_api.schemas import DocumentExtractionResult


class VerificationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "FELLOWSHIFT_SERVICE_TOKEN": "test-token-with-at-least-32-characters",
                "IDEMPOTENCY_DB_PATH": os.path.join(self.temp_dir.name, "idempotency.sqlite3"),
                "AI_REQUEST_TIMEOUT_SECONDS": "1",
            },
        )
        self.environment.start()
        self.previous_processor = main.app.state.document_processor
        self.previous_optimizer = main.app.state.image_optimizer
        self.processed_types: list[str] = []
        main.app.state.document_processor = self.fake_processor
        main.app.state.image_optimizer = self.fake_optimizer
        self.client = TestClient(main.app)
        self.request_number = 0

    def tearDown(self) -> None:
        self.client.close()
        main.app.state.document_processor = self.previous_processor
        main.app.state.image_optimizer = self.previous_optimizer
        self.environment.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def fake_optimizer(content: bytes, mime_type: str) -> tuple[bytes, str]:
        return content, mime_type

    async def fake_processor(
        self,
        content: bytes,
        mime_type: str,
        document_type: str,
    ) -> tuple[DocumentExtractionResult, str]:
        self.processed_types.append(document_type)
        return (
            DocumentExtractionResult(
                is_valid_document=True,
                document_type=document_type,
            ),
            "MISTRAL",
        )

    @staticmethod
    def synthetic_png() -> bytes:
        output = BytesIO()
        Image.new("RGB", (20, 20), color="white").save(output, format="PNG")
        return output.getvalue()

    def post(
        self,
        *,
        criteria: list[dict] | None = None,
        application_data: dict | None = None,
        required_document_types: list[str] | None = None,
        document_types: list[str] | None = None,
        file_content: bytes | None = None,
        filename: str = "synthetic.png",
        content_type: str = "image/png",
        key: str | None = None,
        authorization: bool = True,
    ):
        self.request_number += 1
        headers = {"Idempotency-Key": key or f"request-key-{self.request_number}"}
        if authorization:
            headers["Authorization"] = "Bearer test-token-with-at-least-32-characters"

        data = {
            "application_data": json.dumps(
                {
                    "application_id": "synthetic-application",
                    "application_data": (
                        application_data
                        or {
                            "annual_income": 100,
                            "student_name": "Synthetic Applicant",
                        }
                    ),
                }
            ),
            "criteria": json.dumps(
                criteria
                or [
                    {
                        "field": "annual_income",
                        "operator": "lte",
                        "value": 1000,
                        "reason_code": "INCOME_LIMIT",
                        "explanation": "Income satisfies the synthetic test rule.",
                    }
                ]
            ),
            "required_document_types": json.dumps(
                required_document_types or ["income_certificate"]
            ),
            "document_types": json.dumps(document_types or []),
        }
        files = None
        if file_content is not None:
            data["document_types"] = json.dumps(document_types or ["income_certificate"])
            files = [
                (
                    "documents",
                    (filename, file_content, content_type),
                )
            ]
        return self.client.post(
            "/api/v1/verifications",
            headers=headers,
            data=data,
            files=files,
        )

    def test_valid_application_is_eligible_without_approval(self) -> None:
        response = self.post(
            required_document_types=["income_certificate"],
            document_types=["income_certificate"],
            file_content=self.synthetic_png(),
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["outcome"], "ELIGIBLE")
        self.assertIn("not an approval decision", body["reasons"][0]["explanation"])
        self.assertEqual(body["documents"][0]["provider"], "MISTRAL")

    def test_failed_criterion_is_ineligible(self) -> None:
        response = self.post(
            application_data={"annual_income": 400000},
            required_document_types=["income_certificate"],
            document_types=["income_certificate"],
            file_content=self.synthetic_png(),
            criteria=[
                {
                    "field": "annual_income",
                    "operator": "lte",
                    "value": 1000,
                    "reason_code": "SCHOLARSHIP_INCOME_LIMIT",
                    "explanation": "The synthetic test criterion was not met.",
                }
            ],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outcome"], "INELIGIBLE")
        self.assertEqual(self.processed_types, [])

    def test_unreadable_document_needs_review(self) -> None:
        async def unreadable_processor(content: bytes, mime_type: str, document_type: str):
            return (
                DocumentExtractionResult(
                    is_valid_document=False,
                    document_type=document_type,
                ),
                "MISTRAL",
            )

        main.app.state.document_processor = unreadable_processor
        response = self.post(
            required_document_types=["income_certificate"],
            document_types=["income_certificate"],
            file_content=self.synthetic_png(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outcome"], "NEEDS_REVIEW")
        self.assertEqual(response.json()["reasons"][0]["code"], "OCR_UNCERTAIN")

    def test_document_value_mismatch_needs_manual_review(self) -> None:
        async def mismatched_processor(content: bytes, mime_type: str, document_type: str):
            return (
                DocumentExtractionResult(
                    is_valid_document=True,
                    document_type=document_type,
                    extracted_annual_income=101,
                ),
                "MISTRAL",
            )

        main.app.state.document_processor = mismatched_processor
        response = self.post(
            application_data={"annual_income": 100},
            required_document_types=["income_certificate"],
            document_types=["income_certificate"],
            file_content=self.synthetic_png(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outcome"], "NEEDS_REVIEW")
        self.assertEqual(response.json()["reasons"][0]["code"], "DOCUMENT_DATA_MISMATCH")

    def test_missing_required_document_needs_review(self) -> None:
        response = self.post(required_document_types=["income_certificate"])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outcome"], "NEEDS_REVIEW")
        self.assertEqual(response.json()["reasons"][0]["code"], "MISSING_REQUIRED_DOCUMENT")

    def test_unsupported_file_is_rejected(self) -> None:
        response = self.post(
            required_document_types=["income_certificate"],
            document_types=["income_certificate"],
            file_content=b"not an image",
            filename="document.txt",
            content_type="text/plain",
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"]["error"], "UNSUPPORTED_FILE_TYPE")

    def test_invalid_criteria_is_rejected(self) -> None:
        response = self.post(
            criteria=[
                {
                    "field": "annual_income",
                    "operator": "lte",
                    "value": "not-a-number",
                    "reason_code": "BAD_RULE",
                    "explanation": "Invalid rule.",
                }
            ]
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"], "INVALID_CRITERIA")

    def test_provider_timeout_returns_failed_not_ineligible(self) -> None:
        async def slow_processor(content: bytes, mime_type: str, document_type: str):
            await asyncio.sleep(0.1)
            return (
                DocumentExtractionResult(is_valid_document=True, document_type=document_type),
                "MISTRAL",
            )

        main.app.state.document_processor = slow_processor
        with patch.dict(os.environ, {"AI_REQUEST_TIMEOUT_SECONDS": "0.01"}):
            response = self.post(
                required_document_types=["income_certificate"],
                document_types=["income_certificate"],
                file_content=self.synthetic_png(),
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["outcome"], "FAILED")
        self.assertEqual(response.json()["reasons"][0]["code"], "AI_PROVIDER_TIMEOUT")

    def test_provider_fallback_is_reported(self) -> None:
        extracted = DocumentExtractionResult(
            is_valid_document=True,
            document_type="income_certificate",
        )
        with (
            patch(
                "verification_api.providers.process_document_with_mistral",
                new=AsyncMock(side_effect=RuntimeError("provider detail stays private")),
            ),
            patch(
                "verification_api.providers.process_document_with_gemini",
                new=AsyncMock(return_value=extracted),
            ),
        ):
            result, provider = asyncio.run(
                main.process_document(b"synthetic", "image/jpeg", "income_certificate")
            )

        self.assertEqual(result, extracted)
        self.assertEqual(provider, "GEMINI_FALLBACK")

    def test_unauthorized_request_is_rejected(self) -> None:
        response = self.post(authorization=False)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"]["error"], "UNAUTHORIZED")

    def test_duplicate_request_is_not_processed_twice(self) -> None:
        key = "same-request-key"
        request = {
            "required_document_types": ["income_certificate"],
            "document_types": ["income_certificate"],
            "file_content": self.synthetic_png(),
            "key": key,
        }
        first = self.post(**request)
        second = self.post(**request)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.json()["detail"]["error"], "DUPLICATE_REQUEST")
        self.assertEqual(self.processed_types, ["income_certificate"])

    def test_response_does_not_echo_sensitive_application_data(self) -> None:
        response = self.post(
            application_data={
                "annual_income": 120000,
                "student_name": "Synthetic Private Name",
                "roll_number": "SYNTHETIC-ROLL-001",
            },
            required_document_types=["income_certificate"],
            document_types=["income_certificate"],
            file_content=self.synthetic_png(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Synthetic Private Name", response.text)
        self.assertNotIn("SYNTHETIC-ROLL-001", response.text)
        self.assertNotIn("student_name", response.text)
        self.assertFalse(hasattr(main, "db_applications"))

    def test_legacy_endpoint_uses_supplied_rules_without_persisting(self) -> None:
        extraction_results = [
            (
                DocumentExtractionResult(is_valid_document=True, document_type="ST_CASTE_CERT"),
                "MISTRAL",
            ),
            (
                DocumentExtractionResult(is_valid_document=True, document_type="INCOME_CERT"),
                "MISTRAL",
            ),
            (
                DocumentExtractionResult(is_valid_document=True, document_type="MARKSHEET"),
                "MISTRAL",
            ),
        ]
        data = {
            "application_id": "legacy-synthetic-application",
            "criteria_json": json.dumps(
                [
                    {
                        "field": "annual_income",
                        "operator": "lte",
                        "value": 1000,
                        "reason_code": "SYNTHETIC_LIMIT",
                        "explanation": "Synthetic test criterion.",
                    }
                ]
            ),
            "student_name": "Synthetic Applicant",
            "roll_number": "SYNTHETIC-ROLL",
            "st_subcaste": "Synthetic Category",
            "annual_income": "2000",
            "academic_percentage": "70",
            "course_name": "Synthetic Course",
            "institute_name": "Synthetic Institute",
        }
        files = [
            ("caste_certificate", ("caste.png", self.synthetic_png(), "image/png")),
            ("income_certificate", ("income.png", self.synthetic_png(), "image/png")),
            ("marksheet", ("marks.png", self.synthetic_png(), "image/png")),
        ]
        with patch(
            "main.process_document",
            new=AsyncMock(side_effect=extraction_results),
        ):
            response = self.client.post(
                "/api/scholarship/apply",
                headers={
                    "Authorization": "Bearer test-token-with-at-least-32-characters",
                    "Idempotency-Key": "legacy-request-key",
                },
                data=data,
                files=files,
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["screening_result"]["outcome"], "INELIGIBLE")
        self.assertFalse(hasattr(main, "db_applications"))

    def test_legacy_admin_status_api_is_retired(self) -> None:
        response = self.client.patch(
            "/api/admin/applications/app-123/decision",
            headers={"Authorization": "Bearer test-token-with-at-least-32-characters"},
            json={"status": "APPROVED_BY_OFFICER", "officer_remarks": "synthetic"},
        )

        self.assertEqual(response.status_code, 410)
        self.assertEqual(
            response.json()["detail"]["error"],
            "APPLICATION_STATUS_OWNED_BY_NEXTJS",
        )


if __name__ == "__main__":
    unittest.main()
