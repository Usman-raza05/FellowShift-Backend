import base64
import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from verification_api import providers
from verification_api.schemas import DocumentExtractionResult


class ProviderRegressionTests(unittest.TestCase):
    @staticmethod
    def synthetic_image(image_format: str) -> bytes:
        image_bytes = BytesIO()
        Image.new("RGB", (4, 4), color="white").save(
            image_bytes,
            format=image_format,
        )
        return image_bytes.getvalue()

    @staticmethod
    def extraction_response(**extra_fields: object) -> str:
        result = {
            "is_valid_document": True,
            "document_type": "INCOME_CERT",
            "extracted_full_name": None,
            "extracted_category_caste": None,
            "extracted_annual_income": 12000,
            "extracted_percentage": None,
            "certificate_number": None,
            "issuing_authority": None,
            "discrepancies_or_issues": [],
            **extra_fields,
        }
        return json.dumps(result)

    def test_mistral_keeps_historical_data_url_and_ignores_unknown_output_fields(self) -> None:
        image_bytes = self.synthetic_image("PNG")
        observed: dict[str, object] = {}

        def complete(**kwargs: object) -> SimpleNamespace:
            observed.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=self.extraction_response(
                                model_comment="ignored as by the original Pydantic model"
                            )
                        )
                    )
                ]
            )

        client = SimpleNamespace(chat=SimpleNamespace(complete=complete))
        with patch.object(
            providers,
            "_get_mistral_client",
            return_value=client,
        ):
            result = providers._process_with_mistral_sync(
                image_bytes,
                "image/png",
            )

        self.assertIsInstance(result, DocumentExtractionResult)
        self.assertEqual(result.extracted_annual_income, 12000)
        self.assertEqual(observed["model"], providers.MISTRAL_MODEL)
        image_chunk = observed["messages"][0]["content"][1]
        self.assertEqual(image_chunk["type"], "image_url")
        self.assertEqual(
            image_chunk["image_url"],
            "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        )

    def test_gemini_uses_historical_compatible_schema_and_validates_output(self) -> None:
        mime_types = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }

        for image_format, mime_type in mime_types.items():
            with self.subTest(image_format=image_format):
                image_bytes = self.synthetic_image(image_format)
                observed: dict[str, object] = {}

                def generate_content(**kwargs: object) -> SimpleNamespace:
                    observed.update(kwargs)
                    return SimpleNamespace(
                        text=self.extraction_response(
                            model_comment="ignored as by the original Pydantic model"
                        )
                    )

                client = SimpleNamespace(
                    models=SimpleNamespace(generate_content=generate_content)
                )
                with patch.object(
                    providers,
                    "_get_gemini_client",
                    return_value=client,
                ):
                    result = providers._process_with_gemini_sync(
                        image_bytes,
                        mime_type,
                    )

                self.assertIsInstance(result, DocumentExtractionResult)
                self.assertEqual(result.extracted_annual_income, 12000)
                self.assertEqual(observed["model"], providers.GEMINI_MODEL)
                part = observed["contents"][0]
                self.assertEqual(part.inline_data.data, image_bytes)
                self.assertEqual(part.inline_data.mime_type, mime_type)

                schema_model = observed["config"].response_schema
                self.assertIs(schema_model, providers._GeminiExtractionSchema)
                schema = schema_model.model_json_schema()
                self.assertNotIn("additionalProperties", schema)
                annual_income_schema = schema["properties"]["extracted_annual_income"]
                self.assertNotIn("minimum", str(annual_income_schema))
                percentage_schema = schema["properties"]["extracted_percentage"]
                self.assertNotIn("minimum", str(percentage_schema))
                self.assertNotIn("maximum", str(percentage_schema))

    def test_mistral_still_rejects_out_of_range_extracted_values(self) -> None:
        def complete(**kwargs: object) -> SimpleNamespace:
            _ = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=self.extraction_response(
                                extracted_annual_income=-1
                            )
                        )
                    )
                ]
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(complete=complete)
        )
        with patch.object(
            providers,
            "_get_mistral_client",
            return_value=client,
        ):
            with self.assertRaises(ValidationError):
                providers._process_with_mistral_sync(
                    self.synthetic_image("JPEG"),
                    "image/jpeg",
                )


if __name__ == "__main__":
    unittest.main()
