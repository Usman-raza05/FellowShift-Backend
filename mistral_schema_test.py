import os
import json
import time
import base64
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from PIL import Image
from mistralai.client import Mistral


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not found in .env")

client = Mistral(api_key=MISTRAL_API_KEY)

MODEL = "pixtral-12b-2409"

# Runtime optimization settings
MAX_WIDTH = 1400
JPEG_QUALITY = 85


# ============================================================
# EXACT BACKEND SCHEMA
# ============================================================

class DocumentExtractionResult(BaseModel):
    is_valid_document: bool = False
    document_type: str | None = None
    student_name: str | None = None
    caste_category: str | None = None
    annual_income: str | None = None
    percentage_marks: str | None = None
    roll_number: str | None = None
    certificate_document_number: str | None = None
    issuing_authority: str | None = None
    discrepancies: list[str] = Field(default_factory=list)


# ============================================================
# ORIGINAL DOCUMENTS
# ============================================================

DOCUMENTS = {
    "Caste Certificate": "test_documents/caste_certificate.png",
    "Income Certificate": "test_documents/income_certificate.png",
    "Marksheet": "test_documents/marksheet.png",
}


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are an AI document verification assistant for an Indian
government scholarship system.

Analyze the uploaded document and extract information carefully.

Return ONLY valid JSON.

Use EXACTLY these keys:

{
  "is_valid_document": true,
  "document_type": "...",
  "student_name": "...",
  "caste_category": "...",
  "annual_income": "...",
  "percentage_marks": "...",
  "roll_number": "...",
  "certificate_document_number": "...",
  "issuing_authority": "...",
  "discrepancies": []
}

Rules:

1. Do not invent information.
2. If a field does not exist in the document, return null.
3. Keep extracted values faithful to the document.
4. For percentage_marks, return the final percentage as a string.
   Example: "78.0%"
5. For annual_income, return the amount as a string.
   Example: "INR 180,000"
6. If the document is a sample/demo document, extract it normally.
7. is_valid_document should indicate whether the uploaded image
   appears to be a readable document of the expected type.
8. discrepancies should contain only actual visible inconsistencies
   or suspicious issues. Otherwise return [].
9. Do not add markdown.
10. Do not add explanations outside JSON.
"""


# ============================================================
# RUNTIME IMAGE OPTIMIZATION
# ============================================================

def optimize_image_in_memory(image_path: str):
    """
    Reads the original image from disk and optimizes it entirely
    in memory.

    IMPORTANT:
    - Original file is NOT modified.
    - No optimized image is saved.
    - Returns JPEG bytes ready for API upload.
    """

    with Image.open(image_path) as image:

        original_width, original_height = image.size

        # Convert to RGB because JPEG does not support RGBA directly
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize only if image is wider than MAX_WIDTH
        if image.width > MAX_WIDTH:

            ratio = MAX_WIDTH / image.width

            new_width = MAX_WIDTH
            new_height = int(image.height * ratio)

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )

        # Save optimized JPEG ONLY in RAM
        output = BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True
        )

        jpeg_bytes = output.getvalue()

    return (
        jpeg_bytes,
        original_width,
        original_height,
        len(jpeg_bytes)
    )


# ============================================================
# TEST ONE DOCUMENT
# ============================================================

def test_document(document_name: str, image_path: str):

    print("\n" + "=" * 70)
    print(f"MISTRAL SCHEMA TEST → {document_name}")
    print("=" * 70)

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None

    original_size = os.path.getsize(image_path)

    print(f"Original file : {original_size / (1024 * 1024):.2f} MB")

    # --------------------------------------------------------
    # Runtime optimization
    # --------------------------------------------------------

    try:

        (
            image_bytes,
            original_width,
            original_height,
            optimized_size
        ) = optimize_image_in_memory(image_path)

    except Exception as e:

        print(f"❌ Image optimization failed: {e}")
        return None

    print(
        f"Original size : "
        f"{original_width} × {original_height}"
    )

    print(
        f"Optimized RAM : "
        f"{optimized_size / 1024:.2f} KB"
    )

    reduction = (
        (1 - optimized_size / original_size) * 100
    )

    print(
        f"Compression   : "
        f"{reduction:.2f}% smaller"
    )

    print("Optimized image saved to disk: NO")

    # --------------------------------------------------------
    # Base64
    # --------------------------------------------------------

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    print("\nSending optimized image from RAM to Mistral...")

    start = time.perf_counter()

    try:

        response = client.chat.complete(
            model=MODEL,

            messages=[
                {
                    "role": "user",

                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": (
                                f"data:image/jpeg;base64,"
                                f"{image_base64}"
                            )
                        }
                    ]
                }
            ],

            max_tokens=1000,
            temperature=0
        )

        elapsed = time.perf_counter() - start

        raw_text = response.choices[0].message.content

        # Some SDK responses may return content as a list
        if isinstance(raw_text, list):

            raw_text = "".join(
                item.get("text", "")
                for item in raw_text
                if isinstance(item, dict)
            )

        raw_text = raw_text.strip()

        print(f"\n⏱️ AI response time: {elapsed:.2f}s")

        # ----------------------------------------------------
        # Remove markdown JSON fences if present
        # ----------------------------------------------------

        if raw_text.startswith("```"):

            raw_text = raw_text.replace(
                "```json",
                "",
                1
            )

            raw_text = raw_text.replace(
                "```",
                ""
            )

            raw_text = raw_text.strip()

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        data = json.loads(raw_text)

        print("\nRaw JSON:")
        print(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )
        )

        # ----------------------------------------------------
        # Pydantic validation
        # ----------------------------------------------------

        print(
            "\nValidating against "
            "DocumentExtractionResult..."
        )

        result = DocumentExtractionResult.model_validate(data)

        print(
            "✅ PYDANTIC VALIDATION SUCCESS"
        )

        print("\nNormalized result:")

        print(
            json.dumps(
                result.model_dump(),
                indent=2,
                ensure_ascii=False
            )
        )

        return elapsed

    except json.JSONDecodeError as e:

        elapsed = time.perf_counter() - start

        print("\n❌ INVALID JSON")
        print(f"⏱️ Time: {elapsed:.2f}s")
        print(f"Error: {e}")

        return None

    except Exception as e:

        elapsed = time.perf_counter() - start

        print("\n❌ MISTRAL REQUEST FAILED")
        print(f"⏱️ Time: {elapsed:.2f}s")
        print(f"Error: {e}")

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(" MISTRAL → RUNTIME IMAGE OPTIMIZATION + PYDANTIC TEST")
    print("=" * 70)

    print("\nOriginal images will NOT be modified.")
    print("Optimized JPEG exists only in RAM.\n")

    results = []

    for document_name, image_path in DOCUMENTS.items():

        elapsed = test_document(
            document_name,
            image_path
        )

        if elapsed is not None:

            results.append(
                (
                    document_name,
                    elapsed
                )
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if results:

        total = sum(
            elapsed
            for _, elapsed in results
        )

        average = total / len(results)

        print(
            f"Successful documents : "
            f"{len(results)}/3"
        )

        print(
            f"Total AI time        : "
            f"{total:.2f}s"
        )

        print(
            f"Average AI time      : "
            f"{average:.2f}s"
        )

        print("\nPer document:")

        for name, elapsed in results:

            print(
                f"  {name:<22} "
                f"{elapsed:.2f}s"
            )

    else:

        print("❌ No successful documents.")

    print("=" * 70)


if __name__ == "__main__":
    main()