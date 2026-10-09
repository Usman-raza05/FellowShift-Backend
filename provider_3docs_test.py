import os
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS = [
    {
        "name": "Caste Certificate",
        "path": "test_documents/caste_certificate_optimized.jpg",
        "document_type": "caste certificate",
    },
    {
        "name": "Income Certificate",
        "path": "test_documents/income_certificate_optimized.jpg",
        "document_type": "income certificate",
    },
    {
        "name": "Marksheet",
        "path": "test_documents/marksheet_optimized.jpg",
        "document_type": "academic marksheet",
    },
]

PROMPT_TEMPLATE = """
Analyze this {document_type} image for software testing.

Extract the following information if visible:

- student name
- document type
- caste/category if applicable
- annual income if applicable
- percentage/marks if applicable
- roll number if applicable
- certificate/document number
- issuing authority

Return only JSON.
Do not provide explanations outside the JSON.
"""


def load_image(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def test_mistral(image_base64, document_type):
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        return False, 0, "MISTRAL_API_KEY not found"

    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = PROMPT_TEMPLATE.format(
        document_type=document_type
    )

    payload = {
        "model": "pixtral-12b-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1000,
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30,
        )

        elapsed = time.perf_counter() - start

        if response.ok:
            data = response.json()
            text = data["choices"][0]["message"]["content"]

            return True, elapsed, text

        return (
            False,
            elapsed,
            f"HTTP {response.status_code}: {response.text[:500]}",
        )

    except Exception as error:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(error)


def test_gemini(image_base64, document_type):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return False, 0, "GEMINI_API_KEY not found"

    model = "gemini-3.5-flash"

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    headers = {
        "Content-Type": "application/json",
    }

    prompt = PROMPT_TEMPLATE.format(
        document_type=document_type
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1000,
        },
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30,
        )

        elapsed = time.perf_counter() - start

        if response.ok:
            data = response.json()

            text = data["candidates"][0]["content"]["parts"][0]["text"]

            return True, elapsed, text

        return (
            False,
            elapsed,
            f"HTTP {response.status_code}: {response.text[:500]}",
        )

    except Exception as error:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(error)


def print_result(provider, document, success, elapsed, response):
    print("\n" + "-" * 70)
    print(f"{provider} → {document}")
    print("-" * 70)

    if success:
        print("✅ SUCCESS")
        print(f"⏱️  Time: {elapsed:.2f} seconds")

        preview = response.replace("\n", " ")

        if len(preview) > 350:
            preview = preview[:350] + "..."

        print(f"📄 Response: {preview}")

    else:
        print("❌ FAILED")
        print(f"⏱️  Time: {elapsed:.2f} seconds")
        print(f"Error: {response}")


def main():

    print("=" * 70)
    print("       GEMINI vs MISTRAL — 3 DOCUMENT BENCHMARK")
    print("=" * 70)

    print("\nUsing optimized JPEG documents:")
    print("• Caste Certificate")
    print("• Income Certificate")
    print("• Marksheet")

    results = {
        "Gemini": [],
        "Mistral": [],
    }

    # ---------------------------------------------------------
    # Test each document
    # ---------------------------------------------------------

    for document in DOCUMENTS:

        document_name = document["name"]
        document_path = document["path"]
        document_type = document["document_type"]

        print("\n\n" + "=" * 70)
        print(f"DOCUMENT: {document_name}")
        print("=" * 70)

        if not os.path.exists(document_path):
            print(f"❌ File not found: {document_path}")
            continue

        file_size = os.path.getsize(document_path)

        print(
            f"Image size: "
            f"{file_size / (1024 * 1024):.2f} MB"
        )

        print("Loading image...")
        image_base64 = load_image(document_path)

        # -----------------------------------------------------
        # Gemini
        # -----------------------------------------------------

        print("\nTesting Gemini...")

        success, elapsed, response = test_gemini(
            image_base64,
            document_type,
        )

        print_result(
            "Gemini",
            document_name,
            success,
            elapsed,
            response,
        )

        if success:
            results["Gemini"].append(elapsed)

        # -----------------------------------------------------
        # Mistral
        # -----------------------------------------------------

        print("\nTesting Mistral...")

        success, elapsed, response = test_mistral(
            image_base64,
            document_type,
        )

        print_result(
            "Mistral",
            document_name,
            success,
            elapsed,
            response,
        )

        if success:
            results["Mistral"].append(elapsed)

    # ---------------------------------------------------------
    # Final Results
    # ---------------------------------------------------------

    print("\n\n")
    print("=" * 70)
    print("                    FINAL RESULTS")
    print("=" * 70)

    print(
        f"\n{'Provider':<15}"
        f"{'Documents':<12}"
        f"{'Total Time':<15}"
        f"{'Average':<12}"
    )

    print("-" * 55)

    for provider in ["Gemini", "Mistral"]:

        times = results[provider]

        if times:

            total = sum(times)
            average = total / len(times)

            print(
                f"{provider:<15}"
                f"{len(times):<12}"
                f"{total:>8.2f}s       "
                f"{average:>7.2f}s"
            )

        else:

            print(
                f"{provider:<15}"
                f"0{'':<11}"
                f"FAILED"
            )

    # ---------------------------------------------------------
    # Winner
    # ---------------------------------------------------------

    valid_results = {
        provider: sum(times)
        for provider, times in results.items()
        if times
    }

    if valid_results:

        fastest = min(
            valid_results,
            key=valid_results.get,
        )

        print("\n" + "=" * 70)

        print(
            f"🏆 FASTEST OVERALL: "
            f"{fastest}"
        )

        print(
            f"Total processing time: "
            f"{valid_results[fastest]:.2f} seconds"
        )

        print("=" * 70)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()