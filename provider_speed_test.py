import os
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

IMAGE_PATH = "test_documents/marksheet_optimized.jpg"

PROMPT = """
Analyze this document image for software testing.

Extract the following information if visible:
- student name
- document type
- percentage/marks
- roll number
- certificate/document number
- issuing authority

Return the extracted information in simple JSON.

Do not explain anything outside the JSON.
"""

def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_groq(image_base64):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return False, 0, "GROQ_API_KEY not found"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        elapsed = time.perf_counter() - start

        if response.ok:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return True, elapsed, text

        return False, elapsed, f"HTTP {response.status_code}: {response.text[:500]}"

    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(e)


def test_gemini(image_base64):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return False, 0, "GEMINI_API_KEY not found"

    model = "gemini-3.5-flash"

    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": PROMPT
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0
        }
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        elapsed = time.perf_counter() - start

        if response.ok:
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return True, elapsed, text

        return False, elapsed, f"HTTP {response.status_code}: {response.text[:500]}"

    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(e)


def test_mistral(image_base64):
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        return False, 0, "MISTRAL_API_KEY not found"

    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "pixtral-12b-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        elapsed = time.perf_counter() - start

        if response.ok:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return True, elapsed, text

        return False, elapsed, f"HTTP {response.status_code}: {response.text[:500]}"

    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(e)


def test_openrouter(image_base64):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return False, 0, "OPENROUTER_API_KEY not found"

    model = os.getenv(
        "OPENROUTER_MODEL",
        "google/gemini-2.5-flash"
    )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        elapsed = time.perf_counter() - start

        if response.ok:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return True, elapsed, text

        return False, elapsed, f"HTTP {response.status_code}: {response.text[:500]}"

    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(e)


def test_anthropic(image_base64):
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return False, 0, "ANTHROPIC_API_KEY not found"

    model = os.getenv(
        "ANTHROPIC_MODEL",
        "claude-sonnet-4-6"
    )

    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ]
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        elapsed = time.perf_counter() - start

        if response.ok:
            data = response.json()
            text = data["content"][0]["text"]
            return True, elapsed, text

        return False, elapsed, f"HTTP {response.status_code}: {response.text[:500]}"

    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(e)


def main():

    print("=" * 70)
    print("        AI PROVIDER SPEED BENCHMARK")
    print("=" * 70)

    print(f"\nImage: {IMAGE_PATH}")

    if not os.path.exists(IMAGE_PATH):
        print("\n❌ Image not found!")
        return

    print("Loading image...")
    image_base64 = load_image_base64(IMAGE_PATH)

    image_size_mb = os.path.getsize(IMAGE_PATH) / (1024 * 1024)

    print(f"Image size: {image_size_mb:.2f} MB")

    providers = [
        ("Groq", test_groq),
        ("Gemini", test_gemini),
        ("Mistral", test_mistral),
        ("OpenRouter", test_openrouter),
        ("Anthropic", test_anthropic),
    ]

    results = []

    for name, function in providers:

        print("\n" + "-" * 70)
        print(f"Testing: {name}")
        print("-" * 70)

        success, elapsed, response = function(image_base64)

        if success:
            print(f"✅ SUCCESS")
            print(f"⏱️  Response time: {elapsed:.2f} seconds")

            preview = response.replace("\n", " ")

            if len(preview) > 300:
                preview = preview[:300] + "..."

            print(f"📄 Response: {preview}")

            results.append({
                "name": name,
                "success": True,
                "time": elapsed
            })

        else:
            print(f"❌ FAILED")
            print(f"⏱️  Time: {elapsed:.2f} seconds")
            print(f"Error: {response}")

            results.append({
                "name": name,
                "success": False,
                "time": elapsed
            })

    successful = [
        r for r in results
        if r["success"]
    ]

    print("\n")
    print("=" * 70)
    print("                    FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Provider':<15} {'Status':<12} {'Time':>10}")
    print("-" * 40)

    for result in results:

        status = "SUCCESS" if result["success"] else "FAILED"

        print(
            f"{result['name']:<15} "
            f"{status:<12} "
            f"{result['time']:>8.2f}s"
        )

    if successful:

        fastest = min(
            successful,
            key=lambda x: x["time"]
        )

        print("\n" + "=" * 70)
        print(
            f"🏆 FASTEST: {fastest['name']} "
            f"({fastest['time']:.2f}s)"
        )
        print("=" * 70)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()