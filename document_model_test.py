import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY nahi mili.")
    exit()

client = genai.Client(api_key=api_key)

# Models jo Hello World test mein successfully kaam kar rahe the
models_to_test = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]

# Test documents
documents = {
    "Caste Certificate": "test_documents/caste_certificate.png",
    "Income Certificate": "test_documents/income_certificate.png",
    "Marksheet": "test_documents/marksheet.png",
}

prompt = """
You are testing an AI document verification system.

Analyze this document image carefully.

This is ONLY a fictional software-testing document, not a real
government or academic document.

Extract the following information if visible:

1. Is this a readable document?
2. What type of document is this?
3. Student name
4. Caste/category
5. Annual income
6. Academic percentage
7. Certificate/document number
8. Issuing authority
9. Any suspicious, missing, inconsistent, or unreadable information

Return a simple response with these fields:

Document Valid/Readable:
Document Type:
Student Name:
Caste/Category:
Annual Income:
Academic Percentage:
Certificate Number:
Issuing Authority:
Issues:
"""

print("\n")
print("=" * 80)
print("🚀 GEMINI DOCUMENT MODEL COMPARISON TEST")
print("=" * 80)

for document_name, image_path in documents.items():

    print("\n")
    print("#" * 80)
    print(f"📄 DOCUMENT: {document_name}")
    print(f"📁 FILE: {image_path}")
    print("#" * 80)

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        continue

    try:
        image = Image.open(image_path)

        print(f"📐 Image size: {image.size}")
        print(f"🖼️ Image format: {image.format}")

    except Exception as e:
        print(f"❌ Could not open image: {e}")
        continue

    for model_name in models_to_test:

        print("\n" + "-" * 80)
        print(f"🔍 Testing model: {model_name}")
        print("-" * 80)

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    image
                ]
            )

            print("✅ SUCCESS")
            print("\nAI RESPONSE:")
            print(response.text)

        except Exception as e:

            print("❌ FAILED")
            print("Error:")
            print(str(e)[:1000])

print("\n")
print("=" * 80)
print("🏁 ALL DOCUMENT TESTS COMPLETED")
print("=" * 80)