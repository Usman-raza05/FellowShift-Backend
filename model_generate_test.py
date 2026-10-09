import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY nahi mili.")
    exit()

client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

print("\n🚀 Gemini Model Test")
print("=" * 60)

for model_name in models_to_test:

    print(f"\n🔍 Testing: {model_name}")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Hello! Reply with exactly: Hello, I am working."
        )

        print("✅ WORKING")
        print("Response:", response.text)

    except Exception as e:
        print("❌ FAILED")
        print("Error:", str(e)[:500])

print("\n" + "=" * 60)
print("🏁 Test completed.")