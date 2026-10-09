import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY nahi mili.")
    print("Check karo ki .env file mein API key hai.")
    exit()

print("🔑 API key found.")
print("\n🔍 Aapki API key ke liye available models:\n")

try:
    client = genai.Client(api_key=api_key)

    for model in client.models.list():
        if model.supported_actions and "generateContent" in model.supported_actions:
            print("✅", model.name)

except Exception as e:
    print("\n❌ Error:")
    print(e)