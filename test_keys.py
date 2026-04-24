import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

keys = os.getenv("GEMINI_API_KEYS", "").split(",")
model_name = os.getenv("GEMINI_MODEL_LITE", "gemini-3.1-flash-lite-preview")

print(f"Testing {len(keys)} keys...")

for i, k in enumerate(keys):
    k = k.strip()
    if not k: continue
    print(f"[{i+1}] Testing key: ...{k[-4:]}")
    try:
        client = genai.Client(api_key=k)
        resp = client.models.generate_content(
            model=model_name,
            contents="Say 'Key OK'"
        )
        print(f"   ✅ SUCCESS: {resp.text.strip()}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
