import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY')
print("Key length:", len(api_key) if api_key else 0)

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

try:
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print("Success gemini-1.5-flash:", response.choices[0].message.content)
except Exception as e:
    print("Error gemini-1.5-flash:", e)

try:
    response = client.chat.completions.create(
        model="gemini-1.5-flash-8b",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print("Success gemini-1.5-flash-8b:", response.choices[0].message.content)
except Exception as e:
    print("Error gemini-1.5-flash-8b:", e)

