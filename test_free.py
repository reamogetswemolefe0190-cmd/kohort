import os
import json
import urllib.request

api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get('OPENROUTER_API_KEY')

req = urllib.request.Request('https://openrouter.ai/api/v1/models')
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read())

free_models = []
for m in data['data']:
    pricing = m.get('pricing', {})
    if pricing.get('prompt') == '0' and pricing.get('completion') == '0':
        free_models.append(m['id'])

working_models = []
print(f"Found {len(free_models)} free models. Testing them...")

for model_id in free_models:
    print(f"Testing {model_id}...", end=" ")
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps({
            'model': model_id,
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'max_tokens': 10
        }).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                print("OK")
                working_models.append(model_id)
                if len(working_models) >= 2:
                    break
    except Exception as e:
        print(f"Error: {e}")

if working_models:
    print(f"\nWorking free models: {working_models}")
    with open('working_free.txt', 'w') as f:
        f.write(','.join(working_models))
else:
    print("No working free models found.")
