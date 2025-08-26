import requests, json, os

host = os.getenv("HOST", "127.0.0.1")
port = os.getenv("PORT", "8000")
url = f"http://{host}:{port}/chat"

q = {"question": "이 자료의 핵심 요약은?"}
resp = requests.post(url, json=q, timeout=60)
print(resp.status_code, resp.text)
