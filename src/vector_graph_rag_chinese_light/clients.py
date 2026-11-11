"""HTTP adapters for local Chroma and an OpenAI-compatible chat endpoint."""
import json
from urllib.request import Request, urlopen

COLLECTIONS = "/api/v2/tenants/default_tenant/databases/default_database/collections"


def request_json(url, body=None, headers=None, timeout=25):
    request = Request(
        url,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Chroma:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def __call__(self, path, body):
        return request_json(self.url + path, body, timeout=30)

    def collection(self, name):
        result = self(COLLECTIONS, {
            "name": name,
            "get_or_create": True,
            "configuration": {"hnsw": {"space": "cosine"}},
        })
        return f"{COLLECTIONS}/{result['id']}"


class Chat:
    def __init__(self, key, model, url="https://api.siliconflow.cn/v1/chat/completions"):
        self.key = key
        self.model = model
        self.url = url

    def __call__(self, system, data, max_tokens, timeout=25):
        response = request_json(self.url, {
            "model": self.model,
            "enable_thinking": False,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
            ],
        }, {"Authorization": f"Bearer {self.key}"}, timeout=timeout)
        answer = response["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty model response")
        return answer.strip()
