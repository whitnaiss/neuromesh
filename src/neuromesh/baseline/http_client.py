"""Minimal LLM clients used by the baseline CLI.

Standard library only, sequential, one retry on 429/5xx. They exist so the
baseline can run against any endpoint (vLLM, Ollama, a hosted API) before
Track B's production client lands; once it does, the CLI should switch to it.

* :class:`OpenAICompatibleClient` -- ``POST {base_url}/chat/completions``.
* :class:`OllamaNativeClient` -- ``POST {base_url}/api/chat`` with
  ``think: false``. Needed for Qwen3 on Ollama, whose OpenAI-compatible route
  ignores ``/no_think`` and spends the token budget on hidden reasoning.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from neuromesh.contracts import LLMRequest, LLMResponse


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 300.0,
        retries: int = 1,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout_s = timeout_s
        self.retries = retries

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode())

    def _payload(self, r: LLMRequest) -> dict:
        payload = {
            "model": self.model,
            "messages": r.messages,
            "max_tokens": r.max_tokens,
            "temperature": r.temperature,
        }
        if r.seed is not None:
            payload["seed"] = r.seed
        return payload

    def _parse(self, body: dict) -> tuple[str, int, int]:
        usage = body.get("usage") or {}
        text = body["choices"][0]["message"].get("content") or ""
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    def _one(self, r: LLMRequest) -> LLMResponse:
        payload = self._payload(r)
        last_error = ""
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                text, pt, ct = self._parse(self._post(payload))
                truncated = not text.strip() and ct >= r.max_tokens
                return LLMResponse(
                    agent_id=r.agent_id,
                    text=text,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    latency_s=time.perf_counter() - start,
                    error="empty reply: token budget used up (hidden reasoning?)"
                    if truncated else None,
                )
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code != 429 and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(2.0 * (attempt + 1))
        return LLMResponse(agent_id=r.agent_id, text="", error=last_error)

    def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return [self._one(r) for r in requests]


class OllamaNativeClient(OpenAICompatibleClient):
    """Ollama's native chat API with thinking switched off."""

    def __init__(self, base_url: str, model: str, think: bool = False, **kwargs) -> None:
        super().__init__(base_url, model, **kwargs)
        root = base_url.rstrip("/")
        root = root[: -len("/v1")] if root.endswith("/v1") else root
        self.url = root + "/api/chat"
        self.think = think

    def _payload(self, r: LLMRequest) -> dict:
        options: dict = {"temperature": r.temperature, "num_predict": r.max_tokens}
        if r.seed is not None:
            options["seed"] = r.seed
        return {"model": self.model, "messages": r.messages, "stream": False,
                "think": self.think, "options": options}

    def _parse(self, body: dict) -> tuple[str, int, int]:
        text = (body.get("message") or {}).get("content") or ""
        return text, int(body.get("prompt_eval_count", 0)), int(body.get("eval_count", 0))
