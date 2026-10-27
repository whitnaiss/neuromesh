# LLM endpoint contract (draft)

Owner: Andrew (Track D) · Approver: Armman (Track B) · Due: Oct 23 → to finalize

Both sides build and test against this document and `tests/fixtures/llm/contract_examples.json`.

- **API:** OpenAI-compatible `POST /v1/chat/completions`, health check `GET /v1/models`
- **Request:** `model`, `messages`, `temperature`, `max_tokens`, `seed`, JSON-schema output constraint
  (exact field name: TODO — depends on the deployed server version)
- **Response:** `choices[0].message.content`, `usage.prompt_tokens`, `usage.completion_tokens`
- **Errors:** client retries 429 and 5xx with exponential backoff; server sets max concurrency
- **Config keys:** `llm.base_url`, `llm.model`, `llm.api_key_env`, `llm.max_concurrency`
- **Contract test:** `scripts/check_endpoint.py` (Andrew) runs the shared examples against the live endpoint;
  Armman's client tests run the same examples against an in-process fake server.
