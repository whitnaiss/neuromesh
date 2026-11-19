"""Single-agent baseline tests: run_baseline with fake LLMs and the HTTP client."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from neuromesh.baseline import build_request, run_baseline
from neuromesh.baseline.__main__ import main as baseline_cli
from neuromesh.baseline.http_client import OllamaNativeClient, OpenAICompatibleClient
from neuromesh.contracts import BaselineResult, LLMClient, LLMRequest, LLMResponse
from neuromesh.graphs import assign_roles, generate_graph
from neuromesh.llm import MockLLMClient
from neuromesh.tasks import TASK_NAMES, get_task


def instance(name, seed=0):
    g = assign_roles(generate_graph("er", 14, seed=seed), seed=seed)
    return get_task(name).make_instance(g, seed)


class OracleLLM:
    """Answers every baseline prompt perfectly (reads the answer key)."""

    def __init__(self, inst):
        gt = inst.ground_truth
        if inst.task == "fact_propagation":
            self.text = json.dumps({"facts": {k: f["value"] for k, f in gt["facts"].items()}})
        elif inst.task == "consensus":
            self.text = json.dumps({"vote": 0})
        else:
            self.text = json.dumps({"values": gt["values"], "sum": gt["sum"]})
        self.requests: list[LLMRequest] = []

    def generate(self, requests):
        self.requests.extend(requests)
        return [LLMResponse(agent_id=r.agent_id, text=self.text, prompt_tokens=200,
                            completion_tokens=40, latency_s=0.25) for r in requests]


class FailingLLM:
    def generate(self, requests):
        return [LLMResponse(agent_id=r.agent_id, text="", error="HTTP 503") for r in requests]


@pytest.mark.parametrize("name", TASK_NAMES)
def test_oracle_model_gets_full_marks_and_costs_are_recorded(name):
    inst = instance(name)
    llm = OracleLLM(inst)
    assert isinstance(llm, LLMClient)
    res = run_baseline(inst, llm)
    assert isinstance(res, BaselineResult)
    assert res.task == name and res.seed == 0
    assert res.scores["parse_ok"] == 1.0 and res.scores["llm_error"] == 0.0
    assert (res.prompt_tokens, res.completion_tokens, res.latency_s) == (200, 40, 0.25)
    assert len(llm.requests) == 1  # exactly one call per instance


@pytest.mark.parametrize("name", TASK_NAMES)
def test_placeholder_mock_client_is_scored_as_a_failure_not_a_crash(name):
    res = run_baseline(instance(name), MockLLMClient())
    assert res.scores["parse_ok"] == 1.0  # valid JSON, but an empty AgentTurn ...
    rest = {k: v for k, v in res.scores.items() if k not in ("parse_ok", "llm_error")}
    assert all(v == 0.0 for v in rest.values())  # ... answers nothing


@pytest.mark.parametrize("name", TASK_NAMES)
def test_transport_errors_are_logged_and_scored_zero(name):
    res = run_baseline(instance(name), FailingLLM())
    assert res.scores["llm_error"] == 1.0
    assert res.scores["parse_ok"] == 0.0
    assert res.answer_text.startswith("[error] HTTP 503")


def test_request_shape():
    req = build_request(instance("aggregation", seed=4), max_tokens=123, suffix=" /no_think")
    assert req.max_tokens == 123 and req.seed == 4 and req.temperature == 0.0
    assert [m["role"] for m in req.messages] == ["system", "user"]
    assert req.messages[1]["content"].endswith("/no_think")


# --------------------------------------------------------------------------- HTTP client
class _Handler(BaseHTTPRequestHandler):
    plan: list[int] = []
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(body)
        status = _Handler.plan.pop(0) if _Handler.plan else 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if status == 200 and self.path.endswith("/api/chat"):  # Ollama native shape
            reply = {"message": {"content": '{"vote": 0}'}, "prompt_eval_count": 12,
                     "eval_count": 4}
            self.wfile.write(json.dumps(reply).encode())
        elif status == 200:
            reply = {"choices": [{"message": {"content": '{"vote": 1}'}}],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
            self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _Handler.plan, _Handler.seen = [], []
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_http_client_round_trip(server):
    client = OpenAICompatibleClient(server, "test-model")
    out = client.generate([LLMRequest(agent_id=3, messages=[{"role": "user", "content": "hi"}],
                                      seed=9)])
    assert out[0].text == '{"vote": 1}' and out[0].error is None
    assert (out[0].prompt_tokens, out[0].completion_tokens) == (11, 3)
    assert _Handler.seen[0]["model"] == "test-model" and _Handler.seen[0]["seed"] == 9


def test_http_client_retries_server_errors(server, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    _Handler.plan = [503]
    out = OpenAICompatibleClient(server, "m", retries=1).generate(
        [LLMRequest(agent_id=0, messages=[{"role": "user", "content": "x"}])])
    assert out[0].error is None and len(_Handler.seen) == 2


def test_http_client_does_not_retry_client_errors(server):
    _Handler.plan = [400]
    out = OpenAICompatibleClient(server, "m", retries=3).generate(
        [LLMRequest(agent_id=0, messages=[{"role": "user", "content": "x"}])])
    assert out[0].error == "HTTP 400" and len(_Handler.seen) == 1


def test_cli_writes_one_row_per_task_and_seed(server, tmp_path, capsys):
    assert baseline_cli(["--base-url", server, "--model", "m", "--seeds", "0", "1",
                         "--n", "7", "--out", str(tmp_path)]) == 0
    lines = (tmp_path / "baseline.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 2 * len(TASK_NAMES)
    raw = (tmp_path / "baseline_answers.jsonl").read_text().splitlines()
    answers = [json.loads(x) for x in raw]
    assert {a["task"] for a in answers} == set(TASK_NAMES)


def test_ollama_native_client_disables_thinking(server):
    client = OllamaNativeClient(server, "qwen3:8b")  # base url ends in /v1: stripped
    out = client.generate([LLMRequest(agent_id=1, messages=[{"role": "user", "content": "x"}],
                                      max_tokens=50, seed=3)])
    assert out[0].text == '{"vote": 0}' and (out[0].prompt_tokens, out[0].completion_tokens) == (
        12, 4)
    body = _Handler.seen[0]
    assert body["think"] is False and body["stream"] is False
    assert body["options"] == {"temperature": 0.7, "num_predict": 50, "seed": 3}


def test_budget_exhausted_by_hidden_reasoning_is_flagged(server, monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "_parse", lambda self, body: ("", 0, 400))
    out = OpenAICompatibleClient(server, "m").generate(
        [LLMRequest(agent_id=0, messages=[{"role": "user", "content": "x"}], max_tokens=400)])
    assert out[0].error and "token budget" in out[0].error
