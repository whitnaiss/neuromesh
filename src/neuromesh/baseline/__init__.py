"""Track C (Yutong): single-agent baseline. The runner calls only run_baseline().

The baseline is the n = 1 anchor of the study: one agent, the same model and
the same task instance as the network, but it sees *all* task information in a
single prompt. Any gap between baseline and network is therefore caused by
communication through the graph, not by the model failing the task itself.
"""

from __future__ import annotations

import time

from neuromesh.contracts import BaselineResult, LLMClient, LLMRequest, TaskInstance
from neuromesh.tasks import get_task

SYSTEM_PROMPT = (
    "You are a careful assistant. Follow the output format exactly and answer with a single "
    "JSON object and nothing else."
)


def build_request(
    instance: TaskInstance, max_tokens: int = 300, temperature: float = 0.0, suffix: str = ""
) -> LLMRequest:
    """The one request the baseline sends for ``instance``."""
    prompt = get_task(instance.task).baseline_prompt(instance) + suffix
    return LLMRequest(
        agent_id=-1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        seed=instance.seed,
    )


def run_baseline(
    instance: TaskInstance,
    llm: LLMClient,
    max_tokens: int = 300,
    temperature: float = 0.0,
    prompt_suffix: str = "",
) -> BaselineResult:
    """Give one agent all task information at once, then score its answer.

    Never raises on a bad model reply: a transport error or unparseable answer
    is scored as a failure (``parse_ok = 0``) and the error text is kept in
    ``answer_text`` so it shows up in the logs.
    """
    task = get_task(instance.task)
    request = build_request(instance, max_tokens, temperature, prompt_suffix)
    start = time.perf_counter()
    response = llm.generate([request])[0]
    elapsed = response.latency_s or (time.perf_counter() - start)

    if response.error:
        scores = task.score_baseline(instance, "")
        scores["llm_error"] = 1.0
        answer = f"[error] {response.error}"
    else:
        scores = task.score_baseline(instance, response.text)
        scores["llm_error"] = 0.0
        answer = response.text

    return BaselineResult(
        task=instance.task,
        seed=instance.seed,
        answer_text=answer,
        scores=scores,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_s=elapsed,
    )


__all__ = ["SYSTEM_PROMPT", "build_request", "run_baseline"]
