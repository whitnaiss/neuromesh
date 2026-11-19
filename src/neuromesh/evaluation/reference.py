"""Rule-based reference agents (no LLM).

These deterministic agents follow the same synchronous protocol as the LLM
round engine -- each round every agent reads its neighbours' previous-round
state and updates its own -- and log ``RunResult``/``CallRecord`` objects in the
real format. They serve two purposes:

1. **Validate the evaluation pipeline** end to end on every topology and size
   before any GPU time is spent (the Theme 6 guidance explicitly allows
   rule-based agents inside a population).
2. **Give a communication-limited reference**: perfect relays show the best
   any agent could do in 10 rounds on a given graph, so LLM results can later
   be read as "fraction of the achievable".

Behaviour per task
------------------
* fact_propagation -- adopt any fact a neighbour knows; with probability
  ``relay_error`` a newly adopted code gets one digit changed (telephone noise).
* consensus -- synchronous local majority over self + neighbours; ties keep the
  current vote. Majority dynamics can freeze in split states, as LLMs may.
* aggregation -- merge every value a neighbour knows; report the sum of the
  values known.

``prompt_tokens`` is a *proxy*: 4 characters per token of the neighbour states
an agent would read. It shows where context grows (hubs), not real LLM cost.
"""

from __future__ import annotations

import json
import random

import networkx as nx

from neuromesh.contracts import DEFAULT_ROUNDS, CallRecord, RunResult, TaskInstance

BASE_PROMPT_TOKENS = 150


def _corrupt(code: str, rng: random.Random) -> str:
    i = rng.randrange(len(code))
    digit = rng.choice([d for d in "0123456789" if d != code[i]])
    return code[:i] + digit + code[i + 1 :]


def _update(task: str, own: dict, inbox: list[dict], rng: random.Random, relay_error: float):
    if task == "fact_propagation":
        facts = dict(own.get("facts", {}))
        for msg in inbox:
            for fid, val in msg.get("facts", {}).items():
                if fid not in facts:
                    facts[fid] = _corrupt(val, rng) if rng.random() < relay_error else val
        return {"facts": facts}
    if task == "consensus":
        votes = [own["vote"]] + [m["vote"] for m in inbox if "vote" in m]
        ones = sum(votes)
        zeros = len(votes) - ones
        return {"vote": own["vote"] if ones == zeros else int(ones > zeros)}
    if task == "aggregation":
        values = dict(own.get("values", {}))
        for msg in inbox:
            for k, v in msg.get("values", {}).items():
                values.setdefault(k, v)
        return {"values": values, "sum": sum(values.values())}
    raise ValueError(f"unknown task {task!r}")


def run_reference(
    graph: nx.Graph,
    instance: TaskInstance,
    rounds: int = DEFAULT_ROUNDS,
    relay_error: float = 0.0,
    seed: int = 0,
    run_id: str | None = None,
) -> RunResult:
    """Run the reference agents on ``graph`` and return a log-shaped ``RunResult``."""
    rng = random.Random(f"reference|{instance.task}|{seed}|{relay_error}")
    nodes = sorted(int(v) for v in graph.nodes)
    state = {v: dict(instance.initial_state.get(v, {})) for v in nodes}
    roles = dict(graph.nodes(data="role"))
    calls: list[CallRecord] = []
    rid = run_id or (
        f"ref-{graph.graph.get('topology')}-n{graph.graph.get('n')}-{instance.task}-s{seed}"
    )
    for r in range(1, rounds + 1):
        new_state = {}
        for v in nodes:
            inbox = [state[u] for u in graph.neighbors(v)]
            new_state[v] = _update(instance.task, state[v], inbox, rng, relay_error)
            ctx = BASE_PROMPT_TOKENS + sum(len(json.dumps(m)) for m in inbox) // 4
            calls.append(CallRecord(run_id=rid, round=r, node_id=v, role=str(roles.get(v)),
                                    prompt_tokens=ctx,
                                    completion_tokens=len(json.dumps(new_state[v])) // 4,
                                    latency_s=0.0, parse_ok=True, state=new_state[v]))
        state = new_state
    return RunResult(
        run_id=rid,
        topology=graph.graph.get("topology", "er"),
        n=int(graph.graph.get("n", len(nodes))),
        task=instance.task,
        seed=seed,
        calls=calls,
        final_answers={v: state[v] for v in nodes},
    )
