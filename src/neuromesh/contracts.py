"""Shared types and interfaces for every NeuroMesh module.

Owner: Andrew (Track D). This is the ONLY file all tracks depend on.
Version 0.1 (draft). Frozen as v1.0 on Nov 1; after that, changes go in a
separate `contracts/<topic>` PR approved by all four members.

Conventions
-----------
Graphs are ``networkx.Graph`` objects:
  * nodes are ints ``0 .. n-1``
  * every node has attribute ``role`` with a value from ``YEO7``
  * ``G.graph`` holds ``topology``, ``n``, ``seed`` and ``mean_degree``
Information enters at ``INPUT_ROLE`` agents and is read out at ``READOUT_ROLE`` agents.

Entry points (implemented by their owners, called by the runner)
----------------------------------------------------------------
  Mihir  (A)  neuromesh.graphs.generate_graph / assign_roles / remove_nodes / structural_metrics
  Armman (B)  neuromesh.llm.LLMClient implementations, neuromesh.agents.run_network
  Yutong (C)  neuromesh.tasks.get_task, neuromesh.baseline.run_baseline,
              neuromesh.evaluation.compute_metrics
  Andrew (D)  neuromesh.runner (calls all of the above only through these entry points)
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import networkx as nx
from pydantic import BaseModel, Field

CONTRACTS_VERSION = "0.1.0"

# --------------------------------------------------------------------------- constants
YEO7: tuple[str, ...] = (
    "visual",
    "somatomotor",
    "dorsal_attention",
    "ventral_attention",
    "limbic",
    "frontoparietal",
    "default",
)
INPUT_ROLE = "visual"
READOUT_ROLE = "frontoparietal"

DEFAULT_MEAN_DEGREE = 4.0  # same at every scale; 6 made N = 7 a complete graph
DEFAULT_ROUNDS = 10
DEFAULT_HISTORY_ROUNDS = 3
MAX_MESSAGE_WORDS = 80

Topology = Literal["er", "ws", "ba", "modular", "yeo_sbm"]
TaskName = Literal["fact_propagation", "consensus", "aggregation"]
RemovalMode = Literal["random", "targeted"]


# --------------------------------------------------------------------------- experiment
class RemovalSpec(BaseModel):
    """Node removal for the robustness experiment (N = 112 only)."""

    mode: RemovalMode
    fraction: float = Field(gt=0.0, lt=1.0)


# --------------------------------------------------------------------------- tasks
class TaskInstance(BaseModel):
    """One generated task, ready to run on one graph."""

    task: TaskName
    seed: int
    private_info: dict[int, str]  # text each agent sees at round 0 (may be empty)
    initial_state: dict[int, dict] = Field(default_factory=dict)
    readout_nodes: list[int]
    ground_truth: dict


# --------------------------------------------------------------------------- agents
class AgentTurn(BaseModel):
    """The JSON every agent must return each round."""

    messages: dict[str, str] = Field(default_factory=dict)  # neighbor name -> text
    state: dict = Field(default_factory=dict)  # task-specific, see docs/track_c.md


# --------------------------------------------------------------------------- LLM
class LLMRequest(BaseModel):
    agent_id: int
    messages: list[dict[str, str]]  # OpenAI-style [{"role": ..., "content": ...}]
    json_schema: dict | None = None
    max_tokens: int = 300
    temperature: float = 0.7
    seed: int | None = None


class LLMResponse(BaseModel):
    agent_id: int
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Any LLM backend: mock, replay, or a real OpenAI-compatible endpoint."""

    def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Answer one batch (one round) of requests, in the same order."""
        ...


# --------------------------------------------------------------------------- logs
class CallRecord(BaseModel):
    """One LLM call. One row in calls.parquet."""

    run_id: str
    round: int
    node_id: int
    role: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    parse_ok: bool
    retries: int = 0
    state: dict = Field(default_factory=dict)


class RunResult(BaseModel):
    """Everything one network run produced. One row in runs.parquet (+ its calls)."""

    run_id: str
    topology: Topology
    n: int
    task: TaskName
    seed: int
    removal: RemovalSpec | None = None
    wall_clock_s: float = 0.0
    calls: list[CallRecord] = Field(default_factory=list)
    final_answers: dict[int, dict] = Field(default_factory=dict)


class BaselineResult(BaseModel):
    """Single-agent baseline on one task instance."""

    task: TaskName
    seed: int
    answer_text: str
    scores: dict[str, float]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


# --------------------------------------------------------------------------- task protocol
@runtime_checkable
class Task(Protocol):
    """Implemented once per task by Yutong (C); looked up with tasks.get_task(name)."""

    name: TaskName

    def make_instance(self, graph: nx.Graph, seed: int) -> TaskInstance: ...

    def score(self, instance: TaskInstance, result: RunResult) -> dict[str, float]: ...

    def baseline_prompt(self, instance: TaskInstance) -> str: ...

    def score_baseline(self, instance: TaskInstance, answer_text: str) -> dict[str, float]: ...
