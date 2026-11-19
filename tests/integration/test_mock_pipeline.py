"""End-to-end mock integration test (owner: Mihir, Track A).

What this test is, exactly
--------------------------
It runs the MVP scenario the team planned -- ER graph, N = 7 and N = 14,
balanced roles, a consensus-shaped task, a mock LLM client, synchronous rounds,
a readout -- from one end to the other, with **no API key, no GPU, no network
and no paid service**. It is the test the Week 3 allocation asks for.

It is **not** a demonstration that the team's real pipeline works, because
two of the four tracks have not shipped the modules that pipeline needs. Being
precise about that boundary is the point of this docstring.

Real repository modules exercised here
--------------------------------------
======================================  =======  ===================================
Module                                  Track    What it does here
======================================  =======  ===================================
``neuromesh.graphs.generate_graph``     A        builds the ER graph
``neuromesh.graphs.validate_graph``     A        confirms the generator contract
``neuromesh.graphs.assign_roles``       A        the seven Yeo roles, N/7 each
``neuromesh.graphs.remove_nodes``       A        the Week 3 node removal
``neuromesh.llm.MockLLMClient``         B        answers each round's batch
``neuromesh.contracts.LLMRequest``      D        request objects, as specified
``neuromesh.contracts.LLMResponse``     D        response objects, as specified
``neuromesh.contracts.AgentTurn``       D        the per-round JSON schema
``neuromesh.contracts.TaskInstance``    D        the task hand-off object
``neuromesh.contracts.CallRecord``      D        one row of ``calls.parquet``
``neuromesh.contracts.RunResult``       D        one row of ``runs.parquet``
======================================  =======  ===================================

Production modules that do **not** exist yet
--------------------------------------------
* ``neuromesh.agents.run_network`` -- Track B's round engine. Raises
  ``NotImplementedError``. :func:`test_production_round_engine_is_still_absent`
  asserts that, so the day Armman lands it this test file states plainly that
  the harness below should be retired in favour of the real engine.
* ``neuromesh.tasks.get_task`` -- Track C's task registry. Raises
  ``NotImplementedError``, asserted likewise.
* ``neuromesh.baseline.run_baseline``, ``neuromesh.evaluation.compute_metrics``
  and Track D's experiment execution -- all unimplemented, none needed here.

Test-only scaffolding, named as such
------------------------------------
``MockRoundEngine`` below is a **test harness, not a product.** It is
deliberately the simplest synchronous protocol that can be reasoned about:
every agent forwards everything it knows to every neighbour, once per round.
It lives in this test file, is never importable as a production module, and is
not a substitute for Track B's engine -- its only job is to let the *graph*
side of the integration be exercised against real contract types.

``consensus_instance`` builds a ``TaskInstance`` by hand for the same reason:
``tasks.get_task("consensus")`` raises, so the instance is constructed from the
shared contract type directly. It is a test fixture, not Track C's task.

Where the two meet, and why that is worth testing
-------------------------------------------------
The harness asks ``MockLLMClient`` for a turn every round, parses the reply as
an ``AgentTurn``, and **asserts that every message the client addressed goes to
an actual graph neighbour.** Today's placeholder client returns no messages, so
that check is vacuous on content but live on plumbing. When Armman replaces it
with the real deterministic mock, this test starts checking his routing for
free, and a message sent to a non-neighbour fails here -- in Mihir's directory,
which is what ``CONTRIBUTING.md`` §8 prescribes.

Runtime
-------
The allocation requires under 60 seconds. These tests run in well under one
second locally, because the whole scenario is two small graphs and ten rounds.
There are no sleeps and no wall-clock assertions, so nothing here can make CI
flaky. ``test_whole_pipeline_runtime_is_well_under_the_ci_budget`` measures the
full scenario against a deliberately loose 20-second ceiling -- loose enough
that a slow shared CI runner cannot trip it, tight enough to catch a real
regression. Local runtime does not guarantee GitHub Actions runtime; CI has
not been run on this branch.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field

import networkx as nx
import pytest

from neuromesh.agents import run_network
from neuromesh.contracts import (
    DEFAULT_ROUNDS,
    INPUT_ROLE,
    MAX_MESSAGE_WORDS,
    READOUT_ROLE,
    YEO7,
    AgentTurn,
    CallRecord,
    LLMClient,
    LLMRequest,
    RemovalSpec,
    RunResult,
    TaskInstance,
)
from neuromesh.graphs import (
    assign_roles,
    generate_graph,
    path_metrics,
    remove_nodes,
    roles_are_balanced,
    validate_graph,
)
from neuromesh.llm import MockLLMClient
from neuromesh.tasks import get_task

#: The MVP scenario: the two smallest scales, ER, seed 0.
MVP_SIZES = (7, 14)
MVP_TOPOLOGY = "er"
MVP_SEED = 0

#: The fact that has to cross the network.
FACT = "the-token-is-amber"


# ==========================================================================
# boundary assertions: what is genuinely missing
# ==========================================================================

def test_production_round_engine_is_still_absent():
    """Track B's ``run_network`` is not implemented; the harness stands in.

    When this test starts failing, Armman has landed the real engine and
    ``MockRoundEngine`` below should be replaced by a call to it.
    """
    graph = assign_roles(generate_graph(MVP_TOPOLOGY, 7, seed=MVP_SEED), seed=MVP_SEED)
    instance = consensus_instance(graph, seed=MVP_SEED)
    with pytest.raises(NotImplementedError, match="run_network"):
        run_network(graph, instance, MockLLMClient(), rounds=2)


def test_production_task_registry_builds_a_consensus_instance():
    """Track C's ``get_task`` has landed (Week 5): it builds a valid instance on
    the MVP graph. The hand-built ``consensus_instance`` below stays as the
    harness fixture until the round engine lands."""
    graph = assign_roles(generate_graph(MVP_TOPOLOGY, 7, seed=MVP_SEED), seed=MVP_SEED)
    instance = get_task("consensus").make_instance(graph, MVP_SEED)
    assert instance.task == "consensus"
    assert set(instance.initial_state) == set(graph.nodes)
    assert set(instance.readout_nodes) <= set(graph.nodes)


def test_the_mock_llm_client_is_a_real_repository_module():
    """``MockLLMClient`` is Track B's actual file, and satisfies the protocol."""
    client = MockLLMClient()
    assert isinstance(client, LLMClient)
    responses = client.generate(
        [LLMRequest(agent_id=i, messages=[{"role": "user", "content": "hi"}]) for i in range(3)]
    )
    assert [r.agent_id for r in responses] == [0, 1, 2]
    for response in responses:
        AgentTurn.model_validate(json.loads(response.text))


# ==========================================================================
# test-only scaffolding
# ==========================================================================

def consensus_instance(graph: nx.Graph, seed: int) -> TaskInstance:
    """A consensus-shaped ``TaskInstance``, built by hand. **Test fixture.**

    Not Track C's consensus task. It uses the real shared contract type, and
    follows the project's convention that information enters at ``INPUT_ROLE``
    agents and is read out at ``READOUT_ROLE`` agents, so the hand-off shape is
    the real one even though the task content is not.
    """
    inputs = sorted(
        node for node, data in graph.nodes(data=True) if data["role"] == INPUT_ROLE
    )
    readouts = sorted(
        node for node, data in graph.nodes(data=True) if data["role"] == READOUT_ROLE
    )
    return TaskInstance(
        task="consensus",
        seed=seed,
        private_info={node: FACT for node in inputs},
        initial_state={node: {"known": [FACT]} for node in inputs},
        readout_nodes=readouts,
        ground_truth={"answer": FACT, "input_nodes": inputs},
    )


@dataclass
class RoundLog:
    """What one round did, for the assertions."""

    index: int
    knowledge: dict[int, frozenset] = field(default_factory=dict)
    deliveries: list[tuple[int, int]] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)


class MockRoundEngine:
    """**Test-only** synchronous round engine. Not a production module.

    One round is: every surviving agent asks the LLM client for a turn, then
    forwards everything it currently knows to each of its graph neighbours.
    Deliveries are applied simultaneously at the end of the round, so a fact
    advances exactly one hop per round and the result cannot depend on node
    iteration order.

    Two invariants are enforced *inside* the loop rather than only asserted
    afterwards, so a violation names the round and the agent:

    * no message ever crosses a non-edge, including any message the LLM client
      itself addressed;
    * no message is ever addressed to a node absent from the graph.
    """

    def __init__(self, graph: nx.Graph, instance: TaskInstance, llm: LLMClient):
        self.graph = graph
        self.instance = instance
        self.llm = llm
        self.run_id = f"mock-{instance.task}-n{graph.number_of_nodes()}-s{instance.seed}"
        self.knowledge: dict[int, frozenset] = {
            node: frozenset(instance.initial_state.get(node, {}).get("known", []))
            for node in sorted(graph.nodes())
        }
        self.logs: list[RoundLog] = []
        self.client_messages_seen = 0

    def _requests(self, round_index: int) -> list[LLMRequest]:
        requests = []
        for node in sorted(self.graph.nodes()):
            neighbours = sorted(self.graph.neighbors(node))
            prompt = (
                f"round {round_index}; role {self.graph.nodes[node]['role']}; "
                f"neighbours {neighbours}; known {sorted(self.knowledge[node])}"
            )
            requests.append(
                LLMRequest(
                    agent_id=int(node),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=64,
                    temperature=0.0,
                    seed=self.instance.seed,
                )
            )
        return requests

    def _check_client_routing(self, node: int, turn: AgentTurn) -> None:
        neighbours = {str(v) for v in self.graph.neighbors(node)}
        present = {str(v) for v in self.graph.nodes()}
        for target in turn.messages:
            assert target in present, (
                f"agent {node} addressed {target!r}, which is not a node of the graph"
            )
            assert target in neighbours, (
                f"agent {node} addressed non-neighbour {target!r}; "
                f"its neighbours are {sorted(neighbours)}"
            )
            words = len(turn.messages[target].split())
            assert words <= MAX_MESSAGE_WORDS, (
                f"agent {node} sent {words} words, over the "
                f"{MAX_MESSAGE_WORDS}-word contract limit"
            )
            self.client_messages_seen += 1

    def run(self, rounds: int = DEFAULT_ROUNDS) -> RunResult:
        calls: list[CallRecord] = []
        for round_index in range(1, rounds + 1):
            log = RoundLog(index=round_index)
            requests = self._requests(round_index)
            responses = self.llm.generate(requests)

            assert len(responses) == len(requests), "client dropped a request"
            assert [r.agent_id for r in responses] == [r.agent_id for r in requests], (
                "client returned the batch out of order; contracts.LLMClient "
                "requires the same order"
            )

            pending: dict[int, set] = {node: set() for node in self.graph.nodes()}
            for request, response in zip(requests, responses, strict=True):
                node = request.agent_id
                turn = AgentTurn.model_validate(json.loads(response.text))
                self._check_client_routing(node, turn)

                record = CallRecord(
                    run_id=self.run_id,
                    round=round_index,
                    node_id=node,
                    role=self.graph.nodes[node]["role"],
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_s=response.latency_s,
                    parse_ok=True,
                    state={"known": sorted(self.knowledge[node])},
                )
                calls.append(record)
                log.calls.append(record)

                # The harness's own protocol: flood what is known, one hop.
                # Applied after the client's turn so that a client which does
                # address neighbours is checked above either way.
                if self.knowledge[node]:
                    for neighbour in self.graph.neighbors(node):
                        pending[neighbour] |= self.knowledge[node]
                        log.deliveries.append((node, neighbour))

            for node, incoming in pending.items():
                self.knowledge[node] = self.knowledge[node] | frozenset(incoming)
            log.knowledge = dict(self.knowledge)
            self.logs.append(log)

        final_answers = {
            node: {"answer": sorted(self.knowledge[node])}
            for node in self.instance.readout_nodes
            if node in self.knowledge
        }
        return RunResult(
            run_id=self.run_id,
            topology=self.graph.graph["topology"],
            n=self.graph.graph["n"],
            task=self.instance.task,
            seed=self.instance.seed,
            wall_clock_s=0.0,
            calls=calls,
            final_answers=final_answers,
        )


def build_scenario(n: int, seed: int = MVP_SEED):
    """The MVP scenario: real generator, real validation, real role assignment."""
    graph = generate_graph(MVP_TOPOLOGY, n, seed=seed)
    report = validate_graph(graph, n=n, topology=MVP_TOPOLOGY, seed=seed)
    assert report.ok, report.summary()
    assign_roles(graph, seed=seed)
    assert roles_are_balanced(graph)
    instance = consensus_instance(graph, seed=seed)
    return graph, instance


# ==========================================================================
# the end-to-end scenario
# ==========================================================================

@pytest.mark.parametrize("n", MVP_SIZES)
def test_end_to_end_mock_pipeline(n):
    """Graph -> validation -> roles -> task -> mock rounds -> readout."""
    graph, instance = build_scenario(n)
    engine = MockRoundEngine(graph, instance, MockLLMClient())
    result = engine.run(rounds=DEFAULT_ROUNDS)

    # the log schema Track D will write
    assert isinstance(result, RunResult)
    assert result.topology == MVP_TOPOLOGY
    assert result.n == n
    assert result.task == "consensus"
    assert len(result.calls) == n * DEFAULT_ROUNDS
    assert {c.role for c in result.calls} <= set(YEO7)
    assert all(c.parse_ok for c in result.calls)
    assert all(c.run_id == result.run_id for c in result.calls)
    assert sorted({c.round for c in result.calls}) == list(range(1, DEFAULT_ROUNDS + 1))

    # every agent ends up knowing the fact, and the readout can report it
    assert all(FACT in known for known in engine.knowledge.values())
    assert set(result.final_answers) == set(instance.readout_nodes)
    assert instance.readout_nodes
    for answer in result.final_answers.values():
        assert answer["answer"] == [FACT]

    # the whole thing round-trips as JSON, which is what Parquet needs
    payload = result.model_dump_json()
    assert json.loads(payload)["run_id"] == result.run_id


@pytest.mark.parametrize("n", MVP_SIZES)
def test_information_only_ever_crosses_real_edges(n):
    graph, instance = build_scenario(n)
    engine = MockRoundEngine(graph, instance, MockLLMClient())
    engine.run(rounds=4)

    assert engine.logs
    for log in engine.logs:
        assert log.deliveries
        for sender, receiver in log.deliveries:
            assert graph.has_edge(sender, receiver), (
                f"round {log.index}: delivery {sender}->{receiver} is not an edge"
            )
            assert sender != receiver


@pytest.mark.parametrize("n", MVP_SIZES)
def test_no_nonexistent_node_is_ever_addressed(n):
    graph, instance = build_scenario(n)
    engine = MockRoundEngine(graph, instance, MockLLMClient())
    engine.run(rounds=3)
    nodes = set(graph.nodes())
    for log in engine.logs:
        for sender, receiver in log.deliveries:
            assert sender in nodes and receiver in nodes
        for record in log.calls:
            assert record.node_id in nodes


@pytest.mark.parametrize("n", MVP_SIZES)
def test_propagation_respects_graph_distance_exactly(n):
    """A fact cannot outrun the graph: knowledge after round r is the r-ball.

    This is the integration property that actually matters, and it is checked
    against shortest-path distances computed independently with NetworkX. The
    ``<=`` direction is protocol-independent -- no synchronous one-hop protocol
    can inform a node sooner than its distance from a source. The ``==``
    direction holds for this harness because its rule is an unconditional
    flood; a different protocol would satisfy only ``<=``.
    """
    graph, instance = build_scenario(n)
    sources = sorted(instance.private_info)
    distance = {
        node: min(
            nx.shortest_path_length(graph, source, node) for source in sources
        )
        for node in graph.nodes()
    }

    engine = MockRoundEngine(graph, instance, MockLLMClient())
    engine.run(rounds=DEFAULT_ROUNDS)

    for log in engine.logs:
        informed = {node for node, known in log.knowledge.items() if FACT in known}
        expected = {node for node, d in distance.items() if d <= log.index}
        assert informed <= expected, (
            f"round {log.index}: a fact outran the graph at "
            f"{sorted(informed - expected)}"
        )
        assert informed == expected, (
            f"round {log.index}: flood rule under-delivered at "
            f"{sorted(expected - informed)}"
        )


@pytest.mark.parametrize("n", MVP_SIZES)
def test_the_run_is_deterministic(n):
    graph, instance = build_scenario(n)
    first = MockRoundEngine(graph, instance, MockLLMClient())
    first_result = first.run(rounds=5)
    second = MockRoundEngine(graph, instance, MockLLMClient())
    second_result = second.run(rounds=5)

    assert first.knowledge == second.knowledge
    assert [log.deliveries for log in first.logs] == [
        log.deliveries for log in second.logs
    ]
    assert first_result.final_answers == second_result.final_answers
    assert [c.model_dump() for c in first_result.calls] == [
        c.model_dump() for c in second_result.calls
    ]


def test_each_round_advances_knowledge_by_exactly_one_hop():
    """Determinism of each individual step, not just of the whole run."""
    graph, instance = build_scenario(14)
    engine = MockRoundEngine(graph, instance, MockLLMClient())
    engine.run(rounds=6)

    previous = set(instance.private_info)
    for log in engine.logs:
        informed = {node for node, known in log.knowledge.items() if FACT in known}
        frontier = {
            neighbour
            for node in previous
            for neighbour in graph.neighbors(node)
        }
        assert informed == previous | frontier
        previous = informed


# ==========================================================================
# node removal inside the pipeline
# ==========================================================================

@pytest.mark.parametrize("n", MVP_SIZES)
def test_pipeline_runs_on_an_attacked_graph(n):
    """Remove a node; its incident edges go with it and the run still completes."""
    graph, _ = build_scenario(n)
    attacked = remove_nodes(graph, RemovalSpec(mode="targeted", fraction=0.2), seed=0)

    removed = attacked.graph["removal"]["removed_nodes"]
    assert removed
    for node in removed:
        assert node not in attacked
        # every edge that touched a removed node is gone
        for neighbour in graph.neighbors(node):
            assert not attacked.has_edge(node, neighbour)
    # ...and no other edge was touched: the survivors are the induced subgraph
    surviving_nodes = set(attacked.nodes())
    assert {tuple(sorted(e)) for e in attacked.edges()} == {
        tuple(sorted((u, v)))
        for u, v in graph.edges()
        if u in surviving_nodes and v in surviving_nodes
    }

    # roles survived on the agents that survived, so the task can be rebuilt
    assert all("role" in attacked.nodes[node] for node in attacked.nodes())
    instance = consensus_instance(attacked, seed=0)
    engine = MockRoundEngine(attacked, instance, MockLLMClient())
    result = engine.run(rounds=DEFAULT_ROUNDS)
    assert len(result.calls) == attacked.number_of_nodes() * DEFAULT_ROUNDS


def test_attacked_graph_may_strand_agents_and_that_is_reported():
    """Disconnection is a finding, not a crash.

    At N = 14 a targeted attack can isolate part of the network. The run must
    complete, and the agents in components with no source must end up knowing
    nothing -- which is the robustness effect the study is looking for.
    """
    graph, _ = build_scenario(14)
    attacked = remove_nodes(graph, RemovalSpec(mode="targeted", fraction=0.3), seed=0)
    metrics = path_metrics(attacked)
    assert metrics["n_components"] >= 1

    instance = consensus_instance(attacked, seed=0)
    engine = MockRoundEngine(attacked, instance, MockLLMClient())
    engine.run(rounds=DEFAULT_ROUNDS)

    sources = set(instance.private_info)
    reachable = set()
    for source in sources:
        reachable |= nx.node_connected_component(attacked, source)
    informed = {node for node, known in engine.knowledge.items() if FACT in known}
    assert informed == reachable
    stranded = set(attacked.nodes()) - reachable
    for node in stranded:
        assert engine.knowledge[node] == frozenset()


# ==========================================================================
# the test needs nothing from outside this machine
# ==========================================================================

def test_the_pipeline_makes_no_network_calls(monkeypatch):
    """Any socket creation during the run is a failure.

    Blocks ``socket.socket`` and ``socket.create_connection`` outright, so a
    future client that quietly reaches for an HTTP endpoint fails here rather
    than in someone's CI bill.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("the mock pipeline attempted a network connection")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    graph, instance = build_scenario(14)
    engine = MockRoundEngine(graph, instance, MockLLMClient())
    result = engine.run(rounds=DEFAULT_ROUNDS)
    assert result.final_answers


def test_the_pipeline_needs_no_api_key(monkeypatch):
    """No hosted model, no key. Clearing every plausible key changes nothing."""
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "NEUROMESH_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    graph, instance = build_scenario(7)
    result = MockRoundEngine(graph, instance, MockLLMClient()).run(rounds=4)
    assert len(result.calls) == 7 * 4


def test_whole_pipeline_runtime_is_well_under_the_ci_budget():
    """The allocation requires under 60 s. Measured, with a loose ceiling.

    The ceiling is 20 s rather than 60 s so a regression is caught early, and
    it is loose enough relative to the real runtime (well under one second
    locally) that a slow shared runner cannot make this flaky. There are no
    sleeps anywhere in this file. Local timing does not predict GitHub Actions
    timing, which has not been measured on this branch.
    """
    started = time.perf_counter()
    for n in MVP_SIZES:
        graph, instance = build_scenario(n)
        engine = MockRoundEngine(graph, instance, MockLLMClient())
        engine.run(rounds=DEFAULT_ROUNDS)
        attacked = remove_nodes(graph, RemovalSpec(mode="random", fraction=0.2), seed=0)
        attacked_instance = consensus_instance(attacked, seed=0)
        MockRoundEngine(attacked, attacked_instance, MockLLMClient()).run(
            rounds=DEFAULT_ROUNDS
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 20.0, f"mock pipeline took {elapsed:.2f}s"
