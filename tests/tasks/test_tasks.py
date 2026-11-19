"""Track C task tests: instances, scoring against hand-built transcripts, baseline scoring."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from neuromesh.contracts import YEO7, CallRecord, RemovalSpec, RunResult, Task, TaskInstance
from neuromesh.graphs import assign_roles, generate_graph, remove_nodes
from neuromesh.tasks import TASK_NAMES, get_task
from neuromesh.tasks._common import as_int, hop_distance_from, parse_json_object

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts"


def graph(topology="er", n=14, seed=0):
    return assign_roles(generate_graph(topology, n, seed=seed), seed=seed)


def load_fixture(name):
    fx = json.loads((FIXTURES / name).read_text())
    instance = TaskInstance.model_validate(fx["instance"])
    result = RunResult(**fx["run"], calls=[CallRecord(**c) for c in fx["calls"]])
    return fx, instance, result


# --------------------------------------------------------------------------- registry
@pytest.mark.parametrize("name", TASK_NAMES)
def test_registry_returns_contract_tasks(name):
    task = get_task(name)
    assert isinstance(task, Task)
    assert task.name == name


def test_unknown_task_is_a_clear_error():
    with pytest.raises(KeyError, match="available"):
        get_task("telepathy")


# --------------------------------------------------------------------------- instances
@pytest.mark.parametrize("name", TASK_NAMES)
@pytest.mark.parametrize("topology", ["er", "ws", "ba", "modular", "yeo_sbm"])
def test_instances_cover_every_node_and_are_json_serialisable(name, topology):
    g = graph(topology, 28)
    inst = get_task(name).make_instance(g, seed=3)
    assert set(inst.private_info) == set(g.nodes)
    assert set(inst.initial_state) == set(g.nodes)
    assert inst.readout_nodes == sorted(v for v, r in g.nodes(data="role") if r == "frontoparietal")
    TaskInstance.model_validate_json(inst.model_dump_json())


@pytest.mark.parametrize("name", TASK_NAMES)
def test_instances_are_deterministic_per_seed(name):
    g = graph()
    task = get_task(name)
    assert task.make_instance(g, 1) == task.make_instance(g, 1)
    assert task.make_instance(g, 1).ground_truth != task.make_instance(g, 2).ground_truth


@pytest.mark.parametrize("name", TASK_NAMES)
def test_graph_without_roles_is_rejected(name):
    with pytest.raises(ValueError, match="assign_roles"):
        get_task(name).make_instance(nx.path_graph(7), seed=0)


def test_fact_propagation_only_visual_agents_get_facts():
    g = graph(n=56)
    inst = get_task("fact_propagation").make_instance(g, 0)
    facts = inst.ground_truth["facts"]
    assert len(facts) == 5
    assert len({f["value"] for f in facts.values()}) == 5  # codes are unique
    for v, role in g.nodes(data="role"):
        assert bool(inst.private_info[v]) == (role == "visual")
        assert (inst.ground_truth["hop_distance"][str(v)] == 0) == (role == "visual")
    for f in facts.values():  # every code appears in the Visual brief
        assert f["value"] in inst.private_info[inst.ground_truth["input_nodes"][0]]


def test_consensus_never_starts_unanimous():
    task = get_task("consensus")
    for seed in range(40):
        votes = task.make_instance(graph("ba", 7, seed), seed).ground_truth["initial_votes"]
        assert set(votes.values()) == {0, 1}


def test_aggregation_values_are_distinct_and_known_only_to_their_network():
    g = graph(n=14)
    inst = get_task("aggregation").make_instance(g, 0)
    values = inst.ground_truth["values"]
    assert list(values) == list(YEO7)
    assert len(set(values.values())) == 7
    assert inst.ground_truth["sum"] == sum(values.values())
    for v, role in g.nodes(data="role"):
        assert inst.initial_state[v] == {"values": {role: values[role]}}
        assert str(values[role]) in inst.private_info[v]


def test_aggregation_after_removal_reports_missing_networks():
    g = graph("ba", 112)
    attacked = remove_nodes(g, RemovalSpec(mode="targeted", fraction=0.3), seed=0)
    inst = get_task("aggregation").make_instance(attacked, 0)
    present = {r for _, r in attacked.nodes(data="role")}
    assert set(inst.ground_truth["networks_present"]) == present
    assert set(inst.initial_state) == set(attacked.nodes)


def test_aggregation_separates_recall_from_arithmetic():
    task = get_task("aggregation")
    inst = task.make_instance(graph(), 0)
    gt = inst.ground_truth
    wrong_sum = json.dumps({"values": gt["values"], "sum": gt["sum"] + 10})
    scores = task.score_baseline(inst, wrong_sum)
    assert scores["values_complete"] == 1.0 and scores["partial_accuracy"] == 1.0
    assert scores["sum_accuracy"] == 0.0 and scores["exact_accuracy"] == 0.0


def test_aggregation_difficulty_range_is_configurable():
    from neuromesh.tasks import AggregationTask

    inst = AggregationTask(low=100, high=999).make_instance(graph(), 0)
    assert all(100 <= v <= 999 for v in inst.ground_truth["values"].values())
    with pytest.raises(ValueError):
        AggregationTask(low=1, high=3)


# --------------------------------------------------------------------------- scoring
@pytest.mark.parametrize(
    "fixture", ["fact_propagation_path7.json", "consensus_cycle4.json", "aggregation_ring14.json"]
)
def test_scores_match_hand_derived_values(fixture):
    fx, instance, result = load_fixture(fixture)
    scores = get_task(instance.task).score(instance, result)
    for key, expected in fx["expected"].items():
        assert scores[key] == pytest.approx(expected), key


def test_final_answers_take_precedence_over_the_call_log():
    fx, instance, result = load_fixture("consensus_cycle4.json")
    result.final_answers = {v: {"vote": 0} for v in range(4)}
    scores = get_task("consensus").score(instance, result)
    assert scores["agreement_ratio"] == 1.0
    assert scores["consensus_reached"] == 1.0


def test_unparsed_calls_count_as_empty_state():
    fx, instance, result = load_fixture("aggregation_ring14.json")
    for call in result.calls:
        call.parse_ok = False
    scores = get_task("aggregation").score(instance, result)
    assert scores["exact_accuracy"] == 0.0 and scores["partial_accuracy"] == 0.0


def test_fact_propagation_never_reached_is_censored():
    fx, instance, result = load_fixture("fact_propagation_path7.json")
    for call in result.calls:
        if call.node_id == 6:
            call.state = {"facts": {}}
    scores = get_task("fact_propagation").score(instance, result)
    assert scores["t50_reached"] == 0.0
    assert scores["propagation_delay_t50"] == 4.0  # rounds + 1
    assert scores["readout_fidelity"] == 0.0


def test_empty_run_scores_zero_without_crashing():
    for name in TASK_NAMES:
        inst = get_task(name).make_instance(graph(n=7), 0)
        res = RunResult(run_id="x", topology="er", n=7, task=name, seed=0)
        scores = get_task(name).score(inst, res)
        assert all(isinstance(v, float) for v in scores.values())


# --------------------------------------------------------------------------- baseline scoring
def perfect_answer(inst: TaskInstance) -> str:
    gt = inst.ground_truth
    if inst.task == "fact_propagation":
        return json.dumps({"facts": {k: f["value"] for k, f in gt["facts"].items()}})
    if inst.task == "consensus":
        return json.dumps({"vote": 1})
    return json.dumps({"values": gt["values"], "sum": gt["sum"]})


@pytest.mark.parametrize("name", TASK_NAMES)
def test_baseline_prompt_contains_everything_needed(name):
    inst = get_task(name).make_instance(graph(), 0)
    prompt = get_task(name).baseline_prompt(inst)
    assert "JSON" in prompt
    if name == "fact_propagation":
        assert all(f["value"] in prompt for f in inst.ground_truth["facts"].values())
    if name == "aggregation":
        assert all(str(v) in prompt for v in inst.ground_truth["values"].values())


@pytest.mark.parametrize("name", TASK_NAMES)
@pytest.mark.parametrize(
    "wrap",
    [lambda s: s, lambda s: f"```json\n{s}\n```", lambda s: f"<think>\nhmm\n</think>\n{s}",
     lambda s: f"Sure! Here is the answer: {s} Hope this helps."],
)
def test_baseline_perfect_answers_score_full_marks(name, wrap):
    task = get_task(name)
    inst = task.make_instance(graph(), 0)
    scores = task.score_baseline(inst, wrap(perfect_answer(inst)))
    assert scores["parse_ok"] == 1.0
    headline = {"fact_propagation": "fidelity", "consensus": "agreement_ratio",
                "aggregation": "exact_accuracy"}[name]
    assert scores[headline] == 1.0


@pytest.mark.parametrize("name", TASK_NAMES)
@pytest.mark.parametrize("bad", ["", "I cannot help with that.", "{not json", "[1, 2, 3]"])
def test_baseline_bad_answers_score_zero(name, bad):
    task = get_task(name)
    scores = task.score_baseline(task.make_instance(graph(), 0), bad)
    assert scores["parse_ok"] == 0.0
    assert all(v == 0.0 for v in scores.values())


def test_baseline_distortion_is_measured():
    task = get_task("fact_propagation")
    inst = task.make_instance(graph(), 0)
    answer = json.loads(perfect_answer(inst))
    answer["facts"]["F1"] = "0000"
    scores = task.score_baseline(inst, json.dumps(answer))
    assert scores["fidelity"] == pytest.approx(0.8)
    assert scores["distortion_rate"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- helpers
@pytest.mark.parametrize(
    ("value", "expected"),
    [(7, 7), ("7", 7), (" 1,234 ", 1234), (7.0, 7), (7.5, None), ("seven", None), (True, None),
     (None, None)],
)
def test_as_int(value, expected):
    assert as_int(value) == expected


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('text {"a": {"b": 2}} more') == {"a": {"b": 2}}
    assert parse_json_object("<think>{x}</think>{\"a\": 1}") == {"a": 1}
    assert parse_json_object("[1]") is None
    assert parse_json_object("") is None


def test_hop_distance_marks_unreachable_nodes():
    g = nx.Graph([(0, 1), (1, 2)])
    g.add_node(3)
    assert hop_distance_from(g, [0]) == {0: 0, 1: 1, 2: 2, 3: -1}
    assert hop_distance_from(g, []) == {0: -1, 1: -1, 2: -1, 3: -1}
