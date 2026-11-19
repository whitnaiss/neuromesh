"""Evaluation tests: metrics from logs, summaries, regression, reference agents, figures."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from neuromesh.contracts import CallRecord, RemovalSpec, RunResult, TaskInstance
from neuromesh.evaluation import (
    compute_metrics,
    covariate_regression,
    mean_ci,
    summarize,
    to_frames,
)
from neuromesh.evaluation.pilot import main as pilot_main
from neuromesh.evaluation.plots import plot_baseline_vs_network, plot_fidelity_vs_hop
from neuromesh.evaluation.reference import run_reference
from neuromesh.graphs import assign_roles, generate_graph
from neuromesh.tasks import TASK_NAMES, get_task

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts"
FIXTURE_FILES = ["fact_propagation_path7.json", "consensus_cycle4.json", "aggregation_ring14.json"]


def fixture_run(name):
    fx = json.loads((FIXTURES / name).read_text())
    inst = TaskInstance.model_validate(fx["instance"])
    res = RunResult(**fx["run"], calls=[CallRecord(**c) for c in fx["calls"]])
    return fx, res, inst


# --------------------------------------------------------------------------- metrics
def test_metrics_from_logs_match_the_task_scorers():
    pairs = [fixture_run(f)[1:] for f in FIXTURE_FILES]
    runs, calls = to_frames(pairs)
    assert len(runs) == 3 and len(calls) == sum(len(r.calls) for r, _ in pairs)
    m = compute_metrics(runs, calls).set_index("run_id")
    for f in FIXTURE_FILES:
        fx, res, _ = fixture_run(f)
        for key, expected in fx["expected"].items():
            assert m.loc[res.run_id, key] == pytest.approx(expected), (f, key)


def test_cost_metrics():
    fx, res, inst = fixture_run("fact_propagation_path7.json")
    res.calls[0].parse_ok = False
    res.calls[0].retries = 1
    runs, calls = to_frames([(res, inst)])
    row = compute_metrics(runs, calls).iloc[0]
    assert row["n_calls"] == 21
    assert row["prompt_tokens"] == sum(c.prompt_tokens for c in res.calls)
    assert row["total_tokens"] == row["prompt_tokens"] + 21 * 20
    assert row["max_context_tokens"] == 160  # node 6 prompt in the fixture
    assert row["parse_error_rate"] == pytest.approx(1 / 21)
    assert row["retries"] == 1 and row["wall_clock_s"] == 12.0
    assert row["tokens_per_agent"] == pytest.approx(row["total_tokens"] / 7)


def test_metrics_accept_dict_columns_and_runs_without_calls():
    fx, res, inst = fixture_run("consensus_cycle4.json")
    runs, calls = to_frames([(res, inst)])
    runs["instance"] = runs["instance"].map(json.loads)  # dicts instead of JSON strings
    calls["state"] = calls["state"].map(json.loads)
    assert compute_metrics(runs, calls).iloc[0]["agreement_ratio"] == 0.75
    empty = compute_metrics(runs, calls.iloc[0:0]).iloc[0]  # a run whose calls were lost
    assert empty["n_calls"] == 0 and empty["agreement_ratio"] == 0.0


def test_removal_is_reported():
    fx, res, inst = fixture_run("aggregation_ring14.json")
    res.removal = RemovalSpec(mode="targeted", fraction=0.2)
    row = compute_metrics(*to_frames([(res, inst)])).iloc[0]
    assert (row["removal_mode"], row["removal_fraction"]) == ("targeted", 0.2)


# --------------------------------------------------------------------------- statistics
def test_mean_ci_matches_the_t_interval():
    m, lo, hi = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0
    assert hi - m == pytest.approx(4.302653 / np.sqrt(3), rel=1e-5)
    assert mean_ci([5.0]) == (5.0, 5.0, 5.0)
    assert all(np.isnan(mean_ci([])))


def test_summarize_long_table():
    df = pd.DataFrame({"task": ["a"] * 4, "topology": ["er", "er", "ba", "ba"], "n": [14] * 4,
                       "score": [0.2, 0.4, 1.0, 1.0]})
    s = summarize(df, ["score", "missing"]).set_index("topology")
    assert s.loc["er", "mean"] == pytest.approx(0.3) and s.loc["er", "n_runs"] == 2
    assert s.loc["ba", "ci_low"] == s.loc["ba", "ci_high"] == 1.0


def test_regression_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    rows = []
    for topo, effect in {"er": 0.0, "ws": 0.3, "ba": -0.2}.items():
        for n in (14, 28, 56, 112):
            for _ in range(6):
                x = rng.normal()
                rows.append({"topology": topo, "n": n, "x": x,
                             "y": 1 + 0.5 * np.log2(n) + effect + 2 * x + rng.normal(0, 0.01)})
    res = covariate_regression(pd.DataFrame(rows), "y", covariates=["x"])
    coef = res.table.set_index("term")["coef"]
    assert coef["log2_n"] == pytest.approx(0.5, abs=0.01)
    assert coef["x"] == pytest.approx(2.0, abs=0.01)
    assert coef["topology[ws]"] == pytest.approx(0.3, abs=0.01)
    assert coef["topology[ba]"] == pytest.approx(-0.2, abs=0.01)
    assert res.r2 > 0.999 and "topology[er]" not in coef


def test_regression_drops_constant_and_collinear_columns():
    df = pd.DataFrame({"topology": ["er", "ws"] * 10, "n": [14, 28] * 10,
                       "y": np.arange(20.0), "const_cov": 1.0})
    df["dup"] = np.log2(df["n"])
    res = covariate_regression(df, "y", covariates=["const_cov", "dup"])
    assert "const_cov" in res.dropped and "dup" in res.dropped


# --------------------------------------------------------------------------- reference agents
def path_graph_with_roles():
    g = nx.path_graph(7)
    roles = ["visual", "somatomotor", "dorsal_attention", "ventral_attention", "limbic",
             "default", "frontoparietal"]
    nx.set_node_attributes(g, dict(enumerate(roles)), "role")
    g.graph.update(topology="er", n=7, seed=0)
    return g


def test_ideal_relays_deliver_facts_one_hop_per_round():
    g = path_graph_with_roles()
    inst = get_task("fact_propagation").make_instance(g, 0)
    res = run_reference(g, inst, rounds=10)
    by_round = {}
    for c in res.calls:
        by_round.setdefault(c.round, {})[c.node_id] = c.state
    for r in range(1, 7):
        assert len(by_round[r][r]["facts"]) == 5 and len(by_round[r].get(r + 1, {"facts": {}})
                                                         ["facts"]) == 0 or r == 6
    scores = get_task("fact_propagation").score(inst, res)
    assert scores["readout_fidelity"] == 1.0 and scores["propagation_delay_t50"] == 6.0
    assert scores["distortion_rate"] == 0.0


def test_too_few_rounds_leave_the_readout_uninformed():
    g = path_graph_with_roles()
    inst = get_task("aggregation").make_instance(g, 0)
    assert get_task("aggregation").score(inst, run_reference(g, inst, rounds=3))[
        "exact_accuracy"] == 0.0
    assert get_task("aggregation").score(inst, run_reference(g, inst, rounds=6))[
        "exact_accuracy"] == 1.0


def test_relay_noise_creates_distortion_that_grows_with_hops():
    g = assign_roles(generate_graph("ws", 112, seed=0), seed=0)
    inst = get_task("fact_propagation").make_instance(g, 0)
    s = get_task("fact_propagation").score(inst, run_reference(g, inst, relay_error=0.2, seed=1))
    assert s["distortion_rate"] > 0
    assert s["fidelity_hop_0"] == 1.0 and s["fidelity_hop_slope"] < 0


def test_consensus_majority_dynamics_freeze_on_ties():
    """Leaves of a star see only themselves and the hub: a 1-1 tie keeps their vote,
    so the two dissenting leaves never switch -- a split state the metric must report."""
    g = nx.star_graph(6)  # hub 0
    nx.set_node_attributes(g, {v: "visual" for v in g}, "role")
    g.graph.update(topology="ba", n=7, seed=0)
    inst = TaskInstance(task="consensus", seed=0,
                        private_info={v: "" for v in g},
                        initial_state={0: {"vote": 1}, **{v: {"vote": int(v <= 4)} for v in
                                                          range(1, 7)}},
                        readout_nodes=[], ground_truth={"initial_votes": {},
                                                        "initial_majority_share": 5 / 7})
    s = get_task("consensus").score(inst, run_reference(g, inst, rounds=3))
    assert s["consensus_reached"] == 0.0 and s["convergence_round"] == 4.0  # censored
    assert s["agreement_ratio"] == pytest.approx(5 / 7)


@pytest.mark.parametrize("task", TASK_NAMES)
def test_reference_runs_are_deterministic(task):
    g = assign_roles(generate_graph("ba", 28, seed=2), seed=2)
    inst = get_task(task).make_instance(g, 2)
    a = run_reference(g, inst, relay_error=0.1, seed=2)
    b = run_reference(g, inst, relay_error=0.1, seed=2)
    assert a == b


# --------------------------------------------------------------------------- figures and pilot
def test_figure_functions_write_files(tmp_path):
    hop = pd.DataFrame({"topology": ["er"] * 6, "hop": [0, 1, 2] * 2, "fidelity": [1, .9, .8] * 2})
    assert plot_fidelity_vs_hop(hop, tmp_path / "hop.png", min_runs=1).stat().st_size > 0
    bars = pd.DataFrame({"label": ["FP", "FP"], "system": ["Single agent", "er"],
                         "score": [1.0, 0.6]})
    assert plot_baseline_vs_network(bars, tmp_path / "bars.png").stat().st_size > 0


def test_pilot_pipeline_end_to_end(tmp_path, monkeypatch):
    import neuromesh.evaluation.pilot as pilot

    monkeypatch.setattr(pilot, "SIZES", [14, 28])
    assert pilot_main(["--out", str(tmp_path), "--seeds", "0", "1"]) == 0
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert len(metrics) == 2 * 5 * 2 * 3 * 2  # settings x topologies x sizes x tasks x seeds
    for name in ("summary.csv", "fig_reference_scaling.png", "fig_noisy_fidelity.png",
                 "fig_fidelity_vs_hop.png"):
        assert (tmp_path / name).exists()
    assert list(tmp_path.glob("regression_*.txt"))
