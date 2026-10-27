"""Repository smoke tests (owner: Andrew, Track D). Must always pass on main."""

import json
from pathlib import Path

import pytest

import neuromesh
from neuromesh import contracts
from neuromesh.contracts import AgentTurn, LLMClient, LLMRequest, RemovalSpec
from neuromesh.llm import MockLLMClient
from neuromesh.runner.cli import main
from neuromesh.runner.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_package_imports():
    assert neuromesh.__version__
    assert len(contracts.YEO7) == 7
    assert contracts.INPUT_ROLE in contracts.YEO7
    assert contracts.READOUT_ROLE in contracts.YEO7


def test_mock_client_returns_valid_agent_turn():
    llm = MockLLMClient()
    assert isinstance(llm, LLMClient)
    reqs = [LLMRequest(agent_id=i, messages=[{"role": "user", "content": "hi"}]) for i in range(3)]
    out = llm.generate(reqs)
    assert [r.agent_id for r in out] == [0, 1, 2]
    for r in out:
        AgentTurn.model_validate(json.loads(r.text))


def test_removal_fraction_is_validated():
    with pytest.raises(ValueError):
        RemovalSpec(mode="random", fraction=1.5)


@pytest.mark.parametrize(
    ("name", "expected_runs"),
    [("pilot.yaml", 45), ("core.yaml", 180), ("full.yaml", 75), ("robustness.yaml", 150)],
)
def test_configs_expand_to_expected_run_counts(name, expected_runs):
    cfg = load_config(CONFIGS / name)
    assert len(cfg.matrix()) == expected_runs


def test_cli_dry_run(capsys):
    assert main(["run", str(CONFIGS / "pilot.yaml"), "--dry-run"]) == 0
    assert "45 runs" in capsys.readouterr().out
