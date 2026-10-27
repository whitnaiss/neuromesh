"""Experiment configuration (starter version; Track D owns and extends it)."""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from neuromesh.contracts import (
    DEFAULT_HISTORY_ROUNDS,
    DEFAULT_MEAN_DEGREE,
    DEFAULT_ROUNDS,
    RemovalSpec,
    TaskName,
    Topology,
)


class LLMSettings(BaseModel):
    backend: str = "mock"  # mock | openai_compatible
    model: str = "TBD-after-pilot"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.7
    max_tokens: int = 300
    max_concurrency: int = 64


class ExperimentConfig(BaseModel):
    name: str
    topologies: list[Topology]
    scales: list[int]
    tasks: list[TaskName]
    seeds: list[int]
    rounds: int = DEFAULT_ROUNDS
    mean_degree: float = DEFAULT_MEAN_DEGREE
    history_rounds: int = DEFAULT_HISTORY_ROUNDS
    removals: list[RemovalSpec] = Field(default_factory=list)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    def matrix(self) -> list[dict]:
        """Every (topology, n, task, seed, removal) combination to run."""
        removals: list[RemovalSpec | None] = list(self.removals) or [None]
        return [
            {"topology": t, "n": n, "task": k, "seed": s, "removal": r}
            for t, n, k, s, r in itertools.product(
                self.topologies, self.scales, self.tasks, self.seeds, removals
            )
        ]


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        return ExperimentConfig.model_validate(yaml.safe_load(f))
