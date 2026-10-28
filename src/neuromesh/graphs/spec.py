"""Shared constants and the graph-request specification.

:class:`GraphSpec` is deliberately a *small, isolated* request object that
describes "which graph do I want" without depending on anything outside this
subpackage, so that adapting to the project-wide interfaces requires changing
only the adapter in ``neuromesh.graphs.__init__``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: Population sizes swept by the study.
POPULATION_SIZES: tuple[int, ...] = (7, 14, 28, 56, 112)

#: Mean degree every topology is matched on, at every scale.
TARGET_MEAN_DEGREE: int = 4

#: Relative tolerance permitted on mean degree (+/-10%).
DEGREE_TOLERANCE: float = 0.10

#: N = 7 is a documented boundary case; topology comparisons start at N = 14.
BOUNDARY_N: int = 7

#: Seed used for the fixture set committed to the repository.
DEFAULT_FIXTURE_SEED: int = 0

#: The five seeds the full experiment will use.
EXPERIMENT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Topologies currently implemented.
TOPOLOGIES: tuple[str, ...] = ("er", "ws", "ba")

#: Bumped whenever a change alters the graphs a given (topology, n, seed) yields.
GENERATOR_VERSION: str = "trackA-week1-1.0.0"


class GraphParameterError(ValueError):
    """Raised for structurally invalid generator parameters.

    Subclasses :class:`ValueError` so callers that only catch ``ValueError``
    (including NetworkX's own convention) keep working.
    """


@dataclass(frozen=True)
class GraphSpec:
    """A reproducible request for one graph.

    Attributes
    ----------
    topology:
        One of :data:`TOPOLOGIES`.
    n:
        Number of agents / nodes. Node IDs are always ``0 .. n-1``.
    seed:
        Integer seed. Identical specs always yield identical edge sets.
    target_mean_degree:
        Mean degree the generator aims for, ``2E/N``.
    params:
        Generator-specific overrides, e.g. ``{"rewiring_prob": 0.1}`` for WS or
        ``{"variant": "standard"}`` for BA. Unknown keys raise.
    """

    topology: str
    n: int
    seed: int = DEFAULT_FIXTURE_SEED
    target_mean_degree: int = TARGET_MEAN_DEGREE
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise GraphParameterError(
                f"unknown topology {self.topology!r}; expected one of {TOPOLOGIES}"
            )

    @property
    def key(self) -> str:
        """Stable identifier used for fixture filenames and log joins."""
        return f"{self.topology}_n{self.n:03d}_seed{self.seed}"
