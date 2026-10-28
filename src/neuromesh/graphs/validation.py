"""Automated structural validation for generated graphs.

Implements nine structural checks. Seven of them are
properties of a single graph and live in :func:`validate_graph`; the remaining
two -- seed reproducibility and invalid-parameter handling -- are properties of
a *generator*, so they live in :func:`check_reproducible` and are additionally
asserted in ``tests/test_generators.py``.

Check statuses
--------------
``PASS``
    The property holds.
``FAIL``
    The property does not hold and that is a defect.
``EXCEPTION``
    The property does not hold, and it is a declared, documented consequence of
    the generative model rather than a defect -- for example standard
    Barabasi-Albert cannot reach mean degree 4 at N = 7. An EXCEPTION is never
    silently rewritten as a PASS: it carries the reason with it, and a result
    holding one is :attr:`ValidationResult.ok` (no defect) but *not*
    :attr:`ValidationResult.sweep_eligible` (not usable in the matched-degree
    sweep).
``SKIP``
    The check was not applicable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .generators import mean_degree, target_edge_count
from .spec import DEGREE_TOLERANCE, TARGET_MEAN_DEGREE, GraphParameterError

__all__ = [
    "CheckStatus",
    "Check",
    "ValidationResult",
    "validate_graph",
    "check_reproducible",
    "degree_deviation",
]

PASS = "PASS"
FAIL = "FAIL"
EXCEPTION = "EXCEPTION"
SKIP = "SKIP"

#: Type alias kept as a plain string set for JSON friendliness.
CheckStatus = str


@dataclass
class Check:
    """Outcome of one named validation check."""

    name: str
    status: CheckStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True when the check did not reveal a defect.

        ``EXCEPTION`` counts as "not a defect" but is reported separately so a
        caller can never mistake it for a clean PASS.
        """
        return self.status in (PASS, EXCEPTION, SKIP)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class ValidationResult:
    """Full validation report for one graph."""

    topology: str
    n: int
    seed: int | None
    n_nodes: int
    n_edges: int
    mean_degree: float
    target_mean_degree: float
    connected: bool
    checks: list = field(default_factory=list)

    @property
    def failures(self) -> list:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def exceptions(self) -> list:
        return [c for c in self.checks if c.status == EXCEPTION]

    @property
    def ok(self) -> bool:
        """True when no check reported a defect."""
        return not self.failures

    @property
    def sweep_eligible(self) -> bool:
        """True when the graph is clean *and* carries no declared exception.

        ``ok`` and ``sweep_eligible`` answer different questions and must not be
        confused. ``ok`` means "no defect": a standard-BA graph at N = 7 that
        misses mean degree 4 is ``ok``, because missing it is a documented
        property of the model rather than a bug. But a graph that is only
        ``ok`` by way of an ``EXCEPTION`` is *not* matched on mean degree, and
        mean-degree matching is the study's controlled variable -- so it is not
        eligible for the main experimental sweep. Only
        ``sweep_eligible`` graphs are.
        """
        return not self.failures and not self.exceptions

    def status_of(self, name: str) -> CheckStatus:
        for check in self.checks:
            if check.name == name:
                return check.status
        raise KeyError(f"no check named {name!r}")

    def to_dict(self) -> dict:
        return {
            "topology": self.topology,
            "n": self.n,
            "seed": self.seed,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "mean_degree": self.mean_degree,
            "target_mean_degree": self.target_mean_degree,
            "connected": self.connected,
            "ok": self.ok,
            "sweep_eligible": self.sweep_eligible,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        parts = [f"{c.name}={c.status}" for c in self.checks]
        suffix = "" if self.sweep_eligible else "  [not sweep-eligible]"
        return f"{self.topology} N={self.n}: " + " ".join(parts) + suffix


def degree_deviation(
    graph: nx.Graph, target_mean_degree: float = TARGET_MEAN_DEGREE
) -> float:
    """Relative deviation of 2E/N from the target, signed.

    ``(observed - target) / target``. Negative means the graph is too sparse.
    """
    if target_mean_degree <= 0:
        raise GraphParameterError("target_mean_degree must be positive")
    return (mean_degree(graph) - target_mean_degree) / target_mean_degree


def validate_graph(
    graph: nx.Graph,
    *,
    n: int,
    topology: str = "unknown",
    seed: int | None = None,
    target_mean_degree: float = TARGET_MEAN_DEGREE,
    degree_tolerance: float = DEGREE_TOLERANCE,
    require_connected: bool = True,
    degree_exception_reason: str | None = None,
) -> ValidationResult:
    """Run the per-graph structural checks and return a report.

    Parameters
    ----------
    n:
        The requested number of nodes. Node IDs must be exactly ``0 .. n-1``.
    degree_exception_reason:
        If given, a mean-degree miss is recorded as ``EXCEPTION`` carrying this
        reason instead of ``FAIL``. Use only for a genuine model constraint
        (e.g. standard BA at small N) -- never to make a defect look benign.
    require_connected:
        When False, connectivity is still measured and reported but a
        disconnected graph is an ``EXCEPTION`` rather than a ``FAIL``.
    """
    if not isinstance(graph, nx.Graph):
        raise GraphParameterError(
            f"expected a networkx.Graph, got {type(graph).__name__}"
        )
    if degree_tolerance < 0:
        raise GraphParameterError(
            f"degree_tolerance must be non-negative, got {degree_tolerance}"
        )

    checks: list = []
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # 1. Correct number of nodes, with the agreed integer node IDs 0..n-1.
    expected_nodes = set(range(n))
    actual_nodes = set(graph.nodes())
    if actual_nodes == expected_nodes:
        checks.append(Check("node_count", PASS, f"{n_nodes} nodes, IDs 0..{n - 1}"))
    else:
        missing = sorted(expected_nodes - actual_nodes)[:5]
        extra = sorted(actual_nodes - expected_nodes, key=repr)[:5]
        checks.append(
            Check(
                "node_count",
                FAIL,
                f"expected {n} nodes with IDs 0..{n - 1}, got {n_nodes}; "
                f"missing={missing} unexpected={extra}",
            )
        )

    # 2. Undirected.
    checks.append(
        Check("undirected", PASS, "nx.Graph")
        if not graph.is_directed()
        else Check("undirected", FAIL, "graph reports is_directed() == True")
    )

    # 3. No self-loops.
    loops = nx.number_of_selfloops(graph)
    checks.append(
        Check("no_self_loops", PASS, "0 self-loops")
        if loops == 0
        else Check("no_self_loops", FAIL, f"{loops} self-loop(s)")
    )

    # 4. No parallel edges. A simple nx.Graph cannot hold them by construction,
    #    so this asserts the container type rather than counting duplicates.
    checks.append(
        Check("no_parallel_edges", PASS, "simple graph (not a multigraph)")
        if not graph.is_multigraph()
        else Check("no_parallel_edges", FAIL, "graph is a multigraph")
    )

    # 5. Mean degree 2E/N within tolerance of the target.
    observed = mean_degree(graph) if n_nodes else float("nan")
    if n_nodes == 0 or math.isnan(observed):
        checks.append(Check("mean_degree_tolerance", SKIP, "empty graph"))
    else:
        deviation = (observed - target_mean_degree) / target_mean_degree
        detail = (
            f"2E/N = 2*{n_edges}/{n_nodes} = {observed:.4f} vs target "
            f"{target_mean_degree} ({deviation:+.2%}); "
            f"target edge count {target_edge_count(n, target_mean_degree)}"
        )
        if abs(deviation) <= degree_tolerance:
            checks.append(Check("mean_degree_tolerance", PASS, detail))
        elif degree_exception_reason:
            checks.append(
                Check(
                    "mean_degree_tolerance",
                    EXCEPTION,
                    f"{detail} -- outside +/-{degree_tolerance:.0%}; "
                    f"model constraint: {degree_exception_reason}",
                )
            )
        else:
            checks.append(
                Check(
                    "mean_degree_tolerance",
                    FAIL,
                    f"{detail} -- outside +/-{degree_tolerance:.0%}",
                )
            )

    # 6. Connectivity.
    #    NetworkX's connectivity helpers are undirected-only and raise on a
    #    DiGraph, so a directed input is reported as a SKIP here rather than
    #    crashing the validator -- the "undirected" check above already
    #    records the real defect.
    if n_nodes == 0:
        connected = False
        checks.append(Check("connected", SKIP, "empty graph"))
    elif graph.is_directed():
        connected = False
        checks.append(
            Check("connected", SKIP, "not evaluated: graph is directed")
        )
    else:
        components = nx.number_connected_components(graph)
        connected = components == 1
        if connected:
            checks.append(Check("connected", PASS, "1 connected component"))
        else:
            sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
            detail = f"{components} components, sizes {sizes[:5]}"
            checks.append(
                Check("connected", FAIL if require_connected else EXCEPTION, detail)
            )

    return ValidationResult(
        topology=topology,
        n=n,
        seed=seed,
        n_nodes=n_nodes,
        n_edges=n_edges,
        mean_degree=observed,
        target_mean_degree=float(target_mean_degree),
        connected=connected,
        checks=checks,
    )


def check_reproducible(
    generator: Callable[..., Any],
    *args: Any,
    repeats: int = 2,
    **kwargs: Any,
) -> Check:
    """Call ``generator`` repeatedly with identical arguments and compare edges.

    Generators here return ``(graph, metadata)``; a bare graph is also accepted.
    Reproducibility is judged on the sorted edge set *and* the node set, so a
    generator that produced the same edges on a different node labelling would
    still be caught.
    """
    if repeats < 2:
        raise GraphParameterError(f"repeats must be >= 2, got {repeats}")

    signatures = []
    for _ in range(repeats):
        result = generator(*args, **kwargs)
        graph = result[0] if isinstance(result, tuple) else result
        signatures.append(
            (
                tuple(sorted(graph.nodes())),
                tuple(sorted(tuple(sorted(e)) for e in graph.edges())),
            )
        )

    if all(s == signatures[0] for s in signatures):
        return Check(
            "reproducible",
            PASS,
            f"{repeats} calls produced identical node and edge sets",
        )
    return Check(
        "reproducible",
        FAIL,
        f"{repeats} calls with identical arguments produced differing graphs",
    )
