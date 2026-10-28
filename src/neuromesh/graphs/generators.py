"""Seeded graph generators: ER, WS and BA.

All three generators return ``(G, metadata)`` where ``G`` is a simple,
undirected :class:`networkx.Graph` on integer nodes ``0 .. n-1`` and
``metadata`` is a JSON-serialisable :class:`GraphMetadata`.

Design notes that matter scientifically
---------------------------------------

**Mean degree is defined as 2E/N** throughout. The study compares
topologies *matched in mean degree*, so edge count is a controlled
variable, not an outcome. Each generator therefore targets
``E_target = round(k * N / 2)`` with ``k = 4``, i.e. ``E = 2N``.

**ER model choice.** Two models are supported:

``model="gnm"`` (default)
    The Erdos-Renyi ``G(N, M)`` model: ``M = E_target`` edges chosen uniformly
    at random. Mean degree is *exactly* ``k`` at every N by construction.
``model="gnp"``
    The Gilbert ``G(N, p)`` model with ``p = k / (N - 1)``, which matches
    ``k`` only *in expectation*.

``gnm`` is the default because ``gnp`` cannot satisfy the +/-10% mean-degree
requirement reliably at small N: at N = 7, E is Binomial(21, 2/3) with standard
deviation about 2.16 edges against an acceptance window of only
``E in {13, 14, 15}``. Both models are exposed so the choice stays auditable;
see ``docs/TRACK_A_WEEK1_README.md`` for the measured comparison.

**ER connectivity is in genuine tension with mean degree 4.** ``G(N, M)`` is
connected with high probability only above mean degree ``ln N``, and
``ln(112)`` is about 4.72, above 4. The expected number of isolated vertices at
N = 112 is about ``112 * exp(-4) = 2.0``. With ``ensure_connected=True``
(default) we therefore draw from the *connected* ER ensemble by rejection
sampling: redraw with a derived sub-seed until the graph is connected. This
keeps N and E exactly fixed, so mean degree stays exactly 4, but it is a
conditioned ensemble, not the unconditioned one -- it slightly depletes
low-degree nodes. The number of draws is recorded in ``metadata.attempts`` so
the conditioning is never invisible. Pass ``ensure_connected=False`` for the
raw draw.

**BA cannot reach mean degree 4 at small N under its standard formulation.**
``networkx.barabasi_albert_graph(N, m)`` yields exactly ``E = m(N - m)``, so
mean degree is ``2m(N - m)/N = 4 - 8/N`` for ``m = 2``: 2.857 at N = 7
(-28.6%) and 3.429 at N = 14 (-14.3%), both outside +/-10%. Using ``m = 3``
overshoots. Two variants are therefore provided:

``variant="standard"``
    Textbook BA. Honest about its deviation; fails the tolerance at N = 7, 14.
``variant="degree_matched"`` (default)
    Identical preferential-attachment dynamics, but the seed core is the
    complete graph ``K_(2m+1)`` instead of NetworkX's ``m`` isolated nodes.
    Then ``E = C(2m+1, 2) + m(N - 2m - 1) = m*N`` exactly, for every N, so
    mean degree is exactly ``2m = k``. Derivation: the core contributes
    ``m(2m+1)`` edges and the ``N - 2m - 1`` later arrivals contribute ``m``
    each, giving ``2m^2 + m + mN - 2m^2 - m = mN``.

    The trade-off is that the seed clique is dense. At N = 112 only 5 of 112
    nodes are seeded (4.5%) and the scale-free character is intact; at N = 7
    only 2 of 7 nodes arrive by preferential attachment, so the N = 7
    degree-matched BA graph is a clique-plus-two and is **not** meaningfully
    scale-free. N = 7 is already a declared boundary case, and topology
    comparisons start at N = 14.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from .spec import (
    GENERATOR_VERSION,
    TARGET_MEAN_DEGREE,
    GraphParameterError,
    GraphSpec,
)

__all__ = [
    "GraphMetadata",
    "generate_graph",
    "erdos_renyi",
    "watts_strogatz",
    "barabasi_albert",
    "target_edge_count",
    "mean_degree",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def mean_degree(graph: nx.Graph) -> float:
    """Return 2E/N, the mean degree."""
    n = graph.number_of_nodes()
    if n == 0:
        raise GraphParameterError("mean degree is undefined for an empty graph")
    return 2.0 * graph.number_of_edges() / n


def target_edge_count(n: int, target_mean_degree: float = TARGET_MEAN_DEGREE) -> int:
    """Edge count that realises ``target_mean_degree`` exactly (or nearest).

    ``E = round(k * N / 2)``. For ``k = 4`` this is ``2N`` and always integral,
    so no rounding error is introduced at any of the five population sizes.
    """
    return int(round(target_mean_degree * n / 2.0))


def _derive_seed(seed: int, attempt: int) -> int:
    """Deterministically derive the sub-seed for a rejection-sampling attempt.

    Uses :class:`numpy.random.SeedSequence`, whose stream is documented as
    stable across NumPy versions and platforms, so ``(seed, attempt)`` maps to
    the same graph on every machine. Note ``attempt=0`` does not reuse ``seed``
    verbatim; the mapping is still a pure function of ``seed``.
    """
    sequence = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(attempt),))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _validate_common(n: Any, seed: Any, target_mean_degree: Any) -> None:
    """Reject structurally invalid parameters shared by all three generators."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise GraphParameterError(f"n must be an integer, got {type(n).__name__}")
    if n < 3:
        raise GraphParameterError(
            f"n must be >= 3 to form a connected simple graph, got {n}"
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise GraphParameterError(f"seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise GraphParameterError(f"seed must be non-negative, got {seed}")
    if isinstance(target_mean_degree, bool) or not isinstance(
        target_mean_degree, (int, float, np.integer, np.floating)
    ):
        raise GraphParameterError(
            "target_mean_degree must be numeric, got "
            f"{type(target_mean_degree).__name__}"
        )
    if not math.isfinite(float(target_mean_degree)) or target_mean_degree <= 0:
        raise GraphParameterError(
            f"target_mean_degree must be positive and finite, got {target_mean_degree}"
        )
    if target_mean_degree > n - 1:
        raise GraphParameterError(
            f"target_mean_degree {target_mean_degree} is infeasible for n={n}: "
            f"the complete graph on {n} nodes has mean degree {n - 1}"
        )


def _reject_until_connected(
    draw: Callable[[int], nx.Graph],
    seed: int,
    ensure_connected: bool,
    max_attempts: int,
    label: str,
) -> tuple[nx.Graph, int]:
    """Draw graphs with derived sub-seeds until connected; return (G, attempts).

    With ``ensure_connected=False`` exactly one draw is made and returned as-is,
    connected or not.
    """
    if isinstance(max_attempts, bool) or not isinstance(
        max_attempts, (int, np.integer)
    ):
        raise GraphParameterError(
            f"max_attempts must be an integer, got {type(max_attempts).__name__}"
        )
    if max_attempts < 1:
        raise GraphParameterError(f"max_attempts must be >= 1, got {max_attempts}")

    for attempt in range(int(max_attempts)):
        graph = draw(_derive_seed(seed, attempt))
        if not ensure_connected or nx.is_connected(graph):
            return graph, attempt + 1

    raise RuntimeError(
        f"{label}: no connected graph found in {max_attempts} draws for seed={seed}. "
        "This indicates the requested (n, mean degree) is far below the "
        "connectivity threshold; raise max_attempts, or set ensure_connected=False "
        "and handle disconnection explicitly."
    )


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

@dataclass
class GraphMetadata:
    """Provenance for one generated graph. JSON-serialisable via :meth:`to_dict`.

    Deliberately carries no timestamp, so regenerating a fixture from the same
    seed produces a byte-identical file.
    """

    topology: str
    n: int
    seed: int
    target_mean_degree: float
    n_nodes: int
    n_edges: int
    mean_degree: float
    connected: bool
    n_components: int
    attempts: int
    degree_matched_exactly: bool
    params: dict = field(default_factory=dict)
    generator_version: str = GENERATOR_VERSION
    networkx_version: str = nx.__version__
    numpy_version: str = np.__version__

    def to_dict(self) -> dict:
        return asdict(self)


def _describe(
    graph: nx.Graph,
    *,
    topology: str,
    n: int,
    seed: int,
    target_mean_degree: float,
    attempts: int,
    params: dict,
) -> GraphMetadata:
    components = nx.number_connected_components(graph)
    return GraphMetadata(
        topology=topology,
        n=n,
        seed=seed,
        target_mean_degree=float(target_mean_degree),
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        mean_degree=mean_degree(graph),
        connected=components == 1,
        n_components=components,
        attempts=attempts,
        degree_matched_exactly=(
            graph.number_of_edges() == target_edge_count(n, target_mean_degree)
        ),
        params=params,
    )


# --------------------------------------------------------------------------
# A. Erdos-Renyi
# --------------------------------------------------------------------------

def erdos_renyi(
    n: int,
    seed: int = 0,
    *,
    target_mean_degree: float = TARGET_MEAN_DEGREE,
    model: str = "gnm",
    ensure_connected: bool = True,
    max_attempts: int = 1000,
) -> tuple[nx.Graph, GraphMetadata]:
    """Seeded Erdos-Renyi random graph with mean degree ``target_mean_degree``.

    Parameters
    ----------
    model:
        ``"gnm"`` (default) fixes the edge count at ``round(kN/2)`` so mean
        degree is exact. ``"gnp"`` uses ``p = k/(N-1)`` and matches ``k`` only
        in expectation -- it routinely violates the +/-10% tolerance at small N.
    ensure_connected:
        If True, reject and redraw (with derived sub-seeds) until connected.
        See the module docstring: this conditions the ensemble, and the draw
        count is recorded in ``metadata.attempts``.

    Returns
    -------
    (G, metadata)
    """
    _validate_common(n, seed, target_mean_degree)
    if model not in ("gnm", "gnp"):
        raise GraphParameterError(f"model must be 'gnm' or 'gnp', got {model!r}")

    n = int(n)
    edges = target_edge_count(n, target_mean_degree)
    probability = float(target_mean_degree) / (n - 1)

    if model == "gnm":
        def draw(sub_seed: int) -> nx.Graph:
            return nx.gnm_random_graph(n, edges, seed=sub_seed)

        params: dict = {"model": "gnm", "m": edges}
    else:
        def draw(sub_seed: int) -> nx.Graph:
            return nx.gnp_random_graph(n, probability, seed=sub_seed)

        params = {"model": "gnp", "p": probability}

    params["ensure_connected"] = ensure_connected
    graph, attempts = _reject_until_connected(
        draw, int(seed), ensure_connected, max_attempts, "erdos_renyi"
    )
    # Be explicit that the node set is exactly 0..n-1 even for a raw gnp draw.
    graph.add_nodes_from(range(n))
    return graph, _describe(
        graph,
        topology="er",
        n=n,
        seed=int(seed),
        target_mean_degree=target_mean_degree,
        attempts=attempts,
        params=params,
    )


# --------------------------------------------------------------------------
# B. Watts-Strogatz
# --------------------------------------------------------------------------

def watts_strogatz(
    n: int,
    seed: int = 0,
    *,
    target_mean_degree: float = TARGET_MEAN_DEGREE,
    rewiring_prob: float = 0.1,
    ensure_connected: bool = True,
    max_attempts: int = 1000,
) -> tuple[nx.Graph, GraphMetadata]:
    """Seeded Watts-Strogatz small-world graph.

    Starts from a ring lattice in which every node joins its ``k/2`` nearest
    neighbours on each side, then rewires each edge with probability
    ``rewiring_prob``. Rewiring never changes the edge count, so mean degree is
    exactly ``k`` at every N and for every rewiring probability.

    Parameters
    ----------
    rewiring_prob:
        Default 0.1. Watts and Strogatz (1998) show that the small-world regime
        -- clustering still near the lattice value while characteristic path
        length has already collapsed towards the random-graph value -- occupies
        roughly ``0.01 <= p <= 0.1``. p = 0.1 sits at the top of that window,
        giving the shortest paths available without losing lattice clustering,
        which suits a study of information propagation over 10 rounds. It is a
        parameter, never changed silently; the value used is recorded in
        ``metadata.params``.

    Raises
    ------
    GraphParameterError
        If ``target_mean_degree`` is not an even integer (a ring lattice admits
        only even degree) or is not smaller than ``n``.
    """
    _validate_common(n, seed, target_mean_degree)
    if float(target_mean_degree) != int(target_mean_degree):
        raise GraphParameterError(
            f"Watts-Strogatz needs an integer mean degree, got {target_mean_degree}"
        )
    k = int(target_mean_degree)
    if k % 2 != 0:
        raise GraphParameterError(
            f"Watts-Strogatz needs an even mean degree (ring lattice), got k={k}"
        )
    n = int(n)
    if k >= n:
        raise GraphParameterError(f"Watts-Strogatz needs k < n, got k={k}, n={n}")
    if isinstance(rewiring_prob, bool) or not isinstance(
        rewiring_prob, (int, float, np.integer, np.floating)
    ):
        raise GraphParameterError(
            f"rewiring_prob must be numeric, got {type(rewiring_prob).__name__}"
        )
    if not 0.0 <= float(rewiring_prob) <= 1.0:
        raise GraphParameterError(
            f"rewiring_prob must lie in [0, 1], got {rewiring_prob}"
        )

    def draw(sub_seed: int) -> nx.Graph:
        return nx.watts_strogatz_graph(n, k, float(rewiring_prob), seed=sub_seed)

    graph, attempts = _reject_until_connected(
        draw, int(seed), ensure_connected, max_attempts, "watts_strogatz"
    )
    return graph, _describe(
        graph,
        topology="ws",
        n=n,
        seed=int(seed),
        target_mean_degree=target_mean_degree,
        attempts=attempts,
        params={
            "k": k,
            "rewiring_prob": float(rewiring_prob),
            "ensure_connected": ensure_connected,
        },
    )


# --------------------------------------------------------------------------
# C. Barabasi-Albert
# --------------------------------------------------------------------------

def barabasi_albert(
    n: int,
    seed: int = 0,
    *,
    target_mean_degree: float = TARGET_MEAN_DEGREE,
    variant: str = "degree_matched",
    m: int | None = None,
) -> tuple[nx.Graph, GraphMetadata]:
    """Seeded Barabasi-Albert preferential-attachment graph.

    Parameters
    ----------
    variant:
        ``"degree_matched"`` (default) seeds the growth with the complete graph
        ``K_(2m+1)``, which makes ``E = m*N`` and therefore mean degree exactly
        ``2m = k`` at every N. ``"standard"`` is textbook BA, with
        ``E = m(N - m)`` and mean degree ``k - 2m^2/N`` -- below tolerance at
        N = 7 and N = 14. Neither variant is a silent substitution: the choice
        and its consequence are recorded in ``metadata.params`` and
        ``metadata.degree_matched_exactly``.
    m:
        Edges contributed per arriving node. Defaults to ``k/2`` (= 2 for
        k = 4), the value that makes BA's asymptotic mean degree ``2m`` equal
        the target. An explicit ``m`` with ``2m != k`` is rejected rather than
        quietly producing a graph matched on ``2m``; likewise an odd or
        fractional ``k``, which no integer ``m`` can reach.

    Notes
    -----
    BA graphs are connected by construction (every arrival attaches to existing
    nodes and the seed core is connected), so no rejection sampling is needed;
    connectivity is still measured and reported.
    """
    _validate_common(n, seed, target_mean_degree)
    if variant not in ("degree_matched", "standard"):
        raise GraphParameterError(
            f"variant must be 'degree_matched' or 'standard', got {variant!r}"
        )
    n = int(n)

    explicit_m = m is not None
    if m is None:
        m_value = int(round(float(target_mean_degree) / 2.0))
    else:
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
            raise GraphParameterError(f"m must be an integer, got {type(m).__name__}")
        m_value = int(m)
    if m_value < 1:
        raise GraphParameterError(f"m must be >= 1, got {m_value}")
    if m_value >= n:
        raise GraphParameterError(f"m must be < n, got m={m_value}, n={n}")

    if variant == "degree_matched":
        core_size = 2 * m_value + 1
        if n < core_size:
            raise GraphParameterError(
                f"degree_matched BA needs n >= 2m+1 = {core_size} to seat its "
                f"complete seed core, got n={n}; use variant='standard'"
            )
        initial_graph: nx.Graph | None = nx.complete_graph(core_size)
        params: dict = {
            "variant": "degree_matched",
            "m": m_value,
            "initial_graph": f"complete_graph({core_size})",
            "core_size": core_size,
        }
    else:
        initial_graph = None
        core_size = m_value
        params = {
            "variant": "standard",
            "m": m_value,
            "initial_graph": "networkx_default",
            "core_size": core_size,
        }

    # BA's mean degree is governed by m, not by target_mean_degree: it is 2m
    # exactly for the degree-matched variant and 2m - 2m^2/N for the standard
    # one. So an m that disagrees with the requested mean degree silently
    # produces a graph matched on a *different* degree than the caller asked
    # for -- previously visible only afterwards, via
    # metadata.degree_matched_exactly. Caught here instead, and deliberately
    # after the seed-core feasibility check above so that the more specific
    # error still wins.
    if explicit_m and 2 * m_value != float(target_mean_degree):
        raise GraphParameterError(
            f"inconsistent Barabasi-Albert parameters: m={m_value} yields mean "
            f"degree 2m = {2 * m_value}, not the requested "
            f"target_mean_degree={target_mean_degree}. Pass m={target_mean_degree / 2:g} "
            f"(or leave m unset), or request target_mean_degree={2 * m_value}."
        )
    if not explicit_m and 2 * m_value != float(target_mean_degree):
        raise GraphParameterError(
            f"target_mean_degree={target_mean_degree} is not reachable by "
            "Barabasi-Albert: each arriving node contributes an integer number "
            f"of edges m, so only even mean degrees are attainable (nearest: "
            f"{2 * m_value})."
        )

    graph = nx.barabasi_albert_graph(
        n, m_value, seed=_derive_seed(int(seed), 0), initial_graph=initial_graph
    )
    return graph, _describe(
        graph,
        topology="ba",
        n=n,
        seed=int(seed),
        target_mean_degree=target_mean_degree,
        attempts=1,
        params=params,
    )


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------

_GENERATORS: dict = {
    "er": erdos_renyi,
    "ws": watts_strogatz,
    "ba": barabasi_albert,
}


def generate_graph(spec: GraphSpec) -> tuple[nx.Graph, GraphMetadata]:
    """Generate the graph described by ``spec``.

    This is the single entry point callers should use, so that adapting to
    the project-wide interfaces later touches one function.
    """
    if not isinstance(spec, GraphSpec):
        raise GraphParameterError(f"expected a GraphSpec, got {type(spec).__name__}")
    generator = _GENERATORS[spec.topology]
    try:
        return generator(
            spec.n,
            spec.seed,
            target_mean_degree=spec.target_mean_degree,
            **dict(spec.params),
        )
    except TypeError as exc:  # unknown keyword in spec.params
        raise GraphParameterError(
            f"invalid params {dict(spec.params)!r} for topology "
            f"{spec.topology!r}: {exc}"
        ) from exc
