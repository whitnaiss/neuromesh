"""Block-structured graph sampling, shared by the modular and Yeo-SBM topologies.

Both topologies are stochastic block models over the seven functional groups:
they differ only in the 7x7 weight matrix that sets the relative preference for
each pair of blocks. ``modular`` uses a uniform two-value matrix (one value on
the diagonal, one off it); ``yeo_sbm`` uses an empirically derived matrix. The
sampling machinery below is therefore written once.

Why a fixed edge budget rather than independent Bernoulli trials
----------------------------------------------------------------
A textbook SBM draws every node pair independently with probability
``p[block(i), block(j)]``. Mean degree is then matched only *in expectation*,
and the realised value fluctuates from seed to seed -- exactly the problem that
made ``G(N, p)`` unusable for Erdos-Renyi at small N (see ``generators.py``).
Mean degree is this study's controlled variable, so the same fix is applied
here: the total edge count is fixed at ``E = round(k*N/2)`` and that budget is
divided among the 21 + 7 = 28 block pairs in proportion to

    demand[a][b] = weight[a][b] * capacity[a][b]

where ``capacity`` is the number of distinct node pairs the block pair admits.
Within each block pair the required number of edges is then drawn uniformly
without replacement. The result is an SBM *conditioned on its total edge
count*, which keeps mean degree exactly ``k`` at every N while preserving the
relative block preferences of the weight matrix.

This conditioning is a deliberate, documented departure from the unconditioned
SBM, in the same spirit as the connected-ER ensemble. See
:func:`sample_block_graph` for the two conditioning steps and the provenance
fields that record them.

Finite-size limits
------------------
A block pair cannot receive more edges than it has node pairs. Allocation is
therefore clamped to capacity and the surplus is redistributed to block pairs
that still have room. At small N this binds hard:

* ``N = 7``  -- one node per block, so every within-block capacity is 0 and
  **no within-block edge can exist**. The graph is entirely between-block and
  carries no community structure at all.
* ``N = 14`` -- two nodes per block, so within-block capacity is 1 per block
  (7 in total, against a budget of 28 edges). The within-block share is capped
  at 25% however strong the requested preference.

Neither case is a defect; both are properties of the design at those sizes.
They are reported through :func:`allocate_block_edges` (which records whether
clamping occurred) rather than hidden.
"""

from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np

from .spec import GraphParameterError

__all__ = [
    "N_BLOCKS",
    "allocate_block_edges",
    "balanced_block_sizes",
    "block_assignment",
    "community_profile",
    "pair_capacity",
    "sample_block_graph",
    "validate_block_weights",
]

#: Number of blocks: one per Yeo functional network.
N_BLOCKS: int = 7


# --------------------------------------------------------------------------
# block layout
# --------------------------------------------------------------------------

def balanced_block_sizes(n: int, n_blocks: int = N_BLOCKS) -> tuple[int, ...]:
    """Sizes of ``n_blocks`` equal blocks covering ``n`` nodes.

    The study's population sizes (7, 14, 28, 56, 112) are all divisible by 7,
    so the blocks are exactly equal. An ``n`` that is not divisible is rejected
    rather than silently unbalanced, because balanced functional groups are a
    stated premise of the design.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise GraphParameterError(f"n must be an integer, got {type(n).__name__}")
    if n_blocks < 1:
        raise GraphParameterError(f"n_blocks must be >= 1, got {n_blocks}")
    if int(n) % int(n_blocks) != 0:
        raise GraphParameterError(
            f"n={n} is not divisible by {n_blocks}: balanced functional groups "
            f"require n/{n_blocks} to be an integer (the study uses "
            f"n = 7, 14, 28, 56, 112)"
        )
    return tuple([int(n) // int(n_blocks)] * int(n_blocks))


def block_assignment(n: int, n_blocks: int = N_BLOCKS) -> tuple[int, ...]:
    """Block index of each node ``0 .. n-1``, in contiguous equal runs.

    Node ``i`` belongs to block ``i // (n / n_blocks)``. Contiguous rather than
    interleaved so that a block's nodes form a readable ID range in logs and
    fixtures, and so that block membership is recoverable from a node ID alone.
    """
    sizes = balanced_block_sizes(n, n_blocks)
    assignment: list[int] = []
    for block, size in enumerate(sizes):
        assignment.extend([block] * size)
    return tuple(assignment)


def pair_capacity(sizes: tuple[int, ...]) -> np.ndarray:
    """Number of distinct node pairs available to each block pair.

    ``capacity[a][a] = C(m_a, 2)`` (self-loops excluded) and
    ``capacity[a][b] = m_a * m_b`` for ``a != b``. Returned symmetric; only the
    upper triangle including the diagonal is meaningful for allocation.
    """
    k = len(sizes)
    capacity = np.zeros((k, k), dtype=np.int64)
    for a in range(k):
        capacity[a, a] = sizes[a] * (sizes[a] - 1) // 2
        for b in range(a + 1, k):
            capacity[a, b] = capacity[b, a] = sizes[a] * sizes[b]
    return capacity


# --------------------------------------------------------------------------
# weight matrix validation
# --------------------------------------------------------------------------

def validate_block_weights(weights, n_blocks: int = N_BLOCKS) -> np.ndarray:
    """Return ``weights`` as a validated ``n_blocks x n_blocks`` float array.

    Requirements, each checked rather than assumed: square and of the expected
    size, finite, non-negative, symmetric (the graph is undirected, so an
    asymmetric preference matrix is meaningless), and not identically zero.

    Only *relative* magnitudes matter -- the matrix is rescaled to the edge
    budget by :func:`allocate_block_edges` -- so no normalisation is required
    of the caller.
    """
    array = np.asarray(weights, dtype=float)
    if array.ndim != 2 or array.shape != (n_blocks, n_blocks):
        raise GraphParameterError(
            f"block weights must be a {n_blocks}x{n_blocks} matrix, got shape "
            f"{array.shape if array.ndim else type(weights).__name__}"
        )
    if not np.isfinite(array).all():
        raise GraphParameterError("block weights must all be finite")
    if (array < 0).any():
        raise GraphParameterError(
            f"block weights must be non-negative; minimum is {array.min()}"
        )
    if not np.allclose(array, array.T, rtol=1e-9, atol=1e-12):
        raise GraphParameterError(
            "block weights must be symmetric: the generated graph is undirected, "
            "so weight[a][b] and weight[b][a] describe the same block pair"
        )
    if array.sum() <= 0:
        raise GraphParameterError("block weights must not be identically zero")
    return array


# --------------------------------------------------------------------------
# edge budget allocation
# --------------------------------------------------------------------------

def allocate_block_edges(
    sizes: tuple[int, ...],
    weights: np.ndarray,
    n_edges: int,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Split ``n_edges`` among block pairs in proportion to weight x capacity.

    Returns ``(allocation, info)``. ``allocation`` is a symmetric integer matrix
    whose upper triangle (with diagonal) sums to exactly ``n_edges``.

    Method
    ------
    1. ``demand[a][b] = weight[a][b] * capacity[a][b]``. Capacity enters the
       demand because a weight is a *per-pair* preference: a block pair with
       twice the node pairs should receive twice the edges at equal preference.
    2. Scale demand to the budget and take floors, then hand out the remaining
       edges by largest fractional remainder.
    3. Clamp every block pair to its capacity, redistributing any surplus to
       pairs that still have room. ``info["capacity_clamped"]`` records whether
       this happened, so a caller can tell a freely-sampled graph from one
       whose structure was limited by finite size.

    Tie-breaking
    ------------
    Equal remainders are common whenever demand is uniform, and at N = 7 *every*
    cell ties: each of the 21 between-block pairs is owed 14/21 of an edge and
    can hold exactly one, so step 2 alone decides the entire graph. Breaking
    those ties by block-pair index would then always select the same 14
    low-index pairs, making the N = 7 graph both seed-invariant and
    systematically biased.

    With ``rng`` supplied, ties are therefore broken by a random permutation
    drawn from it, which is what :func:`sample_block_graph` does. Without
    ``rng`` the index order is used, keeping the function a pure, deterministic
    mapping for direct callers and tests.
    """
    k = len(sizes)
    capacity = pair_capacity(sizes)
    total_capacity = int(capacity[np.triu_indices(k)].sum())
    if n_edges > total_capacity:
        raise GraphParameterError(
            f"cannot place {n_edges} edges: the block structure admits at most "
            f"{total_capacity} distinct node pairs"
        )
    if n_edges < 0:
        raise GraphParameterError(f"n_edges must be non-negative, got {n_edges}")

    pairs = [(a, b) for a in range(k) for b in range(a, k)]
    demand = np.array([weights[a, b] * capacity[a, b] for a, b in pairs], dtype=float)
    caps = np.array([capacity[a, b] for a, b in pairs], dtype=np.int64)

    if demand.sum() <= 0:
        # Every block pair with capacity has zero weight: fall back to a
        # capacity-proportional split so the budget can still be placed, and
        # say so in the provenance.
        demand = caps.astype(float)
        degenerate = True
    else:
        degenerate = False

    raw = demand * (n_edges / demand.sum())
    alloc = np.minimum(np.floor(raw).astype(np.int64), caps)
    remainder = raw - np.floor(raw)

    if rng is None:
        order = sorted(range(len(pairs)), key=lambda i: (-remainder[i], i))
    else:
        jitter = rng.permutation(len(pairs))
        order = sorted(range(len(pairs)), key=lambda i: (-remainder[i], jitter[i]))
    deficit = int(n_edges - alloc.sum())
    while deficit > 0:
        progressed = False
        for i in order:
            if deficit == 0:
                break
            if alloc[i] < caps[i]:
                alloc[i] += 1
                deficit -= 1
                progressed = True
        if not progressed:  # pragma: no cover - guarded by the capacity check
            raise GraphParameterError(
                f"cannot place {n_edges} edges within the available capacity"
            )

    # A block pair was limited by capacity when proportional demand asked for
    # more edges than it has node pairs. Comparing the *unrounded* demand with
    # capacity is what detects this: at N = 14 each within-block pair is owed
    # 1.8 edges but can hold only 1, which a comparison of the floored value
    # would miss entirely.
    clamped = bool((raw > caps + 1e-9).any())
    within_capacity = int(sum(capacity[a, a] for a in range(k)))

    allocation = np.zeros((k, k), dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        allocation[a, b] = allocation[b, a] = int(alloc[i])

    within = int(sum(allocation[a, a] for a in range(k)))
    info = {
        "n_within_edges": within,
        "n_between_edges": int(n_edges - within),
        "within_fraction": (within / n_edges) if n_edges else 0.0,
        "capacity_clamped": clamped,
        "within_capacity": within_capacity,
        # False only at N = 7, where each block holds a single node and no
        # within-block edge can exist at any preference.
        "within_structure_possible": within_capacity > 0,
        "degenerate_weights": degenerate,
    }
    return allocation, info


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _draw_block_graph(
    n: int,
    blocks: tuple[int, ...],
    allocation: np.ndarray,
    rng: np.random.Generator,
) -> nx.Graph:
    """Draw one graph with exactly ``allocation[a][b]`` edges in each block pair."""
    k = allocation.shape[0]
    members: list[list[int]] = [[] for _ in range(k)]
    for node, block in enumerate(blocks):
        members[block].append(node)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for a in range(k):
        for b in range(a, k):
            wanted = int(allocation[a, b])
            if wanted == 0:
                continue
            if a == b:
                candidates = list(itertools.combinations(members[a], 2))
            else:
                candidates = list(itertools.product(members[a], members[b]))
            chosen = rng.choice(len(candidates), size=wanted, replace=False)
            for index in chosen:
                u, v = candidates[int(index)]
                graph.add_edge(int(u), int(v))
    return graph


def sample_block_graph(
    n: int,
    weights,
    seed: int = 0,
    *,
    target_mean_degree: float = 4,
    n_blocks: int = N_BLOCKS,
    ensure_connected: bool = True,
    max_attempts: int = 1000,
) -> tuple[nx.Graph, dict]:
    """Sample a connected block-structured graph with an exact edge budget.

    Returns ``(graph, info)``. Every node carries a ``community`` attribute
    holding its block index; ``info`` records the realised block structure and
    both conditioning steps.

    The graph is conditioned twice, and neither conditioning is invisible:

    ``exact edge budget``
        The total edge count is fixed at ``round(k*N/2)`` rather than left to
        fluctuate, so mean degree is exactly ``k``. ``info["allocation"]`` is
        the realised per-block-pair edge count.
    ``connectivity``
        With ``ensure_connected=True`` the draw is repeated with derived
        sub-seeds until the graph is connected, as the Erdos-Renyi generator
        does. ``info["attempts"]`` is the number of draws.

    An unconditioned sample is available with ``ensure_connected=False``; the
    edge budget is always exact and is not optional, because mean-degree
    matching is the study's controlled variable.
    """
    from .generators import _derive_seed, _validate_common, target_edge_count

    _validate_common(n, seed, target_mean_degree)
    n = int(n)
    sizes = balanced_block_sizes(n, n_blocks)
    matrix = validate_block_weights(weights, n_blocks)
    n_edges = target_edge_count(n, target_mean_degree)
    blocks = block_assignment(n, n_blocks)

    if isinstance(max_attempts, bool) or not isinstance(max_attempts, (int, np.integer)):
        raise GraphParameterError(
            f"max_attempts must be an integer, got {type(max_attempts).__name__}"
        )
    if max_attempts < 1:
        raise GraphParameterError(f"max_attempts must be >= 1, got {max_attempts}")

    graph = None
    allocation: np.ndarray | None = None
    info: dict = {}
    attempts = 0
    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(_derive_seed(int(seed), attempt))
        # Allocated inside the loop so that a redraw also redraws the
        # tie-breaking, which is the only source of variation at N = 7.
        allocation, info = allocate_block_edges(sizes, matrix, n_edges, rng=rng)
        graph = _draw_block_graph(n, blocks, allocation, rng)
        attempts = attempt + 1
        if not ensure_connected or nx.is_connected(graph):
            break
    else:
        raise RuntimeError(
            f"block model: no connected graph found in {max_attempts} draws for "
            f"seed={seed}, n={n}. The requested block structure may be too "
            "fragmented at this mean degree; raise max_attempts, or set "
            "ensure_connected=False and handle disconnection explicitly."
        )

    assert graph is not None
    nx.set_node_attributes(graph, {i: b for i, b in enumerate(blocks)}, "community")
    info.update(
        {
            "attempts": attempts,
            "block_sizes": list(sizes),
            "n_blocks": int(n_blocks),
            "allocation": [[int(x) for x in row] for row in allocation],
            "ensure_connected": bool(ensure_connected),
        }
    )
    return graph, info


# --------------------------------------------------------------------------
# realised community structure
# --------------------------------------------------------------------------

def community_profile(graph: nx.Graph) -> dict:
    """Measure the community structure a block-model graph actually realised.

    Returns the within/between edge split and Newman's modularity ``Q`` of the
    planted partition. Reports what a graph *is*, as a check on the generator;
    the study-wide structural-metric covariates are a separate, later piece of
    work and do not live here.

    ``Q`` compares the within-community edge fraction with what the same degree
    sequence would give at random, so ``Q ~ 0`` means "no more modular than
    chance". For seven equal blocks the maximum attainable value is
    ``1 - 1/7 ~ 0.857``.
    """
    if not isinstance(graph, nx.Graph):
        raise GraphParameterError(
            f"expected a networkx.Graph, got {type(graph).__name__}"
        )
    communities = nx.get_node_attributes(graph, "community")
    if len(communities) != graph.number_of_nodes():
        raise GraphParameterError(
            "every node must carry a 'community' attribute; this graph has "
            f"{len(communities)} of {graph.number_of_nodes()}"
        )

    groups: dict[int, set] = {}
    for node, community in communities.items():
        groups.setdefault(community, set()).add(node)
    partition = [groups[key] for key in sorted(groups)]

    within = sum(1 for u, v in graph.edges() if communities[u] == communities[v])
    n_edges = graph.number_of_edges()
    modularity = (
        nx.algorithms.community.modularity(graph, partition) if n_edges else math.nan
    )
    return {
        "n_communities": len(partition),
        "community_sizes": [len(p) for p in partition],
        "n_within_edges": within,
        "n_between_edges": n_edges - within,
        "within_fraction": (within / n_edges) if n_edges else 0.0,
        "modularity": float(modularity),
    }
