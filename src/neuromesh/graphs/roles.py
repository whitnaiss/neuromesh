"""Balanced assignment of the seven Yeo functional-network roles to agents.

Every node receives exactly one role from :data:`neuromesh.contracts.YEO7`,
with exactly ``N/7`` nodes per role at every population size the study uses
(7, 14, 28, 56, 112 are all divisible by 7). Information enters the network at
``INPUT_ROLE`` agents and is read out at ``READOUT_ROLE`` agents, so an
unbalanced or incomplete assignment would silently change what every task
measures.

Two assignment regimes, and why they differ
-------------------------------------------
``er``, ``ws``, ``ba`` -- **randomised, balanced.**
    These topologies have no built-in community structure, so a role is just a
    label attached to a position in the graph. Labels are shuffled with a
    seeded generator and dealt out, which is what makes role placement a
    controlled variable: redrawing per seed averages over the accident of
    which structural position happened to receive the readout role.

``modular``, ``yeo_sbm`` -- **determined by the block structure.**
    These graphs are generated *as* seven blocks, and for ``yeo_sbm`` block
    ``b`` is functional network ``b`` by construction: the generator's weight
    matrix is indexed by the same seven networks, in the same order. Reshuffling
    roles across blocks would break the correspondence between the empirical
    connectivity matrix and the roles the agents actually play, which is the
    entire point of that topology. Role is therefore read off the node's
    ``community`` attribute, and the assignment does not depend on the seed.

The seed-independence of the second regime is a property of the design, not an
oversight: once a node's community is fixed, its role carries no remaining
freedom. :func:`assign_roles` records which regime was used in
``graph.graph["role_assignment"]`` so a consumer never has to guess.

Interpretation warning: the two regimes are a confound
------------------------------------------------------
Because role placement is generated differently for the two groups of
topologies, **a performance difference between a block topology and a
non-block topology cannot be attributed to the arrangement of links alone.**
Two things vary together across that boundary:

1. the topology itself, and
2. how roles sit on it -- randomised and redrawn per seed for ``er``/``ws``/
   ``ba``, so results average over placement; perfectly aligned with the
   communities and identical at every seed for ``modular``/``yeo_sbm``, so
   results carry zero placement variance and maximal role-community alignment.

Comparisons *within* a group (ER vs WS vs BA, or modular vs Yeo-SBM) are not
affected and remain clean reads on topology. Comparisons *across* the boundary
are confounded, and any such claim has to say so.

This is a deliberate consequence of the design rather than a defect: for
``yeo_sbm`` the whole point is that block *b* is functional network *b*, so
the alignment cannot be randomised away without destroying the topology's
meaning. Disentangling the two factors would need an extra condition -- for
example a block topology run with deliberately shuffled role labels -- which
is not part of the current experimental design.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from neuromesh.contracts import YEO7

from .spec import GraphParameterError

__all__ = ["ROLE_SEED_SALT", "assign_roles", "role_counts", "roles_are_balanced"]

#: Mixed into the seed so role assignment draws from a different random stream
#: than graph generation. Without it, ``assign_roles(G, s)`` would consume the
#: same bits that produced ``G`` at seed ``s``, correlating the two.
ROLE_SEED_SALT: int = 0x59454F37  # "YEO7"


def _validate_graph_for_roles(graph) -> list[int]:
    """Check the graph can carry a balanced assignment; return ordered nodes."""
    if not isinstance(graph, nx.Graph):
        raise GraphParameterError(
            f"expected a networkx.Graph, got {type(graph).__name__}"
        )
    n = graph.number_of_nodes()
    if n == 0:
        raise GraphParameterError("cannot assign roles to an empty graph")
    if set(graph.nodes()) != set(range(n)):
        raise GraphParameterError(
            f"nodes must be exactly the integers 0..{n - 1} (the project's agent "
            "ID convention) before roles can be assigned"
        )
    if n % len(YEO7) != 0:
        raise GraphParameterError(
            f"n={n} is not divisible by {len(YEO7)}: a balanced assignment needs "
            f"n/{len(YEO7)} agents per functional network (the study uses "
            "n = 7, 14, 28, 56, 112)"
        )
    # range(n) is already the deterministic order; built explicitly so the
    # result never depends on dict or hash ordering.
    return list(range(n))


def _validate_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise GraphParameterError(f"seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise GraphParameterError(f"seed must be non-negative, got {seed}")
    return int(seed)


def _community_roles(graph: nx.Graph, nodes: list[int]) -> list[str] | None:
    """Roles implied by a planted community structure, or None if there is none."""
    communities = nx.get_node_attributes(graph, "community")
    if not communities:
        return None
    if len(communities) != len(nodes):
        raise GraphParameterError(
            "a partially community-labelled graph is ambiguous: "
            f"{len(communities)} of {len(nodes)} nodes carry a 'community' "
            "attribute"
        )
    values = sorted({int(c) for c in communities.values()})
    if values != list(range(len(YEO7))):
        raise GraphParameterError(
            f"community indices must be exactly 0..{len(YEO7) - 1} to map onto "
            f"the seven functional networks; got {values}"
        )
    return [YEO7[int(communities[node])] for node in nodes]


def assign_roles(graph: nx.Graph, seed: int) -> nx.Graph:
    """Give every node a YEO7 role, exactly ``n/7`` per role.

    Modifies ``graph`` in place and returns it, so both
    ``assign_roles(g, 0)`` and ``g = assign_roles(g, 0)`` behave as expected.
    Only node attributes are written: the node set and **every edge are left
    untouched**.

    Parameters
    ----------
    graph:
        A graph whose nodes are the integers ``0 .. n-1`` with ``n`` divisible
        by 7. A graph carrying a ``community`` attribute on every node (as
        ``modular`` and ``yeo_sbm`` produce) keeps its blocks: role is read off
        the community index.
    seed:
        Non-negative integer. Identical ``(graph, seed)`` always yields an
        identical assignment. For community-structured graphs the assignment is
        fixed by the blocks and the seed has no effect; this is recorded in
        ``graph.graph["role_assignment"]["regime"]``.

    Returns
    -------
    networkx.Graph
        The same object, with a ``role`` attribute on every node and a
        ``role_assignment`` provenance dict in ``graph.graph``.

    Raises
    ------
    GraphParameterError
        For a non-graph input, an empty graph, nodes that are not ``0..n-1``,
        an ``n`` not divisible by 7, an invalid seed, or an inconsistent
        community labelling.
    """
    nodes = _validate_graph_for_roles(graph)
    seed_value = _validate_seed(seed)
    per_role = len(nodes) // len(YEO7)

    community_roles = _community_roles(graph, nodes)
    if community_roles is not None:
        roles = community_roles
        regime = "community"
    else:
        sequence = np.random.SeedSequence(entropy=[seed_value, ROLE_SEED_SALT])
        rng = np.random.default_rng(sequence)
        labels = np.repeat(np.arange(len(YEO7)), per_role)
        rng.shuffle(labels)
        roles = [YEO7[int(index)] for index in labels]
        regime = "randomised_balanced"

    nx.set_node_attributes(graph, dict(zip(nodes, roles, strict=True)), "role")
    graph.graph["role_assignment"] = {
        "regime": regime,
        "seed": seed_value,
        "per_role": per_role,
        "roles": list(YEO7),
        "seed_dependent": regime == "randomised_balanced",
    }
    return graph


def role_counts(graph: nx.Graph) -> dict[str, int]:
    """Number of nodes carrying each YEO7 role; missing roles count as 0."""
    if not isinstance(graph, nx.Graph):
        raise GraphParameterError(
            f"expected a networkx.Graph, got {type(graph).__name__}"
        )
    assigned = nx.get_node_attributes(graph, "role")
    counts = dict.fromkeys(YEO7, 0)
    for role in assigned.values():
        if role not in counts:
            raise GraphParameterError(
                f"node carries unknown role {role!r}; expected one of {list(YEO7)}"
            )
        counts[role] += 1
    return counts


def roles_are_balanced(graph: nx.Graph) -> bool:
    """True when every node has a role and all seven roles appear ``n/7`` times."""
    n = graph.number_of_nodes()
    if n == 0 or n % len(YEO7) != 0:
        return False
    if len(nx.get_node_attributes(graph, "role")) != n:
        return False
    return set(role_counts(graph).values()) == {n // len(YEO7)}
