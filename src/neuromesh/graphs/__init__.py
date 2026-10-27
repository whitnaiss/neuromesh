"""Track A (Mihir): graph generation, roles, node removal, structural metrics.

Public entry points only. Implement them in this package (e.g. generators.py,
yeo_sbm.py, roles.py, removal.py, metrics.py) and keep these signatures.
"""

from __future__ import annotations

import networkx as nx

from neuromesh.contracts import DEFAULT_MEAN_DEGREE, RemovalSpec, Topology


def generate_graph(
    topology: Topology, n: int, mean_degree: float = DEFAULT_MEAN_DEGREE, seed: int = 0
) -> nx.Graph:
    """Connected graph with nodes 0..n-1, mean degree within +/-10% (N >= 14)."""
    raise NotImplementedError("Track A: generate_graph")


def assign_roles(graph: nx.Graph, seed: int) -> nx.Graph:
    """Give every node a YEO7 role, exactly n/7 per role; SBM keeps its blocks."""
    raise NotImplementedError("Track A: assign_roles")


def remove_nodes(graph: nx.Graph, spec: RemovalSpec, seed: int) -> nx.Graph:
    """Random or highest-degree-first removal; never removes every readout node."""
    raise NotImplementedError("Track A: remove_nodes")


def structural_metrics(graph: nx.Graph) -> dict:
    """Path length, clustering, modularity, max degree, input-readout distance, hub roles."""
    raise NotImplementedError("Track A: structural_metrics")


__all__ = ["assign_roles", "generate_graph", "remove_nodes", "structural_metrics"]
