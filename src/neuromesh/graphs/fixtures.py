"""Reusable graph fixtures.

Fixtures are plain JSON so that any consumer can load them with the standard
library alone -- no NetworkX required if only the adjacency is needed.

File format (``schema_version = 1``)
------------------------------------

.. code-block:: json

    {
      "schema_version": 1,
      "metadata": {"topology": "er", "n": 14, "seed": 0, "...": "..."},
      "nodes": [0, 1, 2, "..."],
      "edges": [[0, 3], [0, 9], "..."]
    }

Guarantees
~~~~~~~~~~
* ``nodes`` is exactly ``[0, 1, ..., n-1]`` -- consecutive integer agent IDs.
* Every edge is ``[u, v]`` with ``u < v``; the list is sorted; no duplicates;
  no self-loops. Edges are undirected, so each appears once.
* ``metadata`` carries the full generation provenance, including the seed,
  the generator version and the NetworkX/NumPy versions used.
* No timestamps, so regenerating from the same seed yields byte-identical
  files (the manifest alone records when a set was built).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from .generators import GraphMetadata, generate_graph, mean_degree
from .spec import (
    DEFAULT_FIXTURE_SEED,
    GENERATOR_VERSION,
    POPULATION_SIZES,
    TOPOLOGIES,
    GraphParameterError,
    GraphSpec,
)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_FIXTURE_DIR",
    "fixture_filename",
    "render_record",
    "graph_to_record",
    "record_to_graph",
    "save_fixture",
    "load_fixture",
    "load_graph",
    "generate_fixture_set",
    "load_fixture_set",
]

SCHEMA_VERSION = 1

#: Default location, relative to the repository root.
DEFAULT_FIXTURE_DIR = Path("data") / "fixtures" / "graphs"

MANIFEST_NAME = "manifest.json"


def fixture_filename(topology: str, n: int, seed: int) -> str:
    """Canonical fixture filename, e.g. ``er_n014_seed0.json``.

    Zero-padded N keeps a directory listing in population order.
    """
    return f"{topology}_n{int(n):03d}_seed{int(seed)}.json"


def graph_to_record(graph: nx.Graph, metadata: GraphMetadata) -> dict:
    """Convert a graph plus its metadata into the JSON-serialisable record."""
    edges = sorted(tuple(sorted((int(u), int(v)))) for u, v in graph.edges())
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata.to_dict(),
        "nodes": sorted(int(v) for v in graph.nodes()),
        "edges": [list(e) for e in edges],
    }


def record_to_graph(record: dict) -> tuple[nx.Graph, dict]:
    """Rebuild ``(graph, metadata_dict)`` from a fixture record.

    Validates the record's structural promises rather than trusting them, so a
    corrupted or hand-edited fixture fails loudly at load time.
    """
    if not isinstance(record, dict):
        raise GraphParameterError(f"fixture record must be a dict, got {type(record)}")
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        raise GraphParameterError(
            f"unsupported fixture schema_version {version!r}; "
            f"this build reads version {SCHEMA_VERSION}"
        )
    for key in ("metadata", "nodes", "edges"):
        if key not in record:
            raise GraphParameterError(f"fixture record is missing {key!r}")

    nodes = record["nodes"]
    if nodes != list(range(len(nodes))):
        raise GraphParameterError(
            "fixture nodes must be consecutive integers 0..n-1, "
            f"got {nodes[:8]}{'...' if len(nodes) > 8 else ''}"
        )

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for edge in record["edges"]:
        if len(edge) != 2:
            raise GraphParameterError(f"malformed edge {edge!r}: expected [u, v]")
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            raise GraphParameterError(f"fixture contains a self-loop at node {u}")
        if graph.has_edge(u, v):
            raise GraphParameterError(f"fixture contains a duplicate edge {u}-{v}")
        graph.add_edge(u, v)

    metadata = record["metadata"]
    _check_metadata_against_graph(metadata, graph)
    return graph, metadata


def _check_metadata_against_graph(metadata: Any, graph: nx.Graph) -> None:
    """Cross-check every structural claim in ``metadata`` against ``graph``.

    The edge and node counts were checked here from the start; the remaining
    fields (``n``, ``mean_degree``, ``connected``, ``n_components``) used to be
    trusted, which meant a hand-edited fixture could advertise "connected" or
    "mean degree 4" while holding a graph that is neither. Every structural
    field is now reconstructed from the edge list and compared.

    Fields absent from ``metadata`` are not invented: a record carrying partial
    metadata is still loadable, it simply gets fewer checks.
    """
    if not isinstance(metadata, dict):
        raise GraphParameterError(
            f"fixture metadata must be a dict, got {type(metadata).__name__}"
        )

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    for field_name, observed in (("n_edges", n_edges), ("n_nodes", n_nodes), ("n", n_nodes)):
        declared = metadata.get(field_name)
        if declared is not None and declared != observed:
            noun = "edges" if field_name == "n_edges" else "nodes"
            raise GraphParameterError(
                f"fixture metadata claims {declared} {noun} ({field_name}) but the "
                f"edge/node lists hold {observed}"
            )

    if n_nodes:
        declared_degree = metadata.get("mean_degree")
        observed_degree = mean_degree(graph)
        if declared_degree is not None and not math.isclose(
            float(declared_degree), observed_degree, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise GraphParameterError(
                f"fixture metadata claims mean degree {declared_degree} but 2E/N = "
                f"2*{n_edges}/{n_nodes} = {observed_degree}"
            )

        components = nx.number_connected_components(graph)
        declared_components = metadata.get("n_components")
        if declared_components is not None and declared_components != components:
            raise GraphParameterError(
                f"fixture metadata claims {declared_components} connected "
                f"component(s) but the edge list gives {components}"
            )
        declared_connected = metadata.get("connected")
        if declared_connected is not None and bool(declared_connected) != (components == 1):
            raise GraphParameterError(
                f"fixture metadata claims connected={declared_connected} but the "
                f"edge list gives {components} connected component(s)"
            )


def render_record(record: dict) -> str:
    """Serialise a fixture record as diff-friendly JSON.

    ``json.dumps(indent=2)`` would put every node ID and every edge endpoint on
    its own line, so an N=112 graph becomes a 1000-line file in which a
    one-edge change is unreadable. This writer keeps the metadata block
    indented but puts ``nodes`` on one line and one ``[u, v]`` edge per line.
    The output is ordinary JSON -- ``json.loads`` reads it back unchanged.
    """
    metadata = json.dumps(record["metadata"], indent=2, sort_keys=True)
    metadata = "\n".join(
        line if i == 0 else "  " + line for i, line in enumerate(metadata.splitlines())
    )
    nodes = json.dumps(record["nodes"])
    edges = ",\n    ".join(json.dumps(edge) for edge in record["edges"])
    return (
        "{\n"
        f'  "schema_version": {json.dumps(record["schema_version"])},\n'
        f'  "metadata": {metadata},\n'
        f'  "nodes": {nodes},\n'
        '  "edges": [\n'
        f"    {edges}\n"
        "  ]\n"
        "}\n"
    )


def save_fixture(path: Path, graph: nx.Graph, metadata: GraphMetadata) -> Path:
    """Write one fixture as diff-friendly JSON. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = graph_to_record(graph, metadata)
    path.write_text(render_record(record), encoding="utf-8")
    return path


def load_fixture(path: Path) -> tuple[nx.Graph, dict]:
    """Load one fixture file and return ``(graph, metadata_dict)``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no fixture at {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    return record_to_graph(record)


def load_graph(
    topology: str,
    n: int,
    seed: int = DEFAULT_FIXTURE_SEED,
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
) -> tuple[nx.Graph, dict]:
    """Convenience loader for a single committed fixture.

    >>> from neuromesh.graphs import load_graph
    >>> graph, meta = load_graph("ws", 14)          # doctest: +SKIP
    >>> sorted(graph.neighbors(0))                  # doctest: +SKIP
    [1, 9, 12, 13]
    """
    return load_fixture(Path(fixture_dir) / fixture_filename(topology, n, seed))


def generate_fixture_set(
    out_dir: Path = DEFAULT_FIXTURE_DIR,
    topologies: Iterable[str] = TOPOLOGIES,
    sizes: Iterable[int] = POPULATION_SIZES,
    seeds: Iterable[int] = (DEFAULT_FIXTURE_SEED,),
    params_by_topology: dict | None = None,
    write_manifest: bool = True,
) -> dict:
    """Generate and write a fixture set; return the manifest dict.

    The manifest lists every file with its key structural facts, so a consumer
    can check what is available without opening each graph.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_by_topology = params_by_topology or {}

    entries: list = []
    for topology in topologies:
        for n in sizes:
            for seed in seeds:
                spec = GraphSpec(
                    topology=topology,
                    n=int(n),
                    seed=int(seed),
                    params=params_by_topology.get(topology, {}),
                )
                graph, metadata = generate_graph(spec)
                filename = fixture_filename(topology, n, seed)
                save_fixture(out_dir / filename, graph, metadata)
                entries.append(
                    {
                        "file": filename,
                        "topology": topology,
                        "n": int(n),
                        "seed": int(seed),
                        "n_edges": metadata.n_edges,
                        "mean_degree": metadata.mean_degree,
                        "connected": metadata.connected,
                        "attempts": metadata.attempts,
                        "degree_matched_exactly": metadata.degree_matched_exactly,
                        "params": metadata.params,
                    }
                )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "networkx_version": nx.__version__,
        "count": len(entries),
        "fixtures": entries,
    }
    if write_manifest:
        (out_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def load_fixture_set(fixture_dir: Path = DEFAULT_FIXTURE_DIR) -> dict:
    """Load every fixture listed in a directory's manifest.

    Returns ``{(topology, n, seed): (graph, metadata_dict)}``.
    """
    fixture_dir = Path(fixture_dir)
    manifest_path = fixture_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {fixture_dir}; run "
            "'python -m neuromesh.graphs.cli fixtures' first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded: dict = {}
    for entry in manifest["fixtures"]:
        graph, metadata = load_fixture(fixture_dir / entry["file"])
        loaded[(entry["topology"], entry["n"], entry["seed"])] = (graph, metadata)
    return loaded
