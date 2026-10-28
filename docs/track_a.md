# Track A — Graphs & Network Data (Mihir Mathur)

Modules: `src/neuromesh/graphs/`, `data/`, `configs/robustness.yaml`, `tests/integration/`.

**Status: Week 1 complete — ER, WS and BA generators, validation, fixtures.**
`modular` and `yeo_sbm` are Week 2 and raise `NotImplementedError`.
`assign_roles`, `remove_nodes` and `structural_metrics` are still placeholders.

| Setting | Value |
|---|---|
| Population sizes (N) | 7, 14, 28, 56, 112 |
| Topologies implemented | `er`, `ws`, `ba` |
| Topologies planned (Week 2) | `modular`, `yeo_sbm` |
| Target mean degree | 4, matched **exactly** at every scale |
| Seeds | 0, 1, 2, 3, 4 (fixtures shipped at seed 0) |
| Boundary case | N = 7; topology comparisons start at N = 14 |

---

## 1. The public API

```python
from neuromesh.graphs import generate_graph

graph = generate_graph("ws", 28)                     # topology, N
graph = generate_graph("er", 112, seed=3)
graph = generate_graph("ba", 56, mean_degree=4.0, seed=1)
```

`generate_graph(topology, n, mean_degree=4.0, seed=0) -> nx.Graph` is the
signature in `contracts.py`. It returns **a bare `nx.Graph`, not a
`(graph, metadata)` tuple.** Nodes are the integers `0 .. n-1`.

`G.graph` carries the four keys the contract requires, plus two more:

```python
graph.graph["topology"]            # "ws"
graph.graph["n"]                   # 28
graph.graph["seed"]                # 0
graph.graph["mean_degree"]         # 4.0  -- the REALISED 2E/N
graph.graph["target_mean_degree"]  # 4.0  -- what was REQUESTED
graph.graph["generator_metadata"]  # full provenance dict (see below)
```

`mean_degree` and `target_mean_degree` are separate on purpose. For every
shipped configuration they are equal, because all three Week 1 generators hit
the target exactly; they are kept apart so that a future topology which can
only match *in expectation* cannot misreport itself as matched.

`generator_metadata` is the full `GraphMetadata` dict: `n_edges`, `connected`,
`n_components`, `attempts` (the ER rejection-sampling draw count),
`degree_matched_exactly`, `params` (the exact generator parameters used),
`generator_version`, `networkx_version` and `numpy_version`. Nothing about how
a graph was made is hidden from a consumer.

### Errors

| Call | Result |
|---|---|
| `generate_graph("modular", 28)` | `NotImplementedError` — Week 2, not a typo |
| `generate_graph("yeo_sbm", 28)` | `NotImplementedError` — Week 2 |
| `generate_graph("small_world", 28)` | `GraphParameterError` — unknown topology |
| `generate_graph("er", 2)` | `GraphParameterError` — `n must be >= 3` |
| `generate_graph("er", 14, mean_degree=500)` | `GraphParameterError` — infeasible for n=14 |

`GraphParameterError` subclasses `ValueError`, so a caller that only catches
`ValueError` keeps working.

### The two layers, and why there are two

```
neuromesh.graphs.generate_graph(topology, n, mean_degree, seed) -> nx.Graph
        |  thin adapter; no mathematics happens here
        v
neuromesh.graphs.generate_from_spec(GraphSpec(...)) -> (nx.Graph, GraphMetadata)
        |
        +-- generators.py   erdos_renyi / watts_strogatz / barabasi_albert
            validation.py   validate_graph / check_reproducible
            fixtures.py     save_fixture / load_fixture / load_graph
```

The lower layer returns `(graph, metadata)` so provenance is never lost, and
is reachable when you need it:

```python
from neuromesh.graphs import GraphSpec, generate_from_spec, erdos_renyi

graph, meta = generate_from_spec(GraphSpec("er", n=56, seed=3))
meta.attempts                  # how many draws connectivity needed
meta.degree_matched_exactly    # True

graph, meta = erdos_renyi(112, seed=0, model="gnp", ensure_connected=False)
```

The adapter is tested to produce the same edge set as the layer beneath it
(`tests/graphs/test_public_api.py::test_adapter_and_generator_layer_agree`),
which is what makes the committed fixtures valid for callers of either layer.
`generate_from_spec` is the same function as
`neuromesh.graphs.generators.generate_graph`; it is re-exported under a
different name at package level only because the package-level
`generate_graph` is the team's shared interface.

---

## 2. Generation methods

Mean degree is **always** `2E / N`. For k = 4 the target edge count is
`E = 2N`, integral at all five population sizes, so no rounding error enters
anywhere.

### 2.1 Erdős–Rényi

| Parameter | Value | Why |
|---|---|---|
| `model` | `"gnm"` (default) | `G(N, M)` with `M = 2N`; mean degree **exactly** 4 by construction |
| | `"gnp"` (available) | `G(N, p)`, `p = k/(N-1)`; matches k only in expectation |
| `ensure_connected` | `True` | Rejection-sample from the connected ensemble |
| `max_attempts` | 1000 | Ample; the worst observed case needed 15 draws |

`G(N, M)` is the default because "probability-based ER" and "mean degree within
±10%" are not simultaneously satisfiable at small N under `G(N, p)`: at N = 7
the edge count is Binomial(21, 2/3) with SD ≈ 2.16 edges, while the ±10%
acceptance window is only `E ∈ {13, 14, 15}`. Measured at seed 0, `G(N, p)`
misses the window at N = 28 (+16.07%) and N = 112 (+12.05%). `G(N, M)` is the
original Erdős–Rényi formulation (`G(N, p)` is properly Gilbert's model), so
this is a principled choice rather than a workaround. Both models stay exposed
and both appear in `cli validate` output.

### 2.2 Watts–Strogatz

| Parameter | Value | Why |
|---|---|---|
| `k` | 4 (= target mean degree) | Ring lattice; must be even and < N |
| `rewiring_prob` | **0.1** | Top of the small-world window (see below) |
| `ensure_connected` | `True` | Guard only; never triggered in practice |

Watts & Strogatz (1998) locate the small-world regime — clustering still near
the lattice value while path length has already collapsed towards the random
graph — at roughly `0.01 ≤ p ≤ 0.1`. p = 0.1 sits at the top of that window:
the shortest paths available without surrendering lattice clustering, which
suits information propagation over a fixed 10-round budget. Rewiring never
changes the edge count, so **mean degree is exactly 4 at every N and for every
p.** The value used is recorded in `params` and is never changed silently.

Measured at seed 0 (re-verified in this repository):

| N | WS clustering | ER clustering | WS avg path | ER avg path |
|---|---|---|---|---|
| 28 | 0.400 | 0.126 | 2.99 | 2.42 |
| 56 | 0.335 | 0.075 | 3.74 | 3.03 |
| 112 | 0.374 | 0.027 | 5.19 | 3.51 |

At N = 112 WS retains ~14× ER's clustering at the same mean degree while its
average path is only ~1.5× longer — the small-world signature. WS was connected
in 200/200 unconditioned draws at every N.

### 2.3 Barabási–Albert

| Parameter | Value | Why |
|---|---|---|
| `m` | 2 (= k/2) | Makes BA's asymptotic mean degree `2m` equal the target |
| `variant` | `"degree_matched"` (default) | Seed core `K_(2m+1)` = `K_5` |
| | `"standard"` (available) | Textbook BA; misses tolerance at N = 7, 14 |

**The limitation, stated plainly.** `networkx.barabasi_albert_graph(N, m)`
produces exactly `E = m(N − m)`, so mean degree is `4 − 8/N` for m = 2:

| N | 7 | 14 | 28 | 56 | 112 |
|---|---|----|----|----|-----|
| mean degree | 2.857 | 3.429 | 3.714 | 3.857 | 3.929 |
| deviation | **−28.57%** | **−14.29%** | −7.14% | −3.57% | −1.79% |

Standard BA **cannot** reach mean degree 4 at N = 7 or N = 14, and m = 3
overshoots. This is a property of the model; it is pinned by
`test_ba_standard_small_n_deviation_is_the_predicted_one`, which asserts the
analytically predicted deviation rather than relaxing a bound.

**The fix is a seed core, not a different algorithm.** Preferential attachment
is untouched; only the initial graph changes, from NetworkX's `m` isolated
nodes to the complete graph `K_(2m+1)`. Then

```
E = C(2m+1, 2) + m(N − 2m − 1)
  = m(2m+1) + mN − 2m² − m
  = mN                                  exactly, for every N
```

so mean degree is exactly `2m = k = 4` at every size, and every node after the
core still arrives by preferential attachment. The choice and its consequence
are recorded in `params` and `degree_matched_exactly` — it is not a silent
substitution, and `variant="standard"` remains available.

**Trade-off, measured (seed 0, re-verified here).** The seed clique inflates
clustering at small N:

| N | max degree (matched) | max degree (standard) | clustering (matched) | clustering (standard) |
|---|---|---|---|---|
| 7 | 6 | 4 | 0.867 | 0.190 |
| 14 | 10 | 7 | 0.801 | 0.263 |
| 28 | 14 | 10 | 0.429 | 0.267 |
| 56 | 22 | 13 | 0.170 | 0.112 |
| 112 | 29 | 24 | 0.119 | 0.123 |

At N ≥ 28 the degree-matched variant keeps clear hub structure (max degree 29
against a mean of 4 at N = 112) and its clustering converges on standard BA's.
**At N = 7 it does not** — see §5.

BA is connected by construction, so no rejection sampling is used; connectivity
is still measured and reported.

---

## 3. Exact mean-degree matching

All three default generators hit mean degree 4 **exactly**, not merely within
±10%, at all five sizes and all five seeds:

| Topology | Mechanism | E |
|---|---|---|
| ER | `G(N, M)` with `M = round(kN/2)` | 2N |
| WS | rewiring preserves the ring lattice's edge count | 2N |
| BA | `K_(2m+1)` seed core makes `E = mN` | 2N |

The tests therefore assert equality, not tolerance
(`test_whole_sweep_is_matched_on_mean_degree`,
`test_mean_degree_matches_target_exactly`). If a future change weakens that to
"within tolerance", it has to be a deliberate edit to those assertions.

### Sweep eligibility vs "no defect"

`validate_graph` returns a `ValidationResult` with two distinct flags:

- **`ok`** — no check revealed a *defect*. A documented model constraint (standard
  BA at N = 7 missing mean degree 4) is recorded as `EXCEPTION`, not `FAIL`, and
  still counts as `ok`.
- **`sweep_eligible`** — no failures **and no exceptions**. Only these graphs
  belong in the main experimental sweep, because mean-degree matching is the
  study's controlled variable.

A graph that is `ok` only by way of an `EXCEPTION` is not matched on mean degree
and must not enter the sweep on the strength of `ok` alone. Both flags are in
`result.to_dict()`; `cli validate` prints a sweep-eligible count per table.

---

## 4. Connectivity-conditioned sampling — read before interpreting ER results

`G(N, ·)` is connected with high probability only above mean degree `ln N`, and
`ln(112) ≈ 4.72 > 4`. Expected isolated vertices at N = 112 ≈ `112·e⁻⁴ ≈ 2.0`.
Measured unconditioned P(connected) at N = 112 is **≈ 0.147** (2000 draws,
seeds 0–1999; SE ≈ 0.008 — the Week 1 notes recorded 0.1445 under a different
draw loop, the same quantity within sampling error).

**Mean degree 4 and connectedness are therefore in genuine tension for ER at
large N.** This is a property of the model, not a defect in the generator.

With `ensure_connected=True` (the default) the generator draws from the
**connected ER ensemble** by rejection sampling with deterministically derived
sub-seeds. N and E stay exactly fixed, so mean degree stays exactly 4 — but the
ensemble is **conditioned**, which slightly depletes low-degree nodes relative
to unconditioned ER. **This must be stated in the methods section of the
report.** The draw count is recorded so the conditioning is never invisible:

```python
graph = generate_graph("er", 112, seed=0)
graph.graph["generator_metadata"]["attempts"]    # 15
```

Draws required, measured here (seeds 0–4):

| N | 7 | 14 | 28 | 56 | 112 |
|---|---|----|----|----|-----|
| draws | 1,1,1,1,1 | 1,1,1,1,1 | 1,1,1,3,1 | 5,1,1,3,1 | 15,10,1,3,13 |

Pass `ensure_connected=False` via the generator layer for the raw,
unconditioned draw.

---

## 5. N = 7 boundary case — how to interpret it

N = 7 is a **declared boundary case** throughout the study; topology
comparisons start at N = 14. Three specific consequences:

1. **Degree-matched BA at N = 7 is not scale-free.** The degree sequence is
   `[6, 6, 4, 4, 4, 2, 2]` and the graph verifiably contains `K_5`: only 2 of
   7 nodes arrive by preferential attachment, so it is a clique-plus-two.
   **Do not present N = 7 BA as a hub topology.**
2. **Standard BA at N = 7 misses mean degree 4 by −28.6%**, the largest
   deviation anywhere in the design.
3. **Seed variation is not meaningfully testable at N = 7.** With so few
   distinct realisations, `test_different_seeds_give_different_graphs` is
   skipped at N = 7 — **3 skips**, one per topology, and the only skips in the
   suite. They are deliberate and documented; the same assertion runs at
   N = 14, 28, 56, 112.

N = 7 graphs *are* generated, validated and shipped as fixtures — all three
hit mean degree 4 exactly there. The boundary status is about interpretation,
not about whether the graphs exist.

Week 2 note: balanced role assignment needs N/7 agents per Yeo network, which
is exact at all five sizes. At N = 7 each network has exactly one agent.

---

## 6. Reproducibility and seed handling

- Identical `(topology, n, mean_degree, seed)` always yields an identical node
  and edge set. Asserted for every topology at every size
  (`test_same_seed_gives_the_same_graph`, `check_reproducible`).
- Sub-seeds for rejection-sampling attempts are derived with
  `numpy.random.SeedSequence(entropy=seed, spawn_key=(attempt,))`, whose stream
  is documented as stable across NumPy versions and platforms. `attempt=0` does
  not reuse `seed` verbatim, but the mapping is a pure function of `seed`.
- **Caveat:** this makes fixtures reproducible *given NumPy*, not from a
  pure-Python algorithm. The NetworkX and NumPy versions used are recorded in
  every fixture's metadata. Verified on Python 3.11.9, networkx 3.6.1,
  numpy 2.4.6.
- `check_reproducible` judges on the sorted edge set **and** the node set, so a
  generator producing the same edges under a different node labelling is still
  caught.

---

## 7. Fixtures

**Location: `data/fixtures/graphs/`** — 15 files (3 topologies × 5 sizes, seed 0)
plus `manifest.json`. Kept there rather than under `tests/fixtures/graphs/`
because they are study data consumed by the runner and all three other tracks,
not test-only stubs, and because `DEFAULT_FIXTURE_DIR` is a documented
Week 1 constant. `tests/fixtures/graphs/` is left empty for now.

### Loading — for Armman (B), Yutong (C) and Andrew (D)

```python
from neuromesh.graphs import load_graph

graph, meta = load_graph("ws", 14)          # topology, N (seed 0 by default)
graph, meta = load_graph("ba", 112, seed=0)

graph.number_of_nodes()      # 14
sorted(graph.neighbors(0))   # neighbour list for agent 0
meta["mean_degree"]          # 4.0
meta["attempts"]             # rejection-sampling draws
```

The whole set at once:

```python
from neuromesh.graphs import load_fixture_set

fixtures = load_fixture_set()               # {(topology, n, seed): (graph, meta)}
graph, meta = fixtures[("er", 28, 0)]
```

**Without NetworkX** — the files are plain JSON:

```python
import json
record = json.load(open("data/fixtures/graphs/ws_n014_seed0.json"))
record["nodes"]   # [0, 1, ..., 13]
record["edges"]   # [[0, 1], [0, 2], ...]  each [u, v] with u < v, sorted
```

### File format (`schema_version = 1`)

```json
{
  "schema_version": 1,
  "metadata": { "topology": "er", "n": 14, "seed": 0, "mean_degree": 4.0, "...": "..." },
  "nodes": [0, 1, 2, "..."],
  "edges": [[0, 1], [0, 3], "..."]
}
```

Guarantees other tracks can rely on:

- `nodes` is exactly `[0, 1, ..., N-1]` — consecutive integer agent IDs.
- Every edge is `[u, v]` with `u < v`; sorted; no duplicates; no self-loops.
  Edges are undirected, so each pair appears **once** — iterate both directions
  yourself if you need directed message routing.
- `metadata` carries full provenance: seed, generator version, NetworkX and
  NumPy versions, exact generator parameters, and the draw count.
- **No timestamps in graph files**, so regenerating from the same seed yields
  byte-identical output (asserted by
  `test_regenerating_a_fixture_is_byte_identical`). Only `manifest.json`
  records when a set was built, so it is the one file that changes on
  regeneration.
- `load_fixture` **validates** on read: schema version, contiguous node IDs, no
  self-loops, no duplicate edges, and every structural metadata field
  (`n`, `n_nodes`, `n_edges`, `mean_degree`, `connected`, `n_components`)
  cross-checked against the graph reconstructed from the edge list. A corrupted
  or hand-edited fixture fails loudly instead of feeding a mis-described graph
  into an experiment. Fields absent from `metadata` are not invented; they just
  get fewer checks.

### Regenerating

```bash
python -m neuromesh.graphs.cli fixtures                     # 15 fixtures, seed 0
python -m neuromesh.graphs.cli fixtures --seeds 0 1 2 3 4   # all 75, for later weeks
python -m neuromesh.graphs.cli fixtures --out some/other/dir
python -m neuromesh.graphs.cli validate                     # validation tables
python -m neuromesh.graphs.cli verify-fixtures              # reload + revalidate
```

`validate` prints two tables: the 15 canonical rows that produce the shipped
fixtures, and a diagnostics table for the non-default variants (`G(n,p)` ER and
standard BA) so their known deviations stay visible rather than hidden behind
the defaults.

---

## 8. Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q -rs                        # whole repo; -rs lists skip reasons
python -m pytest tests/graphs -q               # Track A only
ruff check .
```

| Module | Covers |
|---|---|
| `tests/graphs/test_public_api.py` | the shared `generate_graph` interface: return type, `G.graph` keys, all 3 topologies × 5 sizes × 5 seeds, error handling, Week 2 placeholders |
| `tests/graphs/test_generators.py` | generator layer: node/edge validity, exact mean degree, connectivity, reproducibility, invalid inputs, the BA/ER model limitations |
| `tests/graphs/test_validation.py` | the validator's own correctness, against deliberately broken graphs |
| `tests/graphs/test_fixtures.py` | round-trip, corruption handling, and the committed fixture set |

**Expected result: all pass, with exactly 3 skips** — the N = 7 seed-variation
skips described in §5. The validator is itself tested against wrong node
counts, relabelled nodes, self-loops, directed input, multigraphs and
disconnected graphs, so a vacuously-passing validator would be caught.

---

## 9. Yeo connectivity source — unresolved dependency for Week 2

Full findings: **[`data/yeo/README_YEO_SOURCES.md`](../data/yeo/README_YEO_SOURCES.md)**.

- **Verified:** Yeo et al. 2011 (the 7-network *parcellation*, DOI
  10.1152/jn.00338.2011); Schaefer et al. 2018 `*_7Networks_order`
  parcellations on the CBIG GitHub; ENIGMA Toolbox group-average HCP
  **structural** and **functional** matrices in Schaefer 100/200/300/400 via
  `load_sc()` / `load_fc()` (preprint DOI 10.1101/2020.12.21.423838).
- **Central negative finding: no published canonical 7×7 Yeo-network
  connectivity matrix was located.** The 7×7 must be **derived** by aggregating
  a parcel-level matrix over the Yeo labels — a computation Track A owns, with
  an explicitly stated aggregation rule.
- **No numerical connectivity values have been invented anywhere in this
  repository, and the Yeo-SBM generator is not implemented.**
  `generate_graph("yeo_sbm", ...)` raises `NotImplementedError`.
- Six items are unresolved (U1–U6), including the ENIGMA subject count and the
  CBIG/ENIGMA licence terms, both of which must be read before any derived
  matrix is committed.
- If Week 2 cannot obtain the data, **Yeo-SBM will be reported as blocked
  rather than populated with plausible-looking values.**

---

## 10. Flagged risk — WS diameter vs the round budget

At N = 112, seed 0, measured diameters are **ER 7, BA 5, WS 10**, with average
path lengths 3.51 / 2.98 / 5.19. The main experiments run **10 synchronous
rounds**, so WS at N = 112 has a diameter *equal to* the entire round budget: a
fact originating at one end of the ring needs all 10 rounds to reach the
farthest agent, leaving zero slack for the propagation and consensus tasks.

This is evidence for choosing **12 rounds**, or for reporting WS propagation at
N = 112 as diameter-limited. Raised for the team — no parameter has been
changed, and `DEFAULT_ROUNDS` in `contracts.py` is Andrew's to set.

---

## 11. Known limitations and follow-ups

1. Standard BA cannot hit mean degree 4 at N = 7 (−28.6%) or N = 14 (−14.3%).
   Mitigated by `degree_matched`; the standard variant stays available and its
   deviation is reported, never concealed.
2. Degree-matched BA at N = 7 is a clique-plus-two, not a scale-free graph (§5).
3. ER connectivity is conditioned, not natural, at N ≥ 56 (§4). Must be stated
   in the methods section.
4. `G(N, p)` is not the shipped ER model (§2.1). `cli validate`'s diagnostics
   table shows its behaviour for any reviewer who expects it.
5. Seed derivation depends on `numpy.random.SeedSequence` (§6).
6. The Yeo 7×7 matrix does not exist yet (§9).
7. **Follow-up, not done:** fixtures are shipped at seed 0 only. `full.yaml` and
   `robustness.yaml` use seeds 0–4. Generating all 75 is one CLI flag
   (`--seeds 0 1 2 3 4`); whether the runner wants them committed or generated
   on the fly is Andrew's call.
8. **Follow-up, not done:** with a non-default odd `mean_degree`, BA now raises
   rather than silently matching the nearest even value, and WS already
   required an even `k`. ER would still accept it. Harmonising that across the
   three generators would change existing behaviour, so it is logged rather
   than done — the whole study runs at k = 4.
9. The end-to-end integration test (`tests/integration/`) is a Week 3 Track A
   task and is not written yet.
