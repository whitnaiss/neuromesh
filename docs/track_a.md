# Track A — Graphs & Network Data (Mihir Mathur)

Modules: `src/neuromesh/graphs/`, `data/`, `configs/robustness.yaml`, `tests/integration/`.

**Status: Week 4 complete — structural metrics for all 25 conditions, node-level
positional covariates, and graph-only robustness reference curves, on top of
Week 3's node removal, mock integration test and diameter analysis, and
Week 2's five topologies, balanced role assignment, derived Yeo 7×7 matrix,
validation and fixtures.**
**Every Track A entry point in `contracts.py` is now implemented** —
`generate_graph`, `assign_roles`, `remove_nodes` and `structural_metrics`.

| Setting | Value |
|---|---|
| Population sizes (N) | 7, 14, 28, 56, 112 |
| Topologies implemented | `er`, `ws`, `ba`, `modular`, `yeo_sbm` |
| Target mean degree | 4, matched **exactly** at every scale |
| Seeds | 0, 1, 2, 3, 4 (fixtures shipped at seed 0) |
| Roles | seven Yeo networks, exactly N/7 agents each |
| Boundary case | N = 7; topology comparisons start at N = 14 |

## Week 2 additions at a glance

- **Modular** (`generators.modular`) — planted partition over seven equal
  blocks, within/between preference ratio 10 by default. §12.
- **Yeo-SBM** (`generators.yeo_sbm`) — the same machinery driven by an
  empirically derived 7×7 inter-network structural matrix. §13.
- **Role assignment** (`roles.assign_roles`) — exactly N/7 agents per Yeo
  network; randomised for ER/WS/BA, read off the blocks for the two block
  models. §14.
- **Derived matrix** (`data/yeo/yeo7x7_sc.json`) — aggregated from the ENIGMA
  Toolbox HCP group-average Schaefer-200 structural connectome. §13.1.

## Week 3 additions at a glance

- **Node removal** (`robustness.py`) — random failure and *adaptive*
  highest-degree attack, with a documented tie-break, full attack provenance
  and an attacked-graph validator. `remove_nodes` in `contracts.py` is now
  implemented. §16.
- **Mock integration test** (`tests/integration/test_mock_pipeline.py`) —
  the MVP scenario end to end, offline, in 0.40 s. §17.
- **Diameter / round-budget analysis** (`analysis.py`, `cli diameters`) —
  all 25 N = 112 conditions measured. **Headline: 10 rounds covers 23/25 and
  12 rounds covers 24/25; WS seed 4 has diameter 13.** §18.

## Week 4 additions at a glance

- **Structural metrics** (`metrics.py`) — 43 fields per graph over all 125
  instances; `structural_metrics` in `contracts.py` is now implemented. §20.
- **Positional covariates** (`metrics.positional_covariates`) — 22 fields per
  node, 5,425 rows, including participation coefficient and within-module
  z-score against the seven networks. §21.
- **Graph-only robustness curves** (`curves.py`) — 250 trajectories, complete
  over every integer removal count up to 30%, reusing the Week 3 engine. §22.
- **Analysis datasets and figures** (`datasets.py`, `figures.py`,
  `cli week4-analysis`) — 13 byte-reproducible outputs in ~12 s; figures are
  dependency-free SVG. §23.
- **Headline: under random failure topology barely matters (largest-component
  fraction 0.659–0.682 at 30% removal); under targeted attack the families
  separate by more than 10× (WS 0.329 vs BA 0.027).** §22.7.

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

> **Week 3 update — the risk is worse than this section states.** Seed 0 was
> the only seed measured in Week 1. Across all five seeds WS reaches diameter
> **13** (seed 4) and 11 (seed 1), so **12 rounds is not sufficient either**.
> The full 25-condition measurement and the revised recommendation are in
> §18. This section is left as written for the record.

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
9. ~~The end-to-end integration test (`tests/integration/`) is a Week 3 Track A
   task and is not written yet.~~ **Done in Week 3** — see §17.

---

## 12. Modular topology (Week 2)

A planted partition over seven equal blocks of N/7 agents. Every within-block
node pair carries the same preference and every between-block pair the same,
lower one, so the only structure present is community membership itself —
which is what the proposal's "modular network with uniform links between
modules" asks for.

| Parameter | Value | Why |
|---|---|---|
| blocks | 7, equal | One per Yeo functional network |
| `within_between_ratio` | 10.0 | Strong but not extreme preference; recorded in `params`, never changed silently |
| edge budget | `E = 2N`, exact | Mean degree is the controlled variable |
| `ensure_connected` | `True` | Rejection sampling, draws recorded |

### Why a fixed edge budget rather than independent Bernoulli trials

A textbook SBM draws each node pair independently, so mean degree is matched
only *in expectation* — the same defect that ruled out `G(N, p)` for ER. The
total is therefore fixed at `E = round(kN/2)` and divided among the 28 block
pairs in proportion to `weight x capacity`, then drawn uniformly without
replacement inside each pair. **Mean degree is exactly 4 at every N and every
seed**, while the relative block preferences are preserved.

This conditions the ensemble, exactly as connectivity rejection does for ER,
and the conditioning is recorded rather than hidden (`n_within_edges`,
`n_between_edges`, `capacity_clamped`, `attempts`).

### Realised structure (seed 0, measured)

| N | within edges | between | within % | modularity Q | structure |
|---|---|---|---|---|---|
| 7 | 0 | 14 | 0% | −0.148 | **none possible** |
| 14 | 7 | 21 | 25% | +0.107 | capacity-capped |
| 28 | 35 | 21 | 62.5% | +0.482 | as requested |
| 56 | 70 | 42 | 62.5% | +0.482 | as requested |
| 112 | 140 | 84 | 62.5% | +0.482 | as requested |

For seven equal blocks, `Q ≈ within_fraction − 1/7`, which the measured values
match to three decimals — an analytic check on the sampler.

### Finite-size limits, stated plainly

A block pair cannot hold more edges than it has node pairs, so at small N the
preference is capped by **capacity, not by choice**:

- **N = 7** — one agent per block, so within-block capacity is **zero**. No
  within-block edge can exist at any ratio. The graph is entirely
  between-block and its negative modularity says so. **N = 7 modular is not a
  modular graph** and must not be presented as one; it is a labelled random
  graph. `params["within_structure_possible"]` is `False` there and only there.
- **N = 14** — two agents per block, so within-block capacity is 1 per block,
  7 in total against a budget of 28. The within share is capped at **25%**
  however strong the requested ratio. `params["capacity_clamped"]` is `True`.

Both are properties of the design at those sizes, consistent with N = 7 being
a declared boundary case throughout this study.

### Tie-breaking, and a defect that was fixed

Proportional demand rarely divides evenly, so the leftover edges are handed out
by largest fractional remainder. When remainders tie, the tie must be broken.
Breaking it by block-pair **index** was the first implementation and it was
wrong: at N = 7 *every* cell ties (each of the 21 between-block pairs is owed
14/21 of an edge and can hold exactly one), so index order always selected the
same 14 low-index pairs. The N = 7 modular graph was then identical for all
five seeds and systematically biased toward low-numbered agents.

Ties are now broken by a seeded random permutation, so N = 7 modular yields
five distinct graphs across the five study seeds. `allocate_block_edges`
remains a pure deterministic function when called without an `rng`.

---

## 13. Yeo-SBM topology (Week 2)

Identical machinery to §12, but the 7x7 block preferences are **empirically
derived from human structural connectivity** instead of one flat contrast.
Block *b* **is** functional network *b*: the matrix rows and `contracts.YEO7`
share an order, which is what lets `assign_roles` read roles straight off the
blocks.

### 13.1 The derived matrix — provenance

`data/yeo/yeo7x7_sc.json`, produced by
`python -m neuromesh.graphs.cli yeo-derive`.

| Item | Value |
|---|---|
| Source | ENIGMA Toolbox group-average HCP connectome |
| Files | `strucMatrix_ctx_schaefer_200.csv`, `strucLabels_ctx_schaefer_200.csv` |
| Parcellation | Schaefer2018, 200 parcels, `7Networks` order |
| Modality | **Structural** (diffusion tractography), never functional |
| Licence | **BSD-3-Clause** — redistribution permitted with attribution |
| Aggregation | Arithmetic **mean** of parcel-pair entries per block; parcel self-pairs excluded on the diagonal |
| Validation | 200x200, exactly symmetric, all labels parsed, all 7 networks present, all 49 block means non-negative |

**Mean, not sum**, because the seven networks hold unequal numbers of parcels
(12 for limbic to 46 for default at Schaefer-200); sums would re-encode network
size rather than connectivity.

**Structural, not functional**, because the Yeo networks were themselves
defined by clustering resting-state *functional* connectivity, so an
FC-derived matrix would largely re-encode the parcellation instead of
providing independent structural evidence.

Two provider transforms carry into the write-up and are recorded in the file:
the source matrix is **distance-thresholded** (about 88% of parcel pairs are
exactly zero) and **log-transformed**. Block means are therefore means of
transformed values that blend connection *strength* with connection *density*;
the per-block density is stored separately in `provenance["block_density"]` so
the two can be separated. A small number of source entries are negative (16 of
4,822 non-zero, a consequence of the log transform); **every one of the 49
aggregated block means is non-negative**, so no shift or re-transform was
applied, and the derivation refuses to emit a negative block mean.

### 13.2 Derived 7x7 block means

|  | Vis | SomMot | DorsAttn | SalVentAttn | Limbic | Cont | Default |
|---|---|---|---|---|---|---|---|
| **Vis** | 3.108 | 0.056 | 0.416 | 0.216 | 1.168 | 0.202 | 0.501 |
| **SomMot** | 0.056 | 2.051 | 0.864 | 1.192 | 0.052 | 0.361 | 0.200 |
| **DorsAttn** | 0.416 | 0.864 | 1.491 | 0.712 | 0.175 | 0.829 | 0.517 |
| **SalVentAttn** | 0.216 | 1.192 | 0.712 | 1.525 | 0.476 | 0.912 | 0.974 |
| **Limbic** | 1.168 | 0.052 | 0.175 | 0.476 | 2.650 | 0.852 | 1.049 |
| **Cont** | 0.202 | 0.361 | 0.829 | 0.912 | 0.852 | 1.378 | 1.234 |
| **Default** | 0.501 | 0.200 | 0.517 | 0.974 | 1.049 | 1.234 | 1.473 |

Within-network connectivity exceeds between-network by a factor of **3.17** on
average. The off-diagonal is strongly heterogeneous — visual-somatomotor is
near zero (0.056, with only 0.9% of parcel pairs connected) while
default-frontoparietal is high (1.234) — which is what makes Yeo-SBM a
different topology from the uniform modular model rather than a reskin of it.

### 13.3 Realised structure (seed 0, measured)

| N | within edges | within % | modularity Q |
|---|---|---|---|
| 7 | 0 | 0% | −0.153 |
| 14 | 7 | 25% | +0.101 |
| 28 | 16 | 28.6% | +0.141 |
| 56 | 36 | 32.1% | +0.177 |
| 112 | 73 | 32.6% | +0.182 |

Yeo-SBM is **measurably less modular than the uniform modular topology**
(Q ≈ 0.18 against 0.48) because the empirical within/between ratio is 3.17
rather than the modular model's 10. That contrast is the scientific point of
having both.

### 13.4 The empirical matrix sets contrast, not density

Absolute connection density comes from this study's matched mean degree, not
from the brain. The generated graph reproduces the connectome's *relative*
between-network preferences at mean degree 4 — it is a brain-**informed**
topology, not a brain simulation. This must be stated in the methods section.

### 13.5 N = 7 is fully determined by the data

At N = 7 every block pair holds exactly one possible edge, and the 21
off-diagonal empirical weights are all distinct. The allocator therefore has no
ties to break and selects **exactly the 14 strongest inter-network connections**
— the same graph for every seed. This is the faithful result at that size, not
a defect, and it is why `test_different_seeds_give_different_graphs` carries a
documented skip for `yeo_sbm` at N = 7 only.

### 13.6 Without the data, the topology fails loudly

`generate_graph("yeo_sbm", ...)` raises `YeoMatrixUnavailableError` when
`data/yeo/yeo7x7_sc.json` is absent. A caller may pass `matrix=` to exercise
the sampler with a synthetic matrix, and that path sets
`params["matrix_source"] = "caller_supplied"` so no downstream log can mistake
a mechanics test for a data-informed run. **No invented connectivity values
exist anywhere in this repository.**

---

## 14. Balanced role assignment (Week 2)

`assign_roles(graph, seed)` gives every node exactly one of the seven
`contracts.YEO7` roles, with exactly **N/7 agents per network** at all five
population sizes. It modifies the graph in place and returns it; **the node set
and every edge are left untouched**.

Two regimes, recorded in `graph.graph["role_assignment"]["regime"]`:

| Topologies | Regime | Seed-dependent |
|---|---|---|
| `er`, `ws`, `ba` | `randomised_balanced` — labels shuffled with a seeded generator | **Yes** |
| `modular`, `yeo_sbm` | `community` — role read off the node's block | **No** |

The second regime is seed-**in**dependent by design. Once a node's community is
fixed there is no freedom left, and reshuffling roles across blocks would break
the correspondence between the generator's 7x7 matrix and the roles the agents
play — which is the entire point of Yeo-SBM.

Role assignment draws from a **separate random stream** from graph generation
(`SeedSequence(entropy=[seed, ROLE_SEED_SALT])`), so role placement is not
correlated with the draw that produced the graph.

### 14.1 Interpretation warning: role placement is a confound across topology families

Role placement is generated by two different regimes (table above), so
**a performance difference between a block topology and a non-block topology
cannot be attributed to the arrangement of links alone.** Two factors change
together at that boundary:

1. the topology itself, and
2. how roles sit on it — randomised and redrawn per seed for `er`/`ws`/`ba`,
   so results average over placement; perfectly aligned with the communities
   and identical at every seed for `modular`/`yeo_sbm`, so results carry zero
   placement variance and maximal role–community alignment.

| Comparison | Clean read on topology? |
|---|---|
| ER vs WS vs BA | **Yes** — same regime, placement averaged over seeds |
| modular vs Yeo-SBM | **Yes** — same regime, placement fixed by blocks |
| ER/WS/BA vs modular/Yeo-SBM | **No** — topology and placement regime vary together |

This is a consequence of the design, not a defect: for Yeo-SBM the entire
point is that block *b* is functional network *b*, so that alignment cannot be
randomised away without destroying what the topology means. Note also that the
seven roles are *defined* by the Yeo networks, so for the block topologies the
community structure and the role structure are the same object by
construction.

Two practical implications:

- Any cross-family claim in the report must state the confound. Prefer
  within-family statements where the design allows.
- Disentangling the two factors would need an extra condition — for example a
  block topology run with deliberately shuffled role labels, which would break
  role–community alignment while keeping the link arrangement. That condition
  is **not** part of the current experimental design and is not implemented.
  Raised for the team; it is a design question, not a Track A implementation
  choice.

Variance consequence worth recording: because role placement is fixed for the
block topologies, their seed-to-seed variance reflects link placement only,
while ER/WS/BA seed variance reflects link placement *and* role placement.
Error bars from the two families are therefore not measuring the same thing.

```python
from neuromesh.graphs import generate_graph, assign_roles

graph = assign_roles(generate_graph("yeo_sbm", 112, seed=0), seed=0)
graph.nodes[0]["role"]        # 'visual'
graph.nodes[0]["community"]   # 0
```

---

## 15. Week 2 limitations and follow-ups

1. **N = 7 modular carries no community structure** (§12) and **N = 7 Yeo-SBM
   is deterministic** (§13.5). Both are consequences of one agent per network.
2. **N = 14 modular/Yeo-SBM are capacity-capped at 25% within-block edges**
   (§12), so the requested preference is not realised there.
3. **The Yeo block means mix strength and density** (§13.1), because the source
   matrix is thresholded. The per-block density is recorded for anyone who
   needs to separate them.
4. **The ENIGMA HCP subject count is still not published** on the toolbox's
   documentation page, so no cohort size is cited anywhere. Unresolved item U1
   in `data/yeo/README_YEO_SOURCES.md`.
5. **An FC-derived sensitivity matrix has not been produced.** The design names
   SC as primary; FC remains a possible robustness check.
6. **Follow-up, not done:** `DEFAULT_FIXTURE_DIR` is a path relative to the
   working directory, so `load_graph()` only resolves from the repository root.
   The Yeo loader resolves relative to the package instead and works from
   anywhere; harmonising the fixture loader the same way is a one-line change
   left for team review, since it alters existing Week 1 behaviour.
7. **Follow-up, not done:** fixtures remain seed 0 only; `--seeds 0 1 2 3 4`
   produces all 125.

---

## 16. Node removal (Week 3)

`src/neuromesh/graphs/robustness.py`. Two strategies, both deterministic, both
leaving the input graph untouched.

**Everything in this section describes damaging a graph on purpose.** The
generator contracts — connected, mean degree 4, exactly N/7 agents per role —
are properties of the graphs §2 produces and **do not apply to an attacked
graph**. A targeted attack that fails to fragment the network has measured
nothing.

### 16.1 The two strategies

| Strategy | Models | Algorithm |
|---|---|---|
| `random` | indiscriminate failure (an agent crashes, a call times out) | k distinct nodes drawn uniformly without replacement from an isolated seeded RNG |
| `targeted` | an adversary who can see the network | **adaptive** highest-degree-first: remove the current maximum-degree node, recompute degrees on the survivors, repeat |

The word *adaptive* is load-bearing. Sorting the original degree sequence once
and deleting its top k is a different, weaker attack, because it ignores that
removing a hub demotes the hubs that leaned on it. `static_removal_order()`
implements that weaker version — **for comparison and testing only, never for
the experiment** — so the difference is demonstrable rather than asserted.

A worked example, the hand-built graph in `test_robustness.py`:

```
    3  4  5  6        7   8          9  10  11
     \ |  | /          \ /            \  |  /
      \|  |/            1              2
       0 ---------------+
```

Original degrees: node 0 → 5, node 1 → 3, node 2 → 3, everything else → 1.

| | Order | Why |
|---|---|---|
| static | `0, 1, 2` | sorts once; node 1 wins the degree-3 tie on ID |
| **adaptive** | `0, 2, 1` | removing 0 costs node 1 an edge, so it drops to degree 2 while node 2 stays at 3 |

At k = 2 the two strategies remove **different sets** (`{0,1}` vs `{0,2}`); at
k = 3 the same set in a different order. Both are pinned by tests.

### 16.2 Tie-breaking — read this, ties are the normal case

At mean degree 4 most nodes share their degree with several others, so the
tie-break visibly decides the attack. It is therefore explicit, never implicit:

| `tie_break` | Behaviour | Seed |
|---|---|---|
| `smallest_id` | **default**; lowest node ID among the current maxima | not used |
| `largest_id` | highest node ID among the current maxima | not used |
| `random` | uniform among the current maxima, from the isolated seeded RNG | **required** |

On a 4-regular ring of 14 nodes the whole sequence is decided by this choice:
`smallest_id` gives `0, 3, 6, 9, 10` and `largest_id` gives `13, 10, 7, 4, 3`.
Note that neither walks the ring one step at a time — that is the adaptivity
showing, because removing node 0 lowers the degree of nodes 1, 2, 12 and 13.

### 16.3 API

```python
from neuromesh.graphs import (
    remove_nodes,              # the shared contracts.py interface, by fraction
    remove_nodes_by_count,     # the primary Track A entry point, by integer k
    remove_nodes_by_fraction,  # by fraction, returning the full result
    removal_count,             # the fraction -> count rule
    validate_attacked_graph,   # the right validator for an attacked graph
)
```

```python
from neuromesh.graphs import assign_roles, generate_graph, remove_nodes_by_count

graph  = assign_roles(generate_graph("ba", 112, seed=0), seed=0)
result = remove_nodes_by_count(graph, 33, "targeted")

result.n_surviving            # 79
result.removed_nodes          # the adaptive attack sequence, first strike first
result.removed_degrees        # the degree each node had WHEN it was removed
result.original_degrees       # its degree in the original graph
result.connected              # False -- the attack worked
result.n_components           # how badly it fragmented
result.readout_surviving      # surviving frontoparietal agents
result.to_dict()              # JSON-serialisable; excludes the graph
graph.number_of_nodes()       # still 112 -- the input is never modified
```

The shared interface, which is what the runner calls:

```python
from neuromesh.contracts import RemovalSpec
from neuromesh.graphs import remove_nodes

attacked = remove_nodes(graph, RemovalSpec(mode="targeted", fraction=0.3), seed=0)
attacked.graph["removal"]["n_removed"]    # 33
attacked.graph["removal"]["n_surviving"]  # 79
```

`remove_nodes` returns a **bare `nx.Graph`**, like `generate_graph`. Prefer
`remove_nodes_by_count` when you need the removal order, because the adapter
discards everything but the graph.

### 16.4 What is preserved, and what is not

Preserved, because the result *is* the induced subgraph:

- **Node IDs.** No renumbering, no compaction. Node 57 is still node 57 in a
  graph with no node 56, so logs join on node ID across removal levels.
- **Edges** — exactly the original edges whose endpoints both survived. None
  added, none rewired.
- **Node and edge attributes**, so `role` and `community` stay on the agents
  that had them.
- **The input graph.** `subgraph()` returns a view; `.copy()` detaches it.

Deliberately **not** preserved:

- **Connectivity.** Reported, never restored. No edges are added to compensate.
- **Mean degree 4.** It falls, as it must.
- **Role balance.** Not rebalanced. The surviving histogram is reported in
  `result.role_counts`.

`G.graph` on the result: `topology`, `n` and `seed` are **left untouched**
because they identify the experimental condition, not the live node count —
use `G.graph["removal"]["n_surviving"]` for that. `mean_degree` **is** updated
to the realised post-attack 2E/N', because §1 defines it as the realised value
and a stale 4.0 on an attacked graph would be a false claim.

### 16.5 Removal counts, and the rounding rule

Integer `k` is the primary API. `removal_count(n, fraction, rule)` converts a
fraction, with **`floor` as the single rule applied at every scale**:

| N | 10% | 20% | 30% |
|---|---|---|---|
| 112 | 11 (9.82%) | 22 (19.64%) | 33 (29.46%) |

**Quote the realised fraction, not the nominal one.** `floor` is the default
because it never removes more than requested and guarantees `k < N` for every
`fraction < 1`, so a fractional request cannot wipe out the network through
rounding. `rule="nearest"` is available and gives 34 at 30%.

Two consequences worth knowing. `floor` rounds small-N requests to zero — 10%
of N = 7 is `floor(0.7) = 0`, a no-op; the robustness experiment is N = 112
only, so this does not affect it. And random removal is drawn as a truncated
permutation, so **removal sets nest**: at a fixed seed the 10% set is a prefix
of the 20% set, which is a prefix of the 30% set. Adaptive targeted order
nests for the same reason. The Week 6 robustness curve is therefore a curve
over one nested sequence of failures rather than three unrelated draws.

Invalid inputs, documented:

| Input | Result |
|---|---|
| `k = 0` | exact no-op; a detached copy is still returned |
| `k = N` | empty graph; `mean_degree_surviving` is `None` |
| `k > N`, `k < 0` | `GraphParameterError` (a `ValueError`) |
| `k` not an `int` — including `3.0` and `True` | `TypeError` |
| `fraction` outside `[0, 1]` | `GraphParameterError` |
| `fraction` not a number | `TypeError` |
| `strategy="random"` without a seed | `GraphParameterError` |
| `tie_break="random"` without a seed | `GraphParameterError` |
| directed or non-graph input | `GraphParameterError` |

`3.0` raises rather than being read as `3` on purpose: silently rounding here
is how `k = 2.5` becomes an attack nobody requested.

### 16.6 Readout agents are NOT protected — a deliberate deviation

The Week 1 placeholder docstring for `remove_nodes` promised it would "never
remove every readout node". **That promise is deliberately not implemented.**

Silently sparing nodes would make the attack something other than the attack
that was requested, and would hide exactly the failure mode the robustness
experiment exists to find. Instead the outcome is measured:
`result.readout_surviving`, `result.role_counts` and `result.roles_intact`,
all mirrored into `G.graph["removal"]`.

A caller who genuinely wants a protected set can opt in:

```python
readout = [v for v, d in graph.nodes(data=True) if d["role"] == "frontoparietal"]
result  = remove_nodes_by_count(graph, 33, "targeted", protected=readout)
```

**Raised for the team as a design question.** If the consensus is that runs
with no surviving readout agent should be prevented rather than recorded, the
decision belongs to Track C (who score the readout) and Track D (who
orchestrate), not to the removal code. In practice the case is remote at the
experiment's scale — N = 112 has 16 readout agents and the largest removal
level takes 33 nodes — but "remote" is not a reason to promise something the
code does not do.

### 16.7 Validating an attacked graph

Do **not** pass an attacked graph to `validate_graph()`: its `connected` and
`mean_degree_tolerance` checks encode generator contracts, so it would report
a `FAIL` for the attack working as designed. Reporting that FAIL would be a
methodological error.

```python
from neuromesh.graphs import validate_attacked_graph

for check in validate_attacked_graph(graph, result):
    print(check.name, check.status, check.detail)
# partition PASS / induced_subgraph PASS / node_attributes_preserved PASS
# edge_attributes_preserved PASS / no_duplicate_removals PASS
# no_renumbering PASS / connectivity_reported EXCEPTION (expected after an attack)
```

Connectivity is measured and reported as a declared `EXCEPTION`, never a
`FAIL`.

### 16.8 Reproducibility

Identical `(graph, k, strategy, seed, tie_break)` always gives an identical
removal order, surviving graph and metadata. Three things secure that:

1. **Canonical node ordering** before any sampling or tie-break, so nothing
   depends on dict/set iteration order, insertion history or `PYTHONHASHSEED`.
2. **An isolated RNG** — `SeedSequence(entropy=[seed, REMOVAL_SEED_SALT])`,
   never the global `random` or `numpy.random`. The salt mirrors
   `ROLE_SEED_SALT` (§14): without it, removal at seed *s* would consume the
   same bits that generated the graph at seed *s*. A test perturbs both global
   RNGs between calls and asserts nothing changes.
3. **A deterministic default tie-break** (§16.2).

---

## 17. End-to-end mock integration test (Week 3)

`tests/integration/test_mock_pipeline.py`. 20 tests, **0.40 s** measured
locally (0.77 s including interpreter start). The allocation requires under
60 s.

```bash
pytest tests/integration/                      # the integration test alone
pytest tests/integration/ -q --durations=5     # with timings
pytest                                         # CI discovers it automatically
```

**No CI change was needed.** `pyproject.toml` already sets
`testpaths = ["tests"]`, so the plain `pytest --cov=neuromesh` in
`.github/workflows/ci.yml` picks this file up with no edit to the workflow.

### 17.1 What it actually proves, and what it does not

It runs the planned MVP scenario — ER graph, N = 7 and N = 14, balanced roles,
a consensus-shaped task, a mock LLM client, 10 synchronous rounds, a readout —
with **no API key, no GPU, no network, no paid service and no data download**.
A test blocks `socket.socket` outright during a run, so a future client that
quietly reaches for an HTTP endpoint fails here rather than on someone's bill.

It is **not** a demonstration that the team's real pipeline works, because two
tracks have not shipped the modules that pipeline needs.

**Real repository modules exercised:** `generate_graph`, `validate_graph`,
`assign_roles`, `remove_nodes` (Track A); `MockLLMClient` (Track B);
`LLMRequest`, `LLMResponse`, `AgentTurn`, `TaskInstance`, `CallRecord`,
`RunResult` (shared contracts).

**Production modules that do not exist yet**, each asserted to still raise so
the test file announces the day they land:

| Entry point | Track | Status |
|---|---|---|
| `agents.run_network` | B | `NotImplementedError` — the round engine |
| `tasks.get_task` | C | `NotImplementedError` — the task registry |
| `baseline.run_baseline`, `evaluation.compute_metrics` | C | not implemented, not needed here |
| Track D experiment execution | D | not implemented, not needed here |

**Test-only scaffolding, named as such.** `MockRoundEngine` is a test harness,
not a product: the simplest synchronous protocol that can be reasoned about —
every agent forwards everything it knows to every neighbour, once per round,
applied simultaneously at end of round. It lives in the test file, is not
importable as a production module, and is **not** a substitute for Track B's
engine. `consensus_instance` likewise hand-builds a `TaskInstance` from the
shared contract type because `get_task("consensus")` raises; it is a fixture,
not Track C's task.

### 17.2 What is asserted

- **Role alignment** — every agent has a YEO7 role, exactly N/7 per role.
- **Flow only along edges** — every delivery is checked against
  `graph.has_edge`, and **every message the LLM client itself addresses** is
  checked against that agent's neighbour set and the 80-word contract limit.
  That check is vacuous on content today (the placeholder client sends no
  messages) but live on plumbing: when the real deterministic mock lands, a
  message routed to a non-neighbour fails **here**, in Track A's test
  directory — which is what `CONTRIBUTING.md` §8 prescribes.
- **No phantom nodes** — no message, log row or delivery names a node absent
  from the graph.
- **Propagation respects graph distance** — knowledge after round *r* is
  compared against shortest-path distances computed independently with
  NetworkX. The `<=` direction is protocol-independent; `==` holds for this
  harness because its rule is an unconditional flood.
- **Determinism** — identical runs give identical knowledge sets, identical
  delivery lists and identical `CallRecord`s, and each round advances
  knowledge by exactly one hop.
- **Log schema** — a real `RunResult` with N × rounds real `CallRecord`s is
  assembled and round-tripped as JSON, which is what Track D's Parquet writer
  needs.
- **Removal inside the pipeline** — a node is removed, its incident edges are
  confirmed gone and no others touched, the run completes on the attacked
  graph, and agents stranded in a component with no source correctly end up
  knowing nothing.

### 17.3 Timing honesty

There are no sleeps and no wall-clock assertions anywhere in the file, so
nothing in it can make CI flaky. One test measures the whole scenario against
a deliberately loose 20-second ceiling — tight enough to catch a regression,
loose enough that a slow shared runner cannot trip it.

**Local runtime does not predict GitHub Actions runtime, and CI has not been
run on this branch.** All timings here are Windows / Python 3.11.9 / local.

---

## 18. N = 112 diameters and the round budget (Week 3)

`src/neuromesh/graphs/analysis.py`, exposed through the Track A CLI.

```bash
python -m neuromesh.graphs.cli diameters                     # the 25-row study table
python -m neuromesh.graphs.cli diameters --check-fixtures    # + fixture cross-check
python -m neuromesh.graphs.cli diameters --json out.json     # + machine-readable
python -m neuromesh.graphs.cli diameters --n 14 --seeds 0 1  # any scale / seed set
```

Deterministic, offline, no downloads; uses the committed empirical Yeo matrix
for `yeo_sbm`. Graphs come from the public `generate_graph` at default
settings, so the rows describe exactly the graphs the experiment will run.

**Exit code.** 1 means an *execution* problem — a generated graph that breached
the connectivity contract, or a committed fixture whose diameter disagrees with
the regenerated graph. A diameter over the round budget is a **finding, not an
error**, and leaves the exit code 0: reporting it is the command's job.

Equivalent Python:

```python
from neuromesh.graphs import (
    aggregate_path_metrics,
    diameter_table,
    round_budget_assessment,
)

rows = diameter_table(112)                       # 5 topologies x 5 seeds = 25 rows
aggregate_path_metrics(rows)                     # per-topology statistics
round_budget_assessment(rows, (10, 12))          # which conditions fit which budget
```

### 18.1 Method

`networkx.diameter` and `networkx.average_shortest_path_length` on the
unweighted graph (BFS; all edges cost 1 hop). Seeds 0-4, the study's set.
Standard deviation is the **sample** standard deviation (`ddof = 1`,
Bessel-corrected), because the claim is about the topology rather than about
these five graphs; the population value is also returned as `std_population`.

For a disconnected graph the diameter is infinite and **no finite stand-in is
invented**: `diameter` is `None` and `max_finite_distance` — the widest
component's diameter — is reported separately under its own name. All 25
generated graphs are connected, so this path exists for attacked graphs and
for reporting a contract breach honestly.

**Fixture cross-check:** the five committed seed-0 N = 112 fixtures were
reloaded and gave identical diameters and edge counts to the regenerated
graphs (5/5 OK). A test additionally pins the three Week 1 values recorded in
§10 (ER 7, BA 5, WS 10), so a generator change that moves them cannot pass
silently.

### 18.2 Results — all 25 conditions

Every graph: 112 nodes, 224 edges, mean degree exactly 4.0000, connected,
1 component.

| Topology | Seed | Diameter | Avg path | ≤10 rounds | ≤12 rounds |
|---|---|---|---|---|---|
| er | 0 | 7 | 3.5072 | yes | yes |
| er | 1 | 7 | 3.4955 | yes | yes |
| er | 2 | 7 | 3.5253 | yes | yes |
| er | 3 | 8 | 3.5983 | yes | yes |
| er | 4 | 8 | 3.5484 | yes | yes |
| ws | 0 | 10 | 5.1860 | yes | yes |
| ws | 1 | **11** | 5.3274 | **NO** | yes |
| ws | 2 | 10 | 4.9427 | yes | yes |
| ws | 3 | 9 | 4.7954 | yes | yes |
| ws | 4 | **13** | 5.8332 | **NO** | **NO** |
| ba | 0 | 5 | 2.9791 | yes | yes |
| ba | 1 | 6 | 3.1075 | yes | yes |
| ba | 2 | 6 | 3.0224 | yes | yes |
| ba | 3 | 5 | 3.0845 | yes | yes |
| ba | 4 | 6 | 2.9899 | yes | yes |
| modular | 0 | 7 | 3.6427 | yes | yes |
| modular | 1 | 7 | 3.6176 | yes | yes |
| modular | 2 | 7 | 3.5891 | yes | yes |
| modular | 3 | 8 | 3.5349 | yes | yes |
| modular | 4 | 8 | 3.6210 | yes | yes |
| yeo_sbm | 0 | 7 | 3.4596 | yes | yes |
| yeo_sbm | 1 | 7 | 3.5267 | yes | yes |
| yeo_sbm | 2 | 8 | 3.6435 | yes | yes |
| yeo_sbm | 3 | 7 | 3.5510 | yes | yes |
| yeo_sbm | 4 | 7 | 3.5384 | yes | yes |

Aggregate (std dev = sample, `ddof = 1`):

| Topology | Min | Max | Mean | Median | Std dev | ≤10 | ≤12 | >10 | >12 |
|---|---|---|---|---|---|---|---|---|---|
| er | 7 | 8 | 7.40 | 7.0 | 0.5477 | 5 | 5 | 0 | 0 |
| ws | 9 | **13** | 10.60 | 10.0 | 1.5166 | 3 | 4 | **2** | **1** |
| ba | 5 | 6 | 5.60 | 6.0 | 0.5477 | 5 | 5 | 0 | 0 |
| modular | 7 | 8 | 7.40 | 7.0 | 0.5477 | 5 | 5 | 0 | 0 |
| yeo_sbm | 7 | 8 | 7.20 | 7.0 | 0.4472 | 5 | 5 | 0 | 0 |

### 18.3 Ten versus twelve rounds

| Budget | Conditions covered | Verdict |
|---|---|---|
| 10 rounds | **23 / 25** | insufficient — WS seeds 1 (d=11) and 4 (d=13) |
| 12 rounds | **24 / 25** | insufficient — WS seed 4 (d=13) |
| 13 rounds | 25 / 25 | smallest budget covering every measured condition |

**Going from 10 to 12 rounds does not resolve all measured violations.** It
fixes WS seed 1 and leaves WS seed 4 short by one hop. WS is also the only
topology that ever exceeds either budget: ER, BA, modular and Yeo-SBM all peak
at diameter 8, which clears 10 rounds with 2 hops of slack.

WS is the outlier by construction — a rewired ring with p = 0.1 at mean degree
4 keeps long-range structure that the other four do not have, and its
seed-to-seed spread (9 to 13, std 1.52) is 2.8x the next largest
(0.55 for ER, BA and modular).

### 18.4 What "sufficient" means here — read before quoting these numbers

`diameter <= rounds` is **necessary, not sufficient.** It is the weakest
possible requirement: that a shortest path has time to be traversed once.
A graph of diameter *d* needs at least *d* rounds before the most distant pair
can have exchanged anything at all.

It says **nothing** about whether intermediate agents forward information
faithfully, whether a consensus task converges, or whether an LLM reasons
correctly over what arrives. **No LLM convergence has been validated** — the
round engine does not exist yet.

Three further caveats, all real:

1. **The off-by-one is Track B's to settle.** Whether *d* rounds suffice or
   *d + 1* are needed depends on conventions nobody has implemented: whether
   round 1 is the first send or an initialisation turn, whether a message sent
   in round *r* is readable in round *r* or *r + 1*, and whether the final
   readout consumes a round. The numbers above are pure graph distances. The
   tables report `d <= budget`, so if the team later finds an extra round is
   consumed, re-read them against `budget - 1` without remeasuring — at which
   point **10 rounds would cover 23/25 and 12 rounds would also cover only
   23/25**, since WS seed 1 (d = 11) would then fail a 12-round budget too.
2. **Round-trip tasks need more than one traversal.** A task that must combine
   information from the far end and return a verdict to a readout agent needs
   something closer to *d* plus the input-to-readout distance. That is Week 4
   work (positional covariates) and is not computed here.
3. **Diameter is a worst case over pairs.** Average path length is 3.0-5.8
   everywhere, so a diameter violation degrades the tail of the pair
   distribution, not the whole run.

### 18.5 Recommendation

**12 rounds, with WS at N = 112 reported as diameter-limited** — and the team
should decide explicitly rather than inherit the default.

Reasoning. 12 rounds covers 24/25 conditions and costs 20% more LLM calls than
10. Going to 13 buys the 25th condition (WS seed 4) and nothing else, for
another 8% on top; 14 or more would be needed if caveat 1 above turns out to
apply. Against that, every non-WS condition already clears 10 rounds with
slack, so the extra rounds buy coverage for one topology at one scale.

Two alternatives the team may prefer, both defensible:

- **Keep 10 rounds** and report WS N = 112 propagation as diameter-limited for
  2 of 5 seeds. Cheapest, and honest if the limitation is stated.
- **Go to 13 rounds** and have graph-theoretic reachability for every measured
  condition with no caveat.

What Track A will not do unilaterally: `DEFAULT_ROUNDS` lives in
`contracts.py`, which is frozen at v1.0 and owned by all four members, so
changing it needs a `contracts/` PR. **No parameter has been changed.** This
section is the measurement and the recommendation; the decision is the team's.

---

## 19. Week 3 limitations and follow-ups

1. **`remove_nodes` does not protect readout agents**, contrary to its Week 1
   placeholder docstring. Deliberate; measured and reported instead. Raised
   for the team as a design question — §16.6.
2. **The round-budget result is graph-theoretic only.** No LLM convergence,
   consensus or task success has been validated, and the protocol off-by-one
   is unresolved pending Track B — §18.4.
3. **Fraction rounding is `floor`,** so the study's 10/20/30% levels actually
   remove 9.82/19.64/29.46% at N = 112. Report the realised fraction — §16.5.
4. **The integration test is a test-only harness,** not proof that the team's
   real pipeline works. `run_network` and `get_task` remain unimplemented —
   §17.1.
5. **CI has not been run on this branch.** Every number in §17 and §18 is
   local (Windows, Python 3.11.9, NetworkX 3.6.1, NumPy 2.4.6).
6. **Still open from Week 2, unchanged:** the N = 7 / N = 14 block-model
   limits (§15.1-2), fixed-edge-budget block sampling (§12), the
   role-placement confound across topology families (§14.1), the Yeo block
   means mixing strength and density (§13.1), the unpublished ENIGMA subject
   count (§15.4), and `load_graph()` resolving fixtures relative to the
   working directory (§15.6).
7. **Not done, deferred to Week 4:** `structural_metrics` remains a
   placeholder. Positional covariates, robustness curves and AUC analysis are
   Weeks 4-6 and were deliberately not started.

---

## 20. Structural metrics (Week 4)

`src/neuromesh/graphs/metrics.py`. `structural_metrics` — the last Track A
placeholder in `contracts.py` — is implemented. One row per graph, 43 fields.

```python
from neuromesh.graphs import structural_metrics
metrics = structural_metrics(graph)        # the one-argument contract call
```

Identity (`topology`, `n`, `seed`) is read from `graph.graph` when not passed,
so the contract's single-argument call identifies the graph by itself.

### 20.1 What the contract asked for, and where it is

| `contracts.py` wording | Key(s) |
|---|---|
| path length | `average_shortest_path_length`, `diameter`, `radius` |
| clustering | `average_clustering`, `transitivity` |
| modularity | `modularity_role_partition` |
| max degree | `degree_max` (+ min/mean/variance/std/heterogeneity) |
| input-readout distance | `input_readout_mean_distance`, `_min_`, `_max_` |
| hub roles | `hub_role`, `hub_degree` |

### 20.2 Definitions

| Metric | Definition |
|---|---|
| `mean_degree` | 2E/N |
| `edge_density` | 2E / (N(N−1)); `None` for N < 2 |
| `average_clustering` | mean of the local coefficients (`nx.average_clustering`) |
| `transitivity` | 3×triangles / triads — a **global** ratio, not the mean of local values, so it differs from `average_clustering`; both reported |
| `average_shortest_path_length` | whole graph; `None` if disconnected |
| `diameter`, `radius` | whole graph; `None` if disconnected |
| `*_largest_component` | the same two on the largest component, under **separate names** |
| `degree_variance` | **population** variance (ddof=0) |
| `degree_heterogeneity` | ⟨k²⟩/⟨k⟩² — exactly 1.0 for a regular graph, dimensionless so it compares across degree-matched graphs |
| `global_efficiency` | mean over ordered pairs of 1/d, with 1/d = 0 for an unreachable pair. **Defined for disconnected graphs**, which is why the robustness curves use it |
| `degree_assortativity` | Pearson degree-degree correlation; `None` when undefined |
| `modularity_role_partition` | Newman–Girvan Q of the **predefined** seven-network partition, resolution 1.0 |

### 20.3 Undefined values are `None`, never 0.0

Some quantities genuinely do not exist for some graphs. A disconnected graph
has no finite diameter; a regular graph's degree correlation is 0/0. Returning
0.0 would be a **false measurement** that silently pollutes every average
taken over it, and 0.0 assortativity is a strong claim ("measured, and
neutral") rather than an absence.

So every such field is `None`, and `datasets.aggregate_conditions` reports a
`*_n` count of valid observations beside each statistic. **A mean over 3 of 5
seeds is never presented as a mean over 5.** This is not hypothetical: WS at
N = 7 is a regular ring for one of the five seeds, so its assortativity is
defined for 4/5 — visible in `condition_summaries.csv` as
`degree_assortativity_n = 4`.

### 20.4 Full-graph vs largest-component metrics are named apart

`diameter` is `None` for a disconnected graph and
`diameter_largest_component` carries the readable value. They are never
substituted, because quoting a surviving component's diameter as the graph's
diameter understates exactly the damage a robustness experiment measures.

### 20.5 Modularity uses the predefined partition, NOT community detection

The question is *"how modular is the link arrangement with respect to the
seven functional networks the agents actually play"*, which has one answer per
graph. Community **detection** would answer a different question — "what is
the most modular partition available" — which is topology-dependent in a way
that makes the five families incomparable, and which would score ER against a
partition ER does not have.

The same fixed partition is therefore applied to all five topologies. For the
block models it coincides with the planted blocks by construction, so high
values are expected; for ER/WS/BA the roles are a random labelling of
structural positions, so values at or below zero are expected. **That contrast
is the measurement** — and §24 explains why it must not be read as a pure
topology effect.

### 20.6 Measured results, N = 112, mean over 5 seeds

| Topology | Clustering | Avg path | Efficiency | Modularity Q | Degree var | Heterogeneity |
|---|---|---|---|---|---|---|
| er | 0.0345 | 3.5349 | 0.3247 | −0.0145 | 3.364 | 1.210 |
| ws | **0.3653** | **5.2169** | 0.2432 | 0.0096 | **0.400** | **1.025** |
| ba | 0.1357 | 3.0366 | **0.3665** | −0.0133 | **16.514** | **2.032** |
| modular | 0.0732 | 3.6011 | 0.3194 | **0.4821** | 3.118 | 1.195 |
| yeo_sbm | 0.0440 | 3.5439 | 0.3239 | 0.1818 | 3.379 | 1.211 |

Four things worth recording:

1. **WS keeps high clustering at every scale** (0.35–0.38 from N = 14 up)
   while the others decay toward 0 — the small-world signature, and the price
   is the longest average path (5.22) and lowest efficiency (0.243).
2. **BA has 4–40× the degree variance of any other topology** (16.5 vs
   0.40–3.38) and the highest max degree (25.4 vs 5.8–9.2). This is the
   structural feature that varies most across the families, and §22.9 reports
   it as **associated** with the targeted-attack ordering. With only five
   topologies no causal claim is testable from these data.
3. **Modular reaches Q = 0.4821 at N = 28, 56 and 112 — identically.** The
   fixed-edge-budget allocation (§12) splits within/between edges in the same
   proportion at every scale, so Q is scale-invariant above the capacity cap.
   At N = 14 it is only 0.107 and at N = 7 it is −0.151, which is the Week 2
   finite-size limitation (§15.1–2) reappearing as a metric.
4. **Yeo-SBM's Q (0.18) is far below modular's (0.48)** because the empirical
   matrix sets *contrast*, not maximal separation (§13.4): real inter-network
   structural connectivity is not a clean block diagonal.

---

## 21. Node-level positional covariates (Week 4)

`metrics.positional_covariates(graph)` → one row per node, ascending node ID.
**5,425 rows** over the study grid (5 topologies × 5 seeds × 217 nodes).

```python
from neuromesh.graphs import positional_covariates
rows = positional_covariates(graph)        # 22 fields per node
```

### 21.1 Why node-level at all

The study's outcome is measured *at specific agents*: information enters at
`INPUT_ROLE` and is read out at `READOUT_ROLE`. Whether a readout agent landed
on a hub or on a periphery node is a covariate a per-graph average cannot
express — and without it a topology effect and a placement effect are not
separable.

### 21.2 Definitions

| Covariate | Definition |
|---|---|
| `degree_centrality` | degree/(N−1); `None` for N < 2 |
| `betweenness_centrality` | `nx.betweenness_centrality`, **normalised**, so values are in [0,1] and comparable across the five scales |
| `closeness_centrality` | `nx.closeness_centrality` with `wf_improved=True` (Wasserman–Faust). Matters only for disconnected graphs; all 125 study graphs are connected, where it reduces to standard closeness |
| `clustering` | local coefficient; 0 for degree < 2 (NetworkX convention) |
| `core_number` | largest k whose k-core contains the node |
| `eigenvector_centrality` | power iteration, **tol 1e-10, max_iter 2000** |
| `pagerank` | **alpha 0.85, tol 1e-12, max_iter 1000** — fixed as module constants, not left to NetworkX's defaults, so the numbers survive a library upgrade |
| `within_network_degree` | neighbours sharing the node's role |
| `between_network_degree` | neighbours with a different role |
| `n_neighbor_networks` | distinct roles among neighbours (0–7) |
| `participation_coefficient` | `P_i = 1 − Σ_s (k_is/k_i)²` over the seven networks |
| `within_module_z` | `z_i = (k_in,i − mean) / std` over same-role nodes, population std (ddof=0) |

### 21.3 The three edge-case conventions, stated

- **Degree-0 node, degree fractions → `None`.** 0/0 is not a fraction. 0.0
  would read as "this node has no within-network links" when in truth it has
  no links at all.
- **Degree-0 node, participation → `0.0` by definition.** An isolated agent
  participates in nothing. (Note this is a *different* choice from the
  fractions above, deliberately: P has a defined value at zero degree, a
  fraction does not.)
- **Zero-variance module, z-score → `0.0`.** Every member is exactly at its
  module's mean, so a deviation of 0 is the correct reading and the
  alternative is a division by zero. This is the normal case at **N = 7**,
  where each network holds one agent, so every z is exactly 0 there.

`participation_coefficient` lies in **[0, 6/7]** for seven networks, never
1.0 — a bound the tests assert over all 5,425 rows.

**Eigenvector centrality is reported honestly.** The pure-Python power
iteration is used deliberately: `nx.eigenvector_centrality_numpy` needs SciPy,
which is **not** a declared project dependency, so it would work locally and
fail in CI. On non-convergence **every** node's value becomes `None` — never a
partial or unconverged number — and `metrics.eigenvector_status(graph)`
reports which. Measured: it converged for all 125 study graphs.

---

## 22. Graph-only robustness reference curves (Week 4)

`src/neuromesh/graphs/curves.py`. **250 trajectories** (125 graphs × 2
strategies), **3,400 curve points**.

### 22.1 This is a reference curve, not a result about the agent system

It is computed from the adjacency alone. It is **not** agent communication
success, **not** LLM response fidelity and **not** task accuracy. A network
whose largest component survives intact may still fail every task, and a
fragmented network may still score well on a task needing only local
information. Week 6 measures the latter; this measures the structural ceiling
those runs operate under, so a Week 6 score drop can be checked against how
much connectivity was actually lost.

### 22.2 Reuse, not reimplementation

Every removal order comes from Week 3's `robustness.py`:
`random_removal_order` and `targeted_removal_order`. The adaptive attack, the
`smallest_id` tie-break, the salted isolated RNG and the `floor` rounding rule
are inherited unchanged. **No removal logic is duplicated and no new rounding
rule is introduced** — tests assert the trajectory's order is character-for-
character the Week 3 order.

### 22.3 Complete curves, cheaply, via the nesting property

Week 3 established that both orders **nest**: the first k entries of a longer
order are exactly the shorter order. So the order is computed **once** at
k_max and evaluated on each prefix. This is not an approximation — the prefix
of the adaptive attack *is* the shorter adaptive attack — and it is what makes
measuring **every integer removal count** from 0 to `floor(0.30N)` affordable
rather than only three checkpoints.

Points per graph: N = 7 → 3, 14 → 5, 28 → 9, 56 → 17, 112 → 34.

### 22.4 The denominator is the ORIGINAL N

`largest_component_fraction` and `fraction_removed` both divide by the
**original** node count. This matters: a graph reduced to 2 joined nodes has a
largest component holding 100% of its *survivors* but 1.8% of the *original*
network. Survivor-normalising would make total collapse look like perfect
robustness and would make curves at different N incomparable.

### 22.5 Efficiency under both normalisations

- `global_efficiency` — `nx.global_efficiency` of the surviving graph,
  normalised over **surviving** pairs. The standard definition.
- `global_efficiency_original_n` — the same pair sum renormalised over the
  **original** N, so removed agents count as unreachable.

The first asks "how efficiently do the survivors communicate among
themselves"; the second asks "how much of the original network's capacity is
left". They diverge sharply: under random failure at N = 112, BA's survivor
efficiency barely moves (0.371 → ~0.36) while its original-N efficiency falls
0.371 → 0.177. Quoting one as the other would misstate the damage, so both are
exported.

### 22.6 Checkpoints

0%, 10%, 20%, 30%, flagged **within** the complete curve rather than measured
separately, so a checkpoint value and the curve can never disagree. Both the
`requested_fraction` and the realised `k_removed` / `fraction_removed` are
recorded.

At N = 112 the levels are k = 0, 11, 22, 33 → realised 0%, 9.82%, 19.64%,
29.46%. **Quote the realised fraction.**

**N = 7 boundary artefact:** `floor(0.1 × 7) = 0`, so at N = 7 the 10%
checkpoint *is* the 0% checkpoint and only three distinct checkpoints exist.
Asserted by a test so it is not rediscovered later.

### 22.7 Measured results — N = 112, 30% removed, mean over 5 seeds

| Topology | Random: LCC frac | Random: components | Targeted: LCC frac | Targeted: components |
|---|---|---|---|---|
| er | 0.668 | 4.8 | 0.120 | 27.6 |
| ws | 0.659 | 3.0 | **0.329** | **7.2** |
| ba | 0.682 | 3.6 | **0.027** | **59.8** |
| modular | 0.679 | 4.0 | 0.191 | 27.4 |
| yeo_sbm | 0.675 | 3.8 | 0.163 | 25.0 |

**Under random failure, topology barely matters** — every family retains
66–68% of the original network in one component, and the spread (0.023) is
smaller than the across-seed spread. **Under targeted attack the families
separate by more than an order of magnitude**, from WS at 0.329 down to BA at
0.027.

### 22.8 Structural AUC — a Week 4 descriptive reference only

**The principal robustness AUC comparison is Week 6 work over LLM task
scores.** The AUC here is graph-only and the two must never be pooled.

| Field | Definition |
|---|---|
| x axis | `fraction_removed` = k / original N, from 0 to k_max/N (≈0.2946 at N = 112, below 0.30 because node counts are integers) |
| y axis | `largest_component_fraction` (denominator: original N) |
| integration | trapezoidal rule over **measured points only** — no interpolation past the last point, no extrapolation |
| normalisation | `auc_raw / (x_max − x_min)`, i.e. the **mean y over the removal range**. This is what makes values comparable across N, whose `x_max` differ through integer rounding. Raw values at different N are **not** comparable. **1.0 is NOT attainable** — see below |

**The attainable ceiling is `1 − x_max/2`, not 1.0.** Because y is normalised
by the **original** N, deleting a node lowers y even when nothing fragments: a
graph that stays fully connected still has `y = 1 − k/N` at step k, and the
mean of that straight line over `[0, x_max]` is `1 − x_max/2`. That gives

| x_max | Applies to | Ceiling |
|---|---|---|
| 2/7 = 0.285714 | N = 7, 14, 28, 56 | **0.857143** |
| 33/112 = 0.294643 | N = 112 | **0.852679** |

Reaching 1.0 would require the largest component to still hold all N original
nodes after k of them were deleted, which is impossible. **Read a measured AUC
against the ceiling, not against 1.0.** At N = 112, BA's random-failure AUC of
0.8458 sits within 0.007 of the 0.852679 ceiling — i.e. almost no
fragmentation beyond the unavoidable loss of the removed nodes themselves —
while its targeted AUC of 0.3993 indicates severe fragmentation.

**0.0 is likewise a theoretical lower reference, not an attainable result**
under this experiment. The input graphs are connected by generator contract,
so `y(0) = 1.0` exactly; and removal stops at `floor(0.30N)`, so at least
0.70N nodes survive and any survivor is itself a component, giving
`y(k) >= 1/N > 0`. The absolute floor (instant collapse to one node) is about
0.3571 at N = 7, 0.1875 at N = 14, 0.0960 at N = 28, 0.0485 at N = 56 and
0.0239 at N = 112. The lowest value observed anywhere in the sweep is 0.3544
(BA, N = 112, seed 0, targeted).

This also explains why **all 25 measured N = 7 random trajectories pin at
0.8571**: none of them fragmented at any step (0 of 75 curve points had a
largest component smaller than N − k), so each curve sat exactly on its
ceiling. That is an **observed property of these 25 graphs and seeded draws,
not a theorem about 7-node mean-degree-4 graphs** — such graphs *can* be
fragmented by deleting two nodes, and within this very dataset 12 of the 25
N = 7 **targeted** trajectories do fall below the ceiling (range
0.7857–0.8571). The practical consequence stands either way: at N = 7 the
random-removal AUC has almost no room to vary, so **N = 7 discriminates
nothing** in the robustness comparison.

Measured normalised AUC at N = 112 (mean over 5 seeds, min–max):

| Topology | Random | Targeted |
|---|---|---|
| er | 0.8327 (0.823–0.841) | 0.7051 (0.659–0.750) |
| ws | 0.8396 (0.819–0.848) | **0.7760** (0.746–0.814) |
| ba | 0.8458 (0.833–0.853) | **0.3993** (0.354–0.431) |
| modular | 0.8421 (0.836–0.848) | 0.7180 (0.690–0.739) |
| yeo_sbm | 0.8394 (0.828–0.853) | 0.7240 (0.700–0.757) |

### 22.9 An observed association, and a trade-off the team should know about

Targeted-attack AUC is **associated** with degree heterogeneity across the
five families. The table below reports that association; it is not a fitted
model and no hypothesis test was performed.

| Topology | Degree variance | Heterogeneity ⟨k²⟩/⟨k⟩² | Max degree | Targeted AUC |
|---|---|---|---|---|
| ws | 0.400 | 1.025 | 5.8 | 0.7760 |
| modular | 3.118 | 1.195 | 8.6 | 0.7180 |
| er | 3.364 | 1.210 | 8.8 | 0.7051 |
| yeo_sbm | 3.379 | 1.211 | 9.2 | 0.7240 |
| ba | 16.514 | 2.032 | 25.4 | 0.3993 |

Three statements, separated by how well the data support them:

**(a) OBSERVED, and robust to seed choice.** WS has the highest targeted AUC
(0.7760) and BA the lowest (0.3993), and their across-seed ranges
(0.7461–0.8137 and 0.3544–0.4309) do not overlap or come close to
overlapping. The same two extremes also hold the lowest and highest degree
heterogeneity (1.025 and 2.032). This ordering is a direct measurement.

**(b) NOT SEPARATED BY THESE DATA.** The three middle families sit within
0.019 of one another (modular 0.7180, yeo_sbm 0.7240, er 0.7051) with
across-seed ranges that overlap heavily (0.6899–0.7390, 0.6995–0.7574,
0.6594–0.7503). Five seeds per condition do not order them. **No
significance test was performed and none is claimed** — the ranges are
reported so a reader can see the overlap directly. Note too that er and
yeo_sbm have almost identical heterogeneity (1.210 vs 1.211) but AUCs 0.019
apart, which is itself a sign that heterogeneity alone does not determine the
outcome at this resolution.

**(c) INTERPRETATION, consistent with (a) but not demonstrated here.** The
standard network-science account of (a) is that a near-regular graph offers an
attacker no high-degree target, while a hub-dominated graph loses
disproportionate connectivity when its hubs go first — the *robust-yet-fragile*
property of scale-free graphs. That account is consistent with every number in
the table, and BA's behaviour matches it closely (removing 11 nodes, 10%,
already drops its largest component to 34.8%). But this study measured five
topologies, not a controlled sweep of heterogeneity at fixed topology, so the
mechanism is **imported from the literature as an explanation, not established
by this measurement.** Establishing it would need heterogeneity varied
deliberately with other structure held fixed.

The trade-off below rests only on (a):

> **WS has the worst round-budget behaviour (§18: the only topology exceeding
> 10 or 12 rounds, diameter up to 13) but the best targeted-attack robustness
> (AUC 0.776, 7.2 components vs BA's 59.8). BA is the reverse: the shortest
> paths (diameter 5–6) and the most fragile structure.**

Neither is "better". Reporting them together is the honest read, and it is a
more interesting result than either alone.

---

## 23. Week 4 outputs, figures and how to regenerate them

### 23.1 One command

```bash
python -m neuromesh.graphs.cli week4-analysis
```

Deterministic, offline, no downloads; `yeo_sbm` reads the committed empirical
matrix. **All 13 output files are byte-reproducible** across runs (verified by
SHA-256). Measured runtime: **~12 s** total (structural 1.0 s, positional
2.1 s, robustness 9.0 s, figures 0.05 s).

Focused subsets, and configuration:

```bash
python -m neuromesh.graphs.cli structural-analysis      # 125 rows + 25 summaries
python -m neuromesh.graphs.cli positional-covariates    # 5425 node rows
python -m neuromesh.graphs.cli robustness-curves        # 250 trajectories + AUC
python -m neuromesh.graphs.cli week4-analysis --only structural robustness
python -m neuromesh.graphs.cli week4-analysis --out /tmp/w4 --sizes 28 112
python -m neuromesh.graphs.cli week4-analysis --no-write   # validate only
```

Exit code 1 means a **structural invariant failed** — a disconnected original
graph, an unbalanced role assignment, a wrong row count, a mean degree that is
not 4, a curve not starting at k = 0, or a largest component that grew. The
command validates as it goes rather than only writing.

Equivalent Python: `neuromesh.graphs.run_week4_analysis(out_dir, ...)`.

### 23.2 Outputs — `data/analysis/week4/` (2.2 MB, 13 files)

| File | Rows |
|---|---|
| `structural_metrics.csv` | 125 |
| `condition_summaries.csv` | 25 |
| `positional_covariates.csv` | 5,425 |
| `robustness_curves.csv` | 3,400 |
| `robustness_checkpoints.csv` | 950 |
| `robustness_auc.csv` | 250 |
| `figure{1,2,3}_*.svg` | 3 figures |
| `figure_data_{1,2,3}_*.csv` | 100 / 340 / 600 |
| `manifest.json` | grid, conventions, file list, disclaimer |

Determinism rests on three things: fixed iteration order (topology, size,
seed, node — no dict order reaches the output); fixed column tuples; and
**`repr` float formatting**, the shortest string that reads back as the
identical double. `f"{x:.6f}"` was rejected — it silently discards precision,
so a table written and re-read would no longer reproduce its source metric.
CSV `None` is the empty field; JSON `None` is `null`. Line endings are forced
to `\n` so files are identical on Windows and Linux.

Wall-clock timings are deliberately **not** written into `manifest.json`: they
vary per run, which would make the manifest the one non-reproducible output.
They are printed to stdout instead.

### 23.3 Figures are hand-written SVG, and why

**`matplotlib` is NOT a declared project dependency.** `pyproject.toml` lists
`networkx numpy pandas pyarrow pydantic pyyaml` plus dev `pytest pytest-cov
ruff`, and that file is Track D's under `.github/CODEOWNERS`. matplotlib
*happens* to be installed on this development machine, which makes it a trap:
a figure script importing it would work locally and fail in CI, whose install
is exactly `pip install -e ".[dev]"`.

Rather than edit a teammate's file or ship a step that runs on one laptop, the
figures are emitted as SVG from the standard library alone. Side benefits: SVG
is text, so figures diff in review; they are byte-reproducible, so the
determinism tests cover them; and they need no display, backend or font cache
in CI. A test asserts `figures.py` contains no matplotlib/plotly/seaborn
import.

**Every figure also writes its exact plotted values** as `figure_data_*.csv`,
so anyone preferring matplotlib, R or a spreadsheet can reproduce them without
re-running the analysis.

| Figure | Content |
|---|---|
| **1** | 4 panels — clustering, average path length, global efficiency, predefined-partition modularity — against N (log₂ spacing), one line per topology, ±1 SD error bars |
| **2** | Largest-component fraction vs fraction removed, random and targeted panels, 5 topologies, N = 112 seed 0 |
| **3** | Mean LCC fraction with ±1 SD bands across 5 seeds, panels for N = 28/56/112, solid = random, dashed = targeted |

Honesty rules applied to all three: straight segments join **measured points
only** (no smoothing, no spline, nothing drawn past the last point); error bars
are ±1 **sample** SD (ddof=1) with the seed count printed on the figure; a
metric undefined for a condition is **omitted** rather than drawn as zero;
fraction axes start at zero; and each figure states on its face that it is
graph-only with no LLM or task claim.

Figures 2 and 3 pick their cell and panels from the data actually present
(largest N, smallest seed; three largest sizes) rather than hard-coding
112/0 — a fixed default silently produced an **empty** figure on a reduced
grid, and a figure that looks finished and shows nothing is worse than an
error.

### 23.4 Two standard-deviation conventions, both stated

- **Within a graph**, `degree_variance` / `degree_std` are the **population**
  values (ddof=0): a graph's degree sequence is the whole population of its
  nodes.
- **Across the five seeds**, `condition_summaries.csv` uses the **sample**
  standard deviation (ddof=1): five seeds are a sample from the generator's
  ensemble and the claim is about the topology, not those five graphs.

They answer different questions and are not interchangeable.

---

## 24. Role-placement confounding — the Week 4 measurement

§14.1 raised this as an interpretation warning. Week 4 **measured** it. The
three questions the Week 4 brief asks, answered empirically:

### 24.1 Is the node→role mapping identical across families at fixed (N, seed)?

**Within a group, yes. Across the two groups, no.** Measured at N = 28, seeds
0–2:

| Comparison | Identical mapping? |
|---|---|
| er vs ws vs ba | **Yes** — all three share one randomised mapping per seed |
| modular vs yeo_sbm | **Yes** — both read role off the community index |
| {er,ws,ba} vs {modular,yeo_sbm} | **No** |

### 24.2 Does the mapping vary across seeds?

| Family | Distinct mappings over seeds 0–4 | Regime | Seed-dependent |
|---|---|---|---|
| er, ws, ba | **5** | `randomised_balanced` | yes |
| modular, yeo_sbm | **1** | `community` | **no** |

### 24.3 Is block role placement locked to community structure?

**Completely.** For `modular` and `yeo_sbm`, `role == YEO7[community]` for
**every node of every graph** — verified across the grid. ER/WS/BA carry no
`community` attribute at all.

### 24.4 Which comparisons are confounded, and by how much

| Comparison | Clean read on topology? |
|---|---|
| ER vs WS vs BA | **Yes** — same regime, placement averaged over seeds |
| modular vs Yeo-SBM | **Yes** — same regime, placement fixed by blocks |
| ER/WS/BA vs modular/Yeo-SBM | **No** — topology and placement regime vary together |

The magnitude is now quantified rather than asserted. Predefined-partition
modularity at N = 112: **modular 0.482 and Yeo-SBM 0.182, against ER −0.015,
WS 0.010 and BA −0.013.** That gap is *largely definitional*: for the block
models the partition being scored IS the planted structure, while for the
others it is a random labelling. A reader who took "modular is 30× more
modular than ER" as a topology effect would be reading the role-assignment
regime, not the link arrangement.

Two further consequences, both measured:

- **Variance is not comparable across the boundary.** Block topologies'
  seed-to-seed spread reflects link placement only; ER/WS/BA spread reflects
  link placement **and** role placement. Visible in the data: modular's
  modularity SD is 0.0000 at N = 28/56/112 (placement fixed, and the
  edge-budget split is scale-invariant) while ER's is nonzero at every scale.
  Error bars from the two families are not measuring the same thing.
- **Within-network degree differs structurally too**, not only by label: the
  block models concentrate more links inside networks, which is asserted by a
  test over the full grid.

### 24.5 What was NOT done, and what is proposed

**Week 2 role assignment was not altered.** `roles.assign_roles` is untouched;
Week 4 only measured and exported. Changing it would be a scientific change to
a merged interface and is a team decision, not a Track A implementation
choice.

**Proposed controlled sensitivity analysis, for team approval:** add one extra
condition — a block topology (`modular` or `yeo_sbm`) run with **deliberately
shuffled role labels**, i.e. balanced roles assigned by the `randomised_
balanced` regime instead of read off the communities. That breaks
role–community alignment while keeping the link arrangement identical, so the
difference between it and the standard block condition isolates the placement
effect from the topology effect. It needs no new generator — only an optional
flag on `assign_roles` to force the randomised regime on a graph that carries
communities.

Cost estimate: one extra flag, 5 seeds × 5 sizes × 1 topology = 25 extra graph
conditions for the graph-only analysis, and (if the team wants it in the LLM
sweep) a proportional increase in runs. **Raised as a Week 5/6 design question
and not implemented.**

---

## 25. Week 4 limitations and follow-ups

1. **Graph-only, throughout.** Nothing in §20–§23 measures agent behaviour,
   message fidelity, consensus or task accuracy. The robustness curves are a
   structural reference for Week 6, not a substitute for it.
2. **The structural AUC is descriptive and is not the Week 6 AUC.** Different
   y axis (connectivity, not task score), different purpose. Do not pool them.
3. **Five seeds is few for the middle of the AUC ordering.** WS vs BA is
   unambiguous; ER vs modular vs Yeo-SBM under targeted attack (0.705/0.718/
   0.724) is **not resolved** and should not be claimed. WS's N = 112 diameter
   SD of 1.52 (§18) is the same caution in another metric.
4. **Realised removal fractions are 9.82/19.64/29.46%**, not 10/20/30%, under
   the inherited `floor` rule. Quote the realised value.
5. **N = 7 remains a boundary case in new ways:** the 10% checkpoint collapses
   into 0%; every within-module z-score is exactly 0 (one agent per network);
   modular carries no community structure so its Q is negative; and Yeo-SBM is
   fully determined, so its clustering SD across seeds is exactly 0.0000.
6. **Role-placement confound is measured but not controlled** (§24). The
   sensitivity condition is proposed, not implemented.
7. **Still open from Weeks 1–3, unchanged:** fixed-edge-budget block sampling
   (§12); the Yeo block means mixing strength and density (§13.1); the
   unpublished ENIGMA subject count (§15.4); `load_graph()` and now
   `DEFAULT_OUTPUT_DIR` resolving relative to the working directory (§15.6);
   fixtures shipped at seed 0 only; and the unresolved 10-vs-12-vs-13 round
   budget (§18) — **not touched in Week 4.**
8. **`data/analysis/week4/` is 2.2 MB of generated CSV** and is **not** in
   `.gitignore`. It is deterministic and regenerable in 12 s from one command,
   so whether to commit it or treat it as a build artefact is a team call —
   flagged for Track D, who own `.gitignore`. Track A's view: commit it, so
   the progress report's numbers are reviewable without running anything.
9. **Coverage instrumentation makes the suite ~4× slower** (42 s plain vs
   191 s under `--cov`), because the heavy numeric loops are interpreted
   per-line. The three full-grid builds are session-scoped fixtures so each is
   computed once, but this is worth knowing before adding more full-grid
   tests.
10. **Deferred by design:** Week 5 robustness configuration and dry runs, and
    Week 6 random-vs-targeted LLM experiments with task-score AUC. Not
    started.
