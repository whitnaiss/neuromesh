# Track A — Graphs & Network Data (Mihir Mathur)

Modules: `src/neuromesh/graphs/`, `data/`, `configs/robustness.yaml`, `tests/integration/`.

**Status: Week 3 complete — node removal (random and adaptive targeted), the
end-to-end mock integration test, and the N = 112 diameter / round-budget
analysis, on top of Week 2's five topologies, balanced role assignment,
derived Yeo 7×7 matrix, validation and fixtures.**
`structural_metrics` remains a placeholder for Week 4.

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
