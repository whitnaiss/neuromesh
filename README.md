# NeuroMesh

**Brain-Inspired Multi-Agent Networks: Effects of Scale and Topology on Collective Information Processing**

CS 5130 (Applied Programming & Data Processing for AI), Fall 2026 · Team NeuroMesh:
Mihir Mathur, Yutong Dai, Armman Jhamb, Andrew Blaze Pitta

We study how agent population size (7 → 112) and network topology (ER, Watts–Strogatz,
Barabási–Albert, modular, Yeo-based SBM) affect information propagation, consensus,
cross-network integration, robustness and cost in networks of open-source LLM agents whose
roles follow the Yeo 7-network brain parcellation.

> **Status (Nov 19):** graphs, roles and node removal (Track A) and tasks, single-agent baseline and
> evaluation (Track C) are implemented and tested. The LLM client and round engine (Track B) and the
> experiment runner (Track D) are in progress; until they land, `neuromesh run` supports `--dry-run` only.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # smoke tests
ruff check .      # lint
neuromesh run configs/pilot.yaml --dry-run
```

## Usage

**Tasks and evaluation without an LLM.** Build a graph, make a task instance, run the rule-based
reference agents and score the run from its logs:

```python
from neuromesh.graphs import generate_graph, assign_roles
from neuromesh.tasks import get_task
from neuromesh.evaluation import compute_metrics, to_frames
from neuromesh.evaluation.reference import run_reference

graph = assign_roles(generate_graph("ws", 28, seed=0), seed=0)
task = get_task("aggregation")              # or "fact_propagation", "consensus"
instance = task.make_instance(graph, seed=0)
result = run_reference(graph, instance)     # rule-based agents, no LLM
runs, calls = to_frames([(result, instance)])
print(compute_metrics(runs, calls)[["topology", "n", "exact_accuracy", "total_tokens"]])
```

**Single-agent baseline** against any OpenAI-compatible endpoint, or Ollama with thinking disabled:

```bash
python -m neuromesh.baseline --base-url http://localhost:11434 --ollama --model qwen3:8b \
    --seeds 0 1 2 3 4 --n 14 --out results/baseline_n14
```

**Evaluation pilot** (750 reference-agent runs on all 25 graph conditions; metrics, 95% CIs,
covariate regression and figures in about 15 s):

```bash
python -m neuromesh.evaluation.pilot --out results/pilot_reference
```

Task definitions, agent state fields, metrics and the log format are in [docs/track_c.md](docs/track_c.md).

## Repository layout and owners

| Path | What | Owner |
| --- | --- | --- |
| `src/neuromesh/contracts.py` | Shared types and interfaces (frozen Nov 1) | Andrew (D), all four approve changes |
| `src/neuromesh/graphs/`, `data/` | Topologies, roles, node removal, structural metrics | Mihir (A) |
| `src/neuromesh/llm/`, `src/neuromesh/agents/` | LLM clients, prompts, message protocol, round engine | Armman (B) |
| `src/neuromesh/tasks/`, `evaluation/`, `baseline/` | Tasks, scoring, metrics, figures, single-agent baseline | Yutong (C) |
| `src/neuromesh/runner/`, `configs/`, `scripts/` | Config, experiment runner, storage, CLI, job scripts | Andrew (D) |
| `tests/<module>/` | Unit tests, owned with their module | each track |
| `tests/integration/` | End-to-end mock test | Mihir (A) |
| `tests/fixtures/` | Shared stubs: graphs (A), llm (B), transcripts (C), logs (D) | per subfolder |

Track docs: [A](docs/track_a.md) · [B](docs/track_b.md) · [C](docs/track_c.md) · [D](docs/track_d.md) ·
[LLM endpoint contract](docs/llm_endpoint.md)

## Key settings

- Mean degree 4 at every scale; N = 7 is a boundary case (topology comparisons from N = 14)
- 10 synchronous rounds, 3-round history, 5 seeds (3 for the core sweep)
- Outputs go to `results/` (git-ignored) as Parquet

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
