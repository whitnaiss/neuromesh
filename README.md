# NeuroMesh

**Brain-Inspired Multi-Agent Networks: Effects of Scale and Topology on Collective Information Processing**

CS 5130 (Applied Programming & Data Processing for AI), Fall 2026 · Team NeuroMesh:
Mihir Mathur, Yutong Dai, Armman Jhamb, Andrew Blaze Pitta

We study how agent population size (7 → 112) and network topology (ER, Watts–Strogatz,
Barabási–Albert, modular, Yeo-based SBM) affect information propagation, consensus,
cross-network integration, robustness and cost in networks of open-source LLM agents whose
roles follow the Yeo 7-network brain parcellation.

> Status: project skeleton. Entry points raise `NotImplementedError` until each track implements them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # smoke tests
ruff check .      # lint
neuromesh run configs/pilot.yaml --dry-run
```

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
