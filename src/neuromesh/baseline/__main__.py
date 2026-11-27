"""Run the single-agent baseline against an OpenAI-compatible endpoint.

Example (Ollama on a laptop, thinking off via the native API)::

    python -m neuromesh.baseline --base-url http://localhost:11434 --ollama \\
        --model qwen3:8b --seeds 0 1 2 3 4 --out results/baseline

The instances are built exactly as for the network runs (same graph generator,
same task seed), so baseline and network scores are directly comparable.
Writes ``baseline.csv`` (one row per task x seed) and ``baseline_answers.jsonl``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from neuromesh.baseline import run_baseline
from neuromesh.baseline.http_client import OllamaNativeClient, OpenAICompatibleClient
from neuromesh.graphs import assign_roles, generate_graph
from neuromesh.tasks import TASK_NAMES, get_task


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m neuromesh.baseline", description=__doc__)
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:11434/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--tasks", nargs="+", default=list(TASK_NAMES), choices=TASK_NAMES)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n", type=int, default=14, help="graph size the instance is drawn on")
    p.add_argument("--topology", default="er")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3)")
    p.add_argument("--ollama", action="store_true",
                   help="use Ollama's native /api/chat with thinking disabled")
    p.add_argument("--out", default="results/baseline")
    args = p.parse_args(argv)

    client_cls = OllamaNativeClient if args.ollama else OpenAICompatibleClient
    llm = client_cls(args.base_url, args.model, api_key_env=args.api_key_env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = " /no_think" if args.no_think else ""

    rows = []
    with open(out / "baseline_answers.jsonl", "w") as answers:
        for task_name in args.tasks:
            task = get_task(task_name)
            for seed in args.seeds:
                graph = assign_roles(generate_graph(args.topology, args.n, seed=seed), seed=seed)
                instance = task.make_instance(graph, seed)
                res = run_baseline(instance, llm, args.max_tokens, args.temperature, suffix)
                row = {
                    "task": task_name, "seed": seed, "model": args.model,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "latency_s": round(res.latency_s, 3), **res.scores,
                }
                rows.append(row)
                answers.write(json.dumps({**row, "answer_text": res.answer_text}) + "\n")
                print(json.dumps(row), flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(out / "baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out / 'baseline.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
