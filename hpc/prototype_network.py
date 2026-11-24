"""Standalone NeuroMesh proof of concept: neighbor-only consensus with 7/14 agents.

No graph/task/runner code from the rest of the repository is required.
Every agent has separate message history but shares one LocalLLM on the GPU.

Explorer example (from neuromesh/hpc):
    SCRIPT=prototype_network.py N_AGENTS=7 ROUNDS=1 sbatch run.sbatch
    MODEL=deepseek8 SCRIPT=prototype_network.py N_AGENTS=7 ROUNDS=1 sbatch run.sbatch

CPU-only code test (requires no torch, model, or GPU):
    python prototype_network.py --mock --agents 7 --rounds 2 --baseline
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROLES = [
    ("Visual", "Relay received information accurately."),
    ("Somatomotor", "Communicate your position clearly to nearby agents."),
    ("Dorsal Attention", "Notice repeated signals from your neighbors."),
    ("Ventral Attention", "Notice disagreement and new information."),
    ("Limbic", "Prefer stable, evidence-supported decisions."),
    ("Frontoparietal", "Integrate neighbor information into a decision."),
    ("Default Mode", "Remember useful information from past rounds."),
]
ROOT = Path(__file__).resolve().parent


def neighbors_on_ring(n):
    """A simple 4-regular graph for testing the pipeline; NOT an ER/WS/BA graph."""
    return {i: sorted({(i + d) % n for d in (-2, -1, 1, 2)}) for i in range(n)}


def extract_object(text, required_keys):
    """Find the last matching JSON object, allowing fences and reasoning prefixes."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    decoder = json.JSONDecoder()
    matches = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and required_keys.issubset(obj):
            matches.append(obj)
    return matches[-1] if matches else None


def parse_agent_output(text, allowed_neighbors):
    obj = extract_object(text, {"messages", "state"})
    if not obj or not isinstance(obj["messages"], dict) or not isinstance(obj["state"], dict):
        return None, {}, 0, 0, "Missing or malformed messages/state JSON"
    vote = obj["state"].get("vote")
    if type(vote) is not int or vote not in (0, 1):
        return None, {}, 0, 0, "state.vote must be integer 0 or 1"

    outgoing = {}
    dropped = truncated = 0
    for recipient, content in obj["messages"].items():
        try:
            neighbor_id = int(recipient)
        except (ValueError, TypeError):
            dropped += 1
            continue
        if neighbor_id not in allowed_neighbors or not isinstance(content, str):
            dropped += 1
            continue
        words = content.split()
        if len(words) > 80:
            content = " ".join(words[:80])
            truncated += 1
        outgoing[neighbor_id] = content
    return vote, outgoing, dropped, truncated, None


class Agent:
    """One network node. Only its instructions and recent turns are private."""

    def __init__(self, agent_id, role, duty, initial_vote):
        self.id = agent_id
        self.role = role
        self.vote = initial_vote
        self.instructions = (
            f"You are agent {agent_id}, assigned the {role} network. {duty} "
            "Task: reach a common binary vote (0 or 1) by talking ONLY to listed neighbors. "
            "You may retain or update your current vote based on neighbor messages. "
            "Return EXACTLY ONE JSON object with two keys: messages and state. "
            "messages must map neighbor ID strings to short message strings (80 words max each). "
            "state must contain an integer vote equal to 0 or 1. "
            "Send a concise vote message to every listed neighbor. "
            "No Markdown, no explanation, no extra text or reasoning."
        )
        self.turns = []  # (user_message, validated_assistant_JSON); at most 3 rounds.

    def messages_for_round(self, round_num, neighbor_ids, inbox):
        # Incoming messages are from the PREVIOUS completed round only.
        data = {
            "round": round_num,
            "agent_id": self.id,
            "current_vote": self.vote,
            "neighbors": neighbor_ids,
            "received_last_round": inbox,
        }
        user_text = (
            "Use the following data. Return only the required JSON response.\n"
            "CURRENT_DATA_JSON: " + json.dumps(data, separators=(",", ":"))
        )
        messages = [{"role": "system", "content": self.instructions}]
        for previous_user, previous_json in self.turns[-2:]:
            messages += [
                {"role": "user", "content": previous_user},
                {"role": "assistant", "content": previous_json},
            ]
        messages.append({"role": "user", "content": user_text})
        return messages, user_text

    def remember(self, user_text, outgoing, valid):
        # We retain validated state rather than lengthy DeepSeek reasoning.
        response = json.dumps({
            "messages": {str(i): value for i, value in outgoing.items()} if valid else {},
            "state": {"vote": self.vote},
        }, separators=(",", ":"))
        self.turns.append((user_text, response))
        self.turns = self.turns[-3:]


class MockLLM:
    """No-GPU fake backend for checking routing and logs only."""

    def generate_with_stats(self, messages, max_new_tokens=None):
        user_text = messages[-1]["content"]
        if "BASELINE_DATA_JSON: " in user_text:
            data = json.loads(user_text.split("BASELINE_DATA_JSON: ", 1)[1])
            votes = data["votes"]
            text = json.dumps({"vote": int(sum(votes) > len(votes) / 2)})
        else:
            data = json.loads(user_text.split("CURRENT_DATA_JSON: ", 1)[1])
            vote = data["current_vote"]
            text = json.dumps({
                "messages": {str(j): f"I vote {vote}." for j in data["neighbors"]},
                "state": {"vote": vote},
            })
        return {"text": text, "prompt_tokens": 0, "completion_tokens": 0,
                "latency_seconds": 0.0}


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    if args.agents < 7 or args.agents % 7:
        raise SystemExit("--agents must be a multiple of 7, starting with 7")
    if args.rounds < 1 or args.max_new_tokens < 1:
        raise SystemExit("--rounds and --max-new-tokens must be positive")

    model_name = os.getenv("MODEL", "qwen8")
    rng = random.Random(args.seed)
    # A close but non-unanimous starting split makes consensus nontrivial.
    initial_votes = ([1] * (args.agents // 2 + 1) +
                     [0] * (args.agents - (args.agents // 2 + 1)))
    rng.shuffle(initial_votes)
    graph = neighbors_on_ring(args.agents)
    agents = []
    for i, vote in enumerate(initial_votes):
        role, duty = ROLES[i % 7]
        agents.append(Agent(i, role, duty, vote))

    # Distinct per-job directory; ignored by Git. Keep logs even if SLURM times out.
    stamp = os.getenv("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results" / f"{model_name}-n{args.agents}-r{args.rounds}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    calls_path = output_dir / "calls.jsonl"
    rounds_path = output_dir / "rounds.jsonl"
    (output_dir / "config.json").write_text(json.dumps({
        "model": model_name, "agents": args.agents, "rounds": args.rounds,
        "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "topology": "ring-4 (prototype, not ER/WS/BA)",
        "mock": args.mock, "roles": [a.role for a in agents],
        "initial_votes": initial_votes,
        "neighbors": {str(i): v for i, v in graph.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Experiment: {model_name} | N={args.agents} | rounds={args.rounds} "
          f"| seed={args.seed} | mock={args.mock}", flush=True)
    print(f"Logs: {output_dir}", flush=True)

    if args.mock:
        llm = MockLLM()
    else:
        import torch
        from model_pipeline import LocalLLM
        torch.manual_seed(args.seed)
        llm = LocalLLM(model_name)

    incoming = {i: [] for i in range(args.agents)}
    total_prompt = total_completion = good_json = bad_json = 0
    total_inference = 0.0
    dropped_total = truncated_total = 0
    convergence_round = None
    started = time.perf_counter()

    if args.baseline:
        prompt = (
            "You are a single-agent consensus baseline. You see all initial votes. "
            "Pick the majority vote; on ties pick 0. "
            "Respond as JSON only: {\"vote\": 0} or {\"vote\": 1}.\n"
            "BASELINE_DATA_JSON: " + json.dumps({"votes": initial_votes})
        )
        response = llm.generate_with_stats([{"role": "user", "content": prompt}],
                                           max_new_tokens=args.max_new_tokens)
        parsed = extract_object(response["text"], {"vote"})
        baseline_vote = parsed.get("vote") if parsed else None
        valid = type(baseline_vote) is int and baseline_vote in (0, 1)
        majority = int(sum(initial_votes) > len(initial_votes) / 2)
        append_jsonl(calls_path, {
            "kind": "single_agent_baseline", "round": 0, "agent_id": None,
            **response, "parse_ok": valid,
            "vote": baseline_vote if valid else None,
        })
        total_prompt += response["prompt_tokens"]
        total_completion += response["completion_tokens"]
        total_inference += response["latency_seconds"]
        print(f"Baseline: vote={baseline_vote if valid else 'INVALID'} "
              f"| expected majority={majority}", flush=True)

    for round_num in range(1, args.rounds + 1):
        next_incoming = {i: [] for i in range(args.agents)}
        round_valid = 0
        for agent in agents:
            prompt_messages, user_text = agent.messages_for_round(
                round_num, graph[agent.id], incoming[agent.id]
            )
            response = llm.generate_with_stats(prompt_messages,
                                               max_new_tokens=args.max_new_tokens)
            new_vote, outgoing, dropped, truncated, error = parse_agent_output(
                response["text"], set(graph[agent.id])
            )
            parse_ok = error is None
            if parse_ok:
                agent.vote = new_vote
                good_json += 1
                round_valid += 1
                for recipient, message in outgoing.items():
                    next_incoming[recipient].append({"from": agent.id, "text": message})
            else:
                bad_json += 1
            agent.remember(user_text, outgoing, parse_ok)
            dropped_total += dropped
            truncated_total += truncated
            total_prompt += response["prompt_tokens"]
            total_completion += response["completion_tokens"]
            total_inference += response["latency_seconds"]
            append_jsonl(calls_path, {
                "kind": "network", "round": round_num,
                "agent_id": agent.id, "role": agent.role,
                "incoming_count": len(incoming[agent.id]),
                "vote": agent.vote, "parse_ok": parse_ok, "error": error,
                "outgoing_count": len(outgoing),
                "dropped_non_neighbor": dropped, "truncated_messages": truncated,
                **response,
            })

        # Synchronous delivery: no messages from round r are read until r+1.
        incoming = next_incoming
        votes = [agent.vote for agent in agents]
        tally = Counter(votes)
        agreement = max(tally.values()) / args.agents
        if agreement == 1 and convergence_round is None:
            convergence_round = round_num
        append_jsonl(rounds_path, {
            "round": round_num, "votes": votes, "agreement": agreement,
            "valid_json": round_valid,
            "sent_messages": sum(len(v) for v in next_incoming.values()),
        })
        print(f"Round {round_num}: agreement={agreement:.1%} "
              f"| valid JSON={round_valid}/{args.agents} "
              f"| messages={sum(len(v) for v in next_incoming.values())}", flush=True)

    summary = {
        "model": model_name, "N": args.agents, "rounds": args.rounds,
        "seed": args.seed, "mock": args.mock,
        "initial_votes": initial_votes,
        "final_votes": [a.vote for a in agents],
        "final_agreement": max(Counter(a.vote for a in agents).values()) / args.agents,
        "convergence_round": convergence_round,
        "network_calls": args.agents * args.rounds,
        "valid_json_calls": good_json, "invalid_json_calls": bad_json,
        "dropped_non_neighbor_messages": dropped_total,
        "truncated_messages": truncated_total,
        "prompt_tokens": total_prompt, "completion_tokens": total_completion,
        "inference_seconds": round(total_inference, 3),
        "experiment_seconds_excluding_model_load": round(time.perf_counter() - started, 3),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    # Leave invalid JSON visible to students, but do not fail the Slurm job:
    # performance and parse failure are part of the prototype's measurements.
    print("Prototype finished; see summary.json, calls.jsonl, rounds.jsonl.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=int(os.getenv("N_AGENTS", "7")))
    parser.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "1")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
    parser.add_argument("--max-new-tokens", type=int,
                        default=int(os.getenv("MAX_NEW_TOKENS", "600" if
                                             os.getenv("MODEL") == "deepseek8" else "240")))
    parser.add_argument("--baseline", action="store_true",
                        default=os.getenv("BASELINE", "0") == "1")
    parser.add_argument("--mock", action="store_true", help="CPU-only routing test")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
