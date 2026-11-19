"""Consensus (as in AgentsNet).

Every agent starts with a private binary vote (0 or 1) and must end with the
whole network agreeing on one value. Agents report ``state["vote"]``. Both
values are always present at round 0, so agreement has to be produced by
communication rather than inherited from the draw.
"""

from __future__ import annotations

import networkx as nx

from neuromesh.contracts import RunResult, TaskInstance
from neuromesh.tasks import _common as c

NAME = "consensus"


class ConsensusTask:
    name = NAME

    def make_instance(self, graph: nx.Graph, seed: int) -> TaskInstance:
        c.require_roles(graph)
        rng = c.rng_for(NAME, seed, graph)
        nodes = sorted(int(v) for v in graph.nodes)
        votes = {v: rng.randint(0, 1) for v in nodes}
        if len(nodes) >= 2 and len(set(votes.values())) == 1:  # never start unanimous
            flip = rng.choice(nodes)
            votes[flip] = 1 - votes[flip]
        ones = sum(votes.values())
        return TaskInstance(
            task=NAME,
            seed=seed,
            private_info={v: f"Your initial vote is {votes[v]}." for v in nodes},
            initial_state={v: {"vote": votes[v]} for v in nodes},
            readout_nodes=c.readout_nodes(graph),
            ground_truth={
                "initial_votes": {str(v): b for v, b in votes.items()},
                "initial_majority_share": max(ones, len(nodes) - ones) / max(len(nodes), 1),
                "initial_majority": int(ones * 2 > len(nodes)) if ones * 2 != len(nodes) else -1,
            },
        )

    @staticmethod
    def vote_of(state: dict) -> int | None:
        v = c.as_int(state.get("vote")) if isinstance(state, dict) else None
        return v if v in (0, 1) else None

    def _agreement(self, states: dict[int, dict], nodes: list[int]) -> tuple[float, float, bool]:
        """(agreement ratio, valid-vote rate, unanimous) over ``nodes``.

        An agent with a missing or invalid vote counts as not agreeing.
        """
        if not nodes:
            return 0.0, 0.0, False
        votes = [self.vote_of(states.get(v, {})) for v in nodes]
        valid = [b for b in votes if b is not None]
        majority = max(valid.count(0), valid.count(1)) if valid else 0
        unanimous = len(valid) == len(nodes) and len(set(valid)) == 1
        return majority / len(nodes), len(valid) / len(nodes), unanimous

    def score(self, instance: TaskInstance, result: RunResult) -> dict[str, float]:
        final = c.final_states(result)
        nodes = sorted(final) if final else sorted(int(k) for k in instance.initial_state)
        ratio, valid_rate, unanimous = self._agreement(final, nodes)

        by_round = c.states_by_round(result)
        rounds = sorted(by_round)
        known: dict[int, dict] = {}
        converged = None
        for i, r in enumerate(rounds, start=1):
            known.update(by_round[r])
            if converged is None and self._agreement(known, nodes)[2]:
                converged = i
        return {
            "agreement_ratio": ratio,
            "consensus_reached": 1.0 if unanimous else 0.0,
            "valid_vote_rate": valid_rate,
            "initial_majority_share": float(instance.ground_truth["initial_majority_share"]),
            "convergence_round": float(converged if converged is not None else len(rounds) + 1),
            "n_rounds": float(len(rounds)),
        }

    def baseline_prompt(self, instance: TaskInstance) -> str:
        votes = instance.ground_truth["initial_votes"]
        listing = ", ".join(f"agent {k}: {v}" for k, v in votes.items())
        return (
            "You coordinate a group of agents. Each agent proposed a binary value:\n"
            f"{listing}\n\n"
            "Choose the single value (0 or 1) the whole group should agree on. "
            'Answer with JSON only: {"vote": 0} or {"vote": 1}'
        )

    def score_baseline(self, instance: TaskInstance, answer_text: str) -> dict[str, float]:
        parsed = c.parse_json_object(answer_text)
        vote = self.vote_of(parsed) if parsed else None
        majority = instance.ground_truth.get("initial_majority", -1)
        return {
            "parse_ok": 1.0 if parsed is not None else 0.0,
            # one agent always "agrees with itself": success means a valid vote
            "agreement_ratio": 1.0 if vote is not None else 0.0,
            "picked_initial_majority": (
                1.0 if (vote is not None and majority in (0, 1) and vote == majority) else 0.0
            ),
        }
