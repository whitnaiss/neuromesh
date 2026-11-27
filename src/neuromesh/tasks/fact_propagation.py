"""Fact Propagation (broadcast / telephone game).

Only Visual agents (``INPUT_ROLE``) receive five fictional facts at round 0.
Every agent keeps the facts it has heard in ``state["facts"]`` as
``{"F1": "4821", ...}``. We measure how fast the facts reach the Frontoparietal
readout agents and how faithfully they survive each hop.

Facts are invented (fictional project names with random 4-digit codes), so the
model cannot answer from memory and scoring is exact string match on the code.
"""

from __future__ import annotations

import networkx as nx

from neuromesh.contracts import RunResult, TaskInstance
from neuromesh.tasks import _common as c

NAME = "fact_propagation"
N_FACTS = 5

_PREFIXES = ("Kes", "Vor", "Tal", "Mir", "Quen", "Bra", "Sol", "Dex", "Ul", "Ny")
_SUFFIXES = ("trel", "ani", "obar", "ix", "enna", "oth", "ura", "vik", "esto", "ari")
_KINDS = ("access code", "budget code", "launch code", "room number", "batch number")


def fact_ids(n: int = N_FACTS) -> list[str]:
    return [f"F{i + 1}" for i in range(n)]


class FactPropagationTask:
    name = NAME

    def __init__(self, n_facts: int = N_FACTS) -> None:
        self.n_facts = n_facts

    # ------------------------------------------------------------------ instance
    def make_instance(self, graph: nx.Graph, seed: int) -> TaskInstance:
        c.require_roles(graph)
        rng = c.rng_for(NAME, seed, graph)
        sources = c.input_nodes(graph)
        readout = c.readout_nodes(graph)

        codes = rng.sample(range(1000, 10000), self.n_facts)
        names: set[str] = set()
        facts: dict[str, dict[str, str]] = {}
        for fid, code in zip(fact_ids(self.n_facts), codes, strict=True):
            name = ""
            while not name or name in names:
                name = f"{rng.choice(_PREFIXES)}{rng.choice(_SUFFIXES)}-{rng.randint(2, 99)}"
            names.add(name)
            kind = _KINDS[len(facts) % len(_KINDS)]
            facts[fid] = {
                "text": f"The {kind} of Project {name} is {code}.",
                "value": str(code),
            }

        brief = "You received these facts:\n" + "\n".join(
            f"{fid}: {f['text']}" for fid, f in facts.items()
        )
        private_info = {int(v): (brief if v in sources else "") for v in graph.nodes}
        initial_state = {
            int(v): {"facts": ({k: f["value"] for k, f in facts.items()} if v in sources else {})}
            for v in graph.nodes
        }
        hops = c.hop_distance_from(graph, sources)
        return TaskInstance(
            task=NAME,
            seed=seed,
            private_info=private_info,
            initial_state=initial_state,
            readout_nodes=readout,
            ground_truth={
                "facts": facts,
                "input_nodes": sources,
                "hop_distance": {str(k): v for k, v in hops.items()},
            },
        )

    # ------------------------------------------------------------------ scoring
    def node_fidelity(self, instance: TaskInstance, state: dict) -> tuple[float, int, int]:
        """(fraction correct, n reported, n reported wrong) for one agent's state."""
        truth = {k: f["value"] for k, f in instance.ground_truth["facts"].items()}
        reported = state.get("facts") if isinstance(state, dict) else None
        if not isinstance(reported, dict):
            return 0.0, 0, 0
        correct = wrong = n_rep = 0
        for fid, true_value in truth.items():
            if fid not in reported or reported[fid] in (None, ""):
                continue
            n_rep += 1
            got = "".join(ch for ch in str(reported[fid]) if ch.isdigit())
            if got == true_value:
                correct += 1
            else:
                wrong += 1
        return correct / len(truth), n_rep, wrong

    def score(self, instance: TaskInstance, result: RunResult) -> dict[str, float]:
        final = c.final_states(result)
        alive = sorted(final) if final else []
        readout = [v for v in instance.readout_nodes if v in final] or list(instance.readout_nodes)
        hops = {int(k): v for k, v in instance.ground_truth["hop_distance"].items()}

        fid = {v: self.node_fidelity(instance, final.get(v, {})) for v in set(alive) | set(readout)}
        readout_fid = [fid[v][0] for v in readout]
        all_fid = [fid[v][0] for v in alive]
        n_rep = sum(fid[v][1] for v in alive)
        n_wrong = sum(fid[v][2] for v in alive)

        out: dict[str, float] = {
            "readout_fidelity": _mean(readout_fid),
            "network_fidelity": _mean(all_fid),
            "readout_complete": _mean([1.0 if f == 1.0 else 0.0 for f in readout_fid]),
            "distortion_rate": (n_wrong / n_rep) if n_rep else 0.0,
        }

        # propagation delay: first round where mean readout fidelity >= 0.5
        by_round = c.states_by_round(result)
        rounds = sorted(by_round)
        out["n_rounds"] = float(len(rounds))
        t50 = None
        known: dict[int, dict] = {}
        for i, r in enumerate(rounds, start=1):
            known.update(by_round[r])
            level = _mean([self.node_fidelity(instance, known.get(v, {}))[0] for v in readout])
            if t50 is None and level >= 0.5:
                t50 = i
        out["t50_reached"] = 1.0 if t50 is not None else 0.0
        out["propagation_delay_t50"] = float(t50 if t50 is not None else len(rounds) + 1)

        # fidelity versus hop distance from the nearest input agent
        per_hop: dict[int, list[float]] = {}
        for v in alive:
            d = hops.get(v, c.UNREACHABLE)
            if d >= 0:
                per_hop.setdefault(d, []).append(fid[v][0])
        for d, vals in sorted(per_hop.items()):
            out[f"fidelity_hop_{d}"] = _mean(vals)
        out["fidelity_hop_slope"] = _slope(
            [(hops[v], fid[v][0]) for v in alive if hops.get(v, -1) >= 0]
        )
        return out

    # ------------------------------------------------------------------ baseline
    def baseline_prompt(self, instance: TaskInstance) -> str:
        facts = instance.ground_truth["facts"]
        lines = "\n".join(f"{fid}: {f['text']}" for fid, f in facts.items())
        example = ", ".join(f'"{fid}": "<code>"' for fid in facts)
        return (
            "You are the only agent in this system. You have received these facts:\n"
            f"{lines}\n\n"
            "Report the code number of every fact exactly. "
            f'Answer with JSON only, in this form: {{"facts": {{{example}}}}}'
        )

    def score_baseline(self, instance: TaskInstance, answer_text: str) -> dict[str, float]:
        parsed = c.parse_json_object(answer_text)
        if parsed is None:
            return {"parse_ok": 0.0, "fidelity": 0.0, "distortion_rate": 0.0}
        f, n_rep, wrong = self.node_fidelity(instance, parsed)
        return {
            "parse_ok": 1.0,
            "fidelity": f,
            "distortion_rate": (wrong / n_rep) if n_rep else 0.0,
        }


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _slope(points: list[tuple[int, float]]) -> float:
    """Least-squares slope of fidelity on hop distance (0.0 if undefined)."""
    if len(points) < 2:
        return 0.0
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx == 0:
        return 0.0
    return float(sum((p[0] - mx) * (p[1] - my) for p in points) / sxx)
