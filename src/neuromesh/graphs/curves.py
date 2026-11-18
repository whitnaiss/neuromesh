"""Graph-only robustness curves: connectivity decay under node removal (Week 4).

What this measures, and what it does not
----------------------------------------
A robustness curve here answers one narrow structural question: as agents are
deleted, how much of the network stays mutually reachable? It is computed from
the adjacency alone.

It is a **reference curve, not a result about the agent system.** It is not
agent communication success, not LLM response fidelity, and not task accuracy.
A network whose largest component survives an attack intact may still fail
every task, and a fragmented network may still score well on a task that only
needs local information. The Week 6 experiments measure the latter; this module
measures the structural ceiling they operate under, so that a Week 6 drop in
task score can be checked against how much connectivity was actually lost.

Reuse, not reimplementation
---------------------------
Every removal order comes from ``robustness.py`` (Week 3):
:func:`~neuromesh.graphs.robustness.random_removal_order` and
:func:`~neuromesh.graphs.robustness.targeted_removal_order`. The adaptive
attack, the ``smallest_id`` tie-break, the isolated salted RNG and the
``floor`` rounding rule are all inherited unchanged. No removal logic is
duplicated here, and no new rounding rule is introduced.

The nesting property makes complete curves cheap
------------------------------------------------
Week 3 established that both removal orders NEST: the first ``k`` entries of a
longer order are exactly the shorter order. So a complete curve needs the order
computed ONCE at ``k_max`` and then evaluated on each prefix, rather than
re-running the attack at every checkpoint. That is not an approximation -- the
prefix of the adaptive attack IS the shorter adaptive attack -- and it is what
makes measuring every integer removal count from 0 to ``floor(0.30 N)``
affordable instead of only the three checkpoints.

Denominator convention: the ORIGINAL N
--------------------------------------
``largest_component_fraction`` and ``fraction_removed`` are both divided by the
**original** node count, never by the surviving count. This is deliberate and
it matters: a graph reduced to 2 nodes that are joined has a largest component
holding 100% of its *survivors* but 1.8% of the *original* network. Dividing by
the survivor count would make total collapse look like perfect robustness, and
would make curves at different N incomparable.

Efficiency is reported under both normalisations
------------------------------------------------
``global_efficiency`` is ``nx.global_efficiency`` of the surviving graph:
the mean of ``1/d`` over its ordered pairs, with ``1/d = 0`` for an unreachable
pair. That is the standard definition, it is well defined for a disconnected
graph, and it is normalised over the SURVIVING pairs.

``global_efficiency_original_n`` renormalises the same pair sum over the
original node count, so removed agents count as unreachable. The first answers
"how efficiently do the survivors communicate among themselves"; the second
answers "how much of the original network's communication capacity is left".
They diverge sharply under heavy removal, and quoting one as the other would
misstate the damage, so both are exported.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import networkx as nx

from .metrics import graph_id
from .robustness import (
    DEFAULT_ROUNDING_RULE,
    DEFAULT_TIE_BREAK,
    REMOVAL_STRATEGIES,
    random_removal_order,
    removal_count,
    remove_node_set,
    targeted_removal_order,
)
from .spec import GraphParameterError

__all__ = [
    "AUC_FIELDS",
    "CHECKPOINT_FRACTIONS",
    "MAX_REMOVAL_FRACTION",
    "ROBUSTNESS_FIELDS",
    "checkpoint_rows",
    "robustness_auc",
    "robustness_trajectory",
    "trajectory_rows",
]

#: Removal fractions the Week 6 robustness experiment will use, plus the
#: zero-removal baseline. Exported as checkpoints alongside the complete curve.
CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)

#: Curves are measured from 0 up to ``floor(MAX_REMOVAL_FRACTION * N)``
#: inclusive. 0.30 is the largest level the robustness experiment specifies;
#: nothing here extrapolates past it.
MAX_REMOVAL_FRACTION: float = 0.30

#: Column order for the trajectory table.
ROBUSTNESS_FIELDS: tuple[str, ...] = (
    "graph_id",
    "topology",
    "n",
    "seed",
    "strategy",
    "removal_seed",
    "tie_break",
    "k_removed",
    "fraction_removed",
    "n_remaining",
    "fraction_remaining",
    "n_edges_remaining",
    "largest_component_size",
    "largest_component_fraction",
    "n_components",
    "global_efficiency",
    "global_efficiency_original_n",
    "mean_degree_remaining",
    "is_checkpoint",
    "requested_fraction",
)

#: Column order for the AUC table.
AUC_FIELDS: tuple[str, ...] = (
    "graph_id",
    "topology",
    "n",
    "seed",
    "strategy",
    "removal_seed",
    "n_points",
    "x_min",
    "x_max",
    "auc_raw",
    "auc_normalized",
    "integration",
    "x_axis",
    "y_axis",
)


def _efficiency(graph: nx.Graph, n_original: int) -> tuple[float | None, float | None]:
    """``(efficiency over survivors, efficiency renormalised over original N)``."""
    n_surviving = graph.number_of_nodes()
    if n_surviving == 0:
        return None, 0.0
    if n_surviving == 1:
        return 0.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        efficiency = float(nx.global_efficiency(graph))
    # nx.global_efficiency divides the pair sum by n_s(n_s - 1); rescaling to
    # the original node count is therefore exact, not an approximation.
    if n_original < 2:
        return efficiency, None
    pair_sum = efficiency * n_surviving * (n_surviving - 1)
    return efficiency, pair_sum / (n_original * (n_original - 1))


def robustness_trajectory(
    graph: nx.Graph,
    strategy: str,
    *,
    seed: int | None = None,
    tie_break: str = DEFAULT_TIE_BREAK,
    max_fraction: float = MAX_REMOVAL_FRACTION,
    checkpoints: Sequence[float] = CHECKPOINT_FRACTIONS,
    rule: str = DEFAULT_ROUNDING_RULE,
    topology: str | None = None,
    n: int | None = None,
    graph_seed: int | None = None,
) -> dict:
    """One complete connectivity-decay curve for one graph and one strategy.

    Measures every integer removal count from 0 to ``floor(max_fraction * N)``
    inclusive, so the curve is complete rather than sampled at three points.
    The requested checkpoint fractions are flagged within the same curve rather
    than measured separately, which guarantees the checkpoint values and the
    curve cannot disagree.

    Parameters
    ----------
    graph:
        The original graph. **Not modified.**
    strategy:
        ``"random"`` or ``"targeted"`` (``contracts.RemovalMode``).
    seed:
        Required for ``"random"``. For ``"targeted"`` with the default
        deterministic tie-break it is unused but still recorded, so a
        trajectory row always says which seed produced it.
    max_fraction:
        Upper end of the curve. The default 0.30 is the robustness
        experiment's largest level; the curve never goes beyond measured
        points.

    Returns
    -------
    dict
        ``{"points": [...], "order": (...), "strategy": ..., ...}``. Each point
        is a dict whose keys are exactly :data:`ROBUSTNESS_FIELDS`.

    Raises
    ------
    GraphParameterError
        Unknown strategy, missing seed for random removal, or a
        ``max_fraction`` outside ``[0, 1]``.
    """
    if strategy not in REMOVAL_STRATEGIES:
        raise GraphParameterError(
            f"unknown removal strategy {strategy!r}; "
            f"expected one of {REMOVAL_STRATEGIES}"
        )
    if not isinstance(max_fraction, (int, float)) or isinstance(max_fraction, bool):
        raise GraphParameterError("max_fraction must be a number")
    if not 0.0 <= float(max_fraction) <= 1.0:
        raise GraphParameterError(
            f"max_fraction must lie in [0, 1], got {max_fraction}"
        )

    attrs = graph.graph
    topology = topology if topology is not None else attrs.get("topology")
    n_original = graph.number_of_nodes()
    n_label = n if n is not None else attrs.get("n", n_original)
    seed_label = graph_seed if graph_seed is not None else attrs.get("seed")
    gid = (
        graph_id(topology, n_label, seed_label)
        if None not in (topology, n_label, seed_label)
        else None
    )

    k_max = removal_count(n_original, float(max_fraction), rule)

    # The order is computed ONCE at k_max; prefixes are the shorter attacks.
    if strategy == "random":
        order = random_removal_order(graph, k_max, seed)
    else:
        order = targeted_removal_order(graph, k_max, seed=seed, tie_break=tie_break)

    # Checkpoint counts, by the SAME Week 3 rounding rule as the curve.
    checkpoint_k: dict[int, float] = {}
    for fraction in checkpoints:
        k = removal_count(n_original, float(fraction), rule)
        if k <= k_max:
            checkpoint_k.setdefault(k, float(fraction))

    points: list[dict] = []
    for k in range(k_max + 1):
        surviving = remove_node_set(graph, order[:k])
        n_surviving = surviving.number_of_nodes()
        if n_surviving:
            components = sorted(
                (len(c) for c in nx.connected_components(surviving)), reverse=True
            )
        else:
            components = []
        lcc = components[0] if components else 0
        efficiency, efficiency_original = _efficiency(surviving, n_original)

        points.append(
            {
                "graph_id": gid,
                "topology": topology,
                "n": int(n_label) if n_label is not None else None,
                "seed": int(seed_label) if seed_label is not None else None,
                "strategy": strategy,
                "removal_seed": seed,
                "tie_break": tie_break if strategy == "targeted" else "n/a",
                "k_removed": k,
                "fraction_removed": k / n_original if n_original else 0.0,
                "n_remaining": n_surviving,
                "fraction_remaining": (
                    n_surviving / n_original if n_original else 0.0
                ),
                "n_edges_remaining": surviving.number_of_edges(),
                "largest_component_size": lcc,
                # DENOMINATOR IS THE ORIGINAL N -- see the module docstring.
                "largest_component_fraction": (
                    lcc / n_original if n_original else 0.0
                ),
                "n_components": len(components),
                "global_efficiency": efficiency,
                "global_efficiency_original_n": efficiency_original,
                "mean_degree_remaining": (
                    2.0 * surviving.number_of_edges() / n_surviving
                    if n_surviving
                    else None
                ),
                "is_checkpoint": k in checkpoint_k,
                "requested_fraction": checkpoint_k.get(k),
            }
        )

    return {
        "graph_id": gid,
        "topology": topology,
        "n": int(n_label) if n_label is not None else None,
        "seed": int(seed_label) if seed_label is not None else None,
        "strategy": strategy,
        "removal_seed": seed,
        "tie_break": tie_break if strategy == "targeted" else "n/a",
        "rounding_rule": rule,
        "max_fraction": float(max_fraction),
        "k_max": k_max,
        "n_original": n_original,
        "n_edges_original": graph.number_of_edges(),
        "order": tuple(order),
        "points": points,
    }


def trajectory_rows(trajectories: Sequence[dict]) -> list[dict]:
    """Flatten trajectories into one list of table rows, order preserved."""
    rows: list[dict] = []
    for trajectory in trajectories:
        rows.extend(trajectory["points"])
    return rows


def checkpoint_rows(trajectories: Sequence[dict]) -> list[dict]:
    """Only the requested-checkpoint points, for the compact summary table."""
    return [
        point
        for trajectory in trajectories
        for point in trajectory["points"]
        if point["is_checkpoint"]
    ]


def robustness_auc(trajectory: dict, y_field: str = "largest_component_fraction") -> dict:
    """Descriptive graph-only AUC under a trajectory's decay curve.

    **This is a Week 4 structural reference, not the Week 6 robustness-
    experiment AUC.** The allocation reserves the principal AUC comparison for
    Week 6, where it will be computed over LLM task scores. Nothing here says
    anything about task performance, and the two must not be pooled or
    compared.

    Definitions, stated rather than implied
    ---------------------------------------
    x axis
        ``fraction_removed`` -- removed node count divided by the ORIGINAL N.
        Runs from 0.0 to ``k_max / N``, which is at most
        :data:`MAX_REMOVAL_FRACTION` and is usually slightly below it because
        node counts are integers (at N = 112, ``33/112 = 0.2946``).
    y axis
        ``largest_component_fraction`` by default -- largest component size
        over the ORIGINAL N. Any other numeric trajectory field may be named.
    integration
        Trapezoidal rule (``numpy``-free, explicit) over the measured points
        only. **No interpolation beyond the measured range and no
        extrapolation**, so ``x_max`` is reported and must be quoted with the
        number.
    normalisation
        ``auc_normalized = auc_raw / (x_max - x_min)``, i.e. the
        *mean* y over the removal range. This makes values comparable across
        different N, whose ``x_max`` differ because of integer rounding. The
        raw integral is also returned, but raw values at different N are NOT
        comparable.

        **1.0 IS NOT ATTAINABLE, and 1.0 is not the "no damage" reference.**
        Because y is normalised by the ORIGINAL N, deleting a node lowers y
        even when the graph does not fragment at all: a graph that stays fully
        connected throughout still has ``y = 1 - k/N`` at step ``k``. The mean
        of that straight line over ``[0, x_max]`` is

            ceiling = 1 - x_max / 2

        which is the HIGHEST value any curve can reach -- 0.857143 when
        ``x_max = 2/7`` (N = 7, 14, 28, 56 at the 30% level) and 0.852679 when
        ``x_max = 33/112`` (N = 112). Reaching 1.0 would require the largest
        component to still contain all ``N`` original nodes after ``k`` of them
        had been deleted, which is impossible.

        So a measured value should be read AGAINST THAT CEILING, not against
        1.0: at N = 112 an AUC of 0.8458 (BA under random failure) is within
        0.007 of the 0.852679 ceiling and therefore means "almost no
        fragmentation beyond the unavoidable loss of the removed nodes
        themselves", while 0.3993 (BA under targeted attack) means severe
        fragmentation.

        **0.0 IS LIKEWISE A THEORETICAL LOWER REFERENCE, NOT AN ATTAINABLE
        RESULT** under this experiment's documented conditions. Two of those
        conditions bound it away from zero:

        * the generators guarantee a CONNECTED input, so ``y(0) = 1.0``
          exactly and the first trapezoid alone contributes positive area;
        * removal is capped at ``floor(max_fraction * N)``, so at least
          ``N - k_max`` nodes always survive, and any surviving node is itself
          a component of size at least 1, giving ``y(k) >= 1/N > 0``.

        The absolute floor -- a graph collapsing to a single node immediately
        after the first removal and staying there -- is therefore about 0.3571
        at N = 7, 0.1875 at N = 14, 0.0960 at N = 28, 0.0485 at N = 56 and
        0.0239 at N = 112. Reaching 0.0 would require the largest component to
        be empty, i.e. every node removed, which ``max_fraction <= 0.30``
        forbids. The lowest value observed anywhere in the Week 4 sweep is
        0.3544 (BA, N = 112, seed 0, targeted).

    Returns
    -------
    dict
        Keys are exactly :data:`AUC_FIELDS`. ``auc_raw`` and
        ``auc_normalized`` are ``None`` when fewer than two points exist (a
        single point has no area) or when any y value is undefined.
    """
    points = trajectory["points"]
    xs = [p["fraction_removed"] for p in points]
    ys = [p[y_field] for p in points]

    base = {
        "graph_id": trajectory["graph_id"],
        "topology": trajectory["topology"],
        "n": trajectory["n"],
        "seed": trajectory["seed"],
        "strategy": trajectory["strategy"],
        "removal_seed": trajectory["removal_seed"],
        "n_points": len(points),
        "x_min": xs[0] if xs else None,
        "x_max": xs[-1] if xs else None,
        "integration": "trapezoidal over measured points; no extrapolation",
        "x_axis": "fraction_removed (k / original N)",
        "y_axis": f"{y_field} (denominator: original N)",
    }
    if len(points) < 2 or any(y is None for y in ys):
        return {**base, "auc_raw": None, "auc_normalized": None}

    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
    span = xs[-1] - xs[0]
    normalized = area / span if span > 0 else None
    return {
        **base,
        "auc_raw": float(area),
        "auc_normalized": _none_if_nan(normalized),
    }


def _none_if_nan(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
