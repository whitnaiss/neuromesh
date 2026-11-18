"""Command-line entry points for graph generation and validation.

Usage
-----

    python -m neuromesh.graphs.cli fixtures          # write the fixture set
    python -m neuromesh.graphs.cli validate          # print validation tables
    python -m neuromesh.graphs.cli verify-fixtures   # reload and re-check files
    python -m neuromesh.graphs.cli diameters         # N=112 diameters, 10 vs 12 rounds
    python -m neuromesh.graphs.cli week4-analysis    # Week 4 tables + figures

``validate`` prints two tables: the 15 canonical rows (the three topologies
at the five population sizes, using the default generator settings
that produce the shipped fixtures), and a diagnostics table for the
non-default variants (ER under G(n,p) and standard BA) so their known
deviations stay visible rather than hidden behind the defaults.

``diameters`` measures shortest-path structure for every (topology, seed)
condition at one population size and reports which conditions fit inside a
given round budget. Exit code 1 means an *execution* problem -- a generated
graph that breached the connectivity contract, or a fixture whose diameter
disagrees with the regenerated graph. A diameter that exceeds the round budget
is a **finding, not an error**, and does not change the exit code: the point of
the command is to report it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .analysis import (
    DEFAULT_ROUND_BUDGETS,
    STD_CONVENTION,
    aggregate_path_metrics,
    diameter_table,
    measure_graph,
    render_aggregate_table,
    render_measurement_table,
    round_budget_assessment,
)
from .curves import AUC_FIELDS, MAX_REMOVAL_FRACTION, ROBUSTNESS_FIELDS
from .datasets import DEFAULT_OUTPUT_DIR as WEEK4_OUTPUT_DIR
from .fixtures import (
    DEFAULT_FIXTURE_DIR,
    fixture_filename,
    generate_fixture_set,
    load_fixture,
    load_fixture_set,
)
from .generators import (
    barabasi_albert,
    erdos_renyi,
    generate_graph,
    modular,
    watts_strogatz,
    yeo_sbm,
)
from .spec import (
    DEFAULT_FIXTURE_SEED,
    EXPERIMENT_SEEDS,
    POPULATION_SIZES,
    TARGET_MEAN_DEGREE,
    TOPOLOGIES,
    GraphSpec,
)
from .validation import check_reproducible, validate_graph
from .yeo import (
    YeoMatrixUnavailableError,
    derive_yeo7_matrix,
    load_yeo_matrix,
    yeo_matrix_path,
)

# Reasons a mean-degree miss is a model constraint rather than a defect.
_BA_STANDARD_REASON = (
    "standard Barabasi-Albert yields exactly E = m(N-m), so mean degree is "
    "4 - 8/N for m=2 and cannot reach 4 at small N"
)
_ER_GNP_REASON = (
    "G(n,p) matches mean degree only in expectation; per-seed edge count is "
    "Binomial and deviates at small N"
)

#: Rows that correspond to the shipped fixtures.
_CANONICAL = (
    ("ER (G(n,m))", "er", {}, None),
    ("WS (p=0.1)", "ws", {}, None),
    ("BA (degree-matched)", "ba", {}, None),
    ("Modular (7 blocks)", "modular", {}, None),
    ("Yeo-SBM (empirical)", "yeo_sbm", {}, None),
)

#: Rows kept for auditability; these are NOT the shipped fixtures.
_DIAGNOSTIC = (
    ("ER (G(n,p))", "er", {"model": "gnp", "ensure_connected": False}, _ER_GNP_REASON),
    ("BA (standard)", "ba", {"variant": "standard"}, _BA_STANDARD_REASON),
)

_GENERATORS = {
    "er": erdos_renyi,
    "ws": watts_strogatz,
    "ba": barabasi_albert,
    "modular": modular,
    "yeo_sbm": yeo_sbm,
}


def _row(
    label: str,
    topology: str,
    n: int,
    seed: int,
    params: dict,
    exception_reason: str | None,
) -> tuple:
    spec = GraphSpec(topology=topology, n=n, seed=seed, params=params)
    graph, metadata = generate_graph(spec)
    result = validate_graph(
        graph,
        n=n,
        topology=topology,
        seed=seed,
        target_mean_degree=TARGET_MEAN_DEGREE,
        require_connected=params.get("ensure_connected", True),
        degree_exception_reason=exception_reason,
    )
    repro = check_reproducible(
        _GENERATORS[topology], n, seed, target_mean_degree=TARGET_MEAN_DEGREE, **params
    )
    return (
        label,
        str(n),
        str(result.n_nodes),
        str(result.n_edges),
        f"{result.mean_degree:.4f}",
        "yes" if result.connected else "no",
        "yes" if repro.status == "PASS" else "NO",
        result.status_of("mean_degree_tolerance"),
        result,
        repro,
        metadata,
    )


def _render(rows: list, header: list) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row[: len(header)]):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                cell.ljust(widths[i]) for i, cell in enumerate(row[: len(header)])
            )
            + " |"
        )
    return "\n".join(lines)


_HEADER = [
    "Topology",
    "N",
    "Nodes",
    "Edges",
    "Mean Degree",
    "Connected",
    "Reproducible",
    "Degree Tolerance",
]


def cmd_validate(args: argparse.Namespace) -> int:
    seed = args.seed
    overall_ok = True

    for title, table in (
        ("Canonical generators (these produce the shipped fixtures)", _CANONICAL),
        ("Diagnostics: non-default variants (NOT shipped as fixtures)", _DIAGNOSTIC),
    ):
        rows = []
        for label, topology, params, reason in table:
            for n in POPULATION_SIZES:
                row = _row(label, topology, n, seed, params, reason)
                rows.append(row)
                if row[8].failures:
                    overall_ok = False
        print(f"\n### {title}  (seed={seed})\n")
        print(_render(rows, _HEADER))

        exceptions = [r for r in rows if r[7] == "EXCEPTION"]
        failures = [r for r in rows if r[8].failures]
        if exceptions:
            print(
                "\nDeclared model-constraint exceptions "
                "(no defect, but NOT eligible for the matched-degree sweep):"
            )
            for r in exceptions:
                detail = next(
                    c.detail for c in r[8].checks if c.name == "mean_degree_tolerance"
                )
                print(f"  - {r[0]} N={r[1]}: {detail}")

        eligible = [r for r in rows if r[8].sweep_eligible]
        print(
            f"\nSweep-eligible rows: {len(eligible)}/{len(rows)} "
            "(clean AND matched on mean degree)."
        )
        if failures:
            print("\nFAILURES:")
            for r in failures:
                for check in r[8].failures:
                    print(f"  - {r[0]} N={r[1]} [{check.name}] {check.detail}")

    print("\nMean degree is 2E/N. Tolerance is +/-10% of target "
          f"{TARGET_MEAN_DEGREE}. N=7 is a declared boundary case.")
    return 0 if overall_ok else 1


def cmd_fixtures(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    seeds = tuple(args.seeds)
    manifest = generate_fixture_set(out_dir=out_dir, seeds=seeds)
    print(f"Wrote {manifest['count']} fixtures + manifest.json to {out_dir}")
    for entry in manifest["fixtures"]:
        flag = "" if entry["degree_matched_exactly"] else "  [degree not exact]"
        print(
            f"  {entry['file']:<24} E={entry['n_edges']:<4} "
            f"k={entry['mean_degree']:.4f} connected={entry['connected']} "
            f"draws={entry['attempts']}{flag}"
        )
    return 0


def cmd_verify_fixtures(args: argparse.Namespace) -> int:
    fixture_dir = Path(args.out)
    loaded = load_fixture_set(fixture_dir)
    print(f"Reloaded {len(loaded)} fixtures from {fixture_dir}\n")
    failures = 0
    for (topology, n, seed), (graph, metadata) in sorted(loaded.items()):
        result = validate_graph(
            graph,
            n=n,
            topology=topology,
            seed=seed,
            target_mean_degree=metadata.get("target_mean_degree", TARGET_MEAN_DEGREE),
        )
        status = "OK" if result.ok else "FAIL"
        if not result.ok:
            failures += 1
        print(
            f"  [{status}] {topology} N={n:<4} seed={seed} "
            f"nodes={result.n_nodes} edges={result.n_edges} "
            f"k={result.mean_degree:.4f} connected={result.connected}"
        )
        for check in result.failures:
            print(f"        {check.name}: {check.detail}")
    print(f"\n{len(loaded) - failures}/{len(loaded)} fixtures reloaded and revalidated.")
    return 0 if failures == 0 else 1


def cmd_diameters(args: argparse.Namespace) -> int:
    """Measure diameters for every (topology, seed) at one N; assess the budget.

    Deterministic, offline and fixture-free: graphs come from the public
    ``generate_graph`` at its default settings, so the rows describe exactly
    the graphs the experiment will run. ``--check-fixtures`` additionally
    reloads the committed seed-0 fixtures and asserts their diameters agree
    with the regenerated graphs, which catches a generator change that a
    fixture file no longer matches.
    """
    n = args.n
    seeds = tuple(args.seeds)
    budgets = tuple(args.rounds)
    failures: list[str] = []

    rows = diameter_table(n, seeds=seeds, round_budgets=budgets)

    print(f"\n### Shortest-path structure at N = {n}  (seeds {list(seeds)})\n")
    print(render_measurement_table(rows, budgets))
    print(f"\n{len(rows)} configurations measured "
          f"({len(TOPOLOGIES)} topologies x {len(seeds)} seeds).")

    # Every generated graph must satisfy the connectivity contract. A breach is
    # an execution failure, reported rather than concealed.
    for row in rows:
        if not row["connected"]:
            failures.append(
                f"{row['topology']} N={row['n']} seed={row['seed']} is "
                f"disconnected ({row['n_components']} components): the "
                "generator connectivity contract is breached"
            )

    aggregates = aggregate_path_metrics(rows, budgets)
    print(f"\n### Per-topology aggregate  (std dev: {STD_CONVENTION})\n")
    print(render_aggregate_table(aggregates, budgets))

    assessment = round_budget_assessment(rows, budgets)
    print("\n### Round budget\n")
    for budget in budgets:
        entry = assessment["budgets"][budget]
        verdict = "covers all measured" if entry["covers_all_measured"] else "INSUFFICIENT"
        print(
            f"  {budget:>3} rounds: {entry['n_within']}/{entry['n_conditions']} "
            f"conditions have diameter <= {budget}  [{verdict}]"
        )
        for violation in entry["violations"]:
            print(
                f"            exceeded by {violation['topology']} "
                f"seed={violation['seed']} (diameter {violation['diameter']})"
            )
    print(
        f"\n  Largest measured diameter: {assessment['max_diameter']} at "
        + ", ".join(
            f"{a['topology']} seed={a['seed']}" for a in assessment["argmax"]
        )
    )
    print(
        "  Smallest budget covering every measured condition: "
        f"{assessment['minimum_budget_covering_all_measured']} rounds."
    )
    print("\n  " + assessment["interpretation"])

    if args.check_fixtures:
        print("\n### Fixture cross-check (committed seed-0 graphs)\n")
        for topology in TOPOLOGIES:
            path = Path(args.fixture_dir) / fixture_filename(topology, n, 0)
            if not path.is_file():
                print(f"  [SKIP] {topology}: no fixture at {path}")
                continue
            graph, _ = load_fixture(path)
            measured = measure_graph(
                graph,
                topology=topology,
                n=n,
                seed=0,
                source="fixture",
                round_budgets=budgets,
            )
            generated = next(
                r for r in rows if r["topology"] == topology and r["seed"] == 0
            )
            agree = (
                measured["diameter"] == generated["diameter"]
                and measured["n_edges"] == generated["n_edges"]
            )
            status = "OK" if agree else "MISMATCH"
            print(
                f"  [{status}] {topology}: fixture diameter "
                f"{measured['diameter']} / regenerated {generated['diameter']}, "
                f"edges {measured['n_edges']} / {generated['n_edges']}"
            )
            if not agree:
                failures.append(
                    f"{topology} N={n} seed=0: committed fixture disagrees with "
                    "the regenerated graph"
                )

    if args.json:
        payload = {
            "n": n,
            "seeds": list(seeds),
            "round_budgets": list(budgets),
            "std_convention": STD_CONVENTION,
            "rows": [
                {k: v for k, v in row.items() if k != "round_check"}
                | {"round_check": {str(b): row["round_check"][b] for b in budgets}}
                for row in rows
            ],
            "aggregates": aggregates,
            "assessment": assessment,
            "networkx_version": nx_version(),
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {out}")

    if failures:
        print("\nEXECUTION FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


def nx_version() -> str:
    import networkx

    return networkx.__version__


#: Week 4 analysis stages, in dependency order. ``conditions`` is folded into
#: ``structural`` because it is an aggregation of it, and ``figures`` needs the
#: structural and robustness tables, so asking for figures implies both.
_WEEK4_STAGES = ("structural", "positional", "robustness", "figures")


def cmd_week4(args: argparse.Namespace) -> int:
    """Run the Week 4 analysis and write its tables (and optionally figures).

    Deterministic and offline. Every graph is regenerated from its seed and
    ``yeo_sbm`` reads the committed empirical matrix from the package data;
    nothing is downloaded and no LLM is involved in any number produced here.
    """
    from .datasets import (
        CONDITION_FIELDS,
        aggregate_conditions,
        build_positional_table,
        build_robustness_tables,
        build_structural_table,
        expected_node_row_count,
        write_csv,
        write_json,
    )
    from .metrics import POSITIONAL_FIELDS, STRUCTURAL_FIELDS

    stages = set(args.only) if args.only else set(_WEEK4_STAGES)
    if "figures" in stages:
        stages |= {"structural", "robustness"}

    out_dir = Path(args.out)
    topologies = tuple(args.topologies)
    sizes = tuple(args.sizes)
    seeds = tuple(args.seeds)
    expected_graphs = len(topologies) * len(sizes) * len(seeds)

    print(
        f"\n### Track A Week 4 analysis\n\n"
        f"  topologies : {list(topologies)}\n"
        f"  sizes      : {list(sizes)}\n"
        f"  seeds      : {list(seeds)}\n"
        f"  instances  : {expected_graphs}\n"
        f"  stages     : {sorted(stages)}\n"
        f"  output     : {out_dir}\n"
    )

    failures: list[str] = []
    timings: dict[str, float] = {}
    written: list[Path] = []
    analysis: dict = {}

    if "structural" in stages:
        started = time.perf_counter()
        structural = build_structural_table(topologies, sizes, seeds)
        conditions = aggregate_conditions(structural)
        timings["structural"] = time.perf_counter() - started
        analysis["structural"] = structural

        disconnected = [r["graph_id"] for r in structural if r["n_components"] != 1]
        unbalanced = [r["graph_id"] for r in structural if not r["roles_balanced"]]
        mismatched = [
            r["graph_id"]
            for r in structural
            if r["n_edges"] != 2 * r["n_nodes"] or r["mean_degree"] != 4.0
        ]
        if len(structural) != expected_graphs:
            failures.append(
                f"structural table has {len(structural)} rows, expected "
                f"{expected_graphs}"
            )
        for label, bad in (
            ("disconnected original graph(s)", disconnected),
            ("unbalanced role assignment(s)", unbalanced),
            ("mean degree not exactly 4 / E != 2N", mismatched),
        ):
            if bad:
                failures.append(f"{label}: {bad[:5]}")

        if args.write:
            written.append(
                write_csv(
                    out_dir / "structural_metrics.csv", structural, STRUCTURAL_FIELDS
                )
            )
            written.append(
                write_csv(
                    out_dir / "condition_summaries.csv", conditions, CONDITION_FIELDS
                )
            )
        print(
            f"  [structural] {len(structural)} instance rows, "
            f"{len(conditions)} condition rows  ({timings['structural']:.2f}s)"
        )

    if "positional" in stages:
        started = time.perf_counter()
        positional = build_positional_table(topologies, sizes, seeds)
        timings["positional"] = time.perf_counter() - started
        analysis["positional"] = positional

        expected_nodes = expected_node_row_count(topologies, sizes, seeds)
        if len(positional) != expected_nodes:
            failures.append(
                f"positional table has {len(positional)} rows, expected "
                f"{expected_nodes}"
            )
        keys = {(r["graph_id"], r["node_id"]) for r in positional}
        if len(keys) != len(positional):
            failures.append("positional table has duplicate (graph_id, node_id) keys")

        if args.write:
            written.append(
                write_csv(
                    out_dir / "positional_covariates.csv",
                    positional,
                    POSITIONAL_FIELDS,
                )
            )
        print(
            f"  [positional] {len(positional)} node rows "
            f"(expected {expected_nodes})  ({timings['positional']:.2f}s)"
        )

    if "robustness" in stages:
        started = time.perf_counter()
        robustness, auc, trajectories = build_robustness_tables(
            topologies, sizes, seeds, max_fraction=args.max_fraction
        )
        timings["robustness"] = time.perf_counter() - started
        analysis["robustness"] = robustness
        checkpoints = [p for p in robustness if p["is_checkpoint"]]

        expected_trajectories = expected_graphs * 2
        if len(trajectories) != expected_trajectories:
            failures.append(
                f"{len(trajectories)} trajectories, expected "
                f"{expected_trajectories} (2 strategies per instance)"
            )
        for trajectory in trajectories:
            first = trajectory["points"][0]
            if first["k_removed"] != 0:
                failures.append(f"{trajectory['graph_id']}: curve does not start at k=0")
            sizes_seen = [p["largest_component_size"] for p in trajectory["points"]]
            pairs = zip(sizes_seen, sizes_seen[1:], strict=False)
            if any(later > earlier for earlier, later in pairs):
                failures.append(
                    f"{trajectory['graph_id']} ({trajectory['strategy']}): largest "
                    "component grew as nodes were removed"
                )

        if args.write:
            written.append(
                write_csv(
                    out_dir / "robustness_curves.csv", robustness, ROBUSTNESS_FIELDS
                )
            )
            written.append(
                write_csv(
                    out_dir / "robustness_checkpoints.csv",
                    checkpoints,
                    ROBUSTNESS_FIELDS,
                )
            )
            written.append(write_csv(out_dir / "robustness_auc.csv", auc, AUC_FIELDS))
        print(
            f"  [robustness] {len(trajectories)} trajectories, "
            f"{len(robustness)} curve points, {len(checkpoints)} checkpoints, "
            f"{len(auc)} AUC rows  ({timings['robustness']:.2f}s)"
        )

        print("\n  Mean normalised AUC of the largest-component-fraction curve")
        print("  (1.0 = no loss; graph-only, NOT a task-performance measure):\n")
        header = ["Topology", *[f"N={n}" for n in sizes]]
        rows_out = []
        for strategy in ("random", "targeted"):
            for topology in topologies:
                cells = [f"{topology} / {strategy}"]
                for n in sizes:
                    values = [
                        r["auc_normalized"]
                        for r in auc
                        if r["topology"] == topology
                        and r["n"] == n
                        and r["strategy"] == strategy
                        and r["auc_normalized"] is not None
                    ]
                    cells.append(
                        f"{sum(values) / len(values):.4f}" if values else "n/a"
                    )
                rows_out.append(cells)
        print(_render(rows_out, ["Topology / strategy", *header[1:]]))

    if "figures" in stages:
        started = time.perf_counter()
        if args.write:
            from .figures import write_figures

            written.extend(write_figures(out_dir, analysis))
        timings["figures"] = time.perf_counter() - started
        print(f"\n  [figures] 3 SVG + 3 data CSV  ({timings['figures']:.2f}s)")

    if args.write:
        manifest = {
            "generated_by": "python -m neuromesh.graphs.cli week4-analysis",
            "track": "A",
            "week": 4,
            "grid": {
                "topologies": list(topologies),
                "sizes": [int(s) for s in sizes],
                "seeds": [int(s) for s in seeds],
            },
            "stages": sorted(stages),
            "max_removal_fraction": float(args.max_fraction),
            # Wall-clock timings are deliberately NOT written here: they vary
            # run to run, which would make the manifest the one output that is
            # not byte-reproducible. They are printed to stdout instead.
            "files": sorted(p.name for p in written),
            "networkx_version": nx_version(),
            "disclaimer": (
                "Graph-only structural measurements. NOT agent communication "
                "success, LLM response fidelity, or task accuracy."
            ),
        }
        written.append(write_json(out_dir / "manifest.json", manifest))
        print(f"\n  Wrote {len(written)} file(s) to {out_dir}")
        for path in sorted(written, key=lambda p: p.name):
            print(f"    {path.name}")
    else:
        print("\n  --no-write: nothing written to disk")

    total = sum(timings.values())
    print(f"\n  Total analysis time: {total:.2f}s")

    if failures:
        print("\nVALIDATION FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll structural invariants held.")
    return 0


def cmd_yeo_derive(args: argparse.Namespace) -> int:
    """Aggregate a parcel-level connectome into the committed 7x7 matrix."""
    source_sha256 = {}
    for path in (Path(args.matrix), Path(args.labels)):
        if path.is_file():
            source_sha256[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    record = derive_yeo7_matrix(
        args.matrix,
        args.labels,
        source_urls=list(args.source_url or []),
        source_sha256=source_sha256,
        notes=args.notes or "",
    )
    out = Path(args.out) if args.out else yeo_matrix_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    networks = record["networks"]
    matrix = record["matrix"]
    provenance = record["provenance"]
    print(f"Wrote {out}  (schema_version {record['schema_version']})")
    print(f"  parcellation : {provenance['parcellation']} ({provenance['modality']})")
    print(f"  parcels      : {provenance['n_parcels']} -> {provenance['parcels_per_network']}")
    print(f"  aggregation  : {provenance['aggregation']}")
    print(f"  within/between mean ratio : {provenance['within_between_ratio']:.3f}")
    print("\n  7x7 block means (rows and columns in YEO7 order):")
    print("    " + " ".join(f"{name[:8]:>9}" for name in networks))
    for i, name in enumerate(networks):
        print(f"  {name[:11]:11} " + " ".join(f"{v:9.4f}" for v in matrix[i]))
    return 0


def cmd_yeo_show(args: argparse.Namespace) -> int:
    """Print and revalidate the committed 7x7 matrix."""
    try:
        matrix, record = load_yeo_matrix(args.path)
    except YeoMatrixUnavailableError as exc:
        print(f"Yeo matrix unavailable: {exc}")
        return 1
    provenance = record["provenance"]
    print(f"Loaded and validated (schema_version {record['schema_version']})")
    print(f"  source      : {provenance.get('source')}")
    print(f"  parcellation: {provenance.get('parcellation')} ({provenance.get('modality')})")
    print(f"  licence     : {provenance.get('licence')}")
    print(f"  retrieved   : {provenance.get('retrieved_utc')}")
    print("\n  " + " ".join(f"{name[:8]:>9}" for name in record["networks"]))
    for i, name in enumerate(record["networks"]):
        print(f"  {name[:11]:11} " + " ".join(f"{v:9.4f}" for v in matrix[i]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neuromesh.graphs.cli",
        description="Graph generation, validation and fixture utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="print the graph validation tables")
    p_val.add_argument("--seed", type=int, default=DEFAULT_FIXTURE_SEED)
    p_val.set_defaults(func=cmd_validate)

    p_fix = sub.add_parser("fixtures", help="generate the fixture set")
    p_fix.add_argument("--out", default=str(DEFAULT_FIXTURE_DIR))
    p_fix.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[DEFAULT_FIXTURE_SEED],
        help="seeds to generate (default: 0; the full study uses 0 1 2 3 4)",
    )
    p_fix.set_defaults(func=cmd_fixtures)

    p_ver = sub.add_parser("verify-fixtures", help="reload and revalidate fixtures")
    p_ver.add_argument("--out", default=str(DEFAULT_FIXTURE_DIR))
    p_ver.set_defaults(func=cmd_verify_fixtures)

    p_dia = sub.add_parser(
        "diameters",
        help="measure diameters at one N and assess the communication-round budget",
    )
    p_dia.add_argument(
        "--n",
        type=int,
        default=112,
        help="population size to measure (default: 112, the largest scale)",
    )
    p_dia.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(EXPERIMENT_SEEDS),
        help="seeds to measure (default: the study's 0 1 2 3 4)",
    )
    p_dia.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROUND_BUDGETS),
        help="round budgets to check (default: 10 12)",
    )
    p_dia.add_argument(
        "--check-fixtures",
        action="store_true",
        help="also verify the committed seed-0 fixtures give the same diameters",
    )
    p_dia.add_argument("--fixture-dir", default=str(DEFAULT_FIXTURE_DIR))
    p_dia.add_argument(
        "--json", default=None, help="also write the full results to this JSON file"
    )
    p_dia.set_defaults(func=cmd_diameters)

    # Week 4: one primary command plus three focused aliases. All four share
    # cmd_week4; the aliases only preset --only, so there is one code path and
    # the stages cannot drift apart.
    week4_specs = (
        ("week4-analysis", None, "run the whole Week 4 analysis: tables + figures"),
        (
            "structural-analysis",
            ["structural"],
            "per-graph structural metrics + condition summaries",
        ),
        ("positional-covariates", ["positional"], "node-level positional covariates"),
        ("robustness-curves", ["robustness"], "graph-only robustness curves + AUC"),
    )
    for name, preset, help_text in week4_specs:
        parser_w4 = sub.add_parser(name, help=help_text)
        parser_w4.add_argument(
            "--out",
            default=str(WEEK4_OUTPUT_DIR),
            help=f"output directory (default: {WEEK4_OUTPUT_DIR})",
        )
        parser_w4.add_argument(
            "--topologies", nargs="+", default=list(TOPOLOGIES), choices=list(TOPOLOGIES)
        )
        parser_w4.add_argument(
            "--sizes", type=int, nargs="+", default=list(POPULATION_SIZES)
        )
        parser_w4.add_argument(
            "--seeds", type=int, nargs="+", default=list(EXPERIMENT_SEEDS)
        )
        parser_w4.add_argument(
            "--max-fraction",
            type=float,
            default=MAX_REMOVAL_FRACTION,
            dest="max_fraction",
            help=f"largest removal fraction measured (default: {MAX_REMOVAL_FRACTION})",
        )
        parser_w4.add_argument(
            "--no-write",
            action="store_false",
            dest="write",
            help="compute and validate without writing any file",
        )
        if preset is None:
            parser_w4.add_argument(
                "--only",
                nargs="+",
                default=None,
                choices=list(_WEEK4_STAGES),
                help="run only these stages (default: all)",
            )
        else:
            parser_w4.set_defaults(only=preset)
        parser_w4.set_defaults(func=cmd_week4)

    p_der = sub.add_parser(
        "yeo-derive",
        help="derive the 7x7 Yeo structural matrix from a parcel-level connectome",
    )
    p_der.add_argument("--matrix", required=True, help="parcel-level matrix CSV")
    p_der.add_argument("--labels", required=True, help="parcel label CSV")
    p_der.add_argument("--out", default=None, help="output JSON (default: committed path)")
    p_der.add_argument(
        "--source-url", action="append", default=[], help="source URL (repeatable)"
    )
    p_der.add_argument("--notes", default="", help="free-text provenance note")
    p_der.set_defaults(func=cmd_yeo_derive)

    p_show = sub.add_parser("yeo-show", help="print and revalidate the 7x7 Yeo matrix")
    p_show.add_argument("--path", default=None)
    p_show.set_defaults(func=cmd_yeo_show)

    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
