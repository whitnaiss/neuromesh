"""Command-line entry points for graph generation and validation.

Usage
-----

    python -m neuromesh.graphs.cli fixtures          # write the fixture set
    python -m neuromesh.graphs.cli validate          # print validation tables
    python -m neuromesh.graphs.cli verify-fixtures   # reload and re-check files

``validate`` prints two tables: the 15 canonical rows (the three topologies
at the five population sizes, using the default generator settings
that produce the shipped fixtures), and a diagnostics table for the
non-default variants (ER under G(n,p) and standard BA) so their known
deviations stay visible rather than hidden behind the defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fixtures import (
    DEFAULT_FIXTURE_DIR,
    generate_fixture_set,
    load_fixture_set,
)
from .generators import barabasi_albert, erdos_renyi, generate_graph, watts_strogatz
from .spec import (
    DEFAULT_FIXTURE_SEED,
    POPULATION_SIZES,
    TARGET_MEAN_DEGREE,
    GraphSpec,
)
from .validation import check_reproducible, validate_graph

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
)

#: Rows kept for auditability; these are NOT the shipped fixtures.
_DIAGNOSTIC = (
    ("ER (G(n,p))", "er", {"model": "gnp", "ensure_connected": False}, _ER_GNP_REASON),
    ("BA (standard)", "ba", {"variant": "standard"}, _BA_STANDARD_REASON),
)

_GENERATORS = {"er": erdos_renyi, "ws": watts_strogatz, "ba": barabasi_albert}


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

    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
