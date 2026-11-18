"""Week 4 figures, written as dependency-free SVG.

Why SVG written by hand rather than matplotlib
----------------------------------------------
``pyproject.toml`` declares ``networkx numpy pandas pyarrow pydantic pyyaml``
and the dev extra ``pytest pytest-cov ruff``. **matplotlib is not among them**,
and ``pyproject.toml`` is Track D's file under ``.github/CODEOWNERS`` (the
default ``*`` owner). matplotlib happens to be installed on the development
machine, which makes this a trap: a figure script importing it would work
locally and fail in CI, where the install is exactly ``pip install -e ".[dev]"``.

Adding the dependency would mean editing a teammate's file. Rather than do
that, or ship a plotting step that only runs on one laptop, the figures are
emitted as SVG from the standard library alone. That has real advantages
beyond avoiding the dependency question:

* SVG is text, so a figure diffs in review and cannot silently change;
* it is byte-reproducible, so the determinism tests cover the figures too;
* it needs no display, no backend selection and no font cache in CI.

The cost is that this is a small, deliberately limited plotting layer: axes,
polylines, markers, error bars and a legend. That is all these three figures
need.

Replotting elsewhere
--------------------
Every figure also writes its exact plotted values next to it as
``figure_data_*.csv``. Anyone preferring matplotlib, R or a spreadsheet can
reproduce the figures from those files without re-running the analysis, and
without this module's choices being in the way.

Honesty rules applied to every figure
-------------------------------------
* Straight segments connect **measured points only**. No smoothing, no spline,
  no interpolation across a gap, and nothing drawn past the last measured x.
* Error bars are +/- one **sample** standard deviation (``ddof=1``) across the
  five seeds, and the seed count is printed on the figure.
* A metric that is undefined for a condition is omitted from that series
  rather than drawn as zero.
* Axis ranges start at zero where the quantity is a fraction, so bar and line
  heights are proportional and not visually exaggerated by a clipped axis.
* No figure carries any LLM, task-accuracy or agent-performance claim. Each
  one states on its face that it is graph-only.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .datasets import write_csv
from .spec import POPULATION_SIZES, TOPOLOGIES, GraphParameterError

__all__ = [
    "PALETTE",
    "STRATEGY_DASH",
    "svg_document",
    "write_figures",
]

#: One colour per topology, held fixed across every figure so a reader learns
#: the mapping once. Chosen to stay distinguishable in greyscale print (the
#: progress report is a PDF) by also varying lightness, not only hue.
PALETTE: dict[str, str] = {
    "er": "#1f4e9c",
    "ws": "#b8860b",
    "ba": "#1b7a3d",
    "modular": "#a33b8f",
    "yeo_sbm": "#c0392b",
}

#: Removal strategy is encoded by dash pattern, so colour stays free for
#: topology and the two dimensions never compete.
STRATEGY_DASH: dict[str, str] = {"random": "none", "targeted": "6 3"}

_FONT = (
    "font-family=\"DejaVu Sans, Segoe UI, Helvetica, Arial, sans-serif\""
)


def _fmt(value: float) -> str:
    """Coordinate with 2 decimals, so the SVG is stable and not noisy."""
    return f"{value:.2f}"


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values spanning ``[low, high]``."""
    if high <= low:
        return [low]
    raw = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if step >= raw:
            break
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 1e-9:
        if value >= low - step * 1e-9:
            ticks.append(round(value, 10))
        value += step
    return ticks or [low, high]


class _Axes:
    """A single plot panel with a linear x and y axis."""

    def __init__(
        self,
        x0: float,
        y0: float,
        width: float,
        height: float,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        title: str,
        xlabel: str,
        ylabel: str,
        *,
        x_ticks: Sequence[float] | None = None,
        x_tick_labels: Sequence[str] | None = None,
    ):
        self.x0, self.y0 = x0, y0
        self.width, self.height = width, height
        self.xlim = xlim if xlim[1] > xlim[0] else (xlim[0], xlim[0] + 1.0)
        self.ylim = ylim if ylim[1] > ylim[0] else (ylim[0], ylim[0] + 1.0)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_ticks = list(x_ticks) if x_ticks is not None else None
        self.x_tick_labels = list(x_tick_labels) if x_tick_labels is not None else None
        self.parts: list[str] = []

    def px(self, x: float) -> float:
        lo, hi = self.xlim
        return self.x0 + (x - lo) / (hi - lo) * self.width

    def py(self, y: float) -> float:
        lo, hi = self.ylim
        return self.y0 + self.height - (y - lo) / (hi - lo) * self.height

    def frame(self) -> list[str]:
        out = [
            f'<rect x="{_fmt(self.x0)}" y="{_fmt(self.y0)}" '
            f'width="{_fmt(self.width)}" height="{_fmt(self.height)}" '
            'fill="#ffffff" stroke="#333333" stroke-width="1"/>'
        ]
        y_ticks = _nice_ticks(*self.ylim)
        for value in y_ticks:
            y = self.py(value)
            out.append(
                f'<line x1="{_fmt(self.x0)}" y1="{_fmt(y)}" '
                f'x2="{_fmt(self.x0 + self.width)}" y2="{_fmt(y)}" '
                'stroke="#e4e4e4" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{_fmt(self.x0 - 6)}" y="{_fmt(y + 3.5)}" {_FONT} '
                f'font-size="10" fill="#333333" text-anchor="end">'
                f"{_tick_text(value)}</text>"
            )
        ticks = self.x_ticks if self.x_ticks is not None else _nice_ticks(*self.xlim)
        labels = (
            self.x_tick_labels
            if self.x_tick_labels is not None
            else [_tick_text(t) for t in ticks]
        )
        for value, label in zip(ticks, labels, strict=True):
            x = self.px(value)
            out.append(
                f'<line x1="{_fmt(x)}" y1="{_fmt(self.y0 + self.height)}" '
                f'x2="{_fmt(x)}" y2="{_fmt(self.y0 + self.height + 4)}" '
                'stroke="#333333" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{_fmt(x)}" y="{_fmt(self.y0 + self.height + 16)}" '
                f'{_FONT} font-size="10" fill="#333333" text-anchor="middle">'
                f"{label}</text>"
            )
        out.append(
            f'<text x="{_fmt(self.x0 + self.width / 2)}" y="{_fmt(self.y0 - 8)}" '
            f'{_FONT} font-size="12" font-weight="600" fill="#111111" '
            f'text-anchor="middle">{_esc(self.title)}</text>'
        )
        out.append(
            f'<text x="{_fmt(self.x0 + self.width / 2)}" '
            f'y="{_fmt(self.y0 + self.height + 32)}" {_FONT} font-size="11" '
            f'fill="#333333" text-anchor="middle">{_esc(self.xlabel)}</text>'
        )
        cx, cy = self.x0 - 40, self.y0 + self.height / 2
        out.append(
            f'<text x="{_fmt(cx)}" y="{_fmt(cy)}" {_FONT} font-size="11" '
            f'fill="#333333" text-anchor="middle" '
            f'transform="rotate(-90 {_fmt(cx)} {_fmt(cy)})">'
            f"{_esc(self.ylabel)}</text>"
        )
        return out

    def series(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        colour: str,
        *,
        dash: str = "none",
        errors: Sequence[float | None] | None = None,
        markers: bool = True,
        width: float = 1.8,
    ) -> None:
        """Draw one series. Only the points given are drawn or connected."""
        if not xs:
            return
        points = " ".join(
            f"{_fmt(self.px(x))},{_fmt(self.py(y))}"
            for x, y in zip(xs, ys, strict=True)
        )
        dash_attr = "" if dash == "none" else f' stroke-dasharray="{dash}"'
        self.parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}"{dash_attr} stroke-linejoin="round"/>'
        )
        if errors:
            for x, y, err in zip(xs, ys, errors, strict=True):
                if err is None or err <= 0:
                    continue
                lo = self.py(max(y - err, self.ylim[0]))
                hi = self.py(min(y + err, self.ylim[1]))
                px = self.px(x)
                self.parts.append(
                    f'<line x1="{_fmt(px)}" y1="{_fmt(lo)}" x2="{_fmt(px)}" '
                    f'y2="{_fmt(hi)}" stroke="{colour}" stroke-width="1.2" '
                    'opacity="0.85"/>'
                )
                for cap in (lo, hi):
                    self.parts.append(
                        f'<line x1="{_fmt(px - 3)}" y1="{_fmt(cap)}" '
                        f'x2="{_fmt(px + 3)}" y2="{_fmt(cap)}" '
                        f'stroke="{colour}" stroke-width="1.2" opacity="0.85"/>'
                    )
        if markers:
            for x, y in zip(xs, ys, strict=True):
                self.parts.append(
                    f'<circle cx="{_fmt(self.px(x))}" cy="{_fmt(self.py(y))}" '
                    f'r="2.4" fill="{colour}" stroke="#ffffff" '
                    'stroke-width="0.6"/>'
                )

    def render(self) -> str:
        return "\n".join(self.frame() + self.parts)


def _tick_text(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.4g}"
    return text


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_document(
    width: float, height: float, body: Sequence[str], title: str, subtitle: str
) -> str:
    """Wrap panels in a titled SVG document."""
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width)}" '
        f'height="{_fmt(height)}" viewBox="0 0 {_fmt(width)} {_fmt(height)}">',
        f'<rect width="{_fmt(width)}" height="{_fmt(height)}" fill="#ffffff"/>',
        f'<text x="24" y="28" {_FONT} font-size="16" font-weight="700" '
        f'fill="#111111">{_esc(title)}</text>',
        f'<text x="24" y="48" {_FONT} font-size="11" fill="#555555">'
        f"{_esc(subtitle)}</text>",
    ]
    return "\n".join(head + list(body) + ["</svg>"]) + "\n"


def _legend(
    x: float, y: float, entries: Sequence[tuple[str, str, str]]
) -> list[str]:
    """``entries`` are ``(label, colour, dash)``, laid out in one column."""
    out = []
    for index, (label, colour, dash) in enumerate(entries):
        row = y + index * 16
        dash_attr = "" if dash == "none" else f' stroke-dasharray="{dash}"'
        out.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(row)}" x2="{_fmt(x + 22)}" '
            f'y2="{_fmt(row)}" stroke="{colour}" stroke-width="2.2"'
            f"{dash_attr}/>"
        )
        out.append(
            f'<circle cx="{_fmt(x + 11)}" cy="{_fmt(row)}" r="2.4" '
            f'fill="{colour}"/>'
        )
        out.append(
            f'<text x="{_fmt(x + 29)}" y="{_fmt(row + 3.5)}" {_FONT} '
            f'font-size="10" fill="#333333">{_esc(label)}</text>'
        )
    return out


def _stats(values: Sequence[float]) -> tuple[float | None, float | None]:
    defined = [v for v in values if v is not None and math.isfinite(v)]
    if not defined:
        return None, None
    mean = statistics.fmean(defined)
    std = statistics.stdev(defined) if len(defined) > 1 else None
    return mean, std


# --------------------------------------------------------------------------
# Figure 1 — structural metrics by topology and N
# --------------------------------------------------------------------------

_FIGURE1_PANELS = (
    ("average_clustering", "Average clustering coefficient", "C (0-1)"),
    ("average_shortest_path_length", "Average shortest-path length", "hops"),
    ("global_efficiency", "Global efficiency", "E (0-1)"),
    ("modularity_role_partition", "Modularity, 7-network partition", "Q"),
)


def figure_structural_metrics(structural: Sequence[dict]) -> tuple[str, list[dict]]:
    """Four panels: a structural metric against N, one line per topology."""
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in structural:
        for metric, _, _ in _FIGURE1_PANELS:
            grouped.setdefault((metric, row["topology"], row["n"]), []).append(
                row.get(metric)
            )

    data_rows: list[dict] = []
    panel_w, panel_h = 300.0, 190.0
    gap_x, gap_y = 92.0, 86.0
    left, top = 72.0, 86.0
    body: list[str] = []
    sizes = [int(s) for s in POPULATION_SIZES]
    # Log2 x positions: 7, 14, 28, 56, 112 double each step, so a linear axis
    # would crush the four smaller scales into the left margin.
    x_positions = [math.log2(s) for s in sizes]

    for index, (metric, title, ylabel) in enumerate(_FIGURE1_PANELS):
        col, rowi = index % 2, index // 2
        x0 = left + col * (panel_w + gap_x)
        y0 = top + rowi * (panel_h + gap_y)

        values: dict[str, list[tuple[float, float, float | None]]] = {}
        highest = 0.0
        lowest = 0.0
        for topology in TOPOLOGIES:
            series: list[tuple[float, float, float | None]] = []
            for n, xpos in zip(sizes, x_positions, strict=True):
                mean, std = _stats(grouped.get((metric, topology, n), []))
                if mean is None:
                    continue
                series.append((xpos, mean, std))
                highest = max(highest, mean + (std or 0.0))
                lowest = min(lowest, mean - (std or 0.0))
                data_rows.append(
                    {
                        "figure": "figure1_structural_metrics",
                        "panel": metric,
                        "topology": topology,
                        "n": n,
                        "mean": mean,
                        "std_ddof1": std,
                        "n_seeds": len(grouped.get((metric, topology, n), [])),
                    }
                )
            values[topology] = series

        upper = highest * 1.12 if highest > 0 else 1.0
        lower = min(0.0, lowest)
        axes = _Axes(
            x0,
            y0,
            panel_w,
            panel_h,
            (x_positions[0] - 0.25, x_positions[-1] + 0.25),
            (lower, upper),
            title,
            "population size N (log2 spacing)",
            ylabel,
            x_ticks=x_positions,
            x_tick_labels=[str(s) for s in sizes],
        )
        for topology in TOPOLOGIES:
            series = values[topology]
            if not series:
                continue
            axes.series(
                [p[0] for p in series],
                [p[1] for p in series],
                PALETTE[topology],
                errors=[p[2] for p in series],
            )
        body.append(axes.render())

    body.extend(
        _legend(
            left + 2 * panel_w + gap_x + 18,
            top + 10,
            [(t, PALETTE[t], "none") for t in TOPOLOGIES],
        )
    )
    body.append(
        f'<text x="{_fmt(left + 2 * panel_w + gap_x + 18)}" '
        f'y="{_fmt(top + 10 + len(TOPOLOGIES) * 16 + 14)}" {_FONT} '
        'font-size="9" fill="#666666">error bars: +/-1 SD</text>'
    )
    body.append(
        f'<text x="{_fmt(left + 2 * panel_w + gap_x + 18)}" '
        f'y="{_fmt(top + 10 + len(TOPOLOGIES) * 16 + 26)}" {_FONT} '
        'font-size="9" fill="#666666">(sample, ddof=1, n=5 seeds)</text>'
    )

    width = left + 2 * panel_w + gap_x + 180
    height = top + 2 * panel_h + gap_y + 56
    svg = svg_document(
        width,
        height,
        body,
        "Figure 1 - Structural metrics by topology and population size",
        "Graph-only measurements. 5 topologies x 5 sizes x 5 seeds = 125 graphs, "
        "all matched at mean degree 4 (exactly 2N edges) and all connected. "
        "Points are seed means; error bars are +/-1 sample SD (ddof=1, n=5). "
        "Not an LLM or task-performance result.",
    )
    return svg, data_rows


# --------------------------------------------------------------------------
# Figure 2 — random failure vs targeted attack
# --------------------------------------------------------------------------

def figure_robustness_curves(
    robustness: Sequence[dict], n: int | None = None, seed: int | None = None
) -> tuple[str, list[dict]]:
    """LCC fraction against fraction removed, both strategies, one N and seed.

    ``n`` and ``seed`` default to the LARGEST N and the SMALLEST seed actually
    present in ``robustness``, rather than to a hard-coded 112/0. A fixed
    default would silently emit an empty figure whenever the analysis was run
    on a grid that excluded that cell -- a figure that looks finished and shows
    nothing is worse than an error, so the selection adapts and the chosen
    cell is printed in the figure title.
    """
    if not robustness:
        raise GraphParameterError("no robustness rows to plot")
    if n is None:
        n = max(row["n"] for row in robustness)
    available_seeds = [row["seed"] for row in robustness if row["n"] == n]
    if not available_seeds:
        raise GraphParameterError(f"no robustness rows at N = {n}")
    if seed is None:
        seed = min(available_seeds)

    series: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in robustness:
        if row["n"] != n or row["seed"] != seed:
            continue
        series.setdefault((row["topology"], row["strategy"]), []).append(
            (row["fraction_removed"], row["largest_component_fraction"])
        )
    if not series:
        raise GraphParameterError(
            f"no robustness rows at N = {n}, seed = {seed}"
        )

    data_rows: list[dict] = []
    panel_w, panel_h = 380.0, 300.0
    left, top = 78.0, 92.0
    gap_x = 88.0
    body: list[str] = []
    max_x = max(
        (p[0] for points in series.values() for p in points), default=0.3
    )

    for index, strategy in enumerate(("random", "targeted")):
        x0 = left + index * (panel_w + gap_x)
        axes = _Axes(
            x0,
            top,
            panel_w,
            panel_h,
            (0.0, max_x),
            (0.0, 1.0),
            f"{strategy.capitalize()} removal",
            "fraction of original nodes removed",
            "largest component / original N",
            x_ticks=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            x_tick_labels=["0", ".05", ".10", ".15", ".20", ".25", ".30"],
        )
        for topology in TOPOLOGIES:
            points = sorted(series.get((topology, strategy), []))
            if not points:
                continue
            axes.series(
                [p[0] for p in points],
                [p[1] for p in points],
                PALETTE[topology],
                dash=STRATEGY_DASH[strategy],
                markers=False,
                width=2.0,
            )
            for x, y in points:
                data_rows.append(
                    {
                        "figure": "figure2_robustness_curves",
                        "panel": strategy,
                        "topology": topology,
                        "n": n,
                        "seed": seed,
                        "fraction_removed": x,
                        "largest_component_fraction": y,
                    }
                )
        body.append(axes.render())

    body.extend(
        _legend(
            left + 2 * panel_w + gap_x + 14,
            top + 12,
            [(t, PALETTE[t], "none") for t in TOPOLOGIES],
        )
    )
    width = left + 2 * panel_w + gap_x + 160
    height = top + panel_h + 64
    svg = svg_document(
        width,
        height,
        body,
        "Figure 2 - Random failure vs targeted attack "
        f"(N = {n}, seed {seed})",
        "Graph-only connectivity decay. Every integer removal count from 0 to "
        "floor(0.30N) is measured; segments join measured points only, with no "
        "interpolation or extrapolation. Denominator is the ORIGINAL N "
        "throughout. Targeted removal is adaptive highest-degree-first "
        "(degrees recomputed after each removal). This is structural "
        "connectivity, NOT agent communication success or task accuracy.",
    )
    return svg, data_rows


# --------------------------------------------------------------------------
# Figure 3 — robustness by topology, with across-seed uncertainty
# --------------------------------------------------------------------------

def figure_robustness_by_topology(
    robustness: Sequence[dict], sizes: Sequence[int] | None = None
) -> tuple[str, list[dict]]:
    """Mean LCC fraction with +/-1 SD bands across the seeds, per size.

    ``sizes`` defaults to the three LARGEST population sizes actually present,
    for the same reason :func:`figure_robustness_curves` adapts its cell: a
    hard-coded (28, 56, 112) would emit empty panels on a reduced grid.
    """
    if not robustness:
        raise GraphParameterError("no robustness rows to plot")
    if sizes is None:
        present = sorted({row["n"] for row in robustness})
        sizes = present[-3:]
    grouped: dict[tuple[int, str, str, int], list[float]] = {}
    for row in robustness:
        if row["n"] not in sizes:
            continue
        grouped.setdefault(
            (row["n"], row["topology"], row["strategy"], row["k_removed"]), []
        ).append(row["largest_component_fraction"])

    data_rows: list[dict] = []
    panel_w, panel_h = 300.0, 230.0
    left, top = 78.0, 96.0
    gap_x = 84.0
    body: list[str] = []

    for index, n in enumerate(sizes):
        x0 = left + index * (panel_w + gap_x)
        ks = sorted({k for (nn, _, _, k) in grouped if nn == n})
        max_x = (max(ks) / n) if ks else 0.3
        axes = _Axes(
            x0,
            top,
            panel_w,
            panel_h,
            (0.0, max_x),
            (0.0, 1.0),
            f"N = {n}",
            "fraction of original nodes removed",
            "largest component / original N",
        )
        for topology in TOPOLOGIES:
            for strategy in ("random", "targeted"):
                xs, ys, errs = [], [], []
                for k in ks:
                    values = grouped.get((n, topology, strategy, k), [])
                    mean, std = _stats(values)
                    if mean is None:
                        continue
                    xs.append(k / n)
                    ys.append(mean)
                    errs.append(std)
                    data_rows.append(
                        {
                            "figure": "figure3_robustness_by_topology",
                            "panel": f"N={n}",
                            "topology": topology,
                            "strategy": strategy,
                            "n": n,
                            "k_removed": k,
                            "fraction_removed": k / n,
                            "lcc_fraction_mean": mean,
                            "lcc_fraction_std_ddof1": std,
                            "n_seeds": len(values),
                        }
                    )
                if not xs:
                    continue
                axes.series(
                    xs,
                    ys,
                    PALETTE[topology],
                    dash=STRATEGY_DASH[strategy],
                    errors=errs,
                    markers=False,
                    width=1.7,
                )
        body.append(axes.render())

    legend_x = left + len(sizes) * panel_w + (len(sizes) - 1) * gap_x + 14
    body.extend(
        _legend(
            legend_x, top + 12, [(t, PALETTE[t], "none") for t in TOPOLOGIES]
        )
    )
    body.extend(
        _legend(
            legend_x,
            top + 12 + len(TOPOLOGIES) * 16 + 14,
            [
                ("random (solid)", "#444444", "none"),
                ("targeted (dashed)", "#444444", "6 3"),
            ],
        )
    )
    body.append(
        f'<text x="{_fmt(legend_x)}" '
        f'y="{_fmt(top + 12 + len(TOPOLOGIES) * 16 + 62)}" {_FONT} '
        'font-size="9" fill="#666666">bands: +/-1 SD (ddof=1, n=5)</text>'
    )

    width = legend_x + 170
    height = top + panel_h + 64
    svg = svg_document(
        width,
        height,
        body,
        "Figure 3 - Structural robustness by topology, with across-seed spread",
        "Mean largest-component fraction over 5 seeds, +/-1 sample SD "
        "(ddof=1). Solid = random failure, dashed = adaptive targeted attack. "
        "Graph-only structural connectivity; no LLM, no task scores.",
    )
    return svg, data_rows


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def write_figures(out_dir: Path, analysis: dict[str, Any]) -> list[Path]:
    """Write all three figures plus their plotted-value CSVs.

    Returns every path written, in a deterministic order.
    """
    out_dir = Path(out_dir)
    written: list[Path] = []

    svg1, data1 = figure_structural_metrics(analysis["structural"])
    svg2, data2 = figure_robustness_curves(analysis["robustness"])
    svg3, data3 = figure_robustness_by_topology(analysis["robustness"])

    for name, svg in (
        ("figure1_structural_metrics.svg", svg1),
        ("figure2_robustness_curves.svg", svg2),
        ("figure3_robustness_by_topology.svg", svg3),
    ):
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8", newline="\n")
        written.append(path)

    specs = (
        (
            "figure_data_1_structural_metrics.csv",
            data1,
            ("figure", "panel", "topology", "n", "mean", "std_ddof1", "n_seeds"),
        ),
        (
            "figure_data_2_robustness_curves.csv",
            data2,
            (
                "figure",
                "panel",
                "topology",
                "n",
                "seed",
                "fraction_removed",
                "largest_component_fraction",
            ),
        ),
        (
            "figure_data_3_robustness_by_topology.csv",
            data3,
            (
                "figure",
                "panel",
                "topology",
                "strategy",
                "n",
                "k_removed",
                "fraction_removed",
                "lcc_fraction_mean",
                "lcc_fraction_std_ddof1",
                "n_seeds",
            ),
        ),
    )
    for name, rows, fields in specs:
        written.append(write_csv(out_dir / name, rows, fields))
    return written
