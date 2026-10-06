"""Exhibits for the research note.

Four of them, no more. Each answers one question a reader will actually ask, and anything
that would merely decorate the page is left out.

Colour follows a validated categorical palette in fixed slot order (blue, orange, aqua,
yellow). The order is the colourblind-safety mechanism, not a preference: an earlier draft
used orange and red together, which measures ΔE 7.1 in normal vision — below the readable
floor — so the slots are taken in sequence instead. Aqua and yellow sit under 3:1 contrast
on this surface, which obliges visible direct labels; every series carries one.

One y-axis per chart, always. The benchmark is a recessive grey dashed line rather than a
fourth hue, so it reads as the reference it is.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from study.config import BASE_COST_BPS, COST_GRID_BPS, RESULTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
GRID = "#e8e7e3"

# Fixed categorical slots 1-4 — validated as a set; do not reorder or cycle
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Diverging poles for signed values, with a neutral grey midpoint
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"

ARM_LABEL = {
    "rsi": "RSI alone",
    "sentiment": "Sentiment alone",
    "rsi_filtered": "RSI + sentiment filter",
    "rsi_negative_news": "RSI on negative news",
    "buy_and_hold": "Buy and hold",
}


def _style_axes(ax, *, ylabel: str = "", xlabel: str = "") -> None:
    """Recessive frame: no top/right spines, muted grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)


def _title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, color=INK_PRIMARY, fontsize=12.5, fontweight="bold",
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=INK_SECONDARY,
                fontsize=9.5, va="bottom", ha="left")


def _new_figure(width=9.0, height=5.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _place_end_labels(ax, entries, min_gap_frac: float = 0.052) -> None:
    """Direct-label series at their right-hand end, nudged apart so they stay legible.

    Series whose final values are close — which is exactly what happens when two arms
    perform similarly, the interesting case — would otherwise print their labels on top of
    each other. Positions are resolved in axes fractions, so the spacing holds whatever
    the y-scale is, log included.

    `entries` is a list of (y_value, text, colour, weight).
    """
    if not entries:
        return

    y_low, y_high = ax.get_ylim()

    def to_frac(value):
        if ax.get_yscale() == "log":
            import math

            return (math.log10(value) - math.log10(y_low)) / (
                math.log10(y_high) - math.log10(y_low)
            )
        return (value - y_low) / (y_high - y_low)

    placed = sorted(
        ((to_frac(y), text, colour, weight) for y, text, colour, weight in entries),
        key=lambda item: item[0],
    )

    # push upward through the stack until every pair clears the minimum gap
    resolved = []
    previous = -1.0
    for frac, text, colour, weight in placed:
        frac = max(frac, previous + min_gap_frac)
        resolved.append((frac, text, colour, weight))
        previous = frac

    # if the stack overflowed the top, slide the whole thing back down
    overflow = resolved[-1][0] - 0.985
    if overflow > 0:
        resolved = [(f - overflow, t, c, w) for f, t, c, w in resolved]

    for frac, text, colour, weight in resolved:
        ax.annotate(
            text,
            xy=(1.012, frac),
            xycoords="axes fraction",
            color=colour,
            fontsize=9,
            fontweight=weight,
            va="center",
            ha="left",
            annotation_clip=False,
        )


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


# ── Exhibit 1: equity curves ──────────────────────────────────────────────────


def equity_curves(curves: pd.DataFrame, out_dir: Path = RESULTS_DIR) -> Path:
    """Log-scaled growth of 1 unit. Log, because comparing compound paths on a linear
    axis makes late divergence look like the whole story."""
    fig, ax = _new_figure(width=10.0)
    labels = []

    plotted = [c for c in ("rsi", "rsi_filtered", "rsi_negative_news") if c in curves]
    for slot, arm in enumerate(plotted):
        series = curves[arm].dropna()
        if series.empty:
            continue
        ax.plot(series.index, series.values, color=SLOT[slot], linewidth=2.0,
                zorder=3, label=ARM_LABEL[arm])
        labels.append((series.iloc[-1], ARM_LABEL[arm], SLOT[slot], "bold"))

    if "buy_and_hold" in curves:
        series = curves["buy_and_hold"].dropna()
        if not series.empty:
            ax.plot(series.index, series.values, color=INK_MUTED, linewidth=1.6,
                    linestyle=(0, (5, 3)), zorder=2, label=ARM_LABEL["buy_and_hold"])
            labels.append(
                (series.iloc[-1], ARM_LABEL["buy_and_hold"], INK_SECONDARY, "normal")
            )

    ax.set_yscale("log")
    ax.axhline(1.0, color=INK_MUTED, linewidth=0.9, zorder=1)
    _style_axes(ax, ylabel="Growth of 1 unit (log scale)")
    _title(ax, "Equity curves",
           f"Net of {BASE_COST_BPS:.0f}bps round-trip cost · log scale")

    _place_end_labels(ax, labels)
    # Legend below the plot: inside the axes it lands on the data, and arms that perform
    # similarly leave no clear space anywhere in the frame.
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(right=0.78, bottom=0.18)

    return _save(fig, "exhibit_1_equity_curves.png", out_dir)


# ── Exhibit 2: cost sensitivity ───────────────────────────────────────────────


def cost_sensitivity(results: dict, out_dir: Path = RESULTS_DIR) -> Path:
    """Sharpe against round-trip cost — the chart that shows where the edge dies.

    Usually the most persuasive exhibit in a study like this: an edge that evaporates
    between 0 and 10bps was never tradeable, and saying so plainly is worth more than
    any headline number.
    """
    fig, ax = _new_figure(width=10.0, height=5.2)
    costs = list(COST_GRID_BPS)
    labels = []

    for slot, arm in enumerate(results["arms"]):
        sharpes = [results["arms"][arm]["by_cost"][str(c)]["sharpe"] for c in costs]
        if any(s is None for s in sharpes):
            continue
        colour = SLOT[slot % len(SLOT)]
        ax.plot(costs, sharpes, color=colour, linewidth=2.0, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3, label=ARM_LABEL.get(arm, arm))
        labels.append((sharpes[-1], ARM_LABEL.get(arm, arm), colour, "bold"))

    bh = results["benchmark"]["buy_and_hold"]["sharpe"]
    if bh is not None:
        ax.axhline(bh, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
        labels.append((bh, "Buy and hold", INK_SECONDARY, "normal"))
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.0, zorder=2)

    ax.set_xticks(costs)
    ax.set_xticklabels([f"{c:.0f}" for c in costs])
    _style_axes(ax, ylabel="Annualised Sharpe ratio",
                xlabel="Round-trip transaction cost (basis points)")
    _title(ax, "Where the edge dies",
           "Sharpe ratio as realistic trading costs are applied")

    _place_end_labels(ax, labels)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(right=0.76, bottom=0.20)

    return _save(fig, "exhibit_2_cost_sensitivity.png", out_dir)


# ── Exhibit 3: the mechanism ──────────────────────────────────────────────────


def sentiment_buckets(buckets: pd.DataFrame, out_dir: Path = RESULTS_DIR) -> Path:
    """Mean trade outcome by sentiment at signal — diverging, coloured by sign.

    This is the mechanism evidence. If the filter works through the channel the
    hypothesis claims, outcomes should improve as sentiment at entry improves. A flat
    profile here means the headline number, whatever it is, is not coming from this.
    """
    if buckets.empty:
        logger.warning("No sentiment buckets to plot")
        return out_dir / "exhibit_3_sentiment_buckets.png"

    fig, ax = _new_figure(width=8.5, height=5.0)

    x = range(len(buckets))
    values = buckets["mean_return"] * 100
    colours = [POS if v > 0 else NEG for v in values]

    ax.bar(x, values, color=colours, width=0.62, zorder=3,
           edgecolor=SURFACE, linewidth=2.0)   # 2px surface gap between bars

    for i, value in enumerate(values):
        offset = 4 if value >= 0 else -4
        ax.annotate(f"{value:+.2f}%", xy=(i, value), xytext=(0, offset),
                    textcoords="offset points", ha="center",
                    va="bottom" if value >= 0 else "top",
                    color=INK_PRIMARY, fontsize=9, fontweight="bold")

    ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.1, zorder=4)
    ax.set_xticks(list(x))
    # Sample size goes in the tick label rather than floating near the axis, where it
    # collided with the tick text whenever a bar sat close to zero.
    ax.set_xticklabels(
        [
            f"{s:+.2f}\nn={n:,}"
            for s, n in zip(buckets["mean_sentiment"], buckets["trades"])
        ]
    )
    _style_axes(ax, ylabel="Mean net return per trade (%)",
                xlabel="Mean news sentiment at signal (negative → positive)")
    _title(ax, "Does sentiment at entry predict the outcome?",
           f"RSI entries bucketed by sentiment · net of {BASE_COST_BPS:.0f}bps")

    return _save(fig, "exhibit_3_sentiment_buckets.png", out_dir)


# ── Exhibit 4: subperiod stability (a table, because it is one) ────────────────


def subperiod_markdown(results: dict) -> str:
    """Subperiod stability as a markdown table for the note.

    A three-row, four-column grid is a table. Drawing it as a chart would add ink
    without adding information.
    """
    lines = ["| Arm | Period | CAGR | Sharpe | Max drawdown |",
             "|---|---|---:|---:|---:|"]

    def fmt(value, kind):
        if value is None:
            return "—"
        return f"{value:.2%}" if kind == "pct" else f"{value:.2f}"

    for arm, payload in results["arms"].items():
        for row in payload.get("subperiods", []):
            if not row.get("days"):
                continue
            lines.append(
                f"| {ARM_LABEL.get(arm, arm)} | {row['period']} "
                f"| {fmt(row.get('cagr'), 'pct')} | {fmt(row.get('sharpe'), 'num')} "
                f"| {fmt(row.get('max_drawdown'), 'pct')} |"
            )

    for row in results["benchmark"].get("subperiods", []):
        if not row.get("days"):
            continue
        lines.append(
            f"| Buy and hold | {row['period']} | {fmt(row.get('cagr'), 'pct')} "
            f"| {fmt(row.get('sharpe'), 'num')} | {fmt(row.get('max_drawdown'), 'pct')} |"
        )

    return "\n".join(lines)


def build_all(results: dict, out_dir: Path = RESULTS_DIR) -> dict:
    """Every exhibit the note references. Returns a map of name → path."""
    paths = {}

    curves_path = out_dir / "equity_curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path, index_col=0, parse_dates=True)
        paths["equity"] = equity_curves(curves, out_dir)

    paths["cost"] = cost_sensitivity(results, out_dir)

    base_arm = "rsi"
    buckets = pd.DataFrame(results["arms"].get(base_arm, {}).get("sentiment_buckets", []))
    paths["buckets"] = sentiment_buckets(buckets, out_dir)

    (out_dir / "subperiods.md").write_text(subperiod_markdown(results), encoding="utf-8")
    paths["subperiods"] = out_dir / "subperiods.md"

    return paths
