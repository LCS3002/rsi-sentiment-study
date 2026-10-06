"""Render the research note from `results.json`, so no number in it is typed by hand.

The prose lives in `notes/research-note.template.md` with `{{token}}` placeholders. This
module resolves each token against the results file and writes `notes/research-note.md`.

The point is not convenience. A note whose figures are transcribed drifts the moment any
parameter changes, and the drift is invisible — the prose still reads fine. Generating
them means a stale number is impossible by construction, and `--check` makes that
verifiable: it fails if the rendered note differs from the committed one.

Token syntax:

    {{arm.rsi_filtered.sharpe}}        metric at the headline cost
    {{arm.rsi.cagr:pct}}               formatted as a percentage
    {{arm.rsi.cost.20.sharpe}}         a specific cost level
    {{bench.sharpe}}                   buy-and-hold
    {{config.tickers}}                 study configuration
    {{validation.ic.5.mean_ic}}        information coefficient at a horizon
    {{table.headline}}                 a rendered markdown table
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from study.config import BASE_COST_BPS, COST_GRID_BPS, RESULTS_DIR, ROOT

logger = logging.getLogger(__name__)

TEMPLATE_PATH = ROOT / "notes" / "research-note.template.md"
OUTPUT_PATH = ROOT / "notes" / "research-note.md"

ARM_LABEL = {
    "rsi": "RSI alone",
    "sentiment": "Sentiment alone",
    "rsi_filtered": "RSI + sentiment filter",
    "rsi_negative_news": "RSI on negative news",
}

_TOKEN = re.compile(r"\{\{([a-zA-Z0-9_.]+)(?::([a-z0-9.]+))?\}\}")


# ── Formatting ────────────────────────────────────────────────────────────────


def _fmt(value, spec: str | None) -> str:
    """Format a resolved value. `None` becomes an em dash rather than 'None'."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value

    if spec == "pct":
        return f"{value:.2%}"
    if spec == "pct1":
        return f"{value:.1%}"
    if spec == "bps":
        return f"{value:.0f}bps"
    if spec == "int":
        return f"{int(value):,}"
    if spec == "signed":
        return f"{value:+.2f}"
    if spec == "sig4":
        return f"{value:+.4f}"
    if spec:
        return format(value, spec)

    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pct(value) -> str:
    return "—" if value is None else f"{value:.2%}"


def _num(value, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}"


# ── Tables ────────────────────────────────────────────────────────────────────


def _table_headline(results: dict) -> str:
    base = str(BASE_COST_BPS)
    lines = [
        f"| Arm | CAGR | Volatility | Sharpe | Max DD | t (NW) | Trades | Hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in ARM_LABEL.items():
        s = results["arms"].get(arm, {}).get("by_cost", {}).get(base)
        if not s:
            continue
        lines.append(
            f"| {label} | {_pct(s.get('cagr'))} | {_pct(s.get('volatility'))} "
            f"| {_num(s.get('sharpe'))} | {_pct(s.get('max_drawdown'))} "
            f"| {_num(s.get('nw_tstat'))} | {s.get('trades', 0):,} "
            f"| {_pct(s.get('hit_rate'))} |"
        )
    b = results["benchmark"]["buy_and_hold"]
    lines.append(
        f"| *Buy and hold* | {_pct(b.get('cagr'))} | {_pct(b.get('volatility'))} "
        f"| {_num(b.get('sharpe'))} | {_pct(b.get('max_drawdown'))} "
        f"| {_num(b.get('nw_tstat'))} | — | — |"
    )
    return "\n".join(lines)


def _table_cost(results: dict) -> str:
    costs = [str(c) for c in COST_GRID_BPS]
    header = " | ".join(f"{c}bps" for c in costs)
    lines = [f"| Arm | {header} |", "|---|" + "---:|" * len(costs)]
    for arm, label in ARM_LABEL.items():
        payload = results["arms"].get(arm, {}).get("by_cost", {})
        if not payload:
            continue
        cells = " | ".join(_num(payload.get(c, {}).get("sharpe")) for c in costs)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def _table_subperiods(results: dict) -> str:
    lines = ["| Arm | Period | CAGR | Sharpe | Max DD |", "|---|---|---:|---:|---:|"]
    for arm, label in ARM_LABEL.items():
        for row in results["arms"].get(arm, {}).get("subperiods", []):
            if not row.get("days"):
                continue
            lines.append(
                f"| {label} | {row['period']} | {_pct(row.get('cagr'))} "
                f"| {_num(row.get('sharpe'))} | {_pct(row.get('max_drawdown'))} |"
            )
    for row in results["benchmark"].get("subperiods", []):
        if not row.get("days"):
            continue
        lines.append(
            f"| *Buy and hold* | {row['period']} | {_pct(row.get('cagr'))} "
            f"| {_num(row.get('sharpe'))} | {_pct(row.get('max_drawdown'))} |"
        )
    return "\n".join(lines)


def _table_ic(results: dict) -> str:
    rows = results.get("validation", {}).get("information_coefficient", [])
    lines = [
        "| Horizon | Mean IC | t (NW) | IC IR | Days positive | Days |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row.get("days"):
            continue
        lines.append(
            f"| {row['horizon_days']}d | {_num(row.get('mean_ic'), 4)} "
            f"| {_num(row.get('ic_tstat'))} | {_num(row.get('ic_ir'))} "
            f"| {_pct(row.get('share_positive'))} | {row['days']:,} |"
        )
    return "\n".join(lines)


def _table_quintiles(results: dict) -> str:
    rows = results.get("validation", {}).get("quintile_spread_5d", [])
    lines = [
        "| Quintile | Mean sentiment | Mean 5d return | Hit rate | Observations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| Q{row['bucket']} | {_num(row.get('mean_sentiment'), 3)} "
            f"| {_pct(row.get('mean_forward_return'))} | {_pct(row.get('hit_rate'))} "
            f"| {row.get('observations', 0):,} |"
        )
    return "\n".join(lines)


TABLES = {
    "headline": _table_headline,
    "cost": _table_cost,
    "subperiods": _table_subperiods,
    "ic": _table_ic,
    "quintiles": _table_quintiles,
}


# ── Resolution ────────────────────────────────────────────────────────────────


def resolve(path: str, results: dict):
    """Resolve one dotted token path against the results payload."""
    parts = path.split(".")
    head = parts[0]

    if head == "table":
        builder = TABLES.get(parts[1])
        if builder is None:
            raise KeyError(f"unknown table {parts[1]!r}")
        return builder(results)

    if head == "arm":
        arm = results["arms"][parts[1]]
        if len(parts) > 3 and parts[2] == "cost":
            return arm["by_cost"][parts[3]].get(parts[4])
        return arm["by_cost"][str(BASE_COST_BPS)].get(parts[2])

    if head == "bench":
        return results["benchmark"]["buy_and_hold"].get(parts[1])

    if head == "config":
        return results["config"].get(parts[1])

    if head == "validation":
        validation = results.get("validation", {})
        if parts[1] == "ic":
            for row in validation.get("information_coefficient", []):
                if str(row.get("horizon_days")) == parts[2]:
                    return row.get(parts[3])
            return None
        if parts[1] == "dist":
            return validation.get("distribution", {}).get(parts[2])
        if parts[1] == "quintile_spread":
            buckets = validation.get("quintile_spread_5d", [])
            if len(buckets) < 2:
                return None
            return buckets[-1]["mean_forward_return"] - buckets[0]["mean_forward_return"]

    raise KeyError(f"unresolvable token {path!r}")


def render(results: dict, template: str) -> str:
    """Substitute every token. An unresolvable one raises rather than rendering blank —
    a silently empty figure in a research note is worse than a crash."""
    missing = []

    def substitute(match: re.Match) -> str:
        path, spec = match.group(1), match.group(2)
        try:
            return _fmt(resolve(path, results), spec)
        except (KeyError, IndexError, TypeError) as e:
            missing.append(f"{path} ({type(e).__name__})")
            return match.group(0)

    output = _TOKEN.sub(substitute, template)
    if missing:
        raise KeyError(f"unresolved tokens: {', '.join(sorted(set(missing)))}")
    return output


def build(results_path: Path | None = None, check: bool = False) -> Path:
    results_path = results_path or (RESULTS_DIR / "results.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    rendered = render(results, template)

    if check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if current != rendered:
            raise SystemExit(
                f"{OUTPUT_PATH.name} is stale — rerun `python -m study.note` to regenerate"
            )
        logger.info("%s is up to date", OUTPUT_PATH.name)
        return OUTPUT_PATH

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    logger.info("Wrote %s", OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render the research note")
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed note is stale rather than rewriting it")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    build(check=args.check)
