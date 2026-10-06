"""Note rendering.

The note's credibility rests on its figures being generated rather than typed, so the
renderer has to fail loudly. A token that silently renders blank would leave a research
note with a hole in it that still reads as finished prose.
"""

import json

import pytest

from study.note import TEMPLATE_PATH, _fmt, build, render, resolve


@pytest.fixture
def results():
    """A results payload with the shape study.run produces."""
    arm = lambda sharpe: {  # noqa: E731
        "by_cost": {
            "0.0": {"sharpe": sharpe + 0.1, "cagr": 0.08, "volatility": 0.12,
                    "max_drawdown": -0.2, "nw_tstat": 2.1, "trades": 900, "hit_rate": 0.55},
            "10.0": {"sharpe": sharpe, "cagr": 0.06, "volatility": 0.12,
                     "max_drawdown": -0.22, "nw_tstat": 1.8, "trades": 900, "hit_rate": 0.54},
            "20.0": {"sharpe": sharpe - 0.1, "cagr": 0.04, "volatility": 0.12,
                     "max_drawdown": -0.24, "nw_tstat": 1.4, "trades": 900, "hit_rate": 0.53},
            "5.0": {"sharpe": sharpe + 0.05, "cagr": 0.07, "volatility": 0.12,
                    "max_drawdown": -0.21, "nw_tstat": 2.0, "trades": 900, "hit_rate": 0.54},
        },
        "subperiods": [
            {"period": "2010-2014", "days": 1200, "cagr": 0.05,
             "sharpe": sharpe, "max_drawdown": -0.1}
        ],
        "sentiment_buckets": [],
    }

    return {
        "arms": {
            "rsi": arm(0.50),
            "sentiment": arm(-0.20),
            "rsi_filtered": arm(0.62),
            "rsi_negative_news": arm(0.10),
        },
        "benchmark": {
            "buy_and_hold": {"sharpe": 0.70, "cagr": 0.13, "volatility": 0.19,
                             "max_drawdown": -0.34, "nw_tstat": 2.6},
            "subperiods": [{"period": "2010-2014", "days": 1200, "cagr": 0.14,
                            "sharpe": 0.9, "max_drawdown": -0.12}],
        },
        "validation": {
            "distribution": {"coverage": 0.41, "share_near_neutral": 0.33,
                             "scored_ticker_days": 120000},
            "information_coefficient": [
                {"horizon_days": 1, "days": 3000, "mean_ic": 0.012,
                 "ic_tstat": 2.4, "ic_ir": 0.08, "share_positive": 0.53},
                {"horizon_days": 5, "days": 3000, "mean_ic": 0.008,
                 "ic_tstat": 1.1, "ic_ir": 0.05, "share_positive": 0.51},
            ],
            "quintile_spread_5d": [
                {"bucket": 1, "mean_sentiment": -0.6, "mean_forward_return": -0.004,
                 "hit_rate": 0.48, "observations": 20000},
                {"bucket": 5, "mean_sentiment": 0.6, "mean_forward_return": 0.003,
                 "hit_rate": 0.53, "observations": 20000},
            ],
        },
        "config": {
            "tickers": 84, "start": "2010-01-04", "end": "2023-10-23",
            "ticker_days": 274244, "hold_days": 5, "max_positions": 20,
            "base_cost_bps": 10.0, "cost_grid_bps": [0.0, 5.0, 10.0, 20.0],
            "sentiment_threshold": 0.2, "sentiment_halflife_days": 3.0,
            "sentiment_chunk_size": 5000, "newey_west_lags": 10,
        },
    }


# ── Token resolution ──────────────────────────────────────────────────────────


def test_arm_metric_resolves_at_the_headline_cost(results):
    assert resolve("arm.rsi_filtered.sharpe", results) == 0.62


def test_arm_metric_resolves_at_a_named_cost(results):
    assert resolve("arm.rsi_filtered.cost.20.sharpe", results) == pytest.approx(0.52)


def test_benchmark_and_config_resolve(results):
    assert resolve("bench.sharpe", results) == 0.70
    assert resolve("config.tickers", results) == 84


def test_validation_tokens_resolve(results):
    assert resolve("validation.ic.5.mean_ic", results) == 0.008
    assert resolve("validation.dist.coverage", results) == 0.41
    assert resolve("validation.quintile_spread", results) == pytest.approx(0.007)


def test_unknown_token_raises(results):
    with pytest.raises(KeyError):
        resolve("nonsense.path", results)


# ── Formatting ────────────────────────────────────────────────────────────────


def test_none_renders_as_an_em_dash_not_the_word_none():
    assert _fmt(None, None) == "—"
    assert _fmt(None, "pct") == "—"


def test_percentage_and_list_formatting():
    assert _fmt(0.0612, "pct") == "6.12%"
    assert _fmt([0.0, 5.0, 10.0, 20.0], None) == "0 / 5 / 10 / 20"
    assert _fmt(274244, "int") == "274,244"


# ── Rendering ─────────────────────────────────────────────────────────────────


def test_render_substitutes_tokens(results):
    out = render(results, "Sharpe was {{arm.rsi_filtered.sharpe}} on {{config.tickers}} names.")
    assert out == "Sharpe was 0.62 on 84 names."


def test_render_raises_on_an_unresolvable_token(results):
    """A blank figure in a research note still reads as finished prose, which is worse
    than a crash — so this must fail rather than render empty."""
    with pytest.raises(KeyError, match="unresolved tokens"):
        render(results, "Value: {{arm.does_not_exist.sharpe}}")


def test_render_builds_every_table_without_error(results):
    for table in ("headline", "cost", "subperiods", "ic", "quintiles"):
        out = render(results, f"{{{{table.{table}}}}}")
        assert out.startswith("|") and "---" in out


def test_the_real_template_has_no_unresolvable_tokens(results):
    """Catches a typo in the note template without needing a full study run."""
    render(results, TEMPLATE_PATH.read_text(encoding="utf-8"))


def test_check_mode_fails_when_the_note_is_stale(results, tmp_path, monkeypatch):
    import study.note as note

    template = tmp_path / "t.md"
    output = tmp_path / "note.md"
    template.write_text("Sharpe {{arm.rsi.sharpe}}", encoding="utf-8")
    output.write_text("Sharpe 9.99", encoding="utf-8")       # deliberately stale

    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(results), encoding="utf-8")

    monkeypatch.setattr(note, "TEMPLATE_PATH", template)
    monkeypatch.setattr(note, "OUTPUT_PATH", output)

    with pytest.raises(SystemExit, match="stale"):
        build(results_file, check=True)

    build(results_file)                                       # regenerate
    build(results_file, check=True)                           # now passes
