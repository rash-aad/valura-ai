from __future__ import annotations
import datetime
from typing import Any
import yfinance as yf

from src.models import (
    PortfolioHealthResult, ConcentrationRisk,
    Performance, BenchmarkComparison, Observation
)

DISCLAIMER = (
    "This analysis is for informational purposes only and does not constitute "
    "investment advice. Past performance is not indicative of future results. "
    "Please consult a qualified financial adviser before making investment decisions."
)

BENCHMARK_TICKERS = {
    "S&P 500": "^GSPC",
    "QQQ": "QQQ",
    "FTSE 100": "^FTSE",
    "MSCI World": "URTH",
    "NIKKEI 225": "^N225",
}

def _get_prices(tickers: list[str]) -> dict[str, float]:
    prices = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            prices[ticker] = float(info.last_price)
        except Exception:
            try:
                hist = yf.download(ticker, period="2d", progress=False, auto_adjust=True)
                if not hist.empty:
                    prices[ticker] = float(hist["Close"].iloc[-1])
            except Exception:
                prices[ticker] = None
    return prices

def _get_benchmark_return(benchmark_name: str) -> float | None:
    ticker = BENCHMARK_TICKERS.get(benchmark_name, "^GSPC")
    try:
        hist = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return None
        start = float(hist["Close"].iloc[0])
        end = float(hist["Close"].iloc[-1])
        return round((end - start) / start * 100, 2)
    except Exception:
        return None

def _calc_concentration(positions: list[dict], prices: dict[str, float]) -> tuple[ConcentrationRisk, list[tuple[str, float]]]:
    values = []
    for p in positions:
        price = prices.get(p["ticker"])
        if price:
            values.append((p["ticker"], price * p["quantity"]))

    if not values:
        return None, []

    total = sum(v for _, v in values)
    if total == 0:
        return None, []

    sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)
    pct_list = [(t, round(v / total * 100, 2)) for t, v in sorted_vals]

    top1 = pct_list[0][1]
    top3 = sum(p for _, p in pct_list[:3])

    if top1 >= 50:
        flag = "high"
    elif top1 >= 30:
        flag = "medium"
    else:
        flag = "low"

    return ConcentrationRisk(
        top_position_pct=top1,
        top_3_positions_pct=round(top3, 2),
        flag=flag,
    ), pct_list

def _calc_performance(positions: list[dict], prices: dict[str, float]) -> Performance | None:
    total_cost = 0.0
    total_value = 0.0
    earliest_date = None

    for p in positions:
        price = prices.get(p["ticker"])
        if not price:
            continue
        cost = p["avg_cost"] * p["quantity"]
        value = price * p["quantity"]
        total_cost += cost
        total_value += value
        purchased = datetime.date.fromisoformat(p["purchased_at"])
        if earliest_date is None or purchased < earliest_date:
            earliest_date = purchased

    if total_cost == 0 or earliest_date is None:
        return None

    total_return = (total_value - total_cost) / total_cost * 100
    days_held = (datetime.date.today() - earliest_date).days
    years_held = max(days_held / 365.25, 0.01)
    annualized = ((1 + total_return / 100) ** (1 / years_held) - 1) * 100

    return Performance(
        total_return_pct=round(total_return, 2),
        annualized_return_pct=round(annualized, 2),
    )

def _build_observations(
    user: dict,
    concentration: ConcentrationRisk | None,
    performance: Performance | None,
    benchmark: BenchmarkComparison | None,
    pct_list: list[tuple[str, float]],
) -> list[Observation]:
    obs = []

    if concentration:
        top_ticker = pct_list[0][0] if pct_list else "a single position"
        top_pct = pct_list[0][1] if pct_list else 0
        if concentration.flag == "high":
            obs.append(Observation(
                severity="warning",
                text=f"{top_pct:.1f}% of your portfolio is in {top_ticker} — this is highly concentrated. "
                     f"A single stock drop could significantly impact your overall returns."
            ))
        elif concentration.flag == "medium":
            obs.append(Observation(
                severity="info",
                text=f"{top_pct:.1f}% of your portfolio is in {top_ticker}. "
                     f"Consider whether this concentration matches your risk tolerance."
            ))

    if performance:
        if performance.total_return_pct > 0:
            obs.append(Observation(
                severity="info",
                text=f"Your portfolio is up {performance.total_return_pct:.1f}% overall "
                     f"({performance.annualized_return_pct:.1f}% annualized)."
            ))
        else:
            obs.append(Observation(
                severity="warning",
                text=f"Your portfolio is down {abs(performance.total_return_pct):.1f}% overall."
            ))

    if benchmark:
        if benchmark.alpha_pct > 0:
            obs.append(Observation(
                severity="info",
                text=f"You're outperforming the {benchmark.benchmark} by {benchmark.alpha_pct:.1f}% "
                     f"over the past year — good relative performance."
            ))
        else:
            obs.append(Observation(
                severity="info",
                text=f"You're underperforming the {benchmark.benchmark} by {abs(benchmark.alpha_pct):.1f}% "
                     f"over the past year."
            ))

    risk_profile = user.get("risk_profile", "moderate")
    age = user.get("age", 30)
    if risk_profile == "conservative" and concentration and concentration.flag != "low":
        obs.append(Observation(
            severity="warning",
            text="Your concentration level may not match your conservative risk profile. "
                 "Consider diversifying across more asset classes."
        ))
    if age >= 60 and performance and performance.annualized_return_pct < 0:
        obs.append(Observation(
            severity="warning",
            text="As a retiree-age investor, negative returns are a concern. "
                 "Review your defensive holdings and consider capital preservation strategies."
        ))

    return obs

def run(user: dict, llm: Any = None) -> dict:
    positions = user.get("positions", [])
    benchmark_name = user.get("preferences", {}).get("preferred_benchmark", "S&P 500")

    # Empty portfolio — BUILD guidance
    if not positions:
        return PortfolioHealthResult(
            is_empty_portfolio=True,
            observations=[
                Observation(
                    severity="info",
                    text="You don't have any positions yet. That's a great starting point — "
                         "the best time to build good habits is before you invest."
                ),
                Observation(
                    severity="info",
                    text="Consider starting with a low-cost broad market ETF (e.g. VTI or VOO) "
                         "to get diversified exposure with a single instrument."
                ),
                Observation(
                    severity="info",
                    text="Define your goal first: are you investing for retirement, a house, "
                         "or general wealth building? Your time horizon drives everything else."
                ),
            ],
            build_guidance=(
                "To get started: (1) Set a clear goal and time horizon. "
                "(2) Pick an asset allocation matching your risk profile. "
                "(3) Start with broad index funds before adding individual stocks. "
                "(4) Invest consistently — even small amounts compounded over time make a big difference."
            ),
            disclaimer=DISCLAIMER,
        ).model_dump()

    # Fetch prices
    tickers = [p["ticker"] for p in positions]
    prices = _get_prices(tickers)

    # Calculations
    concentration, pct_list = _calc_concentration(positions, prices)
    performance = _calc_performance(positions, prices)

    # Benchmark
    benchmark_return = _get_benchmark_return(benchmark_name)
    benchmark = None
    if performance and benchmark_return is not None:
        alpha = round(performance.total_return_pct - benchmark_return, 2)
        benchmark = BenchmarkComparison(
            benchmark=benchmark_name,
            portfolio_return_pct=performance.total_return_pct,
            benchmark_return_pct=benchmark_return,
            alpha_pct=alpha,
        )

    observations = _build_observations(user, concentration, performance, benchmark, pct_list)

    return PortfolioHealthResult(
        concentration_risk=concentration,
        performance=performance,
        benchmark_comparison=benchmark,
        observations=observations,
        disclaimer=DISCLAIMER,
    ).model_dump()
