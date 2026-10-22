"""Phone-sized poster: fish portfolio against the S&P 500."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from utils import (
    CHARTS_DIR,
    PORTFOLIO_PATH,
    PriceError,
    die,
    format_date,
    format_money,
    format_pct,
    format_shares,
    load_portfolio,
    market_today,
    percent_return,
)

BG = "#101114"
INK = "#F4F1EA"
MUTED = "#A1A6B0"
FISH = "#FF4D2E"
SPY = "#8E949F"
GRID = "#2A2E38"
WIDTH = 1080
HEIGHT = 1920


def price_on(prices: dict[date, float], day: date, ticker: str) -> float:
    if day in prices:
        return prices[day]
    earlier = [item for item in prices if item <= day]
    if earlier:
        return prices[max(earlier)]
    later = [item for item in prices if item >= day]
    if later:
        return prices[min(later)]
    raise PriceError(
        f"No daily close for {ticker} on {format_date(day)}. Nothing was saved."
    )


def history_days(
    spy_days: list[date], anchor: date, fish_dates: list[date]
) -> list[date]:
    """Trading days from the fund's start, plus a point when trades follow the last close."""
    covered = [day for day in spy_days if day >= anchor]
    if not covered:
        late = [day for day in fish_dates if day >= anchor]
        return [max(late) if late else anchor]
    last = covered[-1]
    late = [day for day in fish_dates if day > last]
    if late:
        return covered + [max(late)]
    return covered


def landing_day(trade_date: date, days: list[date]) -> date:
    for day in days:
        if day >= trade_date:
            return day
    return days[-1]


def _trade_date(trade: dict) -> date:
    try:
        return date.fromisoformat(str(trade["date"]))
    except (KeyError, TypeError, ValueError):
        die(f"A trade has no plain YYYY-MM-DD date ({trade.get('date')!r}).")


def build_history(
    portfolio: dict, closes: dict[str, dict[date, float]]
) -> tuple[list[date], list[float], list[float]]:
    if "SPY" not in closes or not closes["SPY"]:
        raise PriceError("SPY returned no daily closes. Nothing was saved.")
    trades = list(portfolio.get("trades", []))
    if not trades:
        raise PriceError("This portfolio has no trade history. Nothing was saved.")
    anchor = min(_trade_date(trade) for trade in trades)
    fish_trades = [
        (index, trade)
        for index, trade in enumerate(trades)
        if trade.get("side") in ("buy", "sell")
    ]
    fish_dates = [_trade_date(trade) for _index, trade in fish_trades]
    days = history_days(sorted(closes["SPY"]), anchor, fish_dates)
    buckets: dict[date, list[dict]] = {day: [] for day in days}
    for _index, trade in fish_trades:
        buckets[landing_day(_trade_date(trade), days)].append(trade)

    cash = float(portfolio["starting_cash"])
    holdings: dict[str, float] = {}
    fish_values: list[float] = []
    spy_values: list[float] = []
    spy_shares = float(portfolio["spy"]["shares"])
    for day in days:
        for trade in buckets[day]:
            ticker = str(trade["ticker"])
            shares = float(trade["shares"])
            amount = float(trade["amount"])
            if trade["side"] == "buy":
                holdings[ticker] = holdings.get(ticker, 0.0) + shares
                cash -= amount
            else:
                holdings[ticker] = holdings.get(ticker, 0.0) - shares
                if holdings[ticker] <= 1e-6:
                    holdings.pop(ticker, None)
                cash += amount
        value = cash
        for ticker, shares in holdings.items():
            if ticker not in closes:
                raise PriceError(
                    f"No daily closes for {ticker}. Nothing was saved."
                )
            value += shares * price_on(closes[ticker], day, ticker)
        fish_values.append(value)
        spy_values.append(spy_shares * price_on(closes["SPY"], day, "SPY"))
    return days, fish_values, spy_values


def tick_days(days: list[date]) -> list[date]:
    """A few of the series dates, so the axis never repeats a label."""
    if len(days) <= 4:
        return list(days)
    slots = (0, len(days) // 3, (2 * len(days)) // 3, len(days) - 1)
    chosen: list[date] = []
    for slot in slots:
        day = days[slot]
        if day not in chosen:
            chosen.append(day)
    return chosen


def _money_formatter(values: list[float]):
    span = max(values) - min(values) if values else 0.0

    def format_tick(value: float, _pos: int) -> str:
        if span < 50:
            return f"${value:,.0f}" if span >= 10 else f"${value:,.2f}"
        return f"${value:,.0f}"

    return FuncFormatter(format_tick)


def _holding_rows(
    portfolio: dict, closes: dict[str, dict[date, float]], day: date
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for ticker, shares in portfolio.get("holdings", {}).items():
        share_count = float(shares)
        if ticker not in closes:
            raise PriceError(f"No daily closes for {ticker}. Nothing was saved.")
        price = price_on(closes[ticker], day, ticker)
        rows.append((ticker, f"{format_shares(share_count)} sh", format_money(share_count * price)))
    rows.append(("Cash", "", format_money(float(portfolio["cash"]))))
    return rows


def write_chart(
    portfolio: dict,
    closes: dict[str, dict[date, float]],
    output: Path,
) -> Path:
    days, fish_values, spy_values = build_history(portfolio, closes)
    start_cash = float(portfolio["starting_cash"])
    if start_cash <= 0:
        die("starting_cash must be greater than zero.")
    fish_pct = percent_return(fish_values[-1], start_cash)
    spy_pct = percent_return(spy_values[-1], start_cash)
    rows = _holding_rows(portfolio, closes, days[-1])

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig = Figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.text(
        0.08,
        0.955,
        "FISH FUND",
        fontsize=44,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.08,
        0.918,
        f"since {format_date(days[0])}",
        fontsize=18,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.text(
        0.08,
        0.875,
        format_pct(fish_pct),
        fontsize=54,
        fontweight="bold",
        color=FISH,
        ha="left",
        va="top",
    )
    fig.text(
        0.54,
        0.875,
        format_pct(spy_pct),
        fontsize=54,
        fontweight="bold",
        color=SPY,
        ha="left",
        va="top",
    )
    fig.text(0.08, 0.812, "FISH", fontsize=16, fontweight="bold", color=FISH, ha="left", va="top")
    fig.text(
        0.54,
        0.812,
        "S&P 500",
        fontsize=16,
        fontweight="bold",
        color=SPY,
        ha="left",
        va="top",
    )

    ax = fig.add_axes([0.12, 0.40, 0.80, 0.36])
    ax.set_facecolor(BG)
    xs = mdates.date2num(days)
    mark = "o" if len(xs) < 3 else None
    (spy_line,) = ax.plot(xs, spy_values, color=SPY, linewidth=2.6, solid_capstyle="round", zorder=2)
    ax.plot(
        xs,
        fish_values,
        color=FISH,
        linewidth=8,
        alpha=0.18,
        solid_capstyle="round",
        zorder=3,
    )
    (fish_line,) = ax.plot(
        xs,
        fish_values,
        color=FISH,
        linewidth=3.2,
        marker=mark,
        markersize=8,
        solid_capstyle="round",
        zorder=4,
    )
    low = min(fish_values + spy_values)
    high = max(fish_values + spy_values)
    pad = max((high - low) * 0.18, 1.0)
    ax.set_ylim(low - pad, high + pad)
    if len(xs) == 1:
        ax.set_xlim(xs[0] - 1, xs[0] + 1)
    else:
        span = xs[-1] - xs[0]
        x_pad = max(span * 0.06, 0.6)
        ax.set_xlim(xs[0] - x_pad, xs[-1] + x_pad)
    ax.yaxis.set_major_formatter(_money_formatter(fish_values + spy_values))
    ax.set_xticks(mdates.date2num(tick_days(days)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(labelsize=13, length=0, pad=8)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for label in ax.get_xticklabels():
        label.set_ha("center")
    legend = ax.legend(
        [fish_line, spy_line],
        ["FISH", "S&P 500"],
        loc="upper left",
        frameon=False,
        fontsize=14,
        borderpad=0.2,
        handlelength=1.6,
    )
    for text, color in zip(legend.get_texts(), (FISH, SPY)):
        text.set_color(color)
        text.set_fontweight("bold")

    fig.text(
        0.08,
        0.345,
        "HOLDINGS",
        fontsize=16,
        fontweight="bold",
        color=MUTED,
        ha="left",
        va="top",
    )
    shown = rows[:9]
    for index, (label, shares, value) in enumerate(shown):
        y = 0.305 - index * 0.028
        fig.text(0.08, y, label, fontsize=22, fontweight="bold", color=INK, ha="left", va="center")
        fig.text(0.42, y, shares, fontsize=18, color=MUTED, ha="left", va="center")
        fig.text(0.92, y, value, fontsize=22, fontweight="bold", color=INK, ha="right", va="center")
    if len(rows) > len(shown):
        fig.text(
            0.08,
            0.305 - len(shown) * 0.028,
            f"and {len(rows) - len(shown)} more",
            fontsize=16,
            color=MUTED,
            ha="left",
            va="center",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=BG, edgecolor=BG)
    return output


def portfolio_tickers(portfolio: dict) -> list[str]:
    symbols: list[str] = []
    for trade in portfolio.get("trades", []):
        if trade.get("side") in ("buy", "sell"):
            ticker = str(trade.get("ticker", "")).upper()
            if ticker and ticker not in symbols:
                symbols.append(ticker)
    for ticker in portfolio.get("holdings", {}):
        if ticker not in symbols:
            symbols.append(str(ticker).upper())
    if "SPY" not in symbols:
        symbols.append("SPY")
    return symbols


def _fetch_closes(portfolio: dict) -> dict[str, dict[date, float]]:
    from utils import daily_closes

    dates = [_trade_date(trade) for trade in portfolio.get("trades", []) if trade.get("date")]
    anchor = min(dates) if dates else market_today()
    return daily_closes(
        portfolio_tickers(portfolio),
        anchor - timedelta(days=10),
        market_today(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Draw a 1080x1920 fish-fund chart against the S&P 500."
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=PORTFOLIO_PATH,
        help="Portfolio file. Default: portfolio.json next to this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path. Default: charts/fish-fund-YYYY-MM-DD.png next to this script.",
    )
    args = parser.parse_args(argv)
    portfolio = load_portfolio(args.portfolio)
    for key in ("cash", "starting_cash", "holdings", "spy", "trades"):
        if key not in portfolio:
            die(f"Portfolio is missing {key}.")
    output = args.output or (CHARTS_DIR / f"fish-fund-{format_date(market_today())}.png")
    try:
        path = write_chart(portfolio, _fetch_closes(portfolio), output)
    except PriceError as exc:
        die(str(exc))
    print(path.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
