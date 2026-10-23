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

BG = "#04100B"
FELT = "#0B2A1D"
FELT_HI = "#134531"
PANEL = "#020806"
INK = "#F4F1EA"
MUTED = "#8FA89C"
FISH = "#F5C542"
HOT = "#FF3B4E"
SPY = "#8FA89C"
GRID = "#1C3A2C"
CHIPS = ("#FF3B4E", "#2EE6D6", "#F5C542", "#A66BFF", "#9BFF4D")
WIDTH = 1080
HEIGHT = 1920
FONT_DIR = Path(__file__).resolve().parent / "fonts"


def _font(name: str, size: float):
    from matplotlib.font_manager import FontProperties

    path = FONT_DIR / name
    if path.exists():
        return FontProperties(fname=str(path), size=size)
    return FontProperties(family="DejaVu Sans", weight="bold", size=size)


def _px(pixels: float) -> float:
    """Font size in points for a cap height of roughly ``pixels`` on this canvas."""
    return pixels * 72 / 100


def _felt(width: int, height: int):
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    d = np.clip(np.sqrt(((xx - width * 0.5) / width) ** 2 + ((yy - height * 0.35) / height) ** 2) * 1.9, 0, 1)[..., None]

    def rgb(value: str):
        value = value.lstrip("#")
        return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], np.float32) / 255

    hi, mid, deep = rgb(FELT_HI), rgb(FELT), rgb(BG)
    image = np.where(d < 0.5, hi * (1 - d * 2) + mid * (d * 2), mid * (2 - d * 2) + deep * (d * 2 - 1))
    image += np.random.default_rng(7).normal(0, 0.016, image.shape)
    return np.clip(image, 0, 1)


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

    from matplotlib.patches import Ellipse, FancyBboxPatch

    anton = lambda px: _font("Anton-Regular.ttf", _px(px))  # noqa: E731
    bold = lambda px: _font("SpaceGrotesk-Bold.ttf", _px(px))  # noqa: E731
    mono = lambda px: _font("JetBrainsMono-Bold.ttf", _px(px))  # noqa: E731

    matplotlib.rcParams.update({"text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = Figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.figimage(_felt(WIDTH, HEIGHT), xo=0, yo=0, zorder=0, origin="upper")

    def fx(px: float) -> float:
        return px / WIDTH

    def fy(px: float) -> float:
        return 1 - px / HEIGHT

    def card(x: float, y: float, w: float, h: float, edge: str = "#41594E", face: str = PANEL, alpha: float = 0.6, lw: float = 2) -> None:
        fig.patches.append(
            FancyBboxPatch(
                (fx(x), fy(y + h)),
                fx(w),
                h / HEIGHT,
                boxstyle="round,pad=0,rounding_size=0.022",
                transform=fig.transFigure,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha,
                zorder=1,
                mutation_aspect=WIDTH / HEIGHT,
            )
        )

    day_count = (days[-1] - days[0]).days
    fig.text(fx(56), fy(60), "FISH FUND", fontproperties=anton(150), color=INK, ha="left", va="top")
    fig.text(
        fx(60),
        fy(250),
        f"5 ZEBRAFISH VS THE S&P 500  ·  DAY {day_count}",
        fontproperties=bold(34),
        color=FISH,
        ha="left",
        va="top",
    )
    fig.text(fx(60), fy(300), f"since {format_date(days[0])}", fontproperties=mono(24), color=MUTED, ha="left", va="top")

    # Score cards.
    lead = fish_pct - spy_pct
    for x, label, pct, colour, winning in (
        (40, "THE FISH", fish_pct, FISH, lead > 0.005),
        (556, "S&P 500", spy_pct, INK, lead < -0.005),
    ):
        card(x, 360, 484, 250, edge=colour if winning else "#41594E", lw=4 if winning else 2, alpha=0.7)
        fig.text(fx(x + 36), fy(400), label, fontproperties=bold(30), color=MUTED, ha="left", va="top")
        fig.text(
            fx(x + 30),
            fy(456),
            format_pct(pct),
            fontproperties=anton(120),
            color=(colour if pct >= 0 else HOT) if label == "THE FISH" else INK,
            ha="left",
            va="top",
        )
    if lead > 0.005:
        verdict, face, ink = "THE FISH ARE BEATING WALL STREET", FISH, PANEL
    elif lead < -0.005:
        verdict, face, ink = "WALL STREET IS AHEAD. FOR NOW.", HOT, INK
    else:
        verdict, face, ink = "DEAD HEAT", INK, PANEL
    card(40, 640, 1000, 84, edge=face, face=face, alpha=1.0)
    fig.text(0.5, fy(682), verdict, fontproperties=anton(46), color=ink, ha="center", va="center")

    # Chart.
    card(40, 760, 1000, 600, alpha=0.55)
    ax = fig.add_axes([fx(120), fy(1320), fx(880), 500 / HEIGHT])
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.set_zorder(5)
    xs = mdates.date2num(days)
    mark = "o" if len(xs) < 3 else None
    (spy_line,) = ax.plot(xs, spy_values, color=SPY, linewidth=3, linestyle=(0, (4, 3)), solid_capstyle="round", zorder=2)
    low = min(fish_values + spy_values)
    high = max(fish_values + spy_values)
    pad = max((high - low) * 0.18, 1.0)
    ax.fill_between(xs, fish_values, low - pad, color=FISH, alpha=0.10, zorder=1, linewidth=0)
    for width, alpha in ((22, 0.05), (14, 0.09), (8, 0.16)):
        ax.plot(xs, fish_values, color=FISH, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=3)
    (fish_line,) = ax.plot(
        xs,
        fish_values,
        color=FISH,
        linewidth=4.5,
        marker=mark,
        markersize=10,
        solid_capstyle="round",
        zorder=4,
    )
    ax.scatter([xs[-1]], [fish_values[-1]], s=260, color=FISH, zorder=5, edgecolors=PANEL, linewidths=4)
    ax.set_ylim(low - pad, high + pad)
    if len(xs) == 1:
        ax.set_xlim(xs[0] - 1, xs[0] + 1)
    else:
        span = xs[-1] - xs[0]
        x_pad = max(span * 0.06, 0.6)
        ax.set_xlim(xs[0] - x_pad, xs[-1] + x_pad)
    ax.yaxis.set_major_formatter(_money_formatter(fish_values + spy_values))
    ax.set_xticks(mdates.date2num(tick_days(days)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(length=0, pad=10)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(mono(19))
        tick.set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=1.2)
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
        prop=bold(24),
        borderpad=0.2,
        handlelength=1.8,
    )
    for text, color in zip(legend.get_texts(), (FISH, SPY)):
        text.set_color(color)

    # Holdings.
    shown = rows[:6]
    card(40, 1390, 1000, 110 + len(shown) * 52 + (40 if len(rows) > len(shown) else 0), alpha=0.55)
    fig.text(fx(76), fy(1424), "WHAT THE FISH OWN", fontproperties=bold(26), color=MUTED, ha="left", va="top")
    for index, (label, shares, value) in enumerate(shown):
        cy = 1500 + index * 52
        colour = CHIPS[index % len(CHIPS)] if label != "Cash" else "#7CE38B"
        fig.patches.append(
            Ellipse((fx(96), fy(cy)), 30 / WIDTH, 30 / HEIGHT, transform=fig.transFigure, facecolor=colour, edgecolor=INK, linewidth=2.5, zorder=3)
        )
        fig.text(fx(130), fy(cy), label.upper(), fontproperties=anton(40), color=INK, ha="left", va="center")
        fig.text(fx(560), fy(cy), shares, fontproperties=mono(24), color=MUTED, ha="right", va="center")
        fig.text(fx(1000), fy(cy), value, fontproperties=anton(40), color=INK, ha="right", va="center")
    if len(rows) > len(shown):
        fig.text(fx(130), fy(1500 + len(shown) * 52), f"and {len(rows) - len(shown)} more", fontproperties=mono(22), color=MUTED, ha="left", va="center")

    fig.text(
        0.5,
        fy(1866),
        "PAPER PORTFOLIO  ·  REAL MARKET PRICES  ·  NOT FINANCIAL ADVICE",
        fontproperties=mono(20),
        color=MUTED,
        ha="center",
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
