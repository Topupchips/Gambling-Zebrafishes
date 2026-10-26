"""Manual fish-fund ledger. Trades happen at a brokerage; this only records them."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

from utils import (
    PORTFOLIO_PATH,
    PriceError,
    clean_price,
    clean_shares,
    die,
    format_money,
    format_pct,
    format_shares,
    latest_prices,
    load_portfolio,
    market_today,
    money,
    normalize_ticker,
    percent_return,
    save_portfolio,
)


class FundError(Exception):
    """The books rejected a command. The portfolio file is left unchanged."""


def init_portfolio(cash: float, spy_price: float, on: date) -> dict:
    if not math.isfinite(cash) or money(cash) <= 0:
        raise FundError("--cash must be greater than zero.")
    if not math.isfinite(spy_price) or spy_price <= 0:
        raise FundError("SPY price was not usable. Nothing was saved.")
    starting = money(cash)
    price = clean_price(spy_price)
    shares = clean_shares(starting / price)
    day = on.isoformat()
    return {
        "cash": starting,
        "starting_cash": starting,
        "holdings": {},
        "spy": {"shares": shares, "start_price": price},
        "trades": [
            {
                "side": "spy",
                "ticker": "SPY",
                "shares": shares,
                "price": price,
                "amount": starting,
                "date": day,
            }
        ],
    }


def buy(portfolio: dict, ticker: str, amount: float, price: float, on: date) -> None:
    spent = money(amount)
    if not math.isfinite(amount) or spent <= 0:
        raise FundError("--amount must be greater than zero. Nothing was saved.")
    if not math.isfinite(price) or price <= 0:
        raise FundError(f"No usable price for {ticker}. Nothing was saved.")
    cash = money(portfolio["cash"])
    if spent - cash > 0:
        raise FundError(
            f"Amount {format_money(spent)} is more than cash {format_money(cash)}. "
            "Nothing was saved."
        )
    shares = clean_shares(spent / price)
    if shares <= 0:
        raise FundError(f"That amount buys no shares of {ticker}. Nothing was saved.")
    holdings = portfolio["holdings"]
    holdings[ticker] = clean_shares(float(holdings.get(ticker, 0.0)) + shares)
    portfolio["cash"] = money(cash - spent)
    portfolio["trades"].append(
        {
            "side": "buy",
            "ticker": ticker,
            "shares": shares,
            "price": clean_price(price),
            "amount": spent,
            "date": on.isoformat(),
        }
    )


def sell(portfolio: dict, ticker: str, shares: float, price: float, on: date) -> None:
    if not math.isfinite(shares) or shares <= 0:
        raise FundError("--shares must be greater than zero. Nothing was saved.")
    if not math.isfinite(price) or price <= 0:
        raise FundError(f"No usable price for {ticker}. Nothing was saved.")
    held = float(portfolio["holdings"].get(ticker, 0.0))
    if shares - held > 1e-6:
        raise FundError(
            f"You hold {format_shares(held)} shares of {ticker}, "
            f"not {format_shares(shares)}. Nothing was saved."
        )
    sold = clean_shares(min(shares, held))
    proceeds = money(sold * price)
    remaining = clean_shares(held - sold)
    if remaining <= 1e-6:
        portfolio["holdings"].pop(ticker, None)
    else:
        portfolio["holdings"][ticker] = remaining
    portfolio["cash"] = money(money(portfolio["cash"]) + proceeds)
    portfolio["trades"].append(
        {
            "side": "sell",
            "ticker": ticker,
            "shares": sold,
            "price": clean_price(price),
            "amount": proceeds,
            "date": on.isoformat(),
        }
    )


def require_portfolio(portfolio: dict) -> None:
    missing = [
        key
        for key in ("cash", "starting_cash", "holdings", "spy", "trades")
        if key not in portfolio
    ]
    if missing:
        raise FundError(
            "Portfolio is missing " + ", ".join(missing) + ". It was not changed."
        )
    if not isinstance(portfolio["holdings"], dict):
        raise FundError("Portfolio holdings must be an object. It was not changed.")
    spy = portfolio["spy"]
    if not isinstance(spy, dict) or "shares" not in spy or "start_price" not in spy:
        raise FundError(
            "Portfolio spy benchmark needs shares and start_price. It was not changed."
        )
    if not isinstance(portfolio["trades"], list):
        raise FundError("Portfolio trades must be a list. It was not changed.")
    if float(portfolio["starting_cash"]) <= 0:
        raise FundError("starting_cash must be greater than zero. It was not changed.")


def summarize(portfolio: dict, prices: dict[str, float]) -> dict:
    rows = []
    holdings_value = 0.0
    for ticker, shares in portfolio["holdings"].items():
        if ticker not in prices:
            raise PriceError(
                f"Could not fetch a price for {ticker}. "
                "Check the ticker and your connection. Nothing was saved."
            )
        price = float(prices[ticker])
        value = float(shares) * price
        holdings_value += value
        rows.append(
            {"ticker": ticker, "shares": float(shares), "price": price, "value": value}
        )
    if "SPY" not in prices:
        raise PriceError(
            "Could not fetch a price for SPY. "
            "Check your connection. Nothing was saved."
        )
    cash = float(portfolio["cash"])
    total = holdings_value + cash
    start = float(portfolio["starting_cash"])
    spy_value = float(portfolio["spy"]["shares"]) * float(prices["SPY"])
    fish_return = percent_return(total, start)
    spy_return = percent_return(spy_value, start)
    return {
        "rows": rows,
        "cash": cash,
        "total": total,
        "starting_cash": start,
        "fish_return": fish_return,
        "spy_return": spy_return,
        "spy_value": spy_value,
        "ahead": ahead(fish_return, spy_return),
    }


def ahead(fish_return: float, spy_return: float) -> str:
    if round(fish_return, 2) == round(spy_return, 2):
        return "TIE"
    if fish_return > spy_return:
        return "FISH"
    return "SPY"


def render_status(summary: dict) -> str:
    header = f"{'HOLDING':<12}{'SHARES':>12}{'PRICE':>12}{'VALUE':>14}"
    lines = [header, "-" * len(header)]
    if summary["rows"]:
        for row in summary["rows"]:
            lines.append(
                f"{row['ticker']:<12}"
                f"{format_shares(row['shares']):>12}"
                f"{format_money(row['price']):>12}"
                f"{format_money(row['value']):>14}"
            )
    else:
        lines.append("(no holdings)")
    lines.append("")
    label, value = 22, 14
    lines.append(f"{'Cash':<{label}}{format_money(summary['cash']):>{value}}")
    lines.append(f"{'Total':<{label}}{format_money(summary['total']):>{value}}")
    lines.append(
        f"{'Fish return':<{label}}{format_pct(summary['fish_return']):>{value}}"
    )
    lines.append(f"{'S&P return':<{label}}{format_pct(summary['spy_return']):>{value}}")
    lines.append(f"{'Ahead':<{label}}{summary['ahead']:>{value}}")
    return "\n".join(lines)


def _portfolio_from_args(argv: list[str]) -> tuple[Path, list[str]]:
    path = PORTFOLIO_PATH
    rest: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--portfolio" or token.startswith("--portfolio="):
            if token.startswith("--portfolio="):
                value = token.split("=", 1)[1]
                index += 1
            else:
                if index + 1 >= len(argv):
                    die("--portfolio needs a file path.")
                value = argv[index + 1]
                index += 2
            if not value:
                die("--portfolio needs a file path.")
            path = Path(value)
            continue
        rest.append(token)
        index += 1
    return path, rest


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log fish-fund trades by hand and compare them with the S&P 500.",
        epilog=(
            "Pass --portfolio before or after the command. "
            "Example: python fund.py --portfolio /tmp/book.json status"
        ),
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=PORTFOLIO_PATH,
        help="Portfolio file. Default: portfolio.json next to this script.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Start a portfolio and a simulated SPY buy.")
    init.add_argument("--cash", type=float, required=True)

    buy_cmd = sub.add_parser("buy", help="Record a manual buy at the latest price.")
    buy_cmd.add_argument("ticker")
    buy_cmd.add_argument("--amount", type=float, required=True)

    sell_cmd = sub.add_parser("sell", help="Record a manual sale at the latest price.")
    sell_cmd.add_argument("ticker")
    sell_cmd.add_argument("--shares", type=float, required=True)

    sub.add_parser("status", help="Show holdings, cash, and fish versus the S&P 500.")
    return parser.parse_args(argv)


def _guard(action) -> None:
    try:
        action()
    except (FundError, PriceError) as exc:
        die(str(exc))


def _cmd_init(path: Path, cash: float) -> None:
    if path.exists():
        die(
            f"{path} already exists. Refusing to overwrite it. "
            "Pass --portfolio to use another file, or move this one aside."
        )
    if not math.isfinite(cash) or money(cash) <= 0:
        die("--cash must be greater than zero.")
    prices = latest_prices(["SPY"])
    portfolio = init_portfolio(cash, prices["SPY"], market_today())
    save_portfolio(path, portfolio)
    spy = portfolio["spy"]
    print(
        f"Started with {format_money(portfolio['starting_cash'])}. "
        f"SPY benchmark: {format_shares(spy['shares'])} shares at {format_money(spy['start_price'])}."
    )
    print(path.resolve())


def _cmd_buy(path: Path, ticker: str, amount: float) -> None:
    symbol = normalize_ticker(ticker)
    if symbol is None:
        die(f"Ticker {ticker!r} does not look like a stock symbol. Nothing was saved.")
    if not math.isfinite(amount) or money(amount) <= 0:
        die("--amount must be greater than zero. Nothing was saved.")
    portfolio = load_portfolio(path)
    require_portfolio(portfolio)
    if money(amount) - money(portfolio["cash"]) > 0:
        die(
            f"Amount {format_money(money(amount))} is more than cash "
            f"{format_money(money(portfolio['cash']))}. Nothing was saved."
        )
    prices = latest_prices([symbol])
    buy(portfolio, symbol, amount, prices[symbol], market_today())
    save_portfolio(path, portfolio)
    trade = portfolio["trades"][-1]
    print(
        f"Bought {format_shares(trade['shares'])} shares of {symbol} "
        f"at {format_money(trade['price'])} ({format_money(trade['amount'])}). "
        f"Cash left: {format_money(portfolio['cash'])}."
    )


def _cmd_sell(path: Path, ticker: str, shares: float) -> None:
    symbol = normalize_ticker(ticker)
    if symbol is None:
        die(f"Ticker {ticker!r} does not look like a stock symbol. Nothing was saved.")
    if not math.isfinite(shares) or shares <= 0:
        die("--shares must be greater than zero. Nothing was saved.")
    portfolio = load_portfolio(path)
    require_portfolio(portfolio)
    held = float(portfolio["holdings"].get(symbol, 0.0))
    if shares - held > 1e-6:
        die(
            f"You hold {format_shares(held)} shares of {symbol}, "
            f"not {format_shares(shares)}. Nothing was saved."
        )
    prices = latest_prices([symbol])
    sell(portfolio, symbol, shares, prices[symbol], market_today())
    save_portfolio(path, portfolio)
    trade = portfolio["trades"][-1]
    print(
        f"Sold {format_shares(trade['shares'])} shares of {symbol} "
        f"at {format_money(trade['price'])} ({format_money(trade['amount'])}). "
        f"Cash now: {format_money(portfolio['cash'])}."
    )


def _cmd_status(path: Path) -> None:
    portfolio = load_portfolio(path)
    require_portfolio(portfolio)
    tickers = list(portfolio["holdings"]) + ["SPY"]
    prices = latest_prices(tickers)
    print(render_status(summarize(portfolio, prices)))


def main(argv: list[str] | None = None) -> None:
    given = list(sys.argv[1:] if argv is None else argv)
    path, rest = _portfolio_from_args(given)
    args = _parse(rest)

    if args.command == "init":
        _guard(lambda: _cmd_init(path, args.cash))
    elif args.command == "buy":
        _guard(lambda: _cmd_buy(path, args.ticker, args.amount))
    elif args.command == "sell":
        _guard(lambda: _cmd_sell(path, args.ticker, args.shares))
    elif args.command == "status":
        _guard(lambda: _cmd_status(path))
    else:
        die("Use init, buy, sell, or status.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
