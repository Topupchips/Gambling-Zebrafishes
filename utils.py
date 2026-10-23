"""Shared config, files, prices, and formatting for fish-fund."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
PORTFOLIO_PATH = ROOT / "portfolio.json"
PICKS_PATH = ROOT / "picks.csv"
RECORDINGS_DIR = ROOT / "recordings"
CHARTS_DIR = ROOT / "charts"
BACKGROUND_PATH = ROOT / "background.png"

MARKET_TZ = ZoneInfo("America/New_York")
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

DEFAULT_CONFIG: dict = {
    "camera_index": 0,
    "countdown_seconds": 15,
    "banner_seconds": 7,
    "min_blob_area": 100,
    "max_blob_area": 8000,
    "background_mode": "mog2",
    "background_image": "",
    "zones": [],
}


class PriceError(Exception):
    """A market price could not be fetched. Callers leave data files untouched."""


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def file_stamp(moment: datetime | None = None) -> str:
    moment = moment or utc_now()
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def iso_timestamp(moment: datetime | None = None) -> str:
    moment = moment or utc_now()
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def market_today() -> date:
    return datetime.now(MARKET_TZ).date()


def format_date(day: date) -> str:
    return day.isoformat()


def format_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def format_pct(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:+.2f}%"


def format_shares(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def money(value: float) -> float:
    return round(float(value) * 100) / 100.0


def clean_shares(value: float) -> float:
    return round(float(value), 8)


def clean_price(value: float) -> float:
    return round(float(value), 4)


def percent_return(current: float, start: float) -> float:
    if start == 0:
        raise ValueError("starting cash is zero")
    return (current - start) / start * 100.0


def normalize_ticker(value: str) -> str | None:
    symbol = value.strip().upper()
    if not TICKER_RE.fullmatch(symbol):
        return None
    return symbol


@dataclass(frozen=True)
class Zone:
    ticker: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
        }


@dataclass
class Config:
    camera_index: int = 0
    countdown_seconds: int = 15
    banner_seconds: int = 7
    min_blob_area: int = 100
    max_blob_area: int = 8000
    background_mode: str = "mog2"
    background_image: str = ""
    zones: list[Zone] = field(default_factory=list)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _read_json_object(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        die(f"{path} is not valid JSON ({exc.lineno}:{exc.colno}).")
    if not isinstance(raw, dict):
        die(f"{path} must contain a JSON object.")
    return raw


def _require_int(data: dict, key: str, minimum: int) -> int:
    try:
        value = int(data[key])
    except (KeyError, TypeError, ValueError):
        die(f"config.json field {key} must be a whole number.")
    if value < minimum:
        die(f"config.json field {key} must be at least {minimum}.")
    return value


def _parse_zones(raw_zones: object) -> list[Zone]:
    if not isinstance(raw_zones, list):
        die("config.json zones must be a list.")
    zones: list[Zone] = []
    for index, item in enumerate(raw_zones, start=1):
        if not isinstance(item, dict):
            die(f"Zone {index} in config.json must be an object.")
        try:
            ticker = str(item["ticker"]).strip().upper()
            zone = Zone(
                ticker=ticker,
                x=int(item["x"]),
                y=int(item["y"]),
                w=int(item["w"]),
                h=int(item["h"]),
            )
        except (KeyError, TypeError, ValueError):
            die(f"Zone {index} in config.json needs ticker, x, y, w, and h.")
        if not ticker:
            die(f"Zone {index} in config.json has a blank ticker.")
        if zone.w <= 0 or zone.h <= 0:
            die(f"Zone {index} in config.json has no area.")
        zones.append(zone)
    return zones


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    raw = _read_json_object(path) if path.exists() else {}
    merged = dict(DEFAULT_CONFIG)
    merged.update(raw)
    if merged.get("background_image") is None:
        merged["background_image"] = ""
    mode = str(merged.get("background_mode", "mog2")).strip().lower()
    if mode not in ("mog2", "frame"):
        die('background_mode in config.json must be "mog2" or "frame".')
    min_area = _require_int(merged, "min_blob_area", 0)
    max_area = _require_int(merged, "max_blob_area", 1)
    if max_area <= min_area:
        die("min_blob_area must be less than max_blob_area in config.json.")
    return Config(
        camera_index=_require_int(merged, "camera_index", 0),
        countdown_seconds=_require_int(merged, "countdown_seconds", 1),
        banner_seconds=_require_int(merged, "banner_seconds", 1),
        min_blob_area=min_area,
        max_blob_area=max_area,
        background_mode=mode,
        background_image=str(merged.get("background_image", "")),
        zones=_parse_zones(merged.get("zones", [])),
    )


def update_config(path: Path, updates: dict) -> Config:
    raw = _read_json_object(path) if path.exists() else {}
    raw.update(updates)
    merged = dict(DEFAULT_CONFIG)
    merged.update(raw)
    text = json.dumps(merged, indent=2) + "\n"
    atomic_write_text(path, text)
    return load_config(path)


def save_zones(path: Path, zones: list[Zone]) -> Config:
    return update_config(path, {"zones": [zone.as_dict() for zone in zones]})


def resolve_background(config: Config, config_path: Path) -> Path | None:
    if not config.background_image:
        return None
    path = Path(config.background_image)
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def append_csv(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_portfolio(path: Path | None = None) -> dict:
    path = path or PORTFOLIO_PATH
    if not path.exists():
        die(
            f"No portfolio at {path}. "
            "Start one with: python fund.py init --cash 200"
        )
    return _read_json_object(path)


def save_portfolio(path: Path, portfolio: dict) -> None:
    atomic_write_text(path, json.dumps(portfolio, indent=2) + "\n")


def _quiet(func):
    import contextlib
    import io
    import warnings

    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func()


def _extract_close(frame, symbol: str):
    import pandas as pd

    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    columns = frame.columns
    if isinstance(columns, pd.MultiIndex):
        level0 = set(map(str, columns.get_level_values(0)))
        if "Close" in level0:
            close = frame["Close"]
            if isinstance(close, pd.Series):
                return close
            for column in close.columns:
                if str(column).upper() == symbol:
                    return close[column]
        level_names = set(map(str, columns.get_level_values(0)))
        if symbol in level_names:
            block = frame[symbol]
            if isinstance(block, pd.DataFrame) and "Close" in block.columns:
                return block["Close"]
            if isinstance(block, pd.Series):
                return block
    if "Close" in frame.columns:
        close = frame["Close"]
        if isinstance(close, pd.Series):
            return close
    return None


def _series_to_dates(series) -> dict[date, float]:
    import pandas as pd

    index = series.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        index = index.tz_convert(MARKET_TZ)
    prices: dict[date, float] = {}
    for stamp, value in zip(index, series.to_numpy()):
        if pd.isna(value):
            continue
        day = stamp.date() if hasattr(stamp, "date") else pd.Timestamp(stamp).date()
        price = float(value)
        if math.isfinite(price) and price > 0:
            prices[day] = price
    return prices


def daily_closes(
    tickers: list[str], start: date, end: date
) -> dict[str, dict[date, float]]:
    """Daily adjusted closes keyed by market date, from start through end inclusive."""
    symbols: list[str] = []
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        return {}
    if end < start:
        start, end = end, start

    def fetch():
        import yfinance as yf

        request = symbols if len(symbols) > 1 else symbols[0]
        return yf.download(
            request,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )

    try:
        frame = _quiet(fetch)
    except Exception as exc:
        raise PriceError(
            "Could not download prices for "
            + ", ".join(symbols)
            + f". Check the ticker and your connection. Nothing was saved. ({exc})"
        ) from exc

    closes: dict[str, dict[date, float]] = {}
    missing: list[str] = []
    for symbol in symbols:
        series = _extract_close(frame, symbol)
        mapped = _series_to_dates(series) if series is not None else {}
        if not mapped:
            missing.append(symbol)
        else:
            closes[symbol] = mapped
    if missing:
        raise PriceError(
            "Could not fetch a price for "
            + ", ".join(missing)
            + ". Check the ticker and your connection. Nothing was saved."
        )
    return closes


def latest_prices(tickers: list[str]) -> dict[str, float]:
    """Latest available close for each ticker. Raises PriceError and saves nothing."""
    today = market_today()
    closes = daily_closes(tickers, today - timedelta(days=14), today)
    prices: dict[str, float] = {}
    for symbol, history in closes.items():
        latest_day = max(history)
        prices[symbol] = clean_price(history[latest_day])
    return prices


def open_camera(index: int):
    """Open a webcam and return (capture, first_frame). Exits if it cannot."""
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
    try:
        import cv2
    except ImportError:
        die("opencv-python is not installed. Run: pip install -r requirements.txt")
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        capture = cv2.VideoCapture(int(index))
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            die(_camera_message(index))
        ok, frame = capture.read()
    except SystemExit:
        raise
    except Exception as exc:
        die(f"Could not open webcam (camera_index={index}): {exc}")
    if not ok or frame is None:
        capture.release()
        die(
            f"Webcam (camera_index={index}) opened but did not return a frame. "
            "Check the cable, camera_index in config.json, and that no other app is using it."
        )
    return capture, frame


def _camera_message(index: int) -> str:
    return (
        f"Could not open webcam (camera_index={index}). "
        "Plug in the camera, close other apps that are using it, "
        "and check camera_index in config.json."
    )
