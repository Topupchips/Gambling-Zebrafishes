"""Film a zebrafish round. The busiest zone at the buzzer wins."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

from utils import (
    CONFIG_PATH,
    PICKS_PATH,
    RECORDINGS_DIR,
    Config,
    Zone,
    append_csv,
    die,
    file_stamp,
    iso_timestamp,
    load_config,
    normalize_ticker,
    open_camera,
    resolve_background,
    update_config,
)
import demo as real_demo
import pixel_tank
from hud import LAYOUTS, Hud, Layout, Quote, RoundInfo

PICK_FIELDS = ["timestamp", "round_type", "subject", "zone_counts", "winner"]
FRAME_THRESHOLD = 28
WRITER_FPS = 30.0
ROUND_LABELS = {
    "hold": ("HOLD", "SELL"),
    "size": ("10%", "25%", "50%"),
    "poker": ("FOLD", "CALL", "RAISE"),
}
DEMO_TICKERS = ("NVDA", "SPY", "TSLA")


def choose_winner(
    samples: list[tuple[float, list[int]]],
    rng: random.Random | None = None,
) -> int:
    """Pick the zone with the most fish on the last sample.

    Ties use the highest average count over the final 3 seconds.
    Zones that are still tied are chosen at random.
    ``samples`` are ``(seconds_from_start, counts)`` in time order.
    """
    if not samples:
        raise ValueError("choose_winner needs at least one sample")
    width = len(samples[0][1])
    if width == 0:
        raise ValueError("choose_winner needs at least one zone")
    for _stamp, counts in samples:
        if len(counts) != width:
            raise ValueError("every sample must cover the same zones")
    end_t, final = samples[-1]
    leaders = [index for index, count in enumerate(final) if count == max(final)]
    if len(leaders) == 1:
        return leaders[0]
    window = [counts for stamp, counts in samples if stamp >= end_t - 3.0]
    if not window:
        window = [final]
    averages = [
        sum(counts[index] for counts in window) / len(window) for index in leaders
    ]
    best_average = max(averages)
    still = [
        index
        for index, average in zip(leaders, averages)
        if math.isclose(average, best_average, abs_tol=1e-9)
    ]
    if len(still) == 1:
        return still[0]
    picker = rng if rng is not None else random.SystemRandom()
    return int(picker.choice(still))


def _kernel(size: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(3))
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _kernel(5))


def frame_mask(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    background_bgr = background
    if background_bgr.ndim == 2:
        background_bgr = cv2.cvtColor(background_bgr, cv2.COLOR_GRAY2BGR)
    elif background_bgr.shape[2] == 4:
        background_bgr = cv2.cvtColor(background_bgr, cv2.COLOR_BGRA2BGR)
    if background_bgr.shape[:2] != frame.shape[:2]:
        background_bgr = cv2.resize(
            background_bgr,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(cv2.absdiff(frame, background_bgr), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _level, mask = cv2.threshold(gray, FRAME_THRESHOLD, 255, cv2.THRESH_BINARY)
    return clean_mask(mask)


class MotionDetector:
    """MOG2 subtractor. It learns the tank, then keeps a low learning rate."""

    def __init__(self) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=16, detectShadows=False
        )
        self.seen = 0

    def mask(self, frame: np.ndarray) -> np.ndarray:
        rate = 0.35 if self.seen < 20 else 0.004
        self.seen += 1
        raw = self.subtractor.apply(frame, learningRate=rate)
        _level, binary = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        return clean_mask(binary)


def centroids_from_mask(
    mask: np.ndarray, min_area: float, max_area: float
) -> list[tuple[int, int]]:
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    points: list[tuple[int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        points.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])))
    return points


def count_zones(
    centroids: list[tuple[int, int]], zones: list[Zone]
) -> tuple[list[int], list[int | None]]:
    counts = [0] * len(zones)
    owners: list[int | None] = []
    for x, y in centroids:
        owner = None
        for index, zone in enumerate(zones):
            if zone.contains(x, y):
                counts[index] += 1
                owner = index
                break
        owners.append(owner)
    return counts, owners


def round_zones(config: Config, round_type: str, ticker: str) -> tuple[list[Zone], str]:
    zones = config.zones
    if round_type == "buy":
        if len(zones) < 1:
            die(
                "A buy round needs at least 1 zone. "
                "Run python calibrate.py and draw one zone per ticker."
            )
        return list(zones), ""
    if round_type == "hold":
        if len(zones) < 2:
            die(
                "A hold round needs at least 2 zones. "
                "The first zone you drew is HOLD and the second is SELL. "
                "Run python calibrate.py."
            )
        if not ticker.strip():
            die("A hold round needs --ticker, for example: python pick.py --round hold --ticker AAPL")
        symbol = normalize_ticker(ticker)
        if symbol is None:
            die(f"Ticker {ticker!r} does not look like a stock symbol.")
        first, second = zones[0], zones[1]
        labeled = [
            Zone("HOLD", first.x, first.y, first.w, first.h),
            Zone("SELL", second.x, second.y, second.w, second.h),
        ]
        return labeled, symbol
    if round_type == "size":
        if len(zones) < 3:
            die(
                "A size round needs at least 3 zones, drawn in order 10%, 25%, 50%. "
                "Run python calibrate.py."
            )
        labeled = [
            Zone(label, zone.x, zone.y, zone.w, zone.h)
            for label, zone in zip(ROUND_LABELS["size"], zones[:3])
        ]
        return labeled, ""
    if round_type == "poker":
        if len(zones) < 3:
            die(
                "A poker round needs at least 3 zones, drawn in order FOLD, CALL, RAISE. "
                "Run python calibrate.py, or rehearse with --demo."
            )
        labeled = [
            Zone(label, zone.x, zone.y, zone.w, zone.h)
            for label, zone in zip(ROUND_LABELS["poker"], zones[:3])
        ]
        return labeled, ""
    die("Round must be buy, hold, size, or poker.")


def even_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    even_w = width - (width % 2)
    even_h = height - (height % 2)
    if even_w == width and even_h == height:
        return frame
    return frame[:even_h, :even_w].copy()


def open_writer(path: Path, width: int, height: int, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        path.unlink(missing_ok=True)
        die(
            f"Could not open the video writer for {path}. "
            "The mp4v codec is unavailable, so this round was not recorded."
        )
    return writer


def load_background(config: Config, config_path: Path) -> np.ndarray:
    path = resolve_background(config, config_path)
    if path is None or not path.exists():
        shown = path if path is not None else "background_image"
        die(
            f"background_mode is frame but no empty-tank image was found at {shown}. "
            "Run: python pick.py --capture-background"
        )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        die(f"Could not read the background image at {path}.")
    return image


def log_pick(
    round_type: str,
    subject: str,
    zones: list[Zone],
    counts: list[int],
    winner: str,
    path: Path = PICKS_PATH,
) -> None:
    append_csv(
        path,
        PICK_FIELDS,
        {
            "timestamp": iso_timestamp(),
            "round_type": round_type,
            "subject": subject,
            "zone_counts": "|".join(
                f"{zone.ticker}:{count}" for zone, count in zip(zones, counts)
            ),
            "winner": winner,
        },
    )


def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = even_frame(frame)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    return frame


def open_video(path: Path):
    """Open a video file and return (capture, first_frame). Exits if it cannot."""
    if not path.exists():
        die(f"Could not open video {path}. The file is not there.")
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    capture = cv2.VideoCapture(str(path))
    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        die(f"Could not open video {path}. It is not a readable video.")
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        die(f"Video {path} opened but did not return a frame.")
    return capture, frame


def _read_frame(capture, width: int, height: int, recording: Path) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        die(
            f"The webcam stopped returning frames. Partial recording kept at {recording}. "
            "No pick was logged."
        )
    return _fit_frame(frame, width, height)


def _read_video_frame(capture, width: int, height: int, recording: Path, path: Path) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    if not ok or frame is None:
        die(
            f"Video {path} stopped returning frames. Partial recording kept at {recording}. "
            "No pick was logged."
        )
    return _fit_frame(frame, width, height)


def phone_copy(recording: Path, crf: int = 18) -> Path | None:
    """Save an H.264 copy that Instagram, TikTok and phones accept. Needs ffmpeg on PATH."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None
    target = recording.with_name(recording.stem + "-h264.mp4")
    result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(recording), "-c:v", "libx264", "-crf", str(crf),
         "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        check=False,
    )
    return target if result.returncode == 0 and target.exists() else None


def count_rounds(path: Path = PICKS_PATH) -> int:
    if not path.exists():
        return 0
    with path.open() as handle:
        return max(0, sum(1 for line in handle if line.strip()) - 1)


def fetch_quotes(symbols: list[str]) -> dict[str, Quote]:
    """Latest close and day change for each symbol. Empty on any failure."""
    from datetime import timedelta

    from utils import PriceError, daily_closes, market_today

    symbols = [s for s in dict.fromkeys(symbols) if normalize_ticker(s)]
    if not symbols:
        return {}
    today = market_today()
    try:
        closes = daily_closes(symbols, today - timedelta(days=14), today)
    except (PriceError, Exception):  # noqa: BLE001 - graphics must never stop a round
        return {}
    quotes: dict[str, Quote] = {}
    for symbol, history in closes.items():
        days = sorted(history)
        last = history[days[-1]]
        change = None
        if len(days) > 1 and history[days[-2]]:
            change = (last / history[days[-2]] - 1) * 100
        quotes[symbol] = Quote(price=last, change_pct=change)
    return quotes


class QuoteLoader:
    """Fetch prices on a thread so the round starts immediately."""

    def __init__(self, symbols: list[str]) -> None:
        self.result: dict[str, Quote] | None = None
        self.delivered = False
        if not symbols:
            self.result = {}
            return
        self.thread = threading.Thread(target=self._run, args=(symbols,), daemon=True)
        self.thread.start()

    def _run(self, symbols: list[str]) -> None:
        self.result = fetch_quotes(symbols)

    def take(self) -> dict[str, Quote] | None:
        if self.delivered or self.result is None:
            return None
        self.delivered = True
        return self.result


def quote_symbols(round_type: str, zones: list[Zone], subject: str) -> list[str]:
    if round_type == "buy":
        return [zone.ticker for zone in zones]
    if round_type == "hold" and subject:
        return [subject]
    return []


def preview_size(layout: Layout, scale: float | None) -> tuple[int, int]:
    w, h = layout.size
    if scale is None:
        scale = 900 / h if layout.name == "reel" else 1280 / w
    return max(1, int(w * scale)), max(1, int(h * scale))


def run_round(
    config: Config,
    config_path: Path,
    round_type: str,
    ticker: str,
    video: Path | None = None,
    *,
    demo: bool = False,
    demo_style: str = "pixel",
    layout_name: str = "reel",
    note: str = "",
    prices: bool = True,
    preview: bool = True,
    preview_scale: float | None = None,
    footer: str | None = None,
) -> Path | None:
    layout = LAYOUTS[layout_name]
    if demo:
        if round_type == "buy":
            labels = [zone.ticker for zone in config.zones] or list(DEMO_TICKERS)
        else:
            labels = list(ROUND_LABELS[round_type])
        tank_module = pixel_tank if demo_style == "pixel" else real_demo
        zones = tank_module.demo_zones(labels)
        subject = (normalize_ticker(ticker) or "NVDA") if round_type == "hold" else ""
        background = None  # set from the tank's empty frame below
        detector = None
    else:
        zones, subject = round_zones(config, round_type, ticker)
        background = None
        detector = None
        if config.background_mode == "frame":
            background = load_background(config, config_path)
        else:
            detector = MotionDetector()

    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

    capture = None
    writer = None
    recording: Path | None = None
    window = "fish-fund | q cancels"
    try:
        if demo:
            capture = (pixel_tank.PixelTank if demo_style == "pixel" else real_demo.DemoTank)(zones, float(config.countdown_seconds))
            # The simulated tank knows exactly what it looks like empty, like --capture-background.
            background = capture.empty_frame
            _ok, frame = capture.read()
        elif video is None:
            capture, frame = open_camera(config.camera_index)
        else:
            capture, frame = open_video(video)
        frame = even_frame(frame)
        height, width = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps = 30.0

        info = RoundInfo(
            round_type=round_type,
            labels=[zone.ticker for zone in zones],
            subject=subject,
            note=note,
            number=count_rounds() + 1,
            countdown=float(config.countdown_seconds),
            reveal_seconds=float(config.banner_seconds),
        )
        if footer is not None:
            info.footer = footer
        start = time.monotonic()
        frames_done = 0

        def clock() -> float:
            # Demo rounds run on a frame clock: every recorded frame is a fresh render at a
            # steady 30 fps, however long each one takes to draw.
            return start + frames_done / WRITER_FPS if demo else time.monotonic()

        if demo:
            info.camera_label = "SIMULATED TANK"
        hud = Hud(layout, info, (width, height), start, pixel_art=bool(getattr(capture, "pixel_art", False)))
        quotes = QuoteLoader(quote_symbols(round_type, zones, subject) if prices else [])

        prefix = "demo-" if demo else ""
        recording = RECORDINGS_DIR / f"{prefix}{file_stamp()}.mp4"
        out_w, out_h = layout.size
        writer = open_writer(recording, out_w, out_h, WRITER_FPS)
        if preview:
            pw, ph = preview_size(layout, preview_scale)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, pw, ph)
            cv2.moveWindow(window, 60, 30)
        wait_ms = 1 if video is None else max(1, int(round(1000 / fps)))
        samples: list[tuple[float, list[int]]] = []
        winner: int | None = None
        banner_until = 0.0
        written = 0
        first = True
        cancelled = False
        while True:
            if not first:
                if demo:
                    _ok, frame = capture.read()
                elif video is None:
                    frame = _read_frame(capture, width, height, recording)
                else:
                    frame = _read_video_frame(capture, width, height, recording, video)
            first = False
            # The demo tank hands over a plain pass for detection, so its lighting never counts as fish.
            seen = capture.detection_frame if demo else frame
            if detector is not None:
                mask = detector.mask(seen)
            elif background is not None:
                mask = frame_mask(seen, background)
            else:
                die("background_mode is frame but no background image is loaded.")
            centroids = centroids_from_mask(mask, config.min_blob_area, config.max_blob_area)
            counts, owners = count_zones(centroids, zones)
            now = clock()
            elapsed = now - start
            fresh = quotes.take()
            if fresh:
                hud.set_quotes(fresh)
            if winner is None:
                samples.append((elapsed, list(counts)))
                if elapsed >= config.countdown_seconds:
                    winner = choose_winner(samples)
                    hud.declare_winner(winner, samples[-1][1], now)
                    banner_until = now + config.banner_seconds
            display = hud.frame(frame, zones, counts, centroids, owners, now)

            # Keep the recording in real time even when a frame takes long to draw.
            due = int((clock() - start) * WRITER_FPS) + 1
            repeats = 1 if demo else max(1, min(4, due - written))
            for _ in range(repeats):
                if writer.write(display) is False:
                    writer.release()
                    writer = None
                    recording.unlink(missing_ok=True)
                    die(f"Could not write a frame to {recording}. The recording was not saved.")
            written += repeats
            frames_done += 1

            key = 255
            if preview:
                cv2.imshow(window, cv2.resize(display, preview_size(layout, preview_scale), interpolation=cv2.INTER_AREA))
                key = cv2.waitKey(wait_ms) & 0xFF
            elif video is not None:
                time.sleep(wait_ms / 1000)
            if winner is None and key in (ord("q"), 27):
                cancelled = True
                break
            if winner is not None and (clock() >= banner_until or key in (ord("q"), 27)):
                break
        if winner is None or recording is None:
            if cancelled and recording is not None:
                print(
                    f"Round cancelled. Partial recording kept at {recording.resolve()}. "
                    "No row was added to picks.csv."
                )
            return None
        final_counts = samples[-1][1]
        label = zones[winner].ticker
        print(f"FISH PICKED: {label}")
        print(", ".join(f"{zone.ticker} {count}" for zone, count in zip(zones, final_counts)))
        print(f"Recording: {recording.resolve()}")
        if writer is not None:
            writer.release()
            writer = None
        phone = phone_copy(recording)
        if phone is not None:
            print(f"Phone-ready copy: {phone.resolve()}")
        else:
            print("Install ffmpeg to also get an H.264 copy for Instagram and TikTok.")
        if demo:
            print("Demo round: nothing was logged.")
        else:
            log_pick(round_type, subject, zones, final_counts, label)
            print(f"Logged: {PICKS_PATH.resolve()}")
        return recording
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        if preview:
            cv2.destroyAllWindows()


def capture_background(config: Config, config_path: Path) -> None:
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    capture = None
    try:
        capture, frame = open_camera(config.camera_index)
        for _ in range(12):
            ok, nxt = capture.read()
            if ok and nxt is not None:
                frame = nxt
    finally:
        if capture is not None:
            capture.release()
    path = config_path.parent / "background.png"
    if not cv2.imwrite(str(path), frame):
        die(f"Could not save the background image to {path}. config.json was not changed.")
    update_config(
        config_path,
        {"background_mode": "frame", "background_image": path.name},
    )
    print(f"Saved an empty-tank frame to {path.resolve()}.")
    print('config.json now uses background_mode "frame". Put the fish back in, then run a round.')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Film a zebrafish pick round.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to config.json.")
    parser.add_argument(
        "--capture-background",
        action="store_true",
        help="Save one empty-tank frame and switch config to frame mode.",
    )
    parser.add_argument("--round", dest="round_type", choices=("buy", "hold", "size", "poker"))
    parser.add_argument("--ticker", default="", help="Stock for a hold round, such as AAPL.")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Play this video through the overlay instead of the webcam.",
    )
    parser.add_argument("--demo", action="store_true", help="Rehearse with a simulated tank. Nothing is logged.")
    parser.add_argument("--demo-style", choices=("pixel", "real"), default="pixel", help="pixel = pixel-art aquarium (default), real = realistic top-down tank.")
    parser.add_argument("--layout", choices=tuple(LAYOUTS), default="reel", help="reel = 1080x1920 vertical (default), wide = 1920x1080.")
    parser.add_argument("--note", default="", help='Line shown under the question, e.g. "POT: 40 CHIPS".')
    parser.add_argument("--footer", default=None, help="Replace the small line at the bottom. Pass \"\" to hide it.")
    parser.add_argument("--no-prices", action="store_true", help="Skip the live price lookup.")
    parser.add_argument("--no-preview", action="store_true", help="Record without opening a window.")
    parser.add_argument("--preview-scale", type=float, default=None, help="Preview window size, e.g. 0.5.")
    args = parser.parse_args(argv)
    if args.capture_background and args.round_type:
        die("Use either --capture-background or --round, not both.")
    if args.capture_background and (args.video is not None or args.demo):
        die("--capture-background needs the real webcam. Drop --video and --demo.")
    if args.demo and args.video is not None:
        die("Use either --demo or --video, not both.")
    config = load_config(args.config)
    if args.capture_background:
        capture_background(config, args.config)
        return
    if not args.round_type:
        die(
            "Pass --round buy, --round hold --ticker AAPL, --round size, --round poker, "
            "or --capture-background. Add --demo to rehearse without a camera."
        )
    if args.ticker and args.round_type != "hold":
        die("--ticker is only used with --round hold.")
    run_round(
        config,
        args.config,
        args.round_type,
        args.ticker,
        args.video,
        demo=args.demo,
        demo_style=args.demo_style,
        layout_name=args.layout,
        note=args.note,
        prices=not args.no_prices,
        preview=not args.no_preview,
        preview_scale=args.preview_scale,
        footer=args.footer,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
