"""Film a zebrafish round. The busiest zone at the buzzer wins."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
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

PICK_FIELDS = ["timestamp", "round_type", "subject", "zone_counts", "winner"]
ZONE_COLORS = (
    (48, 77, 255),
    (80, 200, 255),
    (255, 160, 64),
    (120, 210, 140),
    (210, 110, 240),
)
INK = (16, 16, 18)
WHITE = (244, 241, 234)
GREY = (150, 150, 156)
FRAME_THRESHOLD = 28


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
        labels = ("10%", "25%", "50%")
        labeled = [
            Zone(label, zone.x, zone.y, zone.w, zone.h)
            for label, zone in zip(labels, zones[:3])
        ]
        return labeled, ""
    die("Round must be buy, hold, or size.")


def round_heading(round_type: str, subject: str) -> str:
    if round_type == "buy":
        return "BUY"
    if round_type == "hold":
        return f"HOLD OR SELL  {subject}"
    return "SIZE"


def fit_scale(text: str, max_width: int, max_scale: float, thickness: int = 2) -> float:
    scale = max_scale
    while scale > 0.4:
        (width, _height), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if width <= max_width:
            return scale
        scale -= 0.05
    return 0.4


def draw_chip(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    fg: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    (width, height), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad_x, pad_y = 12, 8
    x = min(max(int(x), 0), max(0, image.shape[1] - width - pad_x * 2))
    y = min(max(int(y), 0), max(0, image.shape[0] - height - base - pad_y * 2))
    cv2.rectangle(
        image,
        (x, y),
        (x + width + pad_x * 2, y + height + base + pad_y * 2),
        INK,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + pad_x, y + height + pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        fg,
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    zones: list[Zone],
    counts: list[int],
    centroids: list[tuple[int, int]],
    owners: list[int | None],
    heading: str,
    seconds_left: int,
) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]
    for index, zone in enumerate(zones):
        color = ZONE_COLORS[index % len(ZONE_COLORS)]
        cv2.rectangle(
            display,
            (zone.x, zone.y),
            (zone.x + zone.w, zone.y + zone.h),
            color,
            max(2, width // 480),
        )
        label_scale = fit_scale(zone.ticker, max(40, zone.w - 20), 0.95)
        draw_chip(display, zone.ticker, zone.x + 8, zone.y + 8, label_scale, WHITE)
        count_text = str(counts[index])
        count_scale = fit_scale(count_text, max(24, zone.w // 2), min(2.2, max(zone.h, 1) / 70))
        (tw, th), _base = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, count_scale, 2)
        draw_chip(
            display,
            count_text,
            zone.x + zone.w // 2 - tw // 2,
            zone.y + zone.h // 2 - th // 2,
            count_scale,
            color,
        )
    for (x, y), owner in zip(centroids, owners):
        color = GREY if owner is None else ZONE_COLORS[owner % len(ZONE_COLORS)]
        cv2.circle(display, (x, y), 8, color, 2, cv2.LINE_AA)

    bar_h = max(72, height // 8)
    cv2.rectangle(display, (0, 0), (width, bar_h), INK, -1)
    head_scale = fit_scale(heading, int(width * 0.52), 1.05)
    cv2.putText(
        display,
        heading,
        (24, int(bar_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        head_scale,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    timer = str(seconds_left)
    timer_scale = fit_scale(timer, int(width * 0.22), 2.1, thickness=3)
    (tw, _th), _base = cv2.getTextSize(timer, cv2.FONT_HERSHEY_SIMPLEX, timer_scale, 3)
    cv2.putText(
        display,
        timer,
        (width - tw - 28, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        timer_scale,
        WHITE,
        3,
        cv2.LINE_AA,
    )
    return display


def draw_banner(frame: np.ndarray, label: str) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]
    veil = np.full_like(display, INK)
    cv2.addWeighted(veil, 0.78, display, 0.22, 0, display)
    text = f"FISH PICKED: {label}"
    scale = fit_scale(text, int(width * 0.88), max(1.2, width / 460), thickness=4)
    (tw, th), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    cv2.putText(
        display,
        text,
        ((width - tw) // 2, (height + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        ZONE_COLORS[0],
        4,
        cv2.LINE_AA,
    )
    return display


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


def _read_frame(capture, width: int, height: int, recording: Path) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        die(
            f"The webcam stopped returning frames. Partial recording kept at {recording}. "
            "No pick was logged."
        )
    frame = even_frame(frame)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    return frame


def run_round(config: Config, config_path: Path, round_type: str, ticker: str) -> None:
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
    try:
        capture, frame = open_camera(config.camera_index)
        frame = even_frame(frame)
        height, width = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps = 30.0
        recording = RECORDINGS_DIR / f"{file_stamp()}.mp4"
        writer = open_writer(recording, width, height, fps)
        window = "fish-fund pick | q cancels"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        heading = round_heading(round_type, subject)
        samples: list[tuple[float, list[int]]] = []
        start = time.monotonic()
        winner: int | None = None
        banner_until = 0.0
        first = True
        cancelled = False
        while True:
            if not first:
                frame = _read_frame(capture, width, height, recording)
            first = False
            if detector is not None:
                mask = detector.mask(frame)
            elif background is not None:
                mask = frame_mask(frame, background)
            else:
                die("background_mode is frame but no background image is loaded.")
            centroids = centroids_from_mask(mask, config.min_blob_area, config.max_blob_area)
            counts, owners = count_zones(centroids, zones)
            elapsed = time.monotonic() - start
            if winner is None:
                samples.append((elapsed, list(counts)))
                if elapsed >= config.countdown_seconds:
                    winner = choose_winner(samples)
                    banner_until = time.monotonic() + config.banner_seconds
            if winner is None:
                seconds_left = max(1, math.ceil(config.countdown_seconds - elapsed - 1e-9))
                display = draw_overlay(
                    frame, zones, counts, centroids, owners, heading, seconds_left
                )
            else:
                display = draw_banner(frame, zones[winner].ticker)
            wrote = writer.write(display)
            if wrote is False:
                writer.release()
                writer = None
                recording.unlink(missing_ok=True)
                die(f"Could not write a frame to {recording}. The recording was not saved.")
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if winner is None and key in (ord("q"), 27):
                cancelled = True
                break
            if winner is not None and (
                time.monotonic() >= banner_until or key in (ord("q"), 27)
            ):
                break
        if winner is None or recording is None:
            if cancelled and recording is not None:
                print(
                    f"Round cancelled. Partial recording kept at {recording.resolve()}. "
                    "No row was added to picks.csv."
                )
            return
        final_counts = samples[-1][1]
        label = zones[winner].ticker
        log_pick(round_type, subject, zones, final_counts, label)
        print(f"FISH PICKED: {label}")
        print(", ".join(f"{zone.ticker} {count}" for zone, count in zip(zones, final_counts)))
        print(f"Recording: {recording.resolve()}")
        print(f"Logged: {PICKS_PATH.resolve()}")
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
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
    parser.add_argument("--round", dest="round_type", choices=("buy", "hold", "size"))
    parser.add_argument("--ticker", default="", help="Stock for a hold round, such as AAPL.")
    args = parser.parse_args(argv)
    if args.capture_background and args.round_type:
        die("Use either --capture-background or --round, not both.")
    config = load_config(args.config)
    if args.capture_background:
        capture_background(config, args.config)
        return
    if not args.round_type:
        die(
            "Pass --round buy, --round hold --ticker AAPL, --round size, "
            "or --capture-background."
        )
    if args.ticker and args.round_type != "hold":
        die("--ticker is only used with --round hold.")
    run_round(config, args.config, args.round_type, args.ticker)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
