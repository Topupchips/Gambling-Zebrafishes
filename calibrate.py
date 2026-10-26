"""Draw tank zones on the webcam and save them into config.json."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2

from utils import CONFIG_PATH, Zone, die, load_config, normalize_ticker, open_camera, save_zones

WINDOW = "fish-fund calibrate | drag zone, u undo, c clear, s save, q quit"
HINT = "drag zone    u undo    c clear    s save    q quit"
MIN_DRAG = 8
INK = (22, 20, 18)
WHITE = (244, 241, 234)
CORAL = (48, 77, 255)


def image_point(window: str, x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Map a click on a resized preview back to camera pixels."""
    try:
        _rx, _ry, shown_w, shown_h = cv2.getWindowImageRect(window)
    except cv2.error:
        shown_w, shown_h = width, height
    if shown_w <= 0 or shown_h <= 0:
        shown_w, shown_h = width, height
    ix = int(round(x * width / shown_w))
    iy = int(round(y * height / shown_h))
    return min(max(ix, 0), width - 1), min(max(iy, 0), height - 1)


def normalized_rect(
    anchor: tuple[int, int],
    current: tuple[int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, right = sorted((anchor[0], current[0]))
    top, bottom = sorted((anchor[1], current[1]))
    left = min(max(left, 0), width - 1)
    top = min(max(top, 0), height - 1)
    right = min(max(right, left + 1), width)
    bottom = min(max(bottom, top + 1), height)
    return left, top, right - left, bottom - top


def fit_scale(text: str, max_width: int, max_scale: float) -> float:
    scale = max_scale
    while scale > 0.4:
        (width, _height), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if width <= max_width:
            return scale
        scale -= 0.05
    return 0.4


class Drawer:
    def __init__(self, window: str) -> None:
        self.window = window
        self.width = 0
        self.height = 0
        self.drawing = False
        self.anchor: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.pending: tuple[int, int, int, int] | None = None
        self.asking = False

    def set_size(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.pending is not None or self.asking or self.width <= 0:
            return
        point = image_point(self.window, x, y, self.width, self.height)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.anchor = point
            self.current = point
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = point
        elif event == cv2.EVENT_LBUTTONUP and self.drawing and self.anchor is not None:
            self.drawing = False
            rect = normalized_rect(self.anchor, point, self.width, self.height)
            self.anchor = None
            self.current = None
            if rect[2] < MIN_DRAG or rect[3] < MIN_DRAG:
                print("That rectangle is too small. Drag a larger zone.")
                return
            self.pending = rect


def draw_chip(image, text: str, x: int, y: int, scale: float) -> None:
    (width, height), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = min(max(x, 0), max(0, image.shape[1] - width - 16))
    y = min(max(y, height + 8), image.shape[0] - base - 6)
    cv2.rectangle(image, (x, y - height - 8), (x + width + 16, y + base + 6), INK, -1)
    cv2.putText(
        image, text, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2, cv2.LINE_AA
    )


def render(frame, zones: list[Zone], drawer: Drawer, pending: tuple[int, int, int, int] | None):
    display = frame.copy()
    height, width = display.shape[:2]
    for index, zone in enumerate(zones, start=1):
        cv2.rectangle(
            display,
            (zone.x, zone.y),
            (zone.x + zone.w, zone.y + zone.h),
            CORAL,
            2,
        )
        draw_chip(display, f"{index} {zone.ticker}", zone.x + 6, zone.y + 28, 0.7)
    if drawer.drawing and drawer.anchor and drawer.current:
        x, y, w, h = normalized_rect(drawer.anchor, drawer.current, width, height)
        cv2.rectangle(display, (x, y), (x + w, y + h), WHITE, 2)
    if pending is not None:
        x, y, w, h = pending
        cv2.rectangle(display, (x, y), (x + w, y + h), WHITE, 2)
        prompt = "TYPE THE TICKER IN THE TERMINAL"
        scale = fit_scale(prompt, width - 48, 0.9)
        (tw, th), _base = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        draw_chip(display, prompt, (width - tw) // 2, height // 2, scale)
    bar_h = max(42, height // 16)
    cv2.rectangle(display, (0, height - bar_h), (width, height), INK, -1)
    hint_scale = fit_scale(HINT, width - 24, 0.62)
    cv2.putText(
        display,
        HINT,
        (16, height - bar_h // 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        hint_scale,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    draw_chip(display, f"{len(zones)} zones", 12, 36, 0.7)
    return display


def ask_ticker(rect: tuple[int, int, int, int]) -> str | None:
    _x, _y, width, height = rect
    print(f"\nZone {width}x{height} at ({_x}, {_y}).")
    while True:
        raw = input("Ticker for this zone (blank skips): ").strip()
        if not raw:
            print("Skipped that rectangle.")
            return None
        symbol = normalize_ticker(raw)
        if symbol is None:
            print("Use a ticker like AAPL, VOO, or BRK.B.")
            continue
        return symbol


def write_zones(path: Path, zones: list[Zone]) -> None:
    save_zones(path, zones)
    noun = "zone" if len(zones) == 1 else "zones"
    print(f"Saved {len(zones)} {noun} to {path}.")


def maybe_save(path: Path, zones: list[Zone]) -> None:
    answer = input("Save zones before quit? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        write_zones(path, zones)


def run(config_path: Path) -> None:
    config = load_config(config_path)
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    capture = None
    try:
        capture, frame = open_camera(config.camera_index)
        zones = list(config.zones)
        dirty = False
        height, width = frame.shape[:2]
        noun = "zone" if len(zones) == 1 else "zones"
        print(f"Camera {width}x{height}. Loaded {len(zones)} {noun} from {config_path}.")
        if zones:
            print("Existing zones: " + ", ".join(zone.ticker for zone in zones))
        print("Drag a rectangle in the preview, then type the ticker in this terminal.")
        print("Keys: u undo, c clear, s save, q or Esc quit.")
        print("Draw hold zones in order HOLD then SELL. Draw size zones as 10%, 25%, 50%.")
        drawer = Drawer(WINDOW)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(width, 1280), min(height, 720))
        cv2.setMouseCallback(WINDOW, drawer.on_mouse)
        first = True
        while True:
            if not first:
                ok, frame = capture.read()
                if not ok or frame is None:
                    die("The webcam stopped returning frames.")
            first = False
            drawer.set_size(frame.shape[1], frame.shape[0])
            display = render(frame, zones, drawer, drawer.pending)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(20) & 0xFF
            if drawer.pending is not None:
                cv2.waitKey(1)
                rect = drawer.pending
                drawer.pending = None
                drawer.asking = True
                try:
                    ticker = ask_ticker(rect)
                finally:
                    drawer.asking = False
                if ticker:
                    x, y, w, h = rect
                    zones.append(Zone(ticker, x, y, w, h))
                    dirty = True
                    print(f"Added {ticker}. {len(zones)} zones so far. Press s to save.")
                continue
            if key in (ord("q"), 27):
                if dirty:
                    maybe_save(config_path, zones)
                break
            if key == ord("u"):
                if not zones:
                    print("No zone to undo.")
                else:
                    removed = zones.pop()
                    dirty = True
                    print(f"Removed {removed.ticker}.")
            elif key == ord("c"):
                zones.clear()
                dirty = True
                print("Cleared zones. Draw new ones, then press s to save.")
            elif key == ord("s"):
                write_zones(config_path, zones)
                dirty = False
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Draw one rectangle per tank zone and save tickers to config.json."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Config file to update. Other settings in the file are kept.",
    )
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
