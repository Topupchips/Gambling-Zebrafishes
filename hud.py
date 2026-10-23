"""Broadcast graphics for fish-fund rounds.

Everything here draws onto BGR uint8 numpy frames, so the same pixels go to the
preview window and to the recording. Two layouts:

- ``reel``: 1080x1920 vertical, ready for Reels / TikTok / Shorts.
- ``wide``: 1920x1080, for YouTube or a screen behind you on set.

The look is a casino broadcast: dark felt, gold rules, poker-chip colours, a live
leaderboard, a countdown ring, and a slot-machine reveal with chip confetti.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {
    "display": "Anton-Regular.ttf",
    "bold": "SpaceGrotesk-Bold.ttf",
    "body": "SpaceGrotesk-Regular.ttf",
    "mono": "JetBrainsMono-Bold.ttf",
}


def hex_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return (b, g, r)


# Palette ---------------------------------------------------------------------
FELT_DEEP = "#04100B"
FELT = "#0B2A1D"
FELT_HI = "#134531"
GOLD = "#F5C542"
GOLD_DEEP = "#B8862B"
INK = "#F4F1EA"
MUTED = "#8FA89C"
DIM = "#41594E"
HOT = "#FF3B4E"
UP = "#3DFF8F"
DOWN = "#FF5A5A"
SHADOW = "#020806"

# Casino chip colours, one per zone.
ZONE_HEX = ("#FF3B4E", "#2EE6D6", "#F5C542", "#A66BFF", "#9BFF4D")


def zone_hex(index: int) -> str:
    return ZONE_HEX[index % len(ZONE_HEX)]


# Easing ----------------------------------------------------------------------
def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_out_quint(t: float) -> float:
    t = clamp(t)
    return 1 - (1 - t) ** 5


def ease_out_back(t: float, overshoot: float = 1.9) -> float:
    t = clamp(t)
    c3 = overshoot + 1
    return 1 + c3 * (t - 1) ** 3 + overshoot * (t - 1) ** 2


def mix_hex(a: str, b: str, t: float) -> str:
    t = clamp(t)
    ca = [int(a.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    cb = [int(b.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb))


# Text sprites ----------------------------------------------------------------
@lru_cache(maxsize=128)
def font(key: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / FONT_FILES[key]
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size)


@lru_cache(maxsize=4096)
def text_sprite(text: str, key: str, size: int, color: str, tracking: int = 0) -> np.ndarray:
    """Render text to a tightly cropped BGRA sprite."""
    face = font(key, size)
    ascent, descent = face.getmetrics()
    pad = max(4, size // 6)
    width = int(sum(face.getlength(ch) for ch in text) + tracking * max(0, len(text) - 1)) + pad * 2
    height = ascent + descent + pad * 2
    image = Image.new("L", (max(1, width), max(1, height)), 0)
    draw = ImageDraw.Draw(image)
    if tracking:
        x = float(pad)
        for ch in text:
            draw.text((x, pad), ch, font=face, fill=255)
            x += face.getlength(ch) + tracking
    else:
        draw.text((pad, pad), text, font=face, fill=255)
    alpha = np.asarray(image)
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return np.zeros((1, 1, 4), np.uint8)
    alpha = alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    sprite = np.zeros((*alpha.shape, 4), np.uint8)
    sprite[..., :3] = hex_bgr(color)
    sprite[..., 3] = alpha
    return sprite


def fit_text(text: str, key: str, max_width: int, max_size: int, color: str, tracking: int = 0, min_size: int = 12) -> np.ndarray:
    size = max_size
    while size > min_size:
        sprite = text_sprite(text, key, size, color, tracking)
        if sprite.shape[1] <= max_width:
            return sprite
        size = int(size * 0.92)
    return text_sprite(text, key, min_size, color, tracking)


@lru_cache(maxsize=512)
def _glow_cached(text: str, key: str, size: int, color: str, tracking: int, radius: int, strength: float) -> np.ndarray:
    return make_glow(text_sprite(text, key, size, color, tracking), radius, color, strength)


def make_glow(sprite: np.ndarray, radius: int, color: str, strength: float = 1.6) -> np.ndarray:
    """A soft halo for a sprite, padded by ``radius`` on each side."""
    h, w = sprite.shape[:2]
    pad = radius * 2
    alpha = np.zeros((h + pad * 2, w + pad * 2), np.float32)
    alpha[pad : pad + h, pad : pad + w] = sprite[..., 3] / 255.0
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(alpha, (k, k), radius / 1.6)
    blurred = np.clip(blurred * strength, 0, 1)
    out = np.zeros((*blurred.shape, 4), np.uint8)
    out[..., :3] = hex_bgr(color)
    out[..., 3] = (blurred * 255).astype(np.uint8)
    return out


def glow_text(text: str, key: str, size: int, color: str, tracking: int = 0, radius: int = 18, strength: float = 1.4) -> np.ndarray:
    return _glow_cached(text, key, size, color, tracking, radius, strength)


# Compositing -----------------------------------------------------------------
def _anchor_origin(x: float, y: float, w: int, h: int, anchor: str) -> tuple[int, int]:
    ax, ay = anchor[0], anchor[1]
    x0 = x - (w / 2 if ax == "c" else w if ax == "r" else 0)
    y0 = y - (h / 2 if ay == "c" else h if ay == "b" else 0)
    return int(round(x0)), int(round(y0))


def blit(canvas: np.ndarray, sprite: np.ndarray, x: float, y: float, anchor: str = "lt", alpha: float = 1.0, scale: float = 1.0, additive: bool = False) -> tuple[int, int, int, int]:
    """Alpha-composite a BGRA sprite. Clips at the canvas edge. Returns the box drawn."""
    if alpha <= 0.003 or scale <= 0.01:
        return (0, 0, 0, 0)
    if abs(scale - 1.0) > 0.003:
        sprite = cv2.resize(
            sprite,
            (max(1, int(sprite.shape[1] * scale)), max(1, int(sprite.shape[0] * scale))),
            interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA,
        )
    h, w = sprite.shape[:2]
    x0, y0 = _anchor_origin(x, y, w, h, anchor)
    H, W = canvas.shape[:2]
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, W), min(y0 + h, H)
    if cx0 >= cx1 or cy0 >= cy1:
        return (x0, y0, w, h)
    part = sprite[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    roi = canvas[cy0:cy1, cx0:cx1]
    a = part[..., 3].astype(np.float32) * np.float32(alpha / 255.0)
    colour = np.ascontiguousarray(part[..., :3])
    if additive:
        roi[:] = cv2.add(roi, (colour.astype(np.float32) * a[..., None]).astype(np.uint8))
    else:
        roi[:] = cv2.blendLinear(colour, np.ascontiguousarray(roi), a, 1.0 - a)
    return (x0, y0, w, h)


@lru_cache(maxsize=256)
def round_mask(w: int, h: int, r: int) -> np.ndarray:
    """Anti-aliased rounded-rect coverage, float32 0..1."""
    w, h = max(1, w), max(1, h)
    r = int(max(0, min(r, w // 2, h // 2)))
    scale = 3
    big = np.zeros((h * scale, w * scale), np.uint8)
    R = r * scale
    W, Hh = w * scale, h * scale
    if R == 0:
        big[:] = 255
    else:
        cv2.rectangle(big, (R, 0), (W - 1 - R, Hh - 1), 255, -1)
        cv2.rectangle(big, (0, R), (W - 1, Hh - 1 - R), 255, -1)
        for cx, cy in ((R, R), (W - 1 - R, R), (R, Hh - 1 - R), (W - 1 - R, Hh - 1 - R)):
            cv2.circle(big, (cx, cy), R, 255, -1)
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


@lru_cache(maxsize=256)
def ring_mask(w: int, h: int, r: int, t: int) -> np.ndarray:
    outer = round_mask(w, h, r)
    if w - 2 * t <= 0 or h - 2 * t <= 0:
        return outer
    inner = np.zeros_like(outer)
    inner[t : h - t, t : w - t] = round_mask(w - 2 * t, h - 2 * t, max(0, r - t))
    return np.clip(outer - inner, 0, 1)


def paint(canvas: np.ndarray, x: int, y: int, mask: np.ndarray, color, alpha: float = 1.0) -> None:
    """Fill ``color`` (hex or BGR array broadcastable to the mask) through ``mask``."""
    if alpha <= 0.003:
        return
    h, w = mask.shape[:2]
    H, W = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    m = mask[cy0 - y : cy1 - y, cx0 - x : cx1 - x]
    if alpha < 0.999:
        m = m * np.float32(alpha)
    m = np.ascontiguousarray(m, np.float32)
    roi = canvas[cy0:cy1, cx0:cx1]
    if isinstance(color, str):
        col = np.empty_like(roi)
        col[:] = hex_bgr(color)
    else:
        col = np.asarray(color)
        if col.ndim == 3:
            col = col[cy0 - y : cy1 - y, cx0 - x : cx1 - x]
        col = np.ascontiguousarray(col, np.uint8)
    roi[:] = cv2.blendLinear(col, np.ascontiguousarray(roi), m, 1.0 - m)


def round_rect(canvas, x, y, w, h, r, color, alpha=1.0) -> None:
    paint(canvas, int(x), int(y), round_mask(int(w), int(h), int(r)), color, alpha)


def round_stroke(canvas, x, y, w, h, r, t, color, alpha=1.0) -> None:
    paint(canvas, int(x), int(y), ring_mask(int(w), int(h), int(r), int(t)), color, alpha)


@lru_cache(maxsize=64)
def h_gradient(w: int, h: int, left: str, right: str) -> np.ndarray:
    a = np.array(hex_bgr(left), np.float32)
    b = np.array(hex_bgr(right), np.float32)
    t = np.linspace(0, 1, max(1, w), dtype=np.float32)[None, :, None]
    row = a * (1 - t) + b * t
    return np.repeat(row, max(1, h), axis=0).astype(np.uint8)


def darken(canvas: np.ndarray, amount: float, box: tuple[int, int, int, int] | None = None) -> None:
    if amount <= 0.003:
        return
    if box is None:
        roi = canvas
    else:
        x, y, w, h = (int(v) for v in box)
        roi = canvas[max(0, y) : y + h, max(0, x) : x + w]
    if roi.size:
        roi[:] = cv2.convertScaleAbs(roi, alpha=1 - clamp(amount))


def tint(canvas: np.ndarray, x: int, y: int, w: int, h: int, color: str, alpha: float) -> None:
    """Fast rectangular colour wash."""
    if alpha <= 0.003:
        return
    x, y, w, h = int(x), int(y), int(w), int(h)
    roi = canvas[max(0, y) : y + h, max(0, x) : x + w]
    if not roi.size:
        return
    solid = np.empty_like(roi)
    solid[:] = hex_bgr(color)
    roi[:] = cv2.addWeighted(solid, alpha, roi, 1 - alpha, 0)


def glow_layer(canvas: np.ndarray, layer: np.ndarray, x: int, y: int, radius: int, strength: float = 1.0) -> None:
    """Additively blend a blurred copy of a BGR layer (black = nothing)."""
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(layer, (k, k), radius / 1.5)
    h, w = blurred.shape[:2]
    H, W = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    part = blurred[cy0 - y : cy1 - y, cx0 - x : cx1 - x].astype(np.float32) * strength
    roi = canvas[cy0:cy1, cx0:cx1]
    roi[:] = np.clip(roi.astype(np.float32) + part, 0, 255).astype(np.uint8)


def triangle(canvas, cx: int, cy: int, size: int, up: bool, color: str) -> None:
    s = size
    if up:
        pts = np.array([[cx, cy - s], [cx - s, cy + s * 0.8], [cx + s, cy + s * 0.8]], np.int32)
    else:
        pts = np.array([[cx, cy + s], [cx - s, cy - s * 0.8], [cx + s, cy - s * 0.8]], np.int32)
    cv2.fillPoly(canvas, [pts], hex_bgr(color), cv2.LINE_AA)


def poker_chip(canvas, cx: int, cy: int, r: int, color: str, label: str = "") -> None:
    """A flat casino chip: coloured disc, white edge spots, inner ring."""
    cv2.circle(canvas, (cx + 3, cy + 5), r, hex_bgr(SHADOW), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r, hex_bgr(color), -1, cv2.LINE_AA)
    for i in range(6):
        angle = i * math.pi / 3
        start = math.degrees(angle) - 11
        cv2.ellipse(canvas, (cx, cy), (r - max(3, r // 7), r - max(3, r // 7)), 0, start, start + 22, hex_bgr(INK), max(4, r // 4), cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(r * 0.62), hex_bgr(mix_hex(color, "#000000", 0.25)), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(r * 0.62), hex_bgr(INK), max(1, r // 18), cv2.LINE_AA)
    if label:
        sprite = fit_text(label, "display", int(r * 1.05), int(r * 0.7), INK)
        blit(canvas, sprite, cx, cy, "cc")


# Particles -------------------------------------------------------------------
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: str
    r: float
    spin: float
    angle: float
    born: float
    life: float
    kind: str  # "chip" | "spark" | "bill"


class Confetti:
    def __init__(self) -> None:
        self.items: list[Particle] = []
        self.last = None

    def burst(self, x: float, y: float, now: float, colors: list[str], count: int = 90, power: float = 1.0) -> None:
        rng = random.Random()
        for _ in range(count):
            angle = rng.uniform(-math.pi, 0) if rng.random() < 0.8 else rng.uniform(0, math.pi)
            speed = rng.uniform(500, 1500) * power
            kind = rng.choices(("chip", "spark", "bill"), (0.45, 0.35, 0.2))[0]
            self.items.append(
                Particle(
                    x=x + rng.uniform(-30, 30),
                    y=y + rng.uniform(-20, 20),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed - rng.uniform(100, 500),
                    color=rng.choice(colors),
                    r=rng.uniform(10, 22) if kind == "chip" else rng.uniform(3, 7),
                    spin=rng.uniform(-9, 9),
                    angle=rng.uniform(0, math.tau),
                    born=now,
                    life=rng.uniform(1.6, 3.2),
                    kind=kind,
                )
            )

    def draw(self, canvas: np.ndarray, now: float) -> None:
        dt = 0.0 if self.last is None else clamp(now - self.last, 0, 0.1)
        self.last = now
        alive = []
        for p in self.items:
            age = now - p.born
            if age > p.life:
                continue
            p.vy += 1700 * dt
            p.vx *= 1 - 1.2 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            fade = 1 - clamp((age - p.life * 0.6) / (p.life * 0.4))
            colour = hex_bgr(mix_hex(SHADOW, p.color, fade))
            center = (int(p.x), int(p.y))
            if p.kind == "chip":
                squash = abs(math.cos(p.angle))
                axes = (int(p.r), max(2, int(p.r * squash)))
                cv2.ellipse(canvas, center, axes, math.degrees(p.angle * 0.3), 0, 360, colour, -1, cv2.LINE_AA)
                cv2.ellipse(canvas, center, (max(1, axes[0] - 4), max(1, axes[1] - 4)), math.degrees(p.angle * 0.3), 0, 360, hex_bgr(mix_hex(SHADOW, INK, fade)), 2, cv2.LINE_AA)
            elif p.kind == "bill":
                w, h = 26, max(3, int(14 * abs(math.sin(p.angle))))
                box = cv2.boxPoints(((p.x, p.y), (w, h), math.degrees(p.angle)))
                cv2.fillPoly(canvas, [box.astype(np.int32)], hex_bgr(mix_hex(SHADOW, "#7CE38B", fade)), cv2.LINE_AA)
            else:
                end = (int(p.x - p.vx * 0.02), int(p.y - p.vy * 0.02))
                cv2.line(canvas, center, end, colour, max(2, int(p.r)), cv2.LINE_AA)
            alive.append(p)
        self.items = alive


# Layouts ---------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    name: str
    size: tuple[int, int]  # (w, h)
    header: tuple[int, int, int, int]
    camera_box: tuple[int, int, int, int]  # area the camera is fitted into
    clock: tuple[int, int, int, int]
    board: tuple[int, int, int, int]
    tape: tuple[int, int, int, int]
    footer_y: int
    reveal: tuple[int, int, int, int]
    k: float  # type scale


REEL = Layout(
    name="reel",
    size=(1080, 1920),
    header=(48, 44, 984, 210),
    camera_box=(40, 270, 1000, 640),
    clock=(40, 0, 1000, 300),  # y filled in at runtime, sits under the camera
    board=(40, 0, 1000, 0),
    tape=(0, 1772, 1080, 64),
    footer_y=1876,
    reveal=(40, 0, 1000, 0),
    k=1.0,
)

WIDE = Layout(
    name="wide",
    size=(1920, 1080),
    header=(48, 30, 1824, 120),
    camera_box=(40, 170, 1240, 790),
    clock=(1320, 170, 560, 300),
    board=(1320, 490, 560, 470),
    tape=(0, 984, 1920, 52),
    footer_y=1058,
    reveal=(1320, 170, 560, 790),
    k=0.78,
)

LAYOUTS = {"reel": REEL, "wide": WIDE}


# Round copy ------------------------------------------------------------------
ROUND_COPY = {
    "buy": ("WHICH STOCK", "DO THE FISH BUY?"),
    "hold": ("HOLD OR SELL", "{subject}?"),
    "size": ("HOW MUCH", "DO THEY BET?"),
    "poker": ("FOLD, CALL", "OR RAISE?"),
}
ROUND_NAME = {"buy": "BUY ROUND", "hold": "HOLD OR SELL", "size": "BET SIZE", "poker": "POKER HAND"}


@dataclass
class RoundInfo:
    round_type: str
    labels: list[str]
    subject: str = ""
    note: str = ""
    number: int = 1
    countdown: float = 15.0
    reveal_seconds: float = 6.0
    fish_expected: int = 5
    footer: str = "PAPER PORTFOLIO  ·  REAL MARKET PRICES  ·  NOT FINANCIAL ADVICE"
    camera_label: str = "CAM 01  ·  THE TANK"


@dataclass
class Quote:
    price: float
    change_pct: float | None = None


@dataclass
class HudState:
    started: float
    shares: list[float] = field(default_factory=list)
    trails: deque = field(default_factory=lambda: deque(maxlen=14))
    last_second: int = -1
    second_changed_at: float = 0.0
    winner: int | None = None
    winner_at: float = 0.0
    winner_counts: list[int] = field(default_factory=list)
    burst_done: bool = False
    quotes: dict[str, Quote] = field(default_factory=dict)
    tape_sprite: np.ndarray | None = None
    tape_key: tuple = ()


class Hud:
    """Stateful renderer. Call ``frame`` once per camera frame."""

    def __init__(self, layout: Layout, info: RoundInfo, source_size: tuple[int, int], started: float, pixel_art: bool = False) -> None:
        self.pixel_art = pixel_art
        self.layout = layout
        self.info = info
        self.src_w, self.src_h = source_size
        self.state = HudState(started=started, shares=[0.0] * len(info.labels))
        self.confetti = Confetti()
        W, H = layout.size
        bx, by, bw, bh = layout.camera_box
        scale = min(bw / self.src_w, bh / self.src_h)
        self.cam_w = int(self.src_w * scale) // 2 * 2
        self.cam_h = int(self.src_h * scale) // 2 * 2
        self.cam_x = bx + (bw - self.cam_w) // 2
        self.cam_y = by + (bh - self.cam_h) // 2 if layout.name == "wide" else by
        self.cam_scale = scale
        if layout.name == "reel":
            clock_y = self.cam_y + self.cam_h + 44
            self.clock = (40, clock_y, 1000, 290)
            board_y = clock_y + 300
            self.board = (40, board_y, 1000, layout.tape[1] - board_y - 28)
            self.reveal = (40, clock_y - 8, 1000, layout.tape[1] - clock_y - 20)
        else:
            self.clock = layout.clock
            self.board = layout.board
            self.reveal = layout.reveal
        self.cam_radius = int(30 * layout.k)
        self.cam_mask = round_mask(self.cam_w, self.cam_h, self.cam_radius)
        self.vignette = self._make_vignette(self.cam_w, self.cam_h)
        self.background = self._make_background()

    # Static layers -----------------------------------------------------------
    def _make_background(self) -> np.ndarray:
        W, H = self.layout.size
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W * 0.5, H * 0.42
        d = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2) * 1.9
        t = np.clip(d, 0, 1)[..., None]
        felt_hi = np.array(hex_bgr(FELT_HI), np.float32)
        felt = np.array(hex_bgr(FELT), np.float32)
        deep = np.array(hex_bgr(FELT_DEEP), np.float32)
        base = np.where(t < 0.5, felt_hi * (1 - t * 2) + felt * (t * 2), felt * (2 - t * 2) + deep * (t * 2 - 1))
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 4.5, (H, W, 1)).astype(np.float32)
        weave = (np.sin(xx * 0.9) * np.sin(yy * 0.9))[..., None] * 1.6
        bg = np.clip(base + noise + weave, 0, 255).astype(np.uint8)

        # Gold halo around the camera window.
        pad = 60
        tmp = np.zeros((self.cam_h + pad * 2, self.cam_w + pad * 2, 3), np.uint8)
        paint(tmp, pad, pad, ring_mask(self.cam_w, self.cam_h, self.cam_radius, 6), GOLD, 1.0)
        glow_layer(bg, tmp, self.cam_x - pad, self.cam_y - pad, 28, 0.55)
        # Drop shadow under the camera.
        shadow = round_mask(self.cam_w + 20, self.cam_h + 20, self.cam_radius + 10)
        shadow = cv2.GaussianBlur(shadow, (61, 61), 20)
        paint(bg, self.cam_x - 10, self.cam_y + 4, shadow, SHADOW, 0.85)

        # Gold hairlines framing the header and the tape.
        x, y, w, h = self.layout.tape
        cv2.rectangle(bg, (x, y), (x + w, y + h), hex_bgr(SHADOW), -1)
        cv2.line(bg, (0, y), (w, y), hex_bgr(GOLD_DEEP), 2, cv2.LINE_AA)
        cv2.line(bg, (0, y + h), (w, y + h), hex_bgr(GOLD_DEEP), 2, cv2.LINE_AA)

        # Panels behind the clock and the board.
        for (px, py, pw, ph) in (self.clock, self.board):
            if ph <= 0:
                continue
            round_rect(bg, px, py, pw, ph, int(28 * self.layout.k), SHADOW, 0.55)
            round_stroke(bg, px, py, pw, ph, int(28 * self.layout.k), 2, DIM, 0.8)
        return bg

    @staticmethod
    def _make_vignette(w: int, h: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = np.minimum(xx, w - 1 - xx) / (w * 0.18)
        dy = np.minimum(yy, h - 1 - yy) / (h * 0.18)
        edge = np.clip(np.minimum(dx, dy), 0, 1)
        v = (0.55 + 0.45 * edge ** 0.6).astype(np.float32)
        return cv2.merge([v, v, v])

    # Inputs ------------------------------------------------------------------
    def set_quotes(self, quotes: dict[str, Quote]) -> None:
        self.state.quotes = dict(quotes)

    def declare_winner(self, index: int, counts: list[int], now: float) -> None:
        self.state.winner = index
        self.state.winner_at = now
        self.state.winner_counts = list(counts)

    # Frame -------------------------------------------------------------------
    def frame(self, source: np.ndarray, zones, counts: list[int], centroids, owners, now: float) -> np.ndarray:
        canvas = self.background.copy()
        elapsed = now - self.state.started
        total = max(1, sum(counts))
        targets = [c / total if sum(counts) else 0.0 for c in counts]
        self.state.shares = [s + (t - s) * 0.22 for s, t in zip(self.state.shares, targets)]
        leader = self._leader(counts)

        self._draw_camera(canvas, source, zones, counts, centroids, owners, leader, now)
        self._draw_header(canvas, elapsed, now)
        if self.state.winner is None:
            self._draw_clock(canvas, elapsed, now, len(centroids))
            self._draw_board(canvas, counts, leader, now)
        else:
            self._draw_reveal(canvas, now)
        self._draw_tape(canvas, now)
        self._draw_footer(canvas)
        self.confetti.draw(canvas, now)
        return canvas

    def _leader(self, counts: list[int]) -> int | None:
        if self.state.winner is not None:
            return self.state.winner
        if not counts or max(counts) == 0:
            return None
        top = max(counts)
        leaders = [i for i, c in enumerate(counts) if c == top]
        return leaders[0] if len(leaders) == 1 else None

    # Camera ------------------------------------------------------------------
    def _draw_camera(self, canvas, source, zones, counts, centroids, owners, leader, now) -> None:
        if self.pixel_art:
            # Keep pixel art crisp: hard-edged scaling, no grading.
            cam = cv2.resize(source, (self.cam_w, self.cam_h), interpolation=cv2.INTER_NEAREST)
        else:
            cam = cv2.resize(source, (self.cam_w, self.cam_h), interpolation=cv2.INTER_AREA)
            cam = cv2.convertScaleAbs(cam, alpha=1.08, beta=-8)
        cam = cv2.multiply(cam, self.vignette, dtype=cv2.CV_8U)
        s = self.cam_scale
        pulse = 0.5 + 0.5 * math.sin(now * 7.0)
        winner = self.state.winner
        k = self.layout.k

        for index, zone in enumerate(zones):
            colour = GOLD if winner == index else zone_hex(index)
            x, y = int(zone.x * s), int(zone.y * s)
            w, h = max(4, int(zone.w * s)), max(4, int(zone.h * s))
            is_lead = leader == index
            fill = 0.05
            if is_lead:
                fill = 0.09 + 0.07 * pulse
            if winner is not None:
                fill = 0.16 + 0.12 * pulse if winner == index else 0.0
            tint(cam, x + 3, y + 3, w - 6, h - 6, colour, fill)
            if winner is not None and winner != index:
                darken(cam, 0.45, (x, y, w, h))
            thickness = 4 if is_lead or winner == index else 2
            round_stroke(cam, x, y, w, h, 14, thickness, colour, 0.9)
            # Target brackets on the corners.
            arm = max(14, min(w, h) // 6)
            for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
                cv2.line(cam, (cx, cy), (cx + arm * dx, cy), hex_bgr(colour), 6, cv2.LINE_AA)
                cv2.line(cam, (cx, cy), (cx, cy + arm * dy), hex_bgr(colour), 6, cv2.LINE_AA)

        # Fish markers with fading trails.
        scaled = [(int(cx * s), int(cy * s), owner) for (cx, cy), owner in zip(centroids, owners)]
        self.state.trails.append(scaled)
        n = len(self.state.trails)
        for age, points in enumerate(self.state.trails):
            fade = (age + 1) / n
            for px, py, owner in points:
                colour = DIM if owner is None else zone_hex(owner)
                if age == n - 1:
                    # A thin four-arc reticle that frames the fish instead of covering it.
                    rr = int(26 * k + 4)
                    for start in (20, 110, 200, 290):
                        cv2.ellipse(cam, (px, py), (rr, rr), 0, start, start + 50, hex_bgr(colour), 2, cv2.LINE_AA)
                else:
                    cv2.circle(cam, (px, py), max(1, int(3 * fade)), hex_bgr(mix_hex(SHADOW, colour, fade)), -1, cv2.LINE_AA)

        for index, zone in enumerate(zones):
            colour = GOLD if winner == index else zone_hex(index)
            x, y = int(zone.x * s), int(zone.y * s)
            w, h = max(4, int(zone.w * s)), max(4, int(zone.h * s))
            is_lead = leader == index
            # Label chip.
            label = fit_text(zone.ticker, "display", max(40, w - 40), int(46 * k), SHADOW)
            lw, lh = label.shape[1] + 26, label.shape[0] + 18
            round_rect(cam, x + 12, y + 12, lw, lh, 10, colour, 0.95)
            blit(cam, label, x + 12 + lw / 2, y + 12 + lh / 2, "cc")
            # Fish-count badge.
            count_sprite = text_sprite(str(counts[index] if index < len(counts) else 0), "display", int(40 * k), INK)
            r = int(30 * k)
            bx, by = x + w - r - 12, y + r + 12
            cv2.circle(cam, (bx, by), r, hex_bgr(SHADOW), -1, cv2.LINE_AA)
            cv2.circle(cam, (bx, by), r, hex_bgr(colour), 3, cv2.LINE_AA)
            blit(cam, count_sprite, bx, by, "cc")
            if winner == index:
                tag = text_sprite("PICKED", "bold", int(30 * k), SHADOW, tracking=6)
                tw, th = tag.shape[1] + 30, tag.shape[0] + 18
                round_rect(cam, x + w / 2 - tw / 2, y + h - th - 14, tw, th, th // 2, GOLD, 1.0)
                blit(cam, tag, x + w / 2, y + h - 14 - th / 2, "cc")
            elif is_lead and winner is None:
                tag = text_sprite("IN THE LEAD", "bold", int(24 * k), SHADOW, tracking=4)
                tw, th = tag.shape[1] + 26, tag.shape[0] + 16
                round_rect(cam, x + w / 2 - tw / 2, y + h - th - 14, tw, th, th // 2, colour, 0.95)
                blit(cam, tag, x + w / 2, y + h - 14 - th / 2, "cc")

        # Paste through the rounded mask, then the gold frame.
        self._paste_camera(canvas, cam)
        frame_colour = GOLD if winner is None else mix_hex(GOLD, INK, pulse * 0.5)
        round_stroke(canvas, self.cam_x, self.cam_y, self.cam_w, self.cam_h, self.cam_radius, 4, frame_colour, 1.0)

        # Captions above the window.
        cap_y = self.cam_y - int(14 * k)
        blit(canvas, text_sprite(self.info.camera_label, "mono", int(22 * k), MUTED, tracking=2), self.cam_x + 6, cap_y, "lb")
        tracking = text_sprite(f"TRACKING {len(centroids)} FISH", "mono", int(22 * k), INK, tracking=2)
        tx = self.cam_x + self.cam_w - 6
        blit(canvas, tracking, tx, cap_y, "rb")
        cv2.circle(canvas, (tx - tracking.shape[1] - 16, cap_y - tracking.shape[0] // 2), 6, hex_bgr(mix_hex(UP, SHADOW, 0.5 - 0.5 * math.sin(now * 5))), -1, cv2.LINE_AA)

    def _paste_camera(self, canvas: np.ndarray, cam: np.ndarray) -> None:
        """Copy the camera in, then blend only the rounded corners."""
        x, y, w, h, r = self.cam_x, self.cam_y, self.cam_w, self.cam_h, self.cam_radius
        canvas[y : y + h, x : x + w] = cam
        for cx, cy in ((0, 0), (w - r, 0), (0, h - r), (w - r, h - r)):
            m = np.ascontiguousarray(self.cam_mask[cy : cy + r, cx : cx + r])
            bg = np.ascontiguousarray(self.background[y + cy : y + cy + r, x + cx : x + cx + r])
            patch = np.ascontiguousarray(cam[cy : cy + r, cx : cx + r])
            canvas[y + cy : y + cy + r, x + cx : x + cx + r] = cv2.blendLinear(patch, bg, m, 1.0 - m)

    # Header ------------------------------------------------------------------
    def _draw_header(self, canvas, elapsed: float, now: float) -> None:
        x, y, w, h = self.layout.header
        k = self.layout.k
        mark = text_sprite("FISH FUND", "display", int(128 * k), INK, tracking=int(6 * k))
        glow = glow_text("FISH FUND", "display", int(128 * k), GOLD, int(6 * k), 22, 0.55)
        blit(canvas, glow, x - 44, y - 44 + 6, "lt", alpha=0.6)
        blit(canvas, mark, x, y + 6, "lt")
        sub_y = y + mark.shape[0] + int(22 * k)
        info = self.info
        sub = f"ROUND {info.number:02d}  ·  {ROUND_NAME.get(info.round_type, info.round_type.upper())}"
        sub_sprite = text_sprite(sub, "bold", int(34 * k), GOLD, tracking=int(6 * k))
        if self.layout.name == "wide":
            blit(canvas, sub_sprite, x + mark.shape[1] + 40, y + mark.shape[0] / 2 + 6, "lc")
        else:
            blit(canvas, sub_sprite, x + 2, sub_y, "lt")

        # LIVE pill + clock, right aligned.
        rx = x + w
        dot = 0.55 + 0.45 * math.sin(now * 6)
        live = text_sprite("LIVE", "bold", int(30 * k), INK, tracking=4)
        pw, ph = live.shape[1] + int(70 * k), live.shape[0] + int(26 * k)
        px, py = rx - pw, y + 14
        round_rect(canvas, px, py, pw, ph, ph // 2, HOT, 1.0)
        cv2.circle(canvas, (px + int(28 * k), py + ph // 2), int(9 * k), hex_bgr(mix_hex(HOT, INK, dot)), -1, cv2.LINE_AA)
        blit(canvas, live, px + int(48 * k), py + ph / 2, "lc")
        rec = f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        blit(canvas, text_sprite(rec, "mono", int(26 * k), INK, tracking=2), rx, py + ph + int(20 * k), "rt")
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d  %H:%M:%S %Z")
        blit(canvas, text_sprite(stamp, "mono", int(22 * k), MUTED, tracking=1), rx, py + ph + int(62 * k), "rt")

    # Countdown ---------------------------------------------------------------
    def _draw_clock(self, canvas, elapsed: float, now: float, fish_seen: int) -> None:
        x, y, w, h = self.clock
        k = self.layout.k
        info = self.info
        remaining = max(0.0, info.countdown - elapsed)
        second = max(0, math.ceil(remaining - 1e-9))
        if second != self.state.last_second:
            self.state.last_second = second
            self.state.second_changed_at = now
        pop = 1.0 + 0.28 * (1 - ease_out_cubic((now - self.state.second_changed_at) / 0.3))
        urgent = remaining <= 5
        colour = mix_hex(GOLD, HOT, clamp((5 - remaining) / 5)) if urgent else GOLD
        r = int(h * 0.29) if self.layout.name == "wide" else 118
        cx = x + (r + 44 if self.layout.name == "reel" else w // 2)
        cy = y + h // 2 if self.layout.name == "reel" else y + r + 44

        # Ring: dim track, progress arc, glow.
        layer = np.zeros((r * 2 + 80, r * 2 + 80, 3), np.uint8)
        lc = (r + 40, r + 40)
        frac = remaining / max(0.001, info.countdown)
        cv2.circle(canvas, (cx, cy), r, hex_bgr(DIM), int(14 * max(k, 0.8)), cv2.LINE_AA)
        cv2.ellipse(layer, lc, (r, r), -90, 0, 360 * frac, hex_bgr(colour), int(16 * max(k, 0.8)), cv2.LINE_AA)
        glow_layer(canvas, layer, cx - lc[0], cy - lc[1], 16, 0.9 + (0.6 * (0.5 + 0.5 * math.sin(now * 12)) if urgent else 0))
        mask = layer.any(axis=2)
        sub = canvas[cy - lc[1] : cy - lc[1] + layer.shape[0], cx - lc[0] : cx - lc[0] + layer.shape[1]]
        if sub.shape[:2] == mask.shape:
            sub[mask] = layer[mask]
        # Tick marks.
        for i in range(60):
            a = math.radians(i * 6 - 90)
            r1 = r + 22
            r2 = r + (34 if i % 5 == 0 else 28)
            cv2.line(canvas, (int(cx + r1 * math.cos(a)), int(cy + r1 * math.sin(a))), (int(cx + r2 * math.cos(a)), int(cy + r2 * math.sin(a))), hex_bgr(DIM if i / 60 > frac else colour), 2, cv2.LINE_AA)
        number = text_sprite(str(second), "display", int(r * 1.15), INK)
        blit(canvas, glow_text(str(second), "display", int(r * 1.15), colour, 0, 20, 0.7), cx, cy - 6, "cc", alpha=0.85 if urgent else 0.45, scale=pop)
        blit(canvas, number, cx, cy - 6, "cc", scale=pop)
        blit(canvas, text_sprite("SECONDS", "mono", int(18 * k + 2), MUTED, tracking=4), cx, cy + r * 0.62, "cc")

        if self.layout.name == "reel":
            tx = cx + r + 64
            tw = x + w - tx - 30
            kicker = "LAST CHANCE TO SWIM" if urgent else "LIVE DECISION"
            kick = text_sprite(kicker, "bold", 26, HOT if urgent else GOLD, tracking=7)
            blit(canvas, kick, tx, y + 44, "lt")
            line1, line2 = ROUND_COPY.get(info.round_type, ("WHERE DO", "THEY GO?"))
            line2 = line2.format(subject=info.subject or "")
            l1 = fit_text(line1, "display", tw, 76, INK)
            l2 = fit_text(line2, "display", tw, 76, INK)
            ly = y + 44 + kick.shape[0] + 18
            blit(canvas, l1, tx, ly, "lt")
            blit(canvas, l2, tx, ly + l1.shape[0] + 14, "lt")
            meta = info.note.upper() if info.note else f"{fish_seen} FISH ON CAMERA  ·  {len(info.labels)} OPTIONS"
            blit(canvas, fit_text(meta, "mono", tw, 22, MUTED, tracking=1), tx, ly + l1.shape[0] + l2.shape[0] + 40, "lt")
        else:
            line1, line2 = ROUND_COPY.get(info.round_type, ("WHERE DO", "THEY GO?"))
            q = f"{line1} {line2.format(subject=info.subject or '')}"
            blit(canvas, fit_text(q, "display", w - 40, 44, INK), x + w // 2, y + h - 18, "cb")

    # Leaderboard -------------------------------------------------------------
    def _draw_board(self, canvas, counts: list[int], leader: int | None, now: float) -> None:
        x, y, w, h = self.board
        k = self.layout.k
        labels = self.info.labels
        n = max(1, len(labels))
        title = text_sprite("THE SCHOOL VOTES", "bold", int(26 * k), MUTED, tracking=8)
        blit(canvas, title, x + 36, y + 30, "lt")
        top = y + 30 + title.shape[0] + int(18 * k)
        row_h = min(int(150 * k), (y + h - top - 16) // n)
        pulse = 0.5 + 0.5 * math.sin(now * 6)
        for i, label in enumerate(labels):
            ry = top + i * row_h
            colour = zone_hex(i)
            lead = leader == i
            if lead:
                round_rect(canvas, x + 14, ry + 4, w - 28, row_h - 8, 20, colour, 0.08 + 0.05 * pulse)
                round_stroke(canvas, x + 14, ry + 4, w - 28, row_h - 8, 20, 2, colour, 0.9)
            chip_r = int(min(40 * k, row_h * 0.3))
            cxc = x + 36 + chip_r
            cyc = ry + row_h // 2
            poker_chip(canvas, cxc, cyc, chip_r, colour, chr(65 + i))
            name_x = cxc + chip_r + int(26 * k)
            name_w = int(230 * k) if self.layout.name == "reel" else 120
            name = fit_text(label, "display", name_w, int(62 * k), INK)
            # Prices sit under the name on the reel; the wide sidebar is too narrow, so they live in the tape.
            quote = self.state.quotes.get(label) if self.layout.name == "reel" else None
            blit(canvas, name, name_x, cyc - (8 if quote else 0), "lc" if not quote else "lb")
            if quote:
                price = text_sprite(f"${quote.price:,.2f}", "mono", int(22 * k), MUTED)
                blit(canvas, price, name_x, cyc + 6, "lt")
                if quote.change_pct is not None:
                    up = quote.change_pct >= 0
                    cc = UP if up else DOWN
                    tx = name_x + price.shape[1] + 22
                    triangle(canvas, tx, cyc + 6 + price.shape[0] // 2, 7, up, cc)
                    blit(canvas, text_sprite(f"{abs(quote.change_pct):.1f}%", "mono", int(22 * k), cc), tx + 14, cyc + 6, "lt")
            # Bar.
            count = counts[i] if i < len(counts) else 0
            bar_x = name_x + name_w + 20
            count_w = int(150 * k)
            bar_w = x + w - 36 - count_w - bar_x
            bar_h = int(26 * k)
            by = cyc - bar_h // 2
            if bar_w > 40:
                round_rect(canvas, bar_x, by, bar_w, bar_h, bar_h // 2, SHADOW, 0.9)
                fill_w = int(bar_w * clamp(self.state.shares[i] if i < len(self.state.shares) else 0))
                if fill_w > bar_h:
                    grad = h_gradient(bar_w, bar_h, mix_hex(colour, "#000000", 0.35), colour)[:, :fill_w]
                    paint(canvas, bar_x, by, round_mask(fill_w, bar_h, bar_h // 2), grad, 1.0)
                    layer = np.zeros((bar_h + 40, fill_w + 40, 3), np.uint8)
                    paint(layer, 20, 20, round_mask(fill_w, bar_h, bar_h // 2), colour, 1.0)
                    glow_layer(canvas, layer, bar_x - 20, by - 20, 10, 0.45 if lead else 0.2)
                    # shine
                    cv2.line(canvas, (bar_x + bar_h // 2, by + 6), (bar_x + fill_w - bar_h // 2, by + 6), hex_bgr(mix_hex(colour, INK, 0.5)), 2, cv2.LINE_AA)
                pct = text_sprite(f"{round(self.state.shares[i] * 100)}%", "mono", int(20 * k), MUTED)
                blit(canvas, pct, bar_x + 4, by + bar_h + 10, "lt")
            # Count.
            num = text_sprite(str(count), "display", int(84 * k), colour)
            if lead:
                blit(canvas, glow_text(str(count), "display", int(84 * k), colour, 0, 14, 0.8), x + w - 36 - count_w // 2 - 20, cyc, "cc", alpha=0.6)
            blit(canvas, num, x + w - 36 - count_w // 2 - 20, cyc, "cc")
            blit(canvas, text_sprite("FISH", "mono", int(18 * k), MUTED, tracking=3), x + w - 36, cyc, "rc")

    # Reveal ------------------------------------------------------------------
    def _draw_reveal(self, canvas, now: float) -> None:
        x, y, w, h = self.reveal
        k = self.layout.k
        info = self.info
        labels = info.labels
        winner = self.state.winner or 0
        t = now - self.state.winner_at
        spin_t = min(1.9, info.reveal_seconds * 0.35)
        intro = ease_out_cubic(t / 0.25)

        # Card.
        round_rect(canvas, x, y, w, h, int(32 * k), SHADOW, 0.94 * intro)
        border = mix_hex(GOLD_DEEP, GOLD, 0.5 + 0.5 * math.sin(now * 8))
        round_stroke(canvas, x, y, w, h, int(32 * k), 3, border, intro)
        cx = x + w // 2
        cy = y + h // 2

        if t < spin_t:
            # Slot machine: cycle the options, decelerate onto the winner.
            n = max(1, len(labels))
            steps = n * (6 if n > 1 else 1) + winner
            p = ease_out_quint(t / spin_t) * steps
            item_h = int(260 * k)
            win_h = int(item_h * 1.15)
            wy = cy - win_h // 2 + int(30 * k)
            round_rect(canvas, x + 40, wy, w - 80, win_h, 24, FELT_DEEP, 1.0)
            view = canvas[wy : wy + win_h, x + 40 : x + w - 40]
            speed = abs(steps / spin_t) * (1 - ease_out_cubic(t / spin_t)) + 0.2
            base = int(math.floor(p))
            frac = p - base
            for offset in (-1, 0, 1, 2):
                idx = (base + offset) % n
                yy = view.shape[0] / 2 + (offset - frac) * item_h
                colour = INK if offset == 0 else MUTED
                sprite = fit_text(labels[idx], "display", view.shape[1] - 80, int(190 * k), colour)
                a = clamp(1 - abs(offset - frac) * 0.6) * (0.5 if speed > 6 else 1)
                blit(view, sprite, view.shape[1] / 2, yy, "cc", alpha=a)
            # Shade the slot edges.
            for i in range(40):
                darken(view, 0.06, (0, i, view.shape[1], 1))
                darken(view, 0.06, (0, view.shape[0] - 1 - i, view.shape[1], 1))
            round_stroke(canvas, x + 40, wy, w - 80, win_h, 24, 4, GOLD, 1.0)
            cv2.fillPoly(canvas, [np.array([[x + 28, wy + win_h // 2 - 22], [x + 62, wy + win_h // 2], [x + 28, wy + win_h // 2 + 22]], np.int32)], hex_bgr(GOLD), cv2.LINE_AA)
            cv2.fillPoly(canvas, [np.array([[x + w - 28, wy + win_h // 2 - 22], [x + w - 62, wy + win_h // 2], [x + w - 28, wy + win_h // 2 + 22]], np.int32)], hex_bgr(GOLD), cv2.LINE_AA)
            tease = text_sprite("THE FISH HAVE DECIDED", "bold", int(34 * k), GOLD, tracking=8)
            blit(canvas, tease, cx, wy - int(40 * k), "cb", alpha=0.6 + 0.4 * math.sin(now * 14))
            return

        # Impact.
        hit = t - spin_t
        if not self.state.burst_done:
            self.state.burst_done = True
            colours = [GOLD, INK, zone_hex(winner), "#7CE38B", GOLD]
            self.confetti.burst(cx, cy, now, colours, count=110, power=1.0)
            self.confetti.burst(x + 80, y + h - 40, now, colours, count=40, power=0.8)
            self.confetti.burst(x + w - 80, y + h - 40, now, colours, count=40, power=0.8)
        flash = 1 - ease_out_cubic(hit / 0.3)
        if flash > 0.01:
            region = canvas[max(0, y) : y + h, max(0, x) : x + w]
            region[:] = np.clip(region.astype(np.float32) + 255 * flash * 0.85, 0, 255).astype(np.uint8)
        wave = ease_out_cubic(hit / 0.7)
        if wave < 1:
            cv2.circle(canvas, (cx, cy), int(40 + wave * 900), hex_bgr(GOLD), max(1, int(30 * (1 - wave))), cv2.LINE_AA)

        label = labels[winner] if labels else "?"
        pop = ease_out_back(hit / 0.45)
        big_size = int(300 * k)
        big = fit_text(label, "display", w - 120, big_size, GOLD)
        size_used = big_size
        while text_sprite(label, "display", size_used, GOLD).shape[1] > w - 120 and size_used > 40:
            size_used = int(size_used * 0.92)
        glow = glow_text(label, "display", size_used, GOLD, 0, 30, 0.9)
        kicker = text_sprite("THE FISH PICKED", "bold", int(36 * k), INK, tracking=10)
        stack_h = kicker.shape[0] + 30 + big.shape[0] + 40 + int(100 * k)
        top = cy - stack_h // 2
        blit(canvas, kicker, cx, top, "ct", alpha=intro)
        big_cy = top + kicker.shape[0] + 30 + big.shape[0] / 2
        blit(canvas, glow, cx, big_cy, "cc", alpha=0.75 + 0.25 * math.sin(now * 5), scale=pop)
        blit(canvas, big, cx, big_cy, "cc", scale=pop)

        counts = self.state.winner_counts
        total = sum(counts) or 1
        won = counts[winner] if winner < len(counts) else 0
        line = f"{won} OF {total} FISH  ·  {round(won / total * 100)}% OF THE SCHOOL"
        detail_y = big_cy + big.shape[0] / 2 + 40
        blit(canvas, fit_text(line, "mono", w - 80, int(30 * k), INK, tracking=2), cx, detail_y, "ct", alpha=clamp((hit - 0.3) / 0.3))
        quote = self.state.quotes.get(label)
        if quote:
            q = f"LAST CLOSE  ${quote.price:,.2f}"
            blit(canvas, fit_text(q, "mono", w - 80, int(26 * k), MUTED, tracking=2), cx, detail_y + int(50 * k), "ct", alpha=clamp((hit - 0.5) / 0.3))

        # Rubber stamp.
        st = hit - 0.7
        if st > 0:
            slam = 1 + 1.4 * (1 - ease_out_cubic(st / 0.22))
            stamp = self._stamp(f"OFFICIAL PICK #{info.number:02d}", int(30 * k))
            blit(canvas, stamp, x + w - int(70 * k), y + h - int(50 * k), "rb", alpha=clamp(st / 0.12), scale=slam)

    @staticmethod
    @lru_cache(maxsize=16)
    def _stamp(text: str, size: int) -> np.ndarray:
        word = text_sprite(text, "display", size, HOT, tracking=4)
        pad = 22
        h, w = word.shape[0] + pad * 2, word.shape[1] + pad * 2
        card = np.zeros((h, w, 4), np.uint8)
        card[..., :3] = hex_bgr(HOT)
        border = (ring_mask(w, h, 10, 5) * 255).astype(np.uint8)
        alpha = border.copy()
        alpha[pad : pad + word.shape[0], pad : pad + word.shape[1]] = np.maximum(alpha[pad : pad + word.shape[0], pad : pad + word.shape[1]], word[..., 3])
        rng = np.random.default_rng(3)
        alpha = (alpha * (rng.random(alpha.shape) > 0.12)).astype(np.uint8)  # ink texture
        card[..., 3] = alpha
        big = max(h, w)
        canvas = np.zeros((big + 40, big + 40, 4), np.uint8)
        oy, ox = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
        canvas[oy : oy + h, ox : ox + w] = card
        M = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), 9, 1.0)
        rotated = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(rotated[..., 3])
        return rotated[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # Tape & footer -----------------------------------------------------------
    def _tape_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        if self.state.quotes:
            for symbol, q in self.state.quotes.items():
                items.append((symbol, GOLD))
                items.append((f"${q.price:,.2f}", INK))
                if q.change_pct is not None:
                    items.append((f"{q.change_pct:+.2f}%", UP if q.change_pct >= 0 else DOWN))
                items.append(("   •   ", DIM))
        else:
            for text in ("5 ZEBRAFISH", "ONE TANK", "REAL MARKET PRICES", "NO HUMANS DECIDE", f"ROUND {self.info.number:02d}"):
                items.append((text, INK if text != "NO HUMANS DECIDE" else GOLD))
                items.append(("   •   ", DIM))
        return items

    def _draw_tape(self, canvas, now: float) -> None:
        x, y, w, h = self.layout.tape
        k = self.layout.k
        key = tuple(sorted((s, q.price) for s, q in self.state.quotes.items()))
        if self.state.tape_sprite is None or key != self.state.tape_key:
            parts = [text_sprite(t, "mono", int(30 * k), c, tracking=2) for t, c in self._tape_items()]
            height = max(p.shape[0] for p in parts)
            width = sum(p.shape[1] for p in parts) + 14 * len(parts)
            strip = np.zeros((height, width, 4), np.uint8)
            cursor = 0
            for p in parts:
                oy = height - p.shape[0]
                strip[oy : oy + p.shape[0], cursor : cursor + p.shape[1]] = p
                cursor += p.shape[1] + 14
            self.state.tape_sprite = strip
            self.state.tape_key = key
        strip = self.state.tape_sprite
        sw = strip.shape[1]
        offset = (now * 140) % sw
        view = canvas[y + 2 : y + h - 1, x : x + w]
        cursor = -offset
        while cursor < w:
            blit(view, strip, cursor, view.shape[0] / 2, "lc")
            cursor += sw
        # Fixed tag on the left.
        tag = text_sprite("FISH FUND", "display", int(30 * k), SHADOW, tracking=2)
        tw = tag.shape[1] + 40
        cv2.rectangle(canvas, (x, y + 2), (x + tw, y + h - 2), hex_bgr(GOLD), -1)
        blit(canvas, tag, x + tw / 2, y + h / 2, "cc")

    def _draw_footer(self, canvas) -> None:
        if not self.info.footer:
            return
        W = self.layout.size[0]
        sprite = fit_text(self.info.footer, "mono", W - 80, int(20 * self.layout.k + 2), MUTED, tracking=2)
        blit(canvas, sprite, W / 2, self.layout.footer_y, "cc", alpha=0.85)
