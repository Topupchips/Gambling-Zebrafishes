"""Studio look for the fish blackjack table.

A cinematic, illustrated card room instead of a dashboard: a brass-framed aquarium
above a lamp-lit felt table, editorial serif type, cards that land the way a dealer
pitches them, real chip stacks, and almost no interface. Everything is drawn at
``scale`` times the logical 1080x1920 frame and can be downsampled for very clean
edges (``scale=2`` then 1080x1920 out), or kept at full size for a 4K master.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache

import cv2
import numpy as np

from blackjack_game import describe, hand_value
from hud import blit, ease_out_cubic, fit_text, hex_bgr, mix_hex, round_mask, text_sprite

W0, H0 = 1080, 1920

# Palette -----------------------------------------------------------------------
INK = "#0B0D0C"
CREAM = "#F2EADB"
CREAM_DIM = "#BDB4A3"
BRASS = "#C9A45C"
BRASS_HI = "#F1D79A"
BRASS_LO = "#6E5326"
FELT = "#0F3B2B"
FELT_DEEP = "#062218"
RAIL = "#3A1A14"
RAIL_HI = "#6B3326"
OXBLOOD = "#8E1F28"
WIN = "#E9C46A"
LOSS = "#D96C5F"
PAPER = "#F7F1E3"
RED_INK = "#B4232F"
BLACK_INK = "#17181B"

CHIP_COLOURS = {1: ("#E9E2D2", "#2B5C8A"), 5: ("#A81E2B", "#F2EADB"), 25: ("#1E6B48", "#F2EADB"), 100: ("#1A1A1C", "#E9C46A")}

RESULT_LINES = {
    "blackjack": "Blackjack.",
    "win": "The fish win.",
    "dealer_bust": "Dealer busts.",
    "push": "Push.",
    "lose": "Dealer wins.",
    "bust": "Bust.",
    "dealer_blackjack": "Dealer blackjack.",
}
RANK_NAMES = {"A": "an Ace", "K": "a King", "Q": "a Queen", "J": "a Jack", "8": "an 8"}


def _ease_out_quart(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 4


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _bgr(value: str) -> np.ndarray:
    return np.array(hex_bgr(value), np.float32)


# ---------------------------------------------------------------------------
# Suits and cards
# ---------------------------------------------------------------------------
def _suit_mask(suit: str, size: int) -> np.ndarray:
    s = max(8, size * 4)
    m = np.zeros((s, s), np.uint8)

    def P(x: float, y: float) -> tuple[int, int]:
        return int(x * s), int(y * s)

    def circle(x, y, r):
        cv2.circle(m, P(x, y), int(r * s), 255, -1, cv2.LINE_AA)

    def poly(*pts):
        cv2.fillPoly(m, [np.array([P(x, y) for x, y in pts], np.int32)], 255, cv2.LINE_AA)

    if suit == "H":
        circle(0.29, 0.34, 0.23)
        circle(0.71, 0.34, 0.23)
        poly((0.07, 0.40), (0.93, 0.40), (0.5, 0.94))
    elif suit == "D":
        poly((0.5, 0.03), (0.86, 0.5), (0.5, 0.97), (0.14, 0.5))
    elif suit == "S":
        circle(0.29, 0.58, 0.22)
        circle(0.71, 0.58, 0.22)
        poly((0.08, 0.55), (0.92, 0.55), (0.5, 0.04))
        poly((0.5, 0.6), (0.34, 0.97), (0.66, 0.97))
    else:
        circle(0.5, 0.29, 0.21)
        circle(0.27, 0.6, 0.21)
        circle(0.73, 0.6, 0.21)
        circle(0.5, 0.55, 0.12)
        poly((0.5, 0.55), (0.34, 0.97), (0.66, 0.97))
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def _stamp(canvas: np.ndarray, mask: np.ndarray, cx: float, cy: float, colour: str, flip: bool = False) -> None:
    if flip:
        mask = mask[::-1, ::-1]
    h, w = mask.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    roi = canvas[y0 : y0 + h, x0 : x0 + w]
    if roi.shape[:2] != mask.shape:
        return
    a = mask[..., None]
    roi[:] = (roi * (1 - a) + _bgr(colour) * a).astype(np.uint8)


PIPS = {
    2: [(0.5, 0.0), (0.5, 1.0)],
    3: [(0.5, 0.0), (0.5, 0.5), (0.5, 1.0)],
    4: [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    5: [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0)],
    6: [(0.0, 0.0), (1.0, 0.0), (0.0, 0.5), (1.0, 0.5), (0.0, 1.0), (1.0, 1.0)],
    7: [(0.0, 0.0), (1.0, 0.0), (0.5, 0.25), (0.0, 0.5), (1.0, 0.5), (0.0, 1.0), (1.0, 1.0)],
    8: [(0.0, 0.0), (1.0, 0.0), (0.5, 0.25), (0.0, 0.5), (1.0, 0.5), (0.5, 0.75), (0.0, 1.0), (1.0, 1.0)],
    9: [(0.0, 0.0), (1.0, 0.0), (0.0, 0.333), (1.0, 0.333), (0.5, 0.5), (0.0, 0.667), (1.0, 0.667), (0.0, 1.0), (1.0, 1.0)],
    10: [(0.0, 0.0), (1.0, 0.0), (0.5, 0.167), (0.0, 0.333), (1.0, 0.333), (0.0, 0.667), (1.0, 0.667), (0.5, 0.833), (0.0, 1.0), (1.0, 1.0)],
}


def _paper(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.ones((h, w, 3), np.float32) * _bgr(PAPER)
    fibre = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8)
    tone = cv2.resize(rng.normal(0, 1, (max(2, h // 40), max(2, w // 40))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    base *= (1 + 0.012 * fibre + 0.01 * tone)[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


def _alpha_card(rgb: np.ndarray, radius: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = (round_mask(w, h, radius) * 255).astype(np.uint8)
    return out


@lru_cache(maxsize=128)
def card_face(rank: str, suit: str, w: int, h: int) -> np.ndarray:
    k = 3
    W, H = w * k, h * k
    u = W / 150  # one design unit
    card = _paper(W, H, hash((rank, suit)) & 0xFFFF)
    ink = RED_INK if suit in ("H", "D") else BLACK_INK
    # Index: serif rank over a small suit, both corners.
    corner = np.zeros((int(74 * u), int(34 * u), 3), np.uint8)
    corner[:] = card[int(8 * u) : int(8 * u) + corner.shape[0], int(7 * u) : int(7 * u) + corner.shape[1]]
    r = fit_text(rank, "serif", int(32 * u), int(46 * u), ink)
    blit(corner, r, corner.shape[1] / 2, 0, "ct")
    _stamp(corner, _suit_mask(suit, int(19 * u)), corner.shape[1] / 2, r.shape[0] + 15 * u, ink)
    ch, cw = corner.shape[:2]
    oy, ox = int(10 * u), int(8 * u)
    card[oy : oy + ch, ox : ox + cw] = corner
    card[H - oy - ch : H - oy, W - ox - cw : W - ox] = corner[::-1, ::-1]
    ix0, ix1 = W * 0.36, W * 0.64
    iy0, iy1 = H * 0.18, H * 0.82
    if rank == "A":
        _stamp(card, _suit_mask(suit, int(70 * u)), W / 2, H / 2, ink)
    elif rank in ("J", "Q", "K"):
        fx, fy, fw, fh = int(W * 0.29), int(H * 0.15), int(W * 0.42), int(H * 0.70)
        frame = card[fy : fy + fh, fx : fx + fw].astype(np.float32)
        frame = frame * 0.9 + _bgr(mix_hex(ink, PAPER, 0.86)) * 0.1
        card[fy : fy + fh, fx : fx + fw] = frame.astype(np.uint8)
        cv2.rectangle(card, (fx, fy), (fx + fw, fy + fh), hex_bgr(ink), max(1, int(1.6 * u)), cv2.LINE_AA)
        cv2.rectangle(card, (fx + int(5 * u), fy + int(5 * u)), (fx + fw - int(5 * u), fy + fh - int(5 * u)), hex_bgr(mix_hex(ink, PAPER, 0.5)), max(1, int(0.8 * u)), cv2.LINE_AA)
        blit(card, text_sprite(rank, "serif", int(100 * u), ink), W / 2, H / 2 + 2 * u, "cc")
        _stamp(card, _suit_mask(suit, int(16 * u)), fx + 15 * u, fy + 17 * u, ink)
        _stamp(card, _suit_mask(suit, int(16 * u)), fx + fw - 15 * u, fy + fh - 17 * u, ink, flip=True)
    else:
        pip = _suit_mask(suit, int(25 * u))
        for px, py in PIPS[int(rank)]:
            _stamp(card, pip, ix0 + (ix1 - ix0) * px, iy0 + (iy1 - iy0) * py, ink, flip=py > 0.5)
    # Fine edge line.
    edge = np.zeros((H, W), np.float32)
    outer = round_mask(W, H, int(12 * u))
    inner = np.pad(round_mask(W - 4, H - 4, int(12 * u) - 2), 2)
    edge = np.clip(outer - inner, 0, 1)
    card = (card * (1 - 0.35 * edge[..., None]) + _bgr("#9C9282") * 0.35 * edge[..., None]).astype(np.uint8)
    small = cv2.resize(card, (w, h), interpolation=cv2.INTER_AREA)
    return _alpha_card(small, int(12 * w / 150))


@lru_cache(maxsize=4)
def card_back(w: int, h: int) -> np.ndarray:
    k = 3
    W, H = w * k, h * k
    u = W / 150
    card = _paper(W, H, 99)
    inset = int(9 * u)
    iw, ih = W - inset * 2, H - inset * 2
    inner = np.zeros((ih, iw, 3), np.uint8)
    inner[:] = hex_bgr(OXBLOOD)
    step = int(11 * u)
    for d in range(-ih, iw + ih, step):
        cv2.line(inner, (d, 0), (d + ih, ih), hex_bgr("#A8333C"), max(1, int(1.2 * u)), cv2.LINE_AA)
        cv2.line(inner, (d + ih, 0), (d, ih), hex_bgr("#A8333C"), max(1, int(1.2 * u)), cv2.LINE_AA)
    cv2.rectangle(inner, (int(6 * u), int(6 * u)), (iw - int(6 * u), ih - int(6 * u)), hex_bgr(BRASS), max(1, int(1.4 * u)), cv2.LINE_AA)
    cx, cy = iw // 2, ih // 2
    cv2.ellipse(inner, (cx, cy), (int(34 * u), int(46 * u)), 0, 0, 360, hex_bgr("#6E141C"), -1, cv2.LINE_AA)
    cv2.ellipse(inner, (cx, cy), (int(34 * u), int(46 * u)), 0, 0, 360, hex_bgr(BRASS), max(1, int(1.4 * u)), cv2.LINE_AA)
    blit(inner, text_sprite("F", "caps_italic", int(56 * u), BRASS_HI), cx, cy, "cc")
    m = round_mask(iw, ih, int(6 * u))[..., None]
    region = card[inset : inset + ih, inset : inset + iw]
    region[:] = (region * (1 - m) + inner * m).astype(np.uint8)
    small = cv2.resize(card, (w, h), interpolation=cv2.INTER_AREA)
    return _alpha_card(small, int(12 * w / 150))


@lru_cache(maxsize=8)
def soft_shadow(w: int, h: int, blur: int) -> np.ndarray:
    pad = blur * 2
    m = np.zeros((h + pad * 2, w + pad * 2), np.float32)
    m[pad : pad + h, pad : pad + w] = round_mask(w, h, int(12 * w / 150))
    m = cv2.GaussianBlur(m, (0, 0), blur * 0.6)
    out = np.zeros((*m.shape, 4), np.uint8)
    out[..., :3] = hex_bgr("#000000")
    out[..., 3] = (np.clip(m, 0, 1) * 200).astype(np.uint8)
    return out


def rotate_sprite(sprite: np.ndarray, degrees: float, sx: float = 1.0) -> np.ndarray:
    h, w = sprite.shape[:2]
    if abs(degrees) < 0.05 and abs(sx - 1) < 0.002:
        return sprite
    side = int(math.hypot(w, h)) + 6
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    # Horizontal squash for the flip, applied before the rotation.
    S = np.array([[sx, 0, (1 - sx) * w / 2], [0, 1, 0], [0, 0, 1]], np.float32)
    R = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    T = np.array([[1, 0, (side - w) / 2], [0, 1, (side - h) / 2], [0, 0, 1]], np.float32)
    A = (T @ R @ S)[:2]
    return cv2.warpAffine(sprite, A, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# Chips
# ---------------------------------------------------------------------------
def chip_breakdown(amount: int) -> list[int]:
    out = []
    for d in (100, 25, 5, 1):
        while amount >= d and len(out) < 14:
            out.append(d)
            amount -= d
    return out


@lru_cache(maxsize=16)
def chip_sprite(value: int, r: int) -> np.ndarray:
    k = 3
    R = r * k
    size = R * 2 + 8 * k
    img = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    c = (size // 2, size // 2)
    base, accent = CHIP_COLOURS[value]
    cv2.circle(img, c, R, hex_bgr(base), -1, cv2.LINE_AA)
    cv2.circle(alpha, c, R, 255, -1, cv2.LINE_AA)
    # Edge inlays.
    for i in range(8):
        a0 = i * 45 - 8
        cv2.ellipse(img, c, (R - int(R * 0.09), R - int(R * 0.09)), 0, a0, a0 + 16, hex_bgr(accent), int(R * 0.18), cv2.LINE_AA)
    cv2.circle(img, c, int(R * 0.72), hex_bgr(mix_hex(base, "#000000", 0.2)), max(1, int(R * 0.03)), cv2.LINE_AA)
    cv2.circle(img, c, int(R * 0.58), hex_bgr(CREAM), -1, cv2.LINE_AA)
    cv2.circle(img, c, int(R * 0.58), hex_bgr(accent if value != 1 else base), max(1, int(R * 0.04)), cv2.LINE_AA)
    label = text_sprite(str(value), "serif", int(R * 0.66), BLACK_INK if value != 5 else OXBLOOD)
    blit(img, label, c[0], c[1] + R * 0.02, "cc")
    # Light: soft top-left highlight and a darker lower rim.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((xx - c[0] + R * 0.35) ** 2 + (yy - c[1] + R * 0.4) ** 2) / R
    light = 1.12 - 0.3 * np.clip(d, 0, 1.5)
    img = np.clip(img.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)
    out = np.dstack([img, alpha])
    return cv2.resize(out, (size // k, size // k), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=16)
def chip_side(value: int, r: int) -> np.ndarray:
    """The visible edge under a stacked chip: a darker disc."""
    spr = chip_sprite(value, r).copy()
    spr[..., :3] = (spr[..., :3].astype(np.float32) * 0.55).astype(np.uint8)
    return spr


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class StudioRenderer:
    FLIP = 0.42
    DEAL = 0.55

    def __init__(self, scale: int, source_size: tuple[int, int], pixel_art: bool = False, out_size: tuple[int, int] | None = None, fps: float = 30.0, tank_label: str = "") -> None:
        self.S = scale
        self.tank_label = tank_label
        self.W, self.H = W0 * scale, H0 * scale
        self.out_size = out_size or (W0, H0)
        self.pixel_art = pixel_art
        self.fps = fps
        self.rng = random.Random(11)
        self.card_w, self.card_h = self.s(150), self.s(210)
        # Aquarium window.
        self.win = (self.s(54), self.s(196), self.s(972), self.s(546))
        self.dealer_cy = self.s(978)
        self.fish_cy = self.s(1520)
        self.circle = (self.s(540), self.s(1712), self.s(84))
        self.shoe = (self.s(902), self.s(828), self.s(128), self.s(96))
        self.card_pose: dict[int, tuple[float, float, float]] = {}
        self.chip_motion: list[dict] = []
        self.last_result_at: float | None = None
        self.shown_bank: float | None = None
        self.background = self._make_background()
        light = cv2.resize(self._make_light(), self.out_size, interpolation=cv2.INTER_AREA)
        self.light_u8 = cv2.merge([np.clip(light / 1.25 * 255, 0, 255).astype(np.uint8)] * 3)
        self.luts = self._make_luts()
        self.grain = []
        for i in range(6):
            g = np.random.default_rng(i).normal(0, 2.6, (self.out_size[1], self.out_size[0])).astype(np.float32)
            g3 = cv2.merge([g, g, g])
            self.grain.append((np.clip(g3, 0, 255).astype(np.uint8), np.clip(-g3, 0, 255).astype(np.uint8)))
        self.frame_no = 0

    def s(self, v: float) -> int:
        return int(round(v * self.S))

    # Static scene -------------------------------------------------------------
    def _make_background(self) -> np.ndarray:
        W, H, S = self.W, self.H, self.S
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # Room: warm dark wall.
        room = np.ones((H, W, 3), np.float32) * _bgr("#141210")
        wall_light = np.exp(-(((xx - W * 0.5) / (W * 0.7)) ** 2 + ((yy - H * 0.18) / (H * 0.3)) ** 2))
        room += _bgr("#3B2E1E") * wall_light[..., None] * 0.9
        img = room

        # Felt: from below the window to the bottom, with the rail arc.
        felt_top = self.s(790)
        felt = np.ones((H, W, 3), np.float32) * _bgr(FELT)
        fibre = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7 * S)
        nap = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 3)
        nap = cv2.resize(nap, (W, H), interpolation=cv2.INTER_CUBIC)
        streak = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=0.6 * S, sigmaY=6 * S)
        felt *= (1 + 0.045 * fibre + 0.035 * nap + 0.02 * streak)[..., None]
        in_felt = (yy >= felt_top)
        img = np.where(in_felt[..., None], felt, img)
        # Dealer edge: a brass rule with a dark lip.
        cv2.rectangle(img, (0, felt_top - self.s(14)), (W, felt_top), tuple(float(v) for v in _bgr("#0C0A08")), -1)
        cv2.line(img, (0, felt_top - self.s(2)), (W, felt_top - self.s(2)), tuple(float(v) for v in _bgr(BRASS)), max(1, self.s(2)), cv2.LINE_AA)

        # Padded rail along the bottom arc.
        cx, cy, R = W * 0.5, self.s(560), self.s(1300)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rail_w = self.s(120)
        t = (d - R) / rail_w
        rail_mask = (t >= 0) & (yy > felt_top)
        shade = np.clip(t, 0, 1)
        rail = _bgr(RAIL) * (0.7 + 0.6 * np.sin(np.clip(shade, 0, 1) * math.pi * 0.85))[..., None]
        leather = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2 * S)
        rail *= (1 + 0.06 * leather)[..., None]
        img = np.where(rail_mask[..., None], rail, img)
        # Rail highlight and stitching.
        hl = np.exp(-(((d - R - rail_w * 0.32) / (rail_w * 0.08)) ** 2)) * (yy > felt_top)
        img += _bgr(RAIL_HI) * hl[..., None] * 0.6
        lip = np.exp(-(((d - R) / self.s(3)) ** 2)) * (yy > felt_top)
        img = img * (1 - 0.6 * lip[..., None])
        stitch = (np.abs(d - R - rail_w * 0.12) < self.s(1.6)) & ((np.arctan2(yy - cy, xx - cx) * R / self.s(9)) % 2 < 1.1) & (yy > felt_top)
        img[stitch] = img[stitch] * 0.4 + _bgr(BRASS) * 0.6

        img = np.clip(img, 0, 255).astype(np.uint8)
        self._print_felt(img)
        self._draw_window_frame(img)
        self._draw_shoe(img)
        return img

    def _arc_text(self, img: np.ndarray, text: str, cx: float, apex_y: float, radius: float, size: int, colour: str, tracking: float) -> None:
        """Gold-foil caps set along a gentle smile, lowest at the centre like a real table print."""
        sprites = [text_sprite(ch, "serif", size, colour) if ch != " " else None for ch in text]
        widths = [sp.shape[1] if sp is not None else size * 0.32 for sp in sprites]
        total = sum(widths) + tracking * (len(text) - 1)
        centre_y = apex_y - radius
        angle = -total / 2 / radius
        for sp, w in zip(sprites, widths):
            a = angle + (w / 2) / radius
            if sp is not None:
                x = cx + radius * math.sin(a)
                y = centre_y + radius * math.cos(a)
                blit(img, rotate_sprite(sp, math.degrees(a)), x, y, "cc", alpha=0.85)
            angle += (w + tracking) / radius

    def _print_felt(self, img: np.ndarray) -> None:
        W = self.W
        # Arc print sits between the dealer's cards and the fish's cards.
        self._arc_text(img, "BLACKJACK PAYS 3 TO 2", W / 2, self.s(1176), self.s(1100), self.s(40), BRASS, self.s(7))
        line = text_sprite("Dealer must stand on all 17s", "serif_italic", self.s(34), mix_hex(BRASS, FELT, 0.15))
        blit(img, line, W / 2, self.s(1226), "cc", alpha=0.75)
        # Betting circle.
        cx, cy, r = self.circle
        cv2.circle(img, (cx, cy), r, hex_bgr(BRASS), max(1, self.s(2)), cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r - self.s(9), hex_bgr(mix_hex(BRASS, FELT, 0.5)), max(1, self.s(1)), cv2.LINE_AA)

    def _draw_window_frame(self, img: np.ndarray) -> None:
        x, y, w, h = self.win
        S = self.S
        bezel = self.s(18)
        R = self.s(26)
        # Drop shadow on the wall.
        sh = np.zeros((h + bezel * 2 + self.s(120), w + bezel * 2 + self.s(120)), np.float32)
        o = self.s(60)
        sh[o : o + h + bezel * 2, o : o + w + bezel * 2] = round_mask(w + bezel * 2, h + bezel * 2, R + bezel)
        sh = cv2.GaussianBlur(sh, (0, 0), self.s(22))
        x0, y0 = x - bezel - o, y - bezel - o + self.s(18)
        roi = img[max(0, y0) : y0 + sh.shape[0], max(0, x0) : x0 + sh.shape[1]]
        part = sh[max(0, -y0) : max(0, -y0) + roi.shape[0], max(0, -x0) : max(0, -x0) + roi.shape[1]]
        roi[:] = (roi * (1 - 0.75 * part[..., None])).astype(np.uint8)
        # Brass bezel with a vertical gradient and an inner bevel.
        bw, bh = w + bezel * 2, h + bezel * 2
        mask = round_mask(bw, bh, R + bezel)
        t = np.linspace(0, 1, bh, dtype=np.float32)[:, None, None]
        grad = _bgr(BRASS_HI) * (1 - t) * 0.6 + _bgr(BRASS) * 0.55 + _bgr(BRASS_LO) * t * 0.5
        grad = np.repeat(grad, bw, axis=1)
        brushed = cv2.GaussianBlur(np.random.default_rng(5).normal(0, 1, (bh, bw)).astype(np.float32), (0, 0), sigmaX=8 * S, sigmaY=0.5 * S)
        grad *= (1 + 0.05 * brushed)[..., None]
        bev = np.clip(mask - np.pad(round_mask(bw - 4 * S, bh - 4 * S, R + bezel - 2 * S), 2 * S), 0, 1)
        grad = grad * (1 - 0.4 * bev[..., None]) + _bgr(BRASS_HI) * 0.4 * bev[..., None]
        roi = img[y - bezel : y - bezel + bh, x - bezel : x - bezel + bw]
        roi[:] = (roi * (1 - mask[..., None]) + np.clip(grad, 0, 255) * mask[..., None]).astype(np.uint8)
        # Screw heads.
        for sx, sy in ((x - bezel // 2, y - bezel // 2), (x + w + bezel // 2, y - bezel // 2), (x - bezel // 2, y + h + bezel // 2), (x + w + bezel // 2, y + h + bezel // 2)):
            cv2.circle(img, (sx + (R // 3 if sx < x else -R // 3), sy + (R // 3 if sy < y else -R // 3)), self.s(5), hex_bgr(BRASS_LO), -1, cv2.LINE_AA)
            cv2.circle(img, (sx + (R // 3 if sx < x else -R // 3), sy + (R // 3 if sy < y else -R // 3)), self.s(5), hex_bgr(BRASS_HI), max(1, S), cv2.LINE_AA)
        self.win_mask = round_mask(w, h, R)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Glass: inner shadow at the rim and a faint diagonal reflection.
        edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
        self.win_shade = (0.62 + 0.38 * np.clip(edge / self.s(70), 0, 1) ** 0.7).astype(np.float32)
        streak = np.exp(-(((xx * 0.55 + yy - h * 0.15) / self.s(60)) ** 2)) * 0.035 + np.exp(-(((xx * 0.55 + yy - h * 0.42) / self.s(16)) ** 2)) * 0.025
        self.win_glare = streak.astype(np.float32)

    def _draw_shoe(self, img: np.ndarray) -> None:
        x, y, w, h = self.shoe
        R = self.s(12)
        body = np.ones((h, w, 3), np.float32) * _bgr("#1B1512")
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        body = body * (1.3 - 0.5 * t)
        m = round_mask(w, h, R)
        roi = img[y : y + h, x : x + w]
        roi[:] = (roi * (1 - m[..., None]) + body * m[..., None]).astype(np.uint8)
        cv2.rectangle(img, (x + self.s(14), y + self.s(16)), (x + w - self.s(14), y + self.s(26)), hex_bgr(BRASS), -1, cv2.LINE_AA)
        back = card_back(self.s(64), self.s(90))
        blit(img, rotate_sprite(back, 90), x + w / 2, y + h / 2 + self.s(10), "cc", alpha=0.9)

    def _make_light(self) -> np.ndarray:
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        lamp = np.exp(-(((xx - W * 0.5) / (W * 0.62)) ** 2 + ((yy - H * 0.66) / (H * 0.30)) ** 2))
        vignette = 1 - 0.55 * np.clip((((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.64)) ** 2) - 0.25, 0, 1)
        light = (0.62 + 0.48 * lamp) * vignette
        return light.astype(np.float32)

    # Per-frame helpers ---------------------------------------------------------
    def _pose(self, tc) -> tuple[float, float, float]:
        key = id(tc)
        if key not in self.card_pose:
            self.card_pose[key] = (self.rng.uniform(-3.2, 3.2), self.rng.uniform(-5, 5) * self.S, self.rng.uniform(-4, 4) * self.S)
        return self.card_pose[key]

    def _draw_hand(self, canvas, hand, cy: int, now: float) -> None:
        dealt = [c for c in hand if now >= c.dealt_at]
        n = len(dealt)
        if not n:
            return
        spacing = min(self.card_w * 0.62, self.s(620) / max(1, n))
        x0 = self.W / 2 - spacing * (n - 1) / 2
        sx = self.shoe[0] + self.shoe[2] / 2
        sy = self.shoe[1] + self.shoe[3] / 2
        for i, tc in enumerate(dealt):
            angle_rest, jx, jy = self._pose(tc)
            tx, ty = x0 + i * spacing + jx, cy + jy
            p = _ease_out_quart((now - tc.dealt_at) / self.DEAL)
            x = sx + (tx - sx) * p
            y = sy + (ty - sy) * p
            angle = angle_rest + (1 - p) * 34
            lift = (1 - p) * self.s(26)
            face = card_face(tc.card.rank, tc.card.suit, self.card_w, self.card_h)
            back = card_back(self.card_w, self.card_h)
            sxs = 1.0
            sprite = face if tc.face_up else back
            if not tc.face_up and tc.flip_at is not None and now >= tc.flip_at:
                f = _clamp((now - tc.flip_at) / self.FLIP)
                sxs = max(0.02, abs(math.cos(math.pi * f)))
                sprite = face if f >= 0.5 else back
                lift = max(lift, math.sin(math.pi * f) * self.s(18))
            elif tc.face_up and p < 1:
                f = _clamp((now - tc.dealt_at) / (self.DEAL * 0.8))
                sxs = max(0.02, abs(math.cos(math.pi * f)))
                sprite = face if f >= 0.5 else back
            shadow = soft_shadow(self.card_w, self.card_h, self.s(14))
            off = self.s(6) + lift * 0.9
            blit(canvas, rotate_sprite(shadow, angle, sxs), x + off * 0.5, y + off, "cc", alpha=0.55 + 0.2 * (1 - lift / max(1, self.s(26))))
            blit(canvas, rotate_sprite(sprite, angle, sxs), x, y - lift, "cc")

    def _draw_stack(self, canvas, amount: int, cx: float, cy: float, alpha: float = 1.0) -> None:
        chips = chip_breakdown(amount)
        r = self.s(40)
        for i, v in enumerate(chips[::-1]):
            dy = -i * self.s(7)
            blit(canvas, chip_side(v, r), cx, cy + dy + self.s(5), "cc", alpha=alpha)
            blit(canvas, chip_sprite(v, r), cx, cy + dy, "cc", alpha=alpha)

    def _label(self, canvas, word: str, value: str, y: int, colour: str) -> None:
        x = self.s(84)
        blit(canvas, text_sprite(word, "mono_light", self.s(20), CREAM_DIM, tracking=self.s(6)), x, y - self.s(34), "lb")
        if value:
            blit(canvas, text_sprite(value, "serif", self.s(70), colour), x, y + self.s(30), "lb")

    # Frame --------------------------------------------------------------------
    def render(self, source: np.ndarray, zones, counts: list[int], now: float, table, hand_label: str) -> np.ndarray:
        canvas = self.background.copy()
        self._draw_aquarium(canvas, source, zones, counts, table, now)
        self._draw_header(canvas, table, hand_label)
        if self.tank_label:
            x, y, w, h = self.win
            blit(canvas, text_sprite(self.tank_label, "mono_light", self.s(18), CREAM_DIM, tracking=self.s(5)), x + w, y + h + self.s(40), "rc", alpha=0.8)

        # Hands and totals.
        dealer_vis = table.visible(table.dealer, now)
        hidden = any(now >= c.dealt_at and not c.showing(now) for c in table.dealer)
        fish_vis = table.visible(table.fish, now)
        d_val = (str(hand_value(dealer_vis)[0]) if hidden else describe(dealer_vis).title()) if dealer_vis else ""
        f_val = describe(fish_vis).title() if fish_vis else ""
        f_total = hand_value(fish_vis)[0] if fish_vis else 0
        self._label(canvas, "DEALER", d_val, self.dealer_cy - self.s(40), LOSS if dealer_vis and hand_value(dealer_vis)[0] > 21 else CREAM)
        self._label(canvas, "THE FISH", f_val, self.fish_cy - self.s(40), LOSS if f_total > 21 else (WIN if f_total == 21 else CREAM))
        self._draw_hand(canvas, table.dealer, self.dealer_cy, now)
        self._draw_hand(canvas, table.fish, self.fish_cy, now)

        # Chips.
        cx, cy, _r = self.circle
        if table.bet:
            self._draw_chips(canvas, table, now, cx, cy)
        self._draw_message(canvas, table, now)

        # Downsample the supersampled frame, then light, grade, and grain at output size.
        out = cv2.resize(canvas, self.out_size, interpolation=cv2.INTER_AREA) if (self.W, self.H) != self.out_size else canvas
        breathe = 1 + 0.012 * math.sin(now * 0.9)
        out = cv2.multiply(out, self.light_u8, scale=1.25 / 255 * breathe)
        plus, minus = self.grain[self.frame_no % len(self.grain)]
        out = cv2.subtract(cv2.add(out, plus), minus)
        self.frame_no += 1
        b, g, r = cv2.split(out)
        return cv2.merge([cv2.LUT(b, self.luts[0]), cv2.LUT(g, self.luts[1]), cv2.LUT(r, self.luts[2])])

    @staticmethod
    def _make_luts() -> list[np.ndarray]:
        """Film-style grade as per-channel curves: soft S-curve, lifted blacks,
        teal in the shadows, warm amber in the highlights."""
        x = np.arange(256, dtype=np.float32) / 255
        s_curve = x * x * (3 - 2 * x)
        base = 0.8 * x + 0.2 * s_curve
        base = 0.025 + base * 0.965
        shadows = (1 - x) ** 2
        highs = x ** 2
        curves = [
            base + 0.030 * shadows - 0.035 * highs,  # blue
            base + 0.012 * shadows - 0.004 * highs,  # green
            base - 0.020 * shadows + 0.030 * highs,  # red
        ]
        return [np.clip(c * 255, 0, 255).astype(np.uint8) for c in curves]

    def _draw_header(self, canvas, table, hand_label: str) -> None:
        blit(canvas, text_sprite("The Fish Table", "serif_italic", self.s(62), CREAM), self.s(56), self.s(64), "lt")
        blit(canvas, text_sprite(hand_label, "mono_light", self.s(20), CREAM_DIM, tracking=self.s(6)), self.s(60), self.s(150), "lt")
        if self.shown_bank is None:
            self.shown_bank = float(table.bankroll)
        self.shown_bank += (table.bankroll - self.shown_bank) * 0.1
        if abs(table.bankroll - self.shown_bank) < 0.5:
            self.shown_bank = float(table.bankroll)
        bank = text_sprite(f"{int(round(self.shown_bank)):,}", "serif", self.s(66), CREAM)
        blit(canvas, bank, self.W - self.s(56), self.s(62), "rt")
        blit(canvas, text_sprite("CHIPS", "mono_light", self.s(20), CREAM_DIM, tracking=self.s(6)), self.W - self.s(58), self.s(150), "rt")

    def _draw_aquarium(self, canvas, source, zones, counts, table, now) -> None:
        x, y, w, h = self.win
        src_h, src_w = source.shape[:2]
        # Grade at source size (cheap), then scale into the window.
        if not hasattr(self, "_tank_matrix"):
            sat, tint = 0.95, np.array((1.08, 1.02, 0.86), np.float32)  # BGR: cool teal water
            grey = np.full((3, 3), (1 - sat) / 3, np.float32)
            self._tank_matrix = (grey + np.eye(3, dtype=np.float32) * sat) * tint[:, None]
            self._win_shade_u8 = cv2.merge([np.clip(self.win_shade * 255, 0, 255).astype(np.uint8)] * 3)
            self._win_glare_u8 = cv2.merge([np.clip(self.win_glare * 255, 0, 255).astype(np.uint8)] * 3)
        graded = cv2.transform(source, self._tank_matrix)
        graded = cv2.convertScaleAbs(graded, alpha=1.14, beta=-6)
        scale = max(w / src_w, h / src_h)
        sw, sh = int(src_w * scale), int(src_h * scale)
        interp = cv2.INTER_NEAREST if self.pixel_art else cv2.INTER_CUBIC
        cam = cv2.resize(graded, (sw, sh), interpolation=interp)
        ox, oy = (sw - w) // 2, (sh - h) // 2
        cam = np.ascontiguousarray(cam[oy : oy + h, ox : ox + w])

        vote = table.vote
        if vote is not None and zones and vote.winner is not None:
            t = _clamp((now - vote.decided_at) / 0.5)
            z = zones[vote.winner]
            zx0 = max(0, int(z.x * scale) - ox)
            zx1 = min(w, int((z.x + z.w) * scale) - ox)
            band = cam[:, zx0:zx1].copy()
            cam = cv2.convertScaleAbs(cam, alpha=1 - 0.5 * t)
            if zx1 > zx0:
                cam[:, zx0:zx1] = cv2.convertScaleAbs(band, alpha=1 + 0.08 * t, beta=6 * t)
        cam = cv2.multiply(cam, self._win_shade_u8, scale=1 / 255)
        cam = cv2.add(cam, self._win_glare_u8)
        if vote is not None and zones:
            self._draw_vote_overlay(cam, zones, counts, vote, scale, ox, now)
        # Copy in, then blend only the rounded corners.
        R = self.s(26)
        canvas[y : y + h, x : x + w] = cam
        bg = self.background
        for cx0, cy0 in ((0, 0), (w - R, 0), (0, h - R), (w - R, h - R)):
            m = np.ascontiguousarray(self.win_mask[cy0 : cy0 + R, cx0 : cx0 + R])
            patch = np.ascontiguousarray(cam[cy0 : cy0 + R, cx0 : cx0 + R])
            under = np.ascontiguousarray(bg[y + cy0 : y + cy0 + R, x + cx0 : x + cx0 + R])
            canvas[y + cy0 : y + cy0 + R, x + cx0 : x + cx0 + R] = cv2.blendLinear(patch, under, m, 1.0 - m)

    def _draw_vote_overlay(self, cam, zones, counts, vote, scale, ox, now) -> None:
        h, w = cam.shape[:2]
        S = self.S
        # Bottom scrim for the labels.
        scrim_h = self.s(150)
        if not hasattr(self, "_scrim"):
            t = np.linspace(0, 1, scrim_h, dtype=np.float32)[:, None]
            self._scrim = cv2.merge([np.repeat(((1 - 0.62 * t ** 1.4) * 255).astype(np.uint8), w, axis=1)] * 3)
        cam[h - scrim_h :] = cv2.multiply(cam[h - scrim_h :], self._scrim, scale=1 / 255)
        leader = None
        if counts and max(counts) > 0 and counts.count(max(counts)) == 1:
            leader = counts.index(max(counts))
        for i, z in enumerate(zones):
            zx0 = int(z.x * scale) - ox
            zx1 = int((z.x + z.w) * scale) - ox
            if i > 0:
                # Hairline divider, dashed.
                xd = max(0, min(w - 1, zx0 - self.s(6)))
                for yd in range(self.s(20), h - self.s(20), self.s(18)):
                    cv2.line(cam, (xd, yd), (xd, yd + self.s(9)), hex_bgr(mix_hex(CREAM, "#000000", 0.35)), max(1, S), cv2.LINE_AA)
            mid = (max(0, zx0) + min(w, zx1)) / 2
            chosen = vote.winner == i
            lead = leader == i and vote.winner is None
            colour = WIN if chosen or lead else CREAM
            label = text_sprite(z.ticker, "caps", self.s(46), colour, tracking=self.s(3))
            blit(cam, label, mid, h - self.s(66), "cb", alpha=1.0 if (chosen or lead or vote.winner is None) else 0.45)
            n = counts[i] if i < len(counts) else 0
            dot_r = self.s(5)
            gap = self.s(16)
            start = mid - gap * (max(n, 1) - 1) / 2
            for k in range(n):
                cv2.circle(cam, (int(start + k * gap), h - self.s(40)), dot_r, hex_bgr(colour), -1, cv2.LINE_AA)
            if lead or chosen:
                cv2.line(cam, (int(mid - label.shape[1] / 2), h - self.s(58)), (int(mid + label.shape[1] / 2), h - self.s(58)), hex_bgr(colour), max(1, S), cv2.LINE_AA)
        # Countdown: a hairline along the top and the seconds in the corner.
        if vote.started is not None and vote.winner is None:
            remaining = max(0.0, vote.seconds - (now - vote.started))
            frac = remaining / max(0.01, vote.seconds)
            colour = LOSS if remaining <= 3 else CREAM
            cv2.line(cam, (self.s(30), self.s(28)), (w - self.s(30), self.s(28)), hex_bgr(mix_hex(CREAM, "#000000", 0.6)), max(1, self.s(2)), cv2.LINE_AA)
            cv2.line(cam, (self.s(30), self.s(28)), (self.s(30) + int((w - self.s(60)) * frac), self.s(28)), hex_bgr(colour), max(1, self.s(3)), cv2.LINE_AA)
            sec = text_sprite(f"0:{int(math.ceil(remaining - 1e-9)):02d}", "mono", self.s(26), colour, tracking=self.s(2))
            blit(cam, sec, w - self.s(30), self.s(46), "rt")

    def _draw_chips(self, canvas, table, now, cx, cy) -> None:
        result = table.result
        base = table.bet
        if result is None:
            self.last_result_at = None
            self._draw_stack(canvas, base, cx, cy)
            return
        outcome, delta, at = result
        t = now - at
        dealer_xy = (self.W / 2, self.dealer_cy - self.s(60))
        if delta < 0:
            # Chips slide away to the dealer.
            p = _ease_out_quart((t - 0.5) / 0.7)
            x = cx + (dealer_xy[0] - cx) * p
            y = cy + (dealer_xy[1] - cy) * p
            self._draw_stack(canvas, base, x, y, alpha=1 - p * 0.9)
        else:
            self._draw_stack(canvas, base, cx - (self.s(52) if delta > 0 else 0) * _clamp((t - 0.3) / 0.3), cy)
            if delta > 0:
                p = _ease_out_quart((t - 0.5) / 0.8)
                x = dealer_xy[0] + (cx + self.s(52) - dealer_xy[0]) * p
                y = dealer_xy[1] + (cy - dealer_xy[1]) * p
                self._draw_stack(canvas, delta, x, y, alpha=_clamp(p * 3))

    def _draw_message(self, canvas, table, now) -> None:
        cy = self.s(1302)
        W = self.W
        vote = table.vote
        if vote is not None and vote.winner is None and vote.started is not None:
            labels = [l.lower() for l in vote.labels]
            if labels and labels[0].isdigit():
                q = "How much do they bet?"
                sub = "Chips on the line"
            else:
                q = (", ".join(l.capitalize() for l in labels[:-1]) + " or " + labels[-1] + "?") if len(labels) > 1 else labels[0]
                q = q[0].upper() + q[1:].lower()
                fish = table.visible(table.fish, now)
                dealer = table.visible(table.dealer, now)
                up = dealer[0].rank if dealer else ""
                sub = f"{describe(fish).lower()} against {RANK_NAMES.get(up, 'a ' + up)}" if fish and up else ""
            t = _clamp((now - vote.started) / 0.4)
            blit(canvas, fit_text(q, "serif", W - self.s(160), self.s(84), CREAM), W / 2, cy, "cc", alpha=t)
            if sub:
                blit(canvas, text_sprite(sub.upper(), "mono_light", self.s(20), CREAM_DIM, tracking=self.s(6)), W / 2, cy + self.s(66), "cc", alpha=t)
            return
        if vote is not None and vote.winner is not None:
            t = now - vote.decided_at
            word = vote.labels[vote.winner]
            word = word.capitalize() + "." if not word.isdigit() else f"{word} chips."
            blit(canvas, text_sprite("THE FISH SAY", "mono_light", self.s(20), CREAM_DIM, tracking=self.s(8)), W / 2, cy + self.s(84), "cc", alpha=_clamp(t / 0.25))
            sc = 1.06 - 0.06 * ease_out_cubic(t / 0.6)
            blit(canvas, text_sprite(word, "serif_italic", self.s(120), WIN), W / 2, cy + self.s(8), "cc", alpha=_clamp(t / 0.3), scale=sc)
            return
        if table.result is not None:
            outcome, delta, at = table.result
            t = now - at
            line = RESULT_LINES.get(outcome, outcome)
            colour = WIN if delta > 0 else (CREAM if delta == 0 else LOSS)
            sc = 1.05 - 0.05 * ease_out_cubic(t / 0.7)
            blit(canvas, text_sprite(line, "serif_italic", self.s(112), colour), W / 2, cy, "cc", alpha=_clamp(t / 0.35), scale=sc)
            sub = f"{'+' if delta > 0 else ''}{delta} CHIPS" if delta else "BET RETURNED"
            blit(canvas, text_sprite(sub, "mono_light", self.s(22), colour, tracking=self.s(8)), W / 2, cy + self.s(80), "cc", alpha=_clamp((t - 0.3) / 0.4))
            return
        if table.status:
            text = table.status.replace("  ·  ", " · ").lower()
            text = text[0].upper() + text[1:]
            if "session over" in text.lower():
                blit(canvas, text_sprite("Session over.", "serif_italic", self.s(96), CREAM), W / 2, cy, "cc")
                diff = table.bankroll - table.start_bankroll
                sub = f"STARTED {table.start_bankroll}  ·  FINISHED {table.bankroll}  ·  {'+' if diff >= 0 else ''}{diff}"
                blit(canvas, text_sprite(sub, "mono_light", self.s(22), WIN if diff > 0 else LOSS if diff < 0 else CREAM, tracking=self.s(6)), W / 2, cy + self.s(76), "cc")
            else:
                blit(canvas, text_sprite(text, "serif_italic", self.s(54), CREAM_DIM), W / 2, cy, "cc", alpha=0.9)
