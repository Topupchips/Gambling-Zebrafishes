"""The fish play blackjack. Play chips only.

A local blackjack table runs on screen. A real dealer follows house rules
(see blackjack_game.py), and every decision the player would make goes to the fish:
when the clock runs out, the busiest zone in the tank decides HIT, STAND, or DOUBLE.
They can also pick the bet size before each hand.

    python blackjack.py --demo --hands 3           # pixel tank, nothing saved
    python blackjack.py --hands 5 --fish-bet       # real camera, bankroll saved

Zones: draw them with calibrate.py, left to right. The first zone is HIT, the second
STAND, the third DOUBLE (and the bet sizes on a bet vote). With only two zones the
fish can't double.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

import demo as real_demo
import pixel_tank
from blackjack_game import (
    BLACKJACK_PATH,
    OUTCOME_TEXT,
    Card,
    Shoe,
    dealer_peeks,
    dealer_should_hit,
    describe,
    hand_value,
    is_blackjack,
    load_bankroll,
    record_hand,
    save_bankroll,
    settle,
)
from hud import (
    DIM,
    GOLD,
    GOLD_DEEP,
    HOT,
    INK,
    MUTED,
    SHADOW,
    Hud,
    Layout,
    RoundInfo,
    blit,
    clamp,
    ease_out_back,
    ease_out_cubic,
    fit_text,
    glow_layer,
    glow_text,
    hex_bgr,
    mix_hex,
    paint,
    poker_chip,
    ring_mask,
    round_mask,
    round_rect,
    round_stroke,
    text_sprite,
    zone_hex,
)
from pick import (
    MotionDetector,
    centroids_from_mask,
    choose_winner,
    count_zones,
    even_frame,
    frame_mask,
    load_background,
    open_video,
    open_writer,
    phone_copy,
)
from studio import StudioRenderer
from utils import CONFIG_PATH, RECORDINGS_DIR, Zone, die, file_stamp, load_config, open_camera

TABLE_LAYOUT = Layout(
    name="reel",
    size=(1080, 1920),
    header=(48, 44, 984, 210),
    camera_box=(40, 250, 1000, 560),
    clock=(0, 0, 0, 0),
    board=(0, 0, 0, 0),
    tape=(0, 1772, 1080, 64),
    footer_y=1876,
    reveal=(0, 0, 0, 0),
    k=1.0,
)

CARD_W, CARD_H = 132, 186
TABLE_FELT = "#0F3A28"
RED_INK = "#D42A3C"
BLACK_INK = "#1B1B22"
WIN_GREEN = "#3DFF8F"


# ---------------------------------------------------------------------------
# Card art
# ---------------------------------------------------------------------------
def _suit_mask(suit: str, size: int) -> np.ndarray:
    """Anti-aliased suit symbol as a float mask (size x size)."""
    s = size * 4
    m = np.zeros((s, s), np.uint8)

    def P(x: float, y: float) -> tuple[int, int]:
        return int(x * s), int(y * s)

    def circle(x: float, y: float, r: float) -> None:
        cv2.circle(m, P(x, y), int(r * s), 255, -1, cv2.LINE_AA)

    def poly(*pts: tuple[float, float]) -> None:
        cv2.fillPoly(m, [np.array([P(x, y) for x, y in pts], np.int32)], 255, cv2.LINE_AA)

    if suit == "H":
        circle(0.29, 0.34, 0.23)
        circle(0.71, 0.34, 0.23)
        poly((0.07, 0.40), (0.93, 0.40), (0.5, 0.94))
    elif suit == "D":
        poly((0.5, 0.03), (0.88, 0.5), (0.5, 0.97), (0.12, 0.5))
    elif suit == "S":
        circle(0.29, 0.58, 0.22)
        circle(0.71, 0.58, 0.22)
        poly((0.08, 0.55), (0.92, 0.55), (0.5, 0.04))
        poly((0.5, 0.6), (0.33, 0.97), (0.67, 0.97))
    else:  # clubs
        circle(0.5, 0.29, 0.21)
        circle(0.27, 0.6, 0.21)
        circle(0.73, 0.6, 0.21)
        circle(0.5, 0.55, 0.12)
        poly((0.5, 0.55), (0.33, 0.97), (0.67, 0.97))
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def _stamp(canvas: np.ndarray, mask: np.ndarray, cx: float, cy: float, colour: str, flip: bool = False) -> None:
    """Paint a suit mask onto a BGR canvas, centred on (cx, cy)."""
    if flip:
        mask = mask[::-1, ::-1]
    h, w = mask.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    roi = canvas[y0 : y0 + h, x0 : x0 + w]
    if roi.shape[:2] != mask.shape:
        return
    col = np.array(hex_bgr(colour), np.float32)
    a = mask[..., None]
    roi[:] = (roi * (1 - a) + col * a).astype(np.uint8)


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


PAPER = "#FBF7EE"


def _with_alpha(rgb: np.ndarray, radius: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = (round_mask(w, h, radius) * 255).astype(np.uint8)
    return out


@lru_cache(maxsize=64)
def card_face(rank: str, suit: str) -> np.ndarray:
    k = 3
    W, H = CARD_W * k, CARD_H * k
    card = np.zeros((H, W, 3), np.uint8)
    card[:] = hex_bgr(PAPER)
    ink = RED_INK if suit in ("H", "D") else BLACK_INK
    round_stroke(card, 0, 0, W, H, 14 * k, 3 * k, "#C9BFAA", 0.6)

    # Corner index, top-left, and the same block turned upside down bottom-right.
    corner = np.zeros((60 * k, 30 * k, 3), np.uint8)
    corner[:] = hex_bgr(PAPER)
    rank_sprite = fit_text(rank, "display", 27 * k, 32 * k, ink)
    blit(corner, rank_sprite, corner.shape[1] / 2, 2 * k, "ct")
    _stamp(corner, _suit_mask(suit, 18 * k), corner.shape[1] / 2, 6 * k + rank_sprite.shape[0] + 10 * k, ink)
    ch, cw = corner.shape[:2]
    card[7 * k : 7 * k + ch, 6 * k : 6 * k + cw] = corner
    card[H - 7 * k - ch : H - 7 * k, W - 6 * k - cw : W - 6 * k] = corner[::-1, ::-1]

    # Face.
    ix0, ix1 = W * 0.38, W * 0.62
    iy0, iy1 = H * 0.17, H * 0.83
    if rank == "A":
        _stamp(card, _suit_mask(suit, 66 * k), W / 2, H / 2, ink)
    elif rank in ("J", "Q", "K"):
        fx, fy, fw, fh = int(W * 0.31), int(H * 0.13), int(W * 0.38), int(H * 0.74)
        round_rect(card, fx, fy, fw, fh, 8 * k, mix_hex(ink, PAPER, 0.88), 1.0)
        round_stroke(card, fx, fy, fw, fh, 8 * k, 2 * k, ink, 0.9)
        blit(card, text_sprite(rank, "display", 70 * k, ink), W / 2, H / 2, "cc")
        _stamp(card, _suit_mask(suit, 16 * k), fx + 13 * k, fy + 15 * k, ink)
        _stamp(card, _suit_mask(suit, 16 * k), fx + fw - 13 * k, fy + fh - 15 * k, ink, flip=True)
    else:
        pip = _suit_mask(suit, 23 * k)
        for px, py in PIPS[int(rank)]:
            _stamp(card, pip, ix0 + (ix1 - ix0) * px, iy0 + (iy1 - iy0) * py, ink, flip=py > 0.5)
    small = cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    return _with_alpha(small, 14)


@lru_cache(maxsize=1)
def card_back() -> np.ndarray:
    k = 3
    W, H = CARD_W * k, CARD_H * k
    card = np.zeros((H, W, 3), np.uint8)
    card[:] = hex_bgr(PAPER)
    inset = 9 * k
    iw, ih = W - inset * 2, H - inset * 2
    inner = np.zeros((ih, iw, 3), np.uint8)
    inner[:] = hex_bgr("#A3132B")
    for d in range(-ih, iw, 14 * k):
        cv2.line(inner, (d, 0), (d + ih, ih), hex_bgr("#C2304A"), 2 * k, cv2.LINE_AA)
        cv2.line(inner, (d + ih, 0), (d, ih), hex_bgr("#C2304A"), 2 * k, cv2.LINE_AA)
    axes = (int(iw * 0.32), int(ih * 0.2))
    cv2.ellipse(inner, (iw // 2, ih // 2), axes, 0, 0, 360, hex_bgr("#7A0C1F"), -1, cv2.LINE_AA)
    cv2.ellipse(inner, (iw // 2, ih // 2), axes, 0, 0, 360, hex_bgr(GOLD), 2 * k, cv2.LINE_AA)
    blit(inner, text_sprite("FF", "display", 40 * k, GOLD, tracking=2 * k), iw / 2, ih / 2, "cc")
    mask = round_mask(iw, ih, 9 * k)[..., None]
    region = card[inset : inset + ih, inset : inset + iw]
    region[:] = (region * (1 - mask) + inner * mask).astype(np.uint8)
    round_stroke(card, inset, inset, iw, ih, 9 * k, 2 * k, GOLD, 0.9)
    round_stroke(card, 0, 0, W, H, 14 * k, 3 * k, "#C9BFAA", 0.6)
    small = cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    return _with_alpha(small, 14)


@lru_cache(maxsize=1)
def card_shadow() -> np.ndarray:
    pad = 24
    m = np.zeros((CARD_H + pad * 2, CARD_W + pad * 2), np.float32)
    m[pad : pad + CARD_H, pad : pad + CARD_W] = round_mask(CARD_W, CARD_H, 14)
    m = cv2.GaussianBlur(m, (31, 31), 9)
    out = np.zeros((*m.shape, 4), np.uint8)
    out[..., :3] = hex_bgr(SHADOW)
    out[..., 3] = (m * 170).astype(np.uint8)
    return out


def _rotate(sprite: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 0.2:
        return sprite
    h, w = sprite.shape[:2]
    side = int(math.hypot(w, h)) + 4
    canvas = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = sprite
    M = cv2.getRotationMatrix2D((side / 2, side / 2), degrees, 1.0)
    return cv2.warpAffine(canvas, M, (side, side), flags=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Table state
# ---------------------------------------------------------------------------
DEAL_SECONDS = 0.42
FLIP_SECONDS = 0.36


@dataclass
class TableCard:
    card: Card
    dealt_at: float
    face_up: bool = True
    flip_at: float | None = None
    x: float | None = None

    def showing(self, now: float) -> bool:
        if self.face_up:
            return True
        return self.flip_at is not None and now >= self.flip_at + FLIP_SECONDS / 2


@dataclass
class Vote:
    labels: list[str]
    question: str
    seconds: float
    started: float | None = None
    samples: list = field(default_factory=list)
    winner: int | None = None
    decided_at: float = 0.0


@dataclass
class Table:
    bankroll: int
    start_bankroll: int
    hand_no: int = 0
    dealer: list[TableCard] = field(default_factory=list)
    fish: list[TableCard] = field(default_factory=list)
    bet: int = 0
    doubled: bool = False
    status: str = ""
    vote: Vote | None = None
    result: tuple[str, int, float] | None = None  # (outcome, delta, at)
    shown_bankroll: float = 0.0

    def visible(self, hand: list[TableCard], now: float) -> list[Card]:
        return [c.card for c in hand if now >= c.dealt_at and c.showing(now)]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class TableHud(Hud):
    def __init__(self, info: RoundInfo, source_size: tuple[int, int], started: float, pixel_art: bool) -> None:
        super().__init__(TABLE_LAYOUT, info, source_size, started, pixel_art=pixel_art)
        top = self.cam_y + self.cam_h + 28
        self.table = (40, top, 1000, TABLE_LAYOUT.tape[1] - 22 - top)
        T = top
        self.dealer_label_y = T + 30
        self.dealer_cards_cy = T + 70 + CARD_H // 2
        self.strip = (64, T + 92 + CARD_H, 952, 178)
        self.fish_label_y = self.strip[1] + self.strip[3] + 18
        self.fish_cards_cy = self.fish_label_y + 40 + CARD_H // 2
        self.bottom_y = self.fish_cards_cy + CARD_H // 2 + 16
        self.shoe = (888, T + 52, 120, 150)
        self.clock = (0, 0, 0, 0)
        self.board = (0, 0, 0, 0)
        self.background = self._make_background()
        self._paint_table()
        self._tape_bankroll: int | None = None
        self.table_state: Table | None = None

    def _paint_table(self) -> None:
        bg = self.background
        x, y, w, h = self.table
        round_rect(bg, x, y, w, h, 40, TABLE_FELT, 0.95)
        layer = np.zeros((h + 80, w + 80, 3), np.uint8)
        paint(layer, 40, 40, ring_mask(w, h, 40, 4), GOLD, 1.0)
        glow_layer(bg, layer, x - 40, y - 40, 18, 0.35)
        round_stroke(bg, x, y, w, h, 40, 4, GOLD, 1.0)
        round_stroke(bg, x + 14, y + 14, w - 28, h - 28, 30, 1, GOLD_DEEP, 0.7)
        # Table print.
        line = text_sprite("BLACKJACK PAYS 3 TO 2  ·  DEALER STANDS ON ALL 17s", "bold", 22, GOLD, tracking=5)
        blit(bg, line, x + w / 2, y + h - 30, "cc", alpha=0.55)
        sx, sy, sw, sh = self.shoe
        round_rect(bg, sx, sy, sw, sh, 14, SHADOW, 0.85)
        round_stroke(bg, sx, sy, sw, sh, 14, 2, GOLD_DEEP, 0.9)
        back = card_back()
        small = cv2.resize(back, (int(CARD_W * 0.62), int(CARD_H * 0.62)), interpolation=cv2.INTER_AREA)
        for i in range(4):
            blit(bg, small, sx + sw / 2 - 6 + i * 3, sy + sh / 2 + 4 - i * 3, "cc")
        blit(bg, text_sprite("SHOE", "mono", 16, MUTED, tracking=3), sx + sw / 2, sy + sh + 14, "cc")

    # Tape --------------------------------------------------------------------
    def _tape_items(self) -> list[tuple[str, str]]:
        t = self.table_state
        bank = f"FISH BANKROLL {t.bankroll:,}" if t else "FISH BANKROLL"
        items = []
        for text in ("BLACKJACK PAYS 3 TO 2", "DEALER STANDS ON 17", "5 ZEBRAFISH", "NO HUMANS DECIDE", bank, "PLAY CHIPS ONLY"):
            items.append((text, GOLD if text in ("NO HUMANS DECIDE", bank) else INK))
            items.append(("   •   ", DIM))
        return items

    # Frame -------------------------------------------------------------------
    def render(self, source, zones, counts, centroids, owners, now: float, table: Table) -> np.ndarray:
        self.table_state = table
        if self._tape_bankroll != table.bankroll:
            self._tape_bankroll = table.bankroll
            self.state.tape_sprite = None
        canvas = self.background.copy()
        elapsed = now - self.state.started
        if counts:
            total = max(1, sum(counts))
            targets = [c / total for c in counts]
            if len(self.state.shares) != len(targets):
                self.state.shares = [0.0] * len(targets)
            self.state.shares = [s + (t - s) * 0.22 for s, t in zip(self.state.shares, targets)]
        leader = self._leader(counts) if counts else None
        self._draw_camera(canvas, source, zones, counts, centroids, owners, leader, now)
        self._draw_header(canvas, elapsed, now)
        self._draw_hands(canvas, table, now)
        self._draw_strip(canvas, table, counts, now)
        self._draw_bottom(canvas, table, now)
        self._draw_tape(canvas, now)
        self._draw_footer(canvas)
        self.confetti.draw(canvas, now)
        return canvas

    # Hands -------------------------------------------------------------------
    def _pill(self, canvas, x: int, y: int, label: str, value: str, colour: str) -> None:
        name = text_sprite(label, "bold", 26, MUTED, tracking=6)
        blit(canvas, name, x, y, "lc")
        if not value:
            return
        val = text_sprite(value, "display", 34, SHADOW)
        pw, ph = val.shape[1] + 30, val.shape[0] + 16
        px = x + name.shape[1] + 18
        round_rect(canvas, px, y - ph / 2, pw, ph, ph // 2, colour, 1.0)
        blit(canvas, val, px + pw / 2, y, "cc")

    def _draw_hand(self, canvas, hand: list[TableCard], cy: float, now: float) -> None:
        dealt = [c for c in hand if now >= c.dealt_at]
        n = len(dealt)
        if not n:
            return
        spacing = min(CARD_W * 0.78, 760 / max(1, n))
        x0 = 540 - spacing * (n - 1) / 2
        sx, sy = self.shoe[0] + self.shoe[2] / 2, self.shoe[1] + self.shoe[3] / 2
        shadow = card_shadow()
        for i, tc in enumerate(dealt):
            target = x0 + i * spacing
            if tc.x is None:
                tc.x = target
            tc.x += (target - tc.x) * 0.25
            p = ease_out_cubic((now - tc.dealt_at) / DEAL_SECONDS)
            x = sx + (tc.x - sx) * p
            y = sy + (cy - sy) * p
            angle = (1 - p) * 28 + (i - (n - 1) / 2) * 2.0
            showing = tc.showing(now)
            sprite = card_face(tc.card.rank, tc.card.suit) if showing else card_back()
            if not tc.face_up and tc.flip_at is not None and now >= tc.flip_at:
                f = clamp((now - tc.flip_at) / FLIP_SECONDS)
                sx_scale = abs(math.cos(math.pi * f))
                if sx_scale < 0.999:
                    sprite = cv2.resize(sprite, (max(2, int(CARD_W * sx_scale)), CARD_H), interpolation=cv2.INTER_AREA)
            elif tc.face_up and p < 1:
                # Dealt face up: flips over as it lands.
                f = clamp((now - tc.dealt_at) / DEAL_SECONDS)
                sx_scale = abs(math.cos(math.pi * f))
                sprite = card_face(tc.card.rank, tc.card.suit) if f > 0.5 else card_back()
                sprite = cv2.resize(sprite, (max(2, int(CARD_W * sx_scale)), CARD_H), interpolation=cv2.INTER_AREA)
            sprite = _rotate(sprite, angle)
            lift = (1 - p) * 30
            blit(canvas, _rotate(shadow, angle), x + 8, y + 14, "cc", alpha=0.9)
            blit(canvas, sprite, x, y - lift, "cc")

    def _draw_hands(self, canvas, table: Table, now: float) -> None:
        dealer_cards = table.visible(table.dealer, now)
        hidden = any(now >= c.dealt_at and not c.showing(now) for c in table.dealer)
        dealer_text = describe(dealer_cards) if dealer_cards else ""
        if hidden and dealer_cards:
            dealer_text = str(hand_value(dealer_cards)[0])
        fish_cards = table.visible(table.fish, now)
        fish_text = describe(fish_cards) if fish_cards else ""
        fish_total = hand_value(fish_cards)[0] if fish_cards else 0
        fish_colour = HOT if fish_total > 21 else (GOLD if fish_total == 21 else INK)
        dealer_colour = HOT if dealer_cards and hand_value(dealer_cards)[0] > 21 else INK
        self._pill(canvas, 84, self.dealer_label_y, "DEALER", dealer_text, dealer_colour)
        self._pill(canvas, 84, self.fish_label_y, "THE FISH", fish_text, fish_colour)
        self._draw_hand(canvas, table.dealer, self.dealer_cards_cy, now)
        self._draw_hand(canvas, table.fish, self.fish_cards_cy, now)

    # Strip -------------------------------------------------------------------
    def _draw_strip(self, canvas, table: Table, counts: list[int], now: float) -> None:
        x, y, w, h = self.strip
        cy = y + h / 2
        vote = table.vote
        if vote is not None and vote.started is not None and vote.winner is None:
            remaining = max(0.0, vote.seconds - (now - vote.started))
            frac = remaining / max(0.01, vote.seconds)
            urgent = remaining <= 3
            colour = mix_hex(GOLD, HOT, clamp((3 - remaining) / 3)) if urgent else GOLD
            r = 64
            rcx, rcy = x + 18 + r, int(cy)
            cv2.circle(canvas, (rcx, rcy), r, hex_bgr(DIM), 10, cv2.LINE_AA)
            layer = np.zeros((r * 2 + 60, r * 2 + 60, 3), np.uint8)
            cv2.ellipse(layer, (r + 30, r + 30), (r, r), -90, 0, 360 * frac, hex_bgr(colour), 12, cv2.LINE_AA)
            glow_layer(canvas, layer, rcx - r - 30, rcy - r - 30, 10, 0.8)
            m = layer.any(axis=2)
            sub = canvas[rcy - r - 30 : rcy + r + 30, rcx - r - 30 : rcx + r + 30]
            if sub.shape[:2] == m.shape:
                sub[m] = layer[m]
            sec = str(max(0, math.ceil(remaining - 1e-9)))
            blit(canvas, text_sprite(sec, "display", 76, INK), rcx, rcy, "cc")
            tx = rcx + r + 34
            tw = x + w - tx - 10
            kicker = text_sprite("LAST CHANCE TO SWIM" if urgent else "THE FISH ARE DECIDING", "bold", 22, HOT if urgent else GOLD, tracking=6)
            blit(canvas, kicker, tx, y + 14, "lt")
            q = fit_text(vote.question, "display", tw, 50, INK)
            blit(canvas, q, tx, y + 14 + kicker.shape[0] + 10, "lt")
            row_y = y + 14 + kicker.shape[0] + 10 + q.shape[0] + 16
            n = len(vote.labels)
            col_w = tw // max(1, n)
            for i, label in enumerate(vote.labels):
                cx0 = tx + i * col_w
                c = counts[i] if i < len(counts) else 0
                share = self.state.shares[i] if i < len(self.state.shares) else 0
                lab = text_sprite(label, "display", 30, zone_hex(i))
                blit(canvas, lab, cx0, row_y, "lt")
                num = text_sprite(str(c), "display", 30, INK)
                blit(canvas, num, cx0 + col_w - 24, row_y, "rt")
                by = row_y + lab.shape[0] + 8
                bw = col_w - 24
                round_rect(canvas, cx0, by, bw, 12, 6, SHADOW, 0.9)
                fw = int(bw * clamp(share))
                if fw > 12:
                    round_rect(canvas, cx0, by, fw, 12, 6, zone_hex(i), 1.0)
            return
        if vote is not None and vote.winner is not None:
            t = now - vote.decided_at
            label = vote.labels[vote.winner]
            colour = zone_hex(vote.winner)
            kick = text_sprite("THE FISH SAY", "bold", 26, GOLD, tracking=8)
            blit(canvas, kick, x + w / 2, y + 10, "ct", alpha=clamp(t / 0.2))
            pop = ease_out_back(t / 0.4)
            big = fit_text(label, "display", w - 80, 120, colour)
            blit(canvas, glow_text(label, "display", 120 if big.shape[1] < w - 80 else 100, colour, 0, 22, 0.8), x + w / 2, cy + 18, "cc", alpha=0.7, scale=pop)
            blit(canvas, big, x + w / 2, cy + 18, "cc", scale=pop)
            return
        if table.result is not None:
            outcome, delta, at = table.result
            t = now - at
            text = OUTCOME_TEXT[outcome]
            colour = GOLD if outcome == "blackjack" else WIN_GREEN if delta > 0 else INK if delta == 0 else HOT
            pop = ease_out_back(t / 0.45)
            big = fit_text(text, "display", w - 60, 112, colour)
            gl = glow_text(text, "display", 112, colour, 0, 24, 0.9) if big.shape[1] < w - 60 else None
            if gl is not None:
                blit(canvas, gl, x + w / 2, cy - 14, "cc", alpha=0.6 + 0.3 * math.sin(now * 5), scale=pop)
            blit(canvas, big, x + w / 2, cy - 14, "cc", scale=pop)
            sub = f"{'+' if delta > 0 else ''}{delta:,} CHIPS" if delta else "BET RETURNED"
            blit(canvas, text_sprite(sub, "mono", 34, colour, tracking=3), x + w / 2, y + h - 22, "cb", alpha=clamp((t - 0.25) / 0.3))
            return
        if table.status:
            s = fit_text(table.status, "bold", w - 60, 40, MUTED, tracking=6)
            blit(canvas, s, x + w / 2, cy, "cc", alpha=0.75 + 0.25 * math.sin(now * 3))

    # Chips and bankroll -----------------------------------------------------
    def _draw_bottom(self, canvas, table: Table, now: float) -> None:
        y = self.bottom_y + 70
        if table.bet:
            stacks = 2 if table.doubled else 1
            per = table.bet // stacks
            colours = ("#FF3B4E", "#2EE6D6", "#F5C542", "#A66BFF", "#9BFF4D")
            for s in range(stacks):
                cx = 128 + s * 96
                chips = max(1, min(9, per // 10))
                for i in range(chips):
                    poker_chip(canvas, cx, y + 20 - i * 9, 36, colours[(i + s) % len(colours)], "")
            label = f"BET {table.bet:,}" + ("  ·  DOUBLED" if table.doubled else "")
            blit(canvas, text_sprite(label, "bold", 26, INK, tracking=4), 96 + stacks * 96, y + 4, "lc")
        # Bankroll counter.
        target = float(table.bankroll)
        if table.shown_bankroll == 0.0:
            table.shown_bankroll = target
        table.shown_bankroll += (target - table.shown_bankroll) * 0.12
        if abs(target - table.shown_bankroll) < 0.5:
            table.shown_bankroll = target
        rx = 1000
        blit(canvas, text_sprite("FISH BANKROLL", "bold", 22, MUTED, tracking=6), rx, y - 34, "rb")
        value = text_sprite(f"{int(round(table.shown_bankroll)):,}", "display", 66, GOLD)
        blit(canvas, value, rx, y - 26, "rt")
        session = table.bankroll - table.start_bankroll
        if session:
            tag = f"{'+' if session > 0 else ''}{session:,} THIS SESSION"
            blit(canvas, text_sprite(tag, "mono", 20, WIN_GREEN if session > 0 else HOT, tracking=1), rx, y - 26 + value.shape[0] + 12, "rt")


# ---------------------------------------------------------------------------
# Game flow
# ---------------------------------------------------------------------------
class Wait:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.until: float | None = None


class Session:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config(args.config)
        self.demo = args.demo
        self.rng = random.Random(args.seed)
        self.shoe = Shoe(6, random.Random(args.seed) if args.seed is not None else None)
        self.bank = {"bankroll": args.bankroll, "start": args.bankroll, "hands": []} if (args.demo or args.reset) else load_bankroll(start=args.bankroll)
        start = int(self.bank["bankroll"])
        self.table = Table(bankroll=start, start_bankroll=start)
        self.bets = sorted({int(b) for b in args.bets.split(",") if b.strip().isdigit() and int(b) > 0}) or [100]

        labels = ["HIT", "STAND", "DOUBLE"]
        if self.demo:
            if args.demo_style is None:
                args.demo_style = "real" if args.look == "studio" else "pixel"
            module = pixel_tank if args.demo_style == "pixel" else real_demo
            self.zones = module.demo_zones(labels)
            tank_cls = pixel_tank.PixelTank if args.demo_style == "pixel" else real_demo.DemoTank
            self.capture = tank_cls(self.zones, float(args.decision_seconds))
            self.capture.end_vote()
            self.background = self.capture.empty_frame
            self.detector = None
        else:
            if len(self.config.zones) < 2:
                die("Blackjack needs at least 2 zones (HIT, STAND), 3 to allow DOUBLE. Run python calibrate.py, or try --demo.")
            self.zones = list(self.config.zones[:3])
            self.capture = None
            self.background = load_background(self.config, args.config) if self.config.background_mode == "frame" else None
            self.detector = MotionDetector() if self.background is None else None

    # Script ------------------------------------------------------------------
    def deal(self, hand: list[TableCard], now: float, face_up: bool = True) -> Card:
        card = self.shoe.draw()
        hand.append(TableCard(card, now, face_up))
        return card

    def script(self):
        for _ in range(self.args.hands):
            if self.table.bankroll <= 0:
                self.table.status = "THE FISH ARE BROKE"
                yield Wait(3.0)
                break
            yield from self.hand()
        t = self.table
        diff = t.bankroll - t.start_bankroll
        t.result = None
        t.status = f"SESSION OVER  ·  {'+' if diff > 0 else ''}{diff:,} CHIPS"
        yield Wait(3.0)

    def hand(self):
        t = self.table
        t.hand_no += 1
        self.hud.info.number = len(self.bank["hands"]) + 1
        self.hand_label = f"BLACKJACK  ·  HAND {len(self.bank['hands']) + 1:02d}"
        t.dealer, t.fish, t.bet, t.doubled, t.result = [], [], 0, False, None
        if self.shoe.needs_shuffle():
            t.status = "SHUFFLING A FRESH SHOE"
            self.shoe.shuffle()
            yield Wait(1.4)
        options = [b for b in self.bets if b <= t.bankroll] or [t.bankroll]
        if self.args.fish_bet and len(options) > 1:
            options = options[: len(self.zones)]
            choice = yield Vote([str(b) for b in options], "HOW MANY CHIPS DO THEY BET?", self.args.decision_seconds)
            bet = int(choice)
        else:
            bet = min(int(self.args.bet), t.bankroll)
        t.bet = bet
        t.status = f"BET {bet:,}  ·  DEALING"
        yield Wait(0.7)
        for hand, face_up in ((t.fish, True), (t.dealer, True), (t.fish, True), (t.dealer, False)):
            self.deal(hand, self.now, face_up)
            yield Wait(0.5)
        yield Wait(0.4)
        dealer_hole = t.dealer[1]
        fish_cards = [c.card for c in t.fish]
        dealer_cards = [c.card for c in t.dealer]
        decisions: list[str] = []

        def reveal():
            dealer_hole.flip_at = self.now
            return Wait(FLIP_SECONDS + 0.6)

        if dealer_peeks(dealer_cards[0]):
            t.status = "DEALER CHECKS FOR BLACKJACK"
            yield Wait(1.3)
        if is_blackjack(dealer_cards) and dealer_peeks(dealer_cards[0]) or is_blackjack(fish_cards):
            t.status = ""
            yield reveal()
        else:
            first = True
            while hand_value([c.card for c in t.fish])[0] < 21:
                can_double = first and len(self.zones) >= 3 and t.bankroll >= bet * 2
                labels = ["HIT", "STAND"] + (["DOUBLE"] if can_double else [])
                fish_total = describe([c.card for c in t.fish])
                question = f"{fish_total} VS {dealer_cards[0].rank}: " + ("HIT, STAND OR DOUBLE?" if can_double else "HIT OR STAND?")
                t.status = ""
                choice = yield Vote(labels, question, self.args.decision_seconds)
                decisions.append(choice)
                if choice == "STAND":
                    break
                if choice == "DOUBLE":
                    t.bet = bet * 2
                    t.doubled = True
                    self.deal(t.fish, self.now)
                    yield Wait(0.9)
                    break
                self.deal(t.fish, self.now)
                yield Wait(0.9)
                first = False
            fish_total = hand_value([c.card for c in t.fish])[0]
            t.status = "BUST!" if fish_total > 21 else "DEALER'S TURN"
            yield reveal()
            if fish_total <= 21:
                while dealer_should_hit([c.card for c in t.dealer]):
                    t.status = "DEALER DRAWS"
                    self.deal(t.dealer, self.now)
                    yield Wait(0.95)
        fish_cards = [c.card for c in t.fish]
        dealer_cards = [c.card for c in t.dealer]
        outcome, delta = settle(fish_cards, dealer_cards, t.bet)
        t.status = ""
        t.result = (outcome, delta, self.now)
        t.bankroll += delta
        record_hand(self.bank, fish_cards, dealer_cards, t.bet, decisions, outcome, delta)
        if not self.demo:
            save_bankroll(self.bank)
        if delta > 0 and self.studio is None:
            x, y, w, h = self.hud.strip
            colours = [GOLD, INK, WIN_GREEN, "#7CE38B", GOLD]
            self.hud.confetti.burst(x + w / 2, y + h / 2, self.now, colours, count=120 if outcome == "blackjack" else 80)
        yield Wait(3.4)
        t.result = None

    # Main loop ---------------------------------------------------------------
    def run(self) -> Path | None:
        args = self.args
        cap = self.capture
        video = None
        if cap is None:
            if args.video is not None:
                cap, frame = open_video(args.video)
                video = args.video
            else:
                cap, frame = open_camera(self.config.camera_index)
        else:
            _ok, frame = cap.read()
        frame = even_frame(frame)
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps < 1 or fps > 120:
            fps = 30.0

        info = RoundInfo(round_type="blackjack", labels=["HIT", "STAND", "DOUBLE"], countdown=float(args.decision_seconds))
        info.number_word = "HAND"
        info.number = len(self.bank["hands"]) + 1
        info.footer = "PLAY CHIPS ONLY  ·  NO REAL MONEY  ·  THE FISH DECIDE EVERY MOVE"
        if self.demo:
            info.camera_label = "SIMULATED TANK"
        start = time.monotonic()
        frames_done = 0
        fps_out = float(args.fps)

        def clock() -> float:
            return start + frames_done / fps_out if self.demo else time.monotonic()

        self.hud = TableHud(info, (width, height), start, bool(getattr(cap, "pixel_art", False)))
        self.hand_label = f"BLACKJACK  ·  HAND {info.number:02d}"
        self.studio = None
        out_size = TABLE_LAYOUT.size
        if args.look == "studio":
            out_size = (2160, 3840) if args.uhd else (1080, 1920)
            scale = 2 if (args.uhd or args.scale >= 2) else 1
            self.studio = StudioRenderer(scale, (width, height), bool(getattr(cap, "pixel_art", False)), out_size=out_size, fps=fps_out, tank_label="SIMULATED TANK" if self.demo else "")
        recording = RECORDINGS_DIR / f"{'demo-' if self.demo else ''}blackjack-{file_stamp()}.mp4"
        writer = open_writer(recording, *out_size, fps_out)
        window = "fish-fund blackjack | q quits"
        preview = not args.no_preview
        if preview:
            preview_dims = (int(out_size[0] * (args.preview_scale or 900 / out_size[1])), int(out_size[1] * (args.preview_scale or 900 / out_size[1])))
            pw, ph = preview_dims
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, pw, ph)
        wait_ms = 1 if video is None else max(1, int(round(1000 / fps)))

        self.now = clock()
        script = self.script()
        command = next(script)
        written = 0
        finished = False
        try:
            while not finished:
                if frames_done:
                    if self.demo:
                        _ok, frame = cap.read()
                    else:
                        ok, frame = cap.read()
                        if (not ok or frame is None) and video is not None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, frame = cap.read()
                        if not ok or frame is None:
                            die(f"The camera stopped returning frames. Partial recording kept at {recording}.")
                        frame = cv2.resize(even_frame(frame), (width, height))
                self.now = now = clock()

                # Advance the script until it is waiting on time or on the fish.
                while True:
                    if isinstance(command, Wait):
                        if command.until is None:
                            command.until = now + command.seconds
                        if now < command.until:
                            break
                        result = None
                    elif isinstance(command, Vote):
                        if command.started is None:
                            command.started = now
                            self.table.vote = command
                            self.hud.state.shares = [0.0] * len(command.labels)
                            if self.demo:
                                cap.start_vote(len(command.labels), command.seconds)
                        if command.winner is None or now - command.decided_at < 1.5:
                            break
                        result = command.labels[command.winner]
                        self.table.vote = None
                        self.hud.state.winner = None
                    try:
                        command = script.send(result)
                    except StopIteration:
                        finished = True
                        break

                # Detection.
                seen = cap.detection_frame if self.demo else frame
                if self.background is not None:
                    mask = frame_mask(seen, self.background)
                else:
                    mask = self.detector.mask(seen)
                centroids = centroids_from_mask(mask, self.config.min_blob_area, self.config.max_blob_area)
                vote = self.table.vote
                zones: list[Zone] = []
                counts: list[int] = []
                owners: list = []
                if vote is not None:
                    zones = [Zone(label, z.x, z.y, z.w, z.h) for label, z in zip(vote.labels, self.zones)]
                    counts, owners = count_zones(centroids, zones)
                    if vote.winner is None:
                        vote.samples.append((now - vote.started, list(counts)))
                        if now - vote.started >= vote.seconds:
                            vote.winner = choose_winner(vote.samples)
                            vote.decided_at = now
                            self.hud.state.winner = vote.winner
                            if self.demo:
                                cap.end_vote()
                else:
                    owners = [None] * len(centroids)

                if self.studio is not None:
                    display = self.studio.render(frame, zones, counts, now, self.table, self.hand_label)
                else:
                    display = self.hud.render(frame, zones, counts, centroids, owners, now, self.table)
                due = int((clock() - start) * fps_out) + 1
                repeats = 1 if self.demo else max(1, min(4, due - written))
                for _ in range(repeats):
                    writer.write(display)
                written += repeats
                frames_done += 1
                if preview:
                    cv2.imshow(window, cv2.resize(display, preview_dims, interpolation=cv2.INTER_AREA))
                    if cv2.waitKey(wait_ms) & 0xFF in (ord("q"), 27):
                        print("Stopped early.")
                        break
        finally:
            writer.release()
            cap.release()
            if preview:
                cv2.destroyAllWindows()

        t = self.table
        print(f"Hands played: {t.hand_no}. Bankroll: {t.bankroll:,} chips ({t.bankroll - t.start_bankroll:+,}).")
        for hand in self.bank["hands"][-t.hand_no:]:
            print(f"  {' '.join(hand['fish'])} vs {' '.join(hand['dealer'])}: {'/'.join(hand['decisions']) or '-'} -> {hand['outcome']} {hand['delta']:+,}")
        print(f"Recording: {recording.resolve()}")
        phone = phone_copy(recording, crf=17 if self.studio is not None else 18)
        if phone is not None:
            print(f"Phone-ready copy: {phone.resolve()}")
        if self.demo:
            print("Demo session: bankroll not saved.")
        else:
            print(f"Bankroll saved to {BLACKJACK_PATH}")
        return recording


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="The fish play blackjack (play chips).")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--hands", type=int, default=3, help="Hands to play in one recording.")
    parser.add_argument("--bet", type=int, default=25, help="Chips per hand when the fish don't choose.")
    parser.add_argument("--fish-bet", action="store_true", help="Let the fish vote on the bet before each hand.")
    parser.add_argument("--bets", default="10,25,50", help="Bet sizes for --fish-bet, one per zone.")
    parser.add_argument("--bankroll", type=int, default=100, help="Starting chips for a new or reset bankroll.")
    parser.add_argument("--reset", action="store_true", help="Start a fresh bankroll.")
    parser.add_argument("--decision-seconds", type=float, default=8.0, help="Length of each fish vote.")
    parser.add_argument("--demo", action="store_true", help="Simulated tank. Nothing is saved.")
    parser.add_argument("--demo-style", choices=("pixel", "real"), default=None, help="Simulated tank style. Default: real for the studio look, pixel for broadcast.")
    parser.add_argument("--look", choices=("studio", "broadcast"), default="studio", help="studio = cinematic card room (default), broadcast = the neon HUD.")
    parser.add_argument("--scale", type=int, default=2, help="Studio supersampling: 2 draws at 4K and downsamples for clean edges (default), 1 is faster.")
    parser.add_argument("--uhd", action="store_true", help="Studio look: export a 2160x3840 master instead of 1080x1920.")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate (demo renders every frame; 60 doubles render time).")
    parser.add_argument("--video", type=Path, default=None, help="Play a tank video instead of the webcam.")
    parser.add_argument("--seed", type=int, default=None, help="Fix the shuffle (and demo fish) for a repeatable run.")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-scale", type=float, default=None)
    args = parser.parse_args(argv)
    if args.hands < 1:
        die("--hands must be at least 1.")
    if args.demo and args.video is not None:
        die("Use either --demo or --video, not both.")
    Session(args).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
