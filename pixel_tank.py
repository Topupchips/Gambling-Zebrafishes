"""A pixel-art aquarium for demo rounds.

The tank is drawn at 250x140 and scaled up 4x with hard edges, so every frame looks
like a hand-made pixel animation: dithered water, swaying plants, rising bubbles,
light rays, and five striped zebrafish sprites with flapping tails.

``python pick.py --round buy --demo`` uses this style by default.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from utils import Zone

LOW_W, LOW_H = 250, 140
SCALE = 4
WIDTH, HEIGHT = LOW_W * SCALE, LOW_H * SCALE  # 1000 x 560, the HUD's camera window
FPS = 30.0
SAND_TOP = 117
WATER_TOP = 8


def hexc(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (4, 2, 0)], np.uint8)  # BGR


WATER = [hexc(c) for c in ("#7fe0dc", "#5fcfd3", "#43b6c6", "#3299b3", "#277d9d", "#1f6487", "#194f72")]
AIR = hexc("#b8ece6")
SURFACE = hexc("#e9fffb")
SAND = [hexc(c) for c in ("#f5dca6", "#e6c58b", "#d0a96e", "#b38953")]
ROCK = [hexc(c) for c in ("#b2b8c4", "#878d9b", "#636978", "#454a57")]
PLANT = [hexc(c) for c in ("#8af0a8", "#3fcf7c", "#25a861", "#187a49", "#10563a")]
BUBBLE = hexc("#eafdff")
WOOD = [hexc(c) for c in ("#f0a85a", "#d9873c", "#b8682a", "#8a4a1d", "#5e3214")]
GOLD = [hexc(c) for c in ("#fff1a6", "#ffd24a", "#e3a224", "#a8701a")]
OUTLINE = hexc("#14203c")

FISH = {
    "o": hexc("#14203c"),  # outline
    "d": hexc("#cfae5c"),  # dorsal olive-gold
    "b": hexc("#2c4fb2"),  # blue stripe
    "c": hexc("#f4e2ab"),  # cream stripe
    "w": hexc("#fff6de"),  # belly
    "f": hexc("#97d3ec"),  # fin
    "s": hexc("#4b7fd0"),  # fin stripe
    "e": hexc("#ffffff"),  # eye white
    "k": hexc("#0b0d14"),  # pupil
}

BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16


def demo_zones(labels: list[str]) -> list[Zone]:
    """Columns over the water, snapped to the pixel grid."""
    n = max(1, len(labels))
    margin, gap = 6, 4  # low-res pixels
    zone_w = (LOW_W - margin * 2 - gap * (n - 1)) // n
    top, bottom = WATER_TOP + 4, SAND_TOP - 2
    return [
        Zone(label, (margin + i * (zone_w + gap)) * SCALE, top * SCALE, zone_w * SCALE, (bottom - top) * SCALE)
        for i, label in enumerate(labels)
    ]


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------
def fish_sprite(tail_shift: int = 0, fin_up: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A 30x14 zebrafish facing right. Returns (BGR, mask)."""
    W, H = 30, 14
    grid = np.full((H, W), ".", dtype="<U1")
    cy = 6.5
    bx0, bx1 = 8, 28
    profile_x = [0.0, 0.3, 0.6, 0.86, 1.0]
    profile_h = [1.6, 3.6, 4.4, 3.6, 1.7]
    for x in range(bx0, bx1 + 1):
        xn = (x - bx0) / (bx1 - bx0)
        hh = float(np.interp(xn, profile_x, profile_h))
        top, bot = cy - hh, cy + hh
        for y in range(H):
            yc = y + 0.5
            if top <= yc <= bot:
                r = (yc - top) / (bot - top)
                if xn > 0.84:
                    grid[y, x] = "d" if r < 0.45 else "w"
                elif r < 0.2:
                    grid[y, x] = "d"
                elif r < 0.34:
                    grid[y, x] = "b"
                elif r < 0.46:
                    grid[y, x] = "c"
                elif r < 0.6:
                    grid[y, x] = "b"
                elif r < 0.72:
                    grid[y, x] = "c"
                elif r < 0.84:
                    grid[y, x] = "b"
                else:
                    grid[y, x] = "w"
    # Tail: forked, striped fin.
    tail_cy = cy + tail_shift
    for x in range(1, bx0 + 1):
        xt = (bx0 - x) / (bx0 - 1)
        spread = 1.5 + 3.6 * xt
        notch = 2.4 * xt ** 1.6 if xt > 0.45 else 0
        for y in range(H):
            dy = abs(y + 0.5 - tail_cy)
            if notch <= dy <= spread:
                grid[y, x] = "s" if int(dy) % 2 == 0 else "f"
    # Dorsal and anal fins.
    lift = 1 if fin_up else 0
    for x in range(15, 20):
        for y in range(1 - lift, 3 - lift):
            if 0 <= y < H and grid[y, x] == "." and (x - 15) >= (2 - y - lift) * 1.5 - 1:
                grid[y, x] = "f"
    for x in range(12, 16):
        y = 11
        if grid[y, x] == ".":
            grid[y, x] = "f"
    # Eye.
    grid[5, 24] = "e"
    grid[5, 25] = "k"
    grid[4, 24] = "e"
    # Outline every empty pixel that touches the fish.
    filled = grid != "."
    pad = np.pad(filled, 1)
    touch = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    grid[(~filled) & touch] = "o"
    img = np.zeros((H, W, 3), np.uint8)
    for key, colour in FISH.items():
        img[grid == key] = colour
    mask = grid != "."
    return img, mask


def rock_sprite(rng: random.Random, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx + 0.5 - w / 2) / (w / 2)
    ny = (yy + 0.5 - h) / h
    inside = nx ** 2 + ny ** 2 <= 1.0
    light = -(nx * 0.6 + ny * 0.8)
    level = np.clip((1 - light) * 1.4 + BAYER[yy.astype(int) % 4, xx.astype(int) % 4] * 0.9, 0, 3).astype(int)
    img = np.zeros((h, w, 3), np.uint8)
    for i in range(4):
        img[level == i] = ROCK[i]
    pad = np.pad(inside, 1)
    edge = inside & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    img[edge] = OUTLINE
    if rng.random() < 0.5:
        img[np.clip(h // 3, 0, h - 1), np.clip(w // 3, 0, w - 1)] = hexc("#e7ebf2")
    return img, inside


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------
class PixelFish:
    def __init__(self, rng: random.Random, x: float, y: float) -> None:
        self.rng = rng
        self.x, self.y = x, y
        self.vx = rng.uniform(-10, 10)
        self.vy = rng.uniform(-4, 4)
        self.cruise = rng.uniform(16, 24)
        self.burst = 0.0
        self.next_burst = rng.uniform(0, 1.0)
        self.facing = 1 if self.vx >= 0 else -1
        self.anim = rng.uniform(0, 10)
        self.target = (x, y)
        self.loyal = rng.random() < 0.82

    def step(self, dt: float, school: list["PixelFish"]) -> None:
        self.next_burst -= dt
        if self.next_burst <= 0:
            self.burst = 1.0
            self.next_burst = self.rng.uniform(0.6, 1.6)
        self.burst = max(0.0, self.burst - dt * 1.8)
        speed = self.cruise * (0.45 + 1.6 * self.burst)
        tx, ty = self.target
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy) + 1e-6
        ax, ay = dx / dist, dy / dist * 0.7
        for other in school:
            if other is self:
                continue
            ox, oy = self.x - other.x, self.y - other.y
            d = math.hypot(ox, oy)
            if 0 < d < 42:
                push = (42 - d) / 42 * 3.2
                ax += ox / d * push
                ay += oy / d * push * 2.0
        # Fish hold depth deliberately: vertical speed follows the depth error directly.
        want_vy = max(-speed * 0.8, min(speed * 0.8, dy * 1.4 + (ay - dy / dist * 0.7) * speed))
        self.vx += (ax * speed - self.vx) * min(1.0, dt * 2.5)
        self.vy += (want_vy - self.vy) * min(1.0, dt * 3.0)
        self.x = min(LOW_W - 18, max(18, self.x + self.vx * dt))
        self.y = min(SAND_TOP - 10, max(WATER_TOP + 9, self.y + self.vy * dt))
        if self.vx > 3:
            self.facing = 1
        elif self.vx < -3:
            self.facing = -1
        self.anim += dt * (5 + 10 * self.burst)


class Bubble:
    def __init__(self, rng: random.Random, x: float) -> None:
        self.x0 = x
        self.y = SAND_TOP - 2
        self.size = rng.choice((1, 1, 2, 2, 3))
        self.speed = rng.uniform(14, 26)
        self.phase = rng.uniform(0, math.tau)


# ---------------------------------------------------------------------------
# Tank
# ---------------------------------------------------------------------------
class PixelTank:
    """Frame source with the same ``read()`` contract as ``cv2.VideoCapture``.

    ``detection_frame`` holds a plain pass (fish on flat water) for detection.
    """

    pixel_art = True

    def __init__(self, zones: list[Zone], countdown: float, fish: int = 5, favourite: int | None = None, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.zones = zones
        self.countdown = countdown
        self.favourite = favourite if favourite is not None else self.rng.randrange(len(zones))
        self.t = 0.0
        self.vote_start = 0.0
        self.vote_seconds = countdown
        self.vote_open = True
        self.frames = {
            (shift, fin): fish_sprite(shift, fin) for shift in (-1, 0, 1) for fin in (False, True)
        }
        self.static = self._make_static()
        self.plants = self._make_plants()
        self.bubble_sources = [133.0, self.rng.uniform(20, 90), self.rng.uniform(170, LOW_W - 20)]
        self.bubbles: list[Bubble] = []
        self.fish = [PixelFish(self.rng, 30 + i * (LOW_W - 60) / max(1, fish - 1), self.rng.uniform(30, 95)) for i in range(fish)]
        for f in self.fish:
            f.target = self._random_point()
        self.plain_bg = np.full((LOW_H, LOW_W, 3), WATER[2], np.uint8)
        self.empty_frame = cv2.resize(self.plain_bg, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        self.detection_frame = self.empty_frame

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def get(self, _prop) -> float:
        return FPS

    def release(self) -> None:
        pass

    def set(self, *_args) -> bool:
        return True

    def _random_point(self, zone: Zone | None = None) -> tuple[float, float]:
        if zone is None:
            return (self.rng.uniform(24, LOW_W - 24), self.rng.uniform(WATER_TOP + 14, SAND_TOP - 14))
        x0, x1 = zone.x / SCALE, (zone.x + zone.w) / SCALE
        y0, y1 = zone.y / SCALE, (zone.y + zone.h) / SCALE
        return (self.rng.uniform(x0 + 12, x1 - 12), self.rng.uniform(y0 + 12, y1 - 12))

    # Static scene -----------------------------------------------------------
    def _make_static(self) -> np.ndarray:
        img = np.zeros((LOW_H, LOW_W, 3), np.uint8)
        yy, xx = np.mgrid[0:LOW_H, 0:LOW_W]
        # Water: depth bands with ordered dithering between neighbours.
        depth = np.clip((yy - WATER_TOP) / (SAND_TOP - WATER_TOP), 0, 0.999) * (len(WATER) - 1)
        base = depth.astype(int)
        frac = depth - base
        pick = np.where(frac > BAYER[yy % 4, xx % 4], np.minimum(base + 1, len(WATER) - 1), base)
        palette = np.stack(WATER)
        img[:] = palette[pick]
        # Wooden tank rim along the top, like a cosy shelf.
        for y in range(WATER_TOP):
            for x in range(LOW_W):
                grain = int(1.4 * math.sin(x * 0.31 + y * 1.7) + BAYER[y % 4, x % 4] * 1.2)
                level = 1 if y in (1, 2) else 2 if y < WATER_TOP - 2 else 3
                img[y, x] = WOOD[min(4, max(0, level + (1 if grain > 1 else 0)))]
            img[0] = WOOD[0]
        img[WATER_TOP - 1] = WOOD[4]
        for x in range(0, LOW_W, 31):
            img[3:WATER_TOP - 2, x] = WOOD[3]
        # Back-glass shimmer lines.
        for x0 in (28, 34, 170):
            for i in range(26):
                x, y = x0 + i, WATER_TOP + 6 + i * 2
                if y < SAND_TOP - 30:
                    img[y, x] = np.minimum(img[y, x].astype(int) + 26, 255).astype(np.uint8)
        # Sand with a bumpy top and texture.
        surface = SAND_TOP + np.round(1.6 * np.sin(np.arange(LOW_W) * 0.09) + 1.2 * np.sin(np.arange(LOW_W) * 0.23 + 1)).astype(int)
        for x in range(LOW_W):
            top = surface[x]
            for y in range(top, LOW_H):
                d = (y - top) / max(1, LOW_H - top)
                level = min(3, int(d * 3.2 + BAYER[y % 4, x % 4] * 0.9))
                img[y, x] = SAND[level]
            img[top, x] = SAND[0]
        for _ in range(140):
            x, y = self.rng.randrange(LOW_W), self.rng.randrange(SAND_TOP + 3, LOW_H)
            img[y, x] = self.rng.choice((SAND[3], SAND[2], hexc("#f7c6b5"), hexc("#fff3d6")))
        # Rocks.
        for x, w, h in ((14, 22, 11), (40, 12, 7), (120, 16, 8), (196, 26, 13), (228, 14, 8), (88, 9, 5), (160, 10, 6)):
            sprite, mask = rock_sprite(self.rng, w, h)
            y0 = SAND_TOP - h + 4
            sub = img[y0 : y0 + h, x : x + w]
            sub[mask[: sub.shape[0], : sub.shape[1]]] = sprite[: sub.shape[0], : sub.shape[1]][mask[: sub.shape[0], : sub.shape[1]]]
        self._draw_chest(img, 124, SAND_TOP - 9)
        return img

    @staticmethod
    def _draw_chest(img: np.ndarray, x: int, y: int) -> None:
        """A sunken treasure chest spilling gold coins."""
        rows = [
            "...oooooooooooo...",
            "..oaaaaaaaaaaaao..",
            ".oabbbbbgbbbbbbao.",
            ".oooooooGooooooo..",
            ".ocyycycgcyyccyo..",
            ".occccccGccccccoy.",
            ".ocyccycgccyccco.y",
            ".oooooooooooooooyy",
            "..y.yy..........yy",
        ]
        cmap = {"o": OUTLINE, "a": WOOD[1], "b": WOOD[2], "c": WOOD[2], "y": GOLD[1], "g": GOLD[2], "G": GOLD[0]}
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch in cmap and 0 <= y + j < LOW_H and 0 <= x + i < LOW_W:
                    img[y + j, x + i] = cmap[ch]

    def _make_plants(self) -> list[dict]:
        plants = []
        for x in (6, 22, 58, 100, 154, 184, 214, 240):
            plants.append(
                {
                    "x": x + self.rng.randint(-3, 3),
                    "height": self.rng.randint(24, 62),
                    "phase": self.rng.uniform(0, math.tau),
                    "amp": self.rng.uniform(2.0, 4.0),
                    "kind": self.rng.choice(("leafy", "grass", "grass")),
                }
            )
        return plants

    # Per frame --------------------------------------------------------------
    def _draw_plants(self, img: np.ndarray) -> None:
        for p in self.plants:
            base_y = SAND_TOP + 2
            strands = 1 if p["kind"] == "leafy" else 3
            for s in range(strands):
                height = p["height"] - s * 7
                sx = p["x"] + (s - 1) * 3
                for j in range(height):
                    y = base_y - j
                    frac = j / height
                    sway = math.sin(self.t * 1.3 + p["phase"] + j * 0.13 + s) * p["amp"] * frac ** 1.4
                    x = int(round(sx + sway))
                    if not (0 <= x < LOW_W and 0 <= y < LOW_H):
                        continue
                    shade = 3 if s == 1 else 2
                    img[y, x] = PLANT[shade]
                    if p["kind"] == "grass" and x + 1 < LOW_W:
                        img[y, x + 1] = PLANT[shade + 1]
                    if p["kind"] == "leafy" and j % 5 == 2 and j > 3:
                        side = 1 if (j // 5) % 2 else -1
                        for k in range(1, 5):
                            lx, ly = x + side * k, y - (k // 2)
                            if 0 <= lx < LOW_W and 0 <= ly < LOW_H:
                                img[ly, lx] = PLANT[1] if k < 3 else PLANT[2]
                    if j == height - 1:
                        img[y, x] = PLANT[0]

    def _draw_rays(self, img: np.ndarray) -> None:
        for i, x0 in enumerate((40, 110, 175)):
            drift = math.sin(self.t * 0.4 + i * 1.7) * 6
            strength = 0.5 + 0.5 * math.sin(self.t * 0.9 + i)
            for y in range(WATER_TOP, SAND_TOP):
                cx = x0 + drift + (y - WATER_TOP) * 0.45
                half = 3 + (y - WATER_TOP) * 0.06
                for x in range(int(cx - half), int(cx + half) + 1):
                    if 0 <= x < LOW_W and BAYER[y % 4, x % 4] < 0.32 * strength * (1 - (y - WATER_TOP) / (SAND_TOP - WATER_TOP)):
                        img[y, x] = np.minimum(img[y, x].astype(int) + 28, 255).astype(np.uint8)

    def _draw_surface(self, img: np.ndarray) -> None:
        for x in range(LOW_W):
            y = WATER_TOP + int(round(math.sin(x * 0.16 + self.t * 2.2) * 0.8 + math.sin(x * 0.05 - self.t) * 0.6))
            img[max(0, y), x] = SURFACE
            if y + 1 < LOW_H and (x + int(self.t * 8)) % 7 == 0:
                img[y + 1, x] = SURFACE

    def _draw_bubbles(self, img: np.ndarray, dt: float) -> None:
        for src in self.bubble_sources:
            if self.rng.random() < dt * 2.2:
                self.bubbles.append(Bubble(self.rng, src + self.rng.uniform(-2, 2)))
        alive = []
        for b in self.bubbles:
            b.y -= b.speed * dt
            if b.y < WATER_TOP + 1:
                continue
            x = int(round(b.x0 + math.sin(self.t * 3 + b.phase) * 1.5))
            y = int(round(b.y))
            if b.size == 1:
                img[y, x] = BUBBLE
            else:
                s = b.size
                for dx in range(-s + 1, s):
                    for dy in range(-s + 1, s):
                        if abs(dx) + abs(dy) == s - 1 or (s == 2 and abs(dx) + abs(dy) <= 1 and (dx, dy) != (0, 0)):
                            px, py = x + dx, y + dy
                            if 0 <= px < LOW_W and 0 <= py < LOW_H:
                                img[py, px] = BUBBLE
                if 0 <= x - 1 < LOW_W and 0 <= y - 1 < LOW_H and s == 3:
                    img[y - 1, x - 1] = hexc("#ffffff")
            alive.append(b)
        self.bubbles = alive

    def _sprite_for(self, f: PixelFish) -> tuple[np.ndarray, np.ndarray]:
        cycle = int(f.anim) % 4
        shift = (0, 1, 0, -1)[cycle]
        img, mask = self.frames[(shift, cycle in (1, 2))]
        if f.facing < 0:
            img, mask = img[:, ::-1], mask[:, ::-1]
        return img, mask

    @staticmethod
    def _paste(img: np.ndarray, sprite: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
        h, w = mask.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(LOW_W, x + w), min(LOW_H, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        m = mask[y0 - y : y1 - y, x0 - x : x1 - x]
        img[y0:y1, x0:x1][m] = sprite[y0 - y : y1 - y, x0 - x : x1 - x][m]

    def start_vote(self, options: int, seconds: float) -> None:
        """Begin a fresh vote over the first ``options`` zones (used by blackjack)."""
        self.favourite = self.rng.randrange(max(1, min(options, len(self.zones))))
        for f in self.fish:
            f.loyal = self.rng.random() < 0.8
        self.vote_start = self.t
        self.vote_seconds = seconds
        self.vote_open = True

    def end_vote(self) -> None:
        """Let the fish wander again between votes."""
        self.vote_open = False

    def read(self) -> tuple[bool, np.ndarray]:
        dt = 1.0 / FPS
        self.t += dt
        deciding = self.vote_open and self.t - self.vote_start > self.vote_seconds * 0.45
        loyal = [f for f in self.fish if f.loyal]
        for f in self.fish:
            if deciding and f.loyal:
                # Each committed fish holds its own lane in the chosen zone, so the school
                # crowds in without piling on top of each other.
                zone = self.zones[self.favourite]
                lane = loyal.index(f)
                x0, x1 = zone.x / SCALE, (zone.x + zone.w) / SCALE
                y0, y1 = zone.y / SCALE, (zone.y + zone.h) / SCALE
                lane_y = y0 + (lane + 0.5) * (y1 - y0) / len(loyal)
                if abs(f.target[1] - lane_y) > 4 or not (x0 + 14 <= f.target[0] <= x1 - 14):
                    f.target = (self.rng.uniform(x0 + 16, x1 - 16), lane_y)
                elif math.hypot(f.target[0] - f.x, f.target[1] - f.y) < 6:
                    f.target = (self.rng.uniform(x0 + 16, x1 - 16), lane_y)
            elif math.hypot(f.target[0] - f.x, f.target[1] - f.y) < 10:
                # Undecided fish drift around the other zones once the vote is on.
                others = [z for i, z in enumerate(self.zones) if i != self.favourite]
                f.target = self._random_point(self.rng.choice(others)) if deciding and others else self._random_point()
            f.step(dt, self.fish)

        img = self.static.copy()
        self._draw_rays(img)
        self._draw_plants(img)
        self._draw_bubbles(img, dt)
        plain = self.plain_bg.copy()
        for f in sorted(self.fish, key=lambda fish: fish.y):
            sprite, mask = self._sprite_for(f)
            x = int(round(f.x)) - sprite.shape[1] // 2
            y = int(round(f.y)) - sprite.shape[0] // 2
            # Soft pixel shadow on the sand under each fish.
            sy = SAND_TOP + 2
            for dx in range(-7, 8):
                sx = int(round(f.x)) + dx
                if 0 <= sx < LOW_W and abs(dx) < 7 - (sy - f.y) / 30:
                    img[sy, sx] = (img[sy, sx].astype(int) * 0.8).astype(np.uint8)
            self._paste(img, sprite, mask, x, y)
            self._paste(plain, np.zeros_like(sprite), mask, x, y)
        self._draw_surface(img)

        self.detection_frame = cv2.resize(plain, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        return True, cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
