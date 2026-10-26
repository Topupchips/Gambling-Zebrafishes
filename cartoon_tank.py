"""A hand-drawn cartoon aquarium.

Five big-eyed cartoon zebrafish in a watercolour tank: ink outlines with a slight
"boil" (the line wobble of hand-drawn animation), fills painted a touch off-register,
paper grain over everything, and the tank drawn on twos (a new drawing every second
frame), like traditional animation.

Two ways to use it:

- Demo: the cartoon fish run their own simulated school and votes
  (``--demo``, the default demo style for the studio look).
- Live: ``drive(points)`` places the cartoon fish where your real fish are, so the
  reel shows your actual fish, drawn (``blackjack.py --cartoon``).
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from pixel_tank import LOW_H, LOW_W, SAND_TOP, WATER_TOP, PixelFish, PixelTank
from utils import Zone

K = 8  # tank pixels per simulation unit
WIDTH, HEIGHT = LOW_W * K, LOW_H * K  # 2000 x 1120
FPS = 30.0
SHIFT = 3  # sub-pixel bits for anti-aliased fills
WATERLINE = 76
SAND_Y = 944


def _c(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))


INK = _c("#1E2228")
INK_SOFT = _c("#3A4350")
PAPER = _c("#F3ECDF")
AIR = _c("#F5F1E6")
BODY = _c("#F4D79C")
BACK = _c("#BCA26A")
STRIPE = _c("#3D68AE")
BELLY = _c("#FFF6E2")
FIN = _c("#BFDDEC")
FIN_LINE = _c("#5A86C2")
BLUSH = _c("#EE9A92")
EYE_WHITE = _c("#FFFDF7")
PLANT = [_c(c) for c in ("#8CC98A", "#6DB178", "#4E9465", "#3B7653")]
SAND_LIGHT = _c("#EBCF96")
SAND_DARK = _c("#D2AE73")
ROCK = [_c(c) for c in ("#B9C0C8", "#9AA3AF", "#7D8794")]
WOOD = [_c(c) for c in ("#C98A4B", "#A86A35", "#7E4D25")]
GOLD = [_c(c) for c in ("#FFE08A", "#F2C14E", "#C9962E")]
BUBBLE_FILL = _c("#EAF7F7")


def demo_zones(labels: list[str]) -> list[Zone]:
    """Columns over the water, in this tank's pixels."""
    n = max(1, len(labels))
    margin, gap = 6, 4
    zone_w = (LOW_W - margin * 2 - gap * (n - 1)) // n
    top, bottom = WATER_TOP + 4, SAND_TOP - 2
    return [
        Zone(label, (margin + i * (zone_w + gap)) * K, top * K, zone_w * K, (bottom - top) * K)
        for i, label in enumerate(labels)
    ]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _noise1d(n: int, rng: np.random.Generator, smooth: int = 2) -> np.ndarray:
    v = rng.normal(0, 1, n + 12)
    k = np.array([1, 4, 6, 4, 1], np.float32) / 16
    for _ in range(smooth):
        v = np.convolve(v, k, mode="same")
    v = v[6 : 6 + n]
    return v / (np.abs(v).max() + 1e-6)


def _fixed(pts: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(pts, np.float64) * (1 << SHIFT)).astype(np.int32)


def fill(img: np.ndarray, pts, colour, offset: tuple[float, float] = (0.0, 0.0)) -> None:
    p = np.asarray(pts, np.float32) + np.array(offset, np.float32)
    cv2.fillPoly(img, [_fixed(p)], colour, cv2.LINE_AA, SHIFT)


def ink(img: np.ndarray, pts, width, colour=INK, seed: int = 0, taper=(0.3, 0.3), boil: float = 1.2) -> None:
    """A brush stroke: varying width, tapered ends, a little hand wobble."""
    pts = np.asarray(pts, np.float32)
    n = len(pts)
    if n < 2:
        return
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    if boil:
        pts = pts + np.stack([_noise1d(n, rng), _noise1d(n, rng)], 1) * boil
    d = np.gradient(pts, axis=0)
    norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / norm
    t = np.linspace(0, 1, n)
    w = np.broadcast_to(np.asarray(width, np.float32), (n,)) * (0.82 + 0.18 * _noise1d(n, rng))
    a, b = taper
    prof = np.ones(n, np.float32)
    if a > 0:
        prof = np.minimum(prof, 0.2 + 0.8 * np.clip(t / a, 0, 1))
    if b > 0:
        prof = np.minimum(prof, 0.2 + 0.8 * np.clip((1 - t) / b, 0, 1))
    w = w * prof
    left = pts + nrm * w[:, None] / 2
    right = pts - nrm * w[:, None] / 2
    cv2.fillPoly(img, [_fixed(np.concatenate([left, right[::-1]]))], colour, cv2.LINE_AA, SHIFT)


def ink_loop(img: np.ndarray, pts, width, colour=INK, seed: int = 0, boil: float = 1.2) -> None:
    """A closed outline drawn as two overlapping strokes, like a real pen line."""
    pts = np.asarray(pts, np.float32)
    n = len(pts)
    if n < 4:
        return
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    if boil:
        jx, jy = _noise1d(n + 6, rng)[:n], _noise1d(n + 6, rng)[:n]
        pts = pts + np.stack([jx, jy], 1) * boil
    h = n // 2
    first = pts[: h + 3]
    second = np.concatenate([pts[h - 1 :], pts[:3]])
    ink(img, first, width, colour, seed + 1, taper=(0.0, 0.0), boil=0)
    ink(img, second, width, colour, seed + 2, taper=(0.0, 0.0), boil=0)


def blend_region(img: np.ndarray, overlay_fn, alpha: float, box: tuple[int, int, int, int]) -> None:
    """Draw with ``overlay_fn`` onto a copy of a region and mix it back at ``alpha``."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1]
    layer = region.copy()
    overlay_fn(layer, x0, y0)
    region[:] = cv2.addWeighted(layer, alpha, region, 1 - alpha, 0)


# ---------------------------------------------------------------------------
# Fish
# ---------------------------------------------------------------------------
B = 205.0  # body length in pixels
HM = 0.205 * B  # half height at the deepest point
TAIL = 84.0
_PROFILE_S = np.array([0.0, 0.015, 0.05, 0.11, 0.2, 0.34, 0.55, 0.78, 1.0], np.float32)
_PROFILE_H = np.array([0.0, 0.34, 0.6, 0.8, 0.93, 1.0, 0.82, 0.5, 0.3], np.float32)
_TAIL_SHAPE = np.array(
    [(0.0, -0.32), (0.3, -0.58), (0.66, -1.05), (0.98, -1.32), (1.0, -1.18), (0.86, -0.62), (0.64, -0.1), (0.6, 0.0),
     (0.64, 0.1), (0.86, 0.62), (1.0, 1.18), (0.98, 1.32), (0.66, 1.05), (0.3, 0.58), (0.0, 0.32)],
    np.float32,
)


class Look:
    """Per-fish drawing state: turn animation, blinking, pitch."""

    def __init__(self, rng: random.Random, index: int, facing: int) -> None:
        self.index = index
        self.dir = float(facing)
        self.pitch = 0.0
        self.next_blink = rng.uniform(1.0, 4.0)
        self.blink_until = 0.0
        self.rng = rng


def _rotate(pts: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c], 1)


class CartoonTank(PixelTank):
    """Frame source with the same ``read()`` contract as ``cv2.VideoCapture``."""

    pixel_art = False
    illustrated = True
    scale = K
    blob_area = (900, 40000)

    def __init__(self, zones: list[Zone], countdown: float, fish: int = 5, favourite: int | None = None, seed: int | None = None) -> None:
        super().__init__(zones, countdown, fish=fish, favourite=favourite, seed=seed)
        self.looks = [Look(self.rng, i, f.facing) for i, f in enumerate(self.fish)]
        self.np_rng = np.random.default_rng(self.rng.randrange(1 << 30))
        self.paper = self._make_paper()
        self.wash = self._make_wash()
        self.statics = [self._make_ink_layer(seed=v) for v in range(3)]
        self.cplants = self._make_cartoon_plants()
        self.cbubbles: list[dict] = []
        self.frame_index = 0
        self.cached: np.ndarray | None = None
        self.plain_bg = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)
        self.empty_frame = self.plain_bg.copy()
        self.detection_frame = self.empty_frame

    # Static art --------------------------------------------------------------
    def _make_paper(self) -> np.ndarray:
        rng = self.np_rng
        fibre = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 0.9)
        blotch = cv2.resize(rng.normal(0, 1, (HEIGHT // 60, WIDTH // 60)).astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        tooth = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 2.2)
        p = 0.955 + 0.022 * fibre + 0.012 * blotch + 0.018 * tooth
        p = np.clip(p, 0.82, 1.0)
        return cv2.merge([(p * 255).astype(np.uint8)] * 3)

    def _make_wash(self) -> np.ndarray:
        """Watercolour water, sand and air, painted once."""
        rng = self.np_rng
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        img = np.ones((HEIGHT, WIDTH, 3), np.float32) * np.array(PAPER, np.float32)

        def noise(cell: float) -> np.ndarray:
            g = rng.normal(0, 1, (max(2, int(HEIGHT / cell)), max(2, int(WIDTH / cell)))).astype(np.float32)
            f = cv2.resize(g, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            return f / (np.abs(f).max() + 1e-6)

        top = np.array(_c("#D3ECE8"), np.float32)
        bottom = np.array(_c("#86C2C4"), np.float32)
        depth = np.clip((yy - WATERLINE) / (SAND_Y - WATERLINE), 0, 1)
        n1, n2 = noise(260), noise(90)
        t = np.clip(depth * 0.9 + 0.12 * n1 + 0.05 * n2, 0, 1)[..., None]
        water = top * (1 - t) + bottom * t
        # Pigment pooling: darker rims where the wash dried, like real watercolour.
        n3 = noise(180)
        rims = cv2.GaussianBlur((np.exp(-((n3 - 0.15) / 0.05) ** 2)).astype(np.float32), (0, 0), 3)
        water *= (1 - 0.025 * rims)[..., None]
        # Light reserved as paper: soft diagonal rays.
        for x0, wd, k in ((380, 120, 0.13), (900, 70, 0.09), (1520, 150, 0.11)):
            band = np.exp(-(((xx - x0 - (yy - WATERLINE) * 0.42) / wd) ** 2)) * np.clip(1 - depth * 1.2, 0, 1)
            water = water * (1 - k * band[..., None]) + np.array(PAPER, np.float32) * k * band[..., None]
        img = np.where((yy >= WATERLINE - 10)[..., None], water, img)
        # Sand.
        sand_top = self._sand_line(np.arange(WIDTH, dtype=np.float32))
        sd = np.clip((yy - sand_top[None, :]) / (HEIGHT - SAND_Y + 30), 0, 1)
        sn = noise(70)
        st = np.clip(sd * 0.85 + 0.15 * sn, 0, 1)[..., None]
        sand = np.array(SAND_LIGHT, np.float32) * (1 - st) + np.array(SAND_DARK, np.float32) * st
        img = np.where((yy >= sand_top[None, :])[..., None], sand, img)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _sand_line(x: np.ndarray) -> np.ndarray:
        return SAND_Y + 12 * np.sin(x * 0.0055) + 7 * np.sin(x * 0.017 + 1.0) + 3 * np.sin(x * 0.05 + 2.0)

    def _make_ink_layer(self, seed: int) -> np.ndarray:
        """Ink and the things that do not move, drawn three times for line boil."""
        img = self.wash.copy()
        rng = random.Random(100 + seed)
        srng = random.Random(7)  # positions stay put across the three drawings
        xs = np.arange(0, WIDTH + 20, 20, dtype=np.float32)
        sand_pts = np.stack([xs, self._sand_line(xs)], 1)
        # Stipple in the sand (same dots every drawing).
        nrng = np.random.default_rng(5)
        for _ in range(900):
            x = float(nrng.uniform(0, WIDTH))
            y = float(nrng.uniform(self._sand_line(np.array([x]))[0] + 12, HEIGHT - 6))
            r = float(nrng.choice((1.4, 1.8, 2.4)))
            cv2.circle(img, (int(x * 8), int(y * 8)), int(r * 8), INK_SOFT, -1, cv2.LINE_AA, 3)
        # Little pebbles.
        for _ in range(26):
            x = srng.uniform(20, WIDTH - 20)
            y = self._sand_line(np.array([x]))[0] + srng.uniform(26, 150)
            r = srng.uniform(7, 15)
            pts = self._blob(x, y, r, r * srng.uniform(0.6, 0.85), srng.random() * 6, 18)
            fill(img, pts, srng.choice(ROCK), (2, 2))
            ink_loop(img, pts, 2.6, INK, seed=rng.randrange(1 << 20), boil=0.8)
        ink(img, sand_pts, 4.2, INK, seed=rng.randrange(1 << 20), taper=(0.0, 0.0), boil=2.0)
        # Rocks sitting on the sand.
        for x, w, h in ((150, 150, 90), (330, 80, 46), (1500, 120, 60), (1800, 190, 112), (1650, 64, 36)):
            base = float(self._sand_line(np.array([x]))[0]) + 14
            pts = self._rock(x, base, w, h, srng)
            fill(img, pts, ROCK[0], (3, 3))
            # Shadow side hatching.
            for i in range(9):
                hx = x + w * 0.12 + i * w * 0.045
                p0 = (hx, base - h * 0.75 + i * 4)
                p1 = (hx + 18, base - 8)
                if cv2.pointPolygonTest(pts.astype(np.float32), p0, False) > 0 and cv2.pointPolygonTest(pts.astype(np.float32), p1, False) > 0:
                    ink(img, [p0, p1], 2.2, INK_SOFT, seed=rng.randrange(1 << 20), boil=0.6)
            ink_loop(img, pts, 4.0, INK, seed=rng.randrange(1 << 20), boil=1.6)
        self._chest(img, 1064, rng)
        # Cartoon glass shine.
        for dx, ln, wd in ((0, 150, 9), (30, 90, 6)):
            p = np.array([(1760 + dx, 120), (1760 + dx + ln * 0.5, 120 + ln)], np.float32)
            ink(img, p, wd, (252, 252, 250), seed=rng.randrange(1 << 20), taper=(0.3, 0.3), boil=0.5)
        return img

    @staticmethod
    def _blob(x: float, y: float, rx: float, ry: float, rot: float, n: int) -> np.ndarray:
        a = np.linspace(0, math.tau, n, endpoint=False)
        pts = np.stack([np.cos(a) * rx, np.sin(a) * ry], 1)
        return _rotate(pts, rot) + np.array([x, y], np.float32)

    @staticmethod
    def _rock(x: float, base: float, w: float, h: float, rng: random.Random) -> np.ndarray:
        a = np.linspace(math.pi, 0, 22)
        bump = np.array([1 + 0.1 * math.sin(i * 1.7 + rng.random()) for i in range(22)], np.float32)
        top = np.stack([x + np.cos(a) * w / 2, base - np.sin(a) * h * bump], 1)
        return np.concatenate([top, [[x + w / 2, base], [x - w / 2, base]]]).astype(np.float32)

    def _chest(self, img: np.ndarray, cx: float, rng: random.Random) -> None:
        base = float(self._sand_line(np.array([cx]))[0]) + 16
        w, h = 170, 92
        x0, x1 = cx - w / 2, cx + w / 2
        body = np.array([(x0 + 4, base - h), (x1 - 4, base - h), (x1, base), (x0, base)], np.float32)
        a = np.linspace(math.pi, 0, 14)
        lid = np.concatenate([
            np.stack([cx + np.cos(a) * (w / 2 - 6), base - h - np.sin(a) * 44 - 6], 1),
            [[x1 - 4, base - h], [x0 + 4, base - h]],
        ]).astype(np.float32)
        fill(img, body, WOOD[0], (3, 3))
        fill(img, lid, WOOD[1], (3, 3))
        for k in (0.35, 0.68):
            y = base - h * k
            ink(img, [(x0 + 6, y), (x1 - 6, y)], 2.4, INK_SOFT, seed=rng.randrange(1 << 20), boil=0.8)
        for bx in (x0 + 26, x1 - 26):
            band = np.array([(bx - 9, base - h - 44), (bx + 9, base - h - 44), (bx + 9, base), (bx - 9, base)], np.float32)
            fill(img, band, WOOD[2], (2, 2))
            ink_loop(img, band, 2.4, INK, seed=rng.randrange(1 << 20), boil=0.8)
        lock = np.array([(cx - 13, base - h - 8), (cx + 13, base - h - 8), (cx + 13, base - h + 22), (cx - 13, base - h + 22)], np.float32)
        fill(img, lock, GOLD[1], (2, 2))
        ink_loop(img, lock, 2.6, INK, seed=rng.randrange(1 << 20), boil=0.8)
        ink_loop(img, body, 4.0, INK, seed=rng.randrange(1 << 20), boil=1.4)
        ink_loop(img, lid, 4.0, INK, seed=rng.randrange(1 << 20), boil=1.4)
        # Spilled coins.
        for i, (dx, dy) in enumerate(((-118, -6), (-98, 2), (-76, -2), (-108, -22), (96, 0), (118, -4))):
            pts = self._blob(cx + dx, base + dy, 13, 8, 0.1 * i, 16)
            fill(img, pts, GOLD[i % 2], (2, 1))
            ink_loop(img, pts, 2.4, INK, seed=rng.randrange(1 << 20), boil=0.6)

    def _make_cartoon_plants(self) -> list[dict]:
        rng = random.Random(21)
        plants = []
        for x, kind in ((60, "kelp"), (250, "grass"), (470, "leafy"), (780, "kelp"), (1290, "grass"), (1390, "leafy"), (1700, "kelp"), (1940, "grass")):
            plants.append({
                "x": x + rng.uniform(-20, 20),
                "h": rng.uniform(330, 560) if kind != "grass" else rng.uniform(150, 260),
                "w": rng.uniform(34, 50),
                "amp": rng.uniform(16, 30),
                "phase": rng.uniform(0, math.tau),
                "kind": kind,
                "shade": rng.randrange(3),
            })
        return plants

    # Per-drawing art -----------------------------------------------------------
    def _draw_cplants(self, img: np.ndarray, t: float, boil: int) -> None:
        for i, p in enumerate(self.cplants):
            base = float(self._sand_line(np.array([p["x"]]))[0]) + 18
            seed = boil * 997 + i * 31
            if p["kind"] == "grass":
                for j in range(3):
                    self._blade(img, p["x"] + (j - 1) * 16, base, p["h"] * (0.75 + 0.18 * j), 14, p["amp"] * 0.6, t, p["phase"] + j, PLANT[(p["shade"] + j) % 4], seed + j)
                continue
            if p["kind"] == "kelp":
                self._blade(img, p["x"], base, p["h"], p["w"], p["amp"], t, p["phase"], PLANT[p["shade"]], seed)
                continue
            # Leafy: a stem with alternating leaves.
            n = 16
            tt = np.linspace(0, 1, n)
            sway = p["amp"] * np.sin(t * 1.2 + p["phase"] + tt * 2.2) * tt ** 1.3
            stem = np.stack([p["x"] + sway, base - tt * p["h"]], 1).astype(np.float32)
            for k in range(2, n - 1, 2):
                side = 1 if (k // 2) % 2 else -1
                cx, cy = stem[k]
                ang = -math.pi / 2 + side * 1.0 + 0.15 * math.sin(t * 1.5 + k)
                leaf = self._leaf(cx, cy, 52 * (1 - 0.4 * tt[k]), 17, ang)
                fill(img, leaf, PLANT[1 + (k % 2)], (2, 2))
                ink_loop(img, leaf, 2.6, INK, seed=seed + k, boil=1.0)
            ink(img, stem, 5.0, PLANT[3], seed=seed + 50, taper=(0.0, 0.5), boil=1.0)
            ink(img, stem + np.array([2.5, 0], np.float32), 2.0, INK, seed=seed + 51, taper=(0.0, 0.5), boil=1.0)

    def _blade(self, img, x, base, h, w, amp, t, phase, colour, seed) -> None:
        n = 18
        tt = np.linspace(0, 1, n, dtype=np.float32)
        sway = amp * np.sin(t * 1.2 + phase + tt * 2.2) * tt ** 1.3
        centre = np.stack([x + sway, base - tt * h], 1)
        d = np.gradient(centre, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        width = w * (0.55 + 0.45 * np.sin(np.minimum(tt * 1.4, 1) * math.pi * 0.95)) * (1 - tt ** 3)
        left = centre + nrm * width[:, None] / 2
        right = centre - nrm * width[:, None] / 2
        fill(img, np.concatenate([left, right[::-1]]), colour, (3, 2))
        ink(img, left, 3.2, INK, seed=seed, taper=(0.0, 0.4), boil=1.1)
        ink(img, right, 3.2, INK, seed=seed + 7, taper=(0.0, 0.4), boil=1.1)
        mid = centre[3:-2]
        ink(img, mid, 1.6, PLANT[3], seed=seed + 9, taper=(0.3, 0.3), boil=0.8)

    @staticmethod
    def _leaf(cx, cy, length, width, angle) -> np.ndarray:
        t = np.linspace(0, 1, 9)
        top = np.stack([t * length, np.sin(t * math.pi) * width / 2], 1)
        bottom = np.stack([t[::-1] * length, -np.sin(t[::-1] * math.pi) * width / 2], 1)
        pts = np.concatenate([top, bottom[1:-1]]).astype(np.float32)
        return _rotate(pts, angle) + np.array([cx, cy], np.float32)

    def _draw_cbubbles(self, img: np.ndarray, dt: float, t: float, boil: int) -> None:
        for sx in (1064.0, 520.0, 1600.0):
            if self.rng.random() < dt * 1.8:
                self.cbubbles.append({"x0": sx + self.rng.uniform(-14, 14), "y": SAND_Y - 40.0, "r": self.rng.choice((5, 7, 9, 12, 15)), "v": self.rng.uniform(80, 150), "ph": self.rng.uniform(0, math.tau)})
        alive = []
        for i, b in enumerate(self.cbubbles):
            b["y"] -= b["v"] * dt
            if b["y"] < WATERLINE + 16:
                continue
            x = b["x0"] + math.sin(t * 2.6 + b["ph"]) * 7
            pts = self._blob(x, b["y"], b["r"], b["r"] * 0.94, 0, 16)
            fill(img, pts, BUBBLE_FILL)
            ink_loop(img, pts, 2.2, INK_SOFT, seed=boil * 13 + i, boil=0.5)
            if b["r"] >= 9:
                hl = np.array([(x - b["r"] * 0.5, b["y"] - b["r"] * 0.15), (x - b["r"] * 0.2, b["y"] - b["r"] * 0.5)], np.float32)
                ink(img, hl, 2.0, (255, 255, 255), seed=i, boil=0)
            alive.append(b)
        self.cbubbles = alive

    def _draw_csurface(self, img: np.ndarray, t: float, boil: int) -> None:
        xs = np.arange(-10, WIDTH + 30, 18, dtype=np.float32)
        ys = WATERLINE + 5 * np.sin(xs * 0.01 + t * 1.6) + 3 * np.sin(xs * 0.023 - t * 1.1)
        line = np.stack([xs, ys], 1)
        air = np.concatenate([line, [[WIDTH + 30, -10], [-10, -10]]]).astype(np.float32)
        fill(img, air, AIR)
        ink(img, line, 4.2, INK, seed=boil * 7 + 3, taper=(0.0, 0.0), boil=1.6)
        for k, x0 in enumerate((240, 690, 1210, 1650)):
            x = x0 + 20 * math.sin(t * 0.7 + k)
            y = WATERLINE + 22 + 4 * math.sin(t * 1.3 + k)
            ink(img, [(x, y), (x + 36, y - 3), (x + 70, y)], 2.4, INK_SOFT, seed=boil * 5 + k, boil=0.8)

    # Fish ------------------------------------------------------------------------
    def _fish_geometry(self, f: PixelFish, look: Look):
        """World-space outline, fins, stripes and eye for one cartoon fish."""
        phase = f.anim * 1.6
        burst = f.burst
        n = 26
        s = np.linspace(0, 1, n, dtype=np.float32)
        x = B * (0.55 - s)
        h = np.interp(s, _PROFILE_S, _PROFILE_H) * HM
        amp = 0.05 * B * (0.45 + 0.75 * burst)
        m = amp * np.sin(phase - 5.0 * s) * s ** 1.6
        top = np.stack([x, m - h], 1)
        bottom = np.stack([x, m + h * 1.07], 1)
        body = np.concatenate([top, bottom[::-1][1:-1]])
        outline = np.concatenate([top[::-1], bottom[1:]])  # open at the tail, so no seam
        # Tail fin follows the body wave.
        slope = (m[-1] - m[-2]) / (x[-2] - x[-1] + 1e-6)
        theta = 0.4 * math.sin(phase - 5.6) * (0.6 + 0.6 * burst) + math.atan(slope)
        tail_local = _TAIL_SHAPE * np.array([TAIL, HM], np.float32)
        tail_local[:, 0] *= -1
        tail = _rotate(tail_local, theta) + np.array([x[-1] + 0.06 * B, m[-1]], np.float32)
        # Dorsal and anal fins.
        def fin_on(edge: np.ndarray, i0: int, i1: int, lift: float) -> np.ndarray:
            base = edge[i0:i1 + 1]
            peak_i = (i0 + i1) // 2
            px = base[:, 0].mean() - B * 0.04
            py = edge[peak_i, 1] + lift
            arc = np.array([[px - B * 0.06, py]], np.float32)
            return np.concatenate([base, arc]).astype(np.float32)
        dorsal = fin_on(top, 11, 16, -0.42 * HM)
        anal = fin_on(np.stack([x, m + h * 1.07], 1), 13, 18, 0.36 * HM)
        # Pectoral fin, flapping.
        flap = 0.45 * math.sin(phase * 1.7)
        pec_base = np.array([x[8] + 2, m[8] + h[8] * 0.35], np.float32)
        tt = np.linspace(0, 1, 9)
        leaf = np.concatenate([np.stack([-tt * 0.2 * B, np.sin(tt * math.pi) * 0.055 * B], 1), np.stack([-tt[::-1][1:-1] * 0.2 * B, -np.sin(tt[::-1][1:-1] * math.pi) * 0.035 * B], 1)]).astype(np.float32)
        pec = _rotate(leaf, 0.4 + flap) + pec_base
        # Stripes and shading paths.
        def band(v: float, s0: float, s1: float) -> np.ndarray:
            sel = (s >= s0) & (s <= s1)
            return np.stack([x[sel], m[sel] + v * h[sel]], 1)
        paths = {
            "back": (band(-0.82, 0.08, 1.0), h[(s >= 0.08) & (s <= 1.0)] * 0.55),
            "s1": (band(-0.36, 0.22, 1.0), h[(s >= 0.22)] * 0.2),
            "s2": (band(0.05, 0.2, 1.0), h[(s >= 0.2)] * 0.2),
            "s3": (band(0.46, 0.26, 1.0), h[(s >= 0.26)] * 0.18),
            "belly": (band(0.82, 0.06, 0.7), h[(s >= 0.06) & (s <= 0.7)] * 0.34),
        }
        eye = np.array([x[3] - 1, m[3] - h[3] * 0.18], np.float32)
        mouth = np.array([(x[0] - 1, m[0] + 3), (x[1] - 2, m[1] + 6), (x[2] - 6, m[2] + 5)], np.float32)
        gill = np.stack([np.full(5, x[7]) + np.array([4, 0, -3, 0, 4]), m[7] + h[7] * np.array([-0.7, -0.35, 0, 0.35, 0.7])], 1).astype(np.float32)
        blush = np.array([x[5] + 2, m[5] + h[5] * 0.5], np.float32)
        return body, outline, tail, dorsal, anal, pec, paths, eye, mouth, gill, blush

    def _to_world(self, pts: np.ndarray, f: PixelFish, look: Look) -> np.ndarray:
        r = _rotate(np.asarray(pts, np.float32), look.pitch)
        return np.stack([f.x * K + look.dir * r[:, 0], f.y * K + r[:, 1]], 1).astype(np.float32)

    def _draw_fish(self, img: np.ndarray, plain: np.ndarray, f: PixelFish, look: Look, t: float, boil: int) -> None:
        body, outline, tail, dorsal, anal, pec, paths, eye, mouth, gill, blush = self._fish_geometry(f, look)
        W = lambda p: self._to_world(p, f, look)  # noqa: E731
        wb, wt = W(body), W(tail)
        seed = boil * 101 + look.index * 17
        reg = (2.5 * look.dir, 2.0)

        # Shadow on the sand.
        cx = f.x * K
        depth = 1 - (SAND_Y - f.y * K) / (SAND_Y - WATERLINE)
        sh_w = B * (0.36 + 0.2 * depth)

        def shadow(layer, ox, oy):
            cv2.ellipse(layer, (int(cx - ox), int(SAND_Y + 18 - oy)), (int(sh_w), int(9 + 5 * depth)), 0, 0, 360, INK_SOFT, -1, cv2.LINE_AA)
        blend_region(img, shadow, 0.10 + 0.14 * depth, (int(cx - sh_w - 4), SAND_Y, int(cx + sh_w + 4), SAND_Y + 40))

        # Fins behind the body.
        for fin in (W(dorsal), W(anal)):
            fill(img, fin, FIN, reg)
            ink_loop(img, fin, 3.0, INK, seed=seed + 3, boil=1.0)
        fill(img, wt, FIN, reg)
        base = W(np.array([[B * -0.39, 0]], np.float32))[0]
        for frac in (0.18, 0.4, 0.6, 0.82):
            tip = wt[int(round(frac * (len(wt) - 1)))]
            ray = np.array([base + (tip - base) * 0.12, base + (tip - base) * 0.85], np.float32)
            ink(img, ray, 2.0, FIN_LINE, seed=seed + int(frac * 100), boil=0.6)
        ink_loop(img, wt, 3.4, INK, seed=seed + 5, boil=1.2)

        # Body: painted fill with stripes, clipped to the outline, a touch off-register.
        x0, y0 = np.floor(wb.min(0) - 12).astype(int)
        x1, y1 = np.ceil(wb.max(0) + 12).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(WIDTH, x1), min(HEIGHT, y1)
        if x1 > x0 and y1 > y0:
            region = img[y0:y1, x0:x1]
            layer = np.empty_like(region)
            layer[:] = BODY
            off = np.array([x0, y0], np.float32)
            for key, colour in (("back", BACK), ("belly", BELLY), ("s1", STRIPE), ("s2", STRIPE), ("s3", STRIPE)):
                pts, width = paths[key]
                if len(pts) > 2:
                    ink(layer, W(pts) - off, width, colour, seed=seed + len(key), taper=(0.15, 0.0) if key.startswith("s") else (0.0, 0.0), boil=0.8)
            mask = np.zeros(region.shape[:2], np.uint8)
            fill(mask, wb - off, 255, reg)
            m = mask.astype(np.float32) / 255
            region[:] = cv2.blendLinear(layer, np.ascontiguousarray(region), m, 1.0 - m)
            pmask = np.zeros(region.shape[:2], np.uint8)
            fill(pmask, wb - off, 255)
            fill(pmask, wt - off, 255)
            plain[y0:y1, x0:x1][pmask > 127] = 0
        # Tail silhouette for detection (outside the body box too).
        fill(plain, wt, (0, 0, 0))
        ink(img, W(outline), 4.6, INK, seed=seed + 11, taper=(0.06, 0.06), boil=1.3)

        # Face and details.
        ink(img, W(gill), 2.4, INK, seed=seed + 13, taper=(0.3, 0.3), boil=0.8)
        ink(img, W(mouth), 2.6, INK, seed=seed + 14, taper=(0.2, 0.4), boil=0.5)
        bw = W(blush[None, :])[0]

        def blush_fn(layer, ox, oy):
            cv2.ellipse(layer, (int(bw[0] - ox), int(bw[1] - oy)), (int(9 * max(abs(look.dir), 0.2)), 5), 0, 0, 360, BLUSH, -1, cv2.LINE_AA)
        blend_region(img, blush_fn, 0.55, (int(bw[0] - 14), int(bw[1] - 10), int(bw[0] + 14), int(bw[1] + 10)))
        wp = W(pec)
        def pec_fn(layer, ox, oy):
            fill(layer, wp - np.array([ox, oy], np.float32), FIN)
        bx0, by0 = np.floor(wp.min(0) - 4).astype(int)
        bx1, by1 = np.ceil(wp.max(0) + 4).astype(int)
        blend_region(img, pec_fn, 0.75, (bx0, by0, bx1, by1))
        ink_loop(img, wp, 2.4, INK, seed=seed + 15, boil=0.7)

        ew = W(eye[None, :])[0]
        re = 0.5 * HM
        sq = max(abs(look.dir), 0.18)
        if t < look.blink_until:
            lid = np.array([(-re, 0), (-re * 0.4, re * 0.35), (re * 0.4, re * 0.35), (re, 0)], np.float32)
            lid[:, 0] *= sq
            ink(img, lid + ew, 3.0, INK, seed=seed + 16, boil=0.4)
        else:
            cv2.ellipse(img, (int(ew[0] * 8), int(ew[1] * 8)), (int(re * sq * 8), int(re * 8)), 0, 0, 360, EYE_WHITE, -1, cv2.LINE_AA, 3)
            ring = self._blob(ew[0], ew[1], re * sq, re, 0, 20)
            ink_loop(img, ring, 3.0, INK, seed=seed + 17, boil=0.6)
            tx, ty = f.target[0] * K - ew[0], f.target[1] * K - ew[1]
            d = math.hypot(tx, ty) + 1e-6
            px = ew[0] + tx / d * re * 0.3 * sq + look.dir * re * 0.12
            py = ew[1] + ty / d * re * 0.28
            pr = re * 0.58
            cv2.ellipse(img, (int(px * 8), int(py * 8)), (int(pr * sq * 8), int(pr * 8)), 0, 0, 360, INK, -1, cv2.LINE_AA, 3)
            cv2.circle(img, (int((px - pr * 0.3 * sq) * 8), int((py - pr * 0.35) * 8)), int(pr * 0.32 * 8), (255, 255, 255), -1, cv2.LINE_AA, 3)

    def _update_looks(self, dt: float, t: float) -> None:
        for f, look in zip(self.fish, self.looks):
            look.dir += (f.facing - look.dir) * min(1.0, dt * 9)
            want = max(-0.4, min(0.4, math.atan2(f.vy, abs(f.vx) + 4)))
            look.pitch += (want - look.pitch) * min(1.0, dt * 5)
            if t >= look.next_blink:
                look.blink_until = t + 0.13
                look.next_blink = t + self.rng.uniform(2.0, 5.0)

    # Frames ----------------------------------------------------------------------
    def _render(self, dt: float) -> tuple[bool, np.ndarray]:
        self._update_looks(dt, self.t)
        self.frame_index += 1
        # Drawn on twos: a new drawing every second frame.
        if self.cached is not None and self.frame_index % 2 == 0:
            return True, self.cached
        drawing = self.frame_index // 2
        boil = drawing // 2  # the ink wobbles to a new drawing every other drawing
        img = self.statics[boil % len(self.statics)].copy()
        self._draw_cplants(img, self.t, boil)
        self._draw_cbubbles(img, dt * 2, self.t, boil)
        plain = self.plain_bg.copy()
        for f in sorted(self.fish, key=lambda fish: fish.y):
            self._draw_fish(img, plain, f, self.looks[self.fish.index(f)], self.t, boil)
        self._draw_csurface(img, self.t, boil)
        img = cv2.multiply(img, self.paper, scale=1 / 255)
        self.detection_frame = plain
        self.cached = img
        return True, img

    def read(self) -> tuple[bool, np.ndarray]:
        dt = 1.0 / FPS
        self._simulate(dt)
        return self._render(dt)

    @staticmethod
    def to_tank(x_norm: float, y_norm: float) -> tuple[float, float]:
        """Camera position (0..1, top-down) to tank units. Left-right is kept exactly so the
        voting columns line up; front-to-back becomes height in the water."""
        x = min(LOW_W - 18, max(18, x_norm * LOW_W))
        y = WATER_TOP + 9 + min(1.0, max(0.0, y_norm)) * (SAND_TOP - 10 - WATER_TOP - 9)
        return x, y

    def drive(self, points: list[tuple[float, float]]) -> tuple[bool, np.ndarray]:
        """Live mode: swim the cartoon fish toward real fish positions (0..1 of the camera frame)."""
        dt = 1.0 / FPS
        self.t += dt
        targets = [self.to_tank(x, y) for x, y in points]
        # Match cartoon fish to real fish, closest pairs first.
        pairs = sorted(
            ((self.fish[i].x - tx) ** 2 + (self.fish[i].y - ty) ** 2, i, j)
            for i in range(len(self.fish))
            for j, (tx, ty) in enumerate(targets)
        )
        used_f, used_t = set(), set()
        for _d, i, j in pairs:
            if i in used_f or j in used_t:
                continue
            used_f.add(i)
            used_t.add(j)
            tx, ty = targets[j]
            fish = self.fish[i]
            fish.target = (tx, ty)
            fish.burst = max(fish.burst, min(1.0, math.hypot(fish.x - tx, fish.y - ty) / 40))
        free = [i for i in range(len(self.fish)) if i not in used_f]
        for i in free:
            fish = self.fish[i]
            if math.hypot(fish.target[0] - fish.x, fish.target[1] - fish.y) < 10:
                fish.target = self._random_point()
        for fish in self.fish:
            fish.step(dt, self.fish)
        return self._render(dt)
