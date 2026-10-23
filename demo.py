"""A simulated tank with five realistic zebrafish, for rehearsing a round without a camera.

``python pick.py --round buy --demo`` runs the real detection and graphics on these
frames, so you can check the look, timing, and recording before the fish are ready.
Demo rounds are never logged to picks.csv, and the HUD labels the feed as simulated.

How the fish are drawn:

- Each fish has a painted skin texture (olive back, blue-black and gold stripes along
  the flanks, silver sheen, translucent fins with rays, eyes with a catch light).
- Every frame the texture is warped along a spine that undulates like a swimming
  zebrafish: the head stays steady, the tail beats, and the body bends into turns.
- They swim in bursts and glides, keep a loose shoal, and cast soft shadows onto a sand
  floor lit by moving caustics.

Detection runs on a plain pass of the same frame (fish over still sand), so the light
show never counts as a fish.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from utils import Zone

WIDTH, HEIGHT = 1280, 720
FPS = 30.0
_g = np.exp(-((np.arange(5) - 2) ** 2) / 1.6).astype(np.float32)
MOTE = (_g[:, None] * _g[None, :])[..., None]


def demo_zones(labels: list[str]) -> list[Zone]:
    n = max(1, len(labels))
    margin, gap = 40, 26
    zone_w = (WIDTH - margin * 2 - gap * (n - 1)) // n
    return [
        Zone(label, margin + i * (zone_w + gap), 48, zone_w, HEIGHT - 96)
        for i, label in enumerate(labels)
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rgb(r: int, g: int, b: int) -> np.ndarray:
    """Colour as a BGR float vector in 0..1."""
    return np.array((b, g, r), np.float32) / 255.0


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _smooth_noise(shape: tuple[int, int], cell: float, rng: np.random.Generator) -> np.ndarray:
    """Band-limited noise in roughly -1..1."""
    h, w = shape
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    coarse = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return field / (np.abs(field).max() + 1e-6)


# ---------------------------------------------------------------------------
# Fish skin
# ---------------------------------------------------------------------------
TEX_L = 384  # texture length in pixels (snout to tail tip)
T_SPAN = 1.9  # texture covers lateral -T..T, in body half-widths
WIDTH_RATIO = 0.085  # one half-width unit as a fraction of body length


def _half_width(u: np.ndarray) -> np.ndarray:
    """Body outline seen from above, in half-width units."""
    head = 0.92 * np.sqrt(np.clip(1 - ((0.11 - u) / 0.11) ** 2, 0, 1))
    grow = 0.92 + 0.08 * _smoothstep(0.11, 0.32, u)
    taper = 0.30 + 0.70 * (0.5 + 0.5 * np.cos(np.pi * np.clip((u - 0.32) / 0.48, 0, 1)))
    out = np.where(u < 0.11, head, np.where(u < 0.32, grow, taper))
    return np.where(u <= 0.80, out, 0.0)


def make_skin(rng: random.Random) -> np.ndarray:
    """Paint a straight zebrafish skin as a BGRA uint8 texture (TEX_L x TEX_H)."""
    unit = TEX_L * WIDTH_RATIO
    tex_h = int(round(2 * T_SPAN * unit))
    u = np.linspace(0, 1, TEX_L, dtype=np.float32)[None, :].repeat(tex_h, 0)
    t = np.linspace(-T_SPAN, T_SPAN, tex_h, dtype=np.float32)[:, None].repeat(TEX_L, 1)
    at = np.abs(t)
    nrng = np.random.default_rng(rng.randrange(1 << 30))

    hue = rng.uniform(-0.04, 0.04)
    olive = _rgb(128, 116, 80) * (1 + hue)
    olive_dark = _rgb(58, 56, 40)
    blue = _rgb(38, 50, 104) * (1 + hue * 2)
    gold = _rgb(226, 204, 146)
    silver = _rgb(196, 204, 210)

    hw = _half_width(u)
    body_in = at < hw
    n = np.where(hw > 0, np.clip(at / np.maximum(hw, 1e-3), 0, 1), 1.0)  # 0 centre .. 1 edge

    # Flank stripes: bands across the half-width, faded in behind the head.
    wobble = 0.035 * np.sin(u * 41 + 1.3) + 0.02 * np.sin(u * 97)
    x = np.clip((n - 0.30) / 0.70 + wobble, 0, 1)
    edges = [0.0, 0.17, 0.31, 0.50, 0.64, 0.82, 1.01]
    cols = [blue, gold, blue, gold, blue, silver]
    flank = np.zeros((*u.shape, 3), np.float32)
    for (a, b), col in zip(zip(edges[:-1], edges[1:]), cols):
        weight = _smoothstep(a - 0.025, a + 0.025, x) * (1 - _smoothstep(b - 0.025, b + 0.025, x))
        flank += weight[..., None] * col
    stripe_on = _smoothstep(0.14, 0.24, u)[..., None] * _smoothstep(0.24, 0.36, n)[..., None]
    colour = olive * (1 - stripe_on) + flank * stripe_on

    # Dorsal midline and a darker saddle.
    midline = (1 - _smoothstep(0.03, 0.09, at)) * _smoothstep(0.12, 0.2, u) * (1 - _smoothstep(0.72, 0.8, u))
    colour = colour * (1 - 0.55 * midline[..., None]) + olive_dark * 0.55 * midline[..., None]

    # Head: paler snout, gill cover arc.
    snout = 1 - _smoothstep(0.0, 0.07, u)
    colour = colour * (1 - 0.35 * snout[..., None]) + _rgb(176, 156, 128) * 0.35 * snout[..., None]
    gill = np.exp(-(((u - 0.15 - 0.025 * (n ** 2)) / 0.006) ** 2)) * _smoothstep(0.3, 0.6, n)
    colour = colour * (1 - 0.25 * gill[..., None]) + _rgb(210, 170, 150) * 0.25 * gill[..., None]

    # Scales and pigment noise.
    lattice = np.sin(u * 420 + t * 9) * np.sin(u * 420 - t * 9)
    speck = cv2.GaussianBlur(nrng.normal(0, 1, u.shape).astype(np.float32), (3, 3), 0.7)
    colour *= (1 + 0.045 * lattice + 0.05 * speck)[..., None]

    # Rounded-body lighting: bright ridge, darker rim, a soft sheen on one flank.
    light = 0.58 + 0.42 * np.sqrt(np.clip(1 - n ** 2, 0, 1))
    colour *= light[..., None]
    sheen = np.exp(-(((t / np.maximum(hw, 0.2)) + 0.38) / 0.16) ** 2) * _smoothstep(0.1, 0.3, u) * (1 - _smoothstep(0.7, 0.8, u))
    colour += _rgb(210, 226, 255) * (0.32 * sheen)[..., None]

    alpha = (1 - _smoothstep(hw - 0.10, hw + 0.02, at)) * (hw > 0)
    # Thin, slightly see-through edge where light passes through the body.
    rim = _smoothstep(0.82, 1.0, n) * (hw > 0)
    colour = colour * (1 - 0.35 * rim[..., None]) + _rgb(222, 214, 190) * 0.35 * rim[..., None]
    alpha *= 1 - 0.25 * rim

    # Caudal fin: translucent, forked, striped, with rays.
    fu = np.clip((u - 0.78) / 0.22, 0, 1)
    fin_hw = 0.30 + 1.20 * np.sin(fu * np.pi * 0.5) ** 1.3
    fork = 0.65 * _smoothstep(0.55, 1.0, fu)
    lobe = (u > 0.78) & (at < fin_hw) & (at > fork * (1 - at / (fin_hw + 1e-3)) * 1.4)
    fin_edge = (1 - _smoothstep(fin_hw - 0.12, fin_hw, at)) * (1 - _smoothstep(0.93, 1.0, u))
    angle = np.arctan2(t, (u - 0.74) * TEX_L / unit)
    rays = 0.86 + 0.14 * np.abs(np.sin(angle * 70))
    fin_bands = 0.5 + 0.5 * np.sin(at / (fin_hw + 1e-3) * np.pi * 5.0)
    fin_col = blue * fin_bands[..., None] + _rgb(214, 196, 150) * (1 - fin_bands[..., None])
    fin_alpha = lobe * fin_edge * rays * (0.36 + 0.20 * fin_bands) * (1 - 0.45 * fu)
    colour = np.where((fin_alpha > alpha)[..., None], fin_col * 0.85, colour)
    alpha = np.maximum(alpha, fin_alpha)

    # Pectoral fins: clear, faintly yellow, swept back.
    for side in (-1, 1):
        du = (u - 0.205) * TEX_L / unit
        dt = t - side * 1.05
        rot = 0.5 * side
        a = du * math.cos(rot) + dt * math.sin(rot)
        b = -du * math.sin(rot) + dt * math.cos(rot)
        pec = np.clip(1 - (a / 0.95) ** 2 - (b / 0.32) ** 2, 0, 1) ** 0.6
        pec_alpha = pec * 0.38 * (~body_in)
        colour = np.where((pec_alpha > alpha)[..., None], _rgb(230, 214, 170), colour)
        alpha = np.maximum(alpha, pec_alpha)

    # Eyes with a silver iris and a catch light.
    for side in (-1, 1):
        dx = (u - 0.07) * TEX_L
        dy = (t - side * 0.66) * unit
        r = np.sqrt(dx ** 2 + dy ** 2)
        er = 0.21 * unit
        iris = r < er
        colour = np.where(iris[..., None], _rgb(176, 160, 118) * (0.6 + 0.4 * (1 - r / er))[..., None], colour)
        colour = np.where((r < er * 0.66)[..., None], _rgb(8, 9, 12), colour)
        glint = np.exp(-((dx + er * 0.25) ** 2 + (dy + er * 0.3) ** 2) / (er * 0.18) ** 2)
        colour = colour * (1 - glint[..., None]) + _rgb(250, 250, 255) * glint[..., None]
        alpha = np.where(iris, 1.0, alpha)

    out = np.zeros((tex_h, TEX_L, 4), np.uint8)
    out[..., :3] = np.clip(colour * 255, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Fish motion and warping
# ---------------------------------------------------------------------------
SPINE_POINTS = 22


class Fish:
    def __init__(self, rng: random.Random, x: float, y: float) -> None:
        self.rng = rng
        self.length = rng.uniform(112, 132)
        self.depth = rng.uniform(0, 1)  # 0 near the floor, 1 near the surface
        self.x, self.y = x, y
        self.heading = rng.uniform(0, math.tau)
        self.turn = 0.0
        self.phase = rng.uniform(0, math.tau)
        self.burst = 0.0
        self.next_burst = rng.uniform(0, 0.8)
        self.cruise = rng.uniform(70, 100)
        self.target = (x, y)
        self.loyal = rng.random() < 0.82
        skin = make_skin(rng)
        tex_l = max(16, int(self.length * 1.6))
        tex_h = max(8, int(skin.shape[0] * tex_l / skin.shape[1]))
        self.skin = cv2.resize(skin, (tex_l, tex_h), interpolation=cv2.INTER_AREA)

    @property
    def unit(self) -> float:
        return self.length * WIDTH_RATIO

    def step(self, dt: float, now: float, school: list["Fish"]) -> None:
        # Burst and glide, like real zebrafish.
        self.next_burst -= dt
        if self.next_burst <= 0:
            self.burst = 1.0
            self.next_burst = self.rng.uniform(0.45, 1.3)
        self.burst = max(0.0, self.burst - dt * 2.2)
        speed = self.cruise * (0.35 + 1.9 * self.burst)

        tx, ty = self.target
        want = math.atan2(ty - self.y, tx - self.x)
        # Keep a loose shoal and some personal space.
        cx = sum(f.x for f in school) / len(school)
        cy = sum(f.y for f in school) / len(school)
        pull = math.atan2(cy - self.y, cx - self.x)
        vx, vy = math.cos(want) + 0.25 * math.cos(pull), math.sin(want) + 0.25 * math.sin(pull)
        for other in school:
            if other is self:
                continue
            d = math.hypot(other.x - self.x, other.y - self.y)
            if 0 < d < self.length * 0.9:
                vx -= (other.x - self.x) / d * 1.2
                vy -= (other.y - self.y) / d * 1.2
        # Walls.
        margin = 70
        if self.x < margin:
            vx += 1.5
        if self.x > WIDTH - margin:
            vx -= 1.5
        if self.y < margin:
            vy += 1.5
        if self.y > HEIGHT - margin:
            vy -= 1.5
        desired = math.atan2(vy, vx)
        delta = (desired - self.heading + math.pi) % math.tau - math.pi
        max_turn = (2.2 + 5.0 * self.burst) * dt
        turn = max(-max_turn, min(max_turn, delta)) + self.rng.gauss(0, 0.6) * dt
        self.heading += turn
        self.turn = self.turn * 0.8 + (turn / max(dt, 1e-3)) * 0.2
        self.x += math.cos(self.heading) * speed * dt
        self.y += math.sin(self.heading) * speed * dt
        self.x = min(WIDTH - 20, max(20, self.x))
        self.y = min(HEIGHT - 20, max(20, self.y))
        self.phase += dt * (7 + 22 * self.burst)

    def spine(self) -> np.ndarray:
        s = np.linspace(0, 1, SPINE_POINTS, dtype=np.float32)
        amp = self.length * (0.045 + 0.085 * self.burst)
        envelope = 0.06 + 0.94 * s ** 2
        lateral = amp * envelope * np.sin(2 * np.pi * 0.9 * s - self.phase)
        lateral += -np.clip(self.turn, -6, 6) * 0.05 * self.length * s ** 2
        fwd = np.array((math.cos(self.heading), math.sin(self.heading)), np.float32)
        left = np.array((-fwd[1], fwd[0]), np.float32)
        head = np.array((self.x, self.y), np.float32) + fwd * self.length * 0.35
        return head[None, :] - (s * self.length)[:, None] * fwd[None, :] + lateral[:, None] * left[None, :]

    def render(self) -> tuple[np.ndarray, int, int] | None:
        """Warp the skin along the current spine. Returns (BGRA patch, x0, y0)."""
        pts = self.spine()
        pad = self.unit * (T_SPAN + 0.4)
        x0 = int(math.floor(pts[:, 0].min() - pad))
        y0 = int(math.floor(pts[:, 1].min() - pad))
        x1 = int(math.ceil(pts[:, 0].max() + pad))
        y1 = int(math.ceil(pts[:, 1].max() + pad))
        w, h = x1 - x0, y1 - y0
        if w <= 2 or h <= 2:
            return None
        # Solve the mapping on a half-resolution grid; the maps are smooth, so upsample them.
        step = 2
        ys, xs = np.mgrid[y0 : y1 + step : step, x0 : x1 + step : step].astype(np.float32)
        gh, gw = ys.shape
        P = np.stack((xs.ravel() + 0.5, ys.ravel() + 0.5), axis=1)  # (M, 2)
        A, B = pts[:-1], pts[1:]
        d = B - A
        dd = (d ** 2).sum(1) + 1e-6
        rel = P[:, None, :] - A[None, :, :]  # (M, N, 2)
        raw = (rel * d[None]).sum(2) / dd[None]
        tc = np.clip(raw, 0, 1)
        closest = A[None] + tc[..., None] * d[None]
        dist2 = ((P[:, None, :] - closest) ** 2).sum(2)
        idx = dist2.argmin(1)
        rows = np.arange(len(P))
        seg_t = tc[rows, idx]
        first, last = idx == 0, idx == len(d) - 1
        seg_t = np.where(first & (raw[rows, idx] < 0), raw[rows, idx], seg_t)
        seg_t = np.where(last & (raw[rows, idx] > 1), raw[rows, idx], seg_t)
        u = (idx + seg_t) / len(d)
        dvec = d[idx]
        rvec = rel[rows, idx]
        cross = (dvec[:, 0] * rvec[:, 1] - dvec[:, 1] * rvec[:, 0]) / np.sqrt(dd[idx])
        v = cross / self.unit
        tex_h, tex_l = self.skin.shape[:2]
        map_x = (u * (tex_l - 1)).reshape(gh, gw).astype(np.float32)
        map_y = ((v + T_SPAN) / (2 * T_SPAN) * (tex_h - 1)).reshape(gh, gw).astype(np.float32)
        full = (gw * step, gh * step)
        map_x = cv2.resize(map_x, full, interpolation=cv2.INTER_LINEAR)[:h, :w]
        map_y = cv2.resize(map_y, full, interpolation=cv2.INTER_LINEAR)[:h, :w]
        patch = cv2.remap(self.skin, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return patch, x0, y0


# ---------------------------------------------------------------------------
# The tank
# ---------------------------------------------------------------------------
class DemoTank:
    """Frame source with the same ``read()`` contract as ``cv2.VideoCapture``.

    ``detection_frame`` holds the plain pass of the latest frame for fish detection.
    """

    def __init__(self, zones: list[Zone], countdown: float, fish: int = 5, favourite: int | None = None, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(self.rng.randrange(1 << 30))
        self.zones = zones
        self.countdown = countdown
        self.favourite = favourite if favourite is not None else self.rng.randrange(len(zones))
        self.t = 0.0
        self.floor = self._make_floor()
        self.plain = self.floor.copy()
        self._make_caustic_fields()
        self.grain = [self.np_rng.normal(0, 3.2, (HEIGHT, WIDTH, 1)).astype(np.float32) for _ in range(4)]
        self.motes = [
            [self.rng.uniform(0, WIDTH), self.rng.uniform(0, HEIGHT), self.rng.uniform(-6, 6), self.rng.uniform(-4, 4), self.rng.uniform(0.6, 2.2)]
            for _ in range(46)
        ]
        cx, cy = self.rng.uniform(300, WIDTH - 300), self.rng.uniform(200, HEIGHT - 200)
        self.fish = [Fish(self.rng, cx + self.rng.uniform(-120, 120), cy + self.rng.uniform(-90, 90)) for _ in range(fish)]
        for f in self.fish:
            f.target = self._random_point()
        self.detection_frame = self.plain

    # cv2.VideoCapture look-alikes -------------------------------------------
    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def get(self, _prop) -> float:
        return FPS

    def release(self) -> None:
        pass

    def set(self, *_args) -> bool:
        return True

    # Scene ------------------------------------------------------------------
    def _random_point(self, zone: Zone | None = None) -> tuple[float, float]:
        if zone is None:
            return (self.rng.uniform(110, WIDTH - 110), self.rng.uniform(110, HEIGHT - 110))
        return (
            self.rng.uniform(zone.x + zone.w * 0.22, zone.x + zone.w * 0.78),
            self.rng.uniform(zone.y + zone.h * 0.22, zone.y + zone.h * 0.78),
        )

    def _make_floor(self) -> np.ndarray:
        rng = self.np_rng
        shape = (HEIGHT, WIDTH)
        sand = _rgb(196, 176, 140)
        img = np.ones((*shape, 3), np.float32) * sand
        # Dunes and ripples at several scales.
        img *= (1 + 0.06 * _smooth_noise(shape, 260, rng) + 0.04 * _smooth_noise(shape, 70, rng))[..., None]
        ripple = np.sin(np.mgrid[0:HEIGHT, 0:WIDTH][1] * 0.045 + 6 * _smooth_noise(shape, 180, rng)) * 0.018
        img *= (1 + ripple)[..., None]
        # Grains: per-pixel tone and a sprinkle of coloured grains.
        grain = rng.normal(0, 1, shape).astype(np.float32)
        img *= (1 + 0.07 * cv2.GaussianBlur(grain, (3, 3), 0.6))[..., None]
        for colour, count in ((_rgb(120, 104, 84), 9000), (_rgb(236, 230, 214), 9000), (_rgb(150, 122, 96), 4000), (_rgb(60, 56, 52), 1500)):
            ys = rng.integers(0, HEIGHT, count)
            xs = rng.integers(0, WIDTH, count)
            img[ys, xs] = img[ys, xs] * 0.4 + colour * 0.6
        img = cv2.GaussianBlur(img, (3, 3), 0.55)

        # Pebbles, kept mostly to the edges so the zones stay readable.
        layer = (img * 255).astype(np.uint8)
        for _ in range(70):
            edge = self.rng.random() < 0.75
            if edge:
                side = self.rng.choice(("top", "bottom", "left", "right"))
                x = self.rng.uniform(0, WIDTH) if side in ("top", "bottom") else (self.rng.uniform(0, 60) if side == "left" else self.rng.uniform(WIDTH - 60, WIDTH))
                y = self.rng.uniform(0, 44) if side == "top" else self.rng.uniform(HEIGHT - 44, HEIGHT) if side == "bottom" else self.rng.uniform(0, HEIGHT)
            else:
                x, y = self.rng.uniform(0, WIDTH), self.rng.uniform(0, HEIGHT)
            r = self.rng.uniform(3, 9) if not edge else self.rng.uniform(5, 16)
            self._pebble(layer, x, y, r)
        return layer

    def _pebble(self, img: np.ndarray, x: float, y: float, r: float) -> None:
        pad = int(r * 2 + 6)
        x0, y0 = int(x) - pad, int(y) - pad
        size = pad * 2
        if x0 < 0 or y0 < 0 or x0 + size > WIDTH or y0 + size > HEIGHT:
            return
        roi = img[y0 : y0 + size, x0 : x0 + size].astype(np.float32) / 255
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        ang = self.rng.uniform(0, math.pi)
        ax, by = r, r * self.rng.uniform(0.6, 0.95)
        dx, dy = xx - pad, yy - pad
        a = (dx * math.cos(ang) + dy * math.sin(ang)) / ax
        b = (-dx * math.sin(ang) + dy * math.cos(ang)) / by
        q = a ** 2 + b ** 2
        mask = 1 - _smoothstep(0.85, 1.0, q)
        shadow = 1 - _smoothstep(0.6, 1.6, ((dx - r * 0.35) / ax) ** 2 + ((dy - r * 0.45) / by) ** 2)
        tone = self.rng.choice((_rgb(132, 124, 112), _rgb(112, 96, 80), _rgb(176, 166, 150), _rgb(92, 86, 78), _rgb(150, 128, 98), _rgb(70, 66, 60)))
        shade = 0.55 + 0.6 * np.clip(1 - q, 0, 1) ** 0.5 - 0.25 * np.clip((dx + dy) / (2 * r), -1, 1)
        spec = np.exp(-(((dx + r * 0.35) ** 2 + (dy + r * 0.4) ** 2) / (r * 0.45) ** 2)) * 0.12
        speck = 1 + 0.12 * self.np_rng.normal(0, 1, (size, size)).astype(np.float32)
        stone = tone * (shade * speck)[..., None] + spec[..., None]
        roi = roi * (1 - 0.35 * shadow[..., None])
        roi = roi * (1 - mask[..., None]) + stone * mask[..., None]
        img[y0 : y0 + size, x0 : x0 + size] = np.clip(roi * 255, 0, 255).astype(np.uint8)

    def _make_caustic_fields(self) -> None:
        qh, qw = HEIGHT // 4, WIDTH // 4
        yy, xx = np.mgrid[0:qh, 0:qw].astype(np.float32)
        scale = 6.28318 * 1.7 / qh
        # Warp the grid a little so the pattern never lines up into rows.
        wx = 0.6 * _smooth_noise((qh, qw), 40, self.np_rng)
        wy = 0.6 * _smooth_noise((qh, qw), 40, self.np_rng)
        self.caustic_grid = (xx * scale + wx - 250.0, yy * scale + wy - 250.0)
        # Soft light falloff: brighter from the top left.
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        d = np.sqrt(((xx - WIDTH * 0.3) / WIDTH) ** 2 + ((yy - HEIGHT * 0.2) / HEIGHT) ** 2)
        self.falloff = (1.08 - 0.32 * d)[..., None].astype(np.float32)
        self.water = np.array((10, 6, -6), np.float32)  # BGR tint push: cooler water

    def _caustics(self, t: float) -> np.ndarray:
        """Animated caustic network (iterated domain warp), solved at quarter resolution."""
        p = self.caustic_grid
        ix, iy = p[0].copy(), p[1].copy()
        c = np.ones_like(ix)
        inten = 0.005
        iterations = 5
        tt = t * 0.32 + 23.0
        for n in range(iterations):
            k = tt * (1.0 - 3.5 / (n + 1))
            nx = p[0] + np.cos(k - ix) + np.sin(k + iy)
            ny = p[1] + np.sin(k - iy) + np.cos(k + ix)
            ix, iy = nx, ny
            c += 1.0 / np.sqrt((p[0] / (np.sin(ix + k) / inten)) ** 2 + (p[1] / (np.cos(iy + k) / inten)) ** 2)
        c /= iterations
        c = 1.17 - np.power(np.abs(c), 1.4)
        light = np.power(np.abs(c), 8.0)
        light = np.clip(light * 1.6 - 0.05, 0, 1.6).astype(np.float32)
        light = cv2.resize(light, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        return cv2.GaussianBlur(light, (5, 5), 1.0)

    # Frames -----------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray]:
        dt = 1.0 / FPS
        self.t += dt
        deciding = self.t > self.countdown * 0.45
        for f in self.fish:
            if deciding and f.loyal:
                zone = self.zones[self.favourite]
                if not zone.contains(int(f.target[0]), int(f.target[1])):
                    f.target = self._random_point(zone)
            if math.hypot(f.target[0] - f.x, f.target[1] - f.y) < 50:
                f.target = self._random_point(self.zones[self.favourite] if deciding and f.loyal else None)
            f.step(dt, self.t, self.fish)

        renders = []
        for f in sorted(self.fish, key=lambda fish: fish.depth):
            r = f.render()
            if r is not None:
                renders.append((f, *r))

        # Lit floor with caustics, then soft shadows.
        floor = self.floor.astype(np.float32)
        light = self.falloff * (0.94 + 0.30 * self._caustics(self.t)[..., None])
        frame = floor * light + self.water
        for f, patch, x0, y0 in renders:
            off_x = int(8 + 16 * f.depth)
            off_y = int(12 + 22 * f.depth)
            k = int(7 + 10 * f.depth) | 1
            a = cv2.GaussianBlur(patch[..., 3].astype(np.float32) / 255, (k, k), 0) * (0.42 - 0.18 * f.depth)
            self._multiply(frame, 1 - a, x0 + off_x, y0 + off_y)

        plain = self.floor.astype(np.float32)
        for f, patch, x0, y0 in renders:
            a = patch[..., 3:4].astype(np.float32) / 255
            colour = patch[..., :3].astype(np.float32)
            # Fish nearer the surface catch a little more light.
            colour = colour * (0.92 + 0.16 * f.depth)
            self._over(frame, colour, a, x0, y0)
            self._over(plain, colour * 0.6, a, x0, y0)

        # Drifting motes and film grain.
        for m in self.motes:
            m[0] = (m[0] + m[2] * dt) % WIDTH
            m[1] = (m[1] + m[3] * dt) % HEIGHT
            x, y = int(m[0]), int(m[1])
            if 2 <= x < WIDTH - 3 and 2 <= y < HEIGHT - 3:
                frame[y - 2 : y + 3, x - 2 : x + 3] += MOTE * (14 * m[4])
        frame += self.grain[int(self.t * FPS) % len(self.grain)]

        self.detection_frame = np.clip(plain, 0, 255).astype(np.uint8)
        return True, np.clip(frame, 0, 255).astype(np.uint8)

    @staticmethod
    def _bounds(x0: int, y0: int, h: int, w: int):
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(WIDTH, x0 + w), min(HEIGHT, y0 + h)
        if cx0 >= cx1 or cy0 >= cy1:
            return None
        return cx0, cy0, cx1, cy1

    def _multiply(self, frame: np.ndarray, factor: np.ndarray, x0: int, y0: int) -> None:
        b = self._bounds(x0, y0, *factor.shape[:2])
        if b is None:
            return
        cx0, cy0, cx1, cy1 = b
        frame[cy0:cy1, cx0:cx1] *= factor[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0, None]

    def _over(self, frame: np.ndarray, colour: np.ndarray, alpha: np.ndarray, x0: int, y0: int) -> None:
        b = self._bounds(x0, y0, *alpha.shape[:2])
        if b is None:
            return
        cx0, cy0, cx1, cy1 = b
        a = alpha[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        c = colour[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        roi = frame[cy0:cy1, cx0:cx1]
        roi[:] = roi * (1 - a) + c * a
