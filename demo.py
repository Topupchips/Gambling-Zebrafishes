"""A fake tank with five swimming fish, for rehearsing a round without a camera.

``python pick.py --round buy --demo`` runs the real detection and graphics on these
frames, so you can check the look, timing, and recording before the fish are ready.
Demo rounds are never logged to picks.csv.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from utils import Zone

WIDTH, HEIGHT = 1280, 720
FPS = 30.0


def demo_zones(labels: list[str]) -> list[Zone]:
    n = max(1, len(labels))
    margin, gap = 40, 26
    zone_w = (WIDTH - margin * 2 - gap * (n - 1)) // n
    return [
        Zone(label, margin + i * (zone_w + gap), 48, zone_w, HEIGHT - 96)
        for i, label in enumerate(labels)
    ]


class DemoTank:
    """Frame source with the same ``read()`` contract as ``cv2.VideoCapture``."""

    def __init__(self, zones: list[Zone], countdown: float, fish: int = 5, favourite: int | None = None, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.zones = zones
        self.countdown = countdown
        self.favourite = favourite if favourite is not None else self.rng.randrange(len(zones))
        self.t = 0.0
        self.background = self._make_background()
        self.fish = [
            {
                "x": self.rng.uniform(120, WIDTH - 120),
                "y": self.rng.uniform(120, HEIGHT - 120),
                "heading": self.rng.uniform(0, math.tau),
                "speed": self.rng.uniform(90, 150),
                "wiggle": self.rng.uniform(0, math.tau),
                "target": self._random_point(),
                "loyal": self.rng.random() < 0.8,
            }
            for _ in range(fish)
        ]

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
            return (self.rng.uniform(80, WIDTH - 80), self.rng.uniform(80, HEIGHT - 80))
        return (
            self.rng.uniform(zone.x + zone.w * 0.2, zone.x + zone.w * 0.8),
            self.rng.uniform(zone.y + zone.h * 0.2, zone.y + zone.h * 0.8),
        )

    def _make_background(self) -> np.ndarray:
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        top = np.array((206, 214, 196), np.float32)  # BGR, pale aqua
        bottom = np.array((168, 190, 186), np.float32)
        t = (yy / HEIGHT)[..., None]
        image = top * (1 - t) + bottom * t
        image += np.random.default_rng(2).normal(0, 3, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        rng = np.random.default_rng(5)
        for _ in range(140):
            cx, cy = int(rng.uniform(0, WIDTH)), int(rng.uniform(0, HEIGHT))
            r = int(rng.uniform(2, 5))
            shade = int(rng.uniform(140, 185))
            cv2.circle(image, (cx, cy), r, (shade, shade + 6, shade + 4), -1, cv2.LINE_AA)
        return cv2.GaussianBlur(image, (3, 3), 0)

    def read(self) -> tuple[bool, np.ndarray]:
        dt = 1.0 / FPS
        self.t += dt
        decide = self.t > self.countdown * 0.45
        frame = self.background.copy()
        for f in self.fish:
            if decide and f["loyal"]:
                zone = self.zones[self.favourite]
                if not zone.contains(int(f["target"][0]), int(f["target"][1])):
                    f["target"] = self._random_point(zone)
            tx, ty = f["target"]
            if math.hypot(tx - f["x"], ty - f["y"]) < 40:
                f["target"] = self._random_point(self.zones[self.favourite] if decide and f["loyal"] else None)
            want = math.atan2(ty - f["y"], tx - f["x"])
            turn = (want - f["heading"] + math.pi) % math.tau - math.pi
            f["heading"] += max(-3.0 * dt, min(3.0 * dt, turn)) + self.rng.uniform(-0.25, 0.25) * dt * 6
            f["wiggle"] += dt * 14
            f["x"] = min(WIDTH - 30, max(30, f["x"] + math.cos(f["heading"]) * f["speed"] * dt))
            f["y"] = min(HEIGHT - 30, max(30, f["y"] + math.sin(f["heading"]) * f["speed"] * dt))
            self._draw_fish(frame, f)
        return True, frame

    @staticmethod
    def _draw_fish(frame: np.ndarray, f: dict) -> None:
        x, y, h = f["x"], f["y"], f["heading"]
        angle = math.degrees(h)
        body = (52, 46, 40)
        cv2.ellipse(frame, (int(x), int(y)), (22, 7), angle, 0, 360, body, -1, cv2.LINE_AA)
        # Stripes.
        for offset in (-8, 0, 8):
            sx = x + math.cos(h) * offset
            sy = y + math.sin(h) * offset
            cv2.ellipse(frame, (int(sx), int(sy)), (2, 6), angle, 0, 360, (84, 76, 68), -1, cv2.LINE_AA)
        # Tail.
        wag = math.sin(f["wiggle"]) * 0.5
        bx, by = x - math.cos(h) * 20, y - math.sin(h) * 20
        tail_dir = h + math.pi + wag
        p1 = (bx + math.cos(tail_dir + 0.45) * 13, by + math.sin(tail_dir + 0.45) * 13)
        p2 = (bx + math.cos(tail_dir - 0.45) * 13, by + math.sin(tail_dir - 0.45) * 13)
        pts = np.array([(bx, by), p1, p2], np.int32)
        cv2.fillPoly(frame, [pts], body, cv2.LINE_AA)
