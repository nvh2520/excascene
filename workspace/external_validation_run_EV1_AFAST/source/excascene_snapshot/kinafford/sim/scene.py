"""Procedural excavation scenes (terrain, materials, piles, trenches, rocks, truck, hazards) and the
BEV ground-truth oracle (height, semantics, material, hazard, visibility, diggable affordance).

Everything is deterministic given the scene seed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

from .kinematics import EXCAVATOR, rot_z, sample_joint_config

# --------------------------------------------------------------------------- constants

TERRAIN_EXTENT = 20.0      # terrain covers [-E, E]^2 in the base frame B
TERRAIN_N = 320            # grid cells per side (res = 0.125 m)
BEV_RES = 0.25
BEV_N = 96
BEV_X0, BEV_Y0 = -8.0, -12.0   # BEV covers x in [-8, 16], y in [-12, 12] in the U frame (front-biased)

# semantic classes used in BEV and image space
SEM_CLASSES = ["terrain", "rock", "truck", "person", "obstacle", "machine", "sky"]
SEM_TERRAIN, SEM_ROCK, SEM_TRUCK, SEM_PERSON, SEM_OBSTACLE, SEM_MACHINE, SEM_SKY = range(7)
N_BEV_SEM = 6              # sky never appears in BEV
MAT_CLASSES = ["sand", "soil", "gravel", "rocky"]
MAT_DIGGABILITY = np.array([1.0, 0.80, 0.65, 0.35])
MAT_ALBEDO = np.array([[0.76, 0.68, 0.50], [0.42, 0.30, 0.20], [0.55, 0.55, 0.53], [0.45, 0.42, 0.40]])
MAT_TEXTURE_AMP = np.array([0.08, 0.15, 0.25, 0.30])

PERSON_EXCL, OBSTACLE_EXCL, TRUCK_BUFFER, MACHINE_BUFFER = 4.0, 1.5, 1.0, 0.5
REACH_MIN, REACH_MAX, REACH_SECTOR = 2.5, 9.5, np.deg2rad(120.0)


# --------------------------------------------------------------------------- helpers


def value_noise(rng: np.random.Generator, n: int, wavelength_cells: float, amp: float) -> np.ndarray:
    """Smooth value noise on an n x n grid."""
    m = int(np.ceil(n / wavelength_cells)) + 2
    coarse = rng.uniform(-1, 1, size=(m, m))
    ys = np.arange(n) / wavelength_cells
    yy, xx = np.meshgrid(ys, ys, indexing="ij")
    return amp * map_coordinates(coarse, [yy, xx], order=3, mode="reflect")


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bev_grid():
    """Cell-centre coordinates (BEV_N, BEV_N) in U: arrays X (columns -> x) and Y (rows -> y)."""
    xs = BEV_X0 + (np.arange(BEV_N) + 0.5) * BEV_RES
    ys = BEV_Y0 + (np.arange(BEV_N) + 0.5) * BEV_RES
    X, Y = np.meshgrid(xs, ys)  # (rows=y, cols=x)
    return X, Y


def point_in_rect(px, py, cx, cy, yaw, hx, hy):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = px - cx, py - cy
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return (np.abs(lx) <= hx) & (np.abs(ly) <= hy)


def dist_to_rect(px, py, cx, cy, yaw, hx, hy):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = px - cx, py - cy
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    ox = np.maximum(np.abs(lx) - hx, 0.0)
    oy = np.maximum(np.abs(ly) - hy, 0.0)
    return np.sqrt(ox * ox + oy * oy)


# --------------------------------------------------------------------------- scene


@dataclass
class Scene:
    seed: int
    h: np.ndarray                 # (N, N) terrain height incl. rocks, base frame, rows->y, cols->x
    mat: np.ndarray               # (N, N) uint8 material
    rock: np.ndarray              # (N, N) bool boulder mask
    z_target: float               # target grade for productivity
    theta_work: float             # working direction (rad, base frame)
    theta_truck: float
    truck: dict | None            # center (x,y), yaw, half sizes, z0
    persons: list = field(default_factory=list)     # [(x,y)]
    obstacles: list = field(default_factory=list)   # [(x,y)]
    light: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.9]))
    tint: np.ndarray = field(default_factory=lambda: np.ones(3))
    ood_terrain: bool = False
    extent: float = TERRAIN_EXTENT

    # --- sampling helpers (base frame) ---
    def _coords(self, x, y):
        n = self.h.shape[0]
        res = 2 * self.extent / n
        cx = (np.asarray(x) + self.extent) / res - 0.5
        cy = (np.asarray(y) + self.extent) / res - 0.5
        return np.clip(cy, 0, n - 1), np.clip(cx, 0, n - 1)

    def height_at(self, x, y):
        cy, cx = self._coords(x, y)
        return map_coordinates(self.h, [cy, cx], order=1, mode="nearest")

    def material_at(self, x, y):
        cy, cx = self._coords(x, y)
        return self.mat[np.round(cy).astype(int), np.round(cx).astype(int)]

    def rock_at(self, x, y):
        cy, cx = self._coords(x, y)
        return self.rock[np.round(cy).astype(int), np.round(cx).astype(int)]

    def normal_at(self, x, y):
        n = self.h.shape[0]
        res = 2 * self.extent / n
        if not hasattr(self, "_gy"):
            gy, gx = np.gradient(self.h, res)
            self._gx, self._gy = gx.astype(np.float32), gy.astype(np.float32)
        cy, cx = self._coords(x, y)
        gx = map_coordinates(self._gx, [cy, cx], order=1, mode="nearest")
        gy = map_coordinates(self._gy, [cy, cx], order=1, mode="nearest")
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        return nrm

    def meta(self) -> dict:
        d = dict(seed=self.seed, z_target=self.z_target, theta_work=self.theta_work,
                 theta_truck=self.theta_truck, truck=self.truck, persons=self.persons,
                 obstacles=self.obstacles, ood_terrain=self.ood_terrain)
        return json.loads(json.dumps(d, default=lambda o: float(o) if np.isscalar(o) else list(o)))


def generate_scene(seed: int, ood_terrain: bool = False) -> Scene:
    rng = np.random.default_rng(1_000_003 * (seed + 1) + 17)
    N, E = TERRAIN_N, TERRAIN_EXTENT
    res = 2 * E / N
    xs = -E + (np.arange(N) + 0.5) * res
    X, Y = np.meshgrid(xs, xs)  # rows -> y, cols -> x

    # base terrain: multi-octave noise scaled by a roughness parameter
    rough = rng.uniform(0.3, 1.0) * (1.4 if ood_terrain else 1.0)
    h = np.zeros((N, N))
    for wl_m, amp in [(12.0, 0.6), (6.0, 0.3), (3.0, 0.15), (1.5, 0.08)]:
        h += value_noise(rng, N, wl_m / res, amp * rough)
    base_level = float(np.median(h))

    theta_work = rng.uniform(-np.pi, np.pi)
    side = rng.choice([-1.0, 1.0])
    theta_truck = theta_work + side * rng.uniform(np.deg2rad(70), np.deg2rad(110))

    # piles (Gaussian bumps) in front of the working direction
    n_piles = rng.integers(1, 4)
    for _ in range(n_piles):
        r = rng.uniform(4.0, 9.0)
        a = theta_work + rng.uniform(-np.deg2rad(60), np.deg2rad(60))
        cx, cy = r * np.cos(a), r * np.sin(a)
        if ood_terrain:
            hp, sg = rng.uniform(3.0, 5.0), rng.uniform(1.0, 2.0)
        else:
            hp, sg = rng.uniform(1.0, 3.5), rng.uniform(1.5, 3.5)
        sgx, sgy = sg * rng.uniform(0.7, 1.3), sg * rng.uniform(0.7, 1.3)
        h += hp * np.exp(-0.5 * (((X - cx) / sgx) ** 2 + ((Y - cy) / sgy) ** 2))

    # pits (negative bumps)
    for _ in range(rng.integers(0, 3)):
        r = rng.uniform(3.0, 11.0)
        a = rng.uniform(-np.pi, np.pi)
        cx, cy = r * np.cos(a), r * np.sin(a)
        h -= rng.uniform(0.4, 1.5) * np.exp(-0.5 * (((X - cx) / rng.uniform(1.0, 2.5)) ** 2 + ((Y - cy) / rng.uniform(1.0, 2.5)) ** 2))

    # optional trench (rectangular depression with soft edges)
    if rng.uniform() < 0.5:
        r = rng.uniform(4.0, 8.5)
        a = theta_work + rng.uniform(-np.deg2rad(50), np.deg2rad(50))
        cx, cy = r * np.cos(a), r * np.sin(a)
        yaw = a + (0.0 if rng.uniform() < 0.5 else np.pi / 2)
        L, W, D = rng.uniform(4.0, 8.0), rng.uniform(1.0, 2.0), rng.uniform(0.6, 2.0)
        c, s = np.cos(yaw), np.sin(yaw)
        lx = c * (X - cx) + s * (Y - cy)
        ly = -s * (X - cx) + c * (Y - cy)
        prof = smoothstep(-np.abs(lx), -L / 2 - 0.4, -L / 2 + 0.4) * smoothstep(-np.abs(ly), -W / 2 - 0.3, -W / 2 + 0.3)
        h -= D * prof

    # materials: Voronoi patches
    n_seed = rng.integers(4, 9)
    sx, sy = rng.uniform(-E, E, n_seed), rng.uniform(-E, E, n_seed)
    if ood_terrain:
        cls = rng.choice(4, size=n_seed, p=[0.15, 0.25, 0.25, 0.35])
    else:
        cls = rng.choice(4, size=n_seed, p=[0.30, 0.35, 0.20, 0.15])
    if len(np.unique(cls)) < 2:
        cls[0] = (cls[0] + 1) % 4
    d2 = (X[None] - sx[:, None, None]) ** 2 + (Y[None] - sy[:, None, None]) ** 2
    mat = cls[np.argmin(d2, axis=0)].astype(np.uint8)

    # boulders: small bumps + rock mask
    rock = np.zeros((N, N), dtype=bool)
    n_rock = rng.integers(15, 31) if ood_terrain else rng.integers(3, 16)
    for _ in range(n_rock):
        r = rng.uniform(2.5, 13.0)
        a = rng.uniform(-np.pi, np.pi)
        cx, cy = r * np.cos(a), r * np.sin(a)
        hr, sr = rng.uniform(0.2, 0.9), rng.uniform(0.3, 0.6)
        bump = hr * np.exp(-0.5 * (((X - cx) / sr) ** 2 + ((Y - cy) / sr) ** 2))
        h += bump
        rock |= bump > 0.12

    # truck: box parked tangentially at the truck direction
    truck = None
    if rng.uniform() < 0.9:
        r = rng.uniform(7.5, 9.5)
        cx, cy = r * np.cos(theta_truck), r * np.sin(theta_truck)
        yaw = theta_truck + np.pi / 2 + rng.normal(0, 0.15)
        z0 = float(map_coordinates(h, [[(cy + E) / res - 0.5], [(cx + E) / res - 0.5]], order=1, mode="nearest")[0])
        truck = dict(center=(float(cx), float(cy)), yaw=float(yaw), half=(3.25, 1.25, 1.30), z0=z0,
                     cab=dict(offset=3.9, half=(0.9, 1.15, 1.35)))

    # persons and obstacles (never inside the machine footprint, never on the truck)
    persons, obstacles = [], []
    for lst, n, rmin in [(persons, rng.integers(0, 4), 3.0), (obstacles, rng.integers(0, 4), 2.5)]:
        for _ in range(n):
            for _try in range(20):
                r = rng.uniform(rmin, 12.0)
                a = theta_work + rng.uniform(-np.deg2rad(150), np.deg2rad(150))
                px, py = r * np.cos(a), r * np.sin(a)
                if truck is not None and dist_to_rect(px, py, truck["center"][0], truck["center"][1], truck["yaw"], 4.5, 1.5) < 0.5:
                    continue
                lst.append((float(px), float(py)))
                break

    # lighting
    el = rng.uniform(np.deg2rad(20), np.deg2rad(70))
    az = rng.uniform(-np.pi, np.pi)
    light = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    tint = rng.uniform(0.85, 1.15, size=3)

    return Scene(seed=seed, h=h.astype(np.float32), mat=mat, rock=rock, z_target=base_level - 0.1,
                 theta_work=float(theta_work), theta_truck=float(theta_truck), truck=truck,
                 persons=persons, obstacles=obstacles, light=light, tint=tint, ood_terrain=ood_terrain)


def sample_frame(scene: Scene, frame_idx: int, ood_artic: bool = False) -> tuple:
    """Deterministic joint configuration for frame `frame_idx` of the scene.  Returns (q, phase)."""
    rng = np.random.default_rng(7_919 * (scene.seed + 1) + 31 * (frame_idx + 1))
    for _ in range(50):
        q3, phase = sample_joint_config(rng, ood_artic=ood_artic)
        # swing: mostly toward the working direction, sometimes toward the truck
        if rng.uniform() < 0.7:
            th = scene.theta_work + rng.normal(0, np.deg2rad(25))
        else:
            th = scene.theta_truck + rng.normal(0, np.deg2rad(12))
            phase = "dump"
        q = np.array([np.arctan2(np.sin(th), np.cos(th)), q3[0], q3[1], q3[2]])
        if _pose_ok(scene, q):
            return q, phase
    return q, phase


def _pose_ok(scene: Scene, q) -> bool:
    """Reject poses whose cameras are below the terrain or whose bucket tip is absurd."""
    from .kinematics import sensor_pose_U, bucket_tip_U, T_B_from_U, transform_points
    TB = T_B_from_U(q)
    tip = transform_points(TB, bucket_tip_U(q)[None])[0]
    if tip[2] > 7.0 or tip[2] < scene.height_at([tip[0]], [tip[1]])[0] - 1.0:
        return False
    for name in ("cab", "boom", "stick", "lidar"):
        p = (TB @ sensor_pose_U(name, q))[:3, 3]
        if p[2] < scene.height_at([p[0]], [p[1]])[0] + 0.35:
            return False
    return True


# --------------------------------------------------------------------------- ground truth in the U frame


def bev_ground_truth(scene: Scene, q) -> dict:
    """Ground-truth BEV maps (rows -> y, cols -> x) in the upper-structure frame for swing q[0]."""
    X, Y = bev_grid()
    R = rot_z(float(q[0]))[:3, :3]
    xb = R[0, 0] * X + R[0, 1] * Y
    yb = R[1, 0] * X + R[1, 1] * Y

    H = scene.height_at(xb.ravel(), yb.ravel()).reshape(X.shape).astype(np.float32)
    mat = scene.material_at(xb.ravel(), yb.ravel()).reshape(X.shape).astype(np.uint8)
    rock = scene.rock_at(xb.ravel(), yb.ravel()).reshape(X.shape)

    sem = np.zeros(X.shape, dtype=np.uint8)
    sem[rock] = SEM_ROCK
    haz = np.zeros(X.shape, dtype=bool)

    if scene.truck is not None:
        t = scene.truck
        cx, cy, yaw, half = t["center"][0], t["center"][1], t["yaw"], t["half"]
        d = dist_to_rect(xb, yb, cx, cy, yaw, half[0] + t["cab"]["half"][0] * 2, half[1])
        sem[d <= 0] = SEM_TRUCK
        haz |= d <= TRUCK_BUFFER
    for (px, py) in scene.persons:
        d = np.hypot(xb - px, yb - py)
        sem[d <= 0.30] = SEM_PERSON
        haz |= d <= PERSON_EXCL
    for (px, py) in scene.obstacles:
        d = np.hypot(xb - px, yb - py)
        sem[d <= 0.35] = SEM_OBSTACLE
        haz |= d <= OBSTACLE_EXCL

    # machine footprint: undercarriage is fixed in B, upper hull in U
    under = (np.abs(xb) <= 2.30) & (np.abs(yb) <= 1.50)
    upper = (X >= -3.10) & (X <= 1.90) & (np.abs(Y) <= 1.35)
    sem[under | upper] = SEM_MACHINE
    under_b = (np.abs(xb) <= 2.30 + MACHINE_BUFFER) & (np.abs(yb) <= 1.50 + MACHINE_BUFFER)
    upper_b = (X >= -3.10 - MACHINE_BUFFER) & (X <= 1.90 + MACHINE_BUFFER) & (np.abs(Y) <= 1.35 + MACHINE_BUFFER)
    haz |= under_b | upper_b

    # slope (deg) from the U-frame height map
    gy, gx = np.gradient(H, BEV_RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    # reachability: radial band, front-biased sector
    r = np.hypot(X, Y)
    ang = np.abs(np.arctan2(Y, X))
    reach = smoothstep(r, REACH_MIN - 0.5, REACH_MIN + 0.5) * (1 - smoothstep(r, REACH_MAX - 0.5, REACH_MAX + 0.5))
    reach *= 1 - smoothstep(ang, REACH_SECTOR - 0.15, REACH_SECTOR + 0.15)

    prod = np.clip((H - scene.z_target) / 1.2, 0.0, 1.0)
    slope_f = 1.0 - smoothstep(slope, 40.0, 60.0)
    dig = MAT_DIGGABILITY[mat]
    aff = dig * slope_f * reach * prod * (~haz) * (sem == SEM_TERRAIN)

    return dict(H=H, sem=sem, mat=mat, haz=haz.astype(np.uint8), aff=aff.astype(np.float32),
                reach=reach.astype(np.float32), prod=prod.astype(np.float32), slope=slope.astype(np.float32))


def compose_affordance(H, sem, mat, haz, reach, z_target):
    """Affordance composed from (predicted or true) component maps -- the 'compositional' selector."""
    gy, gx = np.gradient(H, BEV_RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    prod = np.clip((H - z_target) / 1.2, 0.0, 1.0)
    slope_f = 1.0 - smoothstep(slope, 40.0, 60.0)
    dig = MAT_DIGGABILITY[np.clip(mat, 0, 3)]
    return (dig * slope_f * reach * prod * (haz < 0.5) * (sem == SEM_TERRAIN)).astype(np.float32)
