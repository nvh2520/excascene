"""Ray-casting renderer for ExcaScene (NumPy, multiprocessing-friendly, no GPU needed).

For every sensor ray (in the base frame B) we find the first intersection with
  * the terrain height-field (adaptive marching + bisection refinement),
  * the excavator body (undercarriage, upper hull, cab, boom, arm, bucket) -> self-occlusion,
  * the truck (boxes), persons and obstacles (vertical cylinders).
Cameras return RGB (Lambertian shading with per-material texture), z-depth and per-pixel semantics.
The LiDAR returns a range image with per-return semantics.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .kinematics import (SENSORS, CAMERAS, body_box_poses_U, camera_rays, lidar_rays, sensor_pose_U,
                         T_B_from_U, inv, rot_z)
from .scene import (Scene, MAT_ALBEDO, MAT_TEXTURE_AMP, SEM_TERRAIN, SEM_ROCK, SEM_TRUCK, SEM_PERSON,
                    SEM_OBSTACLE, SEM_MACHINE, SEM_SKY, BEV_N, BEV_RES, BEV_X0, BEV_Y0)

# marching schedule (metres along the ray)
_T_STEPS = np.concatenate([np.arange(0.15, 6.0, 0.15), np.arange(6.3, 18.0, 0.3), np.arange(18.6, 40.6, 0.6)])

OBJECT_COLORS = {
    SEM_TRUCK: np.array([0.95, 0.75, 0.15]),
    SEM_PERSON: np.array([0.95, 0.55, 0.10]),
    SEM_OBSTACLE: np.array([0.85, 0.15, 0.12]),
    SEM_MACHINE: np.array([0.90, 0.68, 0.10]),
    SEM_ROCK: np.array([0.50, 0.48, 0.45]),
}


def _ray_boxes(O, D, boxes):
    """boxes: list of (T_B_from_box (4x4), half (3,), cls).  Returns t (R,), cls (R,), normal (R,3)."""
    R = O.shape[0]
    t_best = np.full(R, np.inf)
    cls_best = np.full(R, -1, dtype=np.int16)
    n_best = np.zeros((R, 3))
    for T, half, cls in boxes:
        Rm = T[:3, :3]
        o = (O - T[:3, 3]) @ Rm  # R^T (O - c)
        d = D @ Rm
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = 1.0 / d
            t1 = (-half - o) * inv_d
            t2 = (half - o) * inv_d
        zero = np.abs(d) < 1e-9
        inside = np.abs(o) <= half
        tmin_ax = np.where(zero, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        tmax_ax = np.where(zero, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        tmin = tmin_ax.max(axis=1)
        tmax = tmax_ax.min(axis=1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax > 1e-6)
        t = np.where(tmin > 1e-6, tmin, tmax)
        better = hit & (t < t_best)
        if better.any():
            ax = np.argmax(tmin_ax, axis=1)
            sgn = -np.sign(d[np.arange(R), ax])
            n_local = np.zeros((R, 3))
            n_local[np.arange(R), ax] = sgn
            n_world = n_local @ Rm.T
            t_best[better] = t[better]
            cls_best[better] = cls
            n_best[better] = n_world[better]
    return t_best, cls_best, n_best


def _ray_cylinders(O, D, cyls):
    """cyls: list of (cx, cy, radius, z0, z1, cls)."""
    R = O.shape[0]
    t_best = np.full(R, np.inf)
    cls_best = np.full(R, -1, dtype=np.int16)
    n_best = np.zeros((R, 3))
    for cx, cy, rad, z0, z1, cls in cyls:
        dx, dy = O[:, 0] - cx, O[:, 1] - cy
        a = D[:, 0] ** 2 + D[:, 1] ** 2
        b = 2 * (dx * D[:, 0] + dy * D[:, 1])
        c = dx * dx + dy * dy - rad * rad
        disc = b * b - 4 * a * c
        ok = (disc >= 0) & (a > 1e-9)
        sq = np.sqrt(np.maximum(disc, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
        t_side = np.full(R, np.inf)
        for t in ts:
            z = O[:, 2] + t * D[:, 2]
            good = ok & (t > 1e-6) & (z >= z0) & (z <= z1) & (t < t_side)
            t_side[good] = t[good]
        # top cap
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cap = (z1 - O[:, 2]) / D[:, 2]
        px = O[:, 0] + t_cap * D[:, 0]
        py = O[:, 1] + t_cap * D[:, 1]
        cap = (D[:, 2] < -1e-9) & (t_cap > 1e-6) & ((px - cx) ** 2 + (py - cy) ** 2 <= rad * rad)
        t_all = np.where(cap & (t_cap < t_side), t_cap, t_side)
        better = np.isfinite(t_all) & (t_all < t_best)
        if better.any():
            hx = O[:, 0] + t_all * D[:, 0]
            hy = O[:, 1] + t_all * D[:, 1]
            n = np.stack([(hx - cx) / rad, (hy - cy) / rad, np.zeros(R)], axis=1)
            n[cap & (t_cap <= t_side)] = np.array([0.0, 0.0, 1.0])
            t_best[better] = t_all[better]
            cls_best[better] = cls
            n_best[better] = n[better]
    return t_best, cls_best, n_best


class Renderer:
    def __init__(self, scene: Scene):
        self.scene = scene
        rng = np.random.default_rng(scene.seed + 424242)
        tex = gaussian_filter(rng.uniform(-1, 1, size=(256, 256)), 1.2, mode="wrap")
        self.tex = (tex / (np.abs(tex).max() + 1e-9)).astype(np.float32)
        self.hmax = float(scene.h.max())
        self._cam_dirs = {c: camera_rays(SENSORS[c]) for c in CAMERAS}
        self._lidar_dirs = lidar_rays(SENSORS["lidar"])

    # ------------------------------------------------------------------ geometry per frame
    def _static_objects(self):
        boxes, cyls = [], []
        sc = self.scene
        if sc.truck is not None:
            t = sc.truck
            cx, cy = t["center"]
            yaw, half, z0 = t["yaw"], t["half"], t["z0"]
            T = rot_z(yaw)
            T[:3, 3] = [cx, cy, z0 + half[2]]
            boxes.append((T, np.array(half), SEM_TRUCK))
            cab = t["cab"]
            Tc = rot_z(yaw)
            off = np.array([cab["offset"], 0.0, 0.0])
            Tc[:3, 3] = [cx, cy, z0 + cab["half"][2]]
            Tc[:3, 3] += rot_z(yaw)[:3, :3] @ off
            boxes.append((Tc, np.array(cab["half"]), SEM_TRUCK))
        for (px, py) in sc.persons:
            z0 = float(sc.height_at([px], [py])[0])
            cyls.append((px, py, 0.30, z0 - 0.2, z0 + 1.75, SEM_PERSON))
        for (px, py) in sc.obstacles:
            z0 = float(sc.height_at([px], [py])[0])
            cyls.append((px, py, 0.35, z0 - 0.2, z0 + 1.0, SEM_OBSTACLE))
        return boxes, cyls

    def _machine_boxes(self, q):
        TB = T_B_from_U(q)
        return [(TB @ T, half, SEM_MACHINE) for T, half in body_box_poses_U(q)]

    # ------------------------------------------------------------------ core casting
    def _march_terrain(self, O, D, tmax):
        sc = self.scene
        R = O.shape[0]
        t_hit = np.full(R, np.inf)
        z0 = sc.height_at(O[:, 0], O[:, 1])
        below0 = O[:, 2] < z0
        t_hit[below0] = 1e-3
        active = ~below0
        active &= ~((D[:, 2] >= 0) & (O[:, 2] > self.hmax + 0.05))
        t_prev = np.zeros(R)
        for tc in _T_STEPS:
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            p = O[idx] + tc * D[idx]
            zt = sc.height_at(p[:, 0], p[:, 1])
            below = p[:, 2] < zt
            if below.any():
                j = idx[below]
                lo, hi = t_prev[j].copy(), np.full(j.size, tc)
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    pm = O[j] + mid[:, None] * D[j]
                    zm = sc.height_at(pm[:, 0], pm[:, 1])
                    b = pm[:, 2] < zm
                    hi = np.where(b, mid, hi)
                    lo = np.where(b, lo, mid)
                t_hit[j] = 0.5 * (lo + hi)
                active[j] = False
            t_prev[idx] = tc
            # rays that rose above the terrain envelope will never hit
            above = (p[:, 2] > self.hmax + 0.05) & (D[idx, 2] >= 0)
            active[idx[above]] = False
            done = tc >= (tmax[idx] if np.ndim(tmax) else tmax)
            active[idx[done]] = False
        return t_hit

    def cast(self, O, D, q, tmax=40.0, machine_boxes=None, static=None):
        """Returns dict(t, cls, normal, P) for rays (O, D) in the base frame."""
        if machine_boxes is None:
            machine_boxes = self._machine_boxes(q)
        if static is None:
            static = self._static_objects()
        boxes, cyls = static
        t_ter = self._march_terrain(O, D, tmax)
        t_box, c_box, n_box = _ray_boxes(O, D, machine_boxes + boxes)
        t_cyl, c_cyl, n_cyl = _ray_cylinders(O, D, cyls) if cyls else (np.full(O.shape[0], np.inf), None, None)
        t = t_ter.copy()
        cls = np.full(O.shape[0], -1, dtype=np.int16)
        cls[np.isfinite(t_ter)] = SEM_TERRAIN
        normal = np.zeros((O.shape[0], 3))
        m = t_box < t
        t[m], cls[m], normal[m] = t_box[m], c_box[m], n_box[m]
        if c_cyl is not None:
            m = t_cyl < t
            t[m], cls[m], normal[m] = t_cyl[m], c_cyl[m], n_cyl[m]
        m = t > tmax
        t[m], cls[m] = np.inf, -1
        P = O + np.where(np.isfinite(t), t, 0.0)[:, None] * D
        ter = cls == SEM_TERRAIN
        if ter.any():
            normal[ter] = self.scene.normal_at(P[ter, 0], P[ter, 1])
            rock = self.scene.rock_at(P[ter, 0], P[ter, 1])
            idx = np.nonzero(ter)[0]
            cls[idx[rock]] = SEM_ROCK
        return dict(t=t, cls=cls, normal=normal, P=P)

    # ------------------------------------------------------------------ shading
    def _texture(self, P, scale=0.12):
        cy = (P[:, 1] / scale) % 256
        cx = (P[:, 0] / scale) % 256
        return map_coordinates(self.tex, [cy, cx], order=1, mode="wrap")

    def shade(self, hit, D, rng: np.random.Generator):
        """RGB in [0,1] for every ray."""
        sc = self.scene
        R = D.shape[0]
        rgb = np.zeros((R, 3))
        cls, P, n = hit["cls"], hit["P"], hit["normal"]
        lamb = 0.35 + 0.65 * np.clip((n * sc.light).sum(axis=1), 0, 1)
        # terrain
        m = cls == SEM_TERRAIN
        if m.any():
            mat = sc.material_at(P[m, 0], P[m, 1])
            tex = self._texture(P[m])
            alb = MAT_ALBEDO[mat] * (1 + MAT_TEXTURE_AMP[mat][:, None] * tex[:, None])
            rgb[m] = alb * lamb[m][:, None]
        m = cls == SEM_ROCK
        if m.any():
            tex = self._texture(P[m], scale=0.05)
            rgb[m] = OBJECT_COLORS[SEM_ROCK] * (1 + 0.3 * tex[:, None]) * lamb[m][:, None]
        for c in (SEM_TRUCK, SEM_PERSON, SEM_OBSTACLE, SEM_MACHINE):
            m = cls == c
            if m.any():
                col = OBJECT_COLORS[c]
                if c == SEM_MACHINE:
                    tex = self._texture(P[m], scale=0.3)
                    rgb[m] = col * (1 + 0.1 * tex[:, None]) * lamb[m][:, None]
                elif c == SEM_PERSON:
                    # hi-vis torso, darker legs
                    top = P[m, 2] > (sc.height_at(P[m, 0], P[m, 1]) + 0.9)
                    rgb[m] = np.where(top[:, None], col, np.array([0.15, 0.18, 0.35])) * lamb[m][:, None]
                else:
                    rgb[m] = col * lamb[m][:, None]
        # sky
        m = cls == -1
        if m.any():
            el = np.clip(D[m, 2], 0, 1)
            rgb[m] = (1 - el[:, None]) * np.array([0.80, 0.86, 0.94]) + el[:, None] * np.array([0.40, 0.58, 0.92])
        # distance fog and tint
        t = np.where(np.isfinite(hit["t"]), hit["t"], 40.0)
        fog = np.clip(t / 60.0, 0, 0.6)
        rgb = rgb * (1 - fog[:, None]) + fog[:, None] * np.array([0.75, 0.78, 0.82])
        rgb = np.clip(rgb * sc.tint, 0, 1) ** (1 / 1.8)
        rgb += rng.normal(0, 0.012, size=rgb.shape)
        return np.clip(rgb, 0, 1)

    # ------------------------------------------------------------------ sensors
    def render_camera(self, name: str, q, rng: np.random.Generator, drift=None, machine_boxes=None, static=None):
        spec = SENSORS[name]
        T_U = sensor_pose_U(name, q, drift=drift)
        T_B = T_B_from_U(q) @ T_U
        dirs = self._cam_dirs[name].reshape(-1, 3)
        D = dirs @ T_B[:3, :3].T
        O = np.repeat(T_B[:3, 3][None], D.shape[0], axis=0)
        hit = self.cast(O, D, q, tmax=40.0, machine_boxes=machine_boxes, static=static)
        rgb = self.shade(hit, D, rng).reshape(spec["H"], spec["W"], 3)
        depth = np.where(np.isfinite(hit["t"]), hit["t"] * dirs[:, 2], 0.0).reshape(spec["H"], spec["W"])
        sem = np.where(hit["cls"] >= 0, hit["cls"], SEM_SKY).reshape(spec["H"], spec["W"]).astype(np.uint8)
        ter = (hit["cls"] == SEM_TERRAIN) | (hit["cls"] == SEM_ROCK)
        return dict(rgb=(rgb * 255).astype(np.uint8), depth=depth.astype(np.float32), sem=sem,
                    T_U_true=T_U, hits_B=hit["P"][ter])

    def render_lidar(self, q, rng: np.random.Generator, drift=None, machine_boxes=None, static=None):
        spec = SENSORS["lidar"]
        T_U = sensor_pose_U("lidar", q, drift=drift)
        T_B = T_B_from_U(q) @ T_U
        dirs = self._lidar_dirs.reshape(-1, 3)
        D = dirs @ T_B[:3, :3].T
        O = np.repeat(T_B[:3, 3][None], D.shape[0], axis=0)
        hit = self.cast(O, D, q, tmax=spec["max_range"], machine_boxes=machine_boxes, static=static)
        rng_img = np.where(np.isfinite(hit["t"]), hit["t"], 0.0).reshape(spec["rings"], spec["azimuths"])
        valid = np.isfinite(hit["t"]).reshape(spec["rings"], spec["azimuths"])
        sem = np.where(hit["cls"] >= 0, hit["cls"], SEM_SKY).reshape(spec["rings"], spec["azimuths"]).astype(np.uint8)
        ter = (hit["cls"] == SEM_TERRAIN) | (hit["cls"] == SEM_ROCK)
        return dict(range=rng_img.astype(np.float32), valid=valid, sem=sem, T_U_true=T_U, hits_B=hit["P"][ter])


def visibility_mask(hits_B_list, q) -> np.ndarray:
    """BEV (rows->y, cols->x) mask of cells that received at least one terrain return, in the U frame."""
    vis = np.zeros((BEV_N, BEV_N), dtype=np.uint8)
    if not hits_B_list:
        return vis
    P = np.concatenate([h for h in hits_B_list if len(h)], axis=0) if any(len(h) for h in hits_B_list) else None
    if P is None:
        return vis
    Rm = rot_z(-float(q[0]))[:3, :3]
    Pu = P @ Rm.T
    cx = np.floor((Pu[:, 0] - BEV_X0) / BEV_RES).astype(int)
    cy = np.floor((Pu[:, 1] - BEV_Y0) / BEV_RES).astype(int)
    ok = (cx >= 0) & (cx < BEV_N) & (cy >= 0) & (cy < BEV_N)
    vis[cy[ok], cx[ok]] = 1
    return vis
