"""Excavator kinematics, sensor mounts and body geometry (NumPy only).

Frames
------
B  : base / undercarriage frame. Origin on the ground under the swing axis, z up. Terrain lives here.
U  : upper-structure frame. Same origin, rotated about z by the swing angle q[0].  ALL BEV maps are in U.
boom / arm / bucket : link frames; x-axis along the link, origin at the proximal pivot.
Camera frames follow the OpenCV convention (x right, y down, z forward).
LiDAR frame: x forward, y left, z up.

q = [swing, boom, arm, bucket] in radians.  boom > 0 raises the boom; arm/bucket are relative angles.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- basic transforms


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def trans(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(v, dtype=np.float64)
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """P: (N,3) -> (N,3)."""
    return P @ T[:3, :3].T + T[:3, 3]


def axis_angle(axis, ang: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


# --------------------------------------------------------------------------- machine description

EXCAVATOR = dict(
    boom_pivot=(0.55, 0.0, 1.95),
    L_boom=5.70,
    L_arm=2.90,
    L_bucket=1.55,
    # nominal configuration used by the STATIC-extrinsic baseline for articulated sensors
    q_nom=np.array([0.0, 0.25, -1.55, 0.0]),
    q_min=np.array([-np.pi, -0.45, -2.70, -1.60]),
    q_max=np.array([np.pi, 1.00, -0.60, 0.80]),
)

# phase-mixture used to sample joint configurations (name, prob, [lo, hi] per boom/arm/bucket)
PHASES = [
    ("approach", 0.30, [(0.20, 0.70), (-1.60, -0.90), (-0.50, 0.30)]),
    ("penetrate", 0.35, [(-0.40, 0.20), (-2.20, -1.40), (-0.80, 0.60)]),
    ("lift", 0.20, [(0.50, 1.00), (-2.60, -1.90), (0.20, 0.80)]),
    ("uniform", 0.15, None),
]

SENSOR_ORDER = ["cab", "boom", "stick", "lidar"]
CAMERAS = ["cab", "boom", "stick"]
IMG_W, IMG_H = 256, 160

SENSORS = {
    # cab-roof camera, fixed on the upper structure
    "cab": dict(kind="camera", parent="upper", pos=(1.45, 1.05, 3.30), yaw=0.0, pitch=np.deg2rad(-18.0),
                hfov=np.deg2rad(95.0), W=IMG_W, H=IMG_H),
    # boom-mounted camera (side of the boom, looking down along the boom toward the bucket)
    "boom": dict(kind="camera", parent="boom", pos=(2.60, 0.45, -0.25), yaw=0.0, pitch=np.deg2rad(-62.0),
                 hfov=np.deg2rad(100.0), W=IMG_W, H=IMG_H),
    # stick/arm-mounted "bucket camera" looking along the arm at the bucket and the ground around it
    "stick": dict(kind="camera", parent="arm", pos=(0.90, 0.35, 0.15), yaw=0.0, pitch=np.deg2rad(-10.0),
                  hfov=np.deg2rad(100.0), W=IMG_W, H=IMG_H),
    # roof LiDAR, fixed on the upper structure
    "lidar": dict(kind="lidar", parent="upper", pos=(-0.40, 0.0, 3.35), yaw=0.0, pitch=0.0,
                  rings=32, azimuths=512, elev=(np.deg2rad(-28.0), np.deg2rad(8.0)), max_range=40.0),
}

# machine body boxes for self-occlusion / self-hits: (frame, center, half extents)
BODY_BOXES = [
    ("base", (0.0, 0.0, 0.55), (2.30, 1.50, 0.55)),          # undercarriage / tracks
    ("upper", (-0.60, 0.0, 1.55), (2.50, 1.35, 0.60)),        # upper structure hull
    ("upper", (0.70, 1.05, 2.45), (0.70, 0.55, 0.80)),        # cab
    ("boom", (EXCAVATOR["L_boom"] / 2, 0.0, 0.0), (EXCAVATOR["L_boom"] / 2, 0.30, 0.55)),
    ("arm", (EXCAVATOR["L_arm"] / 2, 0.0, 0.0), (EXCAVATOR["L_arm"] / 2, 0.22, 0.35)),
    ("bucket", (EXCAVATOR["L_bucket"] / 2, 0.0, 0.0), (EXCAVATOR["L_bucket"] / 2, 0.65, 0.42)),
]

# --------------------------------------------------------------------------- kinematics


def link_transforms(q, params=EXCAVATOR) -> dict:
    """Returns T_U_from_<link> for 'base', 'upper', 'boom', 'arm', 'bucket'."""
    q = np.asarray(q, dtype=np.float64)
    T = {}
    T["upper"] = np.eye(4)
    T["base"] = rot_z(-q[0])  # points fixed in B expressed in U
    T["boom"] = trans(params["boom_pivot"]) @ rot_y(-q[1])
    T["arm"] = T["boom"] @ trans((params["L_boom"], 0, 0)) @ rot_y(-q[2])
    T["bucket"] = T["arm"] @ trans((params["L_arm"], 0, 0)) @ rot_y(-q[3])
    return T


def T_B_from_U(q) -> np.ndarray:
    return rot_z(float(q[0]))


def bucket_tip_U(q, params=EXCAVATOR) -> np.ndarray:
    T = link_transforms(q, params)
    return transform_points(T["bucket"], np.array([[params["L_bucket"], 0.0, 0.0]]))[0]


def mount_transform(pos, yaw: float, pitch: float, kind: str = "camera") -> np.ndarray:
    """T_parent_from_sensor.  Camera: OpenCV axes; LiDAR: x forward, y left, z up."""
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    if kind == "camera":
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        r /= np.linalg.norm(r) + 1e-12
        d = np.cross(f, r)
        T[:3, :3] = np.stack([r, d, f], axis=1)
    else:
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.cross(f, left)
        T[:3, :3] = np.stack([f, left, up], axis=1)
    return T


def sensor_pose_U(name: str, q, params=EXCAVATOR, drift: np.ndarray | None = None) -> np.ndarray:
    """T_U_from_sensor for the given joint configuration.  `drift` (4x4, sensor frame) models a
    calibration error of the physical mount: T_true = T_nominal @ drift."""
    spec = SENSORS[name]
    T_link = link_transforms(q, params)[spec["parent"]]
    T = T_link @ mount_transform(spec["pos"], spec["yaw"], spec["pitch"], spec["kind"])
    if drift is not None:
        T = T @ drift
    return T


def camera_intrinsics(spec) -> np.ndarray:
    W, H = spec["W"], spec["H"]
    fx = (W / 2.0) / np.tan(spec["hfov"] / 2.0)
    K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]])
    return K


def camera_rays(spec) -> np.ndarray:
    """Unit ray directions (H, W, 3) in the camera frame through pixel centres."""
    K = camera_intrinsics(spec)
    W, H = spec["W"], spec["H"]
    u = np.arange(W) + 0.5
    v = np.arange(H) + 0.5
    uu, vv = np.meshgrid(u, v)
    d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def lidar_rays(spec) -> np.ndarray:
    """Unit ray directions (rings, azimuths, 3) in the LiDAR frame."""
    el = np.linspace(spec["elev"][0], spec["elev"][1], spec["rings"])
    az = np.linspace(-np.pi, np.pi, spec["azimuths"], endpoint=False)
    ee, aa = np.meshgrid(el, az, indexing="ij")
    d = np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], axis=-1)
    return d


def body_box_poses_U(q, params=EXCAVATOR) -> list:
    """List of (T_U_from_box, half_extents) for the machine body boxes."""
    T = link_transforms(q, params)
    out = []
    for frame, c, h in BODY_BOXES:
        out.append((T[frame] @ trans(c), np.asarray(h, dtype=np.float64)))
    return out


def sample_joint_config(rng: np.random.Generator, params=EXCAVATOR, ood_artic: bool = False) -> tuple:
    """Sample [boom, arm, bucket] from the phase mixture (swing is sampled by the scene).
    Returns (q3, phase_name)."""
    lo, hi = params["q_min"][1:], params["q_max"][1:]
    if ood_artic:
        # widened, uniform ranges -> unseen articulation configurations
        lo2 = lo - np.array([0.10, 0.10, 0.10])
        hi2 = hi + np.array([0.15, 0.10, 0.10])
        return rng.uniform(lo2, hi2), "ood_uniform"
    probs = np.array([p for _, p, _ in PHASES])
    k = rng.choice(len(PHASES), p=probs / probs.sum())
    name, _, ranges = PHASES[k]
    if ranges is None:
        return rng.uniform(lo, hi), name
    q = np.array([rng.uniform(a, b) for a, b in ranges])
    q = q + rng.normal(0, 0.05, size=3)
    return np.clip(q, lo, hi), name
