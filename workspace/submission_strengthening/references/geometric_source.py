"""Registered median-height anchor; no learned or ground-truth prediction inputs."""
import numpy as np
from scipy.spatial import cKDTree
from kinafford.sim.scene import BEV_N, BEV_RES, BEV_X0, BEV_Y0, smoothstep


def transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def unproject_camera(depth, K, T, present):
    if present <= 0.5:
        return np.empty((0, 3), np.float32)
    v, u = np.indices(depth.shape, dtype=np.float32)
    points = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * depth,
                       (v + 0.5 - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return transform(points[depth > 0.05], T)


def fill_empty(height, occupied):
    if not occupied.any():
        return np.zeros_like(height)
    if occupied.all():
        return height.copy()
    coords = np.column_stack(np.nonzero(occupied))  # row-major sorted
    missing = np.column_stack(np.nonzero(~occupied))
    tree = cKDTree(coords)
    distance, nearest = tree.query(missing, k=1)
    # Explicit exact integer-distance tie resolution, independent of tree traversal.
    candidates = tree.query_ball_point(missing, distance + 1e-9)
    chosen = []
    for point, ids in zip(missing, candidates):
        ids = np.asarray(ids, dtype=int)
        ds = ((coords[ids] - point) ** 2).sum(1)
        chosen.append(ids[ds == ds.min()].min())
    out = height.copy()
    picked = coords[np.asarray(chosen)]
    out[tuple(missing.T)] = height[tuple(picked.T)]
    return out


def median_grid(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    ix = np.floor((points[:, 0] - BEV_X0) / BEV_RES).astype(np.int64)
    iy = np.floor((points[:, 1] - BEV_Y0) / BEV_RES).astype(np.int64)
    valid = (ix >= 0) & (ix < BEV_N) & (iy >= 0) & (iy < BEV_N) & (points[:, 2] < 8)
    ids, z = (iy[valid] * BEV_N + ix[valid]), points[valid, 2]
    height = np.zeros(BEV_N * BEV_N, dtype=np.float32)
    occupied = np.zeros_like(height, dtype=bool)
    # Sort by cell then z; median indices are exact, including even-count arithmetic.
    order = np.lexsort((z, ids))
    cells, first, count = np.unique(ids[order], return_index=True, return_counts=True)
    zs = z[order]
    height[cells] = (zs[first + (count - 1) // 2] + zs[first + count // 2]) * 0.5
    occupied[cells] = True
    height, occupied = height.reshape(BEV_N, BEV_N), occupied.reshape(BEV_N, BEV_N)
    return fill_empty(height, occupied), occupied


def predict_geometry(sample, lidar_dirs):
    points = [unproject_camera(sample['depth'][i], sample['K'][i], sample['T_dyn'][i], sample['cam_present'][i])
              for i in range(3)]
    rng, valid = sample['lidar_range'], sample['lidar_valid']
    points.append(transform((rng[..., None] * lidar_dirs)[(rng > 0.3) & (valid > 0.5)], sample['T_dyn'][3]))
    return median_grid(np.concatenate(points, axis=0))


def compose_geometry(height, reach, z_target):
    gy, gx = np.gradient(height, BEV_RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return (0.70 * (1 - smoothstep(slope, 40, 60)) * reach * np.clip((height-z_target)/1.2, 0, 1)).astype(np.float32)
