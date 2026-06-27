"""NFW profile sampling utilities."""

from __future__ import annotations

import numpy as np


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


def nfw_amp(x: np.ndarray | float) -> np.ndarray | float:
    """Return log(1 + x) - x / (1 + x), the unnormalized NFW mass profile."""

    return np.log1p(x) - np.asarray(x) / (1.0 + np.asarray(x))


def nfw_enclosed_fraction(
    radius: np.ndarray | float,
    rvir: np.ndarray | float,
    concentration: np.ndarray | float,
) -> np.ndarray:
    """NFW enclosed number fraction within ``radius``.

    Parameters are broadcast together. Radii outside ``[0, rvir]`` are clipped.
    """

    radius_arr, rvir_arr, concentration_arr = np.broadcast_arrays(
        np.asarray(radius, dtype=np.float64),
        np.asarray(rvir, dtype=np.float64),
        np.asarray(concentration, dtype=np.float64),
    )
    concentration_arr = np.maximum(concentration_arr, 1.0e-10)
    rvir_arr = np.maximum(rvir_arr, 0.0)

    out = np.zeros_like(radius_arr, dtype=np.float64)
    valid = rvir_arr > 0.0
    if not np.any(valid):
        return out

    clipped_radius = np.clip(radius_arr[valid], 0.0, rvir_arr[valid])
    x = clipped_radius * concentration_arr[valid] / rvir_arr[valid]
    denom = nfw_amp(concentration_arr[valid])
    denom = np.maximum(denom, 1.0e-18)
    out[valid] = nfw_amp(x) / denom
    return out


def sample_unit_vectors(
    size: int,
    rng: np.random.Generator | None = None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Sample isotropic unit vectors on a sphere."""

    rng = _as_rng(rng)
    size = int(size)
    z = rng.uniform(-1.0, 1.0, size=size)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=size)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))

    directions = np.empty((size, 3), dtype=dtype)
    directions[:, 0] = r_xy * np.cos(phi)
    directions[:, 1] = r_xy * np.sin(phi)
    directions[:, 2] = z
    return directions


def sample_nfw_radii(
    u: np.ndarray | float,
    rvir: np.ndarray | float,
    concentration: np.ndarray | float,
    *,
    tolerance: float = 1.0e-5,
    max_iterations: int = 80,
) -> np.ndarray:
    """Invert the NFW enclosed-mass CDF for uniform samples ``u``.

    ``rvir`` and ``concentration`` may be scalar or per-particle arrays.
    """

    u_arr, rvir_arr, concentration_arr = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64),
        np.asarray(rvir, dtype=np.float64),
        np.asarray(concentration, dtype=np.float64),
    )

    original_shape = u_arr.shape
    u_flat = np.clip(u_arr.reshape(-1), 0.0, 1.0)
    rvir_flat = np.maximum(rvir_arr.reshape(-1), 0.0)
    concentration_flat = np.maximum(concentration_arr.reshape(-1), 1.0e-10)

    radii = np.zeros_like(u_flat, dtype=np.float64)
    valid = rvir_flat > 0.0
    if not np.any(valid):
        return radii.reshape(original_shape)

    c = concentration_flat[valid]
    rvir_valid = rvir_flat[valid]
    rs = rvir_valid / c
    target = u_flat[valid] * np.maximum(nfw_amp(c), 1.0e-18)

    low = np.zeros_like(c)
    high = c.copy()
    tol_x = np.maximum(float(tolerance) / np.maximum(rs, 1.0e-30), 1.0e-14)

    for _ in range(int(max_iterations)):
        mid = 0.5 * (low + high)
        move_low = nfw_amp(mid) < target
        low = np.where(move_low, mid, low)
        high = np.where(move_low, high, mid)
        if np.all((high - low) <= tol_x):
            break

    radii_valid = 0.5 * (low + high) * rs
    radii_valid = np.where(u_flat[valid] <= 0.0, 0.0, radii_valid)
    radii_valid = np.where(u_flat[valid] >= 1.0, rvir_valid, radii_valid)
    radii[valid] = radii_valid
    return radii.reshape(original_shape)


def sample_nfw_offsets(
    rvir: np.ndarray | float,
    concentration: np.ndarray | float,
    *,
    rng: np.random.Generator | None = None,
    u: np.ndarray | None = None,
    directions: np.ndarray | None = None,
    tolerance: float = 1.0e-5,
    dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample 3D NFW offsets.

    Returns ``(offsets, radii, unit_vectors)``. ``rvir`` and ``concentration``
    are broadcast to the particle sample shape.
    """

    rng = _as_rng(rng)
    rvir_arr, concentration_arr = np.broadcast_arrays(
        np.asarray(rvir, dtype=np.float64),
        np.asarray(concentration, dtype=np.float64),
    )
    sample_shape = rvir_arr.shape
    size = int(rvir_arr.size)

    if u is None:
        u_arr = rng.random(size)
    else:
        u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_arr.size != size:
            raise ValueError("u must have one sample for each NFW particle.")

    if directions is None:
        directions_arr = sample_unit_vectors(size, rng=rng, dtype=np.float64)
    else:
        directions_arr = np.asarray(directions, dtype=np.float64)
        if directions_arr.shape != (size, 3):
            raise ValueError("directions must have shape (n_particles, 3).")

    radii = sample_nfw_radii(
        u_arr,
        rvir_arr.reshape(-1),
        concentration_arr.reshape(-1),
        tolerance=tolerance,
    )
    offsets = directions_arr * radii.reshape(-1, 1)
    return (
        offsets.astype(dtype, copy=False),
        radii.reshape(sample_shape).astype(dtype, copy=False),
        directions_arr.astype(dtype, copy=False),
    )
