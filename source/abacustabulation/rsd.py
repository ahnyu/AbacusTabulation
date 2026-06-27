"""Redshift-space distortion helpers for AbacusSummit catalog positions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


Axis = int | str


def axis_index(axis: Axis) -> int:
    """Normalize an axis label to an integer index."""

    if isinstance(axis, str):
        key = axis.strip().lower()
        axes = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
        if key not in axes:
            raise ValueError("axis must be one of x, y, z, 0, 1, or 2.")
        return axes[key]

    axis_int = int(axis)
    if axis_int not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    return axis_int


def wrap_positions(
    positions: np.ndarray,
    lbox: float,
    *,
    box_origin: str = "center",
) -> np.ndarray:
    """Periodically wrap positions into the simulation box.

    ``box_origin="center"`` wraps to ``[-Lbox/2, Lbox/2)``. ``"zero"`` wraps
    to ``[0, Lbox)``.
    """

    lbox = float(lbox)
    if lbox <= 0.0:
        raise ValueError("lbox must be positive.")

    positions_arr = np.asarray(positions, dtype=np.float64)
    if box_origin == "center":
        return (positions_arr + 0.5 * lbox) % lbox - 0.5 * lbox
    if box_origin == "zero":
        return positions_arr % lbox
    raise ValueError("box_origin must be 'center' or 'zero'.")


def velzspace_to_kms(header: Mapping[str, object], lbox: float | None = None) -> float:
    """Return the RSD conversion factor from an Abacus catalog header.

    The convention follows the AbacusHOD usage:
    ``velz2kms = header["VelZSpace_to_kms"] / Lbox`` and
    ``s_los = wrap(x_los + v_los / velz2kms, Lbox)``.
    """

    if lbox is None:
        lbox = float(header["BoxSizeHMpc"])
    return float(header["VelZSpace_to_kms"]) / float(lbox)


def apply_rsd(
    positions: np.ndarray,
    velocities: np.ndarray,
    lbox: float,
    *,
    header: Mapping[str, object] | None = None,
    velz2kms: float | None = None,
    los_axis: Axis = 2,
    box_origin: str = "center",
) -> np.ndarray:
    """Apply a periodic line-of-sight RSD shift to positions."""

    positions_arr = np.asarray(positions, dtype=np.float64)
    velocities_arr = np.asarray(velocities, dtype=np.float64)
    if positions_arr.shape != velocities_arr.shape or positions_arr.ndim != 2:
        raise ValueError("positions and velocities must both have shape (n, 3).")
    if positions_arr.shape[1] != 3:
        raise ValueError("positions and velocities must both have shape (n, 3).")

    if velz2kms is None:
        if header is None:
            raise ValueError("Either header or velz2kms must be supplied.")
        velz2kms = velzspace_to_kms(header, lbox=lbox)
    velz2kms = float(velz2kms)
    if velz2kms == 0.0:
        raise ValueError("velz2kms must be non-zero.")

    axis = axis_index(los_axis)
    shifted = positions_arr.copy()
    shifted[:, axis] += velocities_arr[:, axis] / velz2kms
    return wrap_positions(shifted, lbox, box_origin=box_origin)
