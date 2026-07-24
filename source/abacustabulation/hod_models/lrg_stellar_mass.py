"""LRG HOD split by raw or redshift-relative stellar-mass rank.

This module is deliberately self-contained so that the property-dependent HOD
can be exercised before the joint multi-sample fitting infrastructure is
added.  It keeps the existing :mod:`.lrg` Zheng07 parent occupation unchanged
and multiplies it by central and satellite stellar-mass selection
probabilities.

``selection.logMstar_edges`` contains the complete, contiguous partition of
the selected parent sample:

* ``mode: raw`` uses one fixed one-dimensional edge array.
* ``mode: relative`` uses one edge-array row per fine redshift cell and forms
  the explicitly weighted combination of those cell-level filters.

For example, the fixed ``selection`` metadata can be written as::

    selection:
      mode: raw
      bin_index: 0
      logMstar_edges: [10.4, 11.05, 11.30, 11.55, 12.2]

or::

    selection:
      mode: relative
      bin_index: 0
      logMstar_edges:
        - [10.4, 11.00, 11.25, 11.50, 12.2]
        - [10.4, 11.08, 11.33, 11.58, 12.2]
      redshift_weights: [0.4, 0.6]

Omitting ``redshift_weights`` gives every relative redshift cell equal weight.
The relative construction is an effective single-snapshot approximation.  It
preserves one-point occupations and the exact sum-to-parent constraint, but it
is not an exact replacement for cell-by-cell lightcone pair weighting.
Equal observed quartile sizes remain integrated abundance constraints; the
model does not force each split probability to be 0.25 at every halo mass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import param
from .lrg import evaluate as evaluate_parent_lrg

try:  # pragma: no cover - the fallback is for lightweight environments.
    from scipy import special as _special
except Exception:  # pragma: no cover
    _special = None


PARAMETERS = {
    # Existing LRG Zheng07 parent parameters; names intentionally unchanged.
    "logMcut",
    "sigma",
    "a_c",
    "logM1",
    "alpha",
    "kappa",
    # Central conditional stellar-mass distribution.
    "logMstar_cen_bend",
    "logMh_cen_bend",
    "beta_cen_low",
    "beta_cen_high",
    "sigma_logMstar_cen",
    "width_cen_bend",
    # Satellite conditional stellar-mass distribution.
    "alpha_star_sat",
    "delta_cs_pivot",
    "delta_cs_gradient",
    "logMh_delta_pivot",
    # Fixed raw/relative sample definition.
    "selection",
}

_LN10 = math.log(10.0)
_SQRT2 = math.sqrt(2.0)
_SATELLITE_CUTOFF_POWER = 2.0
_QUADRATURE_ORDER = 48
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
_LOG_GL_WEIGHTS = np.log(_GL_WEIGHTS)


def central_mean_logmstar(
    mass: np.ndarray | float,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Mean central ``log10(Mstar/Msun)`` from a smooth broken power law."""

    mass = _validated_mass(mass)
    return _central_mean_logmstar_from_logmass(np.log10(mass), params)


def _central_mean_logmstar_from_logmass(
    log_mass: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    logmh_bend = _finite_param(params, "logMh_cen_bend")
    logmstar_bend = _finite_param(params, "logMstar_cen_bend")
    beta_low = _finite_param(params, "beta_cen_low")
    beta_high = _finite_param(params, "beta_cen_high")
    width = _positive_param(params, "width_cen_bend", default=1.0)

    distance = log_mass - logmh_bend
    scaled_distance = _LN10 * distance / width
    soft_transition = width / _LN10 * (
        np.logaddexp(0.0, scaled_distance) - math.log(2.0)
    )
    return (
        logmstar_bend
        + beta_low * distance
        + (beta_high - beta_low) * soft_transition
    )


def satellite_characteristic_logmstar(
    mass: np.ndarray | float,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Characteristic satellite log stellar mass after the mass-dependent gap."""

    mass = _validated_mass(mass)
    log_mass = np.log10(mass)
    central_mean = _central_mean_logmstar_from_logmass(log_mass, params)
    return _satellite_characteristic_from_central(log_mass, central_mean, params)


def _satellite_characteristic_from_central(
    log_mass: np.ndarray,
    central_mean: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    delta_pivot = _finite_param(params, "delta_cs_pivot")
    delta_gradient = _finite_param(params, "delta_cs_gradient")
    logmh_pivot = _finite_param(params, "logMh_delta_pivot", default=14.0)
    gap = delta_pivot + delta_gradient * (log_mass - logmh_pivot)
    return central_mean - gap


def selection_probabilities(
    mass: np.ndarray | float,
    params: Mapping[str, Any],
    *,
    split_method: str | None = None,
    logmstar_edges: Any = None,
    redshift_weights: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all central and satellite split probabilities.

    The output shape is ``(n_splits, *mass.shape)``.  For a relative split, the
    probability in each redshift cell is normalized over that row's full
    parent support before the rows are combined with normalized
    ``redshift_weights``.  These weights should be fixed from the data using
    the same redshift-cell convention as the relative split, not fitted as HOD
    parameters.
    """

    mass = _validated_mass(mass)
    edges, redshift_weights = _selection_grid(
        params,
        split_method=split_method,
        logmstar_edges=logmstar_edges,
        redshift_weights=redshift_weights,
    )
    mass_shape = mass.shape
    log_mass = np.log10(mass)

    central_mean = _central_mean_logmstar_from_logmass(log_mass, params)
    central_sigma = _positive_param(params, "sigma_logMstar_cen")
    central_cells = _central_cell_probabilities(
        edges,
        central_mean.reshape(-1),
        central_sigma,
    )

    satellite_location = _satellite_characteristic_from_central(
        log_mass,
        central_mean,
        params,
    ).reshape(-1)
    alpha_star_sat = _finite_param(params, "alpha_star_sat")
    satellite_cells = _satellite_cell_probabilities(
        edges,
        satellite_location,
        alpha_star_sat,
    )

    central = _mix_cell_probabilities(central_cells, redshift_weights, mass_shape)
    satellite = _mix_cell_probabilities(satellite_cells, redshift_weights, mass_shape)
    return central, satellite


def evaluate_all_splits(
    mass: np.ndarray | float,
    params: Mapping[str, Any],
    *,
    split_method: str | None = None,
    logmstar_edges: Any = None,
    redshift_weights: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return occupations for every stellar-mass split.

    Summing either returned array over axis zero recovers the corresponding
    unchanged parent LRG occupation pointwise.
    """

    mass = _validated_mass(mass)
    parent_central, parent_satellite = evaluate_parent_lrg(mass, params)
    p_central, p_satellite = selection_probabilities(
        mass,
        params,
        split_method=split_method,
        logmstar_edges=logmstar_edges,
        redshift_weights=redshift_weights,
    )
    return (
        p_central * np.asarray(parent_central)[None, ...],
        p_satellite * np.asarray(parent_satellite)[None, ...],
    )


def evaluate(
    mass: np.ndarray,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the occupation of ``selection.bin_index`` for registry use."""

    central, satellite = evaluate_all_splits(mass, params)
    selection = _selection_mapping(params)
    bin_index = _bin_index(selection, central.shape[0])
    return central[bin_index], satellite[bin_index]


def _selection_mapping(params: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = param(params, "selection")
    if not isinstance(selection, Mapping):
        raise TypeError("selection must be a mapping.")
    return selection


def _selection_grid(
    params: Mapping[str, Any],
    *,
    split_method: str | None = None,
    logmstar_edges: Any = None,
    redshift_weights: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    if split_method is None:
        if logmstar_edges is not None or redshift_weights is not None:
            raise ValueError(
                "Set split_method when passing logmstar_edges or redshift_weights."
            )
        selection = _selection_mapping(params)
    else:
        if logmstar_edges is None:
            raise KeyError("logmstar_edges is required for an explicit split.")
        selection = {
            "mode": split_method,
            "logMstar_edges": logmstar_edges,
        }
        if redshift_weights is not None:
            selection["redshift_weights"] = redshift_weights
    return _selection_grid_from_mapping(selection)


def normalize_stellar_mass_selection(
    split_method: str,
    logmstar_edges: Any,
    redshift_weights: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a split definition and return 2D edges and normalized weights."""

    selection = {
        "mode": split_method,
        "logMstar_edges": logmstar_edges,
    }
    if redshift_weights is not None:
        selection["redshift_weights"] = redshift_weights
    return _selection_grid_from_mapping(selection)


def _selection_grid_from_mapping(
    selection: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(selection.get("mode", "")).lower()
    if mode not in {"raw", "relative"}:
        raise ValueError("split_method must be either 'raw' or 'relative'.")

    try:
        edges = np.asarray(selection["logMstar_edges"], dtype=np.float64)
    except KeyError as exc:
        raise KeyError("logmstar_edges is required.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("logmstar_edges must be a rectangular numeric array.") from exc

    if mode == "raw":
        if edges.ndim != 1:
            raise ValueError("Raw logmstar_edges must be one-dimensional.")
        edges = edges[None, :]
        if "redshift_weights" in selection:
            raise ValueError("A raw split must not define redshift_weights.")
        weights = np.ones(1, dtype=np.float64)
    else:
        if edges.ndim != 2:
            raise ValueError(
                "Relative logmstar_edges must have shape "
                "(n_redshift_cells, n_splits + 1)."
            )
        if edges.shape[0] == 0:
            raise ValueError(
                "Relative logmstar_edges must contain at least one "
                "redshift-cell row."
            )
        weights = np.asarray(
            selection.get("redshift_weights", np.ones(edges.shape[0])),
            dtype=np.float64,
        )

    if edges.shape[1] < 2:
        raise ValueError("logmstar_edges must define at least one split.")
    if not np.all(np.isfinite(edges)):
        raise ValueError(
            "logmstar_edges must be finite; the outer edges define "
            "the selected-parent support."
        )
    if not np.all(np.diff(edges, axis=1) > 0.0):
        raise ValueError(
            "Each row of logmstar_edges must be strictly increasing."
        )

    if weights.ndim != 1 or weights.size != edges.shape[0]:
        raise ValueError(
            "redshift_weights must be one-dimensional and match "
            "the number of relative edge rows."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("redshift_weights must be finite and non-negative.")
    weight_scale = float(np.max(weights))
    if weight_scale <= 0.0:
        raise ValueError("redshift_weights must have positive total weight.")
    scaled_weights = weights / weight_scale
    return edges, scaled_weights / np.sum(scaled_weights)


def _bin_index(selection: Mapping[str, Any], n_splits: int) -> int:
    if "bin_index" not in selection:
        raise KeyError("selection.bin_index is required by evaluate().")
    value = selection["bin_index"]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("selection.bin_index must be a zero-based integer.")
    index = int(value)
    if not 0 <= index < n_splits:
        raise ValueError(
            f"selection.bin_index={index} is outside [0, {n_splits - 1}]."
        )
    return index


def _central_cell_probabilities(
    edges: np.ndarray,
    mean: np.ndarray,
    sigma: float,
) -> np.ndarray:
    lower = (edges[:, :-1, None] - mean[None, None, :]) / sigma
    upper = (edges[:, 1:, None] - mean[None, None, :]) / sigma
    log_integrals = _normal_log_interval(lower, upper)
    return _normalize_log_intervals(log_integrals)


def _satellite_cell_probabilities(
    edges: np.ndarray,
    characteristic: np.ndarray,
    alpha_star_sat: float,
) -> np.ndarray:
    lower = edges[:, :-1]
    upper = edges[:, 1:]
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)

    sample_logmstar = (
        midpoint[:, :, None, None]
        + half_width[:, :, None, None] * _GL_NODES[None, None, None, :]
    )
    offset = sample_logmstar - characteristic[None, None, :, None]
    log_cutoff_argument = _LN10 * _SATELLITE_CUTOFF_POWER * offset
    cutoff_argument = np.exp(np.minimum(log_cutoff_argument, 700.0))
    log_kernel = _LN10 * (alpha_star_sat + 1.0) * offset - cutoff_argument

    log_integrals = (
        np.log(half_width)[:, :, None]
        + _logsumexp(
            log_kernel + _LOG_GL_WEIGHTS[None, None, None, :],
            axis=-1,
        )
    )
    return _normalize_log_intervals(log_integrals)


def _normal_log_interval(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if _special is None:  # pragma: no cover - scipy is present in production.
        return _normal_log_interval_fallback(lower, upper)

    out = np.empty(np.broadcast_shapes(lower.shape, upper.shape), dtype=np.float64)
    lower, upper = np.broadcast_arrays(lower, upper)
    below_zero = upper <= 0.0
    above_zero = lower >= 0.0
    crosses_zero = ~(below_zero | above_zero)

    out[below_zero] = _log_subtract(
        _special.log_ndtr(upper[below_zero]),
        _special.log_ndtr(lower[below_zero]),
    )
    out[above_zero] = _log_subtract(
        _special.log_ndtr(-lower[above_zero]),
        _special.log_ndtr(-upper[above_zero]),
    )
    if np.any(crosses_zero):
        probability = (
            _special.ndtr(upper[crosses_zero])
            - _special.ndtr(lower[crosses_zero])
        )
        out[crosses_zero] = np.log(probability)
    return out


def _normal_log_interval_fallback(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[float])
    erfc = np.vectorize(math.erfc, otypes=[float])
    lower, upper = np.broadcast_arrays(lower, upper)
    probability = np.empty(lower.shape, dtype=np.float64)
    below_zero = upper <= 0.0
    above_zero = lower >= 0.0
    crosses_zero = ~(below_zero | above_zero)
    probability[below_zero] = 0.5 * (
        erfc(-upper[below_zero] / _SQRT2)
        - erfc(-lower[below_zero] / _SQRT2)
    )
    probability[above_zero] = 0.5 * (
        erfc(lower[above_zero] / _SQRT2)
        - erfc(upper[above_zero] / _SQRT2)
    )
    probability[crosses_zero] = 0.5 * (
        erf(upper[crosses_zero] / _SQRT2)
        - erf(lower[crosses_zero] / _SQRT2)
    )
    with np.errstate(divide="ignore"):
        return np.log(probability)


def _log_subtract(log_larger: np.ndarray, log_smaller: np.ndarray) -> np.ndarray:
    difference = np.minimum(log_smaller - log_larger, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return log_larger + np.log(-np.expm1(difference))


def _normalize_log_intervals(log_integrals: np.ndarray) -> np.ndarray:
    maximum = np.max(log_integrals, axis=1, keepdims=True)
    if np.any(~np.isfinite(maximum)):
        raise FloatingPointError(
            "All conditional stellar-mass integrals vanished for at least one "
            "halo mass and redshift cell."
        )
    scaled = np.exp(log_integrals - maximum)
    total = np.sum(scaled, axis=1, keepdims=True)
    return scaled / total


def _mix_cell_probabilities(
    cell_probabilities: np.ndarray,
    redshift_weights: np.ndarray,
    mass_shape: tuple[int, ...],
) -> np.ndarray:
    mixed = np.tensordot(redshift_weights, cell_probabilities, axes=(0, 0))
    normalization = np.sum(mixed, axis=0, keepdims=True)
    tolerance = 32.0 * np.finfo(np.float64).eps
    if (
        np.any(~np.isfinite(mixed))
        or np.any(mixed < -tolerance)
        or np.any(mixed > 1.0 + tolerance)
        or not np.allclose(normalization, 1.0, rtol=0.0, atol=tolerance)
    ):
        raise FloatingPointError(
            "The weighted conditional stellar-mass probabilities did not "
            "form a valid partition."
        )
    mixed /= normalization
    return mixed.reshape((mixed.shape[0], *mass_shape))


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def _validated_mass(mass: np.ndarray | float) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("Halo mass must be finite and strictly positive.")
    return mass


def _finite_param(
    params: Mapping[str, Any],
    name: str,
    *,
    default: Any = None,
) -> float:
    value = param(params, name) if default is None else param(params, name, default=default)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"HOD parameter {name!r} must be finite.")
    return value


def _positive_param(
    params: Mapping[str, Any],
    name: str,
    *,
    default: Any = None,
) -> float:
    value = _finite_param(params, name, default=default)
    if value <= 0.0:
        raise ValueError(f"HOD parameter {name!r} must be positive.")
    return value
