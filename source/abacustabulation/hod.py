"""Vectorized HOD occupation models used by tabulated pair-count weighting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised on cluster envs with scipy installed.
    from scipy import special as _special

    _erf = _special.erf
    _erfc = _special.erfc
except Exception:  # pragma: no cover - small fallback for lightweight envs.
    _erf = np.vectorize(math.erf, otypes=[float])
    _erfc = np.vectorize(math.erfc, otypes=[float])


def _get(params: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in params:
            return params[name]
    if default is not None:
        return default
    raise KeyError(f"Missing HOD parameter. Tried keys: {', '.join(names)}")


def _mass_from_log_or_value(
    params: Mapping[str, Any],
    *,
    log_names: tuple[str, ...],
    value_names: tuple[str, ...],
) -> float:
    for name in value_names:
        if name in params:
            return float(params[name])
    log_value = _get(params, *log_names)
    return 10.0 ** float(log_value)


def n_cen_lrg(
    mass: np.ndarray | float,
    logm_cut: float,
    sigma: float,
    pmax: float = 1.0,
) -> np.ndarray:
    """Zheng-style central occupation, matching ``resource/hodmodel.py``."""

    log_mass = np.log10(np.asarray(mass, dtype=np.float64))
    return float(pmax) * 0.5 * _erfc((float(logm_cut) - log_mass) / (math.sqrt(2.0) * float(sigma)))


def n_cen_qso(
    mass: np.ndarray | float,
    logm_cut: float,
    sigma: float,
    pmax: float = 1.0,
) -> np.ndarray:
    """QSO central occupation from the resource example, with optional ``pmax``."""

    log_mass = np.log10(np.asarray(mass, dtype=np.float64))
    return float(pmax) * 0.5 * (1.0 + _erf((log_mass - float(logm_cut)) / (math.sqrt(2.0) * float(sigma))))


def gaussian_fun(log_mass: np.ndarray | float, logm_cut: float, sigma: float) -> np.ndarray:
    log_mass = np.asarray(log_mass, dtype=np.float64)
    return np.exp(-0.5 * ((log_mass - float(logm_cut)) / float(sigma)) ** 2) / (
        math.sqrt(2.0 * math.pi) * float(sigma)
    )


def phi_fun(log_mass: np.ndarray | float, logm_cut: float, sigma: float) -> np.ndarray:
    return gaussian_fun(log_mass, logm_cut, sigma)


def phi_big_fun(
    log_mass: np.ndarray | float,
    logm_cut: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    log_mass = np.asarray(log_mass, dtype=np.float64)
    return 0.5 * (1.0 + _erf(float(gamma) * (log_mass - float(logm_cut)) / (math.sqrt(2.0) * float(sigma))))


def n_cen_elg_v1(
    mass: np.ndarray | float,
    pmax: float,
    q: float,
    logm_cut: float,
    sigma: float,
    gamma: float,
    anorm: float = 1.0,
) -> np.ndarray:
    """ELG HMQ-style central occupation from the resource example."""

    log_mass = np.log10(np.asarray(mass, dtype=np.float64))
    return (
        2.0
        * (float(pmax) - 1.0 / float(q))
        * phi_fun(log_mass, logm_cut, sigma)
        * phi_big_fun(log_mass, logm_cut, sigma, gamma)
        / float(anorm)
    )


def n_cen_elg_v2(
    mass: np.ndarray | float,
    pmax: float,
    logm_cut: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """ELG central occupation from arXiv:2007.09012 as in the resource example."""

    mass = np.asarray(mass, dtype=np.float64)
    log_mass = np.log10(mass)
    out = np.empty_like(log_mass, dtype=np.float64)
    low = log_mass <= float(logm_cut)
    out[low] = float(pmax) * gaussian_fun(log_mass[low], logm_cut, sigma)
    out[~low] = float(pmax) * (mass[~low] / 10.0 ** float(logm_cut)) ** float(gamma) / (
        math.sqrt(2.0 * math.pi) * float(sigma)
    )
    return out


def n_sat_generic(
    mass: np.ndarray | float,
    m_cut: float,
    kappa: float,
    m1: float,
    alpha: float,
    a_s: float = 1.0,
) -> np.ndarray:
    """Generic Zheng-style satellite occupation from the resource example."""

    mass = np.asarray(mass, dtype=np.float64)
    out = np.zeros_like(mass, dtype=np.float64)
    positive = mass > float(kappa) * float(m_cut)
    out[positive] = float(a_s) * ((mass[positive] - float(kappa) * float(m_cut)) / float(m1)) ** float(alpha)
    return out


def n_sat_lrg_modified(
    mass: np.ndarray | float,
    logm_cut: float,
    m_cut: float,
    m1: float,
    sigma: float,
    alpha: float,
    kappa: float,
    pmax: float = 1.0,
) -> np.ndarray:
    """LRG satellite occupation: power law multiplied by central occupation."""

    return n_sat_generic(mass, m_cut, kappa, m1, alpha) * n_cen_lrg(
        mass,
        logm_cut,
        sigma,
        pmax=pmax,
    )


def n_sat_qso(
    mass: np.ndarray | float,
    m1: float,
    alpha: float,
    m_min: float,
) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    return (mass / float(m1)) ** float(alpha) * np.exp(-float(m_min) / mass)


def evaluate_hod(
    mass: np.ndarray | float,
    params: Mapping[str, Any],
    *,
    model: str = "lrg",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N_central, N_satellite)`` for a named vectorized HOD model.

    Accepted model names are ``lrg``, ``zheng``, ``elg_v1``, ``elg_v2``, and
    ``qso``. Parameter aliases such as ``logMcut``/``logM_cut`` and
    ``logM1``/``logM_1`` are both accepted.
    """

    mass = np.asarray(mass, dtype=np.float64)
    model_key = model.lower()
    logm_cut = float(_get(params, "logM_cut", "logMcut", "logm_cut", "logmcut"))
    sigma = float(_get(params, "sigma"))
    pmax = float(_get(params, "pmax", "p_max", default=1.0))
    m_cut = _mass_from_log_or_value(
        params,
        log_names=("logM_cut", "logMcut", "logm_cut", "logmcut"),
        value_names=("M_cut", "Mcut", "m_cut", "mcut"),
    )

    if model_key in {"lrg", "zheng_lrg", "baseline_lrg"}:
        m1 = _mass_from_log_or_value(
            params,
            log_names=("logM1", "logM_1", "logm1", "logm_1"),
            value_names=("M_1", "M1", "m_1", "m1"),
        )
        alpha = float(_get(params, "alpha"))
        kappa = float(_get(params, "kappa", default=1.0))
        return (
            n_cen_lrg(mass, logm_cut, sigma, pmax=pmax),
            n_sat_lrg_modified(mass, logm_cut, m_cut, m1, sigma, alpha, kappa, pmax=pmax),
        )

    if model_key in {"zheng", "generic", "baseline"}:
        m1 = _mass_from_log_or_value(
            params,
            log_names=("logM1", "logM_1", "logm1", "logm_1"),
            value_names=("M_1", "M1", "m_1", "m1"),
        )
        alpha = float(_get(params, "alpha"))
        kappa = float(_get(params, "kappa", default=1.0))
        satellite = n_sat_generic(mass, m_cut, kappa, m1, alpha, a_s=float(_get(params, "A_s", "As", default=1.0)))
        central = n_cen_lrg(mass, logm_cut, sigma, pmax=pmax)
        if bool(_get(params, "satellite_condition_on_central", default=False)):
            satellite = satellite * central
        return central, satellite

    if model_key in {"qso"}:
        central = n_cen_qso(mass, logm_cut, sigma, pmax=pmax)
        if "logMmin" in params or "Mmin" in params:
            m1 = _mass_from_log_or_value(
                params,
                log_names=("logM1", "logM_1", "logm1", "logm_1"),
                value_names=("M_1", "M1", "m_1", "m1"),
            )
            m_min = _mass_from_log_or_value(
                params,
                log_names=("logMmin", "logM_min", "logmmin", "logm_min"),
                value_names=("Mmin", "M_min", "mmin", "m_min"),
            )
            satellite = n_sat_qso(mass, m1, float(_get(params, "alpha")), m_min)
        else:
            m1 = _mass_from_log_or_value(
                params,
                log_names=("logM1", "logM_1", "logm1", "logm_1"),
                value_names=("M_1", "M1", "m_1", "m1"),
            )
            satellite = n_sat_generic(mass, m_cut, float(_get(params, "kappa", default=1.0)), m1, float(_get(params, "alpha")))
        return central, satellite

    if model_key in {"elg_v1", "elg_hmq"}:
        central = n_cen_elg_v1(
            mass,
            pmax=pmax,
            q=float(_get(params, "Q", "q")),
            logm_cut=logm_cut,
            sigma=sigma,
            gamma=float(_get(params, "gamma")),
            anorm=float(_get(params, "Anorm", "anorm", "maxpdf", default=1.0)),
        )
    elif model_key in {"elg_v2", "elg"}:
        central = n_cen_elg_v2(
            mass,
            pmax=pmax,
            logm_cut=logm_cut,
            sigma=sigma,
            gamma=float(_get(params, "gamma")),
        )
    else:
        raise ValueError(f"Unknown HOD model {model!r}.")

    m1 = _mass_from_log_or_value(
        params,
        log_names=("logM1", "logM_1", "logm1", "logm_1"),
        value_names=("M_1", "M1", "m_1", "m1"),
    )
    satellite = n_sat_generic(
        mass,
        m_cut,
        float(_get(params, "kappa", default=1.0)),
        m1,
        float(_get(params, "alpha")),
        a_s=float(_get(params, "A_s", "As", default=1.0)),
    )
    return central, satellite
