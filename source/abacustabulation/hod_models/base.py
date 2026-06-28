"""Shared vectorized HOD occupation functions."""

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


_MISSING = object()


def param(params: Mapping[str, Any], name: str, default: Any = _MISSING) -> Any:
    if name in params:
        return params[name]
    if default is _MISSING:
        raise KeyError(f"Missing HOD parameter {name!r}.")
    return default


def mass_from_log_param(params: Mapping[str, Any], name: str) -> float:
    return 10.0 ** float(param(params, name))


def logm_cut(params: Mapping[str, Any]) -> float:
    return float(param(params, "logMcut"))


def m_cut(params: Mapping[str, Any]) -> float:
    return mass_from_log_param(params, "logMcut")


def m1(params: Mapping[str, Any]) -> float:
    return mass_from_log_param(params, "logM1")


def sigma(params: Mapping[str, Any]) -> float:
    return float(param(params, "sigma"))


def a_c(params: Mapping[str, Any]) -> float:
    return float(param(params, "a_c"))


def alpha(params: Mapping[str, Any]) -> float:
    return float(param(params, "alpha"))


def kappa(params: Mapping[str, Any]) -> float:
    return float(param(params, "kappa"))


def asat(params: Mapping[str, Any]) -> float:
    return float(param(params, "A_s", default=1.0))


def n_cen_zheng(
    mass: np.ndarray | float,
    logm_cut: float,
    sigma: float,
    a_c: float = 1.0,
) -> np.ndarray:
    """Zheng-style softened central occupation."""

    log_mass = np.log10(np.asarray(mass, dtype=np.float64))
    return float(a_c) * 0.5 * _erfc(
        (float(logm_cut) - log_mass) / (math.sqrt(2.0) * float(sigma))
    )


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
    return 0.5 * (
        1.0
        + _erf(
            float(gamma) * (log_mass - float(logm_cut)) / (math.sqrt(2.0) * float(sigma))
        )
    )


def n_cen_elg_v1(
    mass: np.ndarray | float,
    a_c: float,
    q: float,
    logm_cut: float,
    sigma: float,
    gamma: float,
    maxpdf: float = 1.0,
) -> np.ndarray:
    """ELG HMQ-style central occupation from the resource example."""

    log_mass = np.log10(np.asarray(mass, dtype=np.float64))
    return (
        2.0
        * (float(a_c) - 1.0 / float(q))
        * phi_fun(log_mass, logm_cut, sigma)
        * phi_big_fun(log_mass, logm_cut, sigma, gamma)
        / float(maxpdf)
    )


def n_cen_elg_v2(
    mass: np.ndarray | float,
    a_c: float,
    logm_cut: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """ELG central occupation from arXiv:2007.09012 as in the resource example."""

    mass = np.asarray(mass, dtype=np.float64)
    log_mass = np.log10(mass)
    out = np.empty_like(log_mass, dtype=np.float64)
    low = log_mass <= float(logm_cut)
    out[low] = float(a_c) * gaussian_fun(log_mass[low], logm_cut, sigma)
    out[~low] = float(a_c) * (mass[~low] / 10.0 ** float(logm_cut)) ** float(gamma) / (
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
    out[positive] = (
        float(a_s) * ((mass[positive] - float(kappa) * float(m_cut)) / float(m1)) ** float(alpha)
    )
    return out


def n_sat_lrg_modified(
    mass: np.ndarray | float,
    logm_cut: float,
    m_cut: float,
    m1: float,
    sigma: float,
    alpha: float,
    kappa: float,
    a_c: float = 1.0,
) -> np.ndarray:
    """LRG satellite occupation: power law multiplied by central occupation."""

    return n_sat_generic(mass, m_cut, kappa, m1, alpha) * n_cen_zheng(
        mass,
        logm_cut,
        sigma,
        a_c=a_c,
    )
