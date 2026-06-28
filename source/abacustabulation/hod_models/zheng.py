"""Generic Zheng-style HOD model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import a_c, alpha, asat, kappa, logm_cut, m1, m_cut, n_cen_zheng, n_sat_generic, param, sigma


PARAMETERS = {
    "logMcut",
    "sigma",
    "a_c",
    "logM1",
    "alpha",
    "kappa",
    "A_s",
    "satellite_condition_on_central",
}


def evaluate(mass: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    central = n_cen_zheng(mass, logm_cut(params), sigma(params), a_c=a_c(params))
    satellite = n_sat_generic(
        mass,
        m_cut(params),
        kappa(params),
        m1(params),
        alpha(params),
        a_s=asat(params),
    )
    if bool(param(params, "satellite_condition_on_central", default=False)):
        satellite = satellite * central
    return central, satellite
