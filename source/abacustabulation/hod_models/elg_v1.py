"""ELG v1 HOD model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import a_c, a_s, alpha, kappa, logm_cut, m1, m_cut, n_cen_elg_v1, n_sat_generic, param, sigma


PARAMETERS = {
    "logMcut",
    "sigma",
    "a_c",
    "logM1",
    "alpha",
    "kappa",
    "a_s",
    "Q",
    "gamma",
    "maxpdf",
}


def evaluate(mass: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    central = n_cen_elg_v1(
        mass,
        a_c=a_c(params),
        q=float(param(params, "Q")),
        logm_cut=logm_cut(params),
        sigma=sigma(params),
        gamma=float(param(params, "gamma")),
        maxpdf=float(param(params, "maxpdf", default=1.0)),
    )
    satellite = n_sat_generic(
        mass,
        m_cut(params),
        kappa(params),
        m1(params),
        alpha(params),
        a_s=a_s(params),
    )
    return central, satellite
