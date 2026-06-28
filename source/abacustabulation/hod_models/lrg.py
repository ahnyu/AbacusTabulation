"""LRG HOD model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import a_c, alpha, kappa, logm_cut, m1, m_cut, n_cen_zheng, n_sat_lrg_modified, sigma


PARAMETERS = {"logMcut", "sigma", "a_c", "logM1", "alpha", "kappa"}


def evaluate(mass: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    logmcut = logm_cut(params)
    sigma_value = sigma(params)
    a_c_value = a_c(params)
    return (
        n_cen_zheng(mass, logmcut, sigma_value, a_c=a_c_value),
        n_sat_lrg_modified(
            mass,
            logmcut,
            m_cut(params),
            m1(params),
            sigma_value,
            alpha(params),
            kappa(params),
            a_c=a_c_value,
        ),
    )
