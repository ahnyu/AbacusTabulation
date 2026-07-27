"""Runtime controls that must be applied before numerical libraries import."""

from __future__ import annotations

import os
from pathlib import Path


_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def configure_native_threads(n_threads: int) -> int:
    """Set native math-library thread limits for subsequently imported code."""

    n_threads = int(n_threads)
    if n_threads < 1:
        raise ValueError("Native threads per process must be at least 1.")
    for name in _NATIVE_THREAD_ENV_VARS:
        os.environ[name] = str(n_threads)
    return n_threads


def configure_pocomc_native_threads(
    path2config: str | Path,
    *,
    override: int | None = None,
) -> int:
    """Set inherited native thread limits before importing the fit stack."""

    from .config import load_config

    config = load_config(path2config)
    pocomc = (config.get("fit") or {}).get("mcmc", {}).get("pocomc", {})
    value = (
        override
        if override is not None
        else pocomc.get("native_threads_per_process", 1)
    )
    return configure_native_threads(value)


def configure_optimization_native_threads(
    path2config: str | Path,
    *,
    override: int | None = None,
) -> int:
    """Apply the configured optimization thread cap before numerical imports."""

    from .config import load_config

    config = load_config(path2config)
    optimization = (config.get("fit") or {}).get("optimization", {})
    value = (
        override
        if override is not None
        else optimization.get("native_threads_per_process", 1)
    )
    return configure_native_threads(value)
