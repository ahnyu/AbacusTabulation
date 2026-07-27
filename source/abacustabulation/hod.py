"""High-level HOD model registry and dispatcher."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

import numpy as np

from . import hod_models
from .hod_models.base import (
    gaussian_fun,
    mass_from_log_param,
    n_cen_elg_v1,
    n_cen_elg_v2,
    n_cen_zheng,
    n_sat_generic,
    n_sat_lrg_modified,
    param,
    phi_big_fun,
    phi_fun,
)


HODEvaluator = Callable[[np.ndarray, Mapping[str, Any]], tuple[np.ndarray, np.ndarray]]
HODSplitEvaluator = Callable[..., tuple[np.ndarray, np.ndarray]]
_SKIPPED_MODULES = {"base"}


def _iter_model_modules() -> tuple[tuple[str, ModuleType], ...]:
    modules: list[tuple[str, ModuleType]] = []
    for info in pkgutil.iter_modules(hod_models.__path__):
        if info.ispkg or info.name.startswith("_") or info.name in _SKIPPED_MODULES:
            continue
        module = importlib.import_module(f"{hod_models.__name__}.{info.name}")
        modules.append((info.name, module))
    return tuple(sorted(modules, key=lambda item: item[0]))


def _load_models() -> tuple[
    dict[str, HODEvaluator],
    dict[str, set[str]],
    dict[str, HODSplitEvaluator],
]:
    models: dict[str, HODEvaluator] = {}
    model_parameters: dict[str, set[str]] = {}
    split_models: dict[str, HODSplitEvaluator] = {}
    for name, module in _iter_model_modules():
        evaluator = getattr(module, "evaluate", None)
        parameters = getattr(module, "PARAMETERS", None)
        if not callable(evaluator) or parameters is None:
            raise AttributeError(
                f"HOD model module {module.__name__!r} must define PARAMETERS and evaluate()."
            )
        models[name] = evaluator
        model_parameters[name] = set(parameters)
        split_evaluator = getattr(module, "evaluate_all_splits", None)
        if callable(split_evaluator):
            split_models[name] = split_evaluator
    return models, model_parameters, split_models


HOD_MODELS, HOD_MODEL_PARAMETERS, HOD_SPLIT_MODELS = _load_models()


def _model_key(model: str) -> str:
    model_key = str(model).lower()
    if model_key not in HOD_MODELS:
        known = ", ".join(available_hod_models())
        raise ValueError(f"Unknown HOD model {model!r}. Known models: {known}.")
    return model_key


def _validate_params(model: str, params: Mapping[str, Any]) -> None:
    allowed = HOD_MODEL_PARAMETERS[model]
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported parameter(s) for HOD model {model!r}: {unknown}. "
            f"Allowed names are: {sorted(allowed)}."
        )


def available_hod_models() -> tuple[str, ...]:
    """Return accepted HOD model keys, matching filenames under ``hod_models``."""

    return tuple(HOD_MODELS)


def hod_model_parameters(model: str) -> tuple[str, ...]:
    """Return accepted canonical parameter names for one HOD model."""

    return tuple(sorted(HOD_MODEL_PARAMETERS[_model_key(model)]))


def hod_model_supports_splits(model: str) -> bool:
    """Return whether a model provides batched split occupations."""

    return _model_key(model) in HOD_SPLIT_MODELS


def evaluate_hod_splits(
    mass: np.ndarray | float,
    params: Mapping[str, Any],
    *,
    model: str = "lrg_stellar_mass",
    split_method: str,
    logmstar_edges: Any,
    redshift_weights: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all occupations from a split-capable HOD model."""

    model_key = _model_key(model)
    if model_key not in HOD_SPLIT_MODELS:
        raise ValueError(f"HOD model {model!r} does not provide split occupations.")
    _validate_params(model_key, params)
    return HOD_SPLIT_MODELS[model_key](
        np.asarray(mass, dtype=np.float64),
        params,
        split_method=split_method,
        logmstar_edges=logmstar_edges,
        redshift_weights=redshift_weights,
    )


def evaluate_hod(
    mass: np.ndarray | float,
    params: Mapping[str, Any],
    *,
    model: str = "lrg",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N_central, N_satellite)`` for a named vectorized HOD model.

    Canonical parameter names are intentionally strict and model-specific. Use
    ``a_c`` for central amplitude and ``a_s`` for satellite amplitude. Add
    a new model by creating ``hod_models/<model_name>.py`` with ``PARAMETERS``
    and ``evaluate(mass, params)``.
    """

    model_key = _model_key(model)
    _validate_params(model_key, params)
    return HOD_MODELS[model_key](np.asarray(mass, dtype=np.float64), params)
