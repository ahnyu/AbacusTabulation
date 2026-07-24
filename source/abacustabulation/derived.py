"""High-level HOD-derived quantities for post-processing fit outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hmf import (
    HaloMassFunction,
    StellarMassHODDerivedQuantities,
    find_hmf_file,
    hod_central_median_host_mass,
    hod_central_median_log10_host_mass,
    hod_central_number_density,
    hod_number_density,
    hod_satellite_fraction,
    hod_satellite_median_host_mass,
    hod_satellite_median_log10_host_mass,
    hod_satellite_number_density,
    read_hmf,
    stellar_mass_hod_derived_quantities,
    validate_hmf_file_for_config,
)
from .hod_models.lrg_stellar_mass import normalize_stellar_mass_selection
from .linear_bias import (
    HODLinearBiasTabulator,
    LinearBiasResult,
    StellarMassLinearBiasResult,
    infer_abacus_cosmology_index,
)
from .paircounts import resolve_paircount_path_from_config


DEFAULT_QUANTITIES = (
    "n_cen",
    "n_sat",
    "number_density",
    "satellite_fraction",
    "log10_mh_cen_med",
    "log10_mh_sat_med",
    "mh_cen_med",
    "mh_sat_med",
)
_HMF_QUANTITIES = set(DEFAULT_QUANTITIES)
_LINEAR_BIAS_QUANTITIES = {"linear_bias"}


@dataclass(frozen=True)
class StellarMassDerivedResult:
    """Selected derived quantities for every stellar-mass split."""

    values: Mapping[str, np.ndarray]
    split_method: str
    logmstar_edges: np.ndarray
    redshift_weights: np.ndarray

    @property
    def n_splits(self) -> int:
        if self.split_method == "raw":
            return int(self.logmstar_edges.size - 1)
        return int(self.logmstar_edges.shape[1] - 1)

    def __len__(self) -> int:
        return self.n_splits

    def __getitem__(self, index: int) -> dict[str, float]:
        if not 0 <= int(index) < self.n_splits:
            raise IndexError(index)
        return {name: float(value[index]) for name, value in self.values.items()}


class HODDerivedTabulator:
    """Reusable post-processing object for scalar HOD-derived quantities."""

    def __init__(
        self,
        *,
        hmf: HaloMassFunction | None = None,
        linear_bias_tabulator: HODLinearBiasTabulator | None = None,
        hod_model: str = "lrg",
        quantities: Sequence[str] = DEFAULT_QUANTITIES,
        stellar_mass: Mapping[str, Any] | None = None,
    ):
        self.hmf = hmf
        self.linear_bias_tabulator = linear_bias_tabulator
        self.hod_model = str(hod_model)
        self.quantities = tuple(_canonical_quantity_name(item) for item in quantities)
        self.stellar_mass = (
            None if not stellar_mass else _normalize_stellar_mass_config(stellar_mass)
        )

    @classmethod
    def from_config(
        cls,
        path_or_config: str | Path | Mapping[str, Any],
        *,
        tracer: str | None = None,
    ) -> "HODDerivedTabulator":
        if isinstance(path_or_config, Mapping):
            config = dict(path_or_config)
        else:
            from .config import load_config

            config = load_config(path_or_config)
        tracer = _resolve_tracer(config, tracer)
        derived_config = dict(config.get("derived", {}))
        tracer_config = _derived_tracer_config(config, tracer)
        quantities = tuple(
            _canonical_quantity_name(item)
            for item in tracer_config.get("quantities", derived_config.get("quantities", DEFAULT_QUANTITIES))
        )
        hod_model = str(
            tracer_config.get(
                "hod_model",
                _fit_tracer_theory_config(config, tracer).get("hod_model", config.get("hod", {}).get("model", "lrg")),
            )
        )
        theory_config = _fit_tracer_theory_config(config, tracer)
        stellar_mass = _merged_mapping(
            config.get("hod", {}).get("stellar_mass"),
            theory_config.get("stellar_mass"),
        )
        stellar_mass = _merged_mapping(
            stellar_mass,
            tracer_config.get("stellar_mass"),
        )
        linear_config = _merged_mapping(derived_config.get("linear_bias"), tracer_config.get("linear_bias"))
        if not bool(linear_config.get("enabled", True)):
            quantities = tuple(item for item in quantities if item not in _LINEAR_BIAS_QUANTITIES)

        hmf = None
        if any(item in _HMF_QUANTITIES for item in quantities):
            hmf = read_hmf(_resolve_hmf_path(config, tracer, tracer_config))

        linear_bias_tabulator = None
        linear_requested = any(item in _LINEAR_BIAS_QUANTITIES for item in quantities)
        if linear_requested:
            paircount_path = _resolve_linear_bias_paircount_path(config, linear_config)
            sim_params = config.get("sim_params", {})
            cosmology_index = linear_config.get("cosmology_index")
            if cosmology_index is None:
                cosmology_index = infer_abacus_cosmology_index(str(sim_params.get("sim_name", "")))
            z = float(_first_not_none(linear_config.get("z"), sim_params.get("z_mock"), 0.0))
            linear_bias_tabulator = HODLinearBiasTabulator.from_paircount_file(
                paircount_path,
                cosmology_index=int(cosmology_index),
                z=z,
                engine=str(linear_config.get("engine", "camb")),
                fit_s_min=_optional_float(linear_config.get("fit_s_min")),
                fit_s_max=_optional_float(linear_config.get("fit_s_max")),
                positive_only=bool(linear_config.get("positive_only", True)),
            )

        return cls(
            hmf=hmf,
            linear_bias_tabulator=linear_bias_tabulator,
            hod_model=hod_model,
            quantities=quantities,
            stellar_mass=stellar_mass,
        )

    def number_density(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> float:
        return hod_number_density(self._require_hmf(), hod_params, hod_model=hod_model or self.hod_model)

    def central_number_density(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> float:
        return hod_central_number_density(self._require_hmf(), hod_params, hod_model=hod_model or self.hod_model)

    def satellite_number_density(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> float:
        return hod_satellite_number_density(self._require_hmf(), hod_params, hod_model=hod_model or self.hod_model)

    def satellite_fraction(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> float:
        return hod_satellite_fraction(self._require_hmf(), hod_params, hod_model=hod_model or self.hod_model)

    def median_central_host_mass(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> float:
        return hod_central_median_host_mass(self._require_hmf(), hod_params, hod_model=hod_model or self.hod_model)

    def median_satellite_host_mass(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> float:
        return hod_satellite_median_host_mass(self._require_hmf(), hod_params, hod_model=hod_model or self.hod_model)

    def linear_bias_result(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> LinearBiasResult:
        if self.linear_bias_tabulator is None:
            raise KeyError("Linear bias was requested but no linear-bias tabulator is configured.")
        return self.linear_bias_tabulator.evaluate(hod_params, hod_model=hod_model or self.hod_model)

    def linear_bias(self, hod_params: Mapping[str, Any], *, hod_model: str | None = None) -> float:
        return self.linear_bias_result(hod_params, hod_model=hod_model).bias

    def evaluate(
        self,
        hod_params: Mapping[str, Any],
        *,
        hod_model: str | None = None,
        quantities: Sequence[str] | None = None,
    ) -> dict[str, float]:
        model = hod_model or self.hod_model
        selected = tuple(_canonical_quantity_name(item) for item in (quantities or self.quantities))
        out: dict[str, float] = {}
        for quantity in selected:
            if quantity == "n_cen":
                out[quantity] = self.central_number_density(hod_params, hod_model=model)
            elif quantity == "n_sat":
                out[quantity] = self.satellite_number_density(hod_params, hod_model=model)
            elif quantity == "number_density":
                out[quantity] = self.number_density(hod_params, hod_model=model)
            elif quantity == "satellite_fraction":
                out[quantity] = self.satellite_fraction(hod_params, hod_model=model)
            elif quantity == "log10_mh_cen_med":
                out[quantity] = hod_central_median_log10_host_mass(self._require_hmf(), hod_params, hod_model=model)
            elif quantity == "log10_mh_sat_med":
                out[quantity] = hod_satellite_median_log10_host_mass(self._require_hmf(), hod_params, hod_model=model)
            elif quantity == "mh_cen_med":
                out[quantity] = self.median_central_host_mass(hod_params, hod_model=model)
            elif quantity == "mh_sat_med":
                out[quantity] = self.median_satellite_host_mass(hod_params, hod_model=model)
            elif quantity == "linear_bias":
                out[quantity] = self.linear_bias(hod_params, hod_model=model)
            else:
                raise ValueError(f"Unknown derived quantity {quantity!r}.")
        return out

    @property
    def has_stellar_mass(self) -> bool:
        return self.stellar_mass is not None

    def evaluate_stellar_mass(
        self,
        hod_params: Mapping[str, Any],
        *,
        split_method: str | None = None,
        logmstar_edges: Any = None,
        redshift_weights: Any = None,
        hod_model: str | None = None,
        quantities: Sequence[str] | None = None,
    ) -> StellarMassDerivedResult:
        """Evaluate selected derived quantities for every stellar-mass split."""

        configured = self.stellar_mass or {}
        method = split_method if split_method is not None else configured.get("split_method")
        edges = logmstar_edges if logmstar_edges is not None else configured.get("logmstar_edges")
        weights = (
            redshift_weights
            if redshift_weights is not None
            else configured.get("redshift_weights")
        )
        if method is None or edges is None:
            raise KeyError(
                "Set split_method and logmstar_edges directly or configure "
                "derived.tracers.<TRACER>.stellar_mass."
            )
        normalized_edges, normalized_weights = normalize_stellar_mass_selection(
            str(method),
            edges,
            weights,
        )
        normalized_method = str(method).lower()
        model_edges = (
            normalized_edges[0] if normalized_method == "raw" else normalized_edges
        )
        model_weights = None if normalized_method == "raw" else normalized_weights
        model = hod_model or self.hod_model
        selected = tuple(
            _canonical_quantity_name(item) for item in (quantities or self.quantities)
        )

        hmf_result: StellarMassHODDerivedQuantities | None = None
        if any(item in _HMF_QUANTITIES for item in selected):
            hmf_result = stellar_mass_hod_derived_quantities(
                self._require_hmf(),
                hod_params,
                split_method=normalized_method,
                logmstar_edges=model_edges,
                redshift_weights=model_weights,
                hod_model=model,
            )
        linear_result: StellarMassLinearBiasResult | None = None
        if any(item in _LINEAR_BIAS_QUANTITIES for item in selected):
            if self.linear_bias_tabulator is None:
                raise KeyError(
                    "Linear bias was requested but no linear-bias tabulator is configured."
                )
            linear_result = self.linear_bias_tabulator.evaluate_stellar_mass(
                hod_params,
                split_method=normalized_method,
                logmstar_edges=model_edges,
                redshift_weights=model_weights,
                hod_model=model,
            )

        values: dict[str, np.ndarray] = {}
        for quantity in selected:
            if quantity == "n_cen":
                values[quantity] = _require_stellar_hmf_result(hmf_result).n_cen
            elif quantity == "n_sat":
                values[quantity] = _require_stellar_hmf_result(hmf_result).n_sat
            elif quantity == "number_density":
                values[quantity] = _require_stellar_hmf_result(hmf_result).n_gal
            elif quantity == "satellite_fraction":
                values[quantity] = _require_stellar_hmf_result(hmf_result).f_sat
            elif quantity == "log10_mh_cen_med":
                values[quantity] = _require_stellar_hmf_result(hmf_result).log10_mh_cen_med
            elif quantity == "log10_mh_sat_med":
                values[quantity] = _require_stellar_hmf_result(hmf_result).log10_mh_sat_med
            elif quantity == "mh_cen_med":
                values[quantity] = _require_stellar_hmf_result(hmf_result).mh_cen_med
            elif quantity == "mh_sat_med":
                values[quantity] = _require_stellar_hmf_result(hmf_result).mh_sat_med
            elif quantity == "linear_bias":
                if linear_result is None:
                    raise RuntimeError("Missing stellar-mass linear-bias result.")
                values[quantity] = linear_result.bias
            else:
                raise ValueError(f"Unknown derived quantity {quantity!r}.")

        n_splits = normalized_edges.shape[1] - 1
        for name, value in values.items():
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (n_splits,):
                raise ValueError(
                    f"Derived quantity {name!r} has shape {value.shape}; "
                    f"expected ({n_splits},)."
                )
            values[name] = value
        output_edges = model_edges
        return StellarMassDerivedResult(
            values=values,
            split_method=normalized_method,
            logmstar_edges=output_edges,
            redshift_weights=normalized_weights,
        )

    def _require_hmf(self) -> HaloMassFunction:
        if self.hmf is None:
            raise KeyError("This derived quantity requires an HMF, but no HMF is configured.")
        return self.hmf


def _require_stellar_hmf_result(
    result: StellarMassHODDerivedQuantities | None,
) -> StellarMassHODDerivedQuantities:
    if result is None:
        raise RuntimeError("Missing stellar-mass HMF-derived result.")
    return result


def _normalize_stellar_mass_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "split_method" not in config:
        raise KeyError("stellar_mass.split_method is required.")
    if "logmstar_edges" not in config:
        raise KeyError("stellar_mass.logmstar_edges is required.")
    method = str(config["split_method"]).lower()
    edges, weights = normalize_stellar_mass_selection(
        method,
        config["logmstar_edges"],
        config.get("redshift_weights"),
    )
    return {
        "split_method": method,
        "logmstar_edges": edges[0] if method == "raw" else edges,
        "redshift_weights": None if method == "raw" else weights,
    }


def _stellar_mass_result_payload(result: StellarMassDerivedResult) -> dict[str, Any]:
    return {
        "split_method": result.split_method,
        "logmstar_edges": result.logmstar_edges,
        "redshift_weights": result.redshift_weights,
        "splits": [
            {"index": index, "derived": result[index]}
            for index in range(result.n_splits)
        ],
    }


def write_optimization_derived(problem: Any, theta: Sequence[float], output_dir: str | Path, prefix: str) -> dict[str, Any]:
    """Compute and write derived quantities for one best-fit parameter vector."""

    if not fit_postprocess_enabled(problem, "optimization"):
        return {}
    output_dir = Path(output_dir)
    payload: dict[str, Any] = {}
    for tracer in problem.tracers:
        tabulator = HODDerivedTabulator.from_config(problem.config, tracer=tracer)
        params = problem.params_for_tracer(theta, tracer)
        if tabulator.has_stellar_mass:
            payload[tracer] = {
                "hod_params": params,
                "stellar_mass": _stellar_mass_result_payload(
                    tabulator.evaluate_stellar_mass(params)
                ),
            }
        else:
            payload[tracer] = {
                "hod_params": params,
                "derived": tabulator.evaluate(params),
            }
    path = output_dir / f"{prefix}_optimum_derived.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)
    return payload


def write_mcmc_derived(
    problem: Any,
    samples: np.ndarray,
    weights: np.ndarray,
    loglike: np.ndarray,
    logprior: np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Any]:
    """Compute and write derived quantities aligned with saved posterior samples."""

    if not fit_postprocess_enabled(problem, "mcmc"):
        return {}
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    loglike = np.asarray(loglike, dtype=np.float64).reshape(-1)
    logprior = np.asarray(logprior, dtype=np.float64).reshape(-1)
    output_dir = Path(output_dir)

    n_samples = samples.shape[0]
    if not (weights.size == loglike.size == logprior.size == n_samples):
        raise ValueError("MCMC samples, weights, loglike, and logprior lengths must match.")
    sample_indices = np.arange(n_samples)

    tabulators = {tracer: HODDerivedTabulator.from_config(problem.config, tracer=tracer) for tracer in problem.tracers}
    derived_rows: list[list[float]] = []
    derived_names: list[str] | None = None
    for theta in samples:
        row: list[float] = []
        names: list[str] = []
        for tracer, tabulator in tabulators.items():
            params = problem.params_for_tracer(theta, tracer)
            if tabulator.has_stellar_mass:
                split_result = tabulator.evaluate_stellar_mass(params)
                for split_index in range(split_result.n_splits):
                    for key, value in split_result.values.items():
                        names.append(f"{tracer}.sm{split_index}.{key}")
                        row.append(float(value[split_index]))
            else:
                values = tabulator.evaluate(params)
                for key, value in values.items():
                    names.append(f"{tracer}.{key}")
                    row.append(float(value))
        if derived_names is None:
            derived_names = names
        elif derived_names != names:
            raise RuntimeError("Derived quantity columns changed between samples.")
        derived_rows.append(row)

    derived = np.asarray(derived_rows, dtype=np.float64)
    base = np.column_stack([sample_indices, samples, weights, loglike, logprior])
    table = np.column_stack([base, derived]) if derived.size else base
    header = " ".join(("sample_index", *problem.parameter_names, "weight", "loglike", "logprior", *(derived_names or ())))
    path = output_dir / f"{prefix}_chains_derived.txt"
    np.savetxt(path, table, header=header)

    summary = _weighted_summary(derived, weights, derived_names or [])
    summary_path = output_dir / f"{prefix}_derived_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)
    return summary


def fit_postprocess_enabled(problem: Any, mode: str) -> bool:
    post = problem.fit_config.get("postprocess", {})
    if not post or not bool(post.get("enabled", False)):
        return False
    return bool(post.get(mode, True))


def _canonical_quantity_name(name: Any) -> str:
    value = str(name)
    aliases = {
        "n_gal": "number_density",
        "ng": "number_density",
        "f_sat": "satellite_fraction",
        "mh_cen_median": "mh_cen_med",
        "mh_sat_median": "mh_sat_med",
    }
    return aliases.get(value, value)


def _resolve_tracer(config: Mapping[str, Any], tracer: str | None) -> str:
    if tracer is not None:
        return str(tracer)
    derived_tracers = config.get("derived", {}).get("tracers", {})
    if derived_tracers:
        return str(next(iter(derived_tracers)))
    fit_tracers = (config.get("fit") or {}).get("tracers")
    if fit_tracers:
        return str(fit_tracers[0] if not isinstance(fit_tracers, str) else fit_tracers)
    return "LRG"


def _derived_tracer_config(config: Mapping[str, Any], tracer: str) -> dict[str, Any]:
    return dict(config.get("derived", {}).get("tracers", {}).get(tracer, {}))


def _fit_tracer_theory_config(config: Mapping[str, Any], tracer: str) -> Mapping[str, Any]:
    return (config.get("fit") or {}).get("theory", {}).get("tracers", {}).get(tracer, {})


def _merged_mapping(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(base or {})
    out.update(dict(override or {}))
    return out


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _z_directory(z_mock: float) -> str:
    return f"z{float(z_mock):.3f}"


def _format_values(config: Mapping[str, Any]) -> dict[str, Any]:
    sim_params = config.get("sim_params", {})
    z_mock = float(sim_params.get("z_mock", 0.0))
    return {
        "sim_name": str(sim_params.get("sim_name", "")),
        "z": _z_directory(z_mock),
        "z_mock": z_mock,
    }


def _has_sim_format_fields(value: str) -> bool:
    return any(field in value for field in ("{sim_name", "{z", "{z_mock"))


def _format_exact_path(value: str | Path, config: Mapping[str, Any]) -> Path:
    return Path(str(value).format(**_format_values(config)))


def _format_sim_output_dir(value: str | Path, config: Mapping[str, Any]) -> Path:
    values = _format_values(config)
    template = str(value)
    has_fields = _has_sim_format_fields(template)
    path = Path(template.format(**values))
    if has_fields or tuple(path.parts[-2:]) == (values["sim_name"], values["z"]):
        return path
    return path / values["sim_name"] / values["z"]


def _resolve_hmf_path(config: Mapping[str, Any], tracer: str, tracer_config: Mapping[str, Any]) -> Path:
    hmf_config = _merged_mapping(config.get("hmf"), tracer_config.get("hmf"))
    if hmf_config.get("path") is not None:
        return validate_hmf_file_for_config(
            _format_exact_path(hmf_config["path"], config),
            n_bins=int(hmf_config.get("n_bins", 512)),
            logm_edges=hmf_config.get("logm_edges"),
            logm_min=hmf_config.get("logm_min"),
            logm_max=hmf_config.get("logm_max"),
        )
    output_value = hmf_config.get("output_dir", hmf_config.get("out_dir"))
    if output_value is None:
        raise KeyError(f"Set hmf.output_dir, hmf.path, or derived.tracers.{tracer}.hmf.path.")
    pair_params = config.get("paircounts", {})
    prepare_params = config.get("prepare_profiles", {})
    return find_hmf_file(
        _format_sim_output_dir(output_value, config),
        file_tag=_first_not_none(hmf_config.get("file_tag"), pair_params.get("file_tag")),
        seed=_first_not_none(hmf_config.get("seed"), pair_params.get("seed"), prepare_params.get("seed")),
        n_bins=int(hmf_config.get("n_bins", 512)),
        logm_edges=hmf_config.get("logm_edges"),
        logm_min=hmf_config.get("logm_min"),
        logm_max=hmf_config.get("logm_max"),
    )


def _resolve_linear_bias_paircount_path(config: Mapping[str, Any], linear_config: Mapping[str, Any]) -> Path:
    path = linear_config.get("paircounts_path", linear_config.get("path"))
    job = str(linear_config.get("paircounts_job", "linear_bias"))
    path_config = dict(linear_config.get("paircounts", {}))
    path_config.setdefault("job", job)
    if path is not None:
        path_config["path"] = path
    return resolve_paircount_path_from_config(
        config,
        clustering="smu",
        path_config=path_config,
        job_name=job,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _weighted_summary(values: np.ndarray, weights: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    if values.size == 0:
        return {}
    weights = np.asarray(weights, dtype=np.float64)
    out: dict[str, Any] = {"n_samples": int(values.shape[0]), "columns": list(names)}
    stats: dict[str, Any] = {}
    for i, name in enumerate(names):
        column = values[:, i]
        finite = np.isfinite(column) & np.isfinite(weights)
        if not np.any(finite):
            stats[name] = {"mean": np.nan, "std": np.nan}
            continue
        w = weights[finite]
        if float(np.sum(w)) > 0.0:
            x = column[finite]
            norm = float(np.sum(w))
            mean = float(np.sum(w * x) / norm)
            var = float(np.sum(w * (x - mean) ** 2) / norm)
        else:
            x = column[finite]
            mean = float(np.mean(x))
            var = float(np.var(x))
        stats[name] = {"mean": mean, "std": float(np.sqrt(max(var, 0.0)))}
    out["statistics"] = stats
    return out
