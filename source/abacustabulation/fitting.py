"""Config-driven helpers for HOD fitting against tabulated clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .clustering import HODClusteringTabulator, projected_wp, smu_multipoles
from .config import resolve_hod_model_from_config
from .hmf import (
    HaloMassFunction,
    find_hmf_file,
    hod_number_density,
    read_hmf,
    stellar_mass_hod_number_densities,
    validate_hmf_file_for_config,
)
from .hod import hod_model_supports_splits
from .stellar_mass import StellarMassSelection


ArrayLike = np.ndarray | Sequence[float]
DensityKey = tuple[str, str | None]
_SUPPORTED_STATISTICS = {"wp", "xi0", "xi2"}
_SMU_STATISTICS = {"xi0", "xi2"}


@dataclass(frozen=True)
class FitParameter:
    """One scalar free parameter with a uniform top-hat prior."""

    name: str
    initial: float
    minimum: float
    maximum: float
    shared: bool = False

    def contains(self, value: float) -> bool:
        return self.minimum <= float(value) <= self.maximum


@dataclass(frozen=True)
class OrderedParameterConstraint:
    """Require one fitted parameter value to be no larger than another."""

    lower: str
    upper: str


@dataclass(frozen=True)
class ObservableSpec:
    """Description of one fitted theory-vector segment."""

    name: str
    tracer: str
    statistic: str
    clustering: str
    paircount_path: Path
    hod_model: str
    selection: Any = None
    sample: str | None = None
    split_index: int | None = None

    @property
    def key(self) -> str:
        pieces = [self.tracer]
        if self.sample is not None:
            pieces.append(self.sample)
        pieces.append(self.statistic)
        return ".".join(pieces)

    @property
    def tabulator_key(self) -> Path:
        return self.paircount_path


@dataclass(frozen=True)
class ObservableDataSegment:
    """Observed data segment aligned to one observable spec."""

    spec: ObservableSpec
    values: np.ndarray
    selected_indices: np.ndarray
    full_size: int

    @property
    def size(self) -> int:
        return int(self.values.size)

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.spec.key, self.spec.name, self.spec.statistic)


@dataclass(frozen=True)
class NumberDensityConstraint:
    """Number-density constraint for a tracer or stellar sample."""

    tracer: str
    value: float
    mode: str = "minimum"
    error: float | None = None
    source: str = "paircounts"
    sample: str | None = None

    @property
    def key(self) -> DensityKey:
        return self.tracer, self.sample

    @property
    def label(self) -> str:
        return self.tracer if self.sample is None else f"{self.tracer}.{self.sample}"

    def loglike(self, theory_density: float) -> float:
        mode = self.mode.lower()
        if mode not in {"none", "minimum", "gaussian", "gaussian_two_sided"}:
            raise ValueError(
                f"Unknown number-density mode {self.mode!r} for sample {self.label!r}."
            )
        if mode == "none":
            return 0.0
        if mode == "gaussian_two_sided":
            if self.error is None or self.error <= 0.0:
                raise ValueError(
                    f"number_density.error must be positive for sample {self.label!r}."
                )
            diff = theory_density - self.value
            return float(-0.5 * diff * diff / (self.error * self.error))
        if theory_density >= self.value:
            return 0.0
        if mode == "minimum":
            return -np.inf
        if mode == "gaussian":
            if self.error is None or self.error <= 0.0:
                raise ValueError(
                    f"number_density.error must be positive for sample {self.label!r}."
                )
            diff = theory_density - self.value
            return float(-0.5 * diff * diff / (self.error * self.error))
        raise AssertionError(f"Unhandled number-density mode {mode!r}.")


@dataclass(frozen=True)
class StellarMassFitSpec:
    """One tracer fixed stellar-mass split definition and fitted sample map."""

    tracer: str
    selection: StellarMassSelection
    samples: Mapping[str, int]

    @property
    def n_splits(self) -> int:
        return self.selection.n_splits

    @property
    def split_method(self) -> str:
        return self.selection.split_method

    @property
    def logmstar_edges(self) -> np.ndarray:
        return self.selection.logmstar_edges

    @property
    def redshift_weights(self) -> np.ndarray | None:
        return self.selection.redshift_weights


@dataclass
class _TheoryEvaluation:
    """Cached theory pieces for one HOD parameter vector."""

    theory_vector: np.ndarray
    number_densities: Mapping[DensityKey, float]


@dataclass
class FitDataVector:
    """Observed clustering data vector and inverse covariance."""

    values: np.ndarray
    covariance: np.ndarray
    inverse_covariance: np.ndarray
    names: tuple[str, ...]
    segments: tuple[ObservableDataSegment, ...] = ()

    @property
    def size(self) -> int:
        return int(self.values.size)

    def chi2(self, theory: ArrayLike) -> float:
        theory = np.asarray(theory, dtype=np.float64).reshape(-1)
        if theory.size != self.values.size:
            raise ValueError(
                f"Theory vector has length {theory.size}; data vector has length {self.values.size}."
            )
        diff = theory - self.values
        return float(diff @ self.inverse_covariance @ diff)

    def loglike(self, theory: ArrayLike) -> float:
        chi2 = self.chi2(theory)
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2

    @classmethod
    def from_arrays(
        cls,
        values: ArrayLike,
        covariance: ArrayLike,
        *,
        names: Sequence[str] = (),
        segments: Sequence[ObservableDataSegment] = (),
        inversion: str = "inv",
        precision_scale: float = 1.0,
    ) -> "FitDataVector":
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (values.size, values.size):
            raise ValueError(
                f"Covariance shape {covariance.shape} does not match data length {values.size}."
            )
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(covariance)):
            raise ValueError("Data values and covariance must be finite.")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-8, atol=1.0e-12):
            raise ValueError("Covariance must be symmetric.")
        covariance = 0.5 * (covariance + covariance.T)
        precision_scale = float(precision_scale)
        if not np.isfinite(precision_scale) or precision_scale <= 0.0:
            raise ValueError("covariance.precision_scale must be finite and positive.")
        if inversion == "pinv":
            inverse = np.linalg.pinv(covariance)
        elif inversion == "inv":
            inverse = np.linalg.inv(covariance)
        else:
            raise ValueError("covariance.inversion must be 'inv' or 'pinv'.")
        inverse = precision_scale * inverse
        if not np.all(np.isfinite(inverse)):
            raise ValueError("Inverse covariance contains non-finite values.")
        return cls(
            values=values,
            covariance=covariance,
            inverse_covariance=inverse,
            names=tuple(names),
            segments=tuple(segments),
        )


class HODFittingProblem:
    """Reusable likelihood object for tabulated HOD fitting."""

    def __init__(
        self,
        *,
        data: FitDataVector,
        parameters: Sequence[FitParameter],
        fixed_params: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
        observables: Sequence[ObservableSpec],
        tabulators: Mapping[Any, HODClusteringTabulator],
        density_constraints: Sequence[NumberDensityConstraint] = (),
        hmfs: Mapping[str, HaloMassFunction] | None = None,
        config: Mapping[str, Any] | None = None,
        fit_config: Mapping[str, Any] | None = None,
        tracers: Sequence[str] = (),
        stellar_mass_specs: Mapping[str, StellarMassFitSpec] | None = None,
        hod_models_by_tracer: Mapping[str, str] | None = None,
        parameter_constraints: Sequence[OrderedParameterConstraint] = (),
    ):
        self.data = data
        self.parameters = tuple(parameters)
        self.observables = tuple(observables)
        self.tabulators = dict(tabulators)
        self.density_constraints = tuple(density_constraints)
        self.hmfs = dict(hmfs or {})
        self.config = dict(config or {})
        self.fit_config = dict(fit_config or {})
        self.tracers = tuple(str(item) for item in (tracers or _infer_tracers(self.observables)))
        self.stellar_mass_specs = dict(stellar_mass_specs or {})
        self.hod_models_by_tracer = _normalize_tracer_hod_models(
            self.observables,
            self.tracers,
            hod_models_by_tracer,
        )
        self.fixed_params_by_tracer = _normalize_fixed_params(fixed_params, self.tracers)
        self.parameter_names = tuple(param.name for param in self.parameters)
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError(f"Duplicate fit parameter names: {self.parameter_names}")
        self.parameter_constraints = tuple(parameter_constraints)
        parameter_indices = {
            name: index for index, name in enumerate(self.parameter_names)
        }
        self._parameter_constraint_indices = tuple(
            (parameter_indices[item.lower], parameter_indices[item.upper])
            for item in self.parameter_constraints
        )
        _validate_parameter_ownership(self.parameters, self.tracers)
        _validate_stellar_model_contract(
            self.tracers,
            self.stellar_mass_specs,
            self.hod_models_by_tracer,
        )

    @classmethod
    def from_config(cls, path2config: str | Path, *, validate: bool = True) -> "HODFittingProblem":
        from .config import load_config

        config = load_config(path2config)
        fit_config = config.get("fit")
        if fit_config is None:
            raise KeyError("Set a fit: block in the universal config.")

        tracers = _parse_tracers(fit_config)
        observable_configs = fit_config.get("observables")
        if not observable_configs:
            raise KeyError("Set fit.observables to a list of statistics to fit.")

        parameters = _parse_parameters(_required(fit_config, "parameters"))
        parameter_constraints = _parse_parameter_constraints(
            fit_config.get("parameter_constraints"),
            parameters,
        )
        stellar_mass_specs = _parse_stellar_mass_specs(config, fit_config, tracers)
        hod_models_by_tracer = {
            tracer: resolve_hod_model_from_config(config, tracer=tracer)
            for tracer in tracers
        }
        observables = _parse_observable_specs(
            config,
            fit_config,
            tracers,
            stellar_mass_specs,
            hod_models_by_tracer,
        )
        tabulators = _load_observable_tabulators(observables)
        data_segments = _load_observable_data(config, fit_config, observables, tabulators)
        covariance = _load_fit_covariance(config, fit_config, data_segments)
        covariance_config = fit_config.get("covariance", {})
        values = np.concatenate([segment.values for segment in data_segments])
        data = FitDataVector.from_arrays(
            values,
            covariance,
            names=tuple(segment.spec.key for segment in data_segments),
            segments=data_segments,
            inversion=str(covariance_config.get("inversion", "inv")),
            precision_scale=_precision_scale(covariance_config, values.size),
        )

        density_constraints = _parse_density_constraints(
            fit_config, tracers, stellar_mass_specs
        )
        problem = cls(
            data=data,
            parameters=parameters,
            fixed_params=_parse_fixed_params(config, fit_config, tracers),
            observables=observables,
            tabulators=tabulators,
            density_constraints=density_constraints,
            hmfs=_load_fit_hmfs(config, fit_config, density_constraints),
            config=config,
            fit_config=fit_config,
            tracers=tracers,
            stellar_mass_specs=stellar_mass_specs,
            hod_models_by_tracer=hod_models_by_tracer,
            parameter_constraints=parameter_constraints,
        )
        if validate:
            evaluation = problem._evaluate(problem.initial_vector())
            if evaluation.theory_vector.size != data.size:
                raise ValueError(
                    f"Initial theory vector has length {evaluation.theory_vector.size}; "
                    f"data vector has length {data.size}."
                )
            problem._density_loglike_from_densities(evaluation.number_densities)
        return problem

    def format_path(self, value: str | Path) -> Path:
        return _format_config_path(value, self.config)

    def initial_vector(self) -> np.ndarray:
        return np.array([param.initial for param in self.parameters], dtype=np.float64)

    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((param.minimum, param.maximum) for param in self.parameters)

    def params_from_vector(self, theta: ArrayLike) -> dict[str, Any]:
        theta = self._theta(theta)
        return {param.name: float(value) for param, value in zip(self.parameters, theta, strict=True)}

    def params_for_tracer(self, theta: ArrayLike, tracer: str) -> dict[str, Any]:
        theta = self._theta(theta)
        params = dict(self.fixed_params_by_tracer.get(tracer, {}))
        multi_tracer = len(self.tracers) > 1
        for param, value in zip(self.parameters, theta, strict=True):
            if "." in param.name:
                param_tracer, param_name = param.name.split(".", 1)
                if param_tracer == tracer:
                    params[param_name] = float(value)
            elif not multi_tracer or param.shared:
                params[param.name] = float(value)
        return params

    def logprior(self, theta: ArrayLike) -> float:
        theta = self._theta(theta)
        for value, param in zip(theta, self.parameters, strict=True):
            if not param.contains(float(value)):
                return -np.inf
        if not self._satisfies_parameter_constraints(theta):
            return -np.inf
        return 0.0

    def theory_vector(self, theta: ArrayLike) -> np.ndarray:
        return self._evaluate(theta).theory_vector

    def theory_number_density(
        self,
        theta: ArrayLike,
        tracer: str,
        *,
        sample: str | None = None,
    ) -> float:
        densities = self._evaluate(theta).number_densities
        key = (str(tracer), None if sample is None else str(sample))
        if key not in densities:
            label = key[0] if key[1] is None else f"{key[0]}.{key[1]}"
            raise KeyError(
                f"No observable or HMF density is configured for sample {label!r}."
            )
        return float(densities[key])

    def density_loglike(self, theta: ArrayLike) -> float:
        return self._density_loglike_from_densities(self._evaluate(theta).number_densities)

    def loglike(self, theta: ArrayLike) -> float:
        theta = self._theta(theta)
        if not self._satisfies_parameter_constraints(theta):
            return -np.inf
        evaluation = self._evaluate(theta)
        clustering_value = self.data.loglike(evaluation.theory_vector)
        density_value = self._density_loglike_from_densities(evaluation.number_densities)
        value = clustering_value + density_value
        return float(value) if np.isfinite(value) else -np.inf

    def logposterior(self, theta: ArrayLike) -> float:
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.loglike(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def negative_loglike(self, theta: ArrayLike) -> float:
        value = self.loglike(theta)
        return float(np.inf if not np.isfinite(value) else -value)

    def negative_logposterior(self, theta: ArrayLike) -> float:
        value = self.logposterior(theta)
        return float(np.inf if not np.isfinite(value) else -value)

    def unit_cube_to_parameters(self, unit: ArrayLike) -> np.ndarray:
        unit = self._theta(unit)
        lows = np.array([param.minimum for param in self.parameters], dtype=np.float64)
        highs = np.array([param.maximum for param in self.parameters], dtype=np.float64)
        return lows + unit * (highs - lows)

    def scipy_prior_distributions(self):
        from scipy.stats import uniform

        return [uniform(loc=param.minimum, scale=param.maximum - param.minimum) for param in self.parameters]

    def pocomc_prior(self):
        import pocomc as pc

        return pc.Prior(self.scipy_prior_distributions())

    def _satisfies_parameter_constraints(self, theta: np.ndarray) -> bool:
        return all(
            theta[lower_index] <= theta[upper_index]
            for lower_index, upper_index in self._parameter_constraint_indices
        )

    def _evaluate(self, theta: ArrayLike) -> _TheoryEvaluation:
        theta = self._theta(theta)
        segments = []
        densities: dict[DensityKey, float] = {}
        result_cache: dict[tuple[str, Path, str], Any] = {}
        weight_cache: dict[tuple[str, str, str], Any] = {}
        multipole_cache: dict[int, dict[int, np.ndarray]] = {}
        for spec in self.observables:
            cache_key = (spec.tracer, spec.paircount_path, spec.hod_model)
            stellar_spec = self.stellar_mass_specs.get(spec.tracer)
            if cache_key not in result_cache:
                tabulator = self._tabulator_for_spec(spec)
                weight_key = (
                    spec.tracer,
                    spec.hod_model,
                    tabulator.weight_grid_key,
                )
                if weight_key not in weight_cache:
                    params = self.params_for_tracer(theta, spec.tracer)
                    if stellar_spec is None:
                        weight_cache[weight_key] = tabulator.hod_weights(
                            params,
                            hod_model=spec.hod_model,
                        )
                    else:
                        weight_cache[weight_key] = tabulator.stellar_mass_hod_weights(
                            params,
                            split_method=stellar_spec.split_method,
                            logmstar_edges=stellar_spec.logmstar_edges,
                            redshift_weights=stellar_spec.redshift_weights,
                            hod_model=spec.hod_model,
                        )
                if stellar_spec is None:
                    result_cache[cache_key] = tabulator.correlation_from_weights(
                        weight_cache[weight_key]
                    )
                else:
                    result_cache[cache_key] = tabulator.stellar_mass_correlations_from_weights(
                        weight_cache[weight_key],
                        split_method=stellar_spec.split_method,
                        logmstar_edges=stellar_spec.logmstar_edges,
                        redshift_weights=stellar_spec.redshift_weights,
                    )
            cached = result_cache[cache_key]
            if stellar_spec is None:
                result = cached
                densities.setdefault((spec.tracer, None), float(result.number_density))
            else:
                if spec.split_index is None:
                    raise RuntimeError(f"Missing split index for observable {spec.key!r}.")
                result = cached[spec.split_index]
                for sample, split_index in stellar_spec.samples.items():
                    densities.setdefault(
                        (spec.tracer, sample),
                        float(cached[split_index].number_density),
                    )
            if spec.statistic in _SMU_STATISTICS:
                result_key = id(result)
                if result_key not in multipole_cache:
                    multipole_cache[result_key] = smu_multipoles(
                        result,
                        ells=(0, 2),
                    )
                values = np.asarray(
                    multipole_cache[result_key][
                        0 if spec.statistic == "xi0" else 2
                    ],
                    dtype=np.float64,
                ).reshape(-1)
                segments.append(
                    values[_selection_indices(values.size, spec.selection)]
                )
            else:
                segments.append(_extract_observable(result, spec))

        hmf_tracers = {
            constraint.tracer
            for constraint in self.density_constraints
            if constraint.source == "hmf"
        }
        for tracer in hmf_tracers:
            if tracer not in self.hmfs:
                raise KeyError(f"No HMF loaded for tracer {tracer!r}.")
            params = self.params_for_tracer(theta, tracer)
            stellar_spec = self.stellar_mass_specs.get(tracer)
            if stellar_spec is None:
                densities[(tracer, None)] = hod_number_density(
                    self.hmfs[tracer],
                    params,
                    hod_model=self.hod_model_for_tracer(tracer),
                )
            else:
                values = stellar_mass_hod_number_densities(
                    self.hmfs[tracer],
                    params,
                    split_method=stellar_spec.split_method,
                    logmstar_edges=stellar_spec.logmstar_edges,
                    redshift_weights=stellar_spec.redshift_weights,
                    hod_model=self.hod_model_for_tracer(tracer),
                )
                for sample, split_index in stellar_spec.samples.items():
                    densities[(tracer, sample)] = float(values[split_index])

        theory = np.concatenate(segments) if segments else np.array([], dtype=np.float64)
        return _TheoryEvaluation(theory_vector=theory, number_densities=densities)

    def _density_loglike_from_densities(
        self,
        densities: Mapping[DensityKey, float],
    ) -> float:
        total = 0.0
        for constraint in self.density_constraints:
            if constraint.key not in densities:
                raise KeyError(
                    f"No observable or HMF density is configured for sample "
                    f"{constraint.label!r}."
                )
            term = constraint.loglike(float(densities[constraint.key]))
            if not np.isfinite(term):
                return -np.inf
            total += term
        return float(total)

    def hod_model_for_tracer(self, tracer: str) -> str:
        try:
            return self.hod_models_by_tracer[str(tracer)]
        except KeyError as exc:
            raise KeyError(f"No HOD model is configured for tracer {tracer!r}.") from exc

    def _theta(self, theta: ArrayLike) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.size != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} parameters, got {theta.size}.")
        return theta

    def _tabulator_for_spec(self, spec: ObservableSpec) -> HODClusteringTabulator:
        if spec.tabulator_key in self.tabulators:
            return self.tabulators[spec.tabulator_key]
        if spec.paircount_path in self.tabulators:
            return self.tabulators[spec.paircount_path]
        raise KeyError(f"No tabulator loaded for {spec.paircount_path}.")


def load_fitting_problem_from_config(
    path2config: str | Path,
    *,
    validate: bool = True,
) -> HODFittingProblem:
    return HODFittingProblem.from_config(path2config, validate=validate)


def load_stellar_mass_fitting_problem_from_config(
    path2config: str | Path,
    *,
    validate: bool = True,
) -> HODFittingProblem:
    """Load a fit in which every observable selects a stellar-mass sample."""

    problem = HODFittingProblem.from_config(path2config, validate=validate)
    if not problem.stellar_mass_specs:
        raise KeyError(
            "Configure fit.theory.tracers.<TRACER>.stellar_mass.samples."
        )
    scalar = [spec.key for spec in problem.observables if spec.sample is None]
    if scalar:
        raise ValueError(
            "The stellar-mass fitting runner requires sample on every observable; "
            f"scalar observables are {scalar}."
        )
    return problem


def _extract_observable(result, spec: ObservableSpec) -> np.ndarray:
    statistic = spec.statistic.lower()
    if statistic == "wp":
        if result.paircounts.clustering != "rppi":
            raise ValueError("wp requires rppi paircounts.")
        values = np.asarray(projected_wp(result), dtype=np.float64).reshape(-1)
    elif statistic == "xi0":
        if result.paircounts.clustering != "smu":
            raise ValueError("xi0 requires smu paircounts.")
        values = np.asarray(smu_multipoles(result, ells=(0,))[0], dtype=np.float64).reshape(-1)
    elif statistic == "xi2":
        if result.paircounts.clustering != "smu":
            raise ValueError("xi2 requires smu paircounts.")
        values = np.asarray(smu_multipoles(result, ells=(2,))[2], dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"Unknown observable statistic {spec.statistic!r}.")
    return values[_selection_indices(values.size, spec.selection)]


def _parse_tracers(fit_config: Mapping[str, Any]) -> tuple[str, ...]:
    tracers = fit_config.get("tracers")
    if tracers is None:
        data_tracers = fit_config.get("data", {}).get("tracers", {})
        theory_tracers = fit_config.get("theory", {}).get("tracers", {})
        tracers = list(data_tracers or theory_tracers or {"LRG": {}})
    if isinstance(tracers, str):
        tracers = [tracers]
    out = tuple(str(item) for item in tracers)
    if not out:
        raise ValueError("fit.tracers must contain at least one tracer.")
    return out


def _parse_stellar_mass_specs(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    tracers: Sequence[str],
) -> dict[str, StellarMassFitSpec]:
    global_config = dict((config.get("hod") or {}).get("stellar_mass") or {})
    specs: dict[str, StellarMassFitSpec] = {}
    for tracer in tracers:
        local = _tracer_theory_config(fit_config, tracer).get("stellar_mass")
        if local is None:
            continue
        local = dict(local)
        samples_config = _required(local, "samples")
        if not isinstance(samples_config, Mapping) or not samples_config:
            raise ValueError(
                f"fit.theory.tracers.{tracer}.stellar_mass.samples must be a non-empty mapping."
            )
        selection = dict(global_config)
        selection.update({key: value for key, value in local.items() if key != "samples"})
        normalized_selection = StellarMassSelection.from_values(
            _required(selection, "split_method"),
            _required(selection, "logmstar_edges"),
            selection.get("redshift_weights"),
        )
        n_splits = normalized_selection.n_splits
        samples: dict[str, int] = {}
        for raw_name, raw_index in samples_config.items():
            name = str(raw_name)
            if not name or "." in name or any(char.isspace() for char in name):
                raise ValueError(
                    f"Stellar-mass sample names must be non-empty and contain no dots or whitespace; got {name!r}."
                )
            if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
                raw_index, (int, np.integer)
            ):
                raise TypeError(
                    f"Split index for stellar-mass sample {tracer}.{name} must be an integer."
                )
            index = int(raw_index)
            if not 0 <= index < n_splits:
                raise ValueError(
                    f"Split index {index} for stellar-mass sample {tracer}.{name} "
                    f"is outside [0, {n_splits - 1}]."
                )
            samples[name] = index
        if len(set(samples.values())) != len(samples):
            raise ValueError(
                f"Stellar-mass samples for tracer {tracer!r} must map to unique split indices."
            )
        specs[tracer] = StellarMassFitSpec(
            tracer=tracer,
            selection=normalized_selection,
            samples=samples,
        )
    return specs


def _parse_observable_specs(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    tracers: Sequence[str],
    stellar_mass_specs: Mapping[str, StellarMassFitSpec],
    hod_models_by_tracer: Mapping[str, str],
) -> tuple[ObservableSpec, ...]:
    specs = []
    for item in fit_config.get("observables", []):
        obs = _normalize_observable_item(item, tracers)
        statistic = str(obs.get("statistic")).lower()
        if statistic not in _SUPPORTED_STATISTICS:
            raise ValueError(
                f"Unsupported observable statistic {statistic!r}; use wp, xi0, or xi2."
            )
        tracer = str(obs.get("tracer"))
        if tracer not in tracers:
            raise ValueError(
                f"Observable references tracer {tracer!r}, which is not in fit.tracers."
            )
        if "hod_model" in obs:
            raise ValueError(
                "Set one HOD model per tracer under "
                f"fit.theory.tracers.{tracer}.hod_model or hod.model; "
                "observable-level hod_model is not supported."
            )
        stellar_spec = stellar_mass_specs.get(tracer)
        raw_sample = obs.get("sample")
        if stellar_spec is None:
            if raw_sample is not None:
                raise ValueError(
                    f"Observable for scalar tracer {tracer!r} must not set sample."
                )
            sample = None
            split_index = None
        else:
            if raw_sample is None:
                raise KeyError(
                    f"Set sample on every observable for stellar-mass tracer {tracer!r}."
                )
            sample = str(raw_sample)
            if sample not in stellar_spec.samples:
                raise KeyError(
                    f"Unknown stellar-mass sample {tracer}.{sample}; configured samples are "
                    f"{tuple(stellar_spec.samples)}."
                )
            split_index = stellar_spec.samples[sample]
        clustering = "rppi" if statistic == "wp" else "smu"
        paircount_path = _observable_paircount_path(
            config, fit_config, tracer, clustering, obs
        )
        key_parts = [tracer]
        if sample is not None:
            key_parts.append(sample)
        key_parts.append(statistic)
        specs.append(
            ObservableSpec(
                name=str(obs.get("name", ".".join(key_parts))),
                tracer=tracer,
                statistic=statistic,
                clustering=clustering,
                paircount_path=paircount_path,
                hod_model=hod_models_by_tracer[tracer],
                sample=sample,
                split_index=split_index,
                selection=obs.get("slice", obs.get("selection")),
            )
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Observable names must be unique; got {names}.")
    return tuple(specs)


def _normalize_observable_item(item: Any, tracers: Sequence[str]) -> dict[str, Any]:
    if isinstance(item, str):
        item = {"statistic": item}
    else:
        item = dict(item)
    if "statistic" not in item:
        raise KeyError("Each fit.observable entry must define statistic.")
    if "tracer" not in item:
        if len(tracers) != 1:
            raise KeyError("Set tracer for each observable when fitting multiple tracers.")
        item["tracer"] = tracers[0]
    return item


def _observable_paircount_path(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    tracer: str,
    clustering: str,
    obs: Mapping[str, Any],
) -> Path:
    theory = _tracer_theory_config(fit_config, tracer)
    paircount_config = dict(theory.get("paircounts", {}).get(clustering, {}) or {})
    if obs.get("paircount_path") is not None:
        paircount_config["path"] = obs["paircount_path"]
    return _resolve_paircount_path(config, paircount_config, clustering=clustering)


def _tracer_theory_config(fit_config: Mapping[str, Any], tracer: str) -> Mapping[str, Any]:
    theory = fit_config.get("theory", {})
    return theory.get("tracers", {}).get(tracer, {})


def _load_observable_tabulators(observables: Sequence[ObservableSpec]) -> dict[Path, HODClusteringTabulator]:
    tabulators: dict[Path, HODClusteringTabulator] = {}
    for spec in observables:
        if spec.tabulator_key not in tabulators:
            tabulators[spec.tabulator_key] = HODClusteringTabulator.from_paircount_file(spec.paircount_path)
    return tabulators


def _load_observable_data(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    observables: Sequence[ObservableSpec],
    tabulators: Mapping[Path, HODClusteringTabulator],
) -> tuple[ObservableDataSegment, ...]:
    data_cache: dict[tuple[str, str | None, str, str, str, str], np.ndarray] = {}
    segments = []
    for spec in observables:
        raw_values = _load_statistic_data(config, fit_config, spec, data_cache)
        theory_bins = _observable_theory_bin_count(spec, tabulators)
        indices = _selection_indices(raw_values.size, spec.selection)
        _validate_selection(spec, raw_values.size, indices, theory_bins)
        segments.append(
            ObservableDataSegment(
                spec=spec,
                values=raw_values[indices],
                selected_indices=indices,
                full_size=int(raw_values.size),
            )
        )
    return tuple(segments)


def _load_statistic_data(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    spec: ObservableSpec,
    cache: dict[tuple[str, str | None, str, str, str, str], np.ndarray],
) -> np.ndarray:
    tracer_data = _tracer_data_config(fit_config, spec.tracer)
    sample_data = _sample_data_config(tracer_data, spec)
    if spec.statistic == "wp":
        wp_config = dict(_required(sample_data, "wp"))
        path = _format_config_path(_required(wp_config, "path"), config)
        column = wp_config.get("column", wp_config.get("wp_column", 1))
        key = (
            spec.tracer,
            spec.sample,
            "wp",
            str(path),
            str(wp_config.get("key")),
            repr(column),
        )
        if key not in cache:
            cache[key] = _load_vector(
                path,
                key=wp_config.get("key"),
                usecols=column,
            )
        return cache[key]

    xi02_config = dict(_required(sample_data, "xi02"))
    path = _format_config_path(_required(xi02_config, "path"), config)
    column_key = "xi0_column" if spec.statistic == "xi0" else "xi2_column"
    column = xi02_config.get(column_key)
    if column is None:
        prefix = f"fit.data.tracers.{spec.tracer}"
        if spec.sample is not None:
            prefix += f".samples.{spec.sample}"
        raise KeyError(f"Set {prefix}.xi02.{column_key}.")
    key = (
        spec.tracer,
        spec.sample,
        spec.statistic,
        str(path),
        str(xi02_config.get("key")),
        repr(column),
    )
    if key not in cache:
        cache[key] = _load_vector(
            path,
            key=xi02_config.get("key"),
            usecols=column,
        )
    return cache[key]


def _tracer_data_config(fit_config: Mapping[str, Any], tracer: str) -> Mapping[str, Any]:
    tracers = fit_config.get("data", {}).get("tracers", {})
    if tracer not in tracers:
        raise KeyError(f"Set fit.data.tracers.{tracer}.")
    return tracers[tracer]


def _sample_data_config(
    tracer_data: Mapping[str, Any],
    spec: ObservableSpec,
) -> Mapping[str, Any]:
    if spec.sample is None:
        return tracer_data
    samples = tracer_data.get("samples")
    if not isinstance(samples, Mapping) or spec.sample not in samples:
        raise KeyError(
            f"Set fit.data.tracers.{spec.tracer}.samples.{spec.sample}."
        )
    sample_data = samples[spec.sample]
    if not isinstance(sample_data, Mapping):
        raise TypeError(
            f"fit.data.tracers.{spec.tracer}.samples.{spec.sample} must be a mapping."
        )
    return sample_data


def _observable_theory_bin_count(
    spec: ObservableSpec,
    tabulators: Mapping[Path, HODClusteringTabulator],
) -> int:
    tabulator = tabulators.get(spec.tabulator_key, tabulators.get(spec.paircount_path))
    if tabulator is None:
        raise KeyError(f"No tabulator loaded for {spec.paircount_path}.")
    bins = tabulator.paircounts.bins
    if spec.statistic == "wp":
        if "rp_edges" not in bins:
            raise ValueError(f"Paircount file {spec.paircount_path} is missing bins/rp_edges for wp.")
        return int(len(bins["rp_edges"]) - 1)
    if "s_edges" not in bins:
        raise ValueError(f"Paircount file {spec.paircount_path} is missing bins/s_edges for {spec.statistic}.")
    return int(len(bins["s_edges"]) - 1)


def _validate_selection(
    spec: ObservableSpec,
    data_size: int,
    indices: np.ndarray,
    theory_bins: int,
) -> None:
    if indices.size == 0:
        raise ValueError(f"Observable {spec.key} selection is empty.")
    if np.any(indices < 0) or np.any(indices >= data_size):
        raise ValueError(f"Observable {spec.key} selection is outside data length {data_size}.")
    if np.any(indices >= theory_bins):
        raise ValueError(
            f"Observable {spec.key} selection reaches bin {int(np.max(indices))}, "
            f"but the paircount table only has {theory_bins} bins."
        )
    if spec.selection is None and data_size != theory_bins:
        raise ValueError(
            f"Observable {spec.key} has data length {data_size}, but the paircount table has "
            f"{theory_bins} bins. Set an observable slice to align them."
        )


def _load_fit_covariance(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    segments: Sequence[ObservableDataSegment],
) -> np.ndarray:
    covariance_config = fit_config.get("covariance", {})
    mode = str(covariance_config.get("mode", "joint")).lower()
    if mode == "joint":
        joint = covariance_config.get("joint", covariance_config)
        path = _format_config_path(_required(joint, "path"), config)
        covariance = _load_matrix(path, key=joint.get("key"))
        selected_size = sum(segment.size for segment in segments)
        if covariance.shape != (selected_size, selected_size):
            raise ValueError(
                f"Joint covariance shape {covariance.shape} does not match data vector length {selected_size}."
            )
        return covariance
    if mode != "block":
        raise ValueError("fit.covariance.mode must be 'joint' or 'block'.")

    blocks = covariance_config.get("blocks")
    if not blocks:
        raise KeyError("Set fit.covariance.blocks when covariance.mode is block.")
    covered = []
    for block in blocks:
        labels = _as_string_list(_required(block, "observables"))
        covered.extend(_segment_by_label(label, segments) for label in labels)
    if [id(item) for item in covered] != [id(item) for item in segments]:
        covered_names = [item.spec.key for item in covered]
        expected_names = [item.spec.key for item in segments]
        raise ValueError(
            "Covariance blocks must cover every fitted observable exactly once "
            f"in fit.observables order; got {covered_names}, expected "
            f"{expected_names}."
        )
    return _block_diag([_load_covariance_block(config, block, segments) for block in blocks])


def _load_covariance_block(
    config: Mapping[str, Any],
    block_config: Mapping[str, Any],
    segments: Sequence[ObservableDataSegment],
) -> np.ndarray:
    labels = _as_string_list(_required(block_config, "observables"))
    block_segments = [_segment_by_label(label, segments) for label in labels]
    if list(block_segments) != _ordered_subset(segments, block_segments):
        raise ValueError(
            f"Covariance block observables {labels} must follow fit.observables order."
        )
    covariance = _load_matrix(_format_config_path(_required(block_config, "path"), config), key=block_config.get("key"))
    selected_size = sum(segment.size for segment in block_segments)
    full_size = sum(segment.full_size for segment in block_segments)
    if covariance.shape == (selected_size, selected_size):
        return covariance
    if covariance.shape == (full_size, full_size):
        indices = _block_selected_indices(block_segments)
        return covariance[np.ix_(indices, indices)]
    raise ValueError(
        f"Covariance block {labels} has shape {covariance.shape}; expected "
        f"({selected_size}, {selected_size}) or ({full_size}, {full_size})."
    )


def _segment_by_label(label: str, segments: Sequence[ObservableDataSegment]) -> ObservableDataSegment:
    matches = [segment for segment in segments if label in segment.labels]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No observable segment matches covariance label {label!r}.")
    raise ValueError(
        f"Covariance label {label!r} is ambiguous; use the full sample key "
        "or a unique observable name."
    )


def _ordered_subset(
    segments: Sequence[ObservableDataSegment],
    subset: Sequence[ObservableDataSegment],
) -> list[ObservableDataSegment]:
    wanted = set(id(segment) for segment in subset)
    return [segment for segment in segments if id(segment) in wanted]


def _block_selected_indices(segments: Sequence[ObservableDataSegment]) -> np.ndarray:
    indices = []
    offset = 0
    for segment in segments:
        indices.extend((offset + segment.selected_indices).tolist())
        offset += segment.full_size
    return np.asarray(indices, dtype=int)


def _parse_density_constraints(
    fit_config: Mapping[str, Any],
    tracers: Sequence[str],
    stellar_mass_specs: Mapping[str, StellarMassFitSpec],
) -> tuple[NumberDensityConstraint, ...]:
    constraints = []
    data_tracers = fit_config.get("data", {}).get("tracers", {})
    for tracer in tracers:
        tracer_data = data_tracers.get(tracer, {})
        stellar_spec = stellar_mass_specs.get(tracer)
        if stellar_spec is None:
            entries = ((None, tracer_data.get("number_density")),)
        else:
            samples_data = tracer_data.get("samples", {})
            entries = tuple(
                (
                    sample,
                    (samples_data.get(sample) or {}).get("number_density"),
                )
                for sample in stellar_spec.samples
            )
        for sample, density_config in entries:
            if not density_config:
                continue
            mode = str(density_config.get("mode", "minimum")).lower()
            if mode == "none":
                continue
            source = str(density_config.get("source", "paircounts")).lower()
            label = tracer if sample is None else f"{tracer}.{sample}"
            if source not in {"paircounts", "hmf"}:
                raise ValueError(
                    f"number_density.source must be paircounts or hmf for sample {label!r}."
                )
            value = float(_required(density_config, "value"))
            error = density_config.get("error")
            constraints.append(
                NumberDensityConstraint(
                    tracer=tracer,
                    sample=sample,
                    value=value,
                    mode=mode,
                    error=None if error is None else float(error),
                    source=source,
                )
            )
    return tuple(constraints)


def _load_fit_hmfs(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    density_constraints: Sequence[NumberDensityConstraint],
) -> dict[str, HaloMassFunction]:
    hmfs: dict[str, HaloMassFunction] = {}
    for tracer in sorted({item.tracer for item in density_constraints if item.source == "hmf"}):
        hmfs[tracer] = read_hmf(_resolve_hmf_path(config, fit_config, tracer))
    return hmfs


def _resolve_hmf_path(config: Mapping[str, Any], fit_config: Mapping[str, Any], tracer: str) -> Path:
    hmf_config = dict(config.get("hmf", {}))
    tracer_hmf = _tracer_theory_config(fit_config, tracer).get("hmf", {})
    hmf_config.update(tracer_hmf)
    if hmf_config.get("path") is not None:
        return validate_hmf_file_for_config(
            _format_config_path(hmf_config["path"], config),
            n_bins=int(hmf_config.get("n_bins", 512)),
            logm_edges=hmf_config.get("logm_edges"),
            logm_min=hmf_config.get("logm_min"),
            logm_max=hmf_config.get("logm_max"),
        )
    pair_params = config.get("paircounts", {})
    prepare_params = config.get("prepare_profiles", {})
    output_value = hmf_config.get("output_dir", hmf_config.get("out_dir"))
    if output_value is None:
        raise KeyError(
            f"Set hmf.output_dir, hmf.path, or fit.theory.tracers.{tracer}.hmf.path for HMF density constraints."
        )
    output_dir = _format_sim_output_dir(output_value, config)
    return find_hmf_file(
        output_dir,
        file_tag=_first_not_none(hmf_config.get("file_tag"), pair_params.get("file_tag")),
        seed=_first_not_none(hmf_config.get("seed"), pair_params.get("seed"), prepare_params.get("seed")),
        n_bins=int(hmf_config.get("n_bins", 512)),
        logm_edges=hmf_config.get("logm_edges"),
        logm_min=hmf_config.get("logm_min"),
        logm_max=hmf_config.get("logm_max"),
    )


def _parse_fixed_params(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    tracers: Sequence[str],
) -> dict[str, dict[str, Any]]:
    out = {tracer: dict(config.get("hod", {}).get("params", {})) for tracer in tracers}
    theory_tracers = fit_config.get("theory", {}).get("tracers", {})
    for tracer in tracers:
        out[tracer].update(theory_tracers.get(tracer, {}).get("fixed_params", {}))
    return out


def _normalize_fixed_params(
    fixed_params: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
    tracers: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not tracers:
        return {"default": dict(fixed_params)}
    if all(isinstance(value, Mapping) for value in fixed_params.values()):
        return {tracer: dict(fixed_params.get(tracer, {})) for tracer in tracers}
    return {tracer: dict(fixed_params) for tracer in tracers}


def _infer_tracers(observables: Sequence[ObservableSpec]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(spec.tracer for spec in observables))


def _normalize_tracer_hod_models(
    observables: Sequence[ObservableSpec],
    tracers: Sequence[str],
    configured: Mapping[str, str] | None,
) -> dict[str, str]:
    if configured is not None:
        models = {str(key): str(value) for key, value in configured.items()}
    else:
        models = {}
        for tracer in tracers:
            values = {
                spec.hod_model for spec in observables if spec.tracer == tracer
            }
            if len(values) != 1:
                raise ValueError(
                    f"Tracer {tracer!r} must use exactly one HOD model; got "
                    f"{sorted(values)}."
                )
            models[tracer] = values.pop()
    missing = [tracer for tracer in tracers if tracer not in models]
    if missing:
        raise KeyError(f"No HOD model configured for tracer(s) {missing}.")
    for spec in observables:
        if spec.hod_model != models[spec.tracer]:
            raise ValueError(
                f"Observable {spec.name!r} uses HOD model {spec.hod_model!r}, "
                f"but tracer {spec.tracer!r} uses {models[spec.tracer]!r}."
            )
    return {tracer: models[tracer] for tracer in tracers}


def _validate_stellar_model_contract(
    tracers: Sequence[str],
    stellar_mass_specs: Mapping[str, StellarMassFitSpec],
    hod_models_by_tracer: Mapping[str, str],
) -> None:
    for tracer in tracers:
        model = hod_models_by_tracer[tracer]
        supports_splits = hod_model_supports_splits(model)
        has_selection = tracer in stellar_mass_specs
        if has_selection and not supports_splits:
            raise ValueError(
                f"Tracer {tracer!r} configures stellar-mass samples, but HOD "
                f"model {model!r} does not support split occupations."
            )
        if supports_splits and not has_selection:
            raise ValueError(
                f"Tracer {tracer!r} uses split-capable HOD model {model!r}; "
                "configure fit.theory.tracers.<TRACER>.stellar_mass.samples."
            )


def _validate_parameter_ownership(parameters: Sequence[FitParameter], tracers: Sequence[str]) -> None:
    tracer_set = set(tracers)
    invalid_shared = []
    unknown_tracers = []
    malformed = []
    for param in parameters:
        if "." in param.name:
            param_tracer, param_name = param.name.split(".", 1)
            if not param_tracer or not param_name:
                malformed.append(param.name)
            elif param_tracer not in tracer_set:
                unknown_tracers.append(param.name)
            if param.shared:
                invalid_shared.append(param.name)
    if malformed:
        raise ValueError(f"Malformed tracer-qualified fit parameter names: {malformed}.")
    if unknown_tracers:
        raise ValueError(
            f"Fit parameters reference unknown tracers {unknown_tracers}; configured tracers are {tuple(tracers)}."
        )
    if invalid_shared:
        raise ValueError(f"Tracer-qualified parameters cannot also set shared: true: {invalid_shared}.")
    if len(tracers) <= 1:
        return
    unqualified = [param.name for param in parameters if "." not in param.name and not param.shared]
    if unqualified:
        raise ValueError(
            f"Multi-tracer fits require tracer-qualified parameters or shared: true. "
            f"Unqualified non-shared parameters: {unqualified}."
        )


def _resolve_paircount_path(config: Mapping[str, Any], path_config: Mapping[str, Any], *, clustering: str) -> Path:
    from .paircounts import resolve_paircount_path_from_config

    job_name = path_config.get("job")
    if job_name is None and config.get("paircounts", {}).get("jobs"):
        job_name = "clustering"
    return resolve_paircount_path_from_config(
        config,
        clustering=clustering,
        path_config=path_config,
        job_name=job_name,
    )


def _parse_parameters(config: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[FitParameter, ...]:
    parameters: list[FitParameter] = []
    if isinstance(config, Mapping):
        items = config.items()
    else:
        items = []
        for item in config:
            if "name" not in item:
                raise KeyError("Each fit parameter entry must define name.")
            items.append((item["name"], item))

    for name, spec in items:
        shared = False
        if isinstance(spec, Mapping):
            minimum = float(_required(spec, "min"))
            maximum = float(_required(spec, "max"))
            initial = float(spec.get("initial", 0.5 * (minimum + maximum)))
            shared = bool(spec.get("shared", False))
        else:
            values = list(spec)
            if len(values) == 2:
                minimum, maximum = map(float, values)
                initial = 0.5 * (minimum + maximum)
            elif len(values) == 3:
                initial, minimum, maximum = map(float, values)
            else:
                raise ValueError(f"Parameter {name!r} must be a mapping, [min, max], or [initial, min, max].")
        if not minimum < maximum:
            raise ValueError(f"Parameter {name!r} has invalid bounds [{minimum}, {maximum}].")
        if not minimum <= initial <= maximum:
            raise ValueError(f"Initial value for {name!r} lies outside its prior bounds.")
        parameters.append(FitParameter(str(name), initial, minimum, maximum, shared=shared))
    return tuple(parameters)


def _parse_parameter_constraints(
    config: Sequence[Mapping[str, Any]] | None,
    parameters: Sequence[FitParameter],
) -> tuple[OrderedParameterConstraint, ...]:
    if config is None:
        return ()
    if isinstance(config, (str, bytes, Mapping)) or not isinstance(config, Sequence):
        raise TypeError("fit.parameter_constraints must be a list of mappings.")

    parameter_names = {parameter.name for parameter in parameters}
    constraints = []
    for item in config:
        if not isinstance(item, Mapping):
            raise TypeError("Each fit.parameter_constraints entry must be a mapping.")
        lower = str(_required(item, "lower"))
        upper = str(_required(item, "upper"))
        if lower == upper:
            raise ValueError("An ordered parameter constraint requires two distinct names.")
        missing = sorted({lower, upper} - parameter_names)
        if missing:
            raise ValueError(
                "Ordered parameter constraint references unknown free parameters: "
                f"{missing}."
            )
        constraints.append(OrderedParameterConstraint(lower=lower, upper=upper))
    return tuple(constraints)


def _load_vector(path: str | Path, *, key: str | None = None, usecols: Any = None) -> np.ndarray:
    array = _load_array(path, key=key, usecols=usecols)
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _load_matrix(path: str | Path, *, key: str | None = None) -> np.ndarray:
    array = _load_array(path, key=key, usecols=None)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D matrix in {path}; got shape {array.shape}.")
    return array


def _load_array(path: str | Path, *, key: str | None = None, usecols: Any = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if key is None:
            if len(data.files) != 1:
                raise KeyError(f"Set a key for {path}; available keys are {data.files}.")
            key = data.files[0]
        return data[key]
    return np.loadtxt(path, usecols=_usecols(usecols))


def _usecols(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(int(part) for part in parts) if len(parts) != 1 else int(parts[0])
    if isinstance(value, Sequence):
        return tuple(int(item) for item in value)
    return int(value)


def _selection_indices(size: int, selection: Any) -> np.ndarray:
    if selection is None:
        return np.arange(size)
    if isinstance(selection, Mapping):
        if "indices" in selection:
            return np.asarray(selection["indices"], dtype=int)
        selection = selection.get("slice")
    if isinstance(selection, str):
        selection = [None if part == "" else int(part) for part in selection.split(":")]
    if isinstance(selection, Sequence):
        values = list(selection)
        if len(values) in (2, 3) and all(item is None or isinstance(item, (int, np.integer)) for item in values):
            start = values[0]
            stop = values[1]
            step = values[2] if len(values) == 3 else None
            return np.arange(size)[slice(start, stop, step)]
        return np.asarray(values, dtype=int)
    raise ValueError(f"Unsupported selection {selection!r}.")


def _block_diag(blocks: Sequence[np.ndarray]) -> np.ndarray:
    sizes = [block.shape[0] for block in blocks]
    out = np.zeros((sum(sizes), sum(sizes)), dtype=np.float64)
    start = 0
    for block, size in zip(blocks, sizes, strict=True):
        if block.shape != (size, size):
            raise ValueError(f"Covariance block must be square; got {block.shape}.")
        out[start : start + size, start : start + size] = block
        start += size
    return out


def _precision_scale(covariance_config: Mapping[str, Any], data_size: int) -> float:
    scale = float(covariance_config.get("precision_scale", 1.0))
    n_mocks = covariance_config.get("covariance_n_mocks", covariance_config.get("n_mocks"))
    if n_mocks is None:
        return scale
    n_mocks = int(n_mocks)
    if n_mocks <= data_size + 2:
        raise ValueError("Hartlap correction requires covariance_n_mocks > n_data + 2.")
    return scale * (n_mocks - data_size - 2.0) / (n_mocks - 1.0)


def _required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config or config[key] is None:
        raise KeyError(f"Missing required config key {key!r}.")
    return config[key]


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _z_directory(z_mock: float) -> str:
    return f"z{float(z_mock):.3f}"


def _format_config_path(value: str | Path, config: Mapping[str, Any]) -> Path:
    sim_params = config.get("sim_params", {})
    return Path(
        str(value).format(
            sim_name=str(sim_params.get("sim_name", "")),
            z=_z_directory(float(sim_params.get("z_mock", 0.0))),
            z_mock=float(sim_params.get("z_mock", 0.0)),
        )
    )


def _has_sim_format_fields(value: str) -> bool:
    return any(field in value for field in ("{sim_name", "{z", "{z_mock"))


def _format_sim_output_dir(value: str | Path, config: Mapping[str, Any]) -> Path:
    sim_params = config.get("sim_params", {})
    sim_name = str(sim_params.get("sim_name", ""))
    z_mock = float(sim_params.get("z_mock", 0.0))
    z_dir = _z_directory(z_mock)
    template = str(value)
    has_fields = _has_sim_format_fields(template)
    path = Path(template.format(sim_name=sim_name, z=z_dir, z_mock=z_mock))
    if has_fields or tuple(path.parts[-2:]) == (sim_name, z_dir):
        return path
    return path / sim_name / z_dir
