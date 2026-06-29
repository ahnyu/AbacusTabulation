"""Config-driven helpers for HOD fitting against tabulated clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .clustering import HODClusteringTabulator, projected_wp, smu_multipoles


ArrayLike = np.ndarray | Sequence[float]
_SUPPORTED_STATISTICS = {"wp", "xi0", "xi2"}
_SMU_STATISTICS = {"xi0", "xi2"}


@dataclass(frozen=True)
class FitParameter:
    """One scalar free parameter with a uniform top-hat prior."""

    name: str
    initial: float
    minimum: float
    maximum: float

    def contains(self, value: float) -> bool:
        return self.minimum <= float(value) <= self.maximum


@dataclass(frozen=True)
class ObservableSpec:
    """Description of one fitted theory-vector segment."""

    name: str
    tracer: str
    statistic: str
    clustering: str
    paircount_path: Path
    hod_model: str
    n_subbins: int
    selection: Any = None

    @property
    def key(self) -> str:
        return f"{self.tracer}.{self.statistic}"

    @property
    def tabulator_key(self) -> tuple[Path, int]:
        return (self.paircount_path, self.n_subbins)


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
    """One-sided number-density constraint for a tracer."""

    tracer: str
    value: float
    mode: str = "minimum"
    error: float | None = None

    def loglike(self, theory_density: float) -> float:
        mode = self.mode.lower()
        if mode == "none":
            return 0.0
        if theory_density >= self.value:
            return 0.0
        if mode == "minimum":
            return -np.inf
        if mode == "gaussian":
            if self.error is None or self.error <= 0.0:
                raise ValueError(
                    f"number_density.error must be positive for gaussian mode on tracer {self.tracer!r}."
                )
            diff = theory_density - self.value
            return float(-0.5 * diff * diff / (self.error * self.error))
        raise ValueError(f"Unknown number-density mode {self.mode!r} for tracer {self.tracer!r}.")


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
        if inversion == "pinv":
            inverse = np.linalg.pinv(covariance)
        elif inversion == "inv":
            inverse = np.linalg.inv(covariance)
        else:
            raise ValueError("covariance.inversion must be 'inv' or 'pinv'.")
        inverse = float(precision_scale) * inverse
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
        config: Mapping[str, Any] | None = None,
        fit_config: Mapping[str, Any] | None = None,
        tracers: Sequence[str] = (),
    ):
        self.data = data
        self.parameters = tuple(parameters)
        self.observables = tuple(observables)
        self.tabulators = dict(tabulators)
        self.density_constraints = tuple(density_constraints)
        self.config = dict(config or {})
        self.fit_config = dict(fit_config or {})
        self.tracers = tuple(str(item) for item in (tracers or _infer_tracers(self.observables)))
        self.fixed_params_by_tracer = _normalize_fixed_params(fixed_params, self.tracers)
        self.parameter_names = tuple(param.name for param in self.parameters)
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError(f"Duplicate fit parameter names: {self.parameter_names}")

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
        observables = _parse_observable_specs(config, fit_config, tracers)
        data_segments = _load_observable_data(config, fit_config, observables)
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

        tabulators: dict[tuple[Path, int], HODClusteringTabulator] = {}
        for spec in observables:
            if spec.tabulator_key not in tabulators:
                tabulators[spec.tabulator_key] = HODClusteringTabulator.from_paircount_file(
                    spec.paircount_path,
                    n_subbins=spec.n_subbins,
                )

        problem = cls(
            data=data,
            parameters=parameters,
            fixed_params=_parse_fixed_params(config, fit_config, tracers),
            observables=observables,
            tabulators=tabulators,
            density_constraints=_parse_density_constraints(fit_config, tracers),
            config=config,
            fit_config=fit_config,
            tracers=tracers,
        )
        if validate:
            theory = problem.theory_vector(problem.initial_vector())
            if theory.size != data.size:
                raise ValueError(
                    f"Initial theory vector has length {theory.size}; data vector has length {data.size}."
                )
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
        for param, value in zip(self.parameters, theta, strict=True):
            if "." in param.name:
                param_tracer, param_name = param.name.split(".", 1)
                if param_tracer == tracer:
                    params[param_name] = float(value)
            else:
                params[param.name] = float(value)
        return params

    def logprior(self, theta: ArrayLike) -> float:
        theta = self._theta(theta)
        for value, param in zip(theta, self.parameters, strict=True):
            if not param.contains(float(value)):
                return -np.inf
        return 0.0

    def theory_vector(self, theta: ArrayLike) -> np.ndarray:
        theta = self._theta(theta)
        segments = []
        result_cache: dict[tuple[str, Path, int, str], Any] = {}
        for spec in self.observables:
            cache_key = (spec.tracer, spec.paircount_path, spec.n_subbins, spec.hod_model)
            if cache_key not in result_cache:
                tabulator = self._tabulator_for_spec(spec)
                result_cache[cache_key] = tabulator.correlation(
                    self.params_for_tracer(theta, spec.tracer),
                    hod_model=spec.hod_model,
                )
            segments.append(_extract_observable(result_cache[cache_key], spec))
        return np.concatenate(segments) if segments else np.array([], dtype=np.float64)

    def theory_number_density(self, theta: ArrayLike, tracer: str) -> float:
        theta = self._theta(theta)
        for spec in self.observables:
            if spec.tracer == tracer:
                result = self._tabulator_for_spec(spec).correlation(
                    self.params_for_tracer(theta, tracer),
                    hod_model=spec.hod_model,
                )
                return float(result.number_density)
        raise KeyError(f"No observable is configured for tracer {tracer!r}; cannot compute number density.")

    def density_loglike(self, theta: ArrayLike) -> float:
        theta = self._theta(theta)
        total = 0.0
        densities: dict[str, float] = {}
        for constraint in self.density_constraints:
            if constraint.tracer not in densities:
                densities[constraint.tracer] = self.theory_number_density(theta, constraint.tracer)
            term = constraint.loglike(densities[constraint.tracer])
            if not np.isfinite(term):
                return -np.inf
            total += term
        return float(total)

    def loglike(self, theta: ArrayLike) -> float:
        theta = self._theta(theta)
        try:
            theory = self.theory_vector(theta)
            clustering_value = self.data.loglike(theory)
            density_value = self.density_loglike(theta)
        except Exception:
            return -np.inf
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


def load_fitting_problem_from_config(path2config: str | Path, *, validate: bool = True) -> HODFittingProblem:
    return HODFittingProblem.from_config(path2config, validate=validate)


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


def _parse_observable_specs(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    tracers: Sequence[str],
) -> tuple[ObservableSpec, ...]:
    specs = []
    for i, item in enumerate(fit_config.get("observables", [])):
        obs = _normalize_observable_item(item, tracers)
        statistic = str(obs.get("statistic")).lower()
        if statistic not in _SUPPORTED_STATISTICS:
            raise ValueError(f"Unsupported observable statistic {statistic!r}; use wp, xi0, or xi2.")
        tracer = str(obs.get("tracer"))
        clustering = "rppi" if statistic == "wp" else "smu"
        theory = _tracer_theory_config(fit_config, tracer)
        paircount_path = _observable_paircount_path(config, fit_config, tracer, clustering, obs)
        specs.append(
            ObservableSpec(
                name=str(obs.get("name", f"{tracer}.{statistic}")),
                tracer=tracer,
                statistic=statistic,
                clustering=clustering,
                paircount_path=paircount_path,
                hod_model=str(obs.get("hod_model", theory.get("hod_model", config.get("hod", {}).get("model", "lrg")))),
                n_subbins=int(obs.get("n_subbins", theory.get("n_subbins", config.get("hod", {}).get("n_subbins", 20)))),
                selection=obs.get("slice", obs.get("selection")),
            )
        )
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
    if obs.get("paircount_path") is not None:
        return _format_config_path(obs["paircount_path"], config)
    theory = _tracer_theory_config(fit_config, tracer)
    paircount_config = theory.get("paircounts", {}).get(clustering, {})
    if paircount_config.get("path") is not None:
        return _format_config_path(paircount_config["path"], config)
    return _resolve_paircount_path(config, paircount_config, clustering=clustering)


def _tracer_theory_config(fit_config: Mapping[str, Any], tracer: str) -> Mapping[str, Any]:
    theory = fit_config.get("theory", {})
    return theory.get("tracers", {}).get(tracer, {})


def _load_observable_data(
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    observables: Sequence[ObservableSpec],
) -> tuple[ObservableDataSegment, ...]:
    data_cache: dict[tuple[str, str, str], np.ndarray] = {}
    segments = []
    for spec in observables:
        raw_values = _load_statistic_data(config, fit_config, spec, data_cache)
        max_bins = _configured_bin_count(config, spec)
        indices = _selection_indices(raw_values.size, spec.selection)
        _validate_selection(spec, raw_values.size, indices, max_bins)
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
    cache: dict[tuple[str, str, str], np.ndarray],
) -> np.ndarray:
    tracer_data = _tracer_data_config(fit_config, spec.tracer)
    if spec.statistic == "wp":
        wp_config = dict(_required(tracer_data, "wp"))
        path = _format_config_path(_required(wp_config, "path"), config)
        column = wp_config.get("column", wp_config.get("wp_column", 1))
        key = (spec.tracer, "wp", str(path))
        if key not in cache:
            cache[key] = _load_vector(path, key=wp_config.get("key"), usecols=column)
        return cache[key]

    xi02_config = dict(_required(tracer_data, "xi02"))
    path = _format_config_path(_required(xi02_config, "path"), config)
    column_key = "xi0_column" if spec.statistic == "xi0" else "xi2_column"
    column = xi02_config.get(column_key)
    if column is None:
        raise KeyError(f"Set fit.data.tracers.{spec.tracer}.xi02.{column_key}.")
    key = (spec.tracer, spec.statistic, str(path))
    if key not in cache:
        cache[key] = _load_vector(path, key=xi02_config.get("key"), usecols=column)
    return cache[key]


def _tracer_data_config(fit_config: Mapping[str, Any], tracer: str) -> Mapping[str, Any]:
    tracers = fit_config.get("data", {}).get("tracers", {})
    if tracer not in tracers:
        raise KeyError(f"Set fit.data.tracers.{tracer}.")
    return tracers[tracer]


def _validate_selection(
    spec: ObservableSpec,
    full_size: int,
    indices: np.ndarray,
    configured_bins: int | None,
) -> None:
    if indices.size == 0:
        raise ValueError(f"Observable {spec.key} selection is empty.")
    if np.any(indices < 0) or np.any(indices >= full_size):
        raise ValueError(f"Observable {spec.key} selection is outside data length {full_size}.")
    if configured_bins is not None and indices.size > configured_bins:
        raise ValueError(
            f"Observable {spec.key} uses {indices.size} bins; configured maximum is {configured_bins}."
        )


def _configured_bin_count(config: Mapping[str, Any], spec: ObservableSpec) -> int | None:
    pair_params = config.get("paircounts", {})
    if spec.statistic == "wp":
        value = pair_params.get("rppi", {}).get("nrp_bins")
    else:
        value = pair_params.get("smu", {}).get("ns_bins")
    return None if value is None else int(value)


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
    raise ValueError(f"Covariance label {label!r} is ambiguous; use tracer.statistic or observable name.")


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
) -> tuple[NumberDensityConstraint, ...]:
    constraints = []
    for tracer in tracers:
        tracer_data = fit_config.get("data", {}).get("tracers", {}).get(tracer, {})
        config = tracer_data.get("number_density")
        if not config:
            continue
        mode = str(config.get("mode", "minimum")).lower()
        if mode == "none":
            continue
        value = float(_required(config, "value"))
        error = config.get("error")
        constraints.append(
            NumberDensityConstraint(
                tracer=tracer,
                value=value,
                mode=mode,
                error=None if error is None else float(error),
            )
        )
    return tuple(constraints)


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


def _resolve_paircount_path(config: Mapping[str, Any], path_config: Mapping[str, Any], *, clustering: str) -> Path:
    paths_params = config.get("paths", {})
    pair_params = config.get("paircounts", {})
    mass_params = pair_params.get("mass", {})
    output_value = path_config.get("dir", path_config.get("paircounts_dir", pair_params.get("output_dir", paths_params.get("paircounts_dir"))))
    if output_value is None:
        raise KeyError("Set paircounts.output_dir, paths.paircounts_dir, or theory.tracers.<tracer>.paircounts.<mode>.dir.")
    output_dir = _format_config_path(output_value, config)
    position_dataset = str(path_config.get("position_dataset", pair_params.get("position_dataset", "pos")))
    file_tag = path_config.get("file_tag", pair_params.get("file_tag"))
    nmass_bins = int(path_config.get("nmass_bins", mass_params.get("nmass_bins", pair_params.get("nmass_bins", 20))))
    return _find_paircount_file(
        output_dir,
        clustering=clustering,
        position_dataset=position_dataset,
        file_tag=file_tag,
        nmass_bins=nmass_bins,
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
        if isinstance(spec, Mapping):
            minimum = float(_required(spec, "min"))
            maximum = float(_required(spec, "max"))
            initial = float(spec.get("initial", 0.5 * (minimum + maximum)))
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
        parameters.append(FitParameter(str(name), initial, minimum, maximum))
    return tuple(parameters)


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


def _sanitize_filename_piece(value: object) -> str:
    clean = []
    for char in str(value):
        clean.append(char if char.isalnum() else "-")
    return "".join(clean).strip("-") or "none"


def _find_paircount_file(
    output_dir: Path,
    *,
    clustering: str,
    position_dataset: str,
    file_tag: str | None,
    nmass_bins: int,
) -> Path:
    pos = _sanitize_filename_piece(position_dataset)
    if file_tag is not None:
        tag = _sanitize_filename_piece(file_tag)
        path = output_dir / f"paircounts_{clustering}_{pos}_{tag}_m{nmass_bins}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = sorted(output_dir.glob(f"paircounts_{clustering}_{pos}_*_m{nmass_bins}.h5"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No paircount file matching paircounts_{clustering}_{pos}_*_m{nmass_bins}.h5 in {output_dir}."
        )
    raise ValueError(f"Found multiple matching paircount files in {output_dir}; set paircounts.file_tag.")
