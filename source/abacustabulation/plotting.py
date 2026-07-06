"""Plotting helpers for fitting outputs and HOD diagnostics.

This module intentionally keeps plotting separate from the production tabulation
and fitting code. Config helpers only resolve paths and load the data needed for
plots; plotting functions operate on already-loaded chains, bands, or HOD curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_config
from .fitting import HODFittingProblem
from .hod import evaluate_hod


DEFAULT_FIGSIZE_PER_PANEL = (8.0, 6.0)
DEFAULT_LABELSIZE = 20
DEFAULT_LEGENDSIZE = 20
DEFAULT_TICKSIZE = 18
DEFAULT_BAND_QUANTILES = (16.0, 84.0)
DEFAULT_DERIVED_PARAMETERS = (
    "linear_bias",
    "satellite_fraction",
    "log10_mh_cen_med",
    "log10_mh_sat_med",
)
DEFAULT_PARAMETER_LABELS = {
    "logMcut": r"\log M_{\rm cut}",
    "logM1": r"\log M_1",
    "sigma": r"\sigma",
    "alpha": r"\alpha",
    "kappa": r"\kappa",
    "a_c": r"a_{\rm c}",
    "a_s": r"a_{\rm s}",
    "Q": r"Q",
    "gamma": r"\gamma",
    "maxpdf": r"p_{\rm max}",
    "n_cen": r"n_{\rm cen}",
    "n_sat": r"n_{\rm sat}",
    "number_density": r"n_{\rm g}",
    "satellite_fraction": r"f_{\rm sat}",
    "log10_mh_cen_med": r"\log_{10} \widetilde{M}_{\rm h,cen}",
    "log10_mh_sat_med": r"\log_{10} \widetilde{M}_{\rm h,sat}",
    "mh_cen_med": r"\widetilde{M}_{\rm h,cen}",
    "mh_sat_med": r"\widetilde{M}_{\rm h,sat}",
    "linear_bias": r"b_{\rm lin}",
}


@dataclass(frozen=True)
class ChainSamples:
    """Weighted posterior samples read from ``*_chains.txt``."""

    samples: np.ndarray
    weights: np.ndarray
    parameter_names: tuple[str, ...]
    loglike: np.ndarray | None = None
    logprior: np.ndarray | None = None
    derived: np.ndarray | None = None
    derived_names: tuple[str, ...] = ()
    label: str | None = None
    path: Path | None = None

    @property
    def size(self) -> int:
        return int(self.samples.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.samples.shape[1])

    @property
    def nderived(self) -> int:
        return 0 if self.derived is None else int(self.derived.shape[1])

    @property
    def all_parameter_names(self) -> tuple[str, ...]:
        return (*self.parameter_names, *self.derived_names)

    @property
    def all_samples(self) -> np.ndarray:
        if self.derived is None or self.derived.size == 0:
            return self.samples
        return np.column_stack([self.samples, self.derived])

    def values(self, name: str) -> np.ndarray:
        return _chain_parameter_values(self, name)

    def normalized_weights(self) -> np.ndarray:
        return _normalized_weights(self.weights, self.size)


@dataclass(frozen=True)
class OptimizationResult:
    """One-row optimization result from ``*_optimum.txt``."""

    bestfit: np.ndarray
    parameter_names: tuple[str, ...]
    loglike: float | None = None
    logprior: float | None = None
    logposterior: float | None = None
    path: Path | None = None


@dataclass(frozen=True)
class ObservableFitBand:
    """Data and posterior theory band for one fitted observable."""

    name: str
    tracer: str
    statistic: str
    x: np.ndarray
    data: np.ndarray
    error: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    label: str | None = None


@dataclass(frozen=True)
class HODBand:
    """Posterior HOD occupation band on a log10 mass grid."""

    tracer: str
    logm: np.ndarray
    central_mean: np.ndarray
    central_lower: np.ndarray
    central_upper: np.ndarray
    satellite_mean: np.ndarray
    satellite_lower: np.ndarray
    satellite_upper: np.ndarray
    total_mean: np.ndarray
    total_lower: np.ndarray
    total_upper: np.ndarray
    label: str | None = None


@dataclass(frozen=True)
class ObservableDataPoints:
    """Observed data points for one fitted observable segment."""

    name: str
    tracer: str
    statistic: str
    x: np.ndarray
    values: np.ndarray
    error: np.ndarray


@dataclass(frozen=True)
class ParameterStats:
    """One-dimensional posterior summary for a sampled or derived parameter."""

    name: str
    getdist_name: str
    label: str
    mean: float
    lower: float
    upper: float
    err_minus: float
    err_plus: float
    std: float


@dataclass(frozen=True)
class FitPlotData:
    """Loaded fit outputs for MCMC diagnostics and plotting."""

    problem: HODFittingProblem
    chain: ChainSamples
    data_points: tuple[ObservableDataPoints, ...]
    getdist_sample: Any | None
    parameter_stats: dict[str, ParameterStats]
    label: str | None = None

    @classmethod
    def from_config(
        cls,
        path2config: str | Path,
        *,
        chain_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        prefix: str | None = None,
        label: str | None = None,
        prefer_derived: bool = True,
        validate: bool = False,
        labels: Mapping[str, str] | Sequence[str] | None = None,
        settings: Mapping[str, Any] | None = None,
        build_getdist: bool = True,
    ) -> "FitPlotData":
        problem = load_plotting_problem(path2config, validate=validate)
        path = chain_path or pocomc_chain_path_from_config(
            path2config,
            output_dir=output_dir,
            prefix=prefix,
            prefer_derived=prefer_derived,
        )
        chain = load_chain(path, label=label)
        return cls.from_problem_and_chain(
            problem,
            chain,
            labels=labels,
            settings=settings,
            build_getdist=build_getdist,
        )

    @classmethod
    def from_problem_and_chain(
        cls,
        problem: HODFittingProblem,
        chain: ChainSamples,
        *,
        labels: Mapping[str, str] | Sequence[str] | None = None,
        settings: Mapping[str, Any] | None = None,
        build_getdist: bool = True,
    ) -> "FitPlotData":
        gd_sample = _build_getdist_samples(chain, labels=labels, settings=settings) if build_getdist else None
        stats = (
            _parameter_stats_from_getdist(gd_sample, chain, labels)
            if gd_sample is not None
            else _parameter_stats_from_chain(chain, labels)
        )
        return cls(
            problem=problem,
            chain=chain,
            data_points=tuple(_observable_data_points(problem)),
            getdist_sample=gd_sample,
            parameter_stats=stats,
            label=chain.label,
        )

    @property
    def mcmc_samples(self) -> ChainSamples:
        return self.chain

    @property
    def samples(self) -> np.ndarray:
        return self.chain.all_samples

    @property
    def weights(self) -> np.ndarray:
        return self.chain.weights

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.chain.all_parameter_names

    @property
    def stats(self) -> dict[str, ParameterStats]:
        return self.parameter_stats

    def values(self, name: str) -> np.ndarray:
        return self.chain.values(name)

    def stat(self, name: str) -> ParameterStats:
        return self.parameter_stats[str(name)]

    def stats_table(self) -> dict[str, dict[str, float | str]]:
        return {
            name: {
                "label": stat.label,
                "getdist_name": stat.getdist_name,
                "mean": stat.mean,
                "lower": stat.lower,
                "upper": stat.upper,
                "err_minus": stat.err_minus,
                "err_plus": stat.err_plus,
                "std": stat.std,
            }
            for name, stat in self.parameter_stats.items()
        }

# -----------------------------------------------------------------------------
# Config path helpers


def fit_output_dir_from_config(path_or_config: str | Path | Mapping[str, Any], output_dir: str | Path | None = None) -> Path:
    """Return the configured fit-output directory."""

    config = _as_config(path_or_config)
    fit_output = config.get("fit", {}).get("output", {})
    value = output_dir if output_dir is not None else fit_output.get("output_dir", "fit_outputs")
    return _format_config_path(value, config)


def fit_output_prefix_from_config(path_or_config: str | Path | Mapping[str, Any], prefix: str | None = None) -> str:
    """Return the configured fit-output prefix."""

    if prefix is not None:
        return str(prefix)
    config = _as_config(path_or_config)
    return str(config.get("fit", {}).get("output", {}).get("prefix", "abacustab_fit"))


def pocomc_chain_path_from_config(
    path_or_config: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    prefer_derived: bool = True,
) -> Path:
    """Return the configured pocoMC chain path, preferring ``*_chains_derived.txt``."""

    out = fit_output_dir_from_config(path_or_config, output_dir=output_dir)
    name = fit_output_prefix_from_config(path_or_config, prefix=prefix)
    derived = out / f"{name}_chains_derived.txt"
    if prefer_derived and derived.exists():
        return derived
    return out / f"{name}_chains.txt"


def pocomc_derived_chain_path_from_config(
    path_or_config: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
) -> Path:
    """Return ``<output_dir>/<prefix>_chains_derived.txt`` from the fit config."""

    out = fit_output_dir_from_config(path_or_config, output_dir=output_dir)
    name = fit_output_prefix_from_config(path_or_config, prefix=prefix)
    return out / f"{name}_chains_derived.txt"


def optimization_result_path_from_config(
    path_or_config: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
) -> Path:
    """Return ``<output_dir>/<prefix>_optimum.txt`` from the fit config."""

    out = fit_output_dir_from_config(path_or_config, output_dir=output_dir)
    name = fit_output_prefix_from_config(path_or_config, prefix=prefix)
    return out / f"{name}_optimum.txt"


def theory_vector_path_from_config(
    path_or_config: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
) -> Path:
    """Return ``<output_dir>/<prefix>_theory_vector.txt`` from the fit config."""

    out = fit_output_dir_from_config(path_or_config, output_dir=output_dir)
    name = fit_output_prefix_from_config(path_or_config, prefix=prefix)
    return out / f"{name}_theory_vector.txt"


def load_plotting_problem(path2config: str | Path, *, validate: bool = False) -> HODFittingProblem:
    """Load the fitting problem used to evaluate theory curves for plots."""

    return HODFittingProblem.from_config(path2config, validate=validate)


def load_fit_plot_data(
    path2config: str | Path,
    *,
    chain_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    label: str | None = None,
    prefer_derived: bool = True,
    validate: bool = False,
    labels: Mapping[str, str] | Sequence[str] | None = None,
    settings: Mapping[str, Any] | None = None,
    build_getdist: bool = True,
) -> FitPlotData:
    """Load problem, MCMC chain, observable data, GetDist sample, and parameter stats."""

    return FitPlotData.from_config(
        path2config,
        chain_path=chain_path,
        output_dir=output_dir,
        prefix=prefix,
        label=label,
        prefer_derived=prefer_derived,
        validate=validate,
        labels=labels,
        settings=settings,
        build_getdist=build_getdist,
    )


# -----------------------------------------------------------------------------
# Chain and best-fit readers


def load_chain(path: str | Path, *, label: str | None = None) -> ChainSamples:
    """Read a pocoMC chain, including optional derived-quantity columns."""

    path = Path(path)
    names = _header_names(path)
    array = np.loadtxt(path)
    array = np.atleast_2d(np.asarray(array, dtype=np.float64))
    if names:
        names = tuple(names[: array.shape[1]])
        start = 1 if names and names[0] == "sample_index" else 0
        weight_index = names.index("weight") if "weight" in names else None
        if weight_index is None:
            parameter_columns = list(range(start, len(names)))
            metadata_end = len(names)
            weights = np.ones(array.shape[0])
        else:
            parameter_columns = list(range(start, weight_index))
            metadata_end = max(
                names.index(name) for name in ("weight", "loglike", "logprior") if name in names
            ) + 1
            weights = array[:, weight_index]
        parameter_names = tuple(names[i] for i in parameter_columns)
        derived_columns = [i for i in range(metadata_end, len(names)) if names[i] not in {"weight", "loglike", "logprior"}]
        derived_names = tuple(names[i] for i in derived_columns)
        loglike = array[:, names.index("loglike")] if "loglike" in names else None
        logprior = array[:, names.index("logprior")] if "logprior" in names else None
    else:
        parameter_columns = list(range(array.shape[1]))
        parameter_names = tuple(f"p{i}" for i in parameter_columns)
        derived_columns = []
        derived_names = ()
        weights = np.ones(array.shape[0])
        loglike = None
        logprior = None
    return ChainSamples(
        samples=np.ascontiguousarray(array[:, parameter_columns], dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64).reshape(-1),
        parameter_names=tuple(str(item) for item in parameter_names),
        loglike=None if loglike is None else np.asarray(loglike, dtype=np.float64).reshape(-1),
        logprior=None if logprior is None else np.asarray(logprior, dtype=np.float64).reshape(-1),
        derived=None if not derived_columns else np.ascontiguousarray(array[:, derived_columns], dtype=np.float64),
        derived_names=tuple(str(item) for item in derived_names),
        label=label or path.stem,
        path=path,
    )


def load_chain_from_config(
    path2config: str | Path,
    *,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    label: str | None = None,
    prefer_derived: bool = True,
) -> ChainSamples:
    """Resolve and read a fit chain, preferring ``*_chains_derived.txt`` when present."""

    return load_chain(
        pocomc_chain_path_from_config(
            path2config,
            output_dir=output_dir,
            prefix=prefix,
            prefer_derived=prefer_derived,
        ),
        label=label,
    )


def load_chains(paths: Sequence[str | Path], *, labels: Sequence[str] | None = None) -> list[ChainSamples]:
    """Read several chain files."""

    labels = labels or [None] * len(paths)
    if len(labels) != len(paths):
        raise ValueError("labels must have the same length as paths.")
    return [load_chain(path, label=label) for path, label in zip(paths, labels, strict=True)]


def load_optimization_result(path: str | Path) -> OptimizationResult:
    """Read ``*_optimum.txt`` written by ``scripts/run_fit_optimize.py``."""

    path = Path(path)
    names = _header_names(path)
    row = np.loadtxt(path)
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    special = {"loglike", "logprior", "logposterior"}
    parameter_names = tuple(name for name in names if name not in special) if names else tuple(f"p{i}" for i in range(row.size))
    n_params = len(parameter_names)
    values = {name: row[i] for i, name in enumerate(names[: row.size])} if names else {}
    return OptimizationResult(
        bestfit=row[:n_params],
        parameter_names=parameter_names,
        loglike=float(values["loglike"]) if "loglike" in values else None,
        logprior=float(values["logprior"]) if "logprior" in values else None,
        logposterior=float(values["logposterior"]) if "logposterior" in values else None,
        path=path,
    )


def load_optimization_result_from_config(
    path2config: str | Path,
    *,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
) -> OptimizationResult:
    """Resolve and read ``*_optimum.txt`` from a fit config."""

    return load_optimization_result(
        optimization_result_path_from_config(path2config, output_dir=output_dir, prefix=prefix)
    )


# -----------------------------------------------------------------------------
# getdist helpers


def _build_getdist_samples(
    chain: ChainSamples,
    *,
    params: Sequence[str] | None = None,
    labels: Mapping[str, str] | Sequence[str] | None = None,
    name_tag: str | None = None,
    settings: Mapping[str, Any] | None = None,
):
    try:
        from getdist import MCSamples
    except ImportError as exc:  # pragma: no cover - optional plotting dependency.
        raise ImportError("getdist is required for getdist_samples and triangle plots.") from exc

    selection = _getdist_selection(chain, params)
    selected_names = selection.selected_names
    label_map = _selected_label_map(selected_names, labels)
    getdist_names = _getdist_name_map((*selection.base_names, *selection.derived_names))
    base_getdist_names = [getdist_names[name] for name in selection.base_names]
    base_labels = [label_map.get(name, name) for name in selection.base_names]
    kwargs = dict(settings=settings or {})
    if chain.label is not None:
        kwargs["label"] = str(chain.label)
    samples = MCSamples(
        samples=chain.samples[:, selection.base_indices],
        weights=chain.weights,
        names=base_getdist_names,
        labels=base_labels,
        name_tag=name_tag or chain.label,
        **kwargs,
    )
    if selection.derived_indices:
        if chain.derived is None:
            raise RuntimeError("Derived parameter selection requested, but the chain has no derived array.")
        for index, name in zip(selection.derived_indices, selection.derived_names, strict=True):
            samples.addDerived(
                chain.derived[:, index],
                name=getdist_names[name],
                label=label_map.get(name, name),
            )
    return samples


def getdist_samples(
    fit: FitPlotData,
    *,
    params: Sequence[str] | None = None,
    labels: Mapping[str, str] | Sequence[str] | None = None,
    name_tag: str | None = None,
    settings: Mapping[str, Any] | None = None,
):
    """Return a GetDist sample for a loaded :class:`FitPlotData`."""

    if params is None and labels is None and name_tag is None and settings is None and fit.getdist_sample is not None:
        return fit.getdist_sample
    return _build_getdist_samples(
        fit.chain,
        params=params,
        labels=labels,
        name_tag=name_tag,
        settings=settings,
    )


def plot_triangle(
    fits: FitPlotData | Sequence[FitPlotData],
    *,
    params: Sequence[str] | None = None,
    labels: Mapping[str, str] | Sequence[str] | None = None,
    legend_labels: Sequence[str] | None = None,
    filled: bool = True,
    width_inch: float = 8.0,
    labelsize: int = DEFAULT_LABELSIZE,
    legendsize: int = DEFAULT_LEGENDSIZE,
    ticksize: int = DEFAULT_TICKSIZE,
    settings: Mapping[str, Any] | None = None,
    **kwargs: Any,
):
    """Triangle plot for one loaded fit or several overlaid fits."""

    try:
        from getdist import plots
    except ImportError as exc:  # pragma: no cover - optional plotting dependency.
        raise ImportError("getdist is required for triangle plots.") from exc

    fit_list = _fit_plot_data_list(fits)
    samples = [getdist_samples(fit, params=params, labels=labels, settings=settings) for fit in fit_list]
    plot_params = _getdist_plot_params(fit_list[0].chain, params)
    if legend_labels is None:
        legend_labels = [fit.label for fit in fit_list]
        if not any(label is not None for label in legend_labels):
            legend_labels = None
    plotter = plots.get_subplot_plotter(width_inch=width_inch)
    plotter.settings.axes_fontsize = ticksize
    plotter.settings.lab_fontsize = labelsize
    plotter.settings.legend_fontsize = legendsize
    plotter.triangle_plot(samples, params=plot_params, filled=filled, legend_labels=legend_labels, **kwargs)
    return plotter


# -----------------------------------------------------------------------------
# Fit observable bands


def fit_bands_from_fit(
    fit: FitPlotData,
    *,
    max_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
    quantiles: tuple[float, float] = DEFAULT_BAND_QUANTILES,
    label: str | None = None,
) -> list[ObservableFitBand]:
    """Evaluate a loaded fit into mean and interval theory bands."""

    samples, weights = _selected_samples(fit.chain, max_samples=max_samples, random_state=random_state)
    theory = np.asarray([fit.problem.theory_vector(theta) for theta in samples], dtype=np.float64)
    offsets = _segment_offsets(fit.problem.data.segments)
    diag = np.diag(fit.problem.data.covariance)
    bands = []
    for segment, start, stop in zip(fit.problem.data.segments, offsets[:-1], offsets[1:], strict=True):
        mean, lower, upper = _weighted_mean_interval(theory[:, start:stop], weights, quantiles=quantiles)
        x = _observable_x(fit.problem, segment)
        error = np.sqrt(np.clip(diag[start:stop], 0.0, None))
        bands.append(
            ObservableFitBand(
                name=segment.spec.name,
                tracer=segment.spec.tracer,
                statistic=segment.spec.statistic,
                x=x,
                data=segment.values,
                error=error,
                mean=mean,
                lower=lower,
                upper=upper,
                label=label or fit.label,
            )
        )
    return bands


def fit_bands_from_fits(
    fits: Sequence[FitPlotData],
    *,
    max_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
    quantiles: tuple[float, float] = DEFAULT_BAND_QUANTILES,
) -> list[list[ObservableFitBand]]:
    """Evaluate several loaded fits into fit bands."""

    return [
        fit_bands_from_fit(
            fit,
            max_samples=max_samples,
            random_state=random_state,
            quantiles=quantiles,
        )
        for fit in _fit_plot_data_list(fits)
    ]


def fit_bands_from_theory_vector(
    problem: HODFittingProblem,
    theory: Sequence[float] | np.ndarray,
    *,
    label: str | None = None,
) -> list[ObservableFitBand]:
    """Convert one theory vector, such as an optimization best fit, to plot bands."""

    theory = np.asarray(theory, dtype=np.float64).reshape(-1)
    if theory.size != problem.data.size:
        raise ValueError(f"Theory vector has length {theory.size}; expected {problem.data.size}.")
    offsets = _segment_offsets(problem.data.segments)
    diag = np.diag(problem.data.covariance)
    bands = []
    for segment, start, stop in zip(problem.data.segments, offsets[:-1], offsets[1:], strict=True):
        values = theory[start:stop]
        bands.append(
            ObservableFitBand(
                name=segment.spec.name,
                tracer=segment.spec.tracer,
                statistic=segment.spec.statistic,
                x=_observable_x(problem, segment),
                data=segment.values,
                error=np.sqrt(np.clip(diag[start:stop], 0.0, None)),
                mean=values,
                lower=values,
                upper=values,
                label=label,
            )
        )
    return bands


def fit_bands_from_optimization(
    problem: HODFittingProblem,
    result: OptimizationResult,
    *,
    label: str | None = "best fit",
) -> list[ObservableFitBand]:
    """Evaluate an optimization result into fit-band format."""

    return fit_bands_from_theory_vector(problem, problem.theory_vector(result.bestfit), label=label)


def load_theory_vector(path: str | Path) -> np.ndarray:
    """Read a saved ``*_theory_vector.txt`` file."""

    return np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)


def plot_fit_bands(
    fits: FitPlotData | Sequence[FitPlotData],
    *,
    labels: Sequence[str] | None = None,
    figsize_per_panel: tuple[float, float] = DEFAULT_FIGSIZE_PER_PANEL,
    labelsize: int = DEFAULT_LABELSIZE,
    legendsize: int = DEFAULT_LEGENDSIZE,
    ticksize: int = DEFAULT_TICKSIZE,
    alpha: float = 0.25,
    markersize: float = 4.0,
    show_data: bool = True,
    xscale: str = "log",
    plot_rp_wp: bool = True,
    titles: Sequence[str] | bool | None = None,
    max_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
    quantiles: tuple[float, float] = DEFAULT_BAND_QUANTILES,
):
    """Plot loaded fits and normalized residuals against the measured data."""

    plt = _matplotlib_pyplot()
    fit_list = _fit_plot_data_list(fits)
    band_sets = fit_bands_from_fits(
        fit_list,
        max_samples=max_samples,
        random_state=random_state,
        quantiles=quantiles,
    )
    if not band_sets:
        raise ValueError("No fit bands were provided.")
    n_panels = len(band_sets[0])
    upper_height = float(figsize_per_panel[1])
    fig, axes = plt.subplots(
        2,
        n_panels,
        figsize=(figsize_per_panel[0] * n_panels, upper_height * 4.0 / 3.0),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex="col",
        squeeze=False,
    )
    upper_axes = axes[0]
    residual_axes = axes[1]
    fit_labels = _plot_labels(labels, len(band_sets))
    panel_titles = _panel_titles(titles, n_panels, band_sets[0])
    for i, (ax, rax) in enumerate(zip(upper_axes, residual_axes, strict=True)):
        reference = band_sets[0][i]
        rax.axhspan(-1.0, 1.0, color="lightgray", alpha=0.6, linewidth=0, zorder=0)
        rax.axhline(0.0, color="0.4", linewidth=1.0, zorder=1)
        for band_set, chain_label in zip(band_sets, fit_labels, strict=True):
            band = band_set[i]
            scale = _observable_plot_scale(band, plot_rp_wp=plot_rp_wp)
            line_label = chain_label or band.label
            line = ax.plot(band.x, band.mean * scale, label=line_label)[0]
            color = line.get_color()
            ax.fill_between(
                band.x,
                band.lower * scale,
                band.upper * scale,
                color=color,
                alpha=alpha,
                linewidth=0,
            )
            if show_data:
                ax.errorbar(
                    band.x,
                    band.data * scale,
                    yerr=band.error * np.abs(scale),
                    fmt="o",
                    color=color,
                    ecolor=color,
                    markersize=markersize,
                    linestyle="none",
                    label=None,
                )
            residual = _normalized_fit_residual(band)
            rax.plot(band.x, residual, color=color, marker="o", markersize=markersize, linewidth=1.0)
        if xscale and np.all(reference.x > 0.0):
            ax.set_xscale(xscale)
            rax.set_xscale(xscale)
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel(_observable_y_label(reference.statistic, plot_rp_wp=plot_rp_wp), fontsize=labelsize)
        if panel_titles[i] is not None:
            ax.set_title(panel_titles[i], fontsize=labelsize)
        rax.set_xlabel(_observable_x_label(reference.statistic), fontsize=labelsize)
        rax.set_ylabel(_residual_y_label(reference.statistic), fontsize=labelsize)
        _style_axis(ax, ticksize=ticksize, legendsize=legendsize)
        _style_axis(rax, ticksize=ticksize, legendsize=legendsize)
    fig.tight_layout()
    return fig, axes


def plot_fit(fit: FitPlotData, **kwargs: Any):
    """Compute and plot fit bands for one loaded fit."""

    return plot_fit_bands(fit, **kwargs)


def plot_fits(fits: Sequence[FitPlotData], **kwargs: Any):
    """Compute and plot overlaid fit bands for several loaded fits."""

    return plot_fit_bands(fits, **kwargs)


def plot_fit_from_config(
    path2config: str | Path,
    *,
    chain_path: str | Path | None = None,
    label: str | None = None,
    prefer_derived: bool = True,
    validate: bool = False,
    **kwargs: Any,
):
    """Load the configured fit outputs, then plot the fit bands."""

    plot_data = load_fit_plot_data(
        path2config,
        chain_path=chain_path,
        label=label,
        prefer_derived=prefer_derived,
        validate=validate,
        build_getdist=False,
    )
    return plot_fit(plot_data, **kwargs)


# -----------------------------------------------------------------------------
# HOD bands


def hod_bands_from_fit(
    fit: FitPlotData,
    *,
    tracer: str | None = None,
    logm: np.ndarray | None = None,
    hod_model: str | None = None,
    max_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
    quantiles: tuple[float, float] = DEFAULT_BAND_QUANTILES,
    label: str | None = None,
) -> HODBand:
    """Evaluate central, satellite, and total HOD bands for one loaded fit."""

    tracer = str(tracer or fit.problem.tracers[0])
    if logm is None:
        logm = np.linspace(11.0, 15.5, 256)
    logm = np.asarray(logm, dtype=np.float64)
    mass = 10.0**logm
    model = str(hod_model or fit.problem._hod_model_for_tracer(tracer))
    samples, weights = _selected_samples(fit.chain, max_samples=max_samples, random_state=random_state)
    central = []
    satellite = []
    for theta in samples:
        params = fit.problem.params_for_tracer(theta, tracer)
        cen, sat = evaluate_hod(mass, params, model=model)
        central.append(cen)
        satellite.append(sat)
    central = np.asarray(central, dtype=np.float64)
    satellite = np.asarray(satellite, dtype=np.float64)
    total = central + satellite
    cen_mean, cen_low, cen_high = _weighted_mean_interval(central, weights, quantiles=quantiles)
    sat_mean, sat_low, sat_high = _weighted_mean_interval(satellite, weights, quantiles=quantiles)
    tot_mean, tot_low, tot_high = _weighted_mean_interval(total, weights, quantiles=quantiles)
    return HODBand(
        tracer=tracer,
        logm=logm,
        central_mean=cen_mean,
        central_lower=cen_low,
        central_upper=cen_high,
        satellite_mean=sat_mean,
        satellite_lower=sat_low,
        satellite_upper=sat_high,
        total_mean=tot_mean,
        total_lower=tot_low,
        total_upper=tot_high,
        label=label or fit.label,
    )


def hod_bands_from_fits(fits: Sequence[FitPlotData], **kwargs: Any) -> list[HODBand]:
    """Evaluate HOD bands for several loaded fits."""

    return [hod_bands_from_fit(fit, **kwargs) for fit in _fit_plot_data_list(fits)]


def plot_hod_bands(
    fits: FitPlotData | Sequence[FitPlotData],
    *,
    components: Sequence[str] = ("central", "satellite"),
    labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE_PER_PANEL,
    labelsize: int = DEFAULT_LABELSIZE,
    legendsize: int = DEFAULT_LEGENDSIZE,
    ticksize: int = DEFAULT_TICKSIZE,
    alpha: float = 0.25,
    show_band: Sequence[bool] | bool | None = None,
    yscale: str | None = "log",
    ylim: tuple[float, float] | None = (1.0e-3, 10.0**1.5),
    tracer: str | None = None,
    logm: np.ndarray | None = None,
    hod_model: str | None = None,
    max_samples: int | None = None,
    random_state: int | np.random.Generator | None = None,
    quantiles: tuple[float, float] = DEFAULT_BAND_QUANTILES,
):
    """Plot HOD occupation bands for one or several loaded fits."""

    plt = _matplotlib_pyplot()
    fit_list = _fit_plot_data_list(fits)
    band_list = hod_bands_from_fits(
        fit_list,
        tracer=tracer,
        logm=logm,
        hod_model=hod_model,
        max_samples=max_samples,
        random_state=random_state,
        quantiles=quantiles,
    )
    if not band_list:
        raise ValueError("No HOD bands were provided.")
    fit_labels = _plot_labels(labels, len(band_list))
    show_bands = _show_band_flags(show_band, len(band_list))
    fig, ax = plt.subplots(figsize=figsize)
    linestyles = {"central": "-", "satellite": "--", "total": ":"}
    for band, chain_label, draw_band in zip(band_list, fit_labels, show_bands, strict=True):
        base_label = chain_label or band.label or band.tracer
        for component in components:
            mean, lower, upper = _hod_component_arrays(band, component)
            mean, lower, upper = _hod_plot_values(mean, lower, upper, yscale=yscale)
            line = ax.plot(
                band.logm,
                mean,
                linestyle=linestyles.get(component, "-"),
                label=f"{base_label} {component}",
            )[0]
            if draw_band:
                ax.fill_between(band.logm, lower, upper, color=line.get_color(), alpha=alpha, linewidth=0)
    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\log_{10}(M_h/[M_\odot/h])$", fontsize=labelsize)
    ax.set_ylabel(r"$\langle N \rangle$", fontsize=labelsize)
    _style_axis(ax, ticksize=ticksize, legendsize=legendsize)
    fig.tight_layout()
    return fig, ax


def plot_hod(fit: FitPlotData, **kwargs: Any):
    """Compute and plot HOD bands for one loaded fit."""

    return plot_hod_bands(fit, **kwargs)


def plot_hods(fits: Sequence[FitPlotData], **kwargs: Any):
    """Compute and plot overlaid HOD bands for several loaded fits."""

    return plot_hod_bands(fits, **kwargs)


def plot_derived_parameters(
    fits: FitPlotData | Sequence[FitPlotData],
    x_values: Sequence[float],
    x_label: str,
    *,
    derived_params: Sequence[str] = DEFAULT_DERIVED_PARAMETERS,
    derived_labels: Mapping[str, str] | Sequence[str] | None = None,
    figsize_per_panel: tuple[float, float] = DEFAULT_FIGSIZE_PER_PANEL,
    labelsize: int = DEFAULT_LABELSIZE,
    legendsize: int = DEFAULT_LEGENDSIZE,
    ticksize: int = DEFAULT_TICKSIZE,
    markersize: float = 4.0,
    capsize: float = 3.0,
    color: str | None = None,
    xscale: str | None = None,
):
    """Plot derived parameters with errors against an external x-property."""

    plt = _matplotlib_pyplot()
    fit_list = _fit_plot_data_list(fits)
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    if x.size != len(fit_list):
        raise ValueError(f"x_values has length {x.size}; expected {len(fit_list)} to match the number of fits.")
    params = tuple(str(item) for item in derived_params)
    if not params:
        raise ValueError("derived_params must contain at least one parameter name.")
    ylabels = _derived_parameter_labels(params, derived_labels)
    fig, axes = plt.subplots(
        len(params),
        1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(params)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for ax, param, ylabel in zip(axes, params, ylabels, strict=True):
        stats = [_resolve_parameter_stat(fit, param) for fit in fit_list]
        mean = np.asarray([stat.mean for stat in stats], dtype=np.float64)
        err_minus = np.asarray([stat.err_minus for stat in stats], dtype=np.float64)
        err_plus = np.asarray([stat.err_plus for stat in stats], dtype=np.float64)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([err_minus, err_plus]),
            fmt="o-",
            color=color,
            markersize=markersize,
            capsize=capsize,
        )
        ax.set_ylabel(ylabel, fontsize=labelsize)
        if xscale is not None:
            ax.set_xscale(xscale)
        _style_axis(ax, ticksize=ticksize, legendsize=legendsize)
    axes[-1].set_xlabel(str(x_label), fontsize=labelsize)
    fig.tight_layout()
    return fig, axes


def plot_joint_grid(
    plot_function: Any,
    input_sets: Sequence[Any],
    *,
    nrows: int,
    ncols: int,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    per_set_args: Sequence[Sequence[Any]] | None = None,
    per_set_kwargs: Sequence[Mapping[str, Any]] | None = None,
    figsize_per_panel: tuple[float, float] = DEFAULT_FIGSIZE_PER_PANEL,
    titles: Sequence[str] | None = None,
    close: bool = True,
    interpolation: str = "nearest",
):
    """Compose several plots from one plotting function into a grid figure.

    ``plot_function`` is called once per item in ``input_sets``. Each item is
    passed as the first positional argument, followed by shared ``args`` and any
    matching ``per_set_args`` entry. Shared ``kwargs`` are merged with matching
    ``per_set_kwargs`` entries. The number of input sets must equal
    ``nrows * ncols``.
    """

    plt = _matplotlib_pyplot()
    input_sets = list(input_sets)
    nrows = int(nrows)
    ncols = int(ncols)
    if nrows <= 0 or ncols <= 0:
        raise ValueError("nrows and ncols must be positive integers.")
    if len(input_sets) != nrows * ncols:
        raise ValueError(
            f"Got {len(input_sets)} input sets, but nrows*ncols is {nrows * ncols} "
            f"({nrows}*{ncols})."
        )
    shared_args = tuple(args or ())
    shared_kwargs = dict(kwargs or {})
    per_set_args = _optional_sequence(per_set_args, len(input_sets), name="per_set_args")
    per_set_kwargs = _optional_sequence(per_set_kwargs, len(input_sets), name="per_set_kwargs")
    panel_titles = _grid_titles(titles, len(input_sets))

    images = []
    for i, input_set in enumerate(input_sets):
        call_args = (input_set, *shared_args, *tuple(per_set_args[i] or ()))
        call_kwargs = dict(shared_kwargs)
        call_kwargs.update(dict(per_set_kwargs[i] or {}))
        result = plot_function(*call_args, **call_kwargs)
        panel_fig = _figure_from_plot_result(result)
        images.append(_figure_to_rgb_array(panel_fig))
        if close:
            plt.close(panel_fig)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for ax, image, title in zip(axes.ravel(), images, panel_titles, strict=True):
        ax.imshow(image, interpolation=interpolation)
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title, fontsize=DEFAULT_LABELSIZE)
    fig.tight_layout()
    return fig, axes


# -----------------------------------------------------------------------------
# Internal helpers


def _observable_data_points(problem: HODFittingProblem) -> list[ObservableDataPoints]:
    offsets = _segment_offsets(problem.data.segments)
    diag = np.diag(problem.data.covariance)
    out: list[ObservableDataPoints] = []
    for segment, start, stop in zip(problem.data.segments, offsets[:-1], offsets[1:], strict=True):
        out.append(
            ObservableDataPoints(
                name=segment.spec.name,
                tracer=segment.spec.tracer,
                statistic=segment.spec.statistic,
                x=_observable_x(problem, segment),
                values=segment.values,
                error=np.sqrt(np.clip(diag[start:stop], 0.0, None)),
            )
        )
    return out


def _fit_plot_data_list(value: FitPlotData | Sequence[FitPlotData]) -> list[FitPlotData]:
    if isinstance(value, FitPlotData):
        return [value]
    out = list(value)
    if not out or not all(isinstance(item, FitPlotData) for item in out):
        raise TypeError("Expected FitPlotData or a sequence of FitPlotData.")
    return out


def _optional_sequence(value: Sequence[Any] | None, size: int, *, name: str) -> list[Any | None]:
    if value is None:
        return [None] * int(size)
    out = list(value)
    if len(out) != int(size):
        raise ValueError(f"{name} has length {len(out)}; expected {int(size)} to match input_sets.")
    return out


def _grid_titles(titles: Sequence[str] | None, size: int) -> list[str | None]:
    if titles is None:
        return [None] * int(size)
    out = [None if title is None else str(title) for title in titles]
    if len(out) != int(size):
        raise ValueError(f"titles has length {len(out)}; expected {int(size)} to match input_sets.")
    return out


def _figure_from_plot_result(result: Any):
    if isinstance(result, tuple) and result:
        candidate = result[0]
    else:
        candidate = result
    if hasattr(candidate, "canvas"):
        return candidate
    if hasattr(candidate, "fig") and hasattr(candidate.fig, "canvas"):
        return candidate.fig
    if hasattr(candidate, "figure") and hasattr(candidate.figure, "canvas"):
        return candidate.figure
    raise TypeError("plot_function must return a Matplotlib Figure, (Figure, axes), or object with a .fig Figure.")


def _figure_to_rgb_array(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.array(rgba[..., :3], copy=True)


def _plot_labels(labels: Sequence[str] | None, size: int) -> list[str | None]:
    if labels is None:
        return [None] * int(size)
    out = [None if label is None else str(label) for label in labels]
    if len(out) != int(size):
        raise ValueError(f"labels has length {len(out)}; expected {int(size)} to match the number of fits.")
    return out


def _panel_titles(
    titles: Sequence[str] | bool | None,
    size: int,
    reference_bands: Sequence[ObservableFitBand],
) -> list[str | None]:
    if titles is None or titles is False:
        return [None] * int(size)
    if titles is True:
        return [str(band.name) for band in reference_bands]
    if isinstance(titles, str):
        return [titles] * int(size)
    out = [None if title is None else str(title) for title in titles]
    if len(out) != int(size):
        raise ValueError(f"titles has length {len(out)}; expected {int(size)} to match the number of panels.")
    return out


def _show_band_flags(show_band: Sequence[bool] | bool | None, size: int) -> list[bool]:
    if show_band is None:
        return [True] * int(size)
    if isinstance(show_band, bool):
        return [bool(show_band)] * int(size)
    out = [bool(item) for item in show_band]
    if len(out) != int(size):
        raise ValueError(f"show_band has length {len(out)}; expected {int(size)} to match the number of fits.")
    return out


def _resolve_parameter_stat(fit: FitPlotData, name: str) -> ParameterStats:
    name = str(name)
    if name not in fit.parameter_stats:
        raise KeyError(f"Derived parameter {name!r} not found. Available parameters: {tuple(fit.parameter_stats)}.")
    return fit.parameter_stats[name]


def _derived_parameter_labels(
    params: Sequence[str],
    labels: Mapping[str, str] | Sequence[str] | None,
) -> list[str]:
    if labels is None:
        return [_default_parameter_label(param) for param in params]
    if isinstance(labels, Mapping):
        return [
            str(labels.get(param, labels.get(_parameter_label_key(param), _default_parameter_label(param))))
            for param in params
        ]
    if len(labels) != len(params):
        raise ValueError(f"derived_labels has length {len(labels)}; expected {len(params)}.")
    return [str(label) for label in labels]


def _chain_parameter_values(chain: ChainSamples, name: str) -> np.ndarray:
    name = str(name)
    if name in chain.parameter_names:
        return chain.samples[:, chain.parameter_names.index(name)]
    if name in chain.derived_names:
        if chain.derived is None:
            raise KeyError(f"Derived parameter {name!r} is named but no derived array is loaded.")
        return chain.derived[:, chain.derived_names.index(name)]
    raise KeyError(f"Parameter {name!r} not found. Available parameters: {chain.all_parameter_names}.")


def _parameter_stats_from_chain(
    chain: ChainSamples,
    labels: Mapping[str, str] | Sequence[str] | None = None,
    *,
    quantiles: tuple[float, float] = DEFAULT_BAND_QUANTILES,
) -> dict[str, ParameterStats]:
    names = chain.all_parameter_names
    weights = chain.normalized_weights()
    label_map = _selected_label_map(names, labels)
    getdist_names = _getdist_name_map(names)
    out: dict[str, ParameterStats] = {}
    for name in names:
        values = _chain_parameter_values(chain, name)
        mean = float(np.average(values, weights=weights))
        lower = _weighted_quantile(values, weights, quantiles[0])
        upper = _weighted_quantile(values, weights, quantiles[1])
        variance = float(np.average((values - mean) ** 2, weights=weights))
        std = float(np.sqrt(max(variance, 0.0)))
        out[name] = ParameterStats(
            name=name,
            getdist_name=getdist_names[name],
            label=label_map.get(name, name),
            mean=mean,
            lower=lower,
            upper=upper,
            err_minus=mean - lower,
            err_plus=upper - mean,
            std=std,
        )
    return out


def _parameter_stats_from_getdist(
    samples: Any,
    chain: ChainSamples,
    labels: Mapping[str, str] | Sequence[str] | None = None,
) -> dict[str, ParameterStats]:
    fallback = _parameter_stats_from_chain(chain, labels)
    try:
        marge = samples.getMargeStats()
    except Exception:
        return fallback
    out: dict[str, ParameterStats] = {}
    for name, fallback_stat in fallback.items():
        try:
            par = marge.parWithName(fallback_stat.getdist_name)
        except Exception:
            out[name] = fallback_stat
            continue
        mean = _finite_or(getattr(par, "mean", np.nan), fallback_stat.mean)
        std = _finite_or(getattr(par, "err", np.nan), fallback_stat.std)
        lower, upper = _first_getdist_interval(par)
        lower = _finite_or(lower, fallback_stat.lower)
        upper = _finite_or(upper, fallback_stat.upper)
        out[name] = ParameterStats(
            name=name,
            getdist_name=fallback_stat.getdist_name,
            label=fallback_stat.label,
            mean=mean,
            lower=lower,
            upper=upper,
            err_minus=mean - lower,
            err_plus=upper - mean,
            std=std,
        )
    return out


def _first_getdist_interval(par: Any) -> tuple[float, float]:
    for limit in getattr(par, "limits", ()) or ():
        lower = _as_float_or_nan(getattr(limit, "lower", np.nan))
        upper = _as_float_or_nan(getattr(limit, "upper", np.nan))
        if np.isfinite(lower) and np.isfinite(upper):
            return lower, upper
    return float("nan"), float("nan")


def _finite_or(value: Any, fallback: float) -> float:
    value = _as_float_or_nan(value)
    return value if np.isfinite(value) else float(fallback)


def _as_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_config(path_or_config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_config, Mapping):
        return dict(path_or_config)
    return load_config(path_or_config)


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


def _header_names(path: Path) -> tuple[str, ...]:
    header: tuple[str, ...] = ()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header = tuple(stripped.lstrip("#").strip().split())
                continue
            break
    return header


def _normalized_weights(weights: np.ndarray | None, size: int) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=np.float64) / max(size, 1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size != size:
        raise ValueError(f"Weights length {weights.size} does not match sample size {size}.")
    good = np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return np.ones(size, dtype=np.float64) / max(size, 1)
    out = np.zeros(size, dtype=np.float64)
    out[good] = weights[good] / np.sum(weights[good])
    return out


@dataclass(frozen=True)
class _GetDistSelection:
    base_indices: list[int]
    base_names: tuple[str, ...]
    derived_indices: list[int]
    derived_names: tuple[str, ...]
    selected_names: tuple[str, ...]


def _getdist_selection(chain: ChainSamples, params: Sequence[str] | None) -> _GetDistSelection:
    base_names = tuple(str(item) for item in chain.parameter_names)
    derived_names = tuple(str(item) for item in chain.derived_names)
    available = (*base_names, *derived_names)
    if params is None:
        return _GetDistSelection(
            base_indices=list(range(len(base_names))),
            base_names=base_names,
            derived_indices=list(range(len(derived_names))),
            derived_names=derived_names,
            selected_names=available,
        )

    selected_names = tuple(str(item) for item in params)
    base_indices: list[int] = []
    selected_base_names: list[str] = []
    derived_indices: list[int] = []
    selected_derived_names: list[str] = []
    for name in selected_names:
        if name in base_names:
            base_indices.append(base_names.index(name))
            selected_base_names.append(name)
        elif name in derived_names:
            derived_indices.append(derived_names.index(name))
            selected_derived_names.append(name)
        else:
            raise KeyError(f"Parameter {name!r} not found. Available parameters: {available}.")

    if selected_base_names:
        internal_base_indices = base_indices
        internal_base_names = tuple(selected_base_names)
    else:
        internal_base_indices = list(range(len(base_names)))
        internal_base_names = base_names

    return _GetDistSelection(
        base_indices=internal_base_indices,
        base_names=internal_base_names,
        derived_indices=derived_indices,
        derived_names=tuple(selected_derived_names),
        selected_names=selected_names,
    )


def _selected_label_map(names: Sequence[str], labels: Mapping[str, str] | Sequence[str] | None) -> dict[str, str]:
    selected = tuple(str(item) for item in names)
    if labels is None:
        return {name: _default_parameter_label(name) for name in selected}
    if isinstance(labels, Mapping):
        return {
            name: str(labels.get(name, labels.get(_parameter_label_key(name), _default_parameter_label(name))))
            for name in selected
        }
    if len(labels) != len(selected):
        raise ValueError("Label list must match the number of selected parameters.")
    return {name: str(label) for name, label in zip(selected, labels, strict=True)}


def _unique_getdist_names(names: Sequence[str]) -> list[str]:
    out = []
    used: dict[str, int] = {}
    for name in names:
        clean = re.sub(r"\W", "_", str(name))
        if not clean or clean[0].isdigit():
            clean = f"p_{clean}"
        count = used.get(clean, 0)
        used[clean] = count + 1
        out.append(clean if count == 0 else f"{clean}_{count}")
    return out


def _getdist_name_map(names: Sequence[str]) -> dict[str, str]:
    original = tuple(str(item) for item in names)
    return dict(zip(original, _unique_getdist_names(original), strict=True))


def _getdist_plot_params(chain: ChainSamples, params: Sequence[str] | None) -> list[str] | None:
    if params is None:
        return None
    selection = _getdist_selection(chain, params)
    getdist_names = _getdist_name_map((*selection.base_names, *selection.derived_names))
    return [getdist_names[str(name)] for name in selection.selected_names]


def _selected_samples(
    chain: ChainSamples,
    *,
    max_samples: int | None,
    random_state: int | np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    weights = chain.normalized_weights()
    if max_samples is None or chain.size <= int(max_samples):
        return chain.samples, weights
    positive = np.flatnonzero(weights > 0.0)
    if positive.size <= int(max_samples):
        selected_weights = weights[positive]
        selected_weights = selected_weights / np.sum(selected_weights)
        return chain.samples[positive], selected_weights
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    positive_weights = weights[positive] / np.sum(weights[positive])
    indices = rng.choice(positive, size=int(max_samples), replace=False, p=positive_weights)
    selected_weights = weights[indices]
    selected_weights = selected_weights / np.sum(selected_weights)
    return chain.samples[indices], selected_weights


def _weighted_mean_interval(
    values: np.ndarray,
    weights: np.ndarray | None,
    *,
    quantiles: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    weights = _normalized_weights(weights, values.shape[0])
    mean = np.average(values, axis=0, weights=weights)
    lower = np.array([_weighted_quantile(values[:, i], weights, quantiles[0]) for i in range(values.shape[1])])
    upper = np.array([_weighted_quantile(values[:, i], weights, quantiles[1]) for i in range(values.shape[1])])
    return mean, lower, upper


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return float("nan")
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return float(np.interp(float(quantile) / 100.0, cumulative, values))


def _segment_offsets(segments: Sequence[Any]) -> np.ndarray:
    sizes = [segment.size for segment in segments]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


def _observable_x(problem: HODFittingProblem, segment: Any) -> np.ndarray:
    tabulator = problem._tabulator_for_spec(segment.spec)
    bins = tabulator.paircounts.bins
    if segment.spec.statistic == "wp":
        edges = bins["rp_edges"]
    else:
        edges = bins["s_edges"]
    return _bin_midpoints(edges)[segment.selected_indices]


def _bin_midpoints(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    if np.all(edges > 0.0):
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def _observable_x_label(statistic: str) -> str:
    return r"$r_p\,[h^{-1}{\rm Mpc}]$" if statistic == "wp" else r"$s\,[h^{-1}{\rm Mpc}]$"


def _observable_y_label(statistic: str, *, plot_rp_wp: bool = True) -> str:
    if statistic == "wp" and plot_rp_wp:
        return r"$r_p w_p(r_p)$"
    labels = {"wp": r"$w_p(r_p)$", "xi0": r"$\xi_0(s)$", "xi2": r"$\xi_2(s)$"}
    return labels.get(statistic, statistic)


def _observable_plot_scale(band: ObservableFitBand, *, plot_rp_wp: bool) -> np.ndarray | float:
    if band.statistic == "wp" and plot_rp_wp:
        return band.x
    return 1.0


def _normalized_fit_residual(band: ObservableFitBand) -> np.ndarray:
    error = np.asarray(band.error, dtype=np.float64)
    residual = np.full_like(error, np.nan, dtype=np.float64)
    good = np.isfinite(error) & (error > 0.0)
    residual[good] = (np.asarray(band.mean, dtype=np.float64)[good] - np.asarray(band.data, dtype=np.float64)[good]) / error[good]
    return residual


def _residual_y_label(statistic: str) -> str:
    labels = {
        "wp": r"$\Delta w_p/\sigma_{w_p}$",
        "xi0": r"$\Delta \xi_0/\sigma_{\xi_0}$",
        "xi2": r"$\Delta \xi_2/\sigma_{\xi_2}$",
    }
    return labels.get(statistic, r"$\Delta/\sigma$")


def _parameter_label_key(name: str) -> str:
    return str(name).split(".")[-1]


def _default_parameter_label(name: str) -> str:
    key = _parameter_label_key(name)
    return DEFAULT_PARAMETER_LABELS.get(key, str(name))


def _hod_plot_values(
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    yscale: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if str(yscale).lower() != "log":
        return mean, lower, upper
    floor = 1.0e-8
    return np.clip(mean, floor, None), np.clip(lower, floor, None), np.clip(upper, floor, None)


def _hod_component_arrays(band: HODBand, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    component = str(component).lower()
    if component in {"central", "cen", "ncen"}:
        return band.central_mean, band.central_lower, band.central_upper
    if component in {"satellite", "sat", "nsat"}:
        return band.satellite_mean, band.satellite_lower, band.satellite_upper
    if component in {"total", "all", "ngal"}:
        return band.total_mean, band.total_lower, band.total_upper
    raise ValueError("HOD component must be central, satellite, or total.")


def _matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting dependency.
        raise ImportError("matplotlib is required for plotting helpers.") from exc
    return plt


def _style_axis(ax: Any, *, ticksize: int, legendsize: int) -> None:
    ax.tick_params(axis="both", labelsize=ticksize)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=legendsize)
