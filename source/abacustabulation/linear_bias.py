"""Post-processing helpers for real-space large-scale linear bias."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .clustering import HODClusteringTabulator, smu_multipoles


@dataclass(frozen=True)
class LinearBiasResult:
    bias: float
    s_mid: np.ndarray
    xi0_real: np.ndarray
    xi_linear_matter: np.ndarray
    bias_of_s: np.ndarray
    fit_mask: np.ndarray


@dataclass(frozen=True)
class StellarMassLinearBiasResult:
    """Linear-bias results for every stellar-mass split in edge order."""

    bias: np.ndarray
    s_mid: np.ndarray
    xi0_real: np.ndarray
    xi_linear_matter: np.ndarray
    bias_of_s: np.ndarray
    fit_mask: np.ndarray
    split_method: str
    logmstar_edges: np.ndarray
    redshift_weights: np.ndarray

    def __len__(self) -> int:
        return int(self.bias.size)

    def __getitem__(self, index: int) -> LinearBiasResult:
        return LinearBiasResult(
            bias=float(self.bias[index]),
            s_mid=self.s_mid,
            xi0_real=self.xi0_real[index],
            xi_linear_matter=self.xi_linear_matter,
            bias_of_s=self.bias_of_s[index],
            fit_mask=self.fit_mask[index],
        )


def _linear_bias_from_monopoles(
    s_mid: np.ndarray,
    xi0_real: np.ndarray,
    xi_linear_matter: np.ndarray,
    *,
    fit_s_min: float | None,
    fit_s_max: float | None,
    positive_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi0_real = np.atleast_2d(np.asarray(xi0_real, dtype=np.float64))
    xi_linear_matter = np.asarray(xi_linear_matter, dtype=np.float64)
    if xi0_real.shape[1:] != xi_linear_matter.shape:
        raise ValueError(
            f"xi0_real shape {xi0_real.shape} is incompatible with "
            f"linear matter xi shape {xi_linear_matter.shape}."
        )
    ratio = np.divide(
        xi0_real,
        xi_linear_matter[None, :],
        out=np.full_like(xi0_real, np.nan),
        where=xi_linear_matter[None, :] != 0.0,
    )
    bias_of_s = np.full_like(ratio, np.nan, dtype=np.float64)
    valid_ratio = np.isfinite(ratio) & (ratio >= 0.0)
    bias_of_s[valid_ratio] = np.sqrt(ratio[valid_ratio])

    radial_mask = np.ones_like(s_mid, dtype=bool)
    if fit_s_min is not None:
        radial_mask &= s_mid >= fit_s_min
    if fit_s_max is not None:
        radial_mask &= s_mid <= fit_s_max
    mask = np.broadcast_to(radial_mask, bias_of_s.shape).copy()
    mask &= np.isfinite(bias_of_s)
    if positive_only:
        mask &= (xi0_real > 0.0) & (xi_linear_matter[None, :] > 0.0)

    counts = np.sum(mask, axis=1)
    invalid = np.flatnonzero(counts == 0)
    if invalid.size:
        if xi0_real.shape[0] == 1:
            raise ValueError(
                "No valid bins are available for the configured linear-bias fit range."
            )
        raise ValueError(
            "No valid bins are available for the configured linear-bias fit "
            f"range for stellar-mass split(s) {invalid.tolist()}."
        )
    bias = np.sum(np.where(mask, bias_of_s, 0.0), axis=1) / counts
    return bias, bias_of_s, mask


def infer_abacus_cosmology_index(sim_name: str) -> int:
    """Infer the AbacusSummit cXXX cosmology index from a simulation name."""

    match = re.search(r"(?:^|_)c(\d{3})(?:_|$)", str(sim_name))
    if match is None:
        raise ValueError(f"Cannot infer AbacusSummit cosmology index from sim_name={sim_name!r}.")
    return int(match.group(1))


def linear_matter_xi_abacus(
    s: np.ndarray,
    *,
    z: float,
    cosmology_index: int,
    engine: str = "camb",
) -> np.ndarray:
    """Return linear matter xi(s, z) for an AbacusSummit cosmology."""

    try:
        from cosmoprimo import Fourier
        from cosmoprimo.fiducial import AbacusSummit
    except ImportError as exc:  # pragma: no cover - depends on cluster environment.
        raise ImportError("cosmoprimo is required to compute linear bias.") from exc

    try:
        cosmo = AbacusSummit(int(cosmology_index), engine=engine)
    except TypeError:
        cosmo = AbacusSummit(int(cosmology_index))
    fo = Fourier(cosmo, engine=engine)
    xi_linear = fo.pk_interpolator().to_xi()
    return np.asarray(xi_linear(np.asarray(s, dtype=np.float64), z=float(z)), dtype=np.float64)


def _validate_real_space_smu_paircounts(tabulator: HODClusteringTabulator) -> None:
    paircounts = tabulator.paircounts
    if paircounts.clustering != "smu":
        raise ValueError("Linear-bias postprocessing requires smu paircounts.")
    position_dataset = str(paircounts.attrs.get("position_dataset", "pos"))
    if position_dataset != "pos":
        raise ValueError("Linear-bias postprocessing requires real-space position_dataset='pos'.")
    pos_space = paircounts.attrs.get("pos_dataset_space")
    if pos_space is not None and str(pos_space).lower() != "real":
        raise ValueError(f"Paircount dataset 'pos' is declared as {pos_space!r}, not real.")
    if pos_space is None and str(paircounts.attrs.get("position_space", "")).lower() == "rsd":
        raise ValueError(
            "Linear-bias paircounts look like legacy RSD-only files with RSD coordinates in 'pos'. "
            "Re-prepare catalogs with the current pos/pos_rsd convention."
        )


class HODLinearBiasTabulator:
    """Reusable real-space smu tabulator for HOD linear-bias postprocessing."""

    def __init__(
        self,
        clustering_tabulator: HODClusteringTabulator,
        *,
        cosmology_index: int,
        z: float,
        engine: str = "camb",
        fit_s_min: float | None = None,
        fit_s_max: float | None = None,
        positive_only: bool = True,
        xi_linear_matter: np.ndarray | None = None,
    ):
        _validate_real_space_smu_paircounts(clustering_tabulator)
        self.clustering_tabulator = clustering_tabulator
        self.cosmology_index = int(cosmology_index)
        self.z = float(z)
        self.engine = str(engine)
        self.fit_s_min = None if fit_s_min is None else float(fit_s_min)
        self.fit_s_max = None if fit_s_max is None else float(fit_s_max)
        self.positive_only = bool(positive_only)
        self._xi_linear_matter = None if xi_linear_matter is None else np.asarray(xi_linear_matter, dtype=np.float64)

    @classmethod
    def from_paircount_file(
        cls,
        path: str | Path,
        *,
        cosmology_index: int,
        z: float,
        engine: str = "camb",
        fit_s_min: float | None = None,
        fit_s_max: float | None = None,
        positive_only: bool = True,
    ) -> "HODLinearBiasTabulator":
        tabulator = HODClusteringTabulator.from_paircount_file(path)
        return cls(
            tabulator,
            cosmology_index=cosmology_index,
            z=z,
            engine=engine,
            fit_s_min=fit_s_min,
            fit_s_max=fit_s_max,
            positive_only=positive_only,
        )

    @property
    def s_edges(self) -> np.ndarray:
        return self.clustering_tabulator.paircounts.bins["s_edges"]

    @property
    def s_mid(self) -> np.ndarray:
        edges = self.s_edges
        return 0.5 * (edges[:-1] + edges[1:])

    def linear_matter_xi(self) -> np.ndarray:
        if self._xi_linear_matter is None:
            self._xi_linear_matter = linear_matter_xi_abacus(
                self.s_mid,
                z=self.z,
                cosmology_index=self.cosmology_index,
                engine=self.engine,
            )
        return self._xi_linear_matter

    def xi0_real(self, hod_params: Mapping[str, Any], *, hod_model: str = "lrg") -> np.ndarray:
        result = self.clustering_tabulator.correlation(hod_params, hod_model=hod_model)
        return np.asarray(smu_multipoles(result, ells=(0,))[0], dtype=np.float64)

    def evaluate(
        self,
        hod_params: Mapping[str, Any],
        *,
        hod_model: str = "lrg",
        fit_s_min: float | None = None,
        fit_s_max: float | None = None,
        positive_only: bool | None = None,
    ) -> LinearBiasResult:
        s_mid = self.s_mid
        xi0 = self.xi0_real(hod_params, hod_model=hod_model)
        xi_linear = self.linear_matter_xi()
        lower = self.fit_s_min if fit_s_min is None else float(fit_s_min)
        upper = self.fit_s_max if fit_s_max is None else float(fit_s_max)
        use_positive = self.positive_only if positive_only is None else bool(positive_only)
        bias, bias_of_s, mask = _linear_bias_from_monopoles(
            s_mid,
            xi0,
            xi_linear,
            fit_s_min=lower,
            fit_s_max=upper,
            positive_only=use_positive,
        )
        return LinearBiasResult(
            bias=float(bias[0]),
            s_mid=s_mid,
            xi0_real=xi0,
            xi_linear_matter=xi_linear,
            bias_of_s=bias_of_s[0],
            fit_mask=mask[0],
        )

    def evaluate_stellar_mass(
        self,
        hod_params: Mapping[str, Any],
        *,
        split_method: str,
        logmstar_edges: Any,
        redshift_weights: Any = None,
        hod_model: str = "lrg_stellar_mass",
        fit_s_min: float | None = None,
        fit_s_max: float | None = None,
        positive_only: bool | None = None,
    ) -> StellarMassLinearBiasResult:
        """Return linear bias for every stellar-mass split in one batch."""

        clustering = self.clustering_tabulator.stellar_mass_correlations(
            hod_params,
            split_method=split_method,
            logmstar_edges=logmstar_edges,
            redshift_weights=redshift_weights,
            hod_model=hod_model,
        )
        mu_edges = self.clustering_tabulator.paircounts.bins["mu_edges"]
        xi0 = np.sum(clustering.xi * np.diff(mu_edges)[None, None, :], axis=2)
        xi_linear = self.linear_matter_xi()
        lower = self.fit_s_min if fit_s_min is None else float(fit_s_min)
        upper = self.fit_s_max if fit_s_max is None else float(fit_s_max)
        use_positive = self.positive_only if positive_only is None else bool(positive_only)
        bias, bias_of_s, mask = _linear_bias_from_monopoles(
            self.s_mid,
            xi0,
            xi_linear,
            fit_s_min=lower,
            fit_s_max=upper,
            positive_only=use_positive,
        )
        return StellarMassLinearBiasResult(
            bias=bias,
            s_mid=self.s_mid,
            xi0_real=xi0,
            xi_linear_matter=xi_linear,
            bias_of_s=bias_of_s,
            fit_mask=mask,
            split_method=clustering.split_method,
            logmstar_edges=clustering.logmstar_edges,
            redshift_weights=clustering.redshift_weights,
        )

    def linear_bias(self, hod_params: Mapping[str, Any], *, hod_model: str = "lrg") -> float:
        return self.evaluate(hod_params, hod_model=hod_model).bias

    def stellar_mass_linear_bias(
        self,
        hod_params: Mapping[str, Any],
        **split_options: Any,
    ) -> np.ndarray:
        """Return one linear-bias value per stellar-mass split."""

        return self.evaluate_stellar_mass(hod_params, **split_options).bias
