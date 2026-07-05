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
        n_subbins: int = 20,
        cosmology_index: int,
        z: float,
        engine: str = "camb",
        fit_s_min: float | None = None,
        fit_s_max: float | None = None,
        positive_only: bool = True,
    ) -> "HODLinearBiasTabulator":
        tabulator = HODClusteringTabulator.from_paircount_file(path, n_subbins=n_subbins)
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
        ratio = np.divide(xi0, xi_linear, out=np.full_like(xi0, np.nan), where=xi_linear != 0.0)
        bias_of_s = np.full_like(ratio, np.nan, dtype=np.float64)
        valid_ratio = np.isfinite(ratio) & (ratio >= 0.0)
        bias_of_s[valid_ratio] = np.sqrt(ratio[valid_ratio])

        lower = self.fit_s_min if fit_s_min is None else float(fit_s_min)
        upper = self.fit_s_max if fit_s_max is None else float(fit_s_max)
        mask = np.ones_like(s_mid, dtype=bool)
        if lower is not None:
            mask &= s_mid >= lower
        if upper is not None:
            mask &= s_mid <= upper
        mask &= np.isfinite(bias_of_s)
        use_positive = self.positive_only if positive_only is None else bool(positive_only)
        if use_positive:
            mask &= (xi0 > 0.0) & (xi_linear > 0.0)

        if not np.any(mask):
            raise ValueError("No valid bins are available for the configured linear-bias fit range.")
        bias = float(np.mean(bias_of_s[mask]))
        return LinearBiasResult(
            bias=bias,
            s_mid=s_mid,
            xi0_real=xi0,
            xi_linear_matter=xi_linear,
            bias_of_s=bias_of_s,
            fit_mask=mask,
        )

    def linear_bias(self, hod_params: Mapping[str, Any], *, hod_model: str = "lrg") -> float:
        return self.evaluate(hod_params, hod_model=hod_model).bias
