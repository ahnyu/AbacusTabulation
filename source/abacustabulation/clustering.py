"""Convert tabulated pair counts into HOD-weighted galaxy clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .hod import evaluate_hod, evaluate_hod_splits
from .hod_models.lrg_stellar_mass import normalize_stellar_mass_selection


@dataclass
class PairCountTable:
    clustering: str
    counts_hh: np.ndarray
    counts_hp: np.ndarray
    counts_pp: np.ndarray
    mass_edges_log10: np.ndarray
    mass_mean_log10: np.ndarray
    num_halo: np.ndarray
    num_particle: np.ndarray
    bins: dict[str, np.ndarray]
    attrs: dict[str, Any]
    mass_subbin_edges_log10: np.ndarray | None = None
    mass_subbin_centers_log10: np.ndarray | None = None
    num_halo_subbin: np.ndarray | None = None


@dataclass
class HODBinWeights:
    central: np.ndarray
    satellite: np.ndarray
    particle: np.ndarray
    mass_edges_log10: np.ndarray
    mass_subcenters_log10: np.ndarray
    n_galaxies: float
    n_centrals: float
    n_satellites: float
    subbin_weighting: str = "halo_count"


@dataclass
class GalaxyClusteringResult:
    xi: np.ndarray
    dd: np.ndarray
    rr: np.ndarray
    weights: HODBinWeights
    paircounts: PairCountTable
    n_galaxies: float
    number_density: float
    weights_b: HODBinWeights | None = None
    n_galaxies_b: float | None = None
    number_density_b: float | None = None


@dataclass
class StellarMassClusteringResult:
    """Auto-clustering results for every stellar-mass split in edge order."""

    clusterings: tuple[GalaxyClusteringResult, ...]
    split_method: str
    logmstar_edges: np.ndarray
    redshift_weights: np.ndarray

    def __len__(self) -> int:
        return len(self.clusterings)

    def __getitem__(self, index: int) -> GalaxyClusteringResult:
        return self.clusterings[index]

    @property
    def n_splits(self) -> int:
        return len(self.clusterings)

    @property
    def xi(self) -> np.ndarray:
        return np.stack([result.xi for result in self.clusterings], axis=0)

    @property
    def dd(self) -> np.ndarray:
        return np.stack([result.dd for result in self.clusterings], axis=0)

    @property
    def rr(self) -> np.ndarray:
        return np.stack([result.rr for result in self.clusterings], axis=0)

    @property
    def weights(self) -> tuple[HODBinWeights, ...]:
        return tuple(result.weights for result in self.clusterings)

    @property
    def n_galaxies(self) -> np.ndarray:
        return np.asarray([result.n_galaxies for result in self.clusterings])

    @property
    def number_density(self) -> np.ndarray:
        return np.asarray([result.number_density for result in self.clusterings])


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_paircounts(path: str | Path) -> PairCountTable:
    """Read a paircount HDF5 file produced by ``compute_paircounts.py``."""

    path = Path(path)
    with h5py.File(path, "r") as handle:
        attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
        counts = handle["counts"]
        mass = handle["mass"]
        bins_group = handle["bins"]
        bins = {key: bins_group[key][...] for key in bins_group.keys()}
        mass_subbin_edges = (
            mass["halo_subbin_edges_log10"][...].astype(np.float64)
            if "halo_subbin_edges_log10" in mass
            else None
        )
        mass_subbin_centers = (
            mass["halo_subbin_centers_log10"][...].astype(np.float64)
            if "halo_subbin_centers_log10" in mass
            else None
        )
        num_halo_subbin = (
            mass["num_halo_subbin"][...].astype(np.float64)
            if "num_halo_subbin" in mass
            else None
        )
        return PairCountTable(
            clustering=str(attrs["clustering"]),
            counts_hh=counts["HH"][...].astype(np.float64),
            counts_hp=counts["HP"][...].astype(np.float64),
            counts_pp=counts["PP"][...].astype(np.float64),
            mass_edges_log10=mass["edges_log10"][...].astype(np.float64),
            mass_mean_log10=mass["mean_log10"][...].astype(np.float64),
            num_halo=mass["num_halo"][...].astype(np.float64),
            num_particle=mass["num_particle"][...].astype(np.float64),
            bins=bins,
            attrs=attrs,
            mass_subbin_edges_log10=mass_subbin_edges,
            mass_subbin_centers_log10=mass_subbin_centers,
            num_halo_subbin=num_halo_subbin,
        )


def _hod_weights_from_subcenters(
    mass_edges_log10: np.ndarray,
    mass_subcenters_log10: np.ndarray,
    num_halo: np.ndarray,
    num_particle: np.ndarray,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str,
    num_halo_subbin: np.ndarray,
) -> HODBinWeights:
    edges = np.asarray(mass_edges_log10, dtype=np.float64)
    subcenters = np.asarray(mass_subcenters_log10, dtype=np.float64)
    num_halo = np.asarray(num_halo, dtype=np.float64)
    num_particle = np.asarray(num_particle, dtype=np.float64)

    cen_sub, sat_sub = evaluate_hod(10.0**subcenters, hod_params, model=hod_model)
    cen_sub = np.asarray(cen_sub, dtype=np.float64)
    sat_sub = np.asarray(sat_sub, dtype=np.float64)
    counts = np.asarray(num_halo_subbin, dtype=np.float64)
    if counts.shape != subcenters.shape:
        raise ValueError(
            f"num_halo_subbin shape {counts.shape} does not match HOD subcenters {subcenters.shape}."
        )
    sub_totals = np.sum(counts, axis=1)
    if not np.allclose(sub_totals, num_halo):
        raise ValueError("num_halo_subbin does not sum to the coarse mass-bin num_halo values.")
    central = np.divide(
        np.sum(cen_sub * counts, axis=1),
        sub_totals,
        out=np.zeros_like(sub_totals, dtype=np.float64),
        where=sub_totals > 0.0,
    )
    satellite = np.divide(
        np.sum(sat_sub * counts, axis=1),
        sub_totals,
        out=np.zeros_like(sub_totals, dtype=np.float64),
        where=sub_totals > 0.0,
    )
    particle = np.divide(
        satellite * num_halo,
        num_particle,
        out=np.zeros_like(satellite, dtype=np.float64),
        where=num_particle > 0.0,
    )
    n_centrals = float(np.sum(num_halo * central))
    n_satellites = float(np.sum(num_halo * satellite))
    return HODBinWeights(
        central=central,
        satellite=satellite,
        particle=particle,
        mass_edges_log10=edges,
        mass_subcenters_log10=subcenters,
        n_galaxies=n_centrals + n_satellites,
        n_centrals=n_centrals,
        n_satellites=n_satellites,
        subbin_weighting="halo_count",
    )


def _stellar_mass_weights_from_subcenters(
    mass_edges_log10: np.ndarray,
    mass_subcenters_log10: np.ndarray,
    num_halo: np.ndarray,
    num_particle: np.ndarray,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str,
    split_method: str,
    logmstar_edges: Any,
    redshift_weights: Any,
    num_halo_subbin: np.ndarray,
) -> tuple[HODBinWeights, ...]:
    """Build halo-count-weighted HOD weights for every stellar-mass split."""

    edges = np.asarray(mass_edges_log10, dtype=np.float64)
    subcenters = np.asarray(mass_subcenters_log10, dtype=np.float64)
    num_halo = np.asarray(num_halo, dtype=np.float64)
    num_particle = np.asarray(num_particle, dtype=np.float64)
    counts = np.asarray(num_halo_subbin, dtype=np.float64)
    if counts.shape != subcenters.shape:
        raise ValueError(
            f"num_halo_subbin shape {counts.shape} does not match HOD "
            f"subcenters {subcenters.shape}."
        )
    sub_totals = np.sum(counts, axis=1)
    if not np.allclose(sub_totals, num_halo):
        raise ValueError(
            "num_halo_subbin does not sum to the coarse mass-bin num_halo values."
        )

    central_sub, satellite_sub = evaluate_hod_splits(
        10.0**subcenters,
        hod_params,
        model=hod_model,
        split_method=split_method,
        logmstar_edges=logmstar_edges,
        redshift_weights=redshift_weights,
    )
    central_sub = np.asarray(central_sub, dtype=np.float64)
    satellite_sub = np.asarray(satellite_sub, dtype=np.float64)
    expected_tail = subcenters.shape
    if (
        central_sub.ndim != subcenters.ndim + 1
        or central_sub.shape[1:] != expected_tail
        or satellite_sub.shape != central_sub.shape
    ):
        raise ValueError(
            "Split HOD occupations must have shape "
            f"(n_splits, {expected_tail}), got central {central_sub.shape} "
            f"and satellite {satellite_sub.shape}."
        )
    if (
        not np.all(np.isfinite(central_sub))
        or not np.all(np.isfinite(satellite_sub))
        or np.any(central_sub < 0.0)
        or np.any(satellite_sub < 0.0)
    ):
        raise ValueError("Split HOD occupations must be finite and non-negative.")

    weighted_counts = counts[None, ...]
    denominator = sub_totals[None, :]
    central = np.divide(
        np.sum(central_sub * weighted_counts, axis=2),
        denominator,
        out=np.zeros((central_sub.shape[0], subcenters.shape[0]), dtype=np.float64),
        where=denominator > 0.0,
    )
    satellite = np.divide(
        np.sum(satellite_sub * weighted_counts, axis=2),
        denominator,
        out=np.zeros_like(central),
        where=denominator > 0.0,
    )
    particle = np.divide(
        satellite * num_halo[None, :],
        num_particle[None, :],
        out=np.zeros_like(satellite),
        where=num_particle[None, :] > 0.0,
    )
    n_centrals = central @ num_halo
    n_satellites = satellite @ num_halo

    return tuple(
        HODBinWeights(
            central=central[index],
            satellite=satellite[index],
            particle=particle[index],
            mass_edges_log10=edges,
            mass_subcenters_log10=subcenters,
            n_galaxies=float(n_centrals[index] + n_satellites[index]),
            n_centrals=float(n_centrals[index]),
            n_satellites=float(n_satellites[index]),
            subbin_weighting="halo_count",
        )
        for index in range(central.shape[0])
    )


def refined_hod_bin_weights(
    mass_edges_log10: np.ndarray,
    num_halo: np.ndarray,
    num_particle: np.ndarray,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
    mass_subcenters_log10: np.ndarray,
    num_halo_subbin: np.ndarray,
) -> HODBinWeights:
    """Average HOD occupations over halo-count-weighted subbins inside each mass bin."""

    edges = np.asarray(mass_edges_log10, dtype=np.float64)
    counts = np.asarray(num_halo_subbin, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"num_halo_subbin must be 2D, got shape {counts.shape}.")
    subcenters = np.asarray(mass_subcenters_log10, dtype=np.float64)
    if subcenters.shape != counts.shape:
        raise ValueError(f"mass subbin centers shape {subcenters.shape} does not match counts {counts.shape}.")
    return _hod_weights_from_subcenters(
        edges,
        subcenters,
        num_halo,
        num_particle,
        hod_params,
        hod_model=hod_model,
        num_halo_subbin=counts,
    )


def _paircount_subbin_data(paircounts: PairCountTable) -> tuple[np.ndarray, np.ndarray]:
    if paircounts.num_halo_subbin is None:
        raise ValueError(
            "Paircount table is missing mass/num_halo_subbin. Recompute paircounts with the current "
            "code so HOD subbin refinement can use halo-count weights."
        )
    counts = np.asarray(paircounts.num_halo_subbin, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"num_halo_subbin must be 2D, got shape {counts.shape}.")
    attr_n = paircounts.attrs.get("mass_n_subbins")
    if attr_n is not None and int(attr_n) != counts.shape[1]:
        raise ValueError(
            f"Paircount attr mass_n_subbins={int(attr_n)} does not match mass/num_halo_subbin "
            f"shape {counts.shape}."
        )
    centers = paircounts.mass_subbin_centers_log10
    if centers is None:
        raise ValueError(
            "Paircount table is missing mass/halo_subbin_centers_log10. Recompute paircounts with the "
            "current code so HOD subbin refinement can use stored halo-count subbins."
        )
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != counts.shape:
        raise ValueError(f"mass subbin centers shape {centers.shape} does not match counts {counts.shape}.")
    return centers, counts


def hod_weights_for_paircounts(
    paircounts: PairCountTable,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> HODBinWeights:
    """Compute refined HOD weights matching a paircount table's mass bins."""

    subcenters, subcounts = _paircount_subbin_data(paircounts)
    return refined_hod_bin_weights(
        paircounts.mass_edges_log10,
        paircounts.num_halo,
        paircounts.num_particle,
        hod_params,
        hod_model=hod_model,
        mass_subcenters_log10=subcenters,
        num_halo_subbin=subcounts,
    )


def weighted_galaxy_paircounts(
    paircounts: PairCountTable,
    weights_a: HODBinWeights,
    weights_b: HODBinWeights | None = None,
) -> np.ndarray:
    """Combine HH, HP, and PP tables into HOD-weighted galaxy DD counts.

    If ``weights_b`` is omitted, the old auto-correlation convention is used:
    ``DD = HH*C*C + 2*HP*C*S_particle + PP*S_particle*S_particle``.
    For cross-correlations, central-satellite terms are explicitly computed in
    both component directions.
    """

    hh = paircounts.counts_hh
    hp = paircounts.counts_hp
    pp = paircounts.counts_pp
    ca = weights_a.central
    pa = weights_a.particle

    if weights_b is None:
        dd_hh = np.einsum("ij...,i,j->...", hh, ca, ca, optimize=True)
        dd_hp = np.einsum("ij...,i,j->...", hp, ca, pa, optimize=True)
        dd_pp = np.einsum("ij...,i,j->...", pp, pa, pa, optimize=True)
        return dd_hh + 2.0 * dd_hp + dd_pp

    cb = weights_b.central
    pb = weights_b.particle
    dd_hh = np.einsum("ij...,i,j->...", hh, ca, cb, optimize=True)
    dd_hp_ab = np.einsum("ij...,i,j->...", hp, ca, pb, optimize=True)
    dd_hp_ba = np.einsum("ji...,i,j->...", hp, pa, cb, optimize=True)
    dd_pp = np.einsum("ij...,i,j->...", pp, pa, pb, optimize=True)
    return dd_hh + dd_hp_ab + dd_hp_ba + dd_pp


def rppi_bin_volumes(rp_edges: np.ndarray, pi_edges: np.ndarray) -> np.ndarray:
    """Return cylindrical rp-pi bin volumes for |pi| bins."""

    rp_edges = np.asarray(rp_edges, dtype=np.float64)
    pi_edges = np.asarray(pi_edges, dtype=np.float64)
    d_rp2 = np.diff(rp_edges**2)
    d_pi = np.diff(pi_edges)
    return np.pi * d_rp2[:, None] * (2.0 * d_pi[None, :])


def smu_bin_volumes(s_edges: np.ndarray, mu_edges: np.ndarray) -> np.ndarray:
    """Return spherical-shell volumes for absolute-mu bins."""

    s_edges = np.asarray(s_edges, dtype=np.float64)
    mu_edges = np.asarray(mu_edges, dtype=np.float64)
    d_s3 = np.diff(s_edges**3)
    d_mu = np.diff(mu_edges)
    return (4.0 * np.pi / 3.0) * d_s3[:, None] * d_mu[None, :]


def random_geometry_factor(paircounts: PairCountTable) -> np.ndarray:
    """Return periodic-box random geometry factor, bin_volume / box_volume."""

    boxsize = float(paircounts.attrs["boxsize"])
    box_volume = boxsize**3
    if paircounts.clustering == "rppi":
        bin_volume = rppi_bin_volumes(paircounts.bins["rp_edges"], paircounts.bins["pi_edges"])
    elif paircounts.clustering == "smu":
        bin_volume = smu_bin_volumes(paircounts.bins["s_edges"], paircounts.bins["mu_edges"])
    else:
        raise ValueError(f"Unknown clustering type {paircounts.clustering!r}.")
    return bin_volume / box_volume


def analytic_random_paircounts(
    paircounts: PairCountTable,
    n_galaxies_a: float,
    n_galaxies_b: float | None = None,
) -> np.ndarray:
    """Analytic random pair counts for a periodic box."""

    if n_galaxies_b is None:
        norm = float(n_galaxies_a) ** 2
    else:
        norm = float(n_galaxies_a) * float(n_galaxies_b)
    return norm * random_geometry_factor(paircounts)


class HODClusteringTabulator:
    """In-memory paircount table for repeated HOD evaluations."""

    def __init__(self, paircounts: PairCountTable):
        self.paircounts = paircounts
        self.mass_subcenters_log10, self.num_halo_subbin = _paircount_subbin_data(paircounts)
        self.n_subbins = int(self.num_halo_subbin.shape[1])
        self.bin_shape = tuple(paircounts.counts_hh.shape[2:])
        self.random_geometry = random_geometry_factor(paircounts)
        self._hh = self._flatten_counts(paircounts.counts_hh)
        self._hp = self._flatten_counts(paircounts.counts_hp)
        self._pp = self._flatten_counts(paircounts.counts_pp)

    @classmethod
    def from_paircount_file(cls, path: str | Path) -> "HODClusteringTabulator":
        return cls(read_paircounts(path))

    @staticmethod
    def _flatten_counts(counts: np.ndarray) -> np.ndarray:
        nmass = counts.shape[0]
        return np.ascontiguousarray(counts.reshape(nmass * nmass, -1), dtype=np.float64)

    def hod_weights(
        self,
        hod_params: Mapping[str, Any],
        *,
        hod_model: str = "lrg",
    ) -> HODBinWeights:
        return _hod_weights_from_subcenters(
            self.paircounts.mass_edges_log10,
            self.mass_subcenters_log10,
            self.paircounts.num_halo,
            self.paircounts.num_particle,
            hod_params,
            hod_model=hod_model,
            num_halo_subbin=self.num_halo_subbin,
        )

    def stellar_mass_hod_weights(
        self,
        hod_params: Mapping[str, Any],
        *,
        split_method: str,
        logmstar_edges: Any,
        redshift_weights: Any = None,
        hod_model: str = "lrg_stellar_mass",
    ) -> tuple[HODBinWeights, ...]:
        """Return HOD weights for all raw or relative stellar-mass splits."""

        return _stellar_mass_weights_from_subcenters(
            self.paircounts.mass_edges_log10,
            self.mass_subcenters_log10,
            self.paircounts.num_halo,
            self.paircounts.num_particle,
            hod_params,
            hod_model=hod_model,
            split_method=split_method,
            logmstar_edges=logmstar_edges,
            redshift_weights=redshift_weights,
            num_halo_subbin=self.num_halo_subbin,
        )

    def _weighted_auto_paircounts_batch(
        self,
        weights: tuple[HODBinWeights, ...],
    ) -> np.ndarray:
        if not weights:
            raise ValueError("At least one stellar-mass split is required.")
        central = np.stack([item.central for item in weights], axis=0)
        particle = np.stack([item.particle for item in weights], axis=0)
        w_hh = (central[:, :, None] * central[:, None, :]).reshape(len(weights), -1)
        w_hp = (2.0 * central[:, :, None] * particle[:, None, :]).reshape(
            len(weights), -1
        )
        w_pp = (particle[:, :, None] * particle[:, None, :]).reshape(len(weights), -1)
        dd = w_hh @ self._hh + w_hp @ self._hp + w_pp @ self._pp
        return dd.reshape((len(weights), *self.bin_shape))

    def stellar_mass_correlations(
        self,
        hod_params: Mapping[str, Any],
        *,
        split_method: str,
        logmstar_edges: Any,
        redshift_weights: Any = None,
        hod_model: str = "lrg_stellar_mass",
    ) -> StellarMassClusteringResult:
        """Return auto-clustering for every stellar-mass split in edge order."""

        normalized_edges, normalized_redshift_weights = (
            normalize_stellar_mass_selection(
                split_method,
                logmstar_edges,
                redshift_weights,
            )
        )
        weights = self.stellar_mass_hod_weights(
            hod_params,
            split_method=split_method,
            logmstar_edges=logmstar_edges,
            redshift_weights=redshift_weights,
            hod_model=hod_model,
        )
        expected_splits = normalized_edges.shape[1] - 1
        if len(weights) != expected_splits:
            raise ValueError(
                f"The HOD returned {len(weights)} splits for {expected_splits} intervals."
            )

        dd = self._weighted_auto_paircounts_batch(weights)
        n_galaxies = np.asarray([item.n_galaxies for item in weights])
        normalization_shape = (len(weights),) + (1,) * len(self.bin_shape)
        rr = (n_galaxies**2).reshape(normalization_shape) * self.random_geometry[None, ...]
        xi = np.divide(
            dd,
            rr,
            out=np.full_like(dd, np.nan, dtype=np.float64),
            where=rr > 0.0,
        ) - 1.0
        box_volume = float(self.paircounts.attrs["boxsize"]) ** 3
        clusterings = tuple(
            GalaxyClusteringResult(
                xi=xi[index],
                dd=dd[index],
                rr=rr[index],
                weights=weights[index],
                paircounts=self.paircounts,
                n_galaxies=float(n_galaxies[index]),
                number_density=float(n_galaxies[index] / box_volume),
            )
            for index in range(len(weights))
        )
        normalized_method = str(split_method).lower()
        output_edges = (
            normalized_edges[0] if normalized_method == "raw" else normalized_edges
        )
        return StellarMassClusteringResult(
            clusterings=clusterings,
            split_method=normalized_method,
            logmstar_edges=output_edges,
            redshift_weights=normalized_redshift_weights,
        )

    def weighted_paircounts(
        self,
        weights_a: HODBinWeights,
        weights_b: HODBinWeights | None = None,
    ) -> np.ndarray:
        ca = weights_a.central
        pa = weights_a.particle
        if weights_b is None:
            w_hh = np.outer(ca, ca).ravel()
            w_hp = (2.0 * np.outer(ca, pa)).ravel()
            w_pp = np.outer(pa, pa).ravel()
        else:
            cb = weights_b.central
            pb = weights_b.particle
            w_hh = np.outer(ca, cb).ravel()
            w_hp = (np.outer(ca, pb) + np.outer(cb, pa)).ravel()
            w_pp = np.outer(pa, pb).ravel()

        dd = w_hh @ self._hh + w_hp @ self._hp + w_pp @ self._pp
        return dd.reshape(self.bin_shape)

    def correlation(
        self,
        hod_params: Mapping[str, Any],
        *,
        hod_model: str = "lrg",
    ) -> GalaxyClusteringResult:
        weights = self.hod_weights(hod_params, hod_model=hod_model)
        dd = self.weighted_paircounts(weights)
        rr = weights.n_galaxies**2 * self.random_geometry
        xi = np.divide(dd, rr, out=np.full_like(dd, np.nan, dtype=np.float64), where=rr > 0.0) - 1.0
        boxsize = float(self.paircounts.attrs["boxsize"])
        return GalaxyClusteringResult(
            xi=xi,
            dd=dd,
            rr=rr,
            weights=weights,
            paircounts=self.paircounts,
            n_galaxies=weights.n_galaxies,
            number_density=weights.n_galaxies / boxsize**3,
        )

    def cross_correlation(
        self,
        hod_params_a: Mapping[str, Any],
        hod_params_b: Mapping[str, Any],
        *,
        hod_model_a: str = "lrg",
        hod_model_b: str = "lrg",
    ) -> GalaxyClusteringResult:
        weights_a = self.hod_weights(hod_params_a, hod_model=hod_model_a)
        weights_b = self.hod_weights(hod_params_b, hod_model=hod_model_b)
        dd = self.weighted_paircounts(weights_a, weights_b)
        rr = weights_a.n_galaxies * weights_b.n_galaxies * self.random_geometry
        xi = np.divide(dd, rr, out=np.full_like(dd, np.nan, dtype=np.float64), where=rr > 0.0) - 1.0
        boxsize = float(self.paircounts.attrs["boxsize"])
        return GalaxyClusteringResult(
            xi=xi,
            dd=dd,
            rr=rr,
            weights=weights_a,
            weights_b=weights_b,
            paircounts=self.paircounts,
            n_galaxies=weights_a.n_galaxies,
            n_galaxies_b=weights_b.n_galaxies,
            number_density=weights_a.n_galaxies / boxsize**3,
            number_density_b=weights_b.n_galaxies / boxsize**3,
        )


def galaxy_correlation_from_paircounts(
    paircount_path: str | Path,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> GalaxyClusteringResult:
    """High-level auto-correlation helper returning HOD-weighted ``xi``."""

    tabulator = HODClusteringTabulator.from_paircount_file(paircount_path)
    return tabulator.correlation(hod_params, hod_model=hod_model)


def stellar_mass_correlations_from_paircounts(
    paircount_path: str | Path,
    hod_params: Mapping[str, Any],
    *,
    split_method: str,
    logmstar_edges: Any,
    redshift_weights: Any = None,
    hod_model: str = "lrg_stellar_mass",
) -> StellarMassClusteringResult:
    """Load one paircount table and evaluate every stellar-mass split."""

    tabulator = HODClusteringTabulator.from_paircount_file(paircount_path)
    return tabulator.stellar_mass_correlations(
        hod_params,
        split_method=split_method,
        logmstar_edges=logmstar_edges,
        redshift_weights=redshift_weights,
        hod_model=hod_model,
    )


def galaxy_cross_correlation_from_paircounts(
    paircount_path: str | Path,
    hod_params_a: Mapping[str, Any],
    hod_params_b: Mapping[str, Any],
    *,
    hod_model_a: str = "lrg",
    hod_model_b: str = "lrg",
) -> GalaxyClusteringResult:
    """High-level cross-correlation helper for two HOD parameter sets."""

    tabulator = HODClusteringTabulator.from_paircount_file(paircount_path)
    return tabulator.cross_correlation(
        hod_params_a,
        hod_params_b,
        hod_model_a=hod_model_a,
        hod_model_b=hod_model_b,
    )


def projected_wp(result: GalaxyClusteringResult) -> np.ndarray:
    """Project an rp-pi result to wp(rp)."""

    if result.paircounts.clustering != "rppi":
        raise ValueError("projected_wp requires rppi paircounts.")
    pi_edges = result.paircounts.bins["pi_edges"]
    return 2.0 * np.sum(result.xi * np.diff(pi_edges)[None, :], axis=1)


def smu_multipoles(
    result: GalaxyClusteringResult,
    ells: tuple[int, ...] = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Compute simple mu-integrated multipoles from an smu result."""

    if result.paircounts.clustering != "smu":
        raise ValueError("smu_multipoles requires smu paircounts.")
    mu_edges = result.paircounts.bins["mu_edges"]
    mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = np.diff(mu_edges)
    out: dict[int, np.ndarray] = {}
    for ell in ells:
        if ell == 0:
            legendre = np.ones_like(mu)
        elif ell == 2:
            legendre = 0.5 * (3.0 * mu**2 - 1.0)
        elif ell == 4:
            legendre = (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
        else:
            raise ValueError("Only ell=0,2,4 are currently supported.")
        out[ell] = (2 * ell + 1) * np.sum(result.xi * legendre[None, :] * dmu[None, :], axis=1)
    return out


def galaxy_correlation_from_config(
    path2config: str | Path,
    *,
    paircount_path: str | Path | None = None,
    clustering: str | None = None,
    position_dataset: str | None = None,
) -> GalaxyClusteringResult:
    """Read the universal YAML config and return HOD-weighted galaxy clustering."""

    from .config import load_config

    config = load_config(path2config)

    sim_params = config.get("sim_params", {})
    pair_params = config.get("paircounts", {})
    hod_params = config.get("hod", {})
    if "params" not in hod_params:
        raise KeyError("Set hod.params in the config, or call galaxy_correlation_from_paircounts directly.")

    jobs = pair_params.get("jobs") or {}
    job_params = dict(jobs.get("clustering", {})) if jobs else {}
    configured_modes = job_params.get("clustering", pair_params.get("clustering", ["rppi"]))
    if clustering is not None:
        mode = str(clustering)
    elif isinstance(configured_modes, list):
        mode = str(configured_modes[0])
    else:
        mode = str(configured_modes)
    if "," in mode:
        mode = mode.split(",", 1)[0].strip()

    if paircount_path is None:
        from .paircounts import resolve_paircount_path_from_config

        path_config: dict[str, Any] = {}
        if jobs:
            path_config["job"] = "clustering"
        if position_dataset is not None:
            path_config["position_dataset"] = position_dataset
        paircount_path = resolve_paircount_path_from_config(
            config,
            clustering=mode,
            path_config=path_config,
            job_name="clustering" if jobs else None,
        )

    return galaxy_correlation_from_paircounts(
        paircount_path,
        hod_params["params"],
        hod_model=str(hod_params.get("model", "lrg")),
    )

