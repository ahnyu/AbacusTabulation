"""Convert tabulated pair counts into HOD-weighted galaxy clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .hod import evaluate_hod


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
        )


def _mass_subcenters_log10(edges: np.ndarray, n_subbins: int) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    n_subbins = int(n_subbins)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("mass edges must be one-dimensional with at least two entries.")
    if n_subbins <= 0:
        raise ValueError("n_subbins must be positive.")
    centers = (np.arange(n_subbins, dtype=np.float64) + 0.5) / n_subbins
    return edges[:-1, None] + np.diff(edges)[:, None] * centers[None, :]


def _hod_weights_from_subcenters(
    mass_edges_log10: np.ndarray,
    mass_subcenters_log10: np.ndarray,
    num_halo: np.ndarray,
    num_particle: np.ndarray,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str,
) -> HODBinWeights:
    edges = np.asarray(mass_edges_log10, dtype=np.float64)
    subcenters = np.asarray(mass_subcenters_log10, dtype=np.float64)
    num_halo = np.asarray(num_halo, dtype=np.float64)
    num_particle = np.asarray(num_particle, dtype=np.float64)

    cen_sub, sat_sub = evaluate_hod(10.0**subcenters, hod_params, model=hod_model)
    central = np.mean(np.asarray(cen_sub, dtype=np.float64), axis=1)
    satellite = np.mean(np.asarray(sat_sub, dtype=np.float64), axis=1)
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
    )


def refined_hod_bin_weights(
    mass_edges_log10: np.ndarray,
    num_halo: np.ndarray,
    num_particle: np.ndarray,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
    n_subbins: int = 20,
    subbin_weighting: str = "uniform_log",
) -> HODBinWeights:
    """Average HOD occupations over log-uniform subbins inside each mass bin."""

    if subbin_weighting != "uniform_log":
        raise ValueError("Only subbin_weighting='uniform_log' is currently supported.")
    edges = np.asarray(mass_edges_log10, dtype=np.float64)
    subcenters = _mass_subcenters_log10(edges, n_subbins)
    return _hod_weights_from_subcenters(
        edges,
        subcenters,
        num_halo,
        num_particle,
        hod_params,
        hod_model=hod_model,
    )


def hod_weights_for_paircounts(
    paircounts: PairCountTable,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
    n_subbins: int = 20,
) -> HODBinWeights:
    """Compute refined HOD weights matching a paircount table's mass bins."""

    return refined_hod_bin_weights(
        paircounts.mass_edges_log10,
        paircounts.num_halo,
        paircounts.num_particle,
        hod_params,
        hod_model=hod_model,
        n_subbins=n_subbins,
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

    def __init__(self, paircounts: PairCountTable, *, n_subbins: int = 20):
        self.paircounts = paircounts
        self.n_subbins = int(n_subbins)
        self.mass_subcenters_log10 = _mass_subcenters_log10(
            paircounts.mass_edges_log10,
            self.n_subbins,
        )
        self.bin_shape = tuple(paircounts.counts_hh.shape[2:])
        self.random_geometry = random_geometry_factor(paircounts)
        self._hh = self._flatten_counts(paircounts.counts_hh)
        self._hp = self._flatten_counts(paircounts.counts_hp)
        self._pp = self._flatten_counts(paircounts.counts_pp)

    @classmethod
    def from_paircount_file(
        cls,
        path: str | Path,
        *,
        n_subbins: int = 20,
    ) -> "HODClusteringTabulator":
        return cls(read_paircounts(path), n_subbins=n_subbins)

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
    n_subbins: int = 20,
) -> GalaxyClusteringResult:
    """High-level auto-correlation helper returning HOD-weighted ``xi``."""

    tabulator = HODClusteringTabulator.from_paircount_file(paircount_path, n_subbins=n_subbins)
    return tabulator.correlation(hod_params, hod_model=hod_model)


def galaxy_cross_correlation_from_paircounts(
    paircount_path: str | Path,
    hod_params_a: Mapping[str, Any],
    hod_params_b: Mapping[str, Any],
    *,
    hod_model_a: str = "lrg",
    hod_model_b: str = "lrg",
    n_subbins: int = 20,
) -> GalaxyClusteringResult:
    """High-level cross-correlation helper for two HOD parameter sets."""

    tabulator = HODClusteringTabulator.from_paircount_file(paircount_path, n_subbins=n_subbins)
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


def _z_directory(z_mock: float) -> str:
    return f"z{float(z_mock):.3f}"


def _format_config_path(value: str | Path, *, sim_name: str, z_mock: float) -> Path:
    return Path(
        str(value).format(
            sim_name=sim_name,
            z=_z_directory(z_mock),
            z_mock=float(z_mock),
        )
    )


def _sanitize_filename_piece(value: object) -> str:
    clean = []
    for char in str(value):
        if char.isalnum():
            clean.append(char)
        else:
            clean.append("-")
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
    raise ValueError(
        f"Found multiple matching paircount files in {output_dir}; set paircounts.file_tag."
    )


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
    paths_params = config.get("paths", {})
    pair_params = config.get("paircounts", {})
    mass_params = pair_params.get("mass", {})
    hod_params = config.get("hod", {})
    if "params" not in hod_params:
        raise KeyError("Set hod.params in the config, or call galaxy_correlation_from_paircounts directly.")

    sim_name = str(sim_params.get("sim_name", ""))
    z_mock = float(sim_params.get("z_mock", 0.0))
    configured_modes = pair_params.get("clustering", ["rppi"])
    if clustering is not None:
        mode = str(clustering)
    elif isinstance(configured_modes, list):
        mode = str(configured_modes[0])
    else:
        mode = str(configured_modes)
    if "," in mode:
        mode = mode.split(",", 1)[0].strip()
    pos_dataset = str(position_dataset or pair_params.get("position_dataset", "pos"))
    nmass_bins = int(mass_params.get("nmass_bins", pair_params.get("nmass_bins", 20)))

    if paircount_path is None:
        output_value = pair_params.get("output_dir", paths_params.get("paircounts_dir"))
        if output_value is None:
            raise KeyError("Set paircounts.output_dir or paths.paircounts_dir.")
        output_dir = _format_config_path(output_value, sim_name=sim_name, z_mock=z_mock)
        paircount_path = _find_paircount_file(
            output_dir,
            clustering=mode,
            position_dataset=pos_dataset,
            file_tag=pair_params.get("file_tag"),
            nmass_bins=nmass_bins,
        )

    return galaxy_correlation_from_paircounts(
        paircount_path,
        hod_params["params"],
        hod_model=str(hod_params.get("model", "lrg")),
        n_subbins=int(hod_params.get("n_subbins", 20)),
    )

