"""High-resolution halo mass functions and HOD-derived summary quantities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .hod import evaluate_hod
from .paircounts import find_prepared_file_pairs, parse_logm_edges


@dataclass(frozen=True)
class HaloMassFunction:
    """Binned halo mass function in log10 halo mass."""

    logm_edges: np.ndarray
    logm_centers: np.ndarray
    num_halo: np.ndarray
    dndlog10m: np.ndarray
    volume: float
    attrs: dict[str, Any]

    @property
    def dlogm(self) -> np.ndarray:
        return np.diff(self.logm_edges)

    @property
    def density_per_bin(self) -> np.ndarray:
        return self.dndlog10m * self.dlogm


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_filename_piece(value: object) -> str:
    clean = []
    for char in str(value):
        clean.append(char if char.isalnum() else "-")
    return "".join(clean).strip("-") or "none"


def _z_directory(z_mock: float) -> str:
    return f"z{float(z_mock):.3f}"


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


def default_hmf_output_path(
    output_dir: str | Path,
    *,
    file_tag: str,
    seed: int | str,
    n_bins: int,
) -> Path:
    tag = _sanitize_filename_piece(file_tag)
    seed_tag = _sanitize_filename_piece(f"seed{seed}")
    return Path(output_dir) / f"hmf_{tag}_{seed_tag}_n{int(n_bins)}.h5"


def read_hmf(path: str | Path) -> HaloMassFunction:
    """Read an HMF HDF5 file produced by :func:`compute_hmf_from_prepared`."""

    path = Path(path)
    with h5py.File(path, "r") as handle:
        attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
        group = handle["hmf"]
        return HaloMassFunction(
            logm_edges=group["logm_edges"][...].astype(np.float64),
            logm_centers=group["logm_centers"][...].astype(np.float64),
            num_halo=group["num_halo"][...].astype(np.float64),
            dndlog10m=group["dndlog10m"][...].astype(np.float64),
            volume=float(attrs["volume"]),
            attrs=attrs,
        )


def write_hmf(
    hmf: HaloMassFunction,
    path: str | Path,
    *,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """Write a :class:`HaloMassFunction` to HDF5."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in hmf.attrs.items():
            if value is not None:
                handle.attrs[key] = value
        group = handle.create_group("hmf")
        kwargs = {} if compression is None else {"compression": compression}
        group.create_dataset("logm_edges", data=hmf.logm_edges, **kwargs)
        group.create_dataset("logm_centers", data=hmf.logm_centers, **kwargs)
        group.create_dataset("num_halo", data=hmf.num_halo.astype(np.int64), **kwargs)
        group.create_dataset("dndlog10m", data=hmf.dndlog10m, **kwargs)
    return path


def build_hmf_from_prepared(
    prepared_dir: str | Path,
    *,
    file_tag: str | None = None,
    seed: int | None = None,
    n_bins: int = 512,
    logm_min: float | None = None,
    logm_max: float | None = None,
    logm_edges: object | None = None,
) -> HaloMassFunction:
    """Build a high-resolution HMF from prepared halo slabs."""

    file_pairs = find_prepared_file_pairs(prepared_dir, file_tag=file_tag, seed=seed)
    masses: list[np.ndarray] = []
    lbox: float | None = None
    mpart: float | None = None
    first_attrs: dict[str, Any] = {}
    total_halos = 0

    for index, pair in enumerate(file_pairs):
        with h5py.File(pair.halo_file, "r") as handle:
            attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
            slab_lbox = float(attrs.get("Lbox", attrs.get("BoxSizeHMpc")))
            slab_mpart = float(attrs.get("Mpart", attrs.get("ParticleMassHMsun")))
            if index == 0:
                lbox = slab_lbox
                mpart = slab_mpart
                first_attrs = attrs
            else:
                if not np.isclose(slab_lbox, lbox):
                    raise ValueError(f"Inconsistent Lbox in {pair.halo_file}.")
                if not np.isclose(slab_mpart, mpart):
                    raise ValueError(f"Inconsistent Mpart in {pair.halo_file}.")
            halo_n = handle["halos"]["N"][...].astype(np.float64)
            halo_mass = halo_n * slab_mpart
            if np.any(halo_mass <= 0.0):
                raise ValueError(f"Non-positive halo mass found in {pair.halo_file}.")
            masses.append(halo_mass)
            total_halos += int(halo_mass.size)

    if total_halos == 0:
        raise ValueError("No halos found in prepared files.")

    log_mass = np.log10(np.concatenate(masses))
    edges = parse_logm_edges(logm_edges)
    if edges is None:
        lower = float(np.min(log_mass) if logm_min is None else logm_min)
        upper = float(np.max(log_mass) if logm_max is None else logm_max)
        if upper <= lower:
            raise ValueError("logm_max must be greater than logm_min.")
        edges = np.linspace(lower, np.nextafter(upper, np.inf), int(n_bins) + 1)
    else:
        n_bins = len(edges) - 1

    num_halo, _ = np.histogram(log_mass, bins=edges)
    dlogm = np.diff(edges)
    volume = float(lbox) ** 3
    dndlog10m = num_halo.astype(np.float64) / (volume * dlogm)
    tags = sorted({pair.tag for pair in file_pairs})
    seeds = sorted({pair.seed for pair in file_pairs})
    attrs = dict(first_attrs)
    attrs.update(
        {
            "volume": volume,
            "Lbox": float(lbox),
            "Mpart": float(mpart),
            "n_halos": int(total_halos),
            "n_bins": int(n_bins),
            "prepared_tags": ",".join(tags),
            "prepared_seeds": ",".join(str(item) for item in seeds),
            "prepared_slabs": ",".join(str(pair.slab) for pair in file_pairs),
            "schema_version": "hmf_v1",
        }
    )
    return HaloMassFunction(
        logm_edges=np.asarray(edges, dtype=np.float64),
        logm_centers=0.5 * (edges[:-1] + edges[1:]),
        num_halo=num_halo.astype(np.float64),
        dndlog10m=dndlog10m,
        volume=volume,
        attrs=attrs,
    )


def compute_hmf_from_prepared(
    *,
    prepared_dir: str | Path,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    file_tag: str | None = None,
    seed: int | None = None,
    n_bins: int = 512,
    logm_min: float | None = None,
    logm_max: float | None = None,
    logm_edges: object | None = None,
    overwrite: bool = False,
    hdf5_compression: str | None = None,
) -> Path:
    """Build and write a high-resolution HMF from prepared halo slabs."""

    hmf = build_hmf_from_prepared(
        prepared_dir,
        file_tag=file_tag,
        seed=seed,
        n_bins=n_bins,
        logm_min=logm_min,
        logm_max=logm_max,
        logm_edges=logm_edges,
    )
    if output_path is None:
        if output_dir is None:
            raise KeyError("Set output_path or output_dir when writing an HMF.")
        tags = str(hmf.attrs.get("prepared_tags", "prepared")).split(",")
        seeds = str(hmf.attrs.get("prepared_seeds", "mixed")).split(",")
        output_tag = file_tag or (tags[0] if len(tags) == 1 else "prepared")
        output_seed = seed if seed is not None else (seeds[0] if len(seeds) == 1 else "mixed")
        output_path = default_hmf_output_path(
            output_dir,
            file_tag=output_tag,
            seed=output_seed,
            n_bins=len(hmf.logm_centers),
        )
    return write_hmf(hmf, output_path, overwrite=overwrite, compression=hdf5_compression)


def hmf_from_config(
    path2config: str | Path,
    *,
    prepared_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    file_tag: str | None = None,
    seed: int | None = None,
    n_bins: int | None = None,
    logm_min: float | None = None,
    logm_max: float | None = None,
    logm_edges: object | None = None,
    overwrite: bool | None = None,
    hdf5_compression: str | None = None,
) -> Path:
    """Compute an HMF from the universal YAML config file."""

    from .config import load_config

    config = load_config(path2config)
    hmf_config = config.get("hmf", {})
    pair_params = config.get("paircounts", {})
    paths_params = config.get("paths", {})
    prepare_params = config.get("prepare_profiles", {})

    prepared_value = _first_not_none(
        prepared_dir,
        hmf_config.get("prepared_dir"),
        pair_params.get("prepared_dir"),
        paths_params.get("prepared_dir"),
        prepare_params.get("out_dir"),
        prepare_params.get("output_dir"),
    )
    if prepared_value is None:
        raise KeyError("Set hmf.prepared_dir, paircounts.prepared_dir, or prepare_profiles.out_dir.")
    output_value = _first_not_none(output_dir, hmf_config.get("output_dir"), hmf_config.get("out_dir"))
    path_value = _first_not_none(output_path, hmf_config.get("path"))
    prepared_path = _format_sim_output_dir(prepared_value, config)
    output_path_value = None if path_value is None else Path(str(path_value).format(**_path_format_values(config)))
    output_dir_value = None if output_value is None else _format_sim_output_dir(output_value, config)
    return compute_hmf_from_prepared(
        prepared_dir=prepared_path,
        output_dir=output_dir_value,
        output_path=output_path_value,
        file_tag=_first_not_none(file_tag, hmf_config.get("file_tag"), pair_params.get("file_tag")),
        seed=_first_not_none(seed, hmf_config.get("seed"), pair_params.get("seed"), prepare_params.get("seed")),
        n_bins=int(_first_not_none(n_bins, hmf_config.get("n_bins"), 512)),
        logm_min=_first_not_none(logm_min, hmf_config.get("logm_min")),
        logm_max=_first_not_none(logm_max, hmf_config.get("logm_max")),
        logm_edges=_first_not_none(logm_edges, hmf_config.get("logm_edges")),
        overwrite=bool(_first_not_none(overwrite, hmf_config.get("overwrite"), False)),
        hdf5_compression=_first_not_none(
            hdf5_compression,
            hmf_config.get("hdf5_compression"),
            hmf_config.get("compression"),
        ),
    )


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _path_format_values(config: Mapping[str, Any]) -> dict[str, Any]:
    sim_params = config.get("sim_params", {})
    z_mock = float(sim_params.get("z_mock", 0.0))
    return {
        "sim_name": str(sim_params.get("sim_name", "")),
        "z": _z_directory(z_mock),
        "z_mock": z_mock,
    }


def find_hmf_file(
    output_dir: str | Path,
    *,
    file_tag: str | None = None,
    seed: int | str | None = None,
    n_bins: int = 512,
) -> Path:
    """Find one HMF file in an output directory."""

    output_dir = Path(output_dir)
    tag = _sanitize_filename_piece(file_tag) if file_tag is not None else "*"
    seed_piece = _sanitize_filename_piece(f"seed{seed}") if seed is not None else "seed*"
    pattern = f"hmf_{tag}_{seed_piece}_n{int(n_bins)}.h5"
    matches = sorted(output_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No HMF file matching {pattern} in {output_dir}.")
    raise ValueError(f"Found multiple HMF files matching {pattern} in {output_dir}; set hmf.file_tag and hmf.seed.")


def hmf_density_weights(hmf: HaloMassFunction) -> np.ndarray:
    """Return per-logM-bin number-density weights, dndlog10M * dlog10M."""

    return np.asarray(hmf.density_per_bin, dtype=np.float64)


def hod_occupation_on_hmf(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate central and satellite HOD occupations on HMF bin centers."""

    mass = 10.0 ** np.asarray(hmf.logm_centers, dtype=np.float64)
    cen, sat = evaluate_hod(mass, hod_params, model=hod_model)
    return np.asarray(cen, dtype=np.float64), np.asarray(sat, dtype=np.float64)


def hod_central_number_density(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return int <Ncen> dn/dlog10M dlog10M."""

    cen, _ = hod_occupation_on_hmf(hmf, hod_params, hod_model=hod_model)
    return float(np.sum(cen * hmf_density_weights(hmf)))


def hod_satellite_number_density(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return int <Nsat> dn/dlog10M dlog10M."""

    _, sat = hod_occupation_on_hmf(hmf, hod_params, hod_model=hod_model)
    return float(np.sum(sat * hmf_density_weights(hmf)))


def hod_number_density(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return int <Ncen + Nsat> dn/dlog10M dlog10M."""

    cen, sat = hod_occupation_on_hmf(hmf, hod_params, hod_model=hod_model)
    return float(np.sum((cen + sat) * hmf_density_weights(hmf)))


def hod_satellite_fraction(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return n_sat / (n_cen + n_sat)."""

    n_cen = hod_central_number_density(hmf, hod_params, hod_model=hod_model)
    n_sat = hod_satellite_number_density(hmf, hod_params, hod_model=hod_model)
    total = n_cen + n_sat
    return float(n_sat / total) if total > 0.0 else np.nan


def _weighted_median_log10_host_mass(
    hmf: HaloMassFunction,
    occupation: np.ndarray,
) -> float:
    weights = np.asarray(occupation, dtype=np.float64) * hmf_density_weights(hmf)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    target = 0.5 * total
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(weights) - 1)
    previous = 0.0 if index == 0 else float(cumulative[index - 1])
    current = float(weights[index])
    if current <= 0.0:
        return float(hmf.logm_centers[index])
    fraction = np.clip((target - previous) / current, 0.0, 1.0)
    return float(hmf.logm_edges[index] + fraction * (hmf.logm_edges[index + 1] - hmf.logm_edges[index]))


def hod_central_median_log10_host_mass(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return median log10 host halo mass for centrals."""

    cen, _ = hod_occupation_on_hmf(hmf, hod_params, hod_model=hod_model)
    return _weighted_median_log10_host_mass(hmf, cen)


def hod_satellite_median_log10_host_mass(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return median log10 host halo mass for satellites."""

    _, sat = hod_occupation_on_hmf(hmf, hod_params, hod_model=hod_model)
    return _weighted_median_log10_host_mass(hmf, sat)


def hod_central_median_host_mass(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return median host halo mass for centrals."""

    return float(10.0 ** hod_central_median_log10_host_mass(hmf, hod_params, hod_model=hod_model))


def hod_satellite_median_host_mass(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> float:
    """Return median host halo mass for satellites."""

    return float(10.0 ** hod_satellite_median_log10_host_mass(hmf, hod_params, hod_model=hod_model))


@dataclass(frozen=True)
class HODDerivedQuantities:
    n_cen: float
    n_sat: float
    n_gal: float
    f_sat: float
    log10_mh_cen_med: float
    log10_mh_sat_med: float
    mh_cen_med: float
    mh_sat_med: float


def hod_derived_quantities(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> HODDerivedQuantities:
    """Convenience wrapper returning the common HOD-derived quantities."""

    n_cen = hod_central_number_density(hmf, hod_params, hod_model=hod_model)
    n_sat = hod_satellite_number_density(hmf, hod_params, hod_model=hod_model)
    n_gal = n_cen + n_sat
    log10_cen = hod_central_median_log10_host_mass(hmf, hod_params, hod_model=hod_model)
    log10_sat = hod_satellite_median_log10_host_mass(hmf, hod_params, hod_model=hod_model)
    return HODDerivedQuantities(
        n_cen=n_cen,
        n_sat=n_sat,
        n_gal=n_gal,
        f_sat=float(n_sat / n_gal) if n_gal > 0.0 else np.nan,
        log10_mh_cen_med=log10_cen,
        log10_mh_sat_med=log10_sat,
        mh_cen_med=float(10.0**log10_cen) if np.isfinite(log10_cen) else np.nan,
        mh_sat_med=float(10.0**log10_sat) if np.isfinite(log10_sat) else np.nan,
    )
