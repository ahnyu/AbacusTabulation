"""High-resolution halo mass functions and HOD-derived summary quantities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Any, Mapping

import h5py
import numpy as np

from .hod import evaluate_hod, evaluate_hod_splits
from .paircounts import find_prepared_file_pairs, parse_logm_edges


_HMF_EDGE_FRACTION_WARN = 1.0e-3


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
            "logm_min": float(edges[0]),
            "logm_max": float(edges[-1]),
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


def _assert_close_array(path: Path, name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError(
            f"HMF file {path} has {name} shape/value inconsistent with the current config: "
            f"actual shape {actual.shape}, expected shape {expected.shape}."
        )


def _expected_hmf_edges(
    *,
    n_bins: int,
    logm_edges: object | None = None,
    logm_min: float | None = None,
    logm_max: float | None = None,
) -> np.ndarray | None:
    if logm_edges is not None:
        return parse_logm_edges(logm_edges)
    if logm_min is None or logm_max is None:
        return None
    lower = float(logm_min)
    upper = float(logm_max)
    if upper <= lower:
        raise ValueError("logm_max must be greater than logm_min.")
    return np.linspace(lower, np.nextafter(upper, np.inf), int(n_bins) + 1)


def validate_hmf_file(
    path: str | Path,
    *,
    expected_n_bins: int | None = None,
    expected_logm_edges: np.ndarray | None = None,
    expected_logm_min: float | None = None,
    expected_logm_max: float | None = None,
) -> Path:
    """Validate that an HMF HDF5 file matches the requested HMF tabulation."""

    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "hmf" not in handle:
            raise ValueError(f"HMF file {path} is missing the hmf group.")
        group = handle["hmf"]
        if "logm_edges" not in group or "dndlog10m" not in group:
            raise ValueError(f"HMF file {path} is missing hmf/logm_edges or hmf/dndlog10m.")
        edges = group["logm_edges"][...]
        n_bins = len(edges) - 1
        if expected_n_bins is not None and n_bins != int(expected_n_bins):
            raise ValueError(f"HMF file {path} has {n_bins} bins, expected {int(expected_n_bins)}.")
        if expected_logm_edges is not None:
            _assert_close_array(path, "hmf/logm_edges", edges, expected_logm_edges)
        if expected_logm_min is not None and not np.isclose(edges[0], float(expected_logm_min), rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(f"HMF file {path} has lower mass edge {edges[0]}, expected {float(expected_logm_min)}.")
        if expected_logm_max is not None and not np.isclose(edges[-1], float(expected_logm_max), rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(f"HMF file {path} has upper mass edge {edges[-1]}, expected {float(expected_logm_max)}.")
    return path


def _expected_hmf_edge_bounds(
    *,
    logm_edges: object | None = None,
    logm_min: float | None = None,
    logm_max: float | None = None,
) -> tuple[float | None, float | None]:
    if logm_edges is not None:
        edges = parse_logm_edges(logm_edges)
        return float(edges[0]), float(edges[-1])
    if logm_min is not None and logm_max is not None and float(logm_max) <= float(logm_min):
        raise ValueError("logm_max must be greater than logm_min.")
    return (
        None if logm_min is None else float(logm_min),
        None if logm_max is None else float(logm_max),
    )


def validate_hmf_file_for_config(
    path: str | Path,
    *,
    n_bins: int = 512,
    logm_edges: object | None = None,
    logm_min: float | None = None,
    logm_max: float | None = None,
) -> Path:
    """Validate an explicit HMF path against the configured HMF tabulation."""

    expected_edges = _expected_hmf_edges(
        n_bins=int(n_bins),
        logm_edges=logm_edges,
        logm_min=logm_min,
        logm_max=logm_max,
    )
    expected_min, expected_max = _expected_hmf_edge_bounds(
        logm_edges=logm_edges,
        logm_min=logm_min,
        logm_max=logm_max,
    )
    expected_n_bins = len(expected_edges) - 1 if expected_edges is not None else int(n_bins)
    return validate_hmf_file(
        path,
        expected_n_bins=expected_n_bins,
        expected_logm_edges=expected_edges,
        expected_logm_min=expected_min,
        expected_logm_max=expected_max,
    )


def find_hmf_file(
    output_dir: str | Path,
    *,
    file_tag: str | None = None,
    seed: int | str | None = None,
    n_bins: int = 512,
    logm_edges: object | None = None,
    logm_min: float | None = None,
    logm_max: float | None = None,
) -> Path:
    """Find one HMF file in an output directory and validate known config fields."""

    output_dir = Path(output_dir)
    tag = _sanitize_filename_piece(file_tag) if file_tag is not None else "*"
    seed_piece = _sanitize_filename_piece(f"seed{seed}") if seed is not None else "seed*"
    expected_edges = _expected_hmf_edges(
        n_bins=int(n_bins),
        logm_edges=logm_edges,
        logm_min=logm_min,
        logm_max=logm_max,
    )
    effective_n_bins = len(expected_edges) - 1 if expected_edges is not None else int(n_bins)
    pattern = f"hmf_{tag}_{seed_piece}_n{effective_n_bins}.h5"
    matches = sorted(output_dir.glob(pattern))
    if len(matches) == 1:
        return validate_hmf_file_for_config(
            matches[0],
            n_bins=effective_n_bins,
            logm_edges=logm_edges,
            logm_min=logm_min,
            logm_max=logm_max,
        )
    if not matches:
        raise FileNotFoundError(f"No HMF file matching {pattern} in {output_dir}.")
    raise ValueError(f"Found multiple HMF files matching {pattern} in {output_dir}; set hmf.file_tag and hmf.seed.")


def hmf_density_weights(hmf: HaloMassFunction) -> np.ndarray:
    """Return per-logM-bin number-density weights, dndlog10M * dlog10M."""

    return np.asarray(hmf.density_per_bin, dtype=np.float64)


def _warn_if_hmf_edges_contribute(hmf: HaloMassFunction, occupation: np.ndarray) -> None:
    weights = np.asarray(occupation, dtype=np.float64) * hmf_density_weights(hmf)
    if weights.size == 0 or weights.shape[-1] == 0:
        return
    rows = weights.reshape(-1, weights.shape[-1])
    finite_weights = np.where(np.isfinite(rows), rows, 0.0)
    totals = np.sum(finite_weights, axis=1)
    valid = totals > 0.0
    if not np.any(valid):
        return
    edge_fraction = np.maximum(finite_weights[:, 0], finite_weights[:, -1])
    edge_fraction[valid] /= totals[valid]
    if np.any(edge_fraction[valid] > _HMF_EDGE_FRACTION_WARN):
        warnings.warn(
            "The first or last HMF bin contributes more than 0.1% of the HOD number density. "
            "Consider widening hmf.logm_min/hmf.logm_max for derived quantities.",
            RuntimeWarning,
            stacklevel=3,
        )


def hod_occupation_on_hmf(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate central and satellite HOD occupations on HMF bin centers."""

    mass = 10.0 ** np.asarray(hmf.logm_centers, dtype=np.float64)
    cen, sat = evaluate_hod(mass, hod_params, model=hod_model)
    cen = np.asarray(cen, dtype=np.float64)
    sat = np.asarray(sat, dtype=np.float64)
    _warn_if_hmf_edges_contribute(hmf, cen + sat)
    return cen, sat


def stellar_mass_hod_occupations_on_hmf(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    split_method: str,
    logmstar_edges: Any,
    redshift_weights: Any = None,
    hod_model: str = "lrg_stellar_mass",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate every stellar-mass split on the HMF bin centers."""

    mass = 10.0 ** np.asarray(hmf.logm_centers, dtype=np.float64)
    central, satellite = evaluate_hod_splits(
        mass,
        hod_params,
        model=hod_model,
        split_method=split_method,
        logmstar_edges=logmstar_edges,
        redshift_weights=redshift_weights,
    )
    central = np.asarray(central, dtype=np.float64)
    satellite = np.asarray(satellite, dtype=np.float64)
    expected_tail = (hmf.logm_centers.size,)
    if (
        central.ndim != 2
        or central.shape[1:] != expected_tail
        or satellite.shape != central.shape
    ):
        raise ValueError(
            "Stellar-mass HOD occupations must have shape "
            f"(n_splits, {expected_tail[0]}), got central {central.shape} "
            f"and satellite {satellite.shape}."
        )
    if (
        not np.all(np.isfinite(central))
        or not np.all(np.isfinite(satellite))
        or np.any(central < 0.0)
        or np.any(satellite < 0.0)
    ):
        raise ValueError("Stellar-mass HOD occupations must be finite and non-negative.")
    _warn_if_hmf_edges_contribute(hmf, central + satellite)
    return central, satellite


def _number_densities_from_occupations(
    hmf: HaloMassFunction,
    occupation: np.ndarray,
) -> np.ndarray:
    return np.sum(
        np.asarray(occupation, dtype=np.float64) * hmf_density_weights(hmf),
        axis=-1,
    )


def stellar_mass_hod_central_number_densities(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return central number density for every stellar-mass split."""

    central, _ = stellar_mass_hod_occupations_on_hmf(hmf, hod_params, **split_options)
    return _number_densities_from_occupations(hmf, central)


def stellar_mass_hod_satellite_number_densities(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return satellite number density for every stellar-mass split."""

    _, satellite = stellar_mass_hod_occupations_on_hmf(hmf, hod_params, **split_options)
    return _number_densities_from_occupations(hmf, satellite)


def stellar_mass_hod_number_densities(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return total number density for every stellar-mass split."""

    central, satellite = stellar_mass_hod_occupations_on_hmf(
        hmf, hod_params, **split_options
    )
    return _number_densities_from_occupations(hmf, central + satellite)


def stellar_mass_hod_satellite_fractions(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return satellite fraction for every stellar-mass split."""

    central, satellite = stellar_mass_hod_occupations_on_hmf(
        hmf, hod_params, **split_options
    )
    n_cen = _number_densities_from_occupations(hmf, central)
    n_sat = _number_densities_from_occupations(hmf, satellite)
    total = n_cen + n_sat
    return np.divide(
        n_sat,
        total,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=total > 0.0,
    )


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

    cen, sat = hod_occupation_on_hmf(hmf, hod_params, hod_model=hod_model)
    weights = hmf_density_weights(hmf)
    n_cen = float(np.sum(cen * weights))
    n_sat = float(np.sum(sat * weights))
    total = n_cen + n_sat
    return float(n_sat / total) if total > 0.0 else np.nan


def _weighted_median_log10_host_masses(
    hmf: HaloMassFunction,
    occupations: np.ndarray,
) -> np.ndarray:
    occupations = np.asarray(occupations, dtype=np.float64)
    if occupations.ndim != 2 or occupations.shape[1] != hmf.logm_centers.size:
        raise ValueError(
            "occupations must have shape "
            f"(n_samples, {hmf.logm_centers.size}), got {occupations.shape}."
        )
    weights = occupations * hmf_density_weights(hmf)[None, :]
    totals = np.sum(weights, axis=1)
    result = np.full(occupations.shape[0], np.nan, dtype=np.float64)
    valid = totals > 0.0
    if not np.any(valid):
        return result

    cumulative = np.cumsum(weights[valid], axis=1)
    targets = 0.5 * totals[valid]
    indices = np.sum(cumulative < targets[:, None], axis=1)
    indices = np.minimum(indices, weights.shape[1] - 1)
    rows = np.arange(indices.size)
    previous = np.where(indices == 0, 0.0, cumulative[rows, np.maximum(indices - 1, 0)])
    current = weights[valid][rows, indices]
    fraction = np.divide(
        targets - previous,
        current,
        out=np.zeros_like(targets),
        where=current > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    interpolated = hmf.logm_edges[indices] + fraction * (
        hmf.logm_edges[indices + 1] - hmf.logm_edges[indices]
    )
    interpolated = np.where(
        current > 0.0,
        interpolated,
        hmf.logm_centers[indices],
    )
    result[valid] = interpolated
    return result


def _weighted_median_log10_host_mass(
    hmf: HaloMassFunction,
    occupation: np.ndarray,
) -> float:
    occupation = np.asarray(occupation, dtype=np.float64).reshape(1, -1)
    return float(_weighted_median_log10_host_masses(hmf, occupation)[0])


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


def stellar_mass_hod_central_median_log10_host_masses(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return median log10 central host mass for every stellar-mass split."""

    central, _ = stellar_mass_hod_occupations_on_hmf(hmf, hod_params, **split_options)
    return _weighted_median_log10_host_masses(hmf, central)


def stellar_mass_hod_satellite_median_log10_host_masses(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return median log10 satellite host mass for every stellar-mass split."""

    _, satellite = stellar_mass_hod_occupations_on_hmf(hmf, hod_params, **split_options)
    return _weighted_median_log10_host_masses(hmf, satellite)


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


def stellar_mass_hod_central_median_host_masses(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return median central host mass for every stellar-mass split."""

    log10_mass = stellar_mass_hod_central_median_log10_host_masses(
        hmf, hod_params, **split_options
    )
    return np.where(np.isfinite(log10_mass), 10.0**log10_mass, np.nan)


def stellar_mass_hod_satellite_median_host_masses(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    **split_options: Any,
) -> np.ndarray:
    """Return median satellite host mass for every stellar-mass split."""

    log10_mass = stellar_mass_hod_satellite_median_log10_host_masses(
        hmf, hod_params, **split_options
    )
    return np.where(np.isfinite(log10_mass), 10.0**log10_mass, np.nan)


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


@dataclass(frozen=True)
class StellarMassHODDerivedQuantities:
    n_cen: np.ndarray
    n_sat: np.ndarray
    n_gal: np.ndarray
    f_sat: np.ndarray
    log10_mh_cen_med: np.ndarray
    log10_mh_sat_med: np.ndarray
    mh_cen_med: np.ndarray
    mh_sat_med: np.ndarray


def hod_derived_quantities(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    hod_model: str = "lrg",
) -> HODDerivedQuantities:
    """Convenience wrapper returning the common HOD-derived quantities."""

    central, satellite = hod_occupation_on_hmf(
        hmf,
        hod_params,
        hod_model=hod_model,
    )
    density_weights = hmf_density_weights(hmf)
    n_cen = float(np.sum(central * density_weights))
    n_sat = float(np.sum(satellite * density_weights))
    n_gal = n_cen + n_sat
    log10_cen = _weighted_median_log10_host_mass(hmf, central)
    log10_sat = _weighted_median_log10_host_mass(hmf, satellite)
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


def stellar_mass_hod_derived_quantities(
    hmf: HaloMassFunction,
    hod_params: Mapping[str, Any],
    *,
    split_method: str,
    logmstar_edges: Any,
    redshift_weights: Any = None,
    hod_model: str = "lrg_stellar_mass",
) -> StellarMassHODDerivedQuantities:
    """Return common HMF-derived quantities for every stellar-mass split."""

    central, satellite = stellar_mass_hod_occupations_on_hmf(
        hmf,
        hod_params,
        split_method=split_method,
        logmstar_edges=logmstar_edges,
        redshift_weights=redshift_weights,
        hod_model=hod_model,
    )
    n_cen = _number_densities_from_occupations(hmf, central)
    n_sat = _number_densities_from_occupations(hmf, satellite)
    n_gal = n_cen + n_sat
    f_sat = np.divide(
        n_sat,
        n_gal,
        out=np.full_like(n_gal, np.nan, dtype=np.float64),
        where=n_gal > 0.0,
    )
    log10_cen = _weighted_median_log10_host_masses(hmf, central)
    log10_sat = _weighted_median_log10_host_masses(hmf, satellite)
    return StellarMassHODDerivedQuantities(
        n_cen=n_cen,
        n_sat=n_sat,
        n_gal=n_gal,
        f_sat=f_sat,
        log10_mh_cen_med=log10_cen,
        log10_mh_sat_med=log10_sat,
        mh_cen_med=np.where(np.isfinite(log10_cen), 10.0**log10_cen, np.nan),
        mh_sat_med=np.where(np.isfinite(log10_sat), 10.0**log10_sat, np.nan),
    )
