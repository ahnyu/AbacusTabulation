"""Prepare halo and NFW particle positions for HOD tabulation."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .nfw import sample_nfw_radii, sample_unit_vectors
from .rsd import Axis, apply_rsd, axis_index, velzspace_to_kms, wrap_positions


DEFAULT_HALO_FIELDS = (
    "N",
    "x_L2com",
    "r25_L2com",
    "r98_L2com",
    "id",
)
HEADER_ATTR_KEYS = (
    "BoxSizeHMpc",
    "ParticleMassHMsun",
    "H0",
    "VelZSpace_to_kms",
    "Redshift",
)


@dataclass(frozen=True)
class PreparedSlabPaths:
    """Output files written for one prepared slab."""

    halo_file: Path
    particle_file: Path
    n_halos: int
    n_particles: int


def z_directory(z_mock: float) -> str:
    """Return the Abacus snapshot directory name, e.g. ``z0.800``."""

    return f"z{float(z_mock):.3f}"


def halo_slab_path(
    sim_dir: str | os.PathLike[str],
    sim_name: str,
    z_mock: float,
    slab_index: int,
) -> Path:
    """Return the path to one periodic-box CompaSO halo_info slab."""

    return (
        Path(sim_dir)
        / sim_name
        / "halos"
        / z_directory(z_mock)
        / "halo_info"
        / f"halo_info_{int(slab_index):03d}.asdf"
    )


def discover_halo_slabs(
    sim_dir: str | os.PathLike[str],
    sim_name: str,
    z_mock: float,
) -> list[int]:
    """Find available periodic-box halo slabs and return their slab indices."""

    halo_info_dir = (
        Path(sim_dir) / sim_name / "halos" / z_directory(z_mock) / "halo_info"
    )
    slabs = []
    for path in sorted(halo_info_dir.glob("halo_info_*.asdf")):
        stem = path.stem
        try:
            slabs.append(int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return slabs


def _load_compaso_halos(
    slab_path: str | os.PathLike[str],
    *,
    cleaned_halos: bool,
    extra_fields: tuple[str, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    """Load the halo fields needed by profile preparation."""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    fields = list(dict.fromkeys((*DEFAULT_HALO_FIELDS, *extra_fields)))
    cat = CompaSOHaloCatalog(str(slab_path), fields=fields, cleaned=cleaned_halos)
    halos = cat.halos
    if cleaned_halos:
        halos = halos[halos["N"] > 0]
    return halos, dict(cat.header)


def _default_nfw_count_factor(sim_name: str) -> float:
    factor = 2.0e-7
    if "hugebase" in sim_name.lower().replace(" ", ""):
        factor *= 27.0 * 27.0
    return factor


def nfw_particle_counts(
    halo_n: np.ndarray,
    *,
    sim_name: str,
    factor: float | None = None,
    min_particles_per_halo: int = 1,
    max_particles_per_halo: int | None = None,
) -> np.ndarray:
    """Choose the number of NFW profile samples per halo.

    The default reproduces the resource example's quadratic-in-``N`` rule.
    """

    if factor is None:
        factor = _default_nfw_count_factor(sim_name)

    counts = (float(factor) * np.asarray(halo_n, dtype=np.float64) ** 2 + 1.5).astype(
        np.int64
    )
    if min_particles_per_halo > 0:
        counts = np.maximum(counts, int(min_particles_per_halo))
    if max_particles_per_halo is not None:
        counts = np.minimum(counts, int(max_particles_per_halo))
    return counts


def particle_starts(counts: np.ndarray) -> np.ndarray:
    """Return start offsets for per-halo particle ranges, using -1 for empty halos."""

    counts = np.asarray(counts, dtype=np.int64)
    starts = np.full(counts.shape, -1, dtype=np.int64)
    positive = counts > 0
    if np.any(positive):
        cumulative = np.cumsum(counts, dtype=np.int64)
        starts[positive] = cumulative[positive] - counts[positive]
    return starts


def _valid_concentration(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    concentration = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, 1.0e-10, dtype=np.float64),
        where=denominator > 0.0,
    )
    return np.maximum(concentration, 1.0e-10)


def _sanitize_filename_piece(value: str) -> str:
    clean = []
    for char in str(value):
        if char.isalnum():
            clean.append(char)
        else:
            clean.append("-")
    return "".join(clean).strip("-") or "none"


def concentration_tag(numerator_key: str, denominator_key: str) -> str:
    """Return a filename-safe tag for the concentration definition."""

    return (
        "conc-"
        + _sanitize_filename_piece(numerator_key)
        + "-over-"
        + _sanitize_filename_piece(denominator_key)
    )


def _sigma_components(sigmav3d: np.ndarray) -> np.ndarray:
    sigmav = np.asarray(sigmav3d, dtype=np.float64)
    if sigmav.ndim == 1:
        return np.repeat((sigmav / np.sqrt(3.0))[:, None], 3, axis=1)
    if sigmav.ndim == 2 and sigmav.shape[1] == 3:
        return sigmav
    raise ValueError("sigmav3d_L2com must be a scalar per halo or shape (n, 3).")


def _safe_attrs(header: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in HEADER_ATTR_KEYS:
        if key not in header:
            continue
        value = header[key]
        arr = np.asarray(value)
        if arr.ndim == 0:
            attrs[key] = arr.item()
        elif arr.size <= 16:
            attrs[key] = arr
    return attrs


def _write_group_file(
    path: Path,
    group_name: str,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any],
    *,
    overwrite: bool,
    compression: str | None,
) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        group = handle.create_group(group_name)
        for name, array in arrays.items():
            kwargs = _dataset_kwargs(compression) if np.asarray(array).size > 0 else {}
            group.create_dataset(name, data=array, **kwargs)


def _normalize_dtype(dtype: str | np.dtype | type) -> np.dtype:
    return np.dtype(dtype)


def _validate_position_space(position_space: str) -> str:
    position_space = str(position_space).lower()
    if position_space not in {"real", "rsd", "both"}:
        raise ValueError("position_space must be 'real', 'rsd', or 'both'.")
    return position_space


def _dataset_kwargs(compression: str | None) -> dict[str, Any]:
    if compression is None:
        return {}
    return {"compression": compression, "chunks": True}


def _write_particle_file_streaming(
    path: Path,
    attrs: dict[str, Any],
    *,
    overwrite: bool,
    compression: str | None,
    n_particles: int,
    halo_counts: np.ndarray,
    halo_pos: np.ndarray,
    halo_rvir: np.ndarray,
    halo_concentration: np.ndarray,
    lbox: float,
    box_origin: str,
    rng: np.random.Generator,
    position_dtype: np.dtype,
    index_dtype: np.dtype,
    particle_chunk_size: int,
    position_space: str,
    nfw_tolerance: float,
    write_particle_radius: bool,
    halo_vel: np.ndarray | None,
    halo_sigma_components: np.ndarray | None,
    satellite_velocity_model: str,
    velz2kms: float | None,
    los_axis: int,
) -> None:
    """Write compact particle positions without holding the full table in RAM."""

    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        group = handle.create_group("particles")
        kwargs = _dataset_kwargs(compression)
        halo_index_ds = group.create_dataset(
            "halo_index", shape=(n_particles,), dtype=index_dtype, **kwargs
        )
        pos_ds = group.create_dataset(
            "pos", shape=(n_particles, 3), dtype=position_dtype, **kwargs
        )
        pos_rsd_ds = None
        if position_space == "both":
            pos_rsd_ds = group.create_dataset(
                "pos_rsd", shape=(n_particles, 3), dtype=position_dtype, **kwargs
            )
        radius_ds = None
        if write_particle_radius:
            radius_ds = group.create_dataset(
                "radius", shape=(n_particles,), dtype=position_dtype, **kwargs
            )

        if n_particles == 0:
            return

        n_halos = len(halo_counts)
        chunk_limit = max(1, int(particle_chunk_size))
        h0 = 0
        p0 = 0
        needs_rsd = position_space in {"rsd", "both"}

        while h0 < n_halos:
            while h0 < n_halos and halo_counts[h0] <= 0:
                h0 += 1
            if h0 >= n_halos:
                break

            h1 = h0
            size = 0
            while h1 < n_halos and (size == 0 or size + int(halo_counts[h1]) <= chunk_limit):
                size += int(halo_counts[h1])
                h1 += 1
                if size >= chunk_limit:
                    break

            local_counts = halo_counts[h0:h1]
            parent = np.repeat(np.arange(h0, h1, dtype=index_dtype), local_counts)
            parent_index = parent.astype(np.int64, copy=False)
            u_radius = rng.random(size)
            directions = sample_unit_vectors(size, rng=rng, dtype=np.float64)
            radii = sample_nfw_radii(
                u_radius,
                halo_rvir[parent_index],
                halo_concentration[parent_index],
                tolerance=nfw_tolerance,
            )
            real_pos = wrap_positions(
                halo_pos[parent_index] + directions * radii[:, None],
                lbox,
                box_origin=box_origin,
            )

            halo_index_ds[p0 : p0 + size] = parent
            if radius_ds is not None:
                radius_ds[p0 : p0 + size] = radii.astype(position_dtype, copy=False)

            if needs_rsd:
                if halo_vel is None or velz2kms is None:
                    raise ValueError("RSD output needs halo velocities and velz2kms.")
                if satellite_velocity_model == "gaussian":
                    if halo_sigma_components is None:
                        raise ValueError("Gaussian satellite RSD needs halo velocity dispersions.")
                    virial_vel = rng.normal(
                        loc=0.0,
                        scale=halo_sigma_components[parent_index],
                        size=(size, 3),
                    )
                    particle_vel = halo_vel[parent_index] + virial_vel
                else:
                    particle_vel = halo_vel[parent_index]
                rsd_pos = apply_rsd(
                    real_pos,
                    particle_vel,
                    lbox,
                    velz2kms=velz2kms,
                    los_axis=los_axis,
                    box_origin=box_origin,
                )
                if position_space == "rsd":
                    pos_ds[p0 : p0 + size] = rsd_pos.astype(position_dtype, copy=False)
                else:
                    pos_ds[p0 : p0 + size] = real_pos.astype(position_dtype, copy=False)
                    assert pos_rsd_ds is not None
                    pos_rsd_ds[p0 : p0 + size] = rsd_pos.astype(position_dtype, copy=False)
            else:
                pos_ds[p0 : p0 + size] = real_pos.astype(position_dtype, copy=False)

            p0 += size
            h0 = h1


def prepare_slab(
    slab_index: int,
    *,
    sim_dir: str | os.PathLike[str],
    sim_name: str,
    z_mock: float,
    output_dir: str | os.PathLike[str],
    seed: int = 600,
    cleaned_halos: bool = True,
    overwrite: bool = False,
    position_space: str = "real",
    los_axis: Axis = 2,
    box_origin: str = "center",
    nfw_count_factor: float | None = None,
    min_particles_per_halo: int = 1,
    max_particles_per_halo: int | None = None,
    satellite_velocity_model: str = "gaussian",
    concentration_numerator_key: str = "r98_L2com",
    concentration_denominator_key: str = "r25_L2com",
    nfw_tolerance: float = 1.0e-5,
    position_dtype: str | np.dtype | type = np.float32,
    index_dtype: str | np.dtype | type = np.int32,
    particle_chunk_size: int = 2_000_000,
    write_particle_radius: bool = False,
    hdf5_compression: str | None = None,
    output_tag: str = "abacustab_profiles",
) -> PreparedSlabPaths:
    """Prepare one halo slab and sampled NFW particle positions.

    Compact output schema:
    halos: id, N, pos, npstart, npout, and pos_rsd only for position_space='both'.
    particles: halo_index, pos, optional pos_rsd, and optional radius.
    """

    if satellite_velocity_model not in {"gaussian", "halo"}:
        raise ValueError("satellite_velocity_model must be 'gaussian' or 'halo'.")

    slab_index = int(slab_index)
    axis = axis_index(los_axis)
    position_space = _validate_position_space(position_space)
    position_dtype = _normalize_dtype(position_dtype)
    index_dtype = _normalize_dtype(index_dtype)
    needs_rsd = position_space in {"rsd", "both"}

    extra_fields = [concentration_numerator_key, concentration_denominator_key]
    if needs_rsd:
        extra_fields.extend(["v_L2com", "sigmav3d_L2com"])
    slab_path = halo_slab_path(sim_dir, sim_name, z_mock, slab_index)
    halos, header = _load_compaso_halos(
        slab_path, cleaned_halos=cleaned_halos, extra_fields=tuple(extra_fields)
    )

    lbox = float(header["BoxSizeHMpc"])
    mpart = float(header["ParticleMassHMsun"])
    h0 = float(header["H0"])
    velz2kms = velzspace_to_kms(header, lbox=lbox) if needs_rsd else None

    halo_id = np.asarray(halos["id"], dtype=np.int64)
    halo_n = np.asarray(halos["N"], dtype=np.int64)
    halo_pos = np.asarray(halos["x_L2com"], dtype=np.float64)
    halo_rvir = np.asarray(halos["r98_L2com"], dtype=np.float64)
    concentration_numerator = np.asarray(
        halos[concentration_numerator_key], dtype=np.float64
    )
    concentration_denominator = np.asarray(
        halos[concentration_denominator_key], dtype=np.float64
    )
    halo_concentration = _valid_concentration(
        concentration_numerator,
        concentration_denominator,
    )

    halo_vel = None
    halo_sigma_components = None
    if needs_rsd:
        halo_vel = np.asarray(halos["v_L2com"], dtype=np.float64)
        halo_sigma_components = _sigma_components(np.asarray(halos["sigmav3d_L2com"]))

    halo_counts = nfw_particle_counts(
        halo_n,
        sim_name=sim_name,
        factor=nfw_count_factor,
        min_particles_per_halo=min_particles_per_halo,
        max_particles_per_halo=max_particles_per_halo,
    )
    halo_starts = particle_starts(halo_counts)
    n_particles = int(np.sum(halo_counts, dtype=np.int64))

    if needs_rsd and halo_vel is not None:
        halo_pos_rsd = apply_rsd(
            halo_pos,
            halo_vel,
            lbox,
            velz2kms=velz2kms,
            los_axis=axis,
            box_origin=box_origin,
        )
    else:
        halo_pos_rsd = None

    if position_space == "rsd":
        assert halo_pos_rsd is not None
        halo_output_pos = halo_pos_rsd
    else:
        halo_output_pos = halo_pos

    halo_arrays: dict[str, np.ndarray] = {
        "id": halo_id,
        "N": halo_n,
        "pos": halo_output_pos.astype(position_dtype, copy=False),
        "npstart": halo_starts,
        "npout": halo_counts,
    }
    if position_space == "both":
        assert halo_pos_rsd is not None
        halo_arrays["pos_rsd"] = halo_pos_rsd.astype(position_dtype, copy=False)

    attrs: dict[str, Any] = {
        **_safe_attrs(header),
        "sim_name": sim_name,
        "z_mock": float(z_mock),
        "slab_index": slab_index,
        "seed": int(seed),
        "h": h0 / 100.0,
        "Lbox": lbox,
        "Mpart": mpart,
        "position_space": position_space,
        "rsd_los_axis": axis,
        "box_origin": box_origin,
        "satellite_velocity_model": satellite_velocity_model,
        "concentration_numerator_key": concentration_numerator_key,
        "concentration_denominator_key": concentration_denominator_key,
        "concentration_tag": concentration_tag(
            concentration_numerator_key,
            concentration_denominator_key,
        ),
        "nfw_count_factor": (
            _default_nfw_count_factor(sim_name)
            if nfw_count_factor is None
            else float(nfw_count_factor)
        ),
        "min_particles_per_halo": int(min_particles_per_halo),
        "nfw_tolerance": float(nfw_tolerance),
        "schema_version": "compact_v1",
    }
    if velz2kms is not None:
        attrs["velz2kms"] = velz2kms
    if max_particles_per_halo is not None:
        attrs["max_particles_per_halo"] = int(max_particles_per_halo)

    output_dir = Path(output_dir)
    conc_tag = concentration_tag(
        concentration_numerator_key,
        concentration_denominator_key,
    )
    halo_file = (
        output_dir / f"halos_xcom_{slab_index}_seed{int(seed)}_{output_tag}_{conc_tag}.h5"
    )
    particle_file = (
        output_dir
        / f"particles_xcom_{slab_index}_seed{int(seed)}_{output_tag}_{conc_tag}.h5"
    )

    _write_group_file(
        halo_file,
        "halos",
        halo_arrays,
        attrs,
        overwrite=overwrite,
        compression=hdf5_compression,
    )
    rng = np.random.default_rng(int(seed) + slab_index)
    _write_particle_file_streaming(
        particle_file,
        attrs,
        overwrite=overwrite,
        compression=hdf5_compression,
        n_particles=n_particles,
        halo_counts=halo_counts,
        halo_pos=halo_pos,
        halo_rvir=halo_rvir,
        halo_concentration=halo_concentration,
        lbox=lbox,
        box_origin=box_origin,
        rng=rng,
        position_dtype=position_dtype,
        index_dtype=index_dtype,
        particle_chunk_size=particle_chunk_size,
        position_space=position_space,
        nfw_tolerance=nfw_tolerance,
        write_particle_radius=write_particle_radius,
        halo_vel=halo_vel,
        halo_sigma_components=halo_sigma_components,
        satellite_velocity_model=satellite_velocity_model,
        velz2kms=velz2kms,
        los_axis=axis,
    )
    return PreparedSlabPaths(
        halo_file=halo_file,
        particle_file=particle_file,
        n_halos=len(halo_id),
        n_particles=n_particles,
    )


def prepare_all_slabs(
    *,
    sim_dir: str | os.PathLike[str],
    sim_name: str,
    z_mock: float,
    output_dir: str | os.PathLike[str],
    slab_indices: list[int] | tuple[int, ...] | None = None,
    n_parallel: int = 1,
    **kwargs: Any,
) -> list[PreparedSlabPaths]:
    """Prepare all requested slabs for one snapshot."""

    if slab_indices is None:
        slab_indices = discover_halo_slabs(sim_dir, sim_name, z_mock)
    slab_indices = list(slab_indices)
    if len(slab_indices) == 0:
        raise ValueError("No halo_info slabs were found.")

    if int(n_parallel) <= 1:
        return [
            prepare_slab(
                slab_index,
                sim_dir=sim_dir,
                sim_name=sim_name,
                z_mock=z_mock,
                output_dir=output_dir,
                **kwargs,
            )
            for slab_index in slab_indices
        ]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(n_parallel),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(
                prepare_slab,
                slab_index,
                sim_dir=sim_dir,
                sim_name=sim_name,
                z_mock=z_mock,
                output_dir=output_dir,
                **kwargs,
            )
            for slab_index in slab_indices
        ]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _format_output_dir(template: str, *, sim_name: str, z_mock: float) -> Path:
    return Path(template.format(sim_name=sim_name, z=z_directory(z_mock), z_mock=z_mock))


def prepare_from_config(
    path2config: str | os.PathLike[str],
    *,
    alt_simname: str | None = None,
    alt_z: float | None = None,
    seed: int | None = None,
    overwrite: bool | None = None,
    slab_indices: list[int] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    n_parallel: int | None = None,
    position_space: str | None = None,
    los_axis: Axis | None = None,
    box_origin: str | None = None,
    nfw_count_factor: float | None = None,
    min_particles_per_halo: int | None = None,
    max_particles_per_halo: int | None = None,
    satellite_velocity_model: str | None = None,
    concentration_numerator_key: str | None = None,
    concentration_denominator_key: str | None = None,
    position_dtype: str | np.dtype | type | None = None,
    index_dtype: str | np.dtype | type | None = None,
    particle_chunk_size: int | None = None,
    write_particle_radius: bool | None = None,
    hdf5_compression: str | None = None,
) -> list[PreparedSlabPaths]:
    """Prepare profile catalogs from a YAML config.

    The function understands the resource-style ``sim_params`` block and an
    optional ``prepare_profiles`` block for tabulation-specific options.
    """

    import yaml

    with open(path2config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    sim_params = config.get("sim_params", {})
    paths_params = config.get("paths", {})
    prepare_params = config.get("prepare_profiles", {})

    sim_name = alt_simname or sim_params["sim_name"]
    z_mock = float(alt_z if alt_z is not None else sim_params["z_mock"])
    sim_dir = sim_params["sim_dir"]

    output_value = _first_not_none(
        output_dir,
        prepare_params.get("output_dir"),
        paths_params.get("prepared_dir"),
    )
    if output_value is not None:
        output_path = _format_output_dir(str(output_value), sim_name=sim_name, z_mock=z_mock)
    else:
        output_root = sim_params.get("subsample_dir", prepare_params.get("output_root"))
        if output_root is None:
            raise KeyError(
                "Set prepare_profiles.output_dir, paths.prepared_dir, "
                "or sim_params.subsample_dir."
            )
        output_path = Path(output_root) / sim_name / z_directory(z_mock)

    requested_slabs = slab_indices
    if requested_slabs is None and "slab_indices" in prepare_params:
        requested_slabs = [int(index) for index in prepare_params["slab_indices"]]

    position_space_config = prepare_params.get("position_space")
    if position_space_config is None:
        position_space_config = "both" if prepare_params.get("write_rsd", False) else "real"

    return prepare_all_slabs(
        sim_dir=sim_dir,
        sim_name=sim_name,
        z_mock=z_mock,
        output_dir=output_path,
        slab_indices=requested_slabs,
        n_parallel=int(
            _first_not_none(
                n_parallel,
                prepare_params.get("Nparallel_load"),
                prepare_params.get("n_parallel"),
                1,
            )
        ),
        seed=int(_first_not_none(seed, prepare_params.get("seed"), 600)),
        cleaned_halos=bool(sim_params.get("cleaned_halos", True)),
        overwrite=bool(_first_not_none(overwrite, prepare_params.get("overwrite"), False)),
        position_space=str(_first_not_none(position_space, position_space_config)),
        los_axis=_first_not_none(los_axis, prepare_params.get("los_axis"), 2),
        box_origin=str(_first_not_none(box_origin, prepare_params.get("box_origin"), "center")),
        nfw_count_factor=_first_not_none(nfw_count_factor, prepare_params.get("nfw_count_factor")),
        min_particles_per_halo=int(
            _first_not_none(
                min_particles_per_halo,
                prepare_params.get("min_particles_per_halo"),
                1,
            )
        ),
        max_particles_per_halo=_first_not_none(
            max_particles_per_halo,
            prepare_params.get("max_particles_per_halo"),
        ),
        satellite_velocity_model=str(
            _first_not_none(
                satellite_velocity_model,
                prepare_params.get("satellite_velocity_model"),
                "gaussian",
            )
        ),
        concentration_numerator_key=str(
            _first_not_none(
                concentration_numerator_key,
                prepare_params.get("concentration_numerator_key"),
                "r98_L2com",
            )
        ),
        concentration_denominator_key=str(
            _first_not_none(
                concentration_denominator_key,
                prepare_params.get("concentration_denominator_key"),
                "r25_L2com",
            )
        ),
        nfw_tolerance=float(prepare_params.get("nfw_tolerance", 1.0e-5)),
        position_dtype=_first_not_none(position_dtype, prepare_params.get("position_dtype"), "f4"),
        index_dtype=_first_not_none(index_dtype, prepare_params.get("index_dtype"), "i4"),
        particle_chunk_size=int(
            _first_not_none(
                particle_chunk_size,
                prepare_params.get("particle_chunk_size"),
                2_000_000,
            )
        ),
        write_particle_radius=bool(
            _first_not_none(
                write_particle_radius,
                prepare_params.get("write_particle_radius"),
                False,
            )
        ),
        hdf5_compression=_first_not_none(
            hdf5_compression,
            prepare_params.get("hdf5_compression"),
            prepare_params.get("compression"),
        ),
        output_tag=str(prepare_params.get("output_tag", "abacustab_profiles")),
    )
