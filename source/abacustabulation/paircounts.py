"""Mass-bin tabulated pair counts for prepared AbacusTabulation catalogs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PREPARED_HALO_RE = re.compile(
    r"halos_xcom_(?P<slab>\d+)_seed(?P<seed>\d+)_(?P<tag>.+)\.h5$"
)


@dataclass(frozen=True)
class PreparedFilePair:
    slab: int
    seed: int
    tag: str
    halo_file: Path
    particle_file: Path


@dataclass
class PreparedCatalog:
    halo_pos: np.ndarray
    halo_mass: np.ndarray
    particle_pos: np.ndarray
    particle_halo_index: np.ndarray
    lbox: float
    mpart: float
    files: list[PreparedFilePair]
    attrs: dict[str, Any]


@dataclass
class BinnedPreparedCatalog:
    halo_by_bin: list[np.ndarray]
    particle_by_bin: list[np.ndarray]
    lbox: float
    mpart: float
    files: list[PreparedFilePair]
    attrs: dict[str, Any]
    n_halos: int
    n_particles: int


@dataclass
class MassTabulation:
    edges_log10: np.ndarray
    mean_log10: np.ndarray
    halo_bin: np.ndarray
    particle_bin: np.ndarray
    num_halo: np.ndarray
    num_particle: np.ndarray


def _sanitize_filename_piece(value: object) -> str:
    clean = []
    for char in str(value):
        if char.isalnum():
            clean.append(char)
        else:
            clean.append("-")
    return "".join(clean).strip("-") or "none"


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepared_output_attrs(
    first_attrs: dict[str, Any],
    file_pairs: list[PreparedFilePair],
    position_dataset: str,
) -> dict[str, Any]:
    tags = sorted({pair.tag for pair in file_pairs})
    seeds = sorted({pair.seed for pair in file_pairs})
    attrs = dict(first_attrs)
    attrs["position_dataset"] = position_dataset
    attrs["corrfunc_position_wrap"] = "zero"
    attrs["prepared_tags"] = ",".join(tags)
    attrs["prepared_seeds"] = ",".join(str(item) for item in seeds)
    attrs["prepared_slabs"] = ",".join(str(pair.slab) for pair in file_pairs)
    return attrs


def find_prepared_file_pairs(
    prepared_dir: str | Path,
    *,
    file_tag: str | None = None,
    seed: int | None = None,
) -> list[PreparedFilePair]:
    """Find matching halo/particle prepared files for all slabs."""

    prepared_dir = Path(prepared_dir)
    pairs: list[PreparedFilePair] = []
    seen_slabs: dict[int, Path] = {}

    for halo_file in sorted(prepared_dir.glob("halos_xcom_*_seed*_*.h5")):
        match = PREPARED_HALO_RE.match(halo_file.name)
        if match is None:
            continue
        slab = int(match.group("slab"))
        file_seed = int(match.group("seed"))
        tag = match.group("tag")
        if seed is not None and file_seed != int(seed):
            continue
        if file_tag is not None and tag != file_tag:
            continue
        if slab in seen_slabs:
            raise ValueError(
                "Found more than one prepared halo file for slab "
                f"{slab}: {seen_slabs[slab]} and {halo_file}. "
                "Pass --file-tag and/or --seed to disambiguate."
            )
        particle_file = prepared_dir / f"particles_xcom_{slab}_seed{file_seed}_{tag}.h5"
        if not particle_file.exists():
            raise FileNotFoundError(f"Missing particle file for {halo_file}: {particle_file}")
        seen_slabs[slab] = halo_file
        pairs.append(
            PreparedFilePair(
                slab=slab,
                seed=file_seed,
                tag=tag,
                halo_file=halo_file,
                particle_file=particle_file,
            )
        )

    if not pairs:
        suffix = f" with tag {file_tag!r}" if file_tag is not None else ""
        raise FileNotFoundError(f"No prepared files found in {prepared_dir}{suffix}.")
    return sorted(pairs, key=lambda item: item.slab)


def _read_required_dataset(group: h5py.Group, name: str, file_path: Path) -> np.ndarray:
    if name not in group:
        raise KeyError(f"Dataset {name!r} not found in {file_path}.")
    return group[name][...]


def load_prepared_catalog(
    prepared_dir: str | Path,
    *,
    file_tag: str | None = None,
    seed: int | None = None,
    position_dataset: str = "pos",
    dtype: str | np.dtype | None = None,
) -> PreparedCatalog:
    """Load and concatenate prepared halo and NFW particle files."""

    file_pairs = find_prepared_file_pairs(prepared_dir, file_tag=file_tag, seed=seed)
    halo_pos_all: list[np.ndarray] = []
    halo_mass_all: list[np.ndarray] = []
    particle_pos_all: list[np.ndarray] = []
    particle_halo_index_all: list[np.ndarray] = []
    halo_offset = 0
    lbox: float | None = None
    mpart: float | None = None
    first_attrs: dict[str, Any] = {}

    for index, pair in enumerate(file_pairs):
        with h5py.File(pair.halo_file, "r") as handle:
            attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
            group = handle["halos"]
            halo_pos = _read_required_dataset(group, position_dataset, pair.halo_file)
            halo_n = _read_required_dataset(group, "N", pair.halo_file).astype(np.float64)
            slab_mpart = float(attrs.get("Mpart", attrs.get("ParticleMassHMsun")))
            slab_lbox = float(attrs.get("Lbox", attrs.get("BoxSizeHMpc")))
            if index == 0:
                lbox = slab_lbox
                mpart = slab_mpart
                first_attrs = attrs
            else:
                if not np.isclose(slab_lbox, lbox):
                    raise ValueError(f"Inconsistent Lbox in {pair.halo_file}.")
                if not np.isclose(slab_mpart, mpart):
                    raise ValueError(f"Inconsistent Mpart in {pair.halo_file}.")
            halo_pos_all.append(halo_pos)
            halo_mass_all.append(halo_n * slab_mpart)
            n_halo_slab = len(halo_n)

        with h5py.File(pair.particle_file, "r") as handle:
            group = handle["particles"]
            particle_pos = _read_required_dataset(group, position_dataset, pair.particle_file)
            particle_halo_index = _read_required_dataset(
                group, "halo_index", pair.particle_file
            ).astype(np.int64, copy=False)
            particle_pos_all.append(particle_pos)
            particle_halo_index_all.append(particle_halo_index + halo_offset)

        halo_offset += n_halo_slab

    if dtype is not None:
        out_dtype = np.dtype(dtype)
        halo_pos_concat = np.ascontiguousarray(np.concatenate(halo_pos_all), dtype=out_dtype)
        particle_pos_concat = np.ascontiguousarray(
            np.concatenate(particle_pos_all), dtype=out_dtype
        )
    else:
        halo_pos_concat = np.ascontiguousarray(np.concatenate(halo_pos_all))
        particle_pos_concat = np.ascontiguousarray(np.concatenate(particle_pos_all))

    halo_pos_concat = np.mod(halo_pos_concat, float(lbox))
    particle_pos_concat = np.mod(particle_pos_concat, float(lbox))

    first_attrs = _prepared_output_attrs(first_attrs, file_pairs, position_dataset)

    return PreparedCatalog(
        halo_pos=halo_pos_concat,
        halo_mass=np.concatenate(halo_mass_all),
        particle_pos=particle_pos_concat,
        particle_halo_index=np.concatenate(particle_halo_index_all),
        lbox=float(lbox),
        mpart=float(mpart),
        files=file_pairs,
        attrs=first_attrs,
    )


def parse_logm_edges(value: object | None) -> np.ndarray | None:
    """Parse log10 mass edges from a list, comma-separated string, or text file."""

    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        edges = np.asarray(value, dtype=np.float64)
    else:
        value_str = str(value)
        path = Path(value_str)
        if path.exists():
            edges = np.loadtxt(path, dtype=np.float64)
        else:
            edges = np.array([float(item) for item in value_str.split(",") if item.strip()])
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("Mass-bin edges must be a one-dimensional array with >=2 edges.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("Mass-bin edges must be strictly increasing.")
    return edges


def _assign_log_bins(log_mass: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bin_index = np.searchsorted(edges, log_mass, side="right") - 1
    bin_index[(log_mass < edges[0]) | (log_mass > edges[-1])] = -1
    bin_index[log_mass == edges[-1]] = len(edges) - 2
    return bin_index.astype(np.int32, copy=False)


def build_mass_tabulation(
    halo_mass: np.ndarray,
    particle_halo_index: np.ndarray,
    *,
    nmass_bins: int = 20,
    logm_min: float | None = None,
    logm_max: float | None = None,
    logm_edges: np.ndarray | None = None,
) -> MassTabulation:
    """Assign halos and NFW particles to host-halo mass bins."""

    halo_mass = np.asarray(halo_mass, dtype=np.float64)
    if np.any(halo_mass <= 0.0):
        raise ValueError("All halo masses must be positive.")
    log_mass = np.log10(halo_mass)

    if logm_edges is None:
        lower = float(np.min(log_mass) if logm_min is None else logm_min)
        upper = float(np.max(log_mass) if logm_max is None else logm_max)
        if upper <= lower:
            raise ValueError("logm_max must be greater than logm_min.")
        edges = np.linspace(lower, np.nextafter(upper, np.inf), int(nmass_bins) + 1)
    else:
        edges = np.asarray(logm_edges, dtype=np.float64)
        nmass_bins = len(edges) - 1

    halo_bin = _assign_log_bins(log_mass, edges)
    if np.any((particle_halo_index < 0) | (particle_halo_index >= len(halo_bin))):
        raise ValueError("particle_halo_index contains values outside the halo table.")
    particle_bin = halo_bin[particle_halo_index]

    valid_halo = halo_bin >= 0
    valid_particle = particle_bin >= 0
    num_halo = np.bincount(halo_bin[valid_halo], minlength=nmass_bins).astype(np.int64)
    num_particle = np.bincount(
        particle_bin[valid_particle], minlength=nmass_bins
    ).astype(np.int64)

    mean_log10 = np.empty(nmass_bins, dtype=np.float64)
    for i in range(nmass_bins):
        mask = halo_bin == i
        if np.any(mask):
            mean_log10[i] = np.log10(np.mean(halo_mass[mask]))
        else:
            mean_log10[i] = 0.5 * (edges[i] + edges[i + 1])

    return MassTabulation(
        edges_log10=edges,
        mean_log10=mean_log10,
        halo_bin=halo_bin,
        particle_bin=particle_bin.astype(np.int32, copy=False),
        num_halo=num_halo,
        num_particle=num_particle,
    )


def _append_positions_by_bin(
    chunks: list[list[np.ndarray]],
    positions: np.ndarray,
    bin_index: np.ndarray,
    nmass_bins: int,
) -> None:
    valid = bin_index >= 0
    if not np.any(valid):
        return
    valid_bins = bin_index[valid]
    order = np.argsort(valid_bins, kind="stable")
    sorted_bins = valid_bins[order]
    sorted_positions = np.ascontiguousarray(positions[valid][order])
    for i in range(nmass_bins):
        start = int(np.searchsorted(sorted_bins, i, side="left"))
        stop = int(np.searchsorted(sorted_bins, i, side="right"))
        if stop > start:
            chunks[i].append(sorted_positions[start:stop])


def _concat_position_chunks(
    chunks: list[list[np.ndarray]],
    dtype: np.dtype,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for items in chunks:
        if items:
            out.append(np.ascontiguousarray(np.concatenate(items, axis=0), dtype=dtype))
        else:
            out.append(np.empty((0, 3), dtype=dtype))
        items.clear()
    return out


def load_prepared_binned_catalog(
    prepared_dir: str | Path,
    *,
    file_tag: str | None = None,
    seed: int | None = None,
    position_dataset: str = "pos",
    dtype: str | np.dtype | None = None,
    nmass_bins: int = 20,
    logm_min: float | None = None,
    logm_max: float | None = None,
    logm_edges: object | None = None,
) -> tuple[BinnedPreparedCatalog, MassTabulation]:
    """Read prepared slabs into per-mass-bin position arrays for Corrfunc."""

    file_pairs = find_prepared_file_pairs(prepared_dir, file_tag=file_tag, seed=seed)
    edges = parse_logm_edges(logm_edges)
    lbox: float | None = None
    mpart: float | None = None
    first_attrs: dict[str, Any] = {}
    log_mass_min = np.inf
    log_mass_max = -np.inf
    total_halos = 0

    for index, pair in enumerate(file_pairs):
        with h5py.File(pair.halo_file, "r") as handle:
            attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
            slab_mpart = float(attrs.get("Mpart", attrs.get("ParticleMassHMsun")))
            slab_lbox = float(attrs.get("Lbox", attrs.get("BoxSizeHMpc")))
            if index == 0:
                lbox = slab_lbox
                mpart = slab_mpart
                first_attrs = attrs
            else:
                if not np.isclose(slab_lbox, lbox):
                    raise ValueError(f"Inconsistent Lbox in {pair.halo_file}.")
                if not np.isclose(slab_mpart, mpart):
                    raise ValueError(f"Inconsistent Mpart in {pair.halo_file}.")

            halo_n = _read_required_dataset(handle["halos"], "N", pair.halo_file).astype(np.float64)
            halo_mass = halo_n * slab_mpart
            if np.any(halo_mass <= 0.0):
                raise ValueError(f"Non-positive halo mass found in {pair.halo_file}.")
            log_mass = np.log10(halo_mass)
            log_mass_min = min(log_mass_min, float(np.min(log_mass)))
            log_mass_max = max(log_mass_max, float(np.max(log_mass)))
            total_halos += len(halo_mass)

    if total_halos == 0:
        raise ValueError("No halos found in prepared files.")
    if edges is None:
        lower = float(log_mass_min if logm_min is None else logm_min)
        upper = float(log_mass_max if logm_max is None else logm_max)
        if upper <= lower:
            raise ValueError("logm_max must be greater than logm_min.")
        edges = np.linspace(lower, np.nextafter(upper, np.inf), int(nmass_bins) + 1)
    else:
        nmass_bins = len(edges) - 1

    nmass = len(edges) - 1
    halo_chunks: list[list[np.ndarray]] = [[] for _ in range(nmass)]
    particle_chunks: list[list[np.ndarray]] = [[] for _ in range(nmass)]
    num_halo = np.zeros(nmass, dtype=np.int64)
    num_particle = np.zeros(nmass, dtype=np.int64)
    halo_mass_sum = np.zeros(nmass, dtype=np.float64)
    total_particles = 0
    position_dtype = np.dtype(dtype) if dtype is not None else None

    for pair in file_pairs:
        with h5py.File(pair.halo_file, "r") as handle:
            group = handle["halos"]
            halo_pos = _read_required_dataset(group, position_dataset, pair.halo_file)
            if position_dtype is None:
                position_dtype = halo_pos.dtype
            halo_pos = np.ascontiguousarray(
                np.mod(np.asarray(halo_pos, dtype=position_dtype), float(lbox)),
                dtype=position_dtype,
            )
            halo_n = _read_required_dataset(group, "N", pair.halo_file).astype(np.float64)
            halo_mass = halo_n * float(mpart)
            halo_bin = _assign_log_bins(np.log10(halo_mass), edges)

        valid_halo = halo_bin >= 0
        if np.any(valid_halo):
            num_halo += np.bincount(halo_bin[valid_halo], minlength=nmass)
            halo_mass_sum += np.bincount(
                halo_bin[valid_halo],
                weights=halo_mass[valid_halo],
                minlength=nmass,
            )
            _append_positions_by_bin(halo_chunks, halo_pos, halo_bin, nmass)

        with h5py.File(pair.particle_file, "r") as handle:
            group = handle["particles"]
            particle_pos = _read_required_dataset(group, position_dataset, pair.particle_file)
            particle_pos = np.ascontiguousarray(
                np.mod(np.asarray(particle_pos, dtype=position_dtype), float(lbox)),
                dtype=position_dtype,
            )
            particle_halo_index = _read_required_dataset(
                group, "halo_index", pair.particle_file
            ).astype(np.int64, copy=False)

        if np.any((particle_halo_index < 0) | (particle_halo_index >= len(halo_bin))):
            raise ValueError(f"particle halo_index outside halo table in {pair.particle_file}.")
        particle_bin = halo_bin[particle_halo_index]
        total_particles += len(particle_bin)
        valid_particle = particle_bin >= 0
        if np.any(valid_particle):
            num_particle += np.bincount(particle_bin[valid_particle], minlength=nmass)
            _append_positions_by_bin(particle_chunks, particle_pos, particle_bin, nmass)

    if position_dtype is None:
        position_dtype = np.dtype(dtype or np.float32)
    mean_log10 = np.empty(nmass, dtype=np.float64)
    nonempty = num_halo > 0
    mean_log10[nonempty] = np.log10(halo_mass_sum[nonempty] / num_halo[nonempty])
    mean_log10[~nonempty] = 0.5 * (edges[:-1][~nonempty] + edges[1:][~nonempty])

    mass_tab = MassTabulation(
        edges_log10=edges,
        mean_log10=mean_log10,
        halo_bin=np.empty(0, dtype=np.int32),
        particle_bin=np.empty(0, dtype=np.int32),
        num_halo=num_halo,
        num_particle=num_particle,
    )
    attrs = _prepared_output_attrs(first_attrs, file_pairs, position_dataset)
    catalog = BinnedPreparedCatalog(
        halo_by_bin=_concat_position_chunks(halo_chunks, position_dtype),
        particle_by_bin=_concat_position_chunks(particle_chunks, position_dtype),
        lbox=float(lbox),
        mpart=float(mpart),
        files=file_pairs,
        attrs=attrs,
        n_halos=total_halos,
        n_particles=total_particles,
    )
    return catalog, mass_tab


def _split_positions_by_bin(
    positions: np.ndarray,
    bin_index: np.ndarray,
    nmass_bins: int,
) -> list[np.ndarray]:
    return [np.ascontiguousarray(positions[bin_index == i]) for i in range(nmass_bins)]


def _empty_rppi(nrp: int, npi: int) -> np.ndarray:
    return np.zeros((nrp, npi), dtype=np.uint64)


def _empty_smu(ns: int, nmu: int) -> np.ndarray:
    return np.zeros((ns, nmu), dtype=np.uint64)


def _corrfunc_rppi(
    pos1: np.ndarray,
    pos2: np.ndarray | None,
    *,
    autocorr: bool,
    rp_edges: np.ndarray,
    pi_max: float,
    nthreads: int,
    boxsize: float,
) -> np.ndarray:
    from Corrfunc.theory.DDrppi import DDrppi

    nrp = len(rp_edges) - 1
    npi = int(round(pi_max))
    if autocorr:
        if len(pos1) < 2:
            return _empty_rppi(nrp, npi)
        result = DDrppi(
            1,
            nthreads,
            pi_max,
            rp_edges,
            pos1[:, 0],
            pos1[:, 1],
            pos1[:, 2],
            periodic=True,
            boxsize=boxsize,
        )
    else:
        if pos2 is None:
            raise ValueError("pos2 is required when autocorr=False.")
        if len(pos1) == 0 or len(pos2) == 0:
            return _empty_rppi(nrp, npi)
        result = DDrppi(
            0,
            nthreads,
            pi_max,
            rp_edges,
            X1=pos1[:, 0],
            Y1=pos1[:, 1],
            Z1=pos1[:, 2],
            X2=pos2[:, 0],
            Y2=pos2[:, 1],
            Z2=pos2[:, 2],
            periodic=True,
            boxsize=boxsize,
        )
    counts = np.asarray(result["npairs"], dtype=np.uint64)
    return counts.reshape(nrp, npi)


def _corrfunc_smu(
    pos1: np.ndarray,
    pos2: np.ndarray | None,
    *,
    autocorr: bool,
    s_edges: np.ndarray,
    mu_max: float,
    nmu_bins: int,
    nthreads: int,
    boxsize: float,
) -> np.ndarray:
    from Corrfunc.theory.DDsmu import DDsmu

    ns = len(s_edges) - 1
    if autocorr:
        if len(pos1) < 2:
            return _empty_smu(ns, nmu_bins)
        result = DDsmu(
            1,
            nthreads,
            s_edges,
            mu_max,
            nmu_bins,
            pos1[:, 0],
            pos1[:, 1],
            pos1[:, 2],
            periodic=True,
            boxsize=boxsize,
        )
    else:
        if pos2 is None:
            raise ValueError("pos2 is required when autocorr=False.")
        if len(pos1) == 0 or len(pos2) == 0:
            return _empty_smu(ns, nmu_bins)
        result = DDsmu(
            0,
            nthreads,
            s_edges,
            mu_max,
            nmu_bins,
            X1=pos1[:, 0],
            Y1=pos1[:, 1],
            Z1=pos1[:, 2],
            X2=pos2[:, 0],
            Y2=pos2[:, 1],
            Z2=pos2[:, 2],
            periodic=True,
            boxsize=boxsize,
        )
    counts = np.asarray(result["npairs"], dtype=np.uint64)
    return counts.reshape(ns, nmu_bins)


def _catalog_output_attrs(catalog: PreparedCatalog | BinnedPreparedCatalog) -> dict[str, Any]:
    keys = (
        "position_dataset",
        "corrfunc_position_wrap",
        "prepared_tags",
        "prepared_seeds",
        "prepared_slabs",
        "sim_name",
        "z_mock",
        "Mpart",
        "Lbox",
        "box_origin",
        "position_space",
        "concentration_tag",
        "concentration_numerator_key",
        "concentration_denominator_key",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key in catalog.attrs:
            out[key] = catalog.attrs[key]
    return out


def _create_output_file(
    output_path: Path,
    *,
    overwrite: bool,
    compression: str | None,
    attrs: dict[str, Any],
    mass_tab: MassTabulation,
    count_shape: tuple[int, ...],
) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it.")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(output_path, "w")
    for key, value in attrs.items():
        handle.attrs[key] = value
    mass_group = handle.create_group("mass")
    mass_group.create_dataset("edges_log10", data=mass_tab.edges_log10)
    mass_group.create_dataset("mean_log10", data=mass_tab.mean_log10)
    mass_group.create_dataset("num_halo", data=mass_tab.num_halo)
    mass_group.create_dataset("num_particle", data=mass_tab.num_particle)
    counts_group = handle.create_group("counts")
    kwargs = {"compression": compression, "chunks": True} if compression else {}
    hh = counts_group.create_dataset("HH", shape=count_shape, dtype="u8", **kwargs)
    hp = counts_group.create_dataset("HP", shape=count_shape, dtype="u8", **kwargs)
    pp = counts_group.create_dataset("PP", shape=count_shape, dtype="u8", **kwargs)
    return handle, hh, hp, pp


def _iter_bin_pairs(nmass_bins: int) -> Iterable[tuple[int, int]]:
    for i in range(nmass_bins):
        for j in range(i + 1):
            yield i, j


def _positions_by_mass_bin(
    catalog: PreparedCatalog | BinnedPreparedCatalog,
    mass_tab: MassTabulation,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if isinstance(catalog, BinnedPreparedCatalog):
        return catalog.halo_by_bin, catalog.particle_by_bin
    nmass = len(mass_tab.num_halo)
    return (
        _split_positions_by_bin(catalog.halo_pos, mass_tab.halo_bin, nmass),
        _split_positions_by_bin(catalog.particle_pos, mass_tab.particle_bin, nmass),
    )


def _catalog_sizes(catalog: PreparedCatalog | BinnedPreparedCatalog) -> tuple[int, int]:
    if isinstance(catalog, BinnedPreparedCatalog):
        return catalog.n_halos, catalog.n_particles
    return len(catalog.halo_pos), len(catalog.particle_pos)


def compute_rppi_paircounts(
    catalog: PreparedCatalog | BinnedPreparedCatalog,
    mass_tab: MassTabulation,
    output_path: str | Path,
    *,
    rp_edges: np.ndarray,
    pi_max: float,
    nthreads: int,
    boxsize: float | None = None,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """Compute HH, HP, and PP pair counts in mass bins for rp-pi bins."""

    output_path = Path(output_path)
    nmass = len(mass_tab.num_halo)
    nrp = len(rp_edges) - 1
    npi = int(round(pi_max))
    if npi <= 0:
        raise ValueError("pi_max must round to a positive integer number of pi bins.")
    boxsize = catalog.lbox if boxsize is None else float(boxsize)

    halo_by_bin, particle_by_bin = _positions_by_mass_bin(catalog, mass_tab)
    n_halos, n_particles = _catalog_sizes(catalog)

    attrs = {
        **_catalog_output_attrs(catalog),
        "clustering": "rppi",
        "boxsize": boxsize,
        "nthreads": int(nthreads),
        "n_halos": n_halos,
        "n_particles": n_particles,
        "n_mass_bins": nmass,
        "pi_max": float(pi_max),
        "schema_version": "paircounts_v1",
    }
    handle, hh_ds, hp_ds, pp_ds = _create_output_file(
        output_path,
        overwrite=overwrite,
        compression=compression,
        attrs=attrs,
        mass_tab=mass_tab,
        count_shape=(nmass, nmass, nrp, npi),
    )
    try:
        bins_group = handle.create_group("bins")
        bins_group.create_dataset("rp_edges", data=rp_edges)
        bins_group.create_dataset("pi_edges", data=np.arange(npi + 1, dtype=np.float64))

        for i, j in _iter_bin_pairs(nmass):
            print(f"rppi HH/PP mass bins {i} {j}", flush=True)
            if i == j:
                hh = _corrfunc_rppi(
                    halo_by_bin[i],
                    None,
                    autocorr=True,
                    rp_edges=rp_edges,
                    pi_max=pi_max,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
                pp = _corrfunc_rppi(
                    particle_by_bin[i],
                    None,
                    autocorr=True,
                    rp_edges=rp_edges,
                    pi_max=pi_max,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
            else:
                hh = _corrfunc_rppi(
                    halo_by_bin[i],
                    halo_by_bin[j],
                    autocorr=False,
                    rp_edges=rp_edges,
                    pi_max=pi_max,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
                pp = _corrfunc_rppi(
                    particle_by_bin[i],
                    particle_by_bin[j],
                    autocorr=False,
                    rp_edges=rp_edges,
                    pi_max=pi_max,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
            hh_ds[i, j] = hh
            pp_ds[i, j] = pp
            if i != j:
                hh_ds[j, i] = hh
                pp_ds[j, i] = pp

        for i in range(nmass):
            for j in range(nmass):
                print(f"rppi HP mass bins {i} {j}", flush=True)
                hp_ds[i, j] = _corrfunc_rppi(
                    halo_by_bin[i],
                    particle_by_bin[j],
                    autocorr=False,
                    rp_edges=rp_edges,
                    pi_max=pi_max,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
    finally:
        handle.close()
    return output_path


def compute_smu_paircounts(
    catalog: PreparedCatalog | BinnedPreparedCatalog,
    mass_tab: MassTabulation,
    output_path: str | Path,
    *,
    s_edges: np.ndarray,
    mu_max: float,
    nmu_bins: int,
    nthreads: int,
    boxsize: float | None = None,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    """Compute HH, HP, and PP pair counts in mass bins for s-mu bins."""

    output_path = Path(output_path)
    nmass = len(mass_tab.num_halo)
    ns = len(s_edges) - 1
    boxsize = catalog.lbox if boxsize is None else float(boxsize)

    halo_by_bin, particle_by_bin = _positions_by_mass_bin(catalog, mass_tab)
    n_halos, n_particles = _catalog_sizes(catalog)

    attrs = {
        **_catalog_output_attrs(catalog),
        "clustering": "smu",
        "boxsize": boxsize,
        "nthreads": int(nthreads),
        "n_halos": n_halos,
        "n_particles": n_particles,
        "n_mass_bins": nmass,
        "mu_max": float(mu_max),
        "nmu_bins": int(nmu_bins),
        "schema_version": "paircounts_v1",
    }
    handle, hh_ds, hp_ds, pp_ds = _create_output_file(
        output_path,
        overwrite=overwrite,
        compression=compression,
        attrs=attrs,
        mass_tab=mass_tab,
        count_shape=(nmass, nmass, ns, int(nmu_bins)),
    )
    try:
        bins_group = handle.create_group("bins")
        bins_group.create_dataset("s_edges", data=s_edges)
        bins_group.create_dataset(
            "mu_edges", data=np.linspace(0.0, mu_max, int(nmu_bins) + 1)
        )

        for i, j in _iter_bin_pairs(nmass):
            print(f"smu HH/PP mass bins {i} {j}", flush=True)
            if i == j:
                hh = _corrfunc_smu(
                    halo_by_bin[i],
                    None,
                    autocorr=True,
                    s_edges=s_edges,
                    mu_max=mu_max,
                    nmu_bins=nmu_bins,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
                pp = _corrfunc_smu(
                    particle_by_bin[i],
                    None,
                    autocorr=True,
                    s_edges=s_edges,
                    mu_max=mu_max,
                    nmu_bins=nmu_bins,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
            else:
                hh = _corrfunc_smu(
                    halo_by_bin[i],
                    halo_by_bin[j],
                    autocorr=False,
                    s_edges=s_edges,
                    mu_max=mu_max,
                    nmu_bins=nmu_bins,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
                pp = _corrfunc_smu(
                    particle_by_bin[i],
                    particle_by_bin[j],
                    autocorr=False,
                    s_edges=s_edges,
                    mu_max=mu_max,
                    nmu_bins=nmu_bins,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
            hh_ds[i, j] = hh
            pp_ds[i, j] = pp
            if i != j:
                hh_ds[j, i] = hh
                pp_ds[j, i] = pp

        for i in range(nmass):
            for j in range(nmass):
                print(f"smu HP mass bins {i} {j}", flush=True)
                hp_ds[i, j] = _corrfunc_smu(
                    halo_by_bin[i],
                    particle_by_bin[j],
                    autocorr=False,
                    s_edges=s_edges,
                    mu_max=mu_max,
                    nmu_bins=nmu_bins,
                    nthreads=nthreads,
                    boxsize=boxsize,
                )
    finally:
        handle.close()
    return output_path


def default_output_path(
    output_dir: str | Path,
    *,
    clustering: str,
    position_dataset: str,
    file_tag: str | None,
    nmass_bins: int,
) -> Path:
    tag = _sanitize_filename_piece(file_tag or "prepared")
    pos = _sanitize_filename_piece(position_dataset)
    return Path(output_dir) / f"paircounts_{clustering}_{pos}_{tag}_m{nmass_bins}.h5"


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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


def normalize_clustering_modes(value: object | None) -> list[str]:
    """Normalize a clustering mode config value to a list of modes."""

    if value is None:
        modes = ["rppi"]
    elif isinstance(value, str):
        modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        modes = [str(item).strip().lower() for item in value if str(item).strip()]
    if not modes:
        raise ValueError("At least one clustering mode is required.")
    unknown = sorted(set(modes) - {"rppi", "smu"})
    if unknown:
        raise ValueError(f"Unknown clustering mode(s): {unknown}")
    return modes


def compute_paircounts_from_prepared(
    *,
    prepared_dir: str | Path,
    output_dir: str | Path,
    file_tag: str | None = None,
    seed: int | None = None,
    position_dataset: str = "pos",
    clustering: object | None = None,
    nthreads: int = 32,
    boxsize: float | None = None,
    overwrite: bool = False,
    hdf5_compression: str | None = None,
    corrfunc_dtype: str | np.dtype | None = None,
    nmass_bins: int = 20,
    logm_min: float | None = None,
    logm_max: float | None = None,
    logm_edges: object | None = None,
    nrp_bins: int = 25,
    rp_min: float = -1.5,
    rp_max: float = 1.477,
    pi_max: float = 40.0,
    ns_bins: int = 25,
    s_min: float = -1.5,
    s_max: float = 1.477,
    mu_max: float = 1.0,
    nmu_bins: int = 120,
) -> list[Path]:
    """Load prepared catalogs, build mass bins, and compute requested paircounts."""

    modes = normalize_clustering_modes(clustering)
    print("loading prepared catalog into mass bins", flush=True)
    catalog, mass_tab = load_prepared_binned_catalog(
        prepared_dir,
        file_tag=file_tag,
        seed=seed,
        position_dataset=position_dataset,
        dtype=corrfunc_dtype,
        nmass_bins=int(nmass_bins),
        logm_min=logm_min,
        logm_max=logm_max,
        logm_edges=logm_edges,
    )
    print(
        f"loaded halos={catalog.n_halos:,} particles={catalog.n_particles:,} "
        f"slabs={len(catalog.files):,}",
        flush=True,
    )
    print("mass-bin halo counts:", mass_tab.num_halo, flush=True)
    print("mass-bin particle counts:", mass_tab.num_particle, flush=True)

    output_dir = Path(output_dir)
    unique_tags = sorted({item.tag for item in catalog.files})
    output_tag = file_tag or (unique_tags[0] if len(unique_tags) == 1 else "prepared")
    outputs: list[Path] = []

    if "rppi" in modes:
        rp_edges = np.logspace(float(rp_min), float(rp_max), int(nrp_bins) + 1)
        outputs.append(
            compute_rppi_paircounts(
                catalog,
                mass_tab,
                default_output_path(
                    output_dir,
                    clustering="rppi",
                    position_dataset=position_dataset,
                    file_tag=output_tag,
                    nmass_bins=len(mass_tab.num_halo),
                ),
                rp_edges=rp_edges,
                pi_max=float(pi_max),
                nthreads=int(nthreads),
                boxsize=boxsize,
                overwrite=bool(overwrite),
                compression=hdf5_compression,
            )
        )

    if "smu" in modes:
        s_edges = np.logspace(float(s_min), float(s_max), int(ns_bins) + 1)
        outputs.append(
            compute_smu_paircounts(
                catalog,
                mass_tab,
                default_output_path(
                    output_dir,
                    clustering="smu",
                    position_dataset=position_dataset,
                    file_tag=output_tag,
                    nmass_bins=len(mass_tab.num_halo),
                ),
                s_edges=s_edges,
                mu_max=float(mu_max),
                nmu_bins=int(nmu_bins),
                nthreads=int(nthreads),
                boxsize=boxsize,
                overwrite=bool(overwrite),
                compression=hdf5_compression,
            )
        )
    return outputs


def paircounts_from_config(
    path2config: str | Path,
    *,
    prepared_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    file_tag: str | None = None,
    seed: int | None = None,
    position_dataset: str | None = None,
    clustering: object | None = None,
    nthreads: int | None = None,
    boxsize: float | None = None,
    overwrite: bool | None = None,
    hdf5_compression: str | None = None,
    corrfunc_dtype: str | np.dtype | None = None,
    nmass_bins: int | None = None,
    logm_min: float | None = None,
    logm_max: float | None = None,
    logm_edges: object | None = None,
    nrp_bins: int | None = None,
    rp_min: float | None = None,
    rp_max: float | None = None,
    pi_max: float | None = None,
    ns_bins: int | None = None,
    s_min: float | None = None,
    s_max: float | None = None,
    mu_max: float | None = None,
    nmu_bins: int | None = None,
) -> list[Path]:
    """Compute paircounts from the universal YAML config file."""

    import yaml

    with open(path2config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    sim_params = config.get("sim_params", {})
    paths_params = config.get("paths", {})
    prepare_params = config.get("prepare_profiles", {})
    pair_params = config.get("paircounts", {})
    mass_params = pair_params.get("mass", {})
    rppi_params = pair_params.get("rppi", {})
    smu_params = pair_params.get("smu", {})

    sim_name = str(sim_params.get("sim_name", ""))
    z_mock = float(sim_params.get("z_mock", 0.0))

    prepared_value = _first_not_none(
        prepared_dir,
        pair_params.get("prepared_dir"),
        paths_params.get("prepared_dir"),
        prepare_params.get("output_dir"),
    )
    if prepared_value is None:
        output_root = sim_params.get("subsample_dir", prepare_params.get("output_root"))
        if output_root is None:
            raise KeyError(
                "Set paircounts.prepared_dir, paths.prepared_dir, "
                "prepare_profiles.output_dir, or sim_params.subsample_dir."
            )
        prepared_path = Path(output_root) / sim_name / _z_directory(z_mock)
    else:
        prepared_path = _format_config_path(prepared_value, sim_name=sim_name, z_mock=z_mock)

    output_value = _first_not_none(
        output_dir,
        pair_params.get("output_dir"),
        paths_params.get("paircounts_dir"),
    )
    if output_value is None:
        raise KeyError("Set paircounts.output_dir or paths.paircounts_dir.")
    output_path = _format_config_path(output_value, sim_name=sim_name, z_mock=z_mock)

    return compute_paircounts_from_prepared(
        prepared_dir=prepared_path,
        output_dir=output_path,
        file_tag=_first_not_none(file_tag, pair_params.get("file_tag")),
        seed=_first_not_none(seed, pair_params.get("seed"), prepare_params.get("seed")),
        position_dataset=str(
            _first_not_none(position_dataset, pair_params.get("position_dataset"), "pos")
        ),
        clustering=_first_not_none(clustering, pair_params.get("clustering"), "rppi"),
        nthreads=int(_first_not_none(nthreads, pair_params.get("nthreads"), 32)),
        boxsize=_first_not_none(boxsize, pair_params.get("boxsize")),
        overwrite=bool(_first_not_none(overwrite, pair_params.get("overwrite"), False)),
        hdf5_compression=_first_not_none(
            hdf5_compression,
            pair_params.get("hdf5_compression"),
            pair_params.get("compression"),
        ),
        corrfunc_dtype=_first_not_none(corrfunc_dtype, pair_params.get("corrfunc_dtype")),
        nmass_bins=int(_first_not_none(nmass_bins, mass_params.get("nmass_bins"), pair_params.get("nmass_bins"), 20)),
        logm_min=_first_not_none(logm_min, mass_params.get("logm_min"), pair_params.get("logm_min")),
        logm_max=_first_not_none(logm_max, mass_params.get("logm_max"), pair_params.get("logm_max")),
        logm_edges=_first_not_none(logm_edges, mass_params.get("logm_edges"), pair_params.get("logm_edges")),
        nrp_bins=int(_first_not_none(nrp_bins, rppi_params.get("nrp_bins"), pair_params.get("nrp_bins"), 25)),
        rp_min=float(_first_not_none(rp_min, rppi_params.get("rp_min"), pair_params.get("rp_min"), -1.5)),
        rp_max=float(_first_not_none(rp_max, rppi_params.get("rp_max"), pair_params.get("rp_max"), 1.477)),
        pi_max=float(_first_not_none(pi_max, rppi_params.get("pi_max"), pair_params.get("pi_max"), 40.0)),
        ns_bins=int(_first_not_none(ns_bins, smu_params.get("ns_bins"), pair_params.get("ns_bins"), 25)),
        s_min=float(_first_not_none(s_min, smu_params.get("s_min"), pair_params.get("s_min"), -1.5)),
        s_max=float(_first_not_none(s_max, smu_params.get("s_max"), pair_params.get("s_max"), 1.477)),
        mu_max=float(_first_not_none(mu_max, smu_params.get("mu_max"), pair_params.get("mu_max"), 1.0)),
        nmu_bins=int(_first_not_none(nmu_bins, smu_params.get("nmu_bins"), pair_params.get("nmu_bins"), 120)),
    )

