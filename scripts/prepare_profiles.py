#!/usr/bin/env python
"""CLI wrapper for AbacusTabulation halo/NFW profile preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))



def _parse_slabs(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    slabs: list[int] = []
    for value in values:
        for piece in value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            slabs.append(int(piece))
    return slabs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare AbacusSummit halo positions and compact NFW particle positions."
    )
    parser.add_argument("--path2config", help="YAML config with sim_params/prepare_profiles.")
    parser.add_argument("--sim-dir", help="Directory containing AbacusSummit simulations.")
    parser.add_argument("--sim-name", help="Simulation name, e.g. AbacusSummit_base_c000_ph000.")
    parser.add_argument("--z", "--z-mock", dest="z_mock", type=float, help="Snapshot redshift.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for prepared files. Without --path2config this is used directly; "
            "with config prefer prepare_profiles.output_dir."
        ),
    )
    parser.add_argument("--slabs", nargs="*", help="Slab indices, e.g. --slabs 0 1 2 or 0,1,2.")
    parser.add_argument("--seed", type=int, default=600)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--position-space",
        choices=("real", "rsd", "both"),
        default="real",
        help="Store real-space positions, RSD positions as pos, or both pos and pos_rsd.",
    )
    parser.add_argument("--los-axis", default="z", help="RSD axis: x, y, z, 0, 1, or 2.")
    parser.add_argument(
        "--box-origin",
        choices=("center", "zero"),
        default="center",
        help="Periodic wrap convention for catalog positions.",
    )
    parser.add_argument("--nfw-count-factor", type=float)
    parser.add_argument("--min-particles-per-halo", type=int, default=1)
    parser.add_argument("--max-particles-per-halo", type=int)
    parser.add_argument(
        "--satellite-velocity-model",
        choices=("gaussian", "halo"),
        default="gaussian",
        help="For RSD output, use parent halo plus Gaussian virial draw, or parent halo only.",
    )
    parser.add_argument(
        "--concentration-numerator-key",
        help="Halo catalog key for the numerator of concentration; default r98_L2com.",
    )
    parser.add_argument(
        "--concentration-denominator-key",
        help="Halo catalog key for the denominator of concentration; default r25_L2com.",
    )
    parser.add_argument("--position-dtype", default="f4")
    parser.add_argument("--index-dtype", default="i4")
    parser.add_argument("--particle-chunk-size", type=int, default=2_000_000)
    parser.add_argument(
        "--write-particle-radius",
        action="store_true",
        help="Also store sampled NFW radius; useful for validation, off by default for memory.",
    )
    parser.add_argument("--hdf5-compression")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    slabs = _parse_slabs(args.slabs)

    from abacustabulation.prepare import (
        prepare_all_slabs,
        prepare_from_config,
        z_directory,
    )

    if args.path2config:
        outputs = prepare_from_config(
            args.path2config,
            alt_simname=args.sim_name,
            alt_z=args.z_mock,
            seed=args.seed,
            overwrite=args.overwrite,
            slab_indices=slabs,
            concentration_numerator_key=args.concentration_numerator_key,
            concentration_denominator_key=args.concentration_denominator_key,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--sim-dir", args.sim_dir),
                ("--sim-name", args.sim_name),
                ("--z", args.z_mock),
                ("--output-dir", args.output_dir),
            )
            if value is None
        ]
        if missing:
            parser.error("Missing required arguments without --path2config: " + ", ".join(missing))

        outputs = prepare_all_slabs(
            sim_dir=args.sim_dir,
            sim_name=args.sim_name,
            z_mock=args.z_mock,
            output_dir=Path(args.output_dir) / args.sim_name / z_directory(args.z_mock),
            slab_indices=slabs,
            n_parallel=args.n_parallel,
            seed=args.seed,
            overwrite=args.overwrite,
            position_space=args.position_space,
            los_axis=args.los_axis,
            box_origin=args.box_origin,
            nfw_count_factor=args.nfw_count_factor,
            min_particles_per_halo=args.min_particles_per_halo,
            max_particles_per_halo=args.max_particles_per_halo,
            satellite_velocity_model=args.satellite_velocity_model,
            concentration_numerator_key=(
                args.concentration_numerator_key or "r98_L2com"
            ),
            concentration_denominator_key=(
                args.concentration_denominator_key or "r25_L2com"
            ),
            position_dtype=args.position_dtype,
            index_dtype=args.index_dtype,
            particle_chunk_size=args.particle_chunk_size,
            write_particle_radius=args.write_particle_radius,
            hdf5_compression=args.hdf5_compression,
        )

    for output in sorted(outputs, key=lambda item: item.halo_file.name):
        print(
            f"{output.halo_file}  halos={output.n_halos:,}  "
            f"{output.particle_file}  particles={output.n_particles:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
