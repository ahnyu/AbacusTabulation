#!/usr/bin/env python
"""CLI wrapper for mass-bin HH/HP/PP pair counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("At least one clustering mode is required.")
    unknown = sorted(set(modes) - {"rppi", "smu"})
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown clustering mode(s): {unknown}")
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute mass-bin tabulated HH, HP, and PP pair counts."
    )
    parser.add_argument("--path2config", help="Universal YAML config file.")
    parser.add_argument("--prepared-dir", help="Directory with prepared HDF5 slabs.")
    parser.add_argument("--output-dir", help="Directory for paircount HDF5 files.")
    parser.add_argument(
        "--file-tag",
        help="Prepared file suffix after seed, e.g. abacustab_profiles_conc-r98-L2com-over-r25-L2com.",
    )
    parser.add_argument("--seed", type=int, help="Prepared-file seed to select.")
    parser.add_argument("--position-dataset", help="Position dataset to count, usually pos or pos_rsd.")
    parser.add_argument("--clustering", type=_parse_modes, help="Comma-separated modes: rppi, smu, or rppi,smu.")
    parser.add_argument("--nthreads", type=int)
    parser.add_argument("--boxsize", type=float, help="Override box size; default from prepared attrs.")
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--hdf5-compression", help="Optional HDF5 compression, e.g. gzip.")
    parser.add_argument("--corrfunc-dtype", help="Optional dtype cast for Corrfunc positions, e.g. f4 or f8.")

    parser.add_argument("--nmass-bins", type=int)
    parser.add_argument("--logm-min", type=float)
    parser.add_argument("--logm-max", type=float)
    parser.add_argument("--logm-edges", help="Comma-separated log10 mass edges, or path to a text file of edges.")

    parser.add_argument("--nrp-bins", type=int)
    parser.add_argument("--rp-min", type=float, help="log10 lower rp edge.")
    parser.add_argument("--rp-max", type=float, help="log10 upper rp edge.")
    parser.add_argument("--pi-max", type=float)

    parser.add_argument("--ns-bins", type=int)
    parser.add_argument("--s-min", type=float, help="log10 lower s edge.")
    parser.add_argument("--s-max", type=float, help="log10 upper s edge.")
    parser.add_argument("--mu-max", type=float)
    parser.add_argument("--nmu-bins", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from abacustabulation.paircounts import (
        compute_paircounts_from_prepared,
        paircounts_from_config,
    )

    if args.path2config:
        outputs = paircounts_from_config(
            args.path2config,
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            file_tag=args.file_tag,
            seed=args.seed,
            position_dataset=args.position_dataset,
            clustering=args.clustering,
            nthreads=args.nthreads,
            boxsize=args.boxsize,
            overwrite=args.overwrite,
            hdf5_compression=args.hdf5_compression,
            corrfunc_dtype=args.corrfunc_dtype,
            nmass_bins=args.nmass_bins,
            logm_min=args.logm_min,
            logm_max=args.logm_max,
            logm_edges=args.logm_edges,
            nrp_bins=args.nrp_bins,
            rp_min=args.rp_min,
            rp_max=args.rp_max,
            pi_max=args.pi_max,
            ns_bins=args.ns_bins,
            s_min=args.s_min,
            s_max=args.s_max,
            mu_max=args.mu_max,
            nmu_bins=args.nmu_bins,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--prepared-dir", args.prepared_dir),
                ("--output-dir", args.output_dir),
            )
            if value is None
        ]
        if missing:
            parser.error("Missing required arguments without --path2config: " + ", ".join(missing))

        outputs = compute_paircounts_from_prepared(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            file_tag=args.file_tag,
            seed=args.seed,
            position_dataset=args.position_dataset or "pos",
            clustering=args.clustering or ["rppi"],
            nthreads=args.nthreads if args.nthreads is not None else 32,
            boxsize=args.boxsize,
            overwrite=bool(args.overwrite),
            hdf5_compression=args.hdf5_compression,
            corrfunc_dtype=args.corrfunc_dtype,
            nmass_bins=args.nmass_bins if args.nmass_bins is not None else 20,
            logm_min=args.logm_min,
            logm_max=args.logm_max,
            logm_edges=args.logm_edges,
            nrp_bins=args.nrp_bins if args.nrp_bins is not None else 18,
            rp_min=args.rp_min if args.rp_min is not None else -1.5,
            rp_max=args.rp_max if args.rp_max is not None else 1.5,
            pi_max=args.pi_max if args.pi_max is not None else 40.0,
            ns_bins=args.ns_bins if args.ns_bins is not None else 18,
            s_min=args.s_min if args.s_min is not None else -1.5,
            s_max=args.s_max if args.s_max is not None else 1.5,
            mu_max=args.mu_max if args.mu_max is not None else 1.0,
            nmu_bins=args.nmu_bins if args.nmu_bins is not None else 100,
        )

    for output in outputs:
        print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
