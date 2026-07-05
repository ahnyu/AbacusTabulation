#!/usr/bin/env python
"""CLI wrapper for high-resolution halo mass functions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute a high-resolution HMF from prepared halo slabs.")
    parser.add_argument("--path2config", help="Universal YAML config file.")
    parser.add_argument("--prepared-dir", help="Prepared-file root or expanded directory.")
    parser.add_argument("--output-dir", help="HMF output root or expanded directory.")
    parser.add_argument("--output-path", help="Exact HMF output file path.")
    parser.add_argument("--file-tag", help="Prepared file tag to select.")
    parser.add_argument("--seed", type=int, help="Prepared-file seed to select.")
    parser.add_argument("--n-bins", type=int, help="Number of log10 mass bins.")
    parser.add_argument("--logm-min", type=float)
    parser.add_argument("--logm-max", type=float)
    parser.add_argument("--logm-edges", help="Comma-separated log10 mass edges, or path to a text file of edges.")
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--hdf5-compression", help="Optional HDF5 compression, e.g. gzip.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from abacustabulation.hmf import compute_hmf_from_prepared, hmf_from_config

    if args.path2config:
        output = hmf_from_config(
            args.path2config,
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            output_path=args.output_path,
            file_tag=args.file_tag,
            seed=args.seed,
            n_bins=args.n_bins,
            logm_min=args.logm_min,
            logm_max=args.logm_max,
            logm_edges=args.logm_edges,
            overwrite=args.overwrite,
            hdf5_compression=args.hdf5_compression,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--prepared-dir", args.prepared_dir),
                ("--output-dir or --output-path", args.output_dir or args.output_path),
            )
            if value is None
        ]
        if missing:
            parser.error("Missing required arguments without --path2config: " + ", ".join(missing))
        output = compute_hmf_from_prepared(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            output_path=args.output_path,
            file_tag=args.file_tag,
            seed=args.seed,
            n_bins=args.n_bins if args.n_bins is not None else 512,
            logm_min=args.logm_min,
            logm_max=args.logm_max,
            logm_edges=args.logm_edges,
            overwrite=bool(args.overwrite),
            hdf5_compression=args.hdf5_compression,
        )

    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
