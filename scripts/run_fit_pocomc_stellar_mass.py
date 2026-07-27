#!/usr/bin/env python
"""Run pocoMC for a simultaneous stellar-mass-split HOD fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pocoMC for stellar-mass-split tabulated HOD fitting."
    )
    parser.add_argument(
        "--path2config",
        required=True,
        help="Universal YAML config with a fit: block.",
    )
    parser.add_argument("--output-dir", help="Override fit.output.output_dir.")
    parser.add_argument("--output-prefix", help="Override fit.output.prefix.")
    parser.add_argument(
        "--resume-state-path",
        help="Override fit.mcmc.pocomc.resume_state_path.",
    )
    parser.add_argument(
        "--no-mpi",
        action="store_true",
        help="Disable fit.mcmc.pocomc.use_mpi.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        help="Local worker count; overrides fit.mcmc.pocomc.n_processes.",
    )
    parser.add_argument(
        "--mp-start-method",
        help="Override fit.mcmc.pocomc.multiprocessing_start_method.",
    )
    parser.add_argument(
        "--threads-per-process",
        type=int,
        help="Native math threads per process; defaults to the config value or 1.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip initial theory/data length validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from abacustabulation.runtime import configure_pocomc_native_threads

    native_threads = configure_pocomc_native_threads(
        args.path2config,
        override=args.threads_per_process,
    )
    print(f"limiting native math libraries to {native_threads} thread(s) per process")

    from abacustabulation.fitting import (
        load_stellar_mass_fitting_problem_from_config,
    )
    from abacustabulation.pocomc_runner import run_pocomc

    problem = load_stellar_mass_fitting_problem_from_config(
        args.path2config,
        validate=not args.no_validate,
    )
    return run_pocomc(
        problem,
        output_dir_override=args.output_dir,
        output_prefix_override=args.output_prefix,
        resume_state_path_override=args.resume_state_path,
        disable_mpi=args.no_mpi,
        n_processes_override=args.n_processes,
        multiprocessing_start_method_override=args.mp_start_method,
    )


if __name__ == "__main__":
    raise SystemExit(main())
