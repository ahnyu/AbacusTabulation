#!/usr/bin/env python
"""Run pocoMC using the config-driven tabulated HOD likelihood."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pocoMC for tabulated HOD fitting.")
    parser.add_argument("--path2config", required=True, help="Universal YAML config with a fit: block.")
    parser.add_argument("--output-dir", help="Override fit.output.output_dir.")
    parser.add_argument("--output-prefix", help="Override fit.output.prefix.")
    parser.add_argument("--resume-state-path", help="Override fit.mcmc.pocomc.resume_state_path.")
    parser.add_argument("--no-mpi", action="store_true", help="Disable fit.mcmc.pocomc.use_mpi.")
    parser.add_argument("--no-validate", action="store_true", help="Skip initial theory/data length validation.")
    return parser


def _output_settings(problem, args) -> tuple[Path, str]:
    fit = problem.fit_config
    output = fit.get("output", {})
    output_dir = problem.format_path(args.output_dir or output.get("output_dir", "fit_outputs"))
    prefix = str(args.output_prefix or output.get("prefix", "abacustab_fit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, prefix


def _run_sampler(problem, pc, *, output_dir: Path, prefix: str, pool, resume_state_path: str | None) -> object:
    mcmc_config = problem.fit_config.get("mcmc", {}).get("pocomc", {})
    sampler_kwargs = dict(mcmc_config.get("sampler", {}))
    run_kwargs = dict(mcmc_config.get("run", {"n_total": 4096, "save_every": 5}))
    if resume_state_path is not None:
        run_kwargs["resume_state_path"] = resume_state_path
    sampler = pc.Sampler(
        problem.pocomc_prior(),
        problem.loglike,
        pool=pool,
        output_dir=str(output_dir),
        output_label=prefix,
        **sampler_kwargs,
    )
    sampler.run(**run_kwargs)
    return sampler


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import pocomc as pc
    except ImportError as exc:  # pragma: no cover - depends on cluster env.
        raise SystemExit("pocomc is required for scripts/run_fit_pocomc.py") from exc

    from abacustabulation.fitting import load_fitting_problem_from_config

    problem = load_fitting_problem_from_config(args.path2config, validate=not args.no_validate)
    output_dir, prefix = _output_settings(problem, args)
    pocomc_config = problem.fit_config.get("mcmc", {}).get("pocomc", {})
    resume_state_path = args.resume_state_path or pocomc_config.get("resume_state_path")
    use_mpi = bool(pocomc_config.get("use_mpi", False)) and not args.no_mpi

    if use_mpi:
        with pc.parallel.MPIPool() as pool:
            sampler = _run_sampler(
                problem,
                pc,
                output_dir=output_dir,
                prefix=prefix,
                pool=pool,
                resume_state_path=resume_state_path,
            )
    else:
        sampler = _run_sampler(
            problem,
            pc,
            output_dir=output_dir,
            prefix=prefix,
            pool=None,
            resume_state_path=resume_state_path,
        )

    samples, weights, logl, logp = sampler.posterior()
    chain = np.column_stack([samples, weights, logl, logp])
    header = " ".join((*problem.parameter_names, "weight", "loglike", "logprior"))
    np.savetxt(output_dir / f"{prefix}_chains.txt", chain, header=header)
    try:
        evidence = np.asarray(sampler.evidence())
    except Exception:
        evidence = np.array([], dtype=np.float64)
    if evidence.size:
        np.savetxt(output_dir / f"{prefix}_evidence.txt", evidence.reshape(1, -1))
    print(f"wrote {output_dir / (prefix + '_chains.txt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
