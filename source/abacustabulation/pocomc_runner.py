"""Shared pocoMC execution for config-driven fitting scripts."""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
from typing import Any
import sys

import numpy as np


_MULTIPROCESSING_PROBLEM = None


def _set_multiprocessing_problem(problem: Any) -> None:
    global _MULTIPROCESSING_PROBLEM
    _MULTIPROCESSING_PROBLEM = problem


def _multiprocessing_loglike(theta: Any) -> float:
    if _MULTIPROCESSING_PROBLEM is None:
        raise RuntimeError(
            "Multiprocessing worker was not initialized with a fitting problem."
        )
    return _MULTIPROCESSING_PROBLEM.loglike(theta)


def _output_settings(
    problem: Any,
    output_dir_override: str | None,
    output_prefix_override: str | None,
) -> tuple[Path, str]:
    output = problem.fit_config.get("output", {})
    output_dir = problem.format_path(
        output_dir_override or output.get("output_dir", "fit_outputs")
    )
    prefix = str(output_prefix_override or output.get("prefix", "abacustab_fit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, prefix


def _configured_n_processes(override: int | None, config: dict[str, Any]) -> int:
    value = override if override is not None else config.get("n_processes", 1)
    n_processes = int(value)
    if n_processes < 1:
        raise ValueError("fit.mcmc.pocomc.n_processes must be at least 1.")
    return n_processes


def _multiprocessing_context(start_method: str | None):
    method = "fork" if start_method is None else str(start_method)
    try:
        return get_context(method)
    except ValueError as exc:
        raise ValueError(f"Unsupported multiprocessing start method {method!r}.") from exc


def _run_sampler(
    problem: Any,
    pc: Any,
    *,
    output_dir: Path,
    prefix: str,
    pool: Any,
    loglike: Any,
    resume_state_path: str | None,
) -> Any:
    config = problem.fit_config.get("mcmc", {}).get("pocomc", {})
    sampler_kwargs = dict(config.get("sampler", {}))
    run_kwargs = dict(config.get("run", {"n_total": 4096, "save_every": 5}))
    if resume_state_path is not None:
        run_kwargs["resume_state_path"] = resume_state_path
    sampler = pc.Sampler(
        problem.pocomc_prior(),
        loglike,
        pool=pool,
        output_dir=str(output_dir),
        output_label=prefix,
        **sampler_kwargs,
    )
    sampler.run(**run_kwargs)
    return sampler


def _postprocess_fail_on_error(problem: Any) -> bool:
    post = problem.fit_config.get("postprocess", {})
    return bool(post.get("fail_on_error", post.get("fail_on_postprocess_error", False)))


def run_pocomc(
    problem: Any,
    *,
    output_dir_override: str | None = None,
    output_prefix_override: str | None = None,
    resume_state_path_override: str | None = None,
    disable_mpi: bool = False,
    n_processes_override: int | None = None,
    multiprocessing_start_method_override: str | None = None,
) -> int:
    """Run pocoMC and write chains, evidence, and configured derived outputs."""

    try:
        import pocomc as pc
    except ImportError as exc:  # pragma: no cover - depends on cluster environment.
        raise SystemExit("pocomc is required to run the fitting scripts.") from exc

    output_dir, prefix = _output_settings(
        problem,
        output_dir_override,
        output_prefix_override,
    )
    config = problem.fit_config.get("mcmc", {}).get("pocomc", {})
    resume_state_path = resume_state_path_override or config.get("resume_state_path")
    use_mpi = bool(config.get("use_mpi", False)) and not disable_mpi
    n_processes = _configured_n_processes(n_processes_override, config)
    if use_mpi and n_processes > 1:
        raise ValueError(
            "Use either MPI or local multiprocessing, not both; "
            "disable MPI when setting n_processes above one."
        )

    if use_mpi:
        with pc.parallel.MPIPool() as pool:
            sampler = _run_sampler(
                problem,
                pc,
                output_dir=output_dir,
                prefix=prefix,
                pool=pool,
                loglike=problem.loglike,
                resume_state_path=resume_state_path,
            )
    elif n_processes > 1:
        start_method = (
            multiprocessing_start_method_override
            or config.get("multiprocessing_start_method", "fork")
        )
        context = _multiprocessing_context(start_method)
        _set_multiprocessing_problem(problem)
        print(
            f"using local multiprocessing pool with {n_processes} processes",
            flush=True,
        )
        with context.Pool(
            n_processes,
            initializer=_set_multiprocessing_problem,
            initargs=(problem,),
        ) as pool:
            sampler = _run_sampler(
                problem,
                pc,
                output_dir=output_dir,
                prefix=prefix,
                pool=pool,
                loglike=_multiprocessing_loglike,
                resume_state_path=resume_state_path,
            )
    else:
        sampler = _run_sampler(
            problem,
            pc,
            output_dir=output_dir,
            prefix=prefix,
            pool=None,
            loglike=problem.loglike,
            resume_state_path=resume_state_path,
        )

    samples, weights, loglike, logprior = sampler.posterior()
    chain = np.column_stack([samples, weights, loglike, logprior])
    header = " ".join((*problem.parameter_names, "weight", "loglike", "logprior"))
    chain_path = output_dir / f"{prefix}_chains.txt"
    np.savetxt(chain_path, chain, header=header)
    try:
        evidence = np.asarray(sampler.evidence())
    except Exception:
        evidence = np.array([], dtype=np.float64)
    if evidence.size:
        np.savetxt(
            output_dir / f"{prefix}_evidence.txt",
            evidence.reshape(1, -1),
        )

    from .derived import write_mcmc_derived

    derived_summary = {}
    try:
        derived_summary = write_mcmc_derived(
            problem,
            samples,
            weights,
            loglike,
            logprior,
            output_dir,
            prefix,
        )
    except Exception as exc:
        if _postprocess_fail_on_error(problem):
            raise
        error_path = output_dir / f"{prefix}_postprocess_error.txt"
        with open(error_path, "w", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")
        print(
            f"postprocess failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    print(f"wrote {chain_path}", flush=True)
    if derived_summary:
        print(f"wrote {output_dir / (prefix + '_chains_derived.txt')}", flush=True)
    return 0
