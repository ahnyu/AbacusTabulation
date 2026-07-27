#!/usr/bin/env python
"""Optimize HOD parameters using the config-driven tabulated likelihood."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

INVALID_OBJECTIVE = 1.0e100
_OPTIMIZATION_PROBLEM = None
_OPTIMIZATION_TARGET = None
_OPTIMIZATION_OBJECTIVE = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize tabulated HOD fit parameters.")
    parser.add_argument("--path2config", required=True, help="Universal YAML config with a fit: block.")
    parser.add_argument("--method", help="scipy.optimize method; overrides fit.optimization.method.")
    parser.add_argument("--target", choices=("posterior", "like"), help="Objective target to maximize.")
    parser.add_argument("--output-dir", help="Override fit.output.output_dir.")
    parser.add_argument("--output-prefix", help="Override fit.output.prefix.")
    parser.add_argument(
        "--threads-per-process",
        type=int,
        help="Native math threads for optimization; defaults to the config value or 1.",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip initial theory/data length validation.")
    return parser


def _objective(problem, target: str):
    import numpy as np

    func = problem.negative_logposterior if target == "posterior" else problem.negative_loglike
    bounds = np.asarray(problem.bounds(), dtype=np.float64)

    def wrapped(theta):
        theta = np.asarray(theta, dtype=np.float64)
        if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
            return INVALID_OBJECTIVE
        value = func(theta)
        return INVALID_OBJECTIVE if not np.isfinite(value) else float(value)

    return wrapped


def _initialize_optimization_worker(problem, target: str) -> None:
    global _OPTIMIZATION_PROBLEM, _OPTIMIZATION_TARGET, _OPTIMIZATION_OBJECTIVE
    _OPTIMIZATION_PROBLEM = problem
    _OPTIMIZATION_TARGET = target
    _OPTIMIZATION_OBJECTIVE = _objective(problem, target)


def _parallel_objective(theta):
    if _OPTIMIZATION_OBJECTIVE is None:
        raise RuntimeError("Parallel optimization worker was not initialized.")
    return _OPTIMIZATION_OBJECTIVE(theta)


def _output_settings(problem, args) -> tuple[Path, str]:
    fit = problem.fit_config
    output = fit.get("output", {})
    output_dir = problem.format_path(args.output_dir or output.get("output_dir", "fit_outputs"))
    prefix = str(args.output_prefix or output.get("prefix", "abacustab_fit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, prefix


def _postprocess_fail_on_error(problem) -> bool:
    post = problem.fit_config.get("postprocess", {})
    return bool(post.get("fail_on_error", post.get("fail_on_postprocess_error", False)))


def _json_ready(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from abacustabulation.runtime import configure_optimization_native_threads

    native_threads = configure_optimization_native_threads(
        args.path2config,
        override=args.threads_per_process,
    )
    print(f"limiting native math libraries to {native_threads} thread(s) per process")

    import numpy as np

    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover - depends on cluster env.
        raise SystemExit("scipy is required for scripts/run_fit_optimize.py") from exc

    from abacustabulation.fitting import load_fitting_problem_from_config

    problem = load_fitting_problem_from_config(args.path2config, validate=not args.no_validate)
    opt_config = problem.fit_config.get("optimization", {})
    method = str(args.method or opt_config.get("method", "Nelder-Mead"))
    target = str(args.target or opt_config.get("target", "posterior")).lower()
    if target not in {"posterior", "like"}:
        raise ValueError("fit.optimization.target must be 'posterior' or 'like'.")

    start = np.asarray(opt_config.get("start", problem.initial_vector()), dtype=np.float64)
    if start.size != len(problem.parameters):
        raise ValueError(f"Optimization start has length {start.size}; expected {len(problem.parameters)}.")
    bounds = np.asarray(problem.bounds(), dtype=np.float64)
    if np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
        raise ValueError("Optimization start lies outside fit parameter bounds.")
    options = dict(opt_config.get("options", {}))
    objective = _objective(problem, target)

    if method.lower() in {"differential_evolution", "de"}:
        worker_count = int(options.pop("workers", 1))
        if worker_count < 1:
            raise ValueError("fit.optimization.options.workers must be at least 1.")
        if worker_count == 1:
            result = optimize.differential_evolution(
                objective,
                problem.bounds(),
                **options,
            )
        else:
            from multiprocessing import get_context

            start_method = str(
                opt_config.get("multiprocessing_start_method", "fork")
            )
            options["updating"] = "deferred"
            _initialize_optimization_worker(problem, target)
            context = get_context(start_method)
            print(
                f"using optimization pool with {worker_count} processes",
                flush=True,
            )
            with context.Pool(
                worker_count,
                initializer=_initialize_optimization_worker,
                initargs=(problem, target),
            ) as pool:
                result = optimize.differential_evolution(
                    _parallel_objective,
                    problem.bounds(),
                    workers=pool.map,
                    **options,
                )
        best = np.asarray(result.x, dtype=np.float64)
    else:
        kwargs = {"method": method, "options": options}
        if method.lower() not in {"nelder-mead", "bfgs", "cg"}:
            kwargs["bounds"] = problem.bounds()
        result = optimize.minimize(objective, start, **kwargs)
        best = np.asarray(result.x, dtype=np.float64)

    loglike = problem.loglike(best)
    logprior = problem.logprior(best)
    logposterior = problem.logposterior(best)
    theory = problem.theory_vector(best)
    output_dir, prefix = _output_settings(problem, args)

    header = " ".join((*problem.parameter_names, "loglike", "logprior", "logposterior"))
    np.savetxt(
        output_dir / f"{prefix}_optimum.txt",
        np.concatenate([best, [loglike, logprior, logposterior]])[None, :],
        header=header,
    )
    np.savetxt(output_dir / f"{prefix}_theory_vector.txt", theory)
    summary = {
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "method": method,
        "target": target,
        "parameter_names": list(problem.parameter_names),
        "best_parameters": {name: float(value) for name, value in zip(problem.parameter_names, best, strict=True)},
        "loglike": float(loglike),
        "logprior": float(logprior),
        "logposterior": float(logposterior),
        "n_data": int(problem.data.size),
        "n_density_constraints": int(len(problem.density_constraints)),
        "minus_2_loglike": float(-2.0 * loglike),
    }
    summary_path = output_dir / f"{prefix}_optimization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)

    from abacustabulation.derived import write_optimization_derived

    try:
        derived = write_optimization_derived(problem, best, output_dir, prefix)
    except Exception as exc:
        if _postprocess_fail_on_error(problem):
            raise
        summary["postprocess_error"] = {"type": type(exc).__name__, "message": str(exc)}
        error_path = output_dir / f"{prefix}_postprocess_error.txt"
        with open(error_path, "w", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")
        print(f"postprocess failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    else:
        if derived:
            summary["derived"] = derived
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)
    print(json.dumps(_json_ready(summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
