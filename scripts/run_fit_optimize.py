#!/usr/bin/env python
"""Optimize HOD parameters using the config-driven tabulated likelihood."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize tabulated HOD fit parameters.")
    parser.add_argument("--path2config", required=True, help="Universal YAML config with a fit: block.")
    parser.add_argument("--method", help="scipy.optimize method; overrides fit.optimization.method.")
    parser.add_argument("--target", choices=("posterior", "like"), help="Objective target to maximize.")
    parser.add_argument("--output-dir", help="Override fit.output.output_dir.")
    parser.add_argument("--output-prefix", help="Override fit.output.prefix.")
    parser.add_argument("--no-validate", action="store_true", help="Skip initial theory/data length validation.")
    return parser


def _objective(problem, target: str):
    func = problem.negative_logposterior if target == "posterior" else problem.negative_loglike

    def wrapped(theta):
        value = func(theta)
        return 1.0e300 if not np.isfinite(value) else float(value)

    return wrapped


def _output_settings(problem, args) -> tuple[Path, str]:
    fit = problem.fit_config
    output = fit.get("output", {})
    output_dir = problem.format_path(args.output_dir or output.get("output_dir", "fit_outputs"))
    prefix = str(args.output_prefix or output.get("prefix", "abacustab_fit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, prefix


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
    options = dict(opt_config.get("options", {}))
    objective = _objective(problem, target)

    if method.lower() in {"differential_evolution", "de"}:
        result = optimize.differential_evolution(objective, problem.bounds(), **options)
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
    }
    with open(output_dir / f"{prefix}_optimization_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
