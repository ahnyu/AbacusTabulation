# AbacusTabulation

AbacusTabulation prepares compact AbacusSummit halo/NFW catalogs, tabulates mass-binned pair counts, and evaluates HOD clustering without rebuilding catalogs for every HOD parameter set.

## 1. Make a Config

Use the base config as documentation and keep run-specific changes in an override file.

```bash
cp configs/lrg_fit_override.yaml configs/my_run.yaml
```

Edit `configs/my_run.yaml` and set at least:

- `sim_params.sim_dir`, `sim_params.sim_name`, `sim_params.z_mock`
- `prepare_profiles.out_dir`
- `paircounts.prepared_dir`, `paircounts.output_dir`
- `paircounts.file_tag` and `hmf.file_tag` when more than one prepared tag shares a directory
- `hmf.prepared_dir`, `hmf.output_dir`
- `fit.data`, `fit.covariance`, `fit.output`, and `fit.parameters` for fitting runs

Plain output roots expand to `<root>/{sim_name}/{z}`. For example, `prepare_profiles.out_dir: /data/prepared` writes into `/data/prepared/AbacusSummit_base_c000_ph000/z0.800`.

## 2. Prepare Halo and NFW Profiles

```bash
python3 scripts/prepare_profiles.py --path2config configs/my_run.yaml
```

Useful options:

```bash
python3 scripts/prepare_profiles.py --path2config configs/my_run.yaml --slabs 0,1,2 --overwrite
python3 scripts/prepare_profiles.py --path2config configs/my_run.yaml --position-space both --los-axis z
```

`pos` is always real space. When RSD output is requested, RSD coordinates are written to `pos_rsd`.

## 3. Compute Paircount Tables

```bash
python3 scripts/compute_paircounts.py --path2config configs/my_run.yaml
```

With the default config, this computes both:

- `paircounts.jobs.clustering`: the rppi/smu tables for HOD clustering
- `paircounts.jobs.linear_bias`: the real-space smu table for linear-bias postprocessing

Paircount files must store `mass/num_halo_subbin` and `mass/halo_subbin_centers_log10` for halo-count-weighted HOD refinement. Downstream clustering, fitting, and derived tabulators infer the subbin count from the paircount file; older paircount files without this metadata need to be recomputed.

Set `paircounts.jobs.clustering.position_dataset: pos_rsd` if the fitted clustering should use RSD positions. Keep `paircounts.jobs.linear_bias.position_dataset: pos`.

Avoid global CLI overrides like `--position-dataset pos_rsd` when computing both default jobs, because the linear-bias job must remain real space.

### Evaluate Stellar-Mass Splits

```python
import numpy as np

from abacustabulation import HODClusteringTabulator, projected_wp

tabulator = HODClusteringTabulator.from_paircount_file(paircount_path)
result = tabulator.stellar_mass_correlations(
    hod_params,
    split_method="raw",
    logmstar_edges=[10.4, 11.05, 11.30, 11.55, np.inf],
)
wp_by_split = [projected_wp(item) for item in result.clusterings]
```

Use a terminal `np.inf` edge, or `.inf` in YAML, when the last sample contains
all galaxies above its lower boundary. Finite upper edges remain supported.
For `split_method="relative"`, pass a two-dimensional edge array with one row
per redshift cell; every row must use the same finite or open upper boundary.
`redshift_weights=None` gives all cells equal weight; explicit non-negative
weights are normalized internally. The model evaluates all splits together and
returns them in edge order.

The same calculation can be loaded directly from the universal config:

```python
from abacustabulation import stellar_mass_correlations_from_config

result = stellar_mass_correlations_from_config(
    "configs/lrg_stellar_mass_override.yaml"
)
```

The separate `configs/lrg_stellar_mass_override.yaml` inherits the same base config as the scalar LRG override. It stores the split definition once under `hod.stellar_mass`; HMF and linear-bias derived quantities reuse it:

```python
from abacustabulation import HODDerivedTabulator, load_config

config_path = "configs/lrg_stellar_mass_override.yaml"
config = load_config(config_path)
derived = HODDerivedTabulator.from_config(config, tracer="LRG")
result = derived.evaluate_stellar_mass(config["hod"]["params"])

print(result.values["number_density"])
print(result.values["satellite_fraction"])
print(result.values["linear_bias"])
```

MCMC postprocessing uses the configured sample names in columns such as `LRG.sm0.number_density` and `LRG.sm0.linear_bias`. The dedicated override contains a simultaneous four-split `wp` fit example. `fit.theory.tracers.LRG.stellar_mass.samples` maps sample names to edge-interval indices, while each observable and data block uses the corresponding `sample` name.

## 4. Compute the High-Resolution HMF

Run this before using HMF number-density constraints or derived HOD quantities.

```bash
python3 scripts/compute_hmf.py --path2config configs/my_run.yaml
```

## 5. Fit HOD Parameters

Optimization:

```bash
python3 scripts/run_fit_optimize.py --path2config configs/my_run.yaml
```

pocoMC:

```bash
python3 scripts/run_fit_pocomc.py --path2config configs/my_run.yaml
python3 scripts/run_fit_pocomc.py --path2config configs/my_run.yaml --n-processes 8
python3 scripts/run_fit_pocomc_stellar_mass.py --path2config configs/lrg_stellar_mass_override.yaml --n-processes 8
```

Use `fit.mcmc.pocomc.use_mpi: true` for MPI runs, or `fit.mcmc.pocomc.n_processes > 1` for local multiprocessing on one node.
Keep `fit.mcmc.pocomc.native_threads_per_process: 1` for multiprocessing runs to prevent BLAS/OpenMP oversubscription.
The optimization runner uses the analogous `fit.optimization.native_threads_per_process` setting.

For a fixed HOD parameter, remove it from `fit.parameters` and put it under `fit.theory.tracers.<TRACER>.fixed_params`, for example:

```yaml
fit:
  theory:
    tracers:
      LRG:
        fixed_params:
          a_c: 1.0
```

`hod.model` is the default model for clustering, fitting, and derived quantities. A multi-tracer fit may override it once per tracer with `fit.theory.tracers.<TRACER>.hod_model`.

## 6. Derived Quantities and Linear Bias

Derived output is controlled by:

```yaml
fit:
  postprocess:
    enabled: true
    optimization: true
    mcmc: true
    fail_on_error: false
```

If enabled, optimization writes `*_optimum_derived.json`; MCMC writes `*_chains_derived.txt` and `*_derived_summary.json`. MCMC postprocessing evaluates all posterior samples.

When the pocoMC runner uses local multiprocessing, derived postprocessing reuses the same worker count. Set `fit.postprocess.n_processes` only when a different count is needed.

Linear bias requires the `linear_bias` paircount job and `cosmoprimo` in the runtime environment.

## 7. Plot Fit Outputs

```python
from abacustabulation import load_fit_plot_data, plot_derived_parameters, plot_fit_bands, plot_hod_bands, plot_joint_grid, plot_stellar_mass_hod_bands, plot_triangle

fit = load_fit_plot_data("configs/my_run.yaml", label="LRG")

plot_triangle(fit, params=["logMcut", "sigma", "logM1", "alpha", "LRG.number_density"])
plot_fit_bands(fit, max_samples=1000)  # pass labels=[...] for overlaid runs
plot_hod_bands(fit, max_samples=1000)
# plot_stellar_mass_hod_bands(stellar_fit, max_samples=1000)
plot_derived_parameters([fit], x_values=[0.8], x_label="redshift", derived_params=["LRG.linear_bias", "LRG.satellite_fraction", "LRG.log10_mh_cen_med", "LRG.log10_mh_sat_med"])
# plot_joint_grid(plot_fit_bands, [[fit_a, fit_b], [fit_c, fit_d]], nrows=1, ncols=2)

print(fit.parameter_stats["LRG.number_density"].mean)
```

`load_fit_plot_data` prefers `*_chains_derived.txt` when present. It keeps the original chain in `fit.chain`, all sampled plus derived columns in `fit.samples`, the GetDist object in `fit.getdist_sample`, data points in `fit.data_points`, and GetDist-style summaries in `fit.parameter_stats`. Fitting plots show `r_p w_p` by default for `wp` and include a lower `(model-data)/sigma` residual panel; pass `plot_rp_wp=False` to show raw `w_p` and `titles=True` to show panel titles. For HOD overlays, pass `show_band=[True, False, ...]` to control which fits draw bands. Derived plot parameter names must exist exactly in every fit. Pass `prefer_derived=False` to read the raw `*_chains.txt` file.

## 8. Data and Covariance Notes

- `fit.observables` order defines the theory/data vector order.
- Supported statistics are `wp`, `xi0`, and `xi2`.
- Observable `slice` is applied to both the loaded data and tabulated theory bins.
- `covariance.mode: joint` expects the covariance shape to match the sliced combined data vector.
- `covariance.mode: block` accepts either sliced block covariances or full pre-slice block covariances.
- `covariance.precision_scale` multiplies the inverse covariance; pass a Hartlap factor there, or set `covariance_n_mocks` to let the code compute it.
- Number-density mode `gaussian` penalizes only models below the target; `gaussian_two_sided` penalizes deviations in either direction.
- `fit.parameter_constraints` can impose hard ordering with `{lower: parameter_a, upper: parameter_b}`.

## 9. Quick Checks

```bash
python3 -m compileall source/abacustabulation scripts
python3 -m unittest discover -s tests -v
python3 tests/run_actual_data_smoke.py
python3 scripts/prepare_profiles.py --help
python3 scripts/compute_paircounts.py --help
python3 scripts/compute_hmf.py --help
python3 scripts/run_fit_optimize.py --help
```

The actual-data smoke test reads `AbacusTabulation_data/` and writes
clustering, HOD, and linear-bias diagnostics to `tests/plots/`. Use
`--redshift 0.725` to select another prepared snapshot or
`--skip-linear-bias` to avoid the cosmoprimo/CAMB calculation.
