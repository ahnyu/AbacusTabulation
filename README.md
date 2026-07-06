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
```

Use `fit.mcmc.pocomc.use_mpi: true` for MPI runs, or `fit.mcmc.pocomc.n_processes > 1` for local multiprocessing on one node.

For a fixed HOD parameter, remove it from `fit.parameters` and put it under `fit.theory.tracers.<TRACER>.fixed_params`, for example:

```yaml
fit:
  theory:
    tracers:
      LRG:
        fixed_params:
          a_c: 1.0
```

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

Linear bias requires the `linear_bias` paircount job and `cosmoprimo` in the runtime environment.

## 7. Data and Covariance Notes

- `fit.observables` order defines the theory/data vector order.
- Supported statistics are `wp`, `xi0`, and `xi2`.
- Observable `slice` is applied when data are read.
- `covariance.mode: joint` expects the covariance shape to match the sliced combined data vector.
- `covariance.mode: block` accepts either sliced block covariances or full pre-slice block covariances.
- `covariance.precision_scale` multiplies the inverse covariance; pass a Hartlap factor there, or set `covariance_n_mocks` to let the code compute it.

## 8. Quick Checks

```bash
python3 -m compileall source/abacustabulation scripts
python3 scripts/prepare_profiles.py --help
python3 scripts/compute_paircounts.py --help
python3 scripts/compute_hmf.py --help
python3 scripts/run_fit_optimize.py --help
```
