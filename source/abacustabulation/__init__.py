"""Utilities for AbacusSummit HOD tabulation preparation."""

_EXPORTS = {
    "apply_rsd": (".rsd", "apply_rsd"),
    "available_hod_models": (".hod", "available_hod_models"),
    "build_mass_tabulation": (".paircounts", "build_mass_tabulation"),
    "compute_paircounts_from_prepared": (".paircounts", "compute_paircounts_from_prepared"),
    "evaluate_hod": (".hod", "evaluate_hod"),
    "HODClusteringTabulator": (".clustering", "HODClusteringTabulator"),
    "galaxy_correlation_from_config": (".clustering", "galaxy_correlation_from_config"),
    "galaxy_correlation_from_paircounts": (".clustering", "galaxy_correlation_from_paircounts"),
    "galaxy_cross_correlation_from_paircounts": (".clustering", "galaxy_cross_correlation_from_paircounts"),
    "hod_model_parameters": (".hod", "hod_model_parameters"),
    "hod_weights_for_paircounts": (".clustering", "hod_weights_for_paircounts"),
    "load_prepared_catalog": (".paircounts", "load_prepared_catalog"),
    "load_prepared_binned_catalog": (".paircounts", "load_prepared_binned_catalog"),
    "nfw_enclosed_fraction": (".nfw", "nfw_enclosed_fraction"),
    "paircounts_from_config": (".paircounts", "paircounts_from_config"),
    "prepare_all_slabs": (".prepare", "prepare_all_slabs"),
    "prepare_from_config": (".prepare", "prepare_from_config"),
    "prepare_slab": (".prepare", "prepare_slab"),
    "projected_wp": (".clustering", "projected_wp"),
    "read_paircounts": (".clustering", "read_paircounts"),
    "random_geometry_factor": (".clustering", "random_geometry_factor"),
    "sample_nfw_offsets": (".nfw", "sample_nfw_offsets"),
    "sample_nfw_radii": (".nfw", "sample_nfw_radii"),
    "smu_multipoles": (".clustering", "smu_multipoles"),
    "velzspace_to_kms": (".rsd", "velzspace_to_kms"),
    "wrap_positions": (".rsd", "wrap_positions"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
