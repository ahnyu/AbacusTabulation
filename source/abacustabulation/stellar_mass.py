"""Shared stellar-mass split definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StellarMassSelection:
    """Validated raw or redshift-relative stellar-mass split."""

    split_method: str
    logmstar_edges: np.ndarray
    redshift_weights: np.ndarray | None

    @classmethod
    def from_values(
        cls,
        split_method: str,
        logmstar_edges: Any,
        redshift_weights: Any = None,
    ) -> "StellarMassSelection":
        method = str(split_method).lower()
        if method not in {"raw", "relative"}:
            raise ValueError("split_method must be either 'raw' or 'relative'.")

        try:
            edges = np.asarray(logmstar_edges, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "logmstar_edges must be a rectangular numeric array."
            ) from exc

        if method == "raw":
            if edges.ndim != 1:
                raise ValueError("Raw logmstar_edges must be one-dimensional.")
            if redshift_weights is not None:
                raise ValueError("A raw split must not define redshift_weights.")
            edge_rows = edges[None, :]
            weights = None
        else:
            if edges.ndim != 2 or edges.shape[0] == 0:
                raise ValueError(
                    "Relative logmstar_edges must have shape "
                    "(n_redshift_cells, n_splits + 1)."
                )
            raw_weights = (
                np.ones(edges.shape[0], dtype=np.float64)
                if redshift_weights is None
                else np.asarray(redshift_weights, dtype=np.float64)
            )
            if raw_weights.ndim != 1 or raw_weights.size != edges.shape[0]:
                raise ValueError(
                    "redshift_weights must be one-dimensional and match "
                    "the number of relative edge rows."
                )
            if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
                raise ValueError(
                    "redshift_weights must be finite and non-negative."
                )
            scale = float(np.max(raw_weights))
            if scale <= 0.0:
                raise ValueError(
                    "redshift_weights must have positive total weight."
                )
            scaled = raw_weights / scale
            weights = scaled / np.sum(scaled)
            edge_rows = edges

        if edge_rows.shape[1] < 2:
            raise ValueError("logmstar_edges must define at least one split.")
        if np.any(np.isnan(edge_rows)) or np.any(np.isneginf(edge_rows)):
            raise ValueError(
                "logmstar_edges must not contain NaN or negative infinity."
            )
        if np.any(np.isposinf(edge_rows[:, :-1])):
            raise ValueError(
                "Positive infinity is allowed only as the final "
                "logmstar edge."
            )
        open_upper = np.isposinf(edge_rows[:, -1])
        if np.any(open_upper) and not np.all(open_upper):
            raise ValueError(
                "Relative logmstar edge rows must all use the same finite "
                "or open upper boundary."
            )
        if not np.all(np.diff(edge_rows, axis=1) > 0.0):
            raise ValueError(
                "Each row of logmstar_edges must be strictly increasing."
            )

        return cls(
            split_method=method,
            logmstar_edges=np.array(edges, dtype=np.float64, copy=True),
            redshift_weights=None
            if weights is None
            else np.array(weights, dtype=np.float64, copy=True),
        )

    @property
    def n_splits(self) -> int:
        return int(self.edge_rows.shape[1] - 1)

    @property
    def edge_rows(self) -> np.ndarray:
        if self.split_method == "raw":
            return self.logmstar_edges[None, :]
        return self.logmstar_edges

    @property
    def normalized_redshift_weights(self) -> np.ndarray:
        if self.redshift_weights is None:
            return np.ones(1, dtype=np.float64)
        return self.redshift_weights

    @property
    def has_open_upper_bound(self) -> bool:
        """Return whether the last stellar-mass interval extends to infinity."""

        return bool(np.isposinf(self.edge_rows[0, -1]))


def normalize_stellar_mass_selection(
    split_method: str,
    logmstar_edges: Any,
    redshift_weights: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D edge rows and normalized redshift weights."""

    selection = StellarMassSelection.from_values(
        split_method,
        logmstar_edges,
        redshift_weights,
    )
    return selection.edge_rows, selection.normalized_redshift_weights
