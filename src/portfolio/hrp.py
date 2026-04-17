"""
Hierarchical Risk Parity (HRP) — López de Prado 2016, AFML Ch. 20, design-doc §8.4.1.

Markowitz inverts the covariance matrix, which is numerically fragile on
strongly-correlated assets (the covariance matrix is near-singular) and
produces wildly-concentrated weights. HRP sidesteps the inversion by
building a hierarchical clustering over the correlation matrix, reordering
assets so correlated names are adjacent, then recursively bisecting the
sorted list and allocating risk inversely to each half's cluster variance.

Three-step pipeline
-------------------
1. **Tree clustering**  — distance d(i,j) = sqrt(0.5 · (1 − corr(i,j)))
   feeds into ``scipy.cluster.hierarchy.linkage``.
2. **Quasi-diagonalisation** — walk the linkage tree in post-order so that
   leaves belonging to the same sub-cluster end up adjacent (seriation).
3. **Recursive bisection** — split the sorted list in half and allocate
   weight α to the first half, (1−α) to the second, with
   α = 1 − σ²(left) / (σ²(left) + σ²(right)). σ²(cluster) is the variance
   of the inverse-variance portfolio *within* that cluster. Recurse until
   every cluster is a single asset.

Output weights are non-negative (HRP is long-only by construction) and sum
to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ── pipeline pieces ────────────────────────────────────────────────────


def correlation_to_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Angular distance in [0, 1]; 0 = perfect positive correlation."""

    if not isinstance(corr_matrix, pd.DataFrame):
        raise TypeError("corr_matrix must be a pd.DataFrame")
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("corr_matrix must be square")
    # clip to deal with tiny floating-point overshoots above |1|
    c = corr_matrix.clip(lower=-1.0, upper=1.0)
    d = np.sqrt(0.5 * (1.0 - c))
    return pd.DataFrame(d, index=corr_matrix.index, columns=corr_matrix.columns)


def quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Post-order traversal of the linkage tree → asset ordering."""

    link_i = link.astype(int)
    num_items = int(link_i[-1, 3])  # total leaves

    order: list[int] = [int(link_i[-1, 0]), int(link_i[-1, 1])]
    while max(order) >= num_items:
        expanded: list[int] = []
        for idx in order:
            if idx < num_items:
                expanded.append(idx)
            else:
                row = idx - num_items
                expanded.extend(
                    [int(link_i[row, 0]), int(link_i[row, 1])]
                )
        order = expanded
    return order


def _cluster_variance(cov: pd.DataFrame, items: list) -> float:
    """Variance of the inverse-variance portfolio over ``items``."""
    sub = cov.loc[items, items].to_numpy()
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def get_recursive_bisection_weights(
    cov_matrix: pd.DataFrame,
    sorted_indices: list[int] | list[str],
) -> pd.Series:
    """Recursive bisection over seriated asset indices."""

    # Accept either integer positions or column labels; normalise to labels.
    if len(sorted_indices) == 0:
        return pd.Series(dtype=float)
    if isinstance(sorted_indices[0], (int, np.integer)):
        labels = [cov_matrix.columns[i] for i in sorted_indices]
    else:
        labels = list(sorted_indices)

    weights = pd.Series(1.0, index=labels, dtype=float)
    clusters: list[list] = [labels]

    while clusters:
        new_clusters: list[list] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left = cluster[:half]
            right = cluster[half:]
            v_left = _cluster_variance(cov_matrix, left)
            v_right = _cluster_variance(cov_matrix, right)
            total = v_left + v_right
            alpha = 1.0 - v_left / total if total > 0 else 0.5
            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters

    return weights


def compute_hrp_weights(
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """End-to-end HRP from a returns panel → portfolio weights."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pd.DataFrame")
    if returns.shape[1] < 2:
        raise ValueError("HRP needs at least 2 assets")
    if len(returns) < 2:
        raise ValueError("HRP needs at least 2 observations")

    cov = returns.cov()
    corr = returns.corr()
    dist = correlation_to_distance(corr)

    # scipy.linkage expects condensed distances; squareform extracts them.
    dist_condensed = squareform(dist.to_numpy(), checks=False)
    link = linkage(dist_condensed, method=linkage_method)

    order_pos = quasi_diagonalize(link)
    sorted_labels = [corr.columns[i] for i in order_pos]
    weights = get_recursive_bisection_weights(cov, sorted_labels)
    # Re-index to original column order for a deterministic caller surface.
    return weights.reindex(returns.columns)


# ── stateful optimiser ─────────────────────────────────────────────────


@dataclass
class HRPPortfolioOptimizer:
    """Streaming HRP — append returns, rebalance on a fixed cadence."""

    rebalance_frequency: int = 5
    lookback: int = 252
    linkage_method: str = "single"

    _history: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(dtype=float), init=False
    )
    _cached_weights: pd.Series | None = field(default=None, init=False)
    _updates_since_rebalance: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be >= 1")
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")

    def update(self, returns_row: pd.Series) -> None:
        """Append a single cross-section of returns."""
        row = returns_row.to_frame().T
        if self._history.empty:
            self._history = row.copy()
        else:
            self._history = pd.concat([self._history, row], axis=0)
            if len(self._history) > self.lookback:
                self._history = self._history.iloc[-self.lookback :]
        self._updates_since_rebalance += 1

    def get_weights(self) -> pd.Series:
        """Return the current HRP weights; recompute only at cadence."""
        need_rebalance = (
            self._cached_weights is None
            or self._updates_since_rebalance >= self.rebalance_frequency
        )
        if not need_rebalance:
            return self._cached_weights

        if len(self._history) < 2:
            # degenerate: equal-weight fallback
            cols = self._history.columns
            self._cached_weights = pd.Series(1.0 / len(cols), index=cols)
        else:
            self._cached_weights = compute_hrp_weights(
                self._history, linkage_method=self.linkage_method
            )
        self._updates_since_rebalance = 0
        return self._cached_weights

    def get_target_positions(
        self,
        weights: pd.Series,
        portfolio_nav: float,
        prices: pd.Series,
    ) -> pd.Series:
        """Translate weights into share/unit counts given prices and NAV."""
        if portfolio_nav <= 0:
            raise ValueError("portfolio_nav must be positive")
        aligned_prices = prices.reindex(weights.index)
        if (aligned_prices <= 0).any() or aligned_prices.isna().any():
            raise ValueError("prices must be strictly positive for all assets")
        dollar_alloc = weights * portfolio_nav
        return dollar_alloc / aligned_prices
