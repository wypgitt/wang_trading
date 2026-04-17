"""
PCA-based factor risk model (Isichenko; design-doc §8.4.2).

Given a history of asset returns, fit a latent-factor model:

    r_i(t) ≈ Σ_k B_{i,k} · f_k(t) + ε_i(t)

where the factor returns ``f_k`` and loadings ``B`` are the top principal
components of the asset covariance. PCA is used because the *named* factors
(market, size, value, momentum, vol) are usually collinear and unstable, and
because the solo-operator use case doesn't need a pre-specified factor
universe — we just want a risk decomposition good enough to spot unwanted
tilts ("accidentally long small-cap momentum via two separate strategies").

Two operator-level products sit on top of the model:

* :meth:`get_risk_decomposition` — split portfolio variance into systematic
  (factor-driven) vs idiosyncratic (asset-specific) buckets, and attribute
  the systematic portion across factors.
* :meth:`neutralize_factors` — project a weight vector onto the null space
  of selected factor exposures, producing the closest portfolio (in L₂) with
  zero exposure to those factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorRiskModel:
    n_factors: int = 5
    lookback: int = 252

    factor_returns: pd.DataFrame = field(default=None, init=False)
    factor_loadings: pd.DataFrame = field(default=None, init=False)
    factor_covariance: pd.DataFrame = field(default=None, init=False)
    idiosyncratic_variance: pd.Series = field(default=None, init=False)
    explained_variance_ratio: pd.Series = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.n_factors < 1:
            raise ValueError("n_factors must be >= 1")
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self, returns: pd.DataFrame) -> "FactorRiskModel":
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pd.DataFrame")
        if returns.shape[1] < self.n_factors:
            raise ValueError(
                f"need at least {self.n_factors} assets; got {returns.shape[1]}"
            )

        window = returns.tail(self.lookback)
        centred = window - window.mean()
        cov = centred.cov()

        eigvals, eigvecs = np.linalg.eigh(cov.to_numpy())
        # eigh returns ascending → reverse to descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        k = self.n_factors
        top_vals = eigvals[:k]
        top_vecs = eigvecs[:, :k]

        factor_names = [f"Factor_{i + 1}" for i in range(k)]
        self.factor_loadings = pd.DataFrame(
            top_vecs, index=returns.columns, columns=factor_names
        )
        self.factor_returns = pd.DataFrame(
            centred.to_numpy() @ top_vecs,
            index=window.index,
            columns=factor_names,
        )
        self.factor_covariance = pd.DataFrame(
            np.diag(top_vals), index=factor_names, columns=factor_names
        )

        systematic_cov = top_vecs @ np.diag(top_vals) @ top_vecs.T
        idio = np.maximum(np.diag(cov.to_numpy()) - np.diag(systematic_cov), 0.0)
        self.idiosyncratic_variance = pd.Series(
            idio, index=returns.columns, name="idio_var"
        )

        total_var = float(eigvals.sum())
        self.explained_variance_ratio = pd.Series(
            top_vals / total_var if total_var > 0 else np.zeros(k),
            index=factor_names,
        )
        return self

    # ── exposures & decomposition ──────────────────────────────────────

    def _require_fitted(self) -> None:
        if self.factor_loadings is None:
            raise RuntimeError("call fit() first")

    def get_factor_exposures(self, weights: pd.Series) -> pd.Series:
        self._require_fitted()
        w = weights.reindex(self.factor_loadings.index).fillna(0.0)
        return self.factor_loadings.T @ w

    def get_risk_decomposition(self, weights: pd.Series) -> dict:
        self._require_fitted()
        w = weights.reindex(self.factor_loadings.index).fillna(0.0)
        exposures = self.factor_loadings.T @ w  # B.T w
        fcov = self.factor_covariance.to_numpy()

        systematic = float(exposures.to_numpy() @ fcov @ exposures.to_numpy())
        idio = float((w.to_numpy() ** 2 * self.idiosyncratic_variance.to_numpy()).sum())
        total = systematic + idio

        # Per-factor contribution: e_k · (Σ_f e)_k — additive, sums to systematic
        fcov_e = fcov @ exposures.to_numpy()
        factor_contributions = pd.Series(
            exposures.to_numpy() * fcov_e,
            index=exposures.index,
            name="factor_contribution",
        )

        return {
            "total_risk": total,
            "systematic_risk": systematic,
            "idiosyncratic_risk": idio,
            "factor_contributions": factor_contributions,
            "pct_systematic": systematic / total if total > 0 else 0.0,
        }

    # ── constrained reweighting ────────────────────────────────────────

    def neutralize_factors(
        self,
        weights: pd.Series,
        factors_to_neutralize: list[int] | None = None,
    ) -> pd.Series:
        """Project ``weights`` to the null-space of the selected factor loadings.

        Minimises ‖w − w₀‖² subject to ``A w = 0`` where ``A`` stacks the
        transposed loadings for each factor in ``factors_to_neutralize`` (zero-
        indexed). Closed-form solution:

            w = w₀ − Aᵀ (A Aᵀ)⁻¹ A w₀
        """
        self._require_fitted()
        if factors_to_neutralize is None:
            factors_to_neutralize = [0]

        w0 = weights.reindex(self.factor_loadings.index).fillna(0.0).to_numpy()
        B = self.factor_loadings.to_numpy()  # N × K
        A = B[:, factors_to_neutralize].T  # len(factors) × N

        gram = A @ A.T
        # Regularise to keep well-conditioned when factors are collinear.
        gram += 1e-12 * np.eye(gram.shape[0])
        lam = np.linalg.solve(gram, A @ w0)
        w_new = w0 - A.T @ lam

        return pd.Series(w_new, index=self.factor_loadings.index, name=weights.name)


# ── unintended-tilt detector ───────────────────────────────────────────


def detect_unintended_tilts(
    weights: pd.Series,
    factor_model: FactorRiskModel,
    threshold_std: float = 2.0,
) -> list[dict]:
    """Flag factor exposures that exceed ``threshold_std`` σ from zero.

    The reference σ for each factor is the cross-sectional std of its
    loadings — i.e. how much individual assets disagree on that factor.
    Concentrated portfolios pile into specific loadings, so their
    normalised exposure jumps far above a diversified baseline.
    """
    factor_model._require_fitted()  # pylint: disable=protected-access
    exposures = factor_model.get_factor_exposures(weights)

    warnings: list[dict] = []
    for factor_name in exposures.index:
        exposure = float(exposures[factor_name])
        loadings = factor_model.factor_loadings[factor_name]
        sigma = float(loadings.std(ddof=0))
        if sigma == 0:
            continue
        z = exposure / sigma
        if abs(z) > threshold_std:
            warnings.append(
                {
                    "factor": factor_name,
                    "exposure": exposure,
                    "threshold": threshold_std * sigma,
                    "z_score": z,
                    "warning": (
                        f"{factor_name} exposure {exposure:+.3f} exceeds "
                        f"{threshold_std:.1f}σ ({threshold_std * sigma:+.3f})"
                    ),
                }
            )
    return warnings
