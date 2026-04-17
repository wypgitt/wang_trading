"""
Probability of Backtest Overfitting (PBO) — Bailey et al. 2017, AFML Ch. 14.

Given a matrix of per-period returns for N candidate strategies (e.g. the
hyperparameter grid of a meta-labeler), Combinatorial Symmetric Cross-
Validation (CSCV) asks: *how often does picking the in-sample best strategy
produce an out-of-sample laggard?*

Procedure
---------
1. Split the time axis into S equal-size partitions (S even).
2. Enumerate every C(S, S/2) way of labelling S/2 partitions as IS and the
   rest as OOS.
3. For each split:
      a. Compute each strategy's Sharpe on the IS concatenation.
      b. Pick the IS champion (highest Sharpe).
      c. Rank the champion's OOS Sharpe among all strategies.
      d. Logit = log(rank / (N − rank)) — negative ⇒ champion ranked below
         the OOS median (i.e. IS ranking did not generalise).
4. PBO = fraction of splits with logit < 0. Gate 3 of §9 requires PBO < 0.40.

Selection is worse than a coin flip when PBO > 0.50 — the higher the number,
the more the apparent IS edge is optimization luck, not signal.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.backtesting.walk_forward import WalkForwardBacktester


def _sharpe(returns: np.ndarray) -> float:
    """Sample (non-annualised) Sharpe of a returns vector."""
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=0)
    return float(mu / sigma) if sigma > 0 else 0.0


def _rank_of(value: float, values: np.ndarray) -> float:
    """1-based rank of ``value`` within ``values`` (mid-rank on ties).

    Rank 1 = worst (lowest Sharpe), rank N = best. Ties are placed at the
    midpoint of the tied block so the logit formula stays continuous.
    """
    less = int(np.sum(values < value))
    equal = int(np.sum(values == value))
    return less + (equal + 1) / 2.0


def compute_pbo(
    strategy_returns_matrix: pd.DataFrame,
    n_partitions: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    strategy_returns_matrix : DataFrame
        Rows = time periods (bars), columns = strategy variants.
    n_partitions : int, default 10
        Must be even. Time axis is split into this many contiguous groups;
        C(n_partitions, n_partitions/2) splits are then evaluated.

    Returns
    -------
    (pbo_value, details) : tuple
        ``pbo_value`` in [0, 1]. ``details`` is a DataFrame with one row per
        split — columns ``combination_id, is_best_strategy, oos_rank, logit``.
    """
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be an even integer >= 2")
    if strategy_returns_matrix.empty:
        raise ValueError("strategy_returns_matrix is empty")
    if strategy_returns_matrix.shape[1] < 2:
        raise ValueError("need at least 2 strategies to rank")

    T, N = strategy_returns_matrix.shape
    if T < n_partitions:
        raise ValueError(
            f"need at least {n_partitions} rows, got {T}"
        )

    M = strategy_returns_matrix.to_numpy()
    columns = list(strategy_returns_matrix.columns)

    # contiguous, equal-size partitions (last absorbs remainder)
    size = T // n_partitions
    bounds: list[tuple[int, int]] = []
    for i in range(n_partitions):
        start = i * size
        end = (i + 1) * size if i < n_partitions - 1 else T
        bounds.append((start, end))

    rows: list[dict] = []
    k = n_partitions // 2
    for combo_id, is_groups in enumerate(
        combinations(range(n_partitions), k)
    ):
        is_mask = np.zeros(T, dtype=bool)
        for g in is_groups:
            is_mask[bounds[g][0]:bounds[g][1]] = True
        oos_mask = ~is_mask

        is_sharpes = np.array([_sharpe(M[is_mask, s]) for s in range(N)])
        oos_sharpes = np.array([_sharpe(M[oos_mask, s]) for s in range(N)])

        is_best_idx = int(np.argmax(is_sharpes))
        champion_oos = oos_sharpes[is_best_idx]
        rank = _rank_of(champion_oos, oos_sharpes)  # 1..N

        # Logit = log(rank / (N - rank)), clipped so rank=N doesn't blow up.
        eps = 1e-9
        denom = max(N - rank, eps)
        numer = max(rank, eps)
        logit = float(np.log(numer / denom))

        rows.append(
            {
                "combination_id": combo_id,
                "is_best_strategy": columns[is_best_idx],
                "oos_rank": float(rank),
                "logit": logit,
            }
        )

    details = pd.DataFrame(rows)
    pbo = float((details["logit"] < 0).mean())

    expected_splits = comb(n_partitions, k)
    logger.debug(
        f"PBO: {pbo:.3f} over {len(details)}/{expected_splits} splits "
        f"(N={N} strategies, T={T} periods)"
    )
    return pbo, details


def generate_strategy_variants(
    backtester: WalkForwardBacktester,
    close: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.DataFrame,
    param_grid: dict[str, list],
    n_variants: int = 20,
    run_backtest: Callable | None = None,
) -> pd.DataFrame:
    """Build a (time × variants) return matrix by sweeping hyperparameters.

    ``param_grid`` is a dict of parameter-name → list-of-values. Cartesian
    combinations are sampled (up to ``n_variants`` points). For each
    combination a backtest is run and its per-bar return series becomes one
    column of the output.

    ``run_backtest`` is an optional callable ``(backtester, close, signals,
    features, params) -> BacktestResult`` that the caller can inject for
    custom wiring (e.g. retuning the meta-labeler per variant). The default
    path rebuilds a :class:`WalkForwardBacktester` using the barrier/holding
    knobs from ``params`` and re-runs ``.run``.
    """
    from itertools import product

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    combos = list(product(*value_lists))

    if len(combos) > n_variants:
        # Evenly-spaced subsample so the grid stays representative.
        step = len(combos) / n_variants
        combos = [combos[int(i * step)] for i in range(n_variants)]

    columns: dict[str, pd.Series] = {}
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if run_backtest is not None:
            result = run_backtest(backtester, close, signals, features, params)
        else:
            bt = WalkForwardBacktester(
                cost_model=backtester.cost_model,
                initial_capital=backtester.initial_capital,
                execution_delay_bars=backtester.execution_delay_bars,
                max_positions=backtester.max_positions,
                upper_multiplier=params.get(
                    "upper_barrier_mult", backtester.upper_multiplier
                ),
                lower_multiplier=params.get(
                    "lower_barrier_mult", backtester.lower_multiplier
                ),
                max_holding_period=params.get(
                    "max_holding_period", backtester.max_holding_period
                ),
                asset_class=backtester.asset_class,
            )
            result = bt.run(close=close, signals_df=signals)
        columns[f"variant_{i}"] = result.returns

    return pd.DataFrame(columns)


def validate_pbo(pbo_value: float, max_pbo: float = 0.40) -> tuple[bool, str]:
    """Gate 3 check. Returns ``(passed, message)``."""
    if not 0.0 <= pbo_value <= 1.0:
        raise ValueError(f"pbo_value must be in [0, 1], got {pbo_value}")

    if pbo_value < max_pbo:
        return True, f"PBO {pbo_value:.3f} < {max_pbo} — strategy selection is informative."
    if pbo_value > 0.50:
        return False, (
            f"PBO {pbo_value:.3f} > 0.50 — strategy selection is worse than random; "
            "the IS champion is systematically a poor OOS performer."
        )
    return False, (
        f"PBO {pbo_value:.3f} exceeds {max_pbo} threshold — meaningful risk of overfitting."
    )
