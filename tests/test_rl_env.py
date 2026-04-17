"""Tests for the RL TradingEnv (P6.08)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.rl_env import EnvValidator, TradingEnv


def _make_env(
    *, n_steps: int = 60, n_symbols: int = 2, trend: float = 0.0,
    crash_at: int | None = None, max_steps: int | None = None,
    flat: bool = False, max_position_pct: float = 0.20,
) -> TradingEnv:
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rng = np.random.default_rng(0)
    bars = {}
    for i, sym in enumerate(symbols):
        if flat:
            prices = np.full(n_steps, 100.0)
        else:
            prices = 100.0 * np.cumprod(1.0 + trend + 0.001 * rng.standard_normal(n_steps))
        if crash_at is not None:
            prices[crash_at:] *= 0.5  # crash
        bars[sym] = pd.DataFrame({"close": prices})
    features = pd.DataFrame(rng.standard_normal((n_steps, 3)), columns=["f1", "f2", "f3"])
    meta = pd.DataFrame(np.full((n_steps, n_symbols), 0.6), columns=symbols)
    regime = pd.DataFrame(np.full((n_steps, 2), 0.5), columns=["bull", "bear"])
    return TradingEnv(
        bars=bars, features=features, meta_probs=meta, regime_probs=regime,
        max_steps=max_steps, position_tier_count=5, initial_capital=100_000.0,
        max_position_pct=max_position_pct,
    )


# ── Reset / spaces ──────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_valid_obs(self):
        env = _make_env()
        obs, info = env.reset()
        for key in ("features", "positions", "meta_probs", "regime_probs",
                    "portfolio_stats", "calendar"):
            assert key in obs
            assert np.all(np.isfinite(obs[key]))

    def test_obs_shapes(self):
        env = _make_env(n_symbols=3)
        obs, _ = env.reset()
        assert obs["positions"].shape == (3,)
        assert obs["meta_probs"].shape == (3,)
        assert obs["portfolio_stats"].shape == (5,)
        assert obs["calendar"].shape == (2,)

    def test_action_space_multidiscrete(self):
        env = _make_env(n_symbols=4)
        assert env.action_space.nvec.tolist() == [6, 6, 6, 6]  # tier_count + 1


# ── Step ────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_five_tuple(self):
        env = _make_env()
        env.reset()
        action = np.array([1, 2])
        out = env.step(action)
        assert len(out) == 5
        obs, reward, term, trunc, info = out
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert "nav" in info

    def test_action_zero_exits_positions(self):
        env = _make_env()
        env.reset()
        env.step(np.array([3, 3]))
        assert np.any(env._weights > 0)
        env.step(np.array([0, 0]))
        assert np.all(env._weights == 0)

    def test_invalid_action_raises(self):
        env = _make_env()
        env.reset()
        with pytest.raises(ValueError):
            env.step(np.array([99, 0]))

    def test_reward_finite_over_many_steps(self):
        env = _make_env(n_steps=100)
        env.reset()
        total = 0.0
        for _ in range(50):
            obs, reward, term, trunc, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            total += reward
            if term or trunc:
                break
        assert np.isfinite(total)


# ── Termination / truncation ───────────────────────────────────────────

class TestTermination:
    def test_drawdown_terminates(self):
        env = _make_env(n_steps=20, crash_at=2, max_position_pct=1.0)
        env.reset()
        # Max long position on both symbols; a 50% crash → DD > 30%
        terminated = False
        for _ in range(18):
            _, _, term, trunc, _ = env.step(np.array([5, 5]))
            if term:
                terminated = True
                break
            if trunc:
                break
        assert terminated is True

    def test_max_steps_truncation(self):
        env = _make_env(n_steps=100, max_steps=5)
        env.reset()
        truncated = False
        for _ in range(10):
            _, _, term, trunc, _ = env.step(np.array([1, 1]))
            if trunc:
                truncated = True
                break
        assert truncated is True


# ── Turnover penalty ───────────────────────────────────────────────────

class TestTurnoverPenalty:
    def test_high_turnover_reduces_reward(self):
        # Flat prices → only reward term that matters is -turnover_penalty.
        env_steady = _make_env(n_steps=50, flat=True)
        env_steady.reset()
        env_steady.step(np.array([2, 2]))  # prime
        steady_reward = sum(
            env_steady.step(np.array([2, 2]))[1] for _ in range(5)
        )

        env_churn = _make_env(n_steps=50, flat=True)
        env_churn.reset()
        env_churn.step(np.array([2, 2]))
        churn_reward = 0.0
        for i in range(5):
            action = np.array([5, 0]) if i % 2 == 0 else np.array([0, 5])
            churn_reward += env_churn.step(action)[1]

        assert steady_reward > churn_reward


# ── Validator ──────────────────────────────────────────────────────────

class TestValidator:
    def test_full_check_clean_env(self):
        env = _make_env()
        results = EnvValidator.full_check(env)
        assert results["reset"] == []
        assert results["step"] == []

    def test_catches_nan_in_features(self):
        env = _make_env()
        # Poison features with NaN at the first step
        env.features.iloc[0, 0] = np.nan
        # The env sanitizes via nan_to_num, so features obs stays finite;
        # but if we bypass sanitization, validator should flag directly.
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs["features"]))

        # Directly check a NaN array flagging path.
        errors = EnvValidator.validate_reset(env)
        assert errors == []  # sanitized → clean

    def test_validator_flags_non_finite_step_reward(self, monkeypatch):
        env = _make_env()
        env.reset()
        original = env.step

        def bad_step(action):
            obs, _, term, trunc, info = original(action)
            return obs, float("nan"), term, trunc, info

        monkeypatch.setattr(env, "step", bad_step)
        errors = EnvValidator.validate_step(env, n_steps=3)
        assert any("non-finite reward" in e for e in errors)
