"""Tests for PortfolioPPOAgent / ShadowRLAgent (P6.09).

The PPO training test runs for a tiny number of timesteps to keep the suite
fast. Using a short-episode env keeps each `learn()` call under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml_layer.rl_agent import PortfolioPPOAgent, ShadowRLAgent
from src.ml_layer.rl_env import TradingEnv


def _make_env(n_steps: int = 40, n_symbols: int = 2) -> TradingEnv:
    rng = np.random.default_rng(0)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    bars = {
        s: pd.DataFrame({
            "close": 100.0 * np.cumprod(
                1.0 + 0.001 * rng.standard_normal(n_steps)
            ),
        })
        for s in symbols
    }
    features = pd.DataFrame(rng.standard_normal((n_steps, 3)),
                            columns=["f1", "f2", "f3"])
    meta = pd.DataFrame(np.full((n_steps, n_symbols), 0.5), columns=symbols)
    regime = pd.DataFrame(np.full((n_steps, 2), 0.5), columns=["bull", "bear"])
    return TradingEnv(
        bars=bars, features=features, meta_probs=meta, regime_probs=regime,
        position_tier_count=3, initial_capital=100_000,
    )


def _make_agent(n_steps_hparam: int = 128) -> PortfolioPPOAgent:
    env = _make_env()
    return PortfolioPPOAgent(
        env, learning_rate=3e-4, n_steps=n_steps_hparam,
        batch_size=32, n_epochs=2, seed=42, verbose=0,
    )


# ── Training / prediction ───────────────────────────────────────────────

class TestTrainPredict:
    def test_train_short(self):
        agent = _make_agent()
        agent.train(total_timesteps=256)
        # After training, model should have completed at least one rollout
        assert agent.model is not None

    def test_predict_returns_valid_action(self):
        agent = _make_agent()
        obs, _ = agent.env.reset()
        action = agent.predict(obs, deterministic=True)
        assert agent.env.action_space.contains(action.astype(np.int64))

    def test_predict_before_training_still_works(self):
        agent = _make_agent()
        obs, _ = agent.env.reset()
        action = agent.predict(obs)
        assert action.shape == (agent.env.n_symbols,)


# ── Save / load ─────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        agent = _make_agent()
        agent.train(total_timesteps=128)
        path = tmp_path / "ppo_model"
        agent.save(path)
        assert path.with_suffix(".zip").exists() or path.exists()

        env = _make_env()
        loaded = PortfolioPPOAgent.load(path, env)
        obs, _ = env.reset()
        orig = agent.predict(obs)
        roundtrip = loaded.predict(obs)
        assert roundtrip.shape == orig.shape
        # Meta file written
        assert path.with_suffix(".meta.json").exists()


# ── Evaluate ────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_evaluate_returns_expected_keys(self):
        agent = _make_agent()
        agent.train(total_timesteps=128)
        result = agent.evaluate(_make_env(), n_episodes=2)
        for key in ("mean_reward", "std_reward", "mean_final_nav",
                    "max_drawdown", "n_episodes"):
            assert key in result
        assert result["n_episodes"] == 2
        assert np.isfinite(result["mean_reward"])


# ── Shadow agent ───────────────────────────────────────────────────────

class TestShadowAgent:
    def test_shadow_target_shape_and_bounds(self):
        agent = _make_agent()
        env = agent.env
        shadow = ShadowRLAgent(agent, symbols=env.symbols, max_position_pct=0.2)
        obs, _ = env.reset()
        target = shadow.get_shadow_target_portfolio(obs)
        assert list(target.columns) == ["symbol", "target_weight", "strategy"]
        assert len(target) == env.n_symbols
        # Weights bounded by [0, max_position_pct]
        assert target["target_weight"].min() >= 0.0
        assert target["target_weight"].max() <= 0.2

    def test_log_shadow_decision_records_divergence(self, tmp_path):
        agent = _make_agent()
        env = agent.env
        log_path = tmp_path / "shadow.log"
        shadow = ShadowRLAgent(agent, symbols=env.symbols, log_path=log_path)
        rl = pd.DataFrame({
            "symbol": env.symbols, "target_weight": [0.1, 0.1],
        })
        actual = pd.DataFrame({
            "symbol": env.symbols, "target_weight": [0.0, 0.2],
        })
        decision = shadow.log_shadow_decision(rl, actual)
        assert decision.divergence == pytest.approx(0.2)
        assert len(shadow.decisions) == 1
        assert log_path.exists()
