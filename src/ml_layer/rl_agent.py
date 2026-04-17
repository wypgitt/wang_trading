"""PPO portfolio agent + training pipeline + shadow runner (P6.09).

Wraps ``stable_baselines3.PPO`` with the project-specific defaults, a
training pipeline that can tune hyperparameters via Optuna, and a shadow
agent that runs alongside the production optimizer to log what the RL
policy would have done.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Important on macOS: prevent libomp duplicate-init crashes when torch loads
# alongside numpy/scipy (conftest handles this for tests).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

# Cap torch's intra-op parallelism — mirrors regime_detector's workaround
# for the intermittent libomp segfault on macOS.
try:
    torch.set_num_threads(1)
except RuntimeError:  # pragma: no cover
    pass

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402

from src.ml_layer.rl_env import TradingEnv  # noqa: E402

log = logging.getLogger(__name__)


# ── PPO wrapper ───────────────────────────────────────────────────────────

class PortfolioPPOAgent:
    """Portfolio-optimizer PPO agent with project defaults."""

    def __init__(
        self,
        env: TradingEnv,
        *,
        policy: str = "MultiInputPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: dict | None = None,
        seed: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.hparams: dict[str, Any] = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": policy_kwargs,
        }
        self.model: PPO = PPO(
            policy, env,
            **self.hparams,
            verbose=verbose, seed=seed,
        )

    # ── Training / inference ──────────────────────────────────────────

    def train(
        self, total_timesteps: int, *,
        callback: Any | None = None, progress_bar: bool = False,
    ) -> "PortfolioPPOAgent":
        self.model.learn(
            total_timesteps=int(total_timesteps),
            callback=callback,
            progress_bar=progress_bar,
        )
        return self

    def predict(
        self, observation: Any, *, deterministic: bool = True,
    ) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))
        meta = {
            "policy": self.policy,
            "hparams": {k: v for k, v in self.hparams.items() if k != "policy_kwargs"},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path, env: TradingEnv) -> "PortfolioPPOAgent":
        p = Path(path)
        instance = cls.__new__(cls)
        instance.env = env
        instance.policy = "MultiInputPolicy"
        instance.hparams = {}
        instance.model = PPO.load(str(p), env=env)
        return instance

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(
        self, eval_env: TradingEnv, *, n_episodes: int = 5,
        deterministic: bool = True,
    ) -> dict[str, float]:
        mean_r, std_r = evaluate_policy(
            self.model, eval_env, n_eval_episodes=int(n_episodes),
            deterministic=deterministic, return_episode_rewards=False,
        )
        # Collect extra per-episode stats.
        episode_navs: list[float] = []
        episode_dd: list[float] = []
        for _ in range(int(n_episodes)):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, _, term, trunc, info = eval_env.step(action)
                done = bool(term or trunc)
            episode_navs.append(float(info.get("nav", 0.0)))
            episode_dd.append(float(info.get("drawdown", 0.0)))
        return {
            "mean_reward": float(mean_r),
            "std_reward": float(std_r),
            "mean_final_nav": float(np.mean(episode_navs)) if episode_navs else 0.0,
            "max_drawdown": float(np.max(episode_dd)) if episode_dd else 0.0,
            "n_episodes": int(n_episodes),
        }


# ── Training pipeline ────────────────────────────────────────────────────

@dataclass
class _TrainingSpec:
    total_timesteps: int = 50_000
    eval_episodes: int = 3
    seed: int | None = None


class RLTrainingPipeline:
    """Orchestrates train / eval / hyperparameter search for the PPO agent."""

    def __init__(
        self,
        *,
        env_factory: Callable[..., TradingEnv],
        eval_env_factory: Callable[..., TradingEnv] | None = None,
        spec: _TrainingSpec | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.eval_env_factory = eval_env_factory or env_factory
        self.spec = spec or _TrainingSpec()

    def train_agent(
        self,
        *,
        symbols: list[str] | None = None,
        train_start: Any = None, train_end: Any = None,
        eval_start: Any = None, eval_end: Any = None,
        hparams: dict | None = None,
    ) -> PortfolioPPOAgent:
        env = self.env_factory(
            symbols=symbols, start=train_start, end=train_end,
        ) if symbols is not None else self.env_factory()
        agent = PortfolioPPOAgent(env, seed=self.spec.seed, **(hparams or {}))
        agent.train(self.spec.total_timesteps)
        return agent

    def tune_hyperparameters(
        self, *, n_trials: int = 20, train_steps: int = 5_000,
    ) -> dict[str, Any]:
        """Coarse Optuna search over (learning_rate, clip_range, ent_coef,
        n_steps). Returns the best-params dict plus per-trial scores."""
        try:
            import optuna  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("optuna required for tune_hyperparameters") from exc

        def _objective(trial: "optuna.Trial") -> float:
            hparams = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
                "batch_size": 64,
            }
            env = self.env_factory()
            agent = PortfolioPPOAgent(env, **hparams)
            agent.train(total_timesteps=train_steps)
            eval_env = self.eval_env_factory()
            result = agent.evaluate(eval_env, n_episodes=self.spec.eval_episodes)
            return result["mean_reward"]

        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        return {
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
        }


# ── Shadow agent ─────────────────────────────────────────────────────────

@dataclass
class ShadowDecision:
    timestamp: datetime
    rl_target: dict[str, float]
    actual_target: dict[str, float]
    divergence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ShadowRLAgent:
    """Runs the PPO agent in read-only mode next to production, logging
    what it *would* have traded without actually executing."""

    def __init__(
        self,
        agent: PortfolioPPOAgent,
        *,
        symbols: list[str] | None = None,
        max_position_pct: float = 0.20,
        log_path: Path | str | None = None,
    ) -> None:
        self.agent = agent
        self.symbols = symbols or list(agent.env.symbols)
        self.max_position_pct = float(max_position_pct)
        self.tier_count = int(agent.env.config.position_tier_count)
        self.log_path = Path(log_path) if log_path else None
        self.decisions: list[ShadowDecision] = []

    def get_shadow_target_portfolio(self, context: dict[str, Any]) -> pd.DataFrame:
        """Compute the RL policy's target weights for the given observation.

        ``context`` should be a dict matching the environment's observation
        space (typically captured straight from a live env.reset/step).
        """
        action = self.agent.predict(context, deterministic=True)
        action = np.asarray(action, dtype=int).flatten()
        weights = (action.astype(float) / float(self.tier_count)) * self.max_position_pct
        n = min(len(self.symbols), len(weights))
        return pd.DataFrame({
            "symbol": self.symbols[:n],
            "target_weight": weights[:n],
            "strategy": "rl_shadow",
        })

    def log_shadow_decision(
        self,
        rl_target: pd.DataFrame,
        actual_target: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> ShadowDecision:
        rl_map = dict(zip(rl_target["symbol"], rl_target["target_weight"]))
        act_map = dict(zip(actual_target["symbol"], actual_target["target_weight"]))
        symbols = set(rl_map) | set(act_map)
        divergence = sum(
            abs(rl_map.get(s, 0.0) - act_map.get(s, 0.0)) for s in symbols
        )
        decision = ShadowDecision(
            timestamp=datetime.now(timezone.utc),
            rl_target=rl_map, actual_target=act_map,
            divergence=float(divergence),
            metadata=metadata or {},
        )
        self.decisions.append(decision)
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as fh:
                    fh.write(json.dumps({
                        "ts": decision.timestamp.isoformat(),
                        "rl_target": rl_map,
                        "actual_target": act_map,
                        "divergence": decision.divergence,
                        "metadata": decision.metadata,
                    }) + "\n")
            except Exception:  # pragma: no cover
                log.exception("shadow log write failed")
        return decision
