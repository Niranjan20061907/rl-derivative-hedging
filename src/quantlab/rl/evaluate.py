"""PPO and baseline evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quantlab.backtesting.engine import BacktestEngine
from quantlab.environments.hedging_env import HedgingEnv
from quantlab.metrics.hedging import hedging_metrics
from quantlab.rl.config import ExperimentConfig, load_config
from quantlab.strategies.base import Strategy
from quantlab.strategies.delta_hedging import DeltaHedgingStrategy
from quantlab.strategies.ppo_hedging import PPOHedgingStrategy


class NoHedgeStrategy(Strategy):
    name = "no_hedge"

    def action(self, observation: np.ndarray, info: dict) -> float:
        del observation, info
        return 0.0


def run_env_episode(env: HedgingEnv, strategy: Strategy, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, info = env.reset(seed=seed)
    strategy.reset()
    errors = [0.0]
    pnl = [0.0]
    rewards = []
    done = False
    while not done:
        action = np.array([strategy.action(obs, info)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        errors.append(errors[-1] + info["pnl"])
        pnl.append(info["pnl"])
        rewards.append(reward)
        done = terminated or truncated
    return np.asarray(errors), np.asarray(pnl), np.asarray(rewards)


def evaluate(config: ExperimentConfig, model_path: str | Path) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))
    engine = BacktestEngine(
        K=config.environment.K,
        r=config.environment.r,
        sigma=config.environment.sigma,
        T=config.environment.T,
        cost_rate=config.environment.cost_rate,
    )
    aggregate: dict[str, list[dict[str, float]]] = {"ppo": [], "delta": []}
    if config.evaluation.include_no_hedge:
        aggregate["no_hedge"] = []

    for episode in range(config.evaluation.episodes):
        seed = config.seed + 10_000 + episode
        env = HedgingEnv(config.environment)
        ppo_strategy = PPOHedgingStrategy(model, deterministic=config.evaluation.deterministic)
        ppo_error, ppo_pnl, _ = run_env_episode(env, ppo_strategy, seed)
        aggregate["ppo"].append(hedging_metrics(ppo_error, ppo_pnl))

        prices = env.prices.copy()
        delta_strategy = DeltaHedgingStrategy(
            K=config.environment.K,
            r=config.environment.r,
            sigma=config.environment.sigma,
            T=config.environment.T,
            steps=config.environment.steps,
        )
        delta_result = engine.run(prices, delta_strategy)
        aggregate["delta"].append(hedging_metrics(delta_result.hedging_error, delta_result.portfolio_pnl))

        if config.evaluation.include_no_hedge:
            no_hedge_result = engine.run(prices, NoHedgeStrategy())
            aggregate["no_hedge"].append(hedging_metrics(no_hedge_result.hedging_error, no_hedge_result.portfolio_pnl))

    summary = {
        name: {
            key: float(np.mean([episode_metrics[key] for episode_metrics in values]))
            for key in values[0]
        }
        for name, values in aggregate.items()
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config.results_dir) / f"eval_{timestamp}_seed{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": str(model_path),
        "episodes": config.evaluation.episodes,
        "seed_start": config.seed + 10_000,
        "metrics": summary,
        "per_episode": aggregate,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload | {"output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO hedging policy.")
    parser.add_argument("--config", default="configs/rl.yaml", help="Path to YAML experiment config.")
    parser.add_argument("--model", required=True, help="Path to a saved PPO checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(load_config(args.config), args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
