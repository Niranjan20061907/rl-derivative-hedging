"""Common-path terminal evaluation, chunked simulation, and batched PPO inference."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from quantlab.backtesting.accounting import Portfolio, execute_step
from quantlab.backtesting.engine import BacktestEngine
from quantlab.environments.hedging_env import HedgingEnv
from quantlab.metrics.hedging import paired_bootstrap, terminal_metrics
from quantlab.pricing.black_scholes import call_features
from quantlab.rl.artifacts import load_policy, provenance, run_id
from quantlab.rl.config import ExperimentConfig, load_config
from quantlab.simulators.gbm import simulate_gbm_batch
from quantlab.strategies.base import Strategy
from quantlab.strategies.delta_hedging import DeltaHedgingStrategy


class NoHedgeStrategy(Strategy):
    name = "no_hedge"

    def action(self, observation, info):
        return 0.0


def run_env_episode(env: HedgingEnv, strategy: Strategy, seed: int):
    obs, info = env.reset(seed=seed)
    strategy.reset()
    errors, pnl, rewards = [0.0], [0.0], []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.array([strategy.action(obs, info)], dtype=np.float32))
        errors.append(info["hedging_error"])
        pnl.append(info["pnl"])
        rewards.append(reward)
        done = terminated or truncated
    return np.asarray(errors), np.asarray(pnl), np.asarray(rewards)


def policy_outcomes(prices, params, model, normalizer, deterministic=True):
    """Synchronized independent ledgers. The policy sees only current-row features."""
    batch, width = prices.shape
    steps = width - 1
    times = params.T * (1 - np.arange(width) / steps)
    features = call_features(prices, params.K, times, params.r, params.sigma)
    scale = np.maximum(features[:, 0, 0], 1e-8)
    state = Portfolio(cash=features[:, 0, 0].copy(), position=np.zeros(batch), error=np.zeros(batch))
    costs, turnover = np.zeros(batch), np.zeros(batch)
    inference = 0.0
    for t in range(steps):
        obs = np.stack(
            [
                np.log(prices[:, t] / params.K),
                features[:, t, 1],
                features[:, t, 2],
                np.full(batch, times[t] / params.T),
                np.full(batch, params.sigma),
                state.position,
                state.cash / scale,
            ],
            axis=-1,
        ).astype(np.float32)
        start = time.perf_counter()
        action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=deterministic)
        inference += time.perf_counter() - start
        outcome = execute_step(
            state,
            np.asarray(action[:, 0], dtype=float),
            prices[:, t],
            prices[:, t + 1],
            features[:, t + 1, 0],
            params.T / steps,
            params.r,
            params.cost_rate,
            t == steps - 1,
        )
        costs += outcome["transaction_cost"]
        turnover += outcome["turnover"]
    return state.error, costs, turnover, inference


def validation_rmse(model, normalizer, params, seeds):
    paths = simulate_gbm_batch(params.S0, params.r, params.sigma, params.T, params.steps, seeds)
    errors, _, _, _ = policy_outcomes(paths, params, model, normalizer)
    return float(np.sqrt(np.mean(errors**2)))


def evaluate(config: ExperimentConfig, model_path: str | Path | None = None, output_dir=None, matrix=True) -> dict:
    if config.environment.use_heston:
        raise ValueError("The controlled evaluation suite is GBM-only; Heston remains an exploratory stress test.")
    ev = config.evaluation
    seeds = list(range(ev.seed_start, ev.seed_start + ev.episodes))
    if set(seeds) & (set(range(100000, 100200)) | {11, 22, 33, config.seed}):
        raise ValueError("Test seed range overlaps training or validation seeds.")
    bundle = load_policy(model_path, config.environment) if model_path is not None else None
    output = Path(output_dir) if output_dir is not None else Path(config.results_dir) / f"eval_{run_id(config.seed)}"
    output.mkdir(parents=True, exist_ok=False)
    scenarios = (
        [(sigma, cost) for sigma in (0.1, 0.2, 0.4) for cost in (0.0, 0.001)]
        if matrix
        else [(config.environment.sigma, config.environment.cost_rate)]
    )
    summaries = {}
    with (output / "per_path.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["scenario", "seed", "strategy", "terminal_error", "transaction_cost", "turnover"]
        )
        writer.writeheader()
        for sigma, cost in scenarios:
            params = replace(config.environment, sigma=sigma, cost_rate=cost)
            key = f"sigma{sigma:g}_cost{cost:g}"
            names = ["delta", "delta_5"] + (["no_hedge"] if ev.include_no_hedge else [])
            if bundle:
                names += ["ppo"]
            rows = {name: [] for name in names}
            inference = 0.0
            start = time.perf_counter()
            for offset in range(0, ev.episodes, ev.chunk_size):
                chunk_seeds = seeds[offset : offset + ev.chunk_size]
                paths = simulate_gbm_batch(params.S0, params.r, sigma, params.T, params.steps, chunk_seeds)
                engine = BacktestEngine(params.K, params.r, sigma, params.T, cost)
                for seed, path in zip(chunk_seeds, paths, strict=True):
                    strategies = [
                        DeltaHedgingStrategy(params.K, params.r, sigma, params.T, params.steps, interval)
                        for interval in (1, 5)
                    ]
                    if ev.include_no_hedge:
                        strategies.append(NoHedgeStrategy())
                    for strategy in strategies:
                        result = engine.run(path, strategy)
                        row = [
                            result.terminal_error,
                            float(result.transaction_costs.sum()),
                            float(result.turnover.sum()),
                        ]
                        rows[strategy.name].append(row)
                        writer.writerow(dict(zip(writer.fieldnames, [key, seed, strategy.name, *row], strict=True)))
                if bundle:
                    errors, costs, turns, seconds = policy_outcomes(
                        paths, params, bundle[0], bundle[1], ev.deterministic
                    )
                    inference += seconds
                    for seed, error, total_cost, turn in zip(chunk_seeds, errors, costs, turns, strict=True):
                        row = [float(error), float(total_cost), float(turn)]
                        rows["ppo"].append(row)
                        writer.writerow(dict(zip(writer.fieldnames, [key, seed, "ppo", *row], strict=True)))
                stream.flush()
            arrays = {name: np.asarray(values) for name, values in rows.items()}
            summaries[key] = {
                "parameters": {"sigma": sigma, "cost_rate": cost},
                "metrics": {name: terminal_metrics(*array.T) for name, array in arrays.items()},
                "vs_daily_delta": {
                    name: paired_bootstrap(array[:, 0], arrays["delta"][:, 0], samples=ev.bootstrap_samples)
                    for name, array in arrays.items()
                    if name != "delta"
                },
                "wall_time_seconds": time.perf_counter() - start,
                "ppo_inference_seconds": inference,
            }
            print(f"Finished {key}: {ev.episodes} paired paths", flush=True)
    payload = {
        "provenance": provenance(),
        "config": config.to_dict(),
        "seed_start": ev.seed_start,
        "episodes_per_scenario": ev.episodes,
        "scenarios": summaries,
        "model_manifest": bundle[2] if bundle else None,
        "output_dir": str(output),
        "uncertainty_note": (
            "Path-paired intervals conditional on this trained model; training seeds reported separately."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    if bundle:
        bundle[1].close()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.add_argument("--model", help="v2 checkpoint bundle; omit for analytical baselines")
    parser.add_argument("--output")
    parser.add_argument("--single-scenario", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(load_config(args.config), args.model, args.output, not args.single_scenario), indent=2))


if __name__ == "__main__":
    main()
