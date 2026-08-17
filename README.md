# QUANTLAB

Async-ready backtesting and deep hedging foundation.

QuantLab is a Python package for reproducible option hedging experiments. This repository currently contains the core domain and RL foundation only: market simulation, Black-Scholes pricing, a Gymnasium hedging environment, reusable strategies, a minimal backtesting engine, metrics, PPO training, PPO evaluation, and tests.

It does not yet include FastAPI, PostgreSQL, Celery, Redis, a frontend, or cloud infrastructure.

## Problem Definition

The current RL task is discrete hedging of a European call option under simulated market dynamics.

- Observation: `[log_moneyness, delta, gamma, remaining_time_scaled, sigma, hedge_position]`
- Action: change in hedge position, bounded to `[-1, 1]`
- Transition: one simulated market timestep
- Accounting: option value change minus hedge P&L minus transaction cost
- Reward: `pnl - lambda_risk * pnl**2`
- Termination: after `steps` hedging intervals
- Market models: GBM by default, optional Heston path generation

The environment preserves the prototype's mathematical behavior while making reset seeding, dtype/shape behavior, and Gymnasium compatibility explicit.

## Architecture

```text
src/quantlab/
  environments/   Gymnasium hedging environment
  simulators/     GBM and Heston path simulation
  pricing/        Black-Scholes call price, delta, gamma
  strategies/     Delta, PPO, and SMA strategy adapters
  backtesting/    Reusable path-based backtest engine
  metrics/        Pure hedging metrics
  rl/             Config, train CLI, evaluate CLI
```

Future services can call the backtesting engine and strategies without depending on notebooks, web frameworks, task queues, or databases.

## Installation

Use Python 3.11-3.13. A local Python 3.11 virtual environment was used for verification.

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Configuration

Default experiment settings live in `configs/rl.yaml`. The config records:

- seed
- environment parameters
- PPO hyperparameters
- checkpoint directory
- results directory
- evaluation settings

Increase `ppo.total_timesteps` in YAML for longer runs, e.g. `100000`.

## Training

```bash
.venv/bin/python -m quantlab.rl.train --config configs/rl.yaml
```

Outputs:

- checkpoint: `checkpoints/ppo_hedging_seed42.zip`
- run metadata: `results/run_<timestamp>_seed42/metadata.json`
- config copy: `results/run_<timestamp>_seed42/config.yaml`
- training rewards: `results/run_<timestamp>_seed42/training_log.csv`

Stable-Baselines3 PPO collects complete rollouts, so configured `total_timesteps: 10000` with `n_steps: 256` produced `actual_timesteps: 10240`.

## Evaluation

```bash
.venv/bin/python -m quantlab.rl.evaluate \
  --config configs/rl.yaml \
  --model checkpoints/ppo_hedging_seed42.zip
```

Evaluation runs fresh seeded episodes and compares:

- PPO policy
- Black-Scholes delta hedging
- no-hedge baseline

The PPO policy is evaluated without further training.

## Baselines

`DeltaHedgingStrategy` is a reusable Black-Scholes delta baseline. It can run through `BacktestEngine` without notebooks or Stable-Baselines3.

`SMAStrategy` is included as a minimal deterministic strategy example for the future broader backtesting platform.

## Metrics

Core metrics currently implemented:

- mean absolute hedging error
- 95th percentile absolute hedging error
- P&L mean
- P&L standard deviation
- final hedging error

## Current Results

Verified on August 17, 2026 with seed `42`, GBM environment, 10 evaluation episodes, and configured PPO timesteps `10000` (`10240` actual rollout timesteps):

| Strategy | Mean Abs Hedging Error | P95 Abs Hedging Error | P&L Mean | P&L Std | Final Hedging Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO | 2.2546 | 4.3706 | -0.0175 | 0.0673 | -4.4399 |
| Delta | 2.1245 | 4.1156 | -0.0172 | 0.0203 | -4.3437 |
| No hedge | 10.8687 | 24.1063 | 0.0586 | 1.1329 | 14.8312 |

These results do not show PPO outperforming delta hedging on the default GBM configuration. The PPO run is a reproducibility smoke run, not a tuned research result.

## Testing

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

The test suite covers pricing, simulators, the Gymnasium environment, strategies, metrics, the backtesting engine, and a tiny PPO training smoke test.

## Known Limitations

- PPO has only been smoke-trained, not tuned or validated at research scale.
- The delta baseline and RL accounting are intentionally close to the prototype and should be reviewed before claims about production hedging accuracy.
- Heston simulation is available but the default verified experiment uses GBM.
- Transaction costs are simple proportional costs.
- No persistent database, job queue, REST API, or frontend exists yet.
- Notebooks are exploratory only and are no longer required for training or evaluation.

## Recommended Next Step

Run longer controlled experiments, including 100,000+ PPO timesteps and Heston evaluation, then decide whether the environment reward/accounting should be revised before building the future API and async job layer.
