# QuantLab file guide

This repository is a Python package organized around one experiment pipeline: simulate an underlying price path, price an option, choose hedge trades, account for those trades, and measure the final portfolio error. Application code is under `src/quantlab`; the repository root contains configuration, tests, documentation, and experiment evidence.

## Root files and project setup

- `.gitignore` keeps Python caches, virtual environments, downloaded data, locally generated checkpoints and run output, and macOS `.DS_Store` files out of version control. The `.gitkeep` placeholders below are the exceptions that keep the empty output directories visible in a fresh checkout.
- `README.md` is the setup and user guide. It explains the accounting conventions, observation and action contract, commands, metrics, experiment splits, benchmark methodology, and limitations.
- `FILE_GUIDE.md` (this file) explains the purpose of every maintained project file and how the pieces connect.
- `pyproject.toml` defines package metadata, runtime and optional dependencies, command-line entry points, package discovery under `src`, pytest defaults, and Ruff rules.
- `requirements-lock.txt` records the resolved Python environment used for the measured local run. It lets a Python 3.11 setup reproduce the installed package versions.
- `.env.example` was removed: it only contained a configuration pointer and no code reads environment variables for this project.
- `configs/rl.yaml` is the default experiment file. It holds the market and transaction-cost assumptions, PPO training parameters, training seed, validation interval, and held-out test range and chunk size.
- `.github/workflows/ci.yml` runs Ruff and the pytest suite for pushes and pull requests on Python 3.11.
- `checkpoints/.gitkeep` keeps the ignored local model-output directory in a fresh checkout; model bundles generated during training are ignored.
- `results/.gitkeep` keeps the ignored local training/evaluation-output directory in a fresh checkout; generated runs are ignored.

## Application package: `src/quantlab`

- `src/quantlab/__init__.py` gives the installed package its version.
- `src/quantlab/backtesting/__init__.py` exposes the reusable backtest engine as the package-level backtesting entry point.
- `src/quantlab/backtesting/accounting.py` is the shared ledger for the short-call replication experiment. It funds the position with the option premium, executes bounded stock trades, accrues cash, charges proportional fees, settles the liability, builds the versioned observation, and records terminal replication error.
- `src/quantlab/backtesting/engine.py` runs one strategy over a supplied price path. It offers a scalar reference route and an optimized route that precomputes option features, then returns the complete path and portfolio record.
- `src/quantlab/backtesting/benchmark.py` compares those two routes at 100, 1,000, and 10,000 paths, measures throughput and traced memory, and checks numerical agreement. Path generation is outside the timed section.
- `src/quantlab/backtesting/example.py` is a small runnable example of generating a path, running delta hedging, and printing its terminal error.
- `src/quantlab/environments/__init__.py` exposes the hedging environment and its parameter object.
- `src/quantlab/environments/hedging_env.py` implements the Gymnasium environment used for PPO. It simulates a seeded path, accepts an action that changes the hedge, applies the shared ledger, and returns a seven-value observation and telescoping terminal-error reward. Heston is an explicitly limited stress-test mode.
- `src/quantlab/metrics/__init__.py` exposes the package's metric functions.
- `src/quantlab/metrics/hedging.py` calculates per-trajectory diagnostics, cross-path terminal risk and cost metrics, and paired bootstrap intervals for strategy comparisons.
- `src/quantlab/pricing/__init__.py` exports the public call-pricing functions.
- `src/quantlab/pricing/black_scholes.py` contains validated scalar European call price, delta, and gamma functions, plus vectorized price/Greek calculation for efficient path processing. The scalar implementation is also used as the reference.
- `src/quantlab/simulators/__init__.py` exports both path generators.
- `src/quantlab/simulators/gbm.py` generates reproducible geometric Brownian motion paths, either individually or in independent seeded batches.
- `src/quantlab/simulators/heston.py` generates seeded price and variance paths using a simple Euler Heston scheme. Its output is currently an exploratory fixed-volatility valuation stress test.
- `src/quantlab/strategies/__init__.py` exposes the strategy interface, results, and strategy implementations.
- `src/quantlab/strategies/base.py` defines what an engine-callable strategy must do (`reset` and `action`) and the arrays/terminal error returned in a strategy result.
- `src/quantlab/strategies/delta_hedging.py` implements analytical delta hedging at a configured interval. Interval one gives daily rebalancing; interval five gives the weekly baseline.
- `src/quantlab/strategies/ppo_hedging.py` wraps a trained Stable-Baselines3 PPO policy behind the shared strategy interface, optionally applying saved observation normalization.
- `src/quantlab/strategies/sma.py` is a deterministic moving-average example strategy. It exists to exercise stateful strategy behavior; it is not one of the headline option-hedging baselines.
- `src/quantlab/rl/__init__.py` marks the RL training and evaluation package.
- `src/quantlab/rl/config.py` defines validated experiment, PPO, and evaluation settings; it loads and saves the YAML configuration.
- `src/quantlab/rl/artifacts.py` creates unique run IDs, records source/runtime provenance, hashes files, and checks observation versions and hashes before loading a model/normalizer bundle.
- `src/quantlab/rl/train.py` creates a seeded CPU PPO run, wraps the environment with training-only observation normalization, periodically evaluates on fixed validation paths, saves the best model and its matching normalizer, and streams reward logs in bounded batches.
- `src/quantlab/rl/evaluate.py` evaluates analytical baselines and, if supplied, a versioned PPO bundle on identical seeded paths. It batches policy inference, chunks simulations, and saves per-path CSV outcomes and aggregate metrics with uncertainty intervals.
- `src/quantlab/rl/experiment.py` coordinates the fixed three-training-seed experiment and its held-out test runs.
- `src/quantlab/rl/report.py` reads saved experiment and benchmark JSON/CSV outputs, generates plots, and renders the interview report. It does not retrain policies.

## Tests: `tests`

- `tests/unit/test_accounting.py` checks hand-calculated cash flows, interest, transaction costs, deterministic replication, environment/engine parity, reward telescoping, position limits, invalid actions, batch path seeding, pricing agreement, and the strategy information boundary.
- `tests/unit/test_backtesting.py` checks basic backtest output dimensions and finite results.
- `tests/unit/test_config.py` checks that invalid training/evaluation settings and malformed YAML are rejected.
- `tests/unit/test_environment.py` checks Gymnasium reset/step behavior, seed reproducibility, observations, termination, and action constraints.
- `tests/unit/test_metrics.py` checks hand-calculated metrics, terminal risk, and paired-bootstrap behavior.
- `tests/unit/test_pricing.py` checks known Black–Scholes values, invalid inputs, and nonfinite-input rejection.
- `tests/unit/test_simulators.py` checks path shape, positivity/nonnegative variance, reproducibility, and invalid GBM settings.
- `tests/unit/test_strategies.py` checks delta actions and deterministic moving-average behavior.
- `tests/integration/test_training_smoke.py` trains a tiny model, validates/saves/reloads its policy bundle, checks batched-vs-step inference and evaluation, verifies seed separation and tamper/legacy rejection, and checks repeatability for identical CPU seeds.
- The previous `tests/conftest.py` contained only a docstring and no fixtures or hooks, so it was removed.

## Reports and notebook

- `reports/INTERVIEW_REPORT.md` summarizes measured correctness, runtime, per-seed policy performance, uncertainty, limitations, and defensible interview phrasing.
- `reports/benchmark.json` stores the reference/optimized timings, throughput, memory, provenance, and numerical comparison data.
- `reports/controlled-experiment/experiment.json` records the experiment configuration, splits, training metadata, and aggregate evaluation for all three trained policies.
- `reports/controlled-experiment/seed11/metrics.json`, `seed22/metrics.json`, and `seed33/metrics.json` hold the scenario metrics and paired intervals for each trained policy.
- `reports/controlled-experiment/seed11/per_path.csv`, `seed22/per_path.csv`, and `seed33/per_path.csv` hold the per-seed, per-path results for every evaluated strategy and scenario. They preserve the observations from which the report tables are calculated.
- `reports/figures/terminal_rmse.png` compares held-out terminal RMSE across strategies/scenarios and training seeds.
- `reports/figures/terminal_distribution.png` plots terminal error distributions on identical paths in the representative scenario.
- `reports/figures/validation.png` shows validation RMSE across training checkpoints; the held-out test set is not used for checkpoint selection.
- `notebooks/data_exploration.ipynb` is a short display notebook for the saved evaluator outputs and report plots. It no longer downloads market data or compares incompatible reward proxies.

The former root-level `pricing/`, `simulator/`, `env/`, and `benchmarks/` Python files only re-exported implementations from `src/quantlab`, and no maintained code imported them. They were removed to keep a single canonical package path. `checkpoints/` and `results/` remain as ignored local output locations and are not committed.
