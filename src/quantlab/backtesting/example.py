"""Small runnable backtesting example."""

from quantlab.backtesting.engine import BacktestEngine
from quantlab.simulators.gbm import simulate_gbm
from quantlab.strategies.delta_hedging import DeltaHedgingStrategy


def main() -> None:
    prices = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, steps=32, seed=42)
    strategy = DeltaHedgingStrategy(K=100, r=0.05, sigma=0.2, T=1.0, steps=32)
    result = BacktestEngine(K=100, r=0.05, sigma=0.2, T=1.0).run(prices, strategy)
    print({"strategy": result.name, "final_hedging_error": float(result.hedging_error[-1])})


if __name__ == "__main__":
    main()
