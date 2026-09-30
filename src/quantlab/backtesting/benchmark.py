"""Reproducible corrected-reference vs optimized benchmark (setup excluded)."""

from __future__ import annotations

import argparse
import json
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np

from quantlab.backtesting.engine import BacktestEngine
from quantlab.rl.artifacts import provenance
from quantlab.simulators.gbm import simulate_gbm_batch
from quantlab.strategies.delta_hedging import DeltaHedgingStrategy


def run_benchmark(output="reports/benchmark.json", sizes=(100, 1000, 10000), repeats=5, chunk_size=250):
    results = []
    for count in sizes:
        # Setup is excluded from timing; store only seed IDs, generate bounded chunks first.
        timings = {}
        max_difference = 0.0
        peaks = {}
        for optimized in (False, True):
            label = "optimized" if optimized else "reference"
            engine = BacktestEngine(100, 0.05, 0.2, 1, optimized=optimized)
            strategy = DeltaHedgingStrategy(100, 0.05, 0.2, 1, 252)
            warmup = simulate_gbm_batch(100, 0.05, 0.2, 1, 252, range(200000, 200005))
            for path in warmup:
                engine.run(path, strategy)
            timings[label] = []
            # Feature precomputation is part of engine timing; simulation is not.
            for _ in range(repeats):
                elapsed = 0.0
                for offset in range(0, count, chunk_size):
                    paths = simulate_gbm_batch(
                        100, 0.05, 0.2, 1, 252, range(200000 + offset, 200000 + min(count, offset + chunk_size))
                    )
                    start = time.perf_counter()
                    for path in paths:
                        engine.run(path, strategy)
                    elapsed += time.perf_counter() - start
                timings[label].append(elapsed)
            # Memory instrumentation is a separate run and never contaminates runtime.
            tracemalloc.start()
            for offset in range(0, count, chunk_size):
                paths = simulate_gbm_batch(
                    100, 0.05, 0.2, 1, 252, range(200000 + offset, 200000 + min(count, offset + chunk_size))
                )
                for path in paths:
                    engine.run(path, strategy)
            peaks[label] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        # Compare complete outputs on a representative subset, separately from timing.
        for path in simulate_gbm_batch(100, 0.05, 0.2, 1, 252, range(200000, 200000 + min(count, 100))):
            ref = BacktestEngine(100, 0.05, 0.2, 1, optimized=False).run(
                path, DeltaHedgingStrategy(100, 0.05, 0.2, 1, 252)
            )
            opt = BacktestEngine(100, 0.05, 0.2, 1).run(path, DeltaHedgingStrategy(100, 0.05, 0.2, 1, 252))
            for field in ("option_values", "hedging_error", "cash_balances", "transaction_costs"):
                a, b = getattr(ref, field), getattr(opt, field)
                max_difference = max(max_difference, float(np.max(abs(a - b))))
                np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
        row = {
            "paths": count,
            "steps": 252,
            "chunk_size": chunk_size,
            "speedup": float(np.median(timings["reference"]) / np.median(timings["optimized"])),
            "max_reference_difference": max_difference,
            "engines": {
                label: {
                    "seconds": values,
                    "median_seconds": float(np.median(values)),
                    "p95_seconds": float(np.percentile(values, 95)),
                    "paths_per_second": count / float(np.median(values)),
                    "peak_traced_bytes": peaks[label],
                }
                for label, values in timings.items()
            },
        }
        results.append(row)
        print(f"{count} paths: speedup={row['speedup']:.2f}x", flush=True)
    payload = {
        "provenance": provenance(),
        "repeats": repeats,
        "seed_start": 200000,
        "setup_excluded": "path generation; feature computation included",
        "memory_note": "tracemalloc peak from separate run; native allocator memory may be excluded",
        "process_max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()
    run_benchmark(args.output)


if __name__ == "__main__":
    main()
