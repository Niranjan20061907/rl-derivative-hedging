"""Run the locked three-seed training and six-scenario held-out evaluation suite."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from quantlab.rl.artifacts import provenance, run_id
from quantlab.rl.config import load_config
from quantlab.rl.evaluate import evaluate
from quantlab.rl.train import train


def run_experiment(config_path="configs/rl.yaml", output=None):
    config = load_config(config_path)
    output = Path(output or f"reports/experiment_{run_id(config.seed)}")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "provenance": provenance(),
        "config": config.to_dict(),
        "training_seeds": [11, 22, 33],
        "validation_seeds": [100000, 100199],
        "test_seeds": [config.evaluation.seed_start, config.evaluation.seed_start + config.evaluation.episodes - 1],
        "runs": [],
    }
    for seed in (11, 22, 33):
        run_config = replace(config, seed=seed)
        metadata = train(run_config)
        metrics = evaluate(run_config, metadata["model_path"], output / f"seed{seed}")
        # Three model-specific reports reuse the same paths; they are not 18000 independent scenarios.
        payload["runs"].append({"seed": seed, "training": metadata, "evaluation": metrics})
        (output / "experiment.json").write_text(json.dumps(payload, indent=2))
        print(f"Completed training and test evaluation for seed {seed}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_experiment(args.config, args.output)


if __name__ == "__main__":
    main()
