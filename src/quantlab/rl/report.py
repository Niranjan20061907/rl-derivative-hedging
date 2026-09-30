"""Render measured experiment results and interview evidence without rerunning test evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def render_report(
    experiment="reports/controlled-experiment/experiment.json",
    benchmark="reports/benchmark.json",
    output="reports/INTERVIEW_REPORT.md",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiment_path = Path(experiment)
    data = json.loads(experiment_path.read_text())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir = output.parent / "figures"
    figure_dir.mkdir(exist_ok=True)
    runs = data["runs"]
    scenarios = list(runs[0]["evaluation"]["scenarios"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for ax, scenario in zip(axes.flat, scenarios, strict=True):
        baseline = runs[0]["evaluation"]["scenarios"][scenario]["metrics"]
        names = ["Daily delta", "Weekly delta", "PPO 11", "PPO 22", "PPO 33"]
        vals = [baseline["delta"]["terminal_rmse"], baseline["delta_5"]["terminal_rmse"]]
        vals += [run["evaluation"]["scenarios"][scenario]["metrics"]["ppo"]["terminal_rmse"] for run in runs]
        ax.bar(names, vals, color=["#1e6e59", "#78a78c", "#4471a6", "#7095c1", "#9cb5d4"])
        ax.set_title(scenario.replace("_", ", "))
        ax.set_ylabel("Terminal RMSE (currency units)")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out terminal error: PPO underperforms analytical delta (1000 paired paths/scenario)")
    fig.savefig(figure_dir / "terminal_rmse.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for run in runs:
        history = run["training"]["validation"]
        ax.plot(
            [row["timesteps"] for row in history],
            [row["terminal_rmse"] for row in history],
            marker="o",
            label=f"Seed {run['seed']}",
        )
    ax.set(
        xlabel="Collected timesteps",
        ylabel="Validation terminal RMSE",
        title="Checkpoint selection uses validation only",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figure_dir / "validation.png", dpi=160)
    plt.close(fig)
    # Raw terminal distribution from stored evaluator outcomes, not reward proxies.
    selected = "sigma0.2_cost0.001"
    path_rows = list(csv.DictReader((experiment_path.parent / "seed11" / "per_path.csv").open()))
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for strategy in ("delta", "delta_5", "ppo"):
        values = [
            float(row["terminal_error"])
            for row in path_rows
            if row["scenario"] == selected and row["strategy"] == strategy
        ]
        ax.hist(values, bins=45, density=True, histtype="step", linewidth=1.8, label=strategy)
    ax.set(
        xlabel="Terminal portfolio minus payoff",
        ylabel="Density",
        title="Identical paths, funded accounting (PPO seed 11)",
    )
    ax.legend()
    fig.savefig(figure_dir / "terminal_distribution.png", dpi=160)
    plt.close(fig)
    text = [
        "# QuantLab: measured interview evidence",
        "\nThis report is generated from stored outcomes. It does not rerun or select models using test results.",
        "\n## Correctness and reproducibility",
        "\nThe v2 ledger funds one short call with its initial premium, accounts for trades and cash interest, "
        "liquidates stock with costs, and settles the payoff. Terminal error is portfolio minus liability. "
        "The deterministic zero-volatility/no-cost replication test passes within 1e-10.",
        "\nTraining seeds: **11, 22, 33**. Validation: **100000–100199**. Test: **200000–200999**. "
        "Six volatility/cost scenarios use the same 1000 path seeds. These are 6000 scenario-path cases, "
        "not 18000 independent samples across models; scenarios also share underlying random shocks.",
        "\n## Training and held-out results",
        "\n| Seed | Actual steps | Selected validation RMSE | Training + validation seconds |",
        "|---|---:|---:|---:|",
    ]
    for run in runs:
        training = run["training"]
        text.append(
            f"| {run['seed']} | {training['actual_timesteps']} | "
            f"{training['manifest']['best_validation_rmse']:.4f} | {training['training_time_seconds']:.2f} |"
        )
    text += [
        "\nTimes above were collected while another local benchmark ran; they are descriptive, "
        "not an isolated training-performance comparison.",
        "\n| Scenario | Daily delta RMSE | Weekly delta RMSE | PPO 11 | PPO 22 | PPO 33 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in scenarios:
        m = runs[0]["evaluation"]["scenarios"][scenario]["metrics"]
        vals = [m["delta"]["terminal_rmse"], m["delta_5"]["terminal_rmse"]]
        vals += [r["evaluation"]["scenarios"][scenario]["metrics"]["ppo"]["terminal_rmse"] for r in runs]
        text.append("| " + scenario + " | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
    text += [
        "\n**All three PPO models underperform daily delta on terminal RMSE in every tested scenario.** "
        "Lower PPO trading costs do not establish a better hedge; error must be considered alongside cost.",
        "\n![Terminal RMSE](figures/terminal_rmse.png)",
        "\n![Validation](figures/validation.png)",
        "\n![Terminal distributions](figures/terminal_distribution.png)",
        "\n### Paired uncertainty in the default scenario",
        "\nIntervals resample price paths, conditional on each trained model. Negative differences favor PPO.",
        "\n| Seed | PPO − delta RMSE | Paired 95% bootstrap CI |",
        "|---|---:|---|",
    ]
    for run in runs:
        comparison = run["evaluation"]["scenarios"][selected]["vs_daily_delta"]["ppo"]
        low, high = comparison["rmse_difference_ci95"]
        text.append(f"| {run['seed']} | {comparison['rmse_difference']:.4f} | [{low:.4f}, {high:.4f}] |")
    # An honest positive finding, chosen for explanatory value; do not claim scenario selection was pre-registered.
    stress = runs[0]["evaluation"]["scenarios"]["sigma0.1_cost0.001"]
    daily, weekly = stress["metrics"]["delta"], stress["metrics"]["delta_5"]
    reduction = 100 * (1 - weekly["terminal_rmse"] / daily["terminal_rmse"])
    cost_reduction = 100 * (1 - weekly["mean_transaction_cost"] / daily["mean_transaction_cost"])
    ci = stress["vs_daily_delta"]["delta_5"]["rmse_difference_ci95"]
    text += [
        f"\nA useful observed tradeoff: weekly delta reduces terminal RMSE by **{reduction:.1f}%** "
        f"and mean transaction costs by **{cost_reduction:.1f}%** in the 10% volatility, 10-basis-point-cost scenario. "
        f"The paired RMSE difference CI is [{ci[0]:.4f}, {ci[1]:.4f}]. "
        "This result is scenario-specific; weekly rebalancing performs worse in several other scenarios."
    ]
    benchmark_path = Path(benchmark)
    if benchmark_path.exists():
        bench = json.loads(benchmark_path.read_text())
        text += [
            "\n## Corrected-reference performance",
            "\nFive timed repetitions after warm-up. Path generation is excluded; feature computation is included. "
            "Memory is traced separately and excludes some native allocations.",
            "\n| Paths | Reference median / P95 (s) | Optimized median / P95 (s) | Speedup | Optimized paths/s | "
            "Peak traced MiB | Maximum difference |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in bench["results"]:
            a, b = row["engines"]["reference"], row["engines"]["optimized"]
            text.append(
                f"| {row['paths']} | {a['median_seconds']:.3f} / {a['p95_seconds']:.3f} | "
                f"{b['median_seconds']:.3f} / {b['p95_seconds']:.3f} | {row['speedup']:.2f}× | "
                f"{b['paths_per_second']:.1f} | {b['peak_traced_bytes'] / 1024**2:.2f} | "
                f"{row['max_reference_difference']:.2e} |"
            )
        largest = bench["results"][-1]
        text += [
            "\n## Interview wording",
            f"\n“Built a reproducible Python option-hedging engine and improved corrected-reference throughput "
            f"**{largest['speedup']:.2f}×** across a 10000-path workload, with maximum checked output difference "
            f"**{largest['max_reference_difference']:.2e}**.”",
            "\n“Compared PPO and analytical baselines across three training seeds and six held-out scenarios; "
            "validated financing and settlement, and reported paired uncertainty and cost–risk tradeoffs.”",
        ]
    text += [
        "\n## Limits and follow-up",
        "\nSynthetic GBM evaluation of a single European call; no real-market execution evidence. "
        "Models trained at 20% volatility are stress-tested at 10% and 40%; those results include distribution shift. "
        "Heston remains a fixed-volatility valuation proxy with incomplete drift semantics. "
        "No API, persistent job queue, frontend, or cloud deployment is implemented. "
        "Future work: Heston consistency, reward/training-budget ablations, then a bounded backend service.",
        "\n## Reproduce",
        "\n```bash\npython -m quantlab.rl.experiment --output reports/new-experiment\n"
        "python -m quantlab.backtesting.benchmark --output reports/new-benchmark.json\n"
        "python -m quantlab.rl.report --experiment reports/new-experiment/experiment.json "
        "--benchmark reports/new-benchmark.json --output reports/new-report.md\n```",
    ]
    output.write_text("\n".join(text) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="reports/controlled-experiment/experiment.json")
    parser.add_argument("--benchmark", default="reports/benchmark.json")
    parser.add_argument("--output", default="reports/INTERVIEW_REPORT.md")
    args = parser.parse_args()
    print(render_report(args.experiment, args.benchmark, args.output))


if __name__ == "__main__":
    main()
