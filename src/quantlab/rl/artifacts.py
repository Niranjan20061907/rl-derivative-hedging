"""Run provenance and strict policy/normalization bundle loading."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from quantlab.backtesting.accounting import CONTRACT_VERSION


def run_id(seed):
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_seed{seed}_{uuid.uuid4().hex[:8]}"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    packages = ["numpy", "scipy", "torch", "stable-baselines3", "gymnasium", "PyYAML"]
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": {name: importlib.metadata.version(name) for name in packages},
        "contract_version": CONTRACT_VERSION,
    }


def load_policy(model_path, params):
    import json

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from quantlab.environments.hedging_env import HedgingEnv

    model_path = Path(model_path)
    manifest_path = model_path.with_suffix(".json")
    if not manifest_path.exists():
        raise ValueError("Missing v2 model manifest; legacy checkpoints must be retrained.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Incompatible observation/accounting version.")
    normalization = model_path.parent / manifest["normalization_file"]
    if sha256(model_path) != manifest["model_sha256"] or sha256(normalization) != manifest["normalization_sha256"]:
        raise ValueError("Policy bundle hash mismatch.")
    wrapped = DummyVecEnv([lambda: HedgingEnv(params)])
    normalizer = VecNormalize.load(str(normalization), wrapped)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(str(model_path), device="cpu")
    if model.observation_space.shape != (7,) or normalizer.observation_space.shape != (7,):
        raise ValueError("Incompatible observation shape.")
    return model, normalizer, manifest
