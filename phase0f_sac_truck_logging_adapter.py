"""Run the unchanged official SAC runner with local-only named logging.

The official runner's default unnamed group is None, incompatible with its
save_dir path join. Only logging metadata and upload mode are overridden.
No learner, replay, training, reward, evaluation, or checkpoint code changes.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    root = Path(os.environ["PHASE0F_PROJECT_ROOT"]).resolve()
    plan = json.loads((root / "outputs/PHASE0F_TRUCK_BASELINE_RUN_MANIFEST.json").read_text())
    assert plan["freeze_status"] == "FROZEN_BEFORE_SCIENTIFIC_TRAINING"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan["runner_hashes"][Path(__file__).name]
    import jaxrl_m.wandb as logging
    original = logging.setup_wandb
    seed = next(argument.split("=", 1)[1] for argument in sys.argv[1:] if argument.startswith("--seed="))

    def local_only(hyperparam_dict, **kwargs):
        kwargs.update(offline=True, group="PHASE0F-FROZEN-SAC", unique_identifier=f"seed-{seed}")
        return original(hyperparam_dict, **kwargs)

    logging.setup_wandb = local_only
    source = root / "benchmark/jaxrl_m/examples/mujoco/run_mujoco_sac.py"
    sys.path.insert(0, str(source.parent))
    sys.argv[0] = str(source)
    try:
        runpy.run_path(str(source), run_name="__main__")
        save_arg = next(argument.split("=", 1)[1] for argument in sys.argv[1:] if argument.startswith("--save_dir="))
        save_dir = Path(save_arg).resolve()
        checkpoint = save_dir / "checkpoint_10000000"
        assert checkpoint.exists(), checkpoint
        files = [checkpoint] if checkpoint.is_file() else [path for path in checkpoint.rglob("*") if path.is_file()]
        digest = hashlib.sha256()
        for path in sorted(files):
            digest.update(path.read_bytes())
        receipt = {"method": "sac", "terminal_step": 10000000,
                   "requested_budget": 10000000, "checkpoint": str(checkpoint),
                   "checkpoint_sha256": digest.hexdigest(),
                   "checkpoint_selection": "TERMINAL_EVERY_PREDECLARED_SEED",
                   "training_loop_modified": False, "terminal_save_adapter_only": True,
                   "policy_force_traces_inspected": False}
        (save_dir / "TERMINAL_CHECKPOINT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        logging.setup_wandb = original


if __name__ == "__main__":
    main()
