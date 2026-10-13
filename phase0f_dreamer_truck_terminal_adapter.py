"""Invoke the unchanged official runner, then persist its terminal state.

No training/evaluation callback is modified. No force observations are read.
The policy, optimizer state, and step counter are saved with the official
checkpoint class after normal completion of the official train_eval loop.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    import embodied
    from embodied.agents.dreamerv3 import train

    root = Path(os.environ["PHASE0F_PROJECT_ROOT"]).resolve()
    plan = json.loads((root / "outputs/PHASE0F_TRUCK_BASELINE_RUN_MANIFEST.json").read_text())
    assert plan["freeze_status"] == "FROZEN_BEFORE_SCIENTIFIC_TRAINING"
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert own_hash == plan["runner_hashes"][Path(__file__).name]
    original = embodied.run.train_eval

    def run_and_save(make_agent, make_train_replay, make_eval_replay,
                     make_train_env, make_eval_env, make_logger, args):
        captured = {}

        def capture_agent():
            captured["agent"] = make_agent()
            return captured["agent"]

        def capture_logger():
            captured["logger"] = make_logger()
            return captured["logger"]

        original(capture_agent, make_train_replay, make_eval_replay,
                 make_train_env, make_eval_env, capture_logger, args)
        step = captured["logger"].step
        assert int(step) >= args.steps, (int(step), args.steps)
        filename = embodied.Path(args.logdir) / "terminal.ckpt"
        checkpoint = embodied.Checkpoint(filename, parallel=False)
        checkpoint.agent = captured["agent"]
        checkpoint.step = step
        checkpoint.save()
        receipt = {"terminal_step": int(step), "requested_budget": args.steps,
                   "checkpoint": str(filename),
                   "checkpoint_selection": "TERMINAL_EVERY_PREDECLARED_SEED",
                   "training_loop_modified": False, "policy_force_traces_inspected": False}
        (Path(args.logdir) / "TERMINAL_CHECKPOINT_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2) + "\n")

    embodied.run.train_eval = run_and_save
    try:
        train.main(sys.argv[1:])
    finally:
        embodied.run.train_eval = original


if __name__ == "__main__":
    main()
