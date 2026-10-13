"""TD-MPC2 terminal-save adapter for the frozen truck baseline queue."""
import hashlib, json, os, runpy, sys
from pathlib import Path

def main():
    root = Path(os.environ["PHASE0F_PROJECT_ROOT"]).resolve()
    output = Path(os.environ["PHASE0F_TERMINAL_OUTPUT"]).resolve()
    run_id = os.environ["PHASE0F_INFRA_RUN_ID"]
    sys.path.insert(0, str(root / "benchmark" / "tdmpc2"))
    os.chdir(root / "benchmark")
    from tdmpc2.trainer.online_trainer import OnlineTrainer
    original_train = OnlineTrainer.train
    def train_and_save_terminal(self):
        result = original_train(self)
        terminal_step = int(self._step)
        assert terminal_step >= int(self.cfg.steps), (terminal_step, int(self.cfg.steps))
        checkpoint = Path(self.cfg.work_dir) / "models" / "final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(checkpoint)
        receipt = {
            "method": "tdmpc", "seed": int(self.cfg.seed),
            "terminal_step": terminal_step, "requested_budget": int(self.cfg.steps),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_selection": "TERMINAL_EVERY_PREDECLARED_SEED",
            "training_loop_modified": False, "terminal_save_adapter_only": True,
            "infrastructure_run_id": run_id, "policy_force_traces_inspected": False,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "TERMINAL_CHECKPOINT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return result
    OnlineTrainer.train = train_and_save_terminal
    try:
        runpy.run_module("tdmpc2.train", run_name="__main__")
    finally:
        OnlineTrainer.train = original_train
if __name__ == "__main__":
    main()
