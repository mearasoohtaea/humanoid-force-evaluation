# Humanoid Force Evaluation

**Reproducible motion-return evaluation toward force-aware humanoid policy comparison.**

Research in progress by Sihan Chen, using the official HumanoidBench environment.
The scientific question is whether rankings based on task reward agree with
rankings based on contact-force quality under a matched policy/evaluation protocol.

## Implemented code

- exp1_return_eval.py: frozen terminal-checkpoint evaluation for the Window task.
- truck_exp1_return_eval.py: Truck return-evaluation runner.
- DreamerV3, TD-MPC2 and SAC adapters: preserve the upstream training/action paths
  while saving terminal checkpoints and execution receipts.
- WINDOW_EVALUATION_PROTOCOL.json: common prospective evaluation definition.
- WINDOW_RETURN_REPORT.md: completed Window return results and their interpretation.
- SOURCE_MANIFEST.json: hashes of the released runner/adapter files.

## Current progress — 2026-10-13

Window: three algorithms × three independent training seeds × 30 evaluation
episodes = **270 completed episodes**. All nine checkpoints were restored and
evaluated using their native action paths. See the report for original returns,
native termination and the distinction between execution validity and motion eligibility.

Truck: five existing terminal policies are available (DreamerV3 n=3 and TD-MPC2 n=2).
The common 150-episode return evaluation is pending. Published benchmark numbers
are kept separate from measurements of these same policies.

Force traces, the normalized force-error metric, matched force evaluation,
held-out interaction tests and a ranking-reversal result are **not completed**.
This repository currently publishes the return-evaluation and checkpoint-adapter
code; it does not claim that force-based ranking reversal has been demonstrated.

## Setup and scope

Use the official [HumanoidBench source](https://github.com/carlosferrazza/humanoid-bench)
at commit cb1189039151c8aadaaa987b442da54383c87fab, with its algorithm-specific
dependency environments. This repository does not vendor third-party baseline
code, robot assets, model weights, SSH configuration or cloud credentials.

The preserved runners expect an experiment root with benchmark/, terminal
checkpoints, run receipts and the original frozen manifest under outputs/.
They are not a standalone training implementation. Run python exp1_return_eval.py --help
to inspect supported commands after preparing that experiment layout.
Do not use a later policy or a best-of checkpoint as a replacement for a frozen
terminal checkpoint. Force comparisons must evaluate the same policy set used
for the paired return measurement.

## 中文介绍

围绕人形机器人接触任务，复现DreamerV3、TD-MPC2和SAC基线，并建立统一回报评测、
检查点核验与日志追溯流程。Window任务已完成9组策略、270回合评价。
力交互评价和跨接触条件验证仍在推进，尚未建立算法排序反转结论。

## License and citation

First-party runners/adapters: MIT. Upstream algorithms and assets keep their own
licenses. The software citation records an ongoing development toolkit, not a
published or accepted paper.
