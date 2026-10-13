"""Resume the frozen Truck original-return evaluation, then stop at EXP-1.

This adapter imports only policy loading/actions and runtime setup from the
existing evaluator. It never calls its Window freeze, run, or summary commands.
Scientific records live exclusively in outputs/TRUCK_EXP1. The parent creates
terminal_inventory.json from the existing Truck terminal artifacts.

Run with the existing Linux environment, for example:
  .venv_phase0f/bin/python truck_exp1_return_eval.py prepare --root PROJECT
  .venv_phase0f/bin/python -u truck_exp1_return_eval.py run --root PROJECT

There are zero automatic retries. A failed worker keeps its log and stops run.
"""

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import traceback

from exp1_return_eval import checkpoint_digest, digest, make_policy, setup_environment


TASK = "h1hand-truck-v0"
COMMIT = "cb1189039151c8aadaaa987b442da54383c87fab"
METHODS = ("DreamerV3", "TD-MPC2", "SAC")
KEYS = {"DreamerV3": "dreamer", "TD-MPC2": "tdmpc", "SAC": "sac"}
BUDGETS = {"DreamerV3": 10_000_000, "TD-MPC2": 2_000_000, "SAC": 10_000_000}
EXPECTED_JOBS = [(method, seed) for method in METHODS for seed in range(3)]
SEEDS = list(range(1000, 1030))
HORIZON = 1000


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def paths(root):
    return root / "outputs/TRUCK_EXP1", root / "tmp/truck_exp1_return_eval"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def local_path(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact is outside the declared project: {path}")
    return path


def inputs(root, manifest_path=None):
    out, runtime = paths(root)
    protocol_path = root / "outputs/NEXT_TASK/TRUCK_RETURN_PROTOCOL.json"
    plan = read_json(root / "outputs/NEXT_TASK/BASELINE_TRAINING_PLAN.json")
    queue = read_json(root / "outputs/NEXT_TASK/BASELINE_QUEUE.json")
    protocol = read_json(protocol_path)
    inventory = read_json(out / "terminal_inventory.json")
    manifest_path = local_path(root, manifest_path or "outputs/PHASE0F_TRUCK_BASELINE_RUN_MANIFEST.json")
    manifest = read_json(manifest_path)

    for name, data in (("protocol", protocol), ("training plan", plan), ("queue", queue)):
        if data.get("task") != TASK or data.get("source_commit") != COMMIT:
            raise ValueError(f"Unexpected frozen Truck task/source in {name}")
    if inventory.get("task") != TASK:
        raise ValueError("The terminal inventory must describe Truck only")
    if manifest.get("task", TASK) != TASK:
        raise ValueError("The training manifest is not the Truck manifest")
    if manifest.get("commit", manifest.get("source_commit")) != COMMIT:
        raise ValueError("Unexpected manifest source commit")
    if protocol.get("eval_seeds") != SEEDS or protocol.get("episodes_per_checkpoint") != 30:
        raise ValueError("The frozen 30-episode seed definition does not match")
    if protocol.get("maximum_total_episodes") != 270 or protocol["environment"]["horizon"] != HORIZON:
        raise ValueError("The frozen total episodes or native horizon does not match")
    if protocol["environment"].get("force_instrumentation") is not False:
        raise ValueError("The protocol must exclude force instrumentation")
    if plan.get("budgets") != BUDGETS or queue.get("automatic_retries") != 0:
        raise ValueError("Unexpected frozen budgets or retry policy")
    expected_order = [(item["method"], item["seed"]) for item in queue["order"]]
    if expected_order != EXPECTED_JOBS:
        raise ValueError("Unexpected frozen serial order")
    if any(item["budget_steps"] != BUDGETS[item["method"]] for item in queue["order"]):
        raise ValueError("Unexpected budget in the frozen queue")
    protocol_hash = digest(protocol_path)
    manifest_hash = digest(manifest_path)
    if inventory.get("protocol_sha256") != protocol_hash:
        raise ValueError("Inventory protocol hash does not match the frozen Truck protocol")
    if inventory.get("manifest_sha256") != manifest_hash:
        raise ValueError("Inventory manifest hash does not match the frozen Truck manifest")
    jobs = inventory.get("jobs", [])
    identities = [(entry["METHOD"], entry["SEED"]) for entry in jobs]
    if len(identities) != 9 or set(identities) != set(EXPECTED_JOBS):
        raise ValueError("Inventory must contain exactly the nine predeclared Truck jobs")
    by_identity = dict(zip(identities, jobs))
    jobs = [by_identity[key] for key in EXPECTED_JOBS]
    for entry in jobs:
        method = entry["METHOD"]
        if entry["TARGET_STEP"] != BUDGETS[method]:
            raise ValueError(f"Inventory budget mismatch: {method} seed {entry['SEED']}")
        if entry.get("CHECKPOINT_VALIDITY") == "VALID":
            step = entry.get("TERMINAL_STEP")
            if type(step) is not int or step < entry["TARGET_STEP"]:
                raise ValueError("A VALID terminal checkpoint must reach the official runner budget")
            checkpoint = local_path(root, entry["TERMINAL_CHECKPOINT"])
            relative = checkpoint.relative_to(root).parts
            is_truck_output = relative[:2] == ("outputs", "BASELINES_TRUCK")
            is_native_tdmpc_output = (method == "TD-MPC2" and relative[:3] ==
                                     ("benchmark", "logs", "humanoid_h1hand-truck-v0"))
            if not (is_truck_output or is_native_tdmpc_output):
                raise ValueError("Terminal checkpoint is not in its existing Truck output tree")
            if not checkpoint.exists() or not entry.get("TERMINAL_RECEIPT"):
                raise ValueError("VALID terminal checkpoint/receipt is absent")
            if not local_path(root, entry["TERMINAL_RECEIPT"]).is_file():
                raise ValueError("VALID terminal receipt is not an existing file")
            checksum = entry.get("CHECKPOINT_SHA256", "")
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError("VALID terminal checkpoint lacks its SHA256")
    return {"out": out, "runtime": runtime, "protocol": protocol,
            "protocol_hash": protocol_hash, "manifest": manifest,
            "manifest_path": manifest_path, "manifest_hash": manifest_hash,
            "inventory": inventory, "jobs": jobs}


def boolean_or_unknown(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError("Native success must be an exposed 0/1 value or UNKNOWN")


def row_identity(row):
    return (row["METHOD"], row["TRAIN_SEED"], row["CHECKPOINT_SHA256"], row["EVAL_SEED"])


def read_rows(data):
    path = data["out"] / "original_return_raw.jsonl"
    if not path.exists():
        return []
    expected = {(entry["METHOD"], entry["SEED"], entry.get("CHECKPOINT_SHA256"), seed)
                for entry in data["jobs"] for seed in SEEDS}
    rows, observed = [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                identity = row_identity(row)
                if identity not in expected or identity in observed:
                    raise ValueError("Unexpected checkpoint/seed or duplicate valid-record identity")
                if row.get("TASK", TASK) != TASK:
                    raise ValueError("Existing record describes a different task")
                if row.get("PROTOCOL_SHA256") != data["protocol_hash"]:
                    raise ValueError("Existing record lacks matching frozen-protocol provenance")
                if row.get("EVALUATION_MODE") != data["protocol"]["policy_modes"][row["METHOD"]]:
                    raise ValueError("Existing record uses a different policy mode")
                if type(row["EPISODE_INDEX"]) is not int or row["EPISODE_INDEX"] != SEEDS.index(row["EVAL_SEED"]):
                    raise ValueError("Incorrect evaluation episode index")
                if isinstance(row["EPISODE_RETURN"], bool) or not math.isfinite(row["EPISODE_RETURN"]):
                    raise ValueError("Nonfinite or invalid native return")
                if type(row["EPISODE_LENGTH"]) is not int or not 1 <= row["EPISODE_LENGTH"] <= HORIZON:
                    raise ValueError("Incorrect native episode length")
                if type(row["TERMINATED"]) is not bool or type(row["TRUNCATED"]) is not bool:
                    raise ValueError("Missing native termination flags")
                if not (row["TERMINATED"] or row["TRUNCATED"]):
                    raise ValueError("Record does not describe a finished native episode")
                boolean_or_unknown(row["TASK_SUCCESS"])
                subtask = row["SUCCESS_SUBTASKS"]
                if subtask is not None and (type(subtask) is not int or not 0 <= subtask <= 5):
                    raise ValueError("Incorrect exposed Truck subtask count")
                if boolean_or_unknown(row["TASK_SUCCESS"]) is True and subtask not in (None, 5):
                    raise ValueError("Inconsistent native five-package success record")
                observed.add(identity)
                rows.append(row)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                raise ValueError(f"Invalid existing raw record at line {line_number}; preserve it and stop: {error}") from error
    return rows


def valid_count(data):
    return sum(entry.get("CHECKPOINT_VALIDITY") == "VALID" for entry in data["jobs"])


def require_ready(data):
    if valid_count(data) != 9:
        missing = [f"{entry['METHOD']} seed {entry['SEED']}: {entry.get('REMAINING_ACTION')}"
                   for entry in data["jobs"] if entry.get("CHECKPOINT_VALIDITY") != "VALID"]
        raise ValueError("Truck evaluation awaits nine valid terminal checkpoints: " + "; ".join(missing))


def scalar_native(info, key, np):
    """Access only the two explicitly authorized native Truck info values."""
    if key not in info:
        return None
    value = info[key]
    if isinstance(value, np.generic):
        value = value.item()
    if key == "success":
        return boolean_or_unknown(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Exposed native success_subtasks is not a numeric count")
    if not math.isfinite(value) or value != int(value) or not 0 <= value <= 5:
        raise ValueError("Exposed native success_subtasks is outside the original five packages")
    return int(value)


def evaluate_checkpoint(root, index, manifest_path=None):
    data = inputs(root, manifest_path)
    require_ready(data)
    rows = read_rows(data)
    entry = data["jobs"][index]
    method, seed = entry["METHOD"], entry["SEED"]
    done = {row["EVAL_SEED"] for row in rows if row["METHOD"] == method and row["TRAIN_SEED"] == seed}
    if done == set(SEEDS):
        print("CHECKPOINT_ALREADY_EVALUATED", method, seed, flush=True)
        return
    checkpoint = local_path(root, entry["TERMINAL_CHECKPOINT"])
    if checkpoint_digest(checkpoint) != entry["CHECKPOINT_SHA256"]:
        raise ValueError("Terminal checkpoint no longer matches its inventoried SHA256")
    candidates = [job for job in data["manifest"]["jobs"]
                  if job["method"] == KEYS[method] and job["seed"] == seed]
    if len(candidates) != 1:
        raise ValueError("A unique frozen method/seed configuration is required")
    cfg = candidates[0]["configuration"]
    setup_environment(root)
    import numpy as np
    import gymnasium as gym
    import humanoid_bench  # registers the original benchmark; no task instrumentation

    # Loading once preserves the established native policy path and avoids 30
    # repeated model loads/JAX compilations. This template is never stepped.
    policy_env = gym.make(TASK, render_mode=None)
    try:
        if policy_env.spec.max_episode_steps != HORIZON or policy_env.action_space.shape != (61,):
            raise ValueError("Unexpected frozen Truck horizon/action interface")
        reset_policy, policy = make_policy(method, seed, policy_env, checkpoint, cfg, root)
        print("CHECKPOINT_LOADED", method, seed, entry["CHECKPOINT_SHA256"], flush=True)
        for episode_index, eval_seed in enumerate(SEEDS):
            if eval_seed in done:
                continue
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            # A new original environment per episode resets Truck's persistent
            # package lists through its constructor; no initialization edits.
            env = gym.make(TASK, render_mode=None)
            try:
                if env.spec.max_episode_steps != HORIZON or env.action_space.shape != (61,):
                    raise ValueError("Unexpected frozen Truck horizon/action interface")
                env.action_space.seed(eval_seed)
                env.observation_space.seed(eval_seed)
                reset_policy(eval_seed)
                observation, _ = env.reset(seed=eval_seed)
                total, reward, length = 0.0, 0.0, 0
                terminated = truncated = False
                print("EPISODE_START", method, seed, eval_seed, now(), flush=True)
                while not (terminated or truncated):
                    action = np.asarray(policy(observation, reward, length == 0))
                    if action.shape != (61,) or not np.isfinite(action).all():
                        raise ValueError("Invalid native policy action")
                    action = np.clip(action, -1, 1)  # existing native boundary clipping
                    observation, reward, terminated, truncated, info = env.step(action)
                    reward = float(reward)
                    if not math.isfinite(reward):
                        raise ValueError("Nonfinite original env.step reward")
                    total += reward
                    length += 1
                    if length > HORIZON:
                        raise ValueError("Original native horizon was not enforced")
                task_success = scalar_native(info, "success", np)
                success_subtasks = scalar_native(info, "success_subtasks", np)
                if task_success is True and success_subtasks not in (None, 5):
                    raise ValueError("Inconsistent exposed native Truck success fields")
                five_complete = (success_subtasks == 5 if success_subtasks is not None else task_success)
                record = {"TASK": TASK, "METHOD": method, "TRAIN_SEED": seed,
                          "CHECKPOINT_SHA256": entry["CHECKPOINT_SHA256"],
                          "EVAL_SEED": eval_seed, "EPISODE_INDEX": episode_index,
                          "EPISODE_RETURN": total, "EPISODE_LENGTH": length,
                          "TERMINATED": bool(terminated), "TRUNCATED": bool(truncated),
                          "TASK_SUCCESS": task_success, "SUCCESS_SUBTASKS": success_subtasks,
                          "FIVE_PACKAGES_COMPLETE": five_complete,
                          "TASK_SUCCESS_STATUS": "EXPLICIT_NATIVE_INFO" if task_success is not None else "TASK_SUCCESS_NOT_EXPLICITLY_EXPOSED",
                          "SUBTASK_STATUS": "EXPLICIT_NATIVE_INFO" if success_subtasks is not None else "SUCCESS_SUBTASKS_NOT_EXPLICITLY_EXPOSED",
                          "EVALUATION_MODE": data["protocol"]["policy_modes"][method],
                          "PROTOCOL_SHA256": data["protocol_hash"],
                          "COMPLETED_AT_UTC": now()}
                if not math.isfinite(total):
                    raise ValueError("Nonfinite original episode return")
                data["out"].mkdir(parents=True, exist_ok=True)
                with (data["out"] / "original_return_raw.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print("EPISODE_DONE", method, seed, eval_seed, "return", total,
                      "length", length, "success", task_success, "subtasks", success_subtasks, flush=True)
            finally:
                env.close()
    finally:
        policy_env.close()
    if checkpoint_digest(checkpoint) != entry["CHECKPOINT_SHA256"]:
        raise ValueError("Terminal checkpoint changed during evaluation")


def summarize(root, manifest_path=None, errors=None):
    data = inputs(root, manifest_path)
    rows = read_rows(data)
    previous_path = data["out"] / "checkpoint_summary.json"
    previous = read_json(previous_path) if previous_path.exists() else {}
    if errors is None:
        errors = previous.get("active_runtime_errors", previous.get("runtime_errors", []))
    error_history = list(previous.get("runtime_errors", []))
    for error in errors:
        if error not in error_history:
            error_history.append(error)
    checkpoint_summaries = []
    for index, entry in enumerate(data["jobs"]):
        subset = [row for row in rows if row["METHOD"] == entry["METHOD"] and row["TRAIN_SEED"] == entry["SEED"]]
        values = [row["EPISODE_RETURN"] for row in subset]
        known_success = [boolean_or_unknown(row["TASK_SUCCESS"]) for row in subset if row["TASK_SUCCESS"] is not None]
        known_subtasks = [row["SUCCESS_SUBTASKS"] for row in subset if row["SUCCESS_SUBTASKS"] is not None]
        five_flags = [(row["SUCCESS_SUBTASKS"] == 5 if row["SUCCESS_SUBTASKS"] is not None else boolean_or_unknown(row["TASK_SUCCESS"])) for row in subset]
        summary = {"METHOD": entry["METHOD"], "TRAIN_SEED": entry["SEED"],
                   "CHECKPOINT_SHA256": entry.get("CHECKPOINT_SHA256"),
                   "TERMINAL_CHECKPOINT": entry.get("TERMINAL_CHECKPOINT"),
                   "TERMINAL_STEP": entry.get("TERMINAL_STEP"), "TARGET_STEP": entry["TARGET_STEP"],
                   "CHECKPOINT_VALIDITY": entry.get("CHECKPOINT_VALIDITY"), "EXIT_CODE": entry.get("EXIT_CODE", "UNKNOWN"),
                   "episodes_complete": len(subset), "episodes_expected": 30,
                   "execution_complete": len(subset) == 30 and not any(error.get("index") == index for error in errors),
                   "mean_return": statistics.mean(values) if values else None,
                   "median_return": statistics.median(values) if values else None,
                   "sample_sd_ddof_1": statistics.stdev(values) if len(values) > 1 else None,
                   "min_return": min(values) if values else None, "max_return": max(values) if values else None,
                   "mean_episode_length": statistics.mean(row["EPISODE_LENGTH"] for row in subset) if subset else None,
                   "task_success_count": sum(known_success), "task_success_known_episodes": len(known_success),
                   "task_success_unknown_episodes": len(subset) - len(known_success),
                   "task_success_fraction_among_exposed": sum(known_success) / len(known_success) if known_success else None,
                   "success_subtasks_mean_among_exposed": statistics.mean(known_subtasks) if known_subtasks else None,
                   "success_subtasks_distribution": {str(count): known_subtasks.count(count) for count in range(6)},
                   "success_subtasks_unknown_episodes": len(subset) - len(known_subtasks),
                   "five_packages_completion_count": sum(flag is True for flag in five_flags),
                   "five_packages_completion_unknown_episodes": sum(flag is None for flag in five_flags)}
        if values:
            import numpy as np
            array = np.asarray(values, dtype=np.float64)
            draws = np.random.default_rng(20260929 + index).choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
            summary["descriptive_mean_bootstrap_ci95"] = np.quantile(draws, [0.025, 0.975], method="linear").tolist()
        else:
            summary["descriptive_mean_bootstrap_ci95"] = None
        checkpoint_summaries.append(summary)

    method_summaries = []
    for method in METHODS:
        subset = [summary for summary in checkpoint_summaries if summary["METHOD"] == method]
        complete = all(summary["execution_complete"] for summary in subset)
        means = [summary["mean_return"] for summary in subset]
        success_fractions = [summary["task_success_fraction_among_exposed"] for summary in subset]
        subtask_means = [summary["success_subtasks_mean_among_exposed"] for summary in subset]
        method_summaries.append({"METHOD": method, "independent_training_replicates": 3,
            "aggregation": "equal weights over train seeds 0/1/2; 90 evaluation episodes are not 90 training replicates",
            "episodes_complete": sum(summary["episodes_complete"] for summary in subset),
            "execution_complete": complete,
            "checkpoint_results": [{"TRAIN_SEED": summary["TRAIN_SEED"], "mean_return": summary["mean_return"],
                                    "task_success_count": summary["task_success_count"],
                                    "task_success_known_episodes": summary["task_success_known_episodes"],
                                    "success_subtasks_mean_among_exposed": summary["success_subtasks_mean_among_exposed"]}
                                   for summary in subset],
            "equal_weight_mean_return": statistics.mean(means) if complete else None,
            "median_of_three_checkpoint_means": statistics.median(means) if complete else None,
            "training_seed_sample_sd": statistics.stdev(means) if complete else None,
            "equal_weight_native_success_fraction_among_exposed": statistics.mean(success_fractions) if complete and all(value is not None for value in success_fractions) else None,
            "equal_weight_native_subtask_mean_among_exposed": statistics.mean(subtask_means) if complete and all(value is not None for value in subtask_means) else None,
            "task_success_count": sum(summary["task_success_count"] for summary in subset),
            "task_success_unknown_episodes": sum(summary["task_success_unknown_episodes"] for summary in subset),
            "five_packages_completion_count": sum(summary["five_packages_completion_count"] for summary in subset)})

    all_complete = valid_count(data) == 9 and len(rows) == 270 and all(summary["execution_complete"] for summary in checkpoint_summaries)
    status = "TRUCK_ORIGINAL_RETURN_COMPLETE" if all_complete else "TRUCK_ORIGINAL_RETURN_EXECUTION_FAILED" if errors else "TRUCK_EXP1_INCOMPLETE"
    confirmed_exit = sum(type(entry.get("EXIT_CODE")) is int for entry in data["jobs"])
    blockers = [f"{entry['METHOD']} seed {entry['SEED']}: {entry.get('REMAINING_ACTION', 'terminal checkpoint not valid')}"
                for entry in data["jobs"] if entry.get("CHECKPOINT_VALIDITY") != "VALID"]
    blockers += [f"Evaluation worker {error.get('index')}: exit {error.get('exit_code')}; log {error.get('log')}" for error in errors]
    missing = 270 - len(rows)
    if not all_complete and not blockers and missing:
        blockers.append(f"Original-return evaluation pending: {missing} episodes")
    metadata = {"task": TASK, "status": status, "updated_at_utc": now(),
                "protocol_sha256": data["protocol_hash"], "manifest_sha256": data["manifest_hash"],
                "WINDOW_TRAINING_COMPLETE": "9/9", "WINDOW_RETURN_EVAL_COMPLETE": "270/270",
                "TRUCK_TERMINAL_CHECKPOINTS_VALIDATED": f"{valid_count(data)}/9",
                "TRUCK_RUNS_WITH_CONFIRMED_EXIT": f"{confirmed_exit}/9",
                "TRUCK_RETURN_EPISODES_COMPLETE": f"{len(rows)}/270",
                "BLOCKER": blockers or None, "FORCE_WORK_PERFORMED": False,
                "NEXT_STEP": "STOP" if all_complete else "Complete missing Truck training/evaluation only"}
    atomic_json(data["out"] / "checkpoint_summary.json", {**metadata, "checkpoints": checkpoint_summaries,
                "runtime_errors": error_history, "active_runtime_errors": errors})
    atomic_json(data["out"] / "method_summary.json", {**metadata, "methods": method_summaries})
    report = ["# Truck EXP-1 original-return report", "", f"Status: **{status}**", "",
              "Window training 9/9 and return evaluation 270/270 were already completed and are retained; this adapter performs only the frozen Truck task.",
              "The Window→Truck design revision uses native five-package success/completion fields and continuous original return. No Window return >=650 gate is applied.", "",
              "Independent training replicates: n=3 per method. Thirty episodes per checkpoint use the same environment seeds 1000–1029. Within-checkpoint bootstrap intervals are descriptive episode-resampling intervals, not independent-training uncertainty estimates.",
              "Native termination is recorded directly; Truck can terminate on native task completion. Termination is not relabeled as robot/tool failure.",
              "Missing native success/subtask fields remain UNKNOWN; low return or low success remains a scientific result, not an infrastructure failure.", "",
              "| Method | Seed | Terminal step / target | Checkpoint validity | Training exit | Episodes | Mean return | Median | Sample SD | Min / max | Descriptive mean CI95 | Native success / exposed | Subtasks mean | Five-package completions |",
              "|---|---:|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|"]
    for summary in checkpoint_summaries:
        report.append(f"| {summary['METHOD']} | {summary['TRAIN_SEED']} | {summary['TERMINAL_STEP']} / {summary['TARGET_STEP']} | {summary['CHECKPOINT_VALIDITY']} | {summary['EXIT_CODE']} | {summary['episodes_complete']}/30 | {summary['mean_return']} | {summary['median_return']} | {summary['sample_sd_ddof_1']} | {summary['min_return']} / {summary['max_return']} | {summary['descriptive_mean_bootstrap_ci95']} | {summary['task_success_count']} / {summary['task_success_known_episodes']} (unknown {summary['task_success_unknown_episodes']}) | {summary['success_subtasks_mean_among_exposed']} | {summary['five_packages_completion_count']} |")
    report += ["", "| Method | Seed 0/1/2 checkpoint means | Equal-weight mean return | Median checkpoint mean | Equal-weight exposed native success fraction | Equal-weight exposed subtask mean |", "|---|---|---:|---:|---:|---:|"]
    for summary in method_summaries:
        report.append(f"| {summary['METHOD']} | {[item['mean_return'] for item in summary['checkpoint_results']]} | {summary['equal_weight_mean_return']} | {summary['median_of_three_checkpoint_means']} | {summary['equal_weight_native_success_fraction_among_exposed']} | {summary['equal_weight_native_subtask_mean_among_exposed']} |")
    report += ["", "Final status fields:", ""]
    for key in ("WINDOW_TRAINING_COMPLETE", "WINDOW_RETURN_EVAL_COMPLETE", "TRUCK_TERMINAL_CHECKPOINTS_VALIDATED", "TRUCK_RUNS_WITH_CONFIRMED_EXIT", "TRUCK_RETURN_EPISODES_COMPLETE"):
        report.append(f"- {key} = {metadata[key]}")
    report += ["- METHOD_RETURN_AND_SUCCESS_SUMMARY = method_summary.json (three training seeds, equally weighted)",
               f"- BLOCKER = {metadata['BLOCKER']}", "- FORCE_WORK_PERFORMED = FALSE", f"- NEXT_STEP = {metadata['NEXT_STEP']}", "",
               "Unknown training exit codes remain UNKNOWN and do not invalidate an otherwise evidenced frozen full-budget terminal policy. No exit code was reconstructed or invented.",
               "Checkpoint files, exact steps, native terminal receipts, hashes and missing evidence are listed in terminal_inventory.json. Native final subtask-count distributions and unknown counts are retained in checkpoint_summary.json.",
               "This completes only EXP-1. It makes no determination about ranking reversal. EXP-2, EXP-3 nominal, EXP-4 held-out and analysis remain outside this execution. Native Truck success is not a force-comparison eligibility gate.",
               "No force data was queried or saved; no nIAE_F, reference-force design, force comparison, held-out-force scenario, or new algorithm was performed.", ""]
    if error_history:
        report += ["Preserved evaluation runtime-error history (no automatic retries):", "",
                   *[f"- {json.dumps(error, ensure_ascii=False)}" for error in error_history], ""]
    (data["out"] / "TRUCK_EXP1_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("SUMMARY", status, "terminals", valid_count(data), "episodes", len(rows), flush=True)
    return all_complete


def run(root, manifest_path=None):
    data = inputs(root, manifest_path)
    require_ready(data)
    errors = []
    for index, entry in enumerate(data["jobs"]):
        rows = read_rows(data)
        done = [row for row in rows if row["METHOD"] == entry["METHOD"] and row["TRAIN_SEED"] == entry["SEED"]]
        if len(done) == 30:
            continue
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log = data["runtime"] / f"{index}-{KEYS[entry['METHOD']]}-{entry['SEED']}-{stamp}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--root", str(root),
                   "--index", str(index), "--manifest", str(data["manifest_path"])]
        print("CHECKPOINT_START", index, entry["METHOD"], entry["SEED"], str(log), flush=True)
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            print("WORKER_PID", process.pid, flush=True)
            code = process.wait()
        print("CHECKPOINT_EXIT", index, code, flush=True)
        if code != 0:
            errors.append({"index": index, "METHOD": entry["METHOD"], "TRAIN_SEED": entry["SEED"],
                           "exit_code": code, "log": str(log), "automatic_retries": 0})
            break
    complete = summarize(root, manifest_path, errors)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    out, runtime = paths(root)
    runtime.mkdir(parents=True, exist_ok=True)
    # Linux advisory locks prevent overlapping coordinators or manual workers;
    # importing the module or checking Python syntax does not require fcntl.
    import fcntl
    lock_name = "worker.lock" if args.command == "worker" else "coordinator.lock"
    with (runtime / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.command == "worker":
                if args.index is None or not 0 <= args.index < 9:
                    parser.error("worker requires --index 0..8")
                evaluate_checkpoint(root, args.index, args.manifest)
                return 0
            if args.command == "run":
                return run(root, args.manifest)
            data = inputs(root, args.manifest)
            read_rows(data)  # Invalid prior records always block; never overwrite.
            if args.command == "prepare":
                raw_path = out / "original_return_raw.jsonl"
                if not raw_path.exists():
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with raw_path.open("x", encoding="utf-8"):
                        pass
            complete = summarize(root, args.manifest)
            if args.command == "prepare":
                print("EVALUATION_READY", valid_count(data) == 9, "AUTOMATIC_RETRIES", 0, flush=True)
            return 0 if args.command == "prepare" or complete else 2
        except Exception:
            traceback.print_exc()
            print("STOP: preserved existing scientific records; automatic_retries=0", flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
