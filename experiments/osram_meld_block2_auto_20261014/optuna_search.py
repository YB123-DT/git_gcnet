"""Two-candidate, finite TPE optimization of real paired worker results."""
from __future__ import annotations

import math
from pathlib import Path

from .methods import build_config
from .scheduler import write_json


INITIAL_CONFIGS = (
    {"lr": 1e-3, "weight_decay": 1e-5, "dropout": .5},
    {"lr": 3e-4, "weight_decay": 1e-4, "dropout": .3},
    {"lr": 2e-3, "weight_decay": 1e-6, "dropout": .2},
    {"lr": 7e-4, "weight_decay": 1e-3, "dropout": .6},
    {"lr": 1.5e-3, "weight_decay": 1e-4, "dropout": .4},
)


def paired_objective(summary, flat_paired):
    if summary.get("status") != "complete":
        raise ValueError("unfinished workers cannot produce an HPO objective")
    paired = summary["paired"]
    if not flat_paired.get("mask_digest") or paired.get("mask_digest") != flat_paired["mask_digest"]:
        raise ValueError("paired comparison requires identical common mask digest")
    def metrics(value):
        if "on" in value:
            value = value["on"]["mean"]
        return [float(value[key] if key in value else value[key.upper()])
                for key in ("acc", "ua")]
    values = [candidate - baseline for candidate, baseline in
              zip(metrics(paired), metrics(flat_paired))]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired objective must be finite")
    return .6 * values[0] + .4 * values[1]


def run_search(root, selected_methods, base_config, flat_paired, scheduler,
               trials_per_model=20, top_k=2):
    """Persist all trials. First five finish singly; later batches hold <=2."""
    methods = list(selected_methods)
    if not 1 <= top_k <= 2 or len(methods) > top_k or len(set(methods)) != len(methods):
        raise ValueError("HPO is restricted to at most two selected methods")
    if not 16 <= trials_per_model <= 20:
        raise ValueError("finite HPO budget must be 16 through 20 trials per method")
    if not methods:
        return []
    if base_config.get("epochs", 50) != 50 or base_config.get("seed", 66) != 66:
        raise ValueError("HPO keeps 50 epochs and seed66")
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Optuna is required in the authorized training environment") from error
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_records = []
    for method in methods:
        if not method or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in method):
            raise ValueError("method must be a safe identifier")
        study = optuna.create_study(study_name=f"meld_block2_{method}_seed66",
            storage=f"sqlite:///{root / (method + '.sqlite3')}", direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=66, n_startup_trials=5), load_if_exists=True)
        protocol = {"base_config": base_config, "flat_paired": flat_paired,
                    "trials_per_model": trials_per_model, "initial_configs": list(INITIAL_CONFIGS)}
        if study.user_attrs.get("protocol", protocol) != protocol:
            raise ValueError("persisted Optuna study protocol differs from this invocation")
        study.set_user_attr("protocol", protocol)
        if not study.trials:
            for config in INITIAL_CONFIGS:
                study.enqueue_trial(config)

        def job(trial):
            identifier = f"hpo_{method}_{trial.number:03d}"
            output = str(root / "runs" / identifier)
            config = build_config(base_config, method, output, overrides={
                "learning_rate": trial.params["lr"],
                "weight_decay": trial.params["weight_decay"],
                "dropout": trial.params["dropout"]}, epochs=50)
            return {"id": identifier, "method": method, "config": config,
                    "output": output}

        def execute(trials):
            records = scheduler.run_jobs([job(trial) for trial in trials])
            for trial, record in zip(trials, records):
                study._storage.set_trial_user_attr(trial._trial_id, "queue_record", record)
                try:
                    value = paired_objective(record["summary"], flat_paired)
                    if record["status"] != "complete":
                        raise ValueError("worker failed")
                except (KeyError, ValueError, TypeError) as error:
                    study._storage.set_trial_user_attr(trial._trial_id, "failure", str(error))
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial.number, value)

        # RUNNING trials already consume budget. Queue identity recovers their
        # existing process or completed summary instead of submitting a duplicate.
        running = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.RUNNING:
                continue
            if set(trial.params) != {"lr", "weight_decay", "dropout"}:
                study._storage.set_trial_user_attr(trial._trial_id, "failure",
                    "interrupted before all hyperparameters were persisted; no worker submitted")
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            else:
                running.append(trial)
        for start in range(0, len(running), 2):
            execute(running[start:start + 2])
        while True:
            issued = [trial for trial in study.trials if trial.state != optuna.trial.TrialState.WAITING]
            remaining = trials_per_model - len(issued)
            if remaining <= 0:
                break
            width = 1 if len(issued) < 5 else min(2, remaining)
            trials = []
            for _ in range(width):
                trial = study.ask()
                trial.suggest_float("lr", 3e-4, 2e-3, log=True)
                trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                trial.suggest_float("dropout", .2, .6)
                trials.append(trial)
            execute(trials)
        rows = [{"method": method, "number": t.number, "state": t.state.name,
                 "params": t.params, "value": t.value, "user_attrs": t.user_attrs}
                for t in study.trials if t.state != optuna.trial.TrialState.WAITING]
        complete = sorted((r for r in rows if r["state"] == "COMPLETE"),
                          key=lambda r: (-r["value"], r["number"]))
        write_json(root / f"{method}_TRIALS.json", {"method": method, "trials": rows,
            "best": complete[0] if complete else None, "budget": trials_per_model})
        all_records.extend(rows)
    write_json(root / "SEARCH.json", {"selected_methods": methods, "trials": all_records,
                                      "status": "complete"})
    return all_records
