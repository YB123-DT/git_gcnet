"""Report the finite MELD Block2 screen without mixing selection protocols.

ACC/UA/WF1 are percentages; all selection deltas are percentage points.
The preregistered screen remains fixed when later HPO records become available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path


METHODS = ("Flat", "B1", "B2", "B3", "B4", "B5", "B6")
RATES = tuple(f"{i / 10:.1f}" for i in range(8))
HIGH_RATES = ("0.5", "0.6", "0.7")
METRICS = ("ACC", "UA", "WF1", "CE")
EPS = 1e-10
SELECTION = {
    "comparison": "common-mask paired on, eight-rate arithmetic mean versus new Flat",
    "score": "0.6 * delta_ACC + 0.4 * delta_UA",
    "exclude_below_delta_ACC_pp": -0.30,
    "exclude_below_delta_UA_pp": -0.50,
    "signal": "score > 0 and (delta_ACC > 0 or delta_UA > 0)",
    "tie_break": "method ID ascending",
    "high_rates": list(HIGH_RATES),
}


def _index(reports):
    if isinstance(reports, dict):
        reports = reports.values()
    indexed = {}
    for row in reports:
        method = row.get("method")
        if method not in METHODS:
            raise ValueError(f"Unknown screen method: {method!r}")
        if method in indexed:
            raise ValueError(f"Duplicate screen method: {method}")
        indexed[method] = row
    return indexed


def _average(block, rates=RATES):
    return {key: sum(float(block["rates"][rate][key]) for rate in rates) / len(rates)
            for key in METRICS}


def _provenance(row):
    raw = row.get("provenance", {})
    return {
        "seed": raw.get("seed", row.get("config", {}).get("seed")),
        "code": raw.get("code_hashes", raw.get("code_sha256", raw.get("git_sha"))),
        "data": raw.get("dataset_sha256", raw.get("data_sha256")),
        "mask_protocol": raw.get("mask_protocol"),
    }


def _validate(row, flat):
    reasons = []
    provenance = _provenance(row)
    for key, value in provenance.items():
        if value is None or value == "" or value == {}:
            reasons.append(f"missing provenance {key}")
        elif flat and value != _provenance(flat)[key]:
            reasons.append(f"provenance mismatch: {key}")
    if provenance["seed"] != 66:
        reasons.append("screen seed must be 66")
    if provenance["mask_protocol"] != "gcnet-random-mask-v1":
        reasons.append("screen mask protocol must be gcnet-random-mask-v1")
    paired = row.get("paired", {})
    blocks = {"testbest": row.get("testbest", {}),
              "on": paired.get("on", {}), "off": paired.get("off", {})}
    for mode, block in blocks.items():
        rates = block.get("rates", {})
        if set(rates) != set(RATES):
            reasons.append(f"{mode}: exactly eight rates 0.0–0.7 required")
        for rate in RATES:
            metrics = rates.get(rate, {})
            for key in METRICS:
                value = metrics.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    reasons.append(f"{mode}/{rate}: missing or nonfinite {key}")
                elif key != "CE" and not 0 <= value <= 100:
                    reasons.append(f"{mode}/{rate}: {key} must be a percentage")
            n = metrics.get("N")
            support = metrics.get("support")
            recall = metrics.get("recall")
            if not isinstance(n, int) or n <= 0:
                reasons.append(f"{mode}/{rate}: positive integer N required")
            if (not isinstance(support, list) or len(support) != 7
                    or any(not isinstance(v, int) or v < 0 for v in support)
                    or sum(support) != n):
                reasons.append(f"{mode}/{rate}: seven-class support must sum to N")
            if (not isinstance(recall, list) or len(recall) != 7
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in recall)):
                reasons.append(f"{mode}/{rate}: seven finite class recalls required")
            if not metrics.get("mask_sha256"):
                reasons.append(f"{mode}/{rate}: mask_sha256 required")
            if not isinstance(metrics.get("epoch"), int):
                reasons.append(f"{mode}/{rate}: selected checkpoint epoch required")
            if mode in ("on", "off") and flat:
                reference = flat.get("paired", {}).get("on", {}).get("rates", {}).get(rate, {})
                for key in ("mask_sha256", "N", "support"):
                    if metrics.get(key) != reference.get(key):
                        reasons.append(f"{mode}/{rate}: common-mask Flat {key} mismatch")
    for rate in RATES:
        on = blocks["on"].get("rates", {}).get(rate, {})
        off = blocks["off"].get("rates", {}).get(rate, {})
        testbest = blocks["testbest"].get("rates", {}).get(rate, {})
        if on.get("epoch") != off.get("epoch") or on.get("epoch") != testbest.get("epoch"):
            reasons.append(f"{rate}: on/off/Test-best checkpoint epochs differ")
        if row["method"] == "Flat" and on != off:
            reasons.append(f"{rate}: Flat on/off must be identical")
        changes = paired.get("changes", {}).get(rate, {})
        counts = [changes.get(key) for key in ("corrections", "harms", "both_correct", "both_wrong")]
        if (any(not isinstance(v, int) or v < 0 for v in counts)
                or sum(counts) != on.get("N") or changes.get("N") != on.get("N")):
            reasons.append(f"{rate}: paired correctness counts must sum to N")
    return list(dict.fromkeys(reasons))


def pick_candidates(reports, top_k=2):
    """Apply the frozen gate; unfinished comparisons defer all HPO selection.

    Terminal failed methods are listed but excluded. Any malformed complete
    report invalidates comparison; waiting methods yield an incomplete result.
    """
    if not isinstance(top_k, int) or not 0 <= top_k <= 2:
        raise ValueError("top_k must be an integer between 0 and the registered maximum 2")
    indexed = _index(reports)
    flat = indexed.get("Flat")
    states = {}
    complete = []
    for method in METHODS:
        row = indexed.get(method)
        status = row.get("status", "unknown") if row else "missing"
        reasons = _validate(row, flat) if row and status == "complete" else []
        states[method] = {"status": status, "reasons": reasons}
        if status == "complete" and not reasons:
            complete.append(method)
    invalid = any(state["reasons"] for state in states.values())
    waiting = any(state["status"] not in ("complete", "failed", "cancelled") for state in states.values())
    if not flat or flat.get("status") != "complete":
        waiting = True
    means = {method: _average(indexed[method]["paired"]["on"]) for method in complete}
    rankings = []
    if "Flat" in means:
        for method in complete:
            if method == "Flat":
                continue
            da = means[method]["ACC"] - means["Flat"]["ACC"]
            du = means[method]["UA"] - means["Flat"]["UA"]
            score = .6 * da + .4 * du
            reasons = []
            if da < -.30 - EPS:
                reasons.append("delta_ACC < -0.30 pp")
            if du < -.50 - EPS:
                reasons.append("delta_UA < -0.50 pp")
            signal = score > 0 and (da > 0 or du > 0)
            rankings.append({"method": method, "delta_ACC": da, "delta_UA": du,
                             "score": score, "eligible": not reasons, "signal": signal,
                             "reasons": reasons, "mean": means[method]})
    rankings.sort(key=lambda row: (-round(row["score"], 10), row["method"]))
    pareto = [method for method in complete if not any(
        other != method and means[other]["ACC"] >= means[method]["ACC"] - EPS
        and means[other]["UA"] >= means[method]["UA"] - EPS
        and (means[other]["ACC"] > means[method]["ACC"] + EPS
             or means[other]["UA"] > means[method]["UA"] + EPS)
        for other in complete)]
    candidates = [row["method"] for row in rankings if row["eligible"] and row["signal"]]
    status = "invalid_comparison" if invalid else "incomplete" if waiting else (
        "ready_for_hpo" if candidates else "no_positive_signal")
    return {"status": status, "selected": candidates[:top_k] if status == "ready_for_hpo" else [],
            "selection_protocol": SELECTION, "rankings": rankings, "pareto": pareto,
            "methods": states, "complete_methods": complete}


def _load_reports(root):
    rows = []
    for method in METHODS:
        candidates = (root / "screen" / method / "SUMMARY.json", root / method / "SUMMARY.json")
        for path in candidates:
            if path.exists():
                try:
                    row = json.loads(path.read_text())
                    row.setdefault("method", method)
                    rows.append(row)
                except (OSError, ValueError) as exc:
                    rows.append({"method": method, "status": "failed", "error": str(exc)})
                break
    return rows


def _sqlite_trials(path):
    """Export every Optuna trial without importing Optuna or writing its DB."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"studies", "trials"} <= tables:
            return []
        studies = []
        for study in db.execute("SELECT * FROM studies ORDER BY study_id"):
            record = dict(study)
            record["trials"] = []
            for trial in db.execute("SELECT * FROM trials WHERE study_id=? ORDER BY number", (study["study_id"],)):
                item = dict(trial)
                for table in ("trial_params", "trial_values", "trial_user_attributes",
                              "trial_system_attributes", "trial_intermediate_values", "trial_heartbeats"):
                    if table in tables:
                        item[table] = [dict(v) for v in db.execute(
                            f"SELECT * FROM {table} WHERE trial_id=?", (trial["trial_id"],))]
                record["trials"].append(item)
            studies.append(record)
        return studies


def _load_optuna(root, supplied):
    if supplied is not None:
        return supplied
    path = root / "optuna_search.json"
    result = json.loads(path.read_text()) if path.exists() else {"studies": []}
    databases = sorted(set(root.rglob("*.sqlite")) | set(root.rglob("*.sqlite3")) | set(root.rglob("*.db")))
    if databases:
        result = {"result_json": result, "studies": result.get("studies", []) if isinstance(result, dict) else [],
                  "sqlite_exports": []}
        for database in databases:
            try:
                result["sqlite_exports"].append({"path": str(database), "studies": _sqlite_trials(database)})
            except sqlite3.Error as exc:
                result["sqlite_exports"].append({"path": str(database), "error": str(exc)})
    return result


def _hpo_records(optuna):
    records = {}
    if isinstance(optuna, list):
        studies = optuna
    elif isinstance(optuna, dict):
        studies = optuna.get("studies", optuna.get("results", []))
        if isinstance(studies, dict):
            studies = [dict(value, method=key) for key, value in studies.items()]
    else:
        return records
    for study in studies:
        method = study.get("method")
        if method in METHODS[1:]:
            best = study.get("best_config")
            records[method] = {"status": study.get("status", "reported (status not supplied)" if best is not None else "pending"),
                               "best_config": best, "best_trial": study.get("best_trial"),
                               "trial_count": len(study.get("trials", []))}
    return records


def _table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |")
    lines.append("")


def _format(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def _markdown(result):
    lines = ["# MELD Block2 finite screen", "",
             "INTERNAL EXPLORATORY: single-seed (66), 50 epochs, Test-best ACC oracle screening. "
             "Future validation-based selection and independent seeds are required before performance claims.", "",
             f"Screen status: **{result['status']}**. Selected: {', '.join(result['selected']) or 'none'}.", "",
             "The pre-registered gate uses common-mask paired eight-rate mean versus the new Flat. "
             "Exclude delta ACC < -0.30 pp or delta UA < -0.50 pp; inclusive boundaries. "
             "Score = 0.6 delta ACC + 0.4 delta UA. Positive score and at least one positive delta required. "
             "Select at most two; equal scores use method ID. HPO cannot change the initial gate.", "",
             "High-missing summary uses rates 0.5, 0.6 and 0.7. All UA/W-F1/CE and recalls use the "
             "ACC-selected checkpoint; no independent UA checkpoint selection or epoch mixing.", "",
             "Historical original Flat (three seeds: ACC 60.693, UA 36.255, W-F1 57.045) is reference only. "
             "It is not the paired comparator and is not assumed to share this run's code or masks.", "",
             "## Run status and resources", ""]
    _table(lines, ["Method", "Status", "parameters", "Train seconds", "Peak bytes", "Issues"], [
        [method, state["status"], result["reports"].get(method, {}).get("parameters", "—"),
         result["reports"].get(method, {}).get("totaltrainsec", "—"),
         result["reports"].get(method, {}).get("peakmemorybytes", "—"), "; ".join(state["reasons"])]
        for method, state in result["methods"].items()])
    lines += ["## Fixed screening ranking", ""]
    _table(lines, ["Method", "delta ACC pp", "delta UA pp", "Score", "Eligible", "Signal", "Reason"], [
        [r["method"], _format(r["delta_ACC"]), _format(r["delta_UA"]), _format(r["score"]),
         r["eligible"], r["signal"], "; ".join(r["reasons"])] for r in result["rankings"]])
    lines += ["Pareto frontier on paired mean ACC and UA (all valid complete methods including Flat; "
              "independent of the exclusion gate): " + (", ".join(result["pareto"]) or "none") + ".", ""]
    for label, accessor in (("Test-best ACC selected (winning-epoch masks)", lambda r: r["testbest"]),
                            ("Paired common-mask Block2 on", lambda r: r["paired"]["on"]),
                            ("Paired common-mask Block2 off", lambda r: r["paired"]["off"])):
        lines += ["## " + label, ""]
        rows = []
        for method in result["complete_methods"]:
            block = accessor(result["reports"][method])
            for rate, values in [(r, block["rates"][r]) for r in RATES] + [
                    ("mean", _average(block)), ("high", _average(block, HIGH_RATES))]:
                rows.append([method, rate] + [_format(values[key]) for key in METRICS] + [
                    values.get("epoch", "—"), values.get("N", "—"),
                    json.dumps(values.get("recall", [])), json.dumps(values.get("support", [])),
                    values.get("mask_sha256", "—")])
        _table(lines, ["Method", "Rate", *METRICS, "Epoch", "N", "Class recall", "Class support", "Mask SHA256"], rows)
    lines += ["## Block2 on/off paired correctness changes", ""]
    _table(lines, ["Method", "Rate", "Corrections", "Harms", "Both correct", "Both wrong", "N"], [
        [method, rate] + [result["reports"][method]["paired"]["changes"][rate][key]
                         for key in ("corrections", "harms", "both_correct", "both_wrong", "N")]
        for method in result["complete_methods"] for rate in RATES])
    lines += ["## Optuna and chosen configurations", ""]
    if result["hpo"]:
        _table(lines, ["Method", "Status", "Trial count", "Best configuration"], [
            [method, row["status"], row["trial_count"], json.dumps(row["best_config"], sort_keys=True)]
            for method, row in result["hpo"].items()])
    else:
        lines += ["HPO not started; no best configurations are available.", ""]
    lines += ["Full Optuna trial records (including failed/pruned/incomplete trials):", "", "```json",
              json.dumps(result["optuna"], indent=2, sort_keys=True), "```", "",
              "Full run configurations and provenance are retained in SUMMARY.json.", ""]
    return "\n".join(lines)


def summarize(root, top_k=2, reports=None, optuna_results=None):
    """Write SUMMARY.json, SUMMARY.csv and RESULT.md; return the JSON payload."""
    root = Path(root)
    indexed = _index(_load_reports(root) if reports is None else reports)
    result = pick_candidates(indexed, top_k=top_k)
    result["reports"] = indexed
    result["optuna"] = _load_optuna(root, optuna_results)
    result["hpo"] = _hpo_records(result["optuna"])
    for method in result["selected"]:
        result["hpo"].setdefault(method, {"status": "not started", "best_config": None,
                                          "best_trial": None, "trial_count": 0})
    result["warning"] = "single-seed exploratory Test-oracle screen; future validation required"
    root.mkdir(parents=True, exist_ok=True)
    (root / "SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "RESULT.md").write_text(_markdown(result))
    with (root / "SUMMARY.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "status", "mode", "rate", *METRICS, "epoch", "N", "recall", "support",
                         "mask_sha256", "parameters", "totaltrainsec", "peakmemorybytes"])
        for method in METHODS:
            row = indexed.get(method, {})
            if method not in result["complete_methods"]:
                writer.writerow([method, result["methods"][method]["status"]])
                continue
            for mode, block in (("testbest", row["testbest"]), ("paired_on", row["paired"]["on"]),
                                ("paired_off", row["paired"]["off"])):
                values = [(r, block["rates"][r]) for r in RATES] + [
                    ("mean", _average(block)), ("high", _average(block, HIGH_RATES))]
                for rate, metrics in values:
                    writer.writerow([method, "complete", mode, rate, *[metrics[k] for k in METRICS],
                                     metrics.get("epoch", ""), metrics.get("N", ""),
                                     json.dumps(metrics.get("recall", [])), json.dumps(metrics.get("support", [])),
                                     metrics.get("mask_sha256", ""), row.get("parameters", ""),
                                     row.get("totaltrainsec", ""), row.get("peakmemorybytes", "")])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    result = summarize(args.root, top_k=args.top_k)
    print(json.dumps({key: result[key] for key in ("status", "selected", "pareto")}))


if __name__ == "__main__":
    main()
