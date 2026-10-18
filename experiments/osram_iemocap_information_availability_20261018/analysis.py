"""Strict macro-fold summaries for frozen IEMOCAP representation probes.

Metrics use percentage units, except CE. No prediction, training or tensor
dependency is needed here. Cross-counts are summed over disjoint test folds,
then averaged over rates; repeated masks/rates are never unique utterances.
"""

from __future__ import annotations

import copy
import math
import statistics


# Same six real/control pairs as the original MELD information probe.
PROBE_PAIRS = {
    "Base": "BaseDonor", "Memory": "MemoryDonor",
    "LocalBase": "LocalBaseDonor", "LocalMemory": "LocalMemoryDonor",
    "RawHistory": "RawHistoryDonor", "LocalRawHistory": "LocalRawHistoryDonor",
}
VARIANTS = ("Local", *[name for pair in PROBE_PAIRS.items() for name in pair])
DATASETS = ("IEMOCAPFour", "IEMOCAPSix")
SEEDS = (66, 67, 68)
RATES = tuple(f"{i / 10:.1f}" for i in range(8))
METRICS = ("ACC", "UA", "WF1", "CE")
COUNTS = ("N", "corrections", "harms", "both_correct", "both_wrong")


def _summary(values):
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("summary requires nonempty finite values")
    return {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values}


def _validated_grid(folds):
    grid = {}
    for fold_summary in folds:
        dataset = fold_summary.get("dataset")
        fold = fold_summary.get("fold")
        if dataset not in DATASETS or fold not in range(1, 6) or fold_summary.get("status") != "complete":
            raise ValueError("expected complete IEMOCAPFour/Six fold 1..5")
        if (dataset, fold) in grid:
            raise ValueError(f"duplicate fold: {dataset}/{fold}")
        rates = fold_summary.get("rates", {})
        if set(rates) != set(RATES):
            raise ValueError(f"incomplete or noncanonical rate grid: {dataset}/{fold}")
        rows = {}
        for rate, cell in rates.items():
            if cell.get("status") != "complete":
                raise ValueError(f"incomplete rate: {dataset}/{fold}/{rate}")
            for fit in cell.get("fits", []):
                seed, variant = fit.get("seed"), fit.get("variant")
                key = (rate, seed, variant)
                if seed not in SEEDS or variant not in VARIANTS or key in rows:
                    raise ValueError(f"invalid or duplicate probe fit: {dataset}/{fold}/{key}")
                metrics = fit.get("metrics", {})
                for metric in METRICS:
                    value = float(metrics.get(metric, math.nan))
                    if not math.isfinite(value) or value < 0 or (metric != "CE" and value > 100):
                        raise ValueError(f"invalid {metric}: {key}")
                n = metrics.get("N")
                if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                    raise ValueError(f"invalid N: {key}")
                for field in ("cross_vs_local", "cross_vs_donor"):
                    cross = fit.get(field)
                    if cross is None:
                        if field == "cross_vs_local":
                            raise ValueError(f"missing aligned {field}: {key}")
                        continue
                    if any(isinstance(cross.get(k), bool) or not isinstance(cross.get(k), int)
                           or cross[k] < 0 for k in COUNTS):
                        raise ValueError(f"invalid {field} counts: {key}")
                    if cross["N"] != sum(cross[k] for k in COUNTS[1:]):
                        raise ValueError(f"{field} four-cell counts do not sum to N: {key}")
                    if cross["N"] > n:
                        raise ValueError(f"{field} cohort exceeds metric cohort: {key}")
                rows[key] = fit
            expected = {(rate, seed, name) for seed in SEEDS for name in VARIANTS}
            if {key for key in rows if key[0] == rate} != expected:
                raise ValueError(f"incomplete probe grid: {dataset}/{fold}/{rate}")
            for seed in SEEDS:
                for real, donor in PROBE_PAIRS.items():
                    if rows[(rate, seed, real)]["metrics"]["N"] != rows[(rate, seed, donor)]["metrics"]["N"]:
                        raise ValueError(f"real/donor cohort mismatch: {dataset}/{fold}/{rate}/{real}")
        grid[(dataset, fold)] = rows
    if set(grid) != {(dataset, fold) for dataset in DATASETS for fold in range(1, 6)}:
        raise ValueError("all five folds of both IEMOCAP variants are required")
    return grid


def _metric_summary(values):
    # values[(seed, rate, metric)] is already the equal-weight mean of five folds.
    def across(rates):
        return {metric: _summary(statistics.mean(values[(seed, rate, metric)] for rate in rates)
                                 for seed in SEEDS) for metric in METRICS}
    return {"per_rate": {rate: {metric: _summary(values[(seed, rate, metric)] for seed in SEEDS)
                                for metric in METRICS} for rate in RATES},
            "eight_rate": across(RATES), "high_missing": across(("0.5", "0.6", "0.7"))}


def _cross_summary(grid, dataset, variant, field):
    cells = {}
    for seed in SEEDS:
        for rate in RATES:
            rows = [grid[(dataset, fold)][(rate, seed, variant)].get(field) for fold in range(1, 6)]
            if any(row is None for row in rows):
                if all(grid[(dataset, fold)][(r, s, variant)].get(field) is None
                       for fold in range(1, 6) for r in RATES for s in SEEDS):
                    return None
                raise ValueError(f"partially missing {field}: {dataset}/{variant}")
            cells[(seed, rate)] = {key: sum(row[key] for row in rows) for key in COUNTS}

    def across(rates):
        return {key: _summary(statistics.mean(cells[(seed, rate)][key] for rate in rates)
                              for seed in SEEDS) for key in COUNTS}
    return {"per_rate": {rate: {key: _summary(cells[(seed, rate)][key] for seed in SEEDS)
                                for key in COUNTS} for rate in RATES},
            "eight_rate": across(RATES), "high_missing": across(("0.5", "0.6", "0.7")),
            "event_totals_by_probe_seed": {str(seed): {key: sum(cells[(seed, rate)][key] for rate in RATES)
                                                       for key in COUNTS} for seed in SEEDS},
            "count_semantics": "sum disjoint folds, mean rates, then mean probe initializations; totals are repeated rate-events"}


def aggregate(folds):
    """Validate 10 fold summaries and return JSON-ready complete probe results."""
    grid = _validated_grid(folds)
    datasets = {}
    for dataset in DATASETS:
        variants = {}
        for variant in VARIANTS:
            values = {(seed, rate, metric): statistics.mean(
                float(grid[(dataset, fold)][(rate, seed, variant)]["metrics"][metric])
                for fold in range(1, 6))
                for seed in SEEDS for rate in RATES for metric in METRICS}
            variants[variant] = _metric_summary(values)
        paired = {}
        for real, donor in PROBE_PAIRS.items():
            values = {(seed, rate, metric): statistics.mean(
                float(grid[(dataset, fold)][(rate, seed, real)]["metrics"][metric])
                - float(grid[(dataset, fold)][(rate, seed, donor)]["metrics"][metric])
                for fold in range(1, 6))
                for seed in SEEDS for rate in RATES for metric in METRICS}
            paired[real] = {"control": donor, **_metric_summary(values)}
        cross_local = {variant: _cross_summary(grid, dataset, variant, "cross_vs_local")
                       for variant in VARIANTS}
        cross_donor = {variant: _cross_summary(grid, dataset, variant, "cross_vs_donor")
                       for variant in PROBE_PAIRS}
        datasets[dataset] = {"variants": variants, "paired_deltas": paired,
                             "cross_vs_local": cross_local,
                             "cross_vs_donor": {key: value for key, value in cross_donor.items() if value is not None}}
    return {"status": "complete", "label": "INTERNAL DIAGNOSTIC ONLY", "backbone_seed": 66,
            "probe_seeds": list(SEEDS), "mask_protocol": "gcnet-random-mask-v1",
            "aggregation": "equal-weight five-fold mean, equal-weight rates, then mean and sample std across three probe initializations",
            "units": {"ACC": "percent", "UA": "percent", "WF1": "percent", "CE": "raw"},
            "datasets": datasets, "fold_summaries": copy.deepcopy(folds)}


def _formatted(summary):
    return f"{summary['mean']:.3f} ± {summary['sample_std']:.3f}"


def render_markdown(report):
    """Render complete result tables without interpreting probe scores causally."""
    lines = ["# IEMOCAP Frozen Information Availability Probe", "", "**INTERNAL DIAGNOSTIC ONLY**", "",
             "主干 seed66；五折、八率、gcnet-random-mask-v1。不重新训练主干，仅训练离线分类 Probe。",
             "Probe initialization = 66/67/68；三次 probe initialization 不是三种主干随机种子。",
             "主干及 Probe 使用 Test-oracle 选择；结果不是独立测试集论文成绩。每个 Probe 训练30 epochs。",
             "所有指标先对五折等权平均、再对八率等权平均，最后对三个 Probe 初始化给出均值及样本标准差。",
             "ACC/UA/W-F1 使用百分数，CE 使用原数。UA 是完整4/6类的 Macro Recall。",
             "Local 与联合输入头容量不完全匹配；每个 real/donor 配对使用相同输入维度和同容量分类头。",
             "真实/Donor 比较必须使用同一匹配 cohort；与 Local 的四格统计使用交集样本，不把未对齐样本相减。", ""]
    for dataset, data in report["datasets"].items():
        lines += [f"## {dataset}", "", "### Overall / high missing (.5/.6/.7)", "",
                  "| Probe | ACC | UA | W-F1 | high ACC | high UA | high W-F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, values in data["variants"].items():
            parts = [_formatted(values[section][metric])
                     for section in ("eight_rate", "high_missing") for metric in ("ACC", "UA", "WF1")]
            lines.append(f"| {name} | " + " | ".join(parts) + " |")
        lines += ["", "### Per-rate core probes", "",
                  "| Rate | Probe | ACC | UA | W-F1 | CE |", "|---|---|---:|---:|---:|---:|"]
        for rate in RATES:
            for name in ("Local", "LocalBase", "LocalMemory", "LocalMemoryDonor"):
                values = data["variants"][name]["per_rate"][rate]
                lines.append(f"| {rate} | {name} | " + " | ".join(_formatted(values[k]) for k in METRICS) + " |")
        lines += ["", "### Real minus donor (percentage points; CE raw difference)", "",
                  "| Real | Donor | ΔACC | ΔUA | ΔW-F1 | ΔCE | high ΔACC | high ΔUA |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for real, values in data["paired_deltas"].items():
            parts = [_formatted(values["eight_rate"][key]) for key in METRICS]
            parts += [_formatted(values["high_missing"][key]) for key in ("ACC", "UA")]
            lines.append(f"| {real} | {values['control']} | " + " | ".join(parts) + " |")
        for field, title, names in (("cross_vs_local", "Compared with Local", ("LocalBase", "LocalMemory")),
                                    ("cross_vs_donor", "Real compared with donor", tuple(PROBE_PAIRS))):
            lines += ["", f"### {title}: corrections / harms", "",
                      "Counts sum five disjoint folds, then average rates and Probe initializations; not unique-utterance totals.", "",
                      "| Probe | N | corrections | harms | both correct | both wrong |",
                      "|---|---:|---:|---:|---:|---:|"]
            for name in names:
                values = data[field].get(name)
                if values is not None:
                    lines.append(f"| {name} | " + " | ".join(_formatted(values["eight_rate"][key]) for key in COUNTS) + " |")
    lines += ["", "## Interpretation limits", "",
              "可解码性不等于原任务头实际利用。Real 超过 Donor 仅支持这一 Probe 类别下的真实历史预测价值。",
              "LocalMemory 是否超过 Local、纠错是否超过致错必须按实际结果判断；不能据此直接断言写入或读取失败。",
              "机器可读结果保留所有 fold_summaries（含原始指标、逐类 recall/support、四格计数及来源引用）；不合并不同协议结果。", ""]
    return "\n".join(lines)
