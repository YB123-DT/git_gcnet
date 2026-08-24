# GCNet MOSI Full-Fused Reconstruction 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将已验证的 GCNet Full-Fused runner 最小参数化到 CMUMOSI，复用既有 baseline，只新增并完成 80 个正式任务及坍塌敏感性分析。

**架构：** 保留单一 runner 和现有锁、恢复、配对审计逻辑。每个 job 显式携带 dataset；dataset 决定训练命令、预期 fold 和 baseline 根目录。默认 IEMOCAPSix 行为由回归测试锁定，CMUMOSI 仅增加数据集配置，不改变 GCNet 或 Full-Fused loss。

**技术栈：** Python 3.8/3.10、PyTorch、pytest、现有 GCNet official protocol、biggpu V100。

---

## 文件边界

- 修改 `scripts/run_iemocap6_full_fused_sweep.py`：增加受限 dataset 配置并消除 fold5 硬编码。
- 修改 `tests/test_iemocap6_full_fused_runner.py`：新增 MOSI matrix、command、manifest、baseline-root 测试，并保留 IEMOCAP 回归测试。
- 创建 `docs/experiments/2026-08-24-mosi-full-fused-reconstruction.md`：记录命令、完整性、原始结果和坍塌敏感性分析。

### 任务 1：用失败测试定义 MOSI runner 行为

**文件：**
- 修改：`tests/test_iemocap6_full_fused_runner.py`
- 测试：`scripts/run_iemocap6_full_fused_sweep.py`

- [ ] **步骤 1：编写 CMUMOSI 80-job matrix 测试**

```python
def test_mosi_matrix_has_80_full_fused_fold1_jobs(tmp_path):
    jobs = sweep.build_jobs(
        output_root=tmp_path / "out",
        baseline_root=tmp_path / "baseline",
        python="python",
        dataset="CMUMOSI",
        gpus=(0, 1, 2),
    )
    assert len(jobs) == 80
    assert {job.dataset for job in jobs} == {"CMUMOSI"}
    assert {job.condition for job in jobs} == {"full_fused"}
    assert all("--fold" not in job.command for job in jobs)
    assert all(job.command[job.command.index("--dataset") + 1] == "CMUMOSI" for job in jobs)
```

- [ ] **步骤 2：编写默认 IEMOCAPSix 不变测试**

```python
def test_default_dataset_remains_iemocap_six(tmp_path):
    jobs = sweep.build_jobs(tmp_path, python="python")
    assert len(jobs) == 80
    assert {job.dataset for job in jobs} == {"IEMOCAPSix"}
    assert all(job.command[job.command.index("--fold") + 1] == "5" for job in jobs)
```

- [ ] **步骤 3：编写 fold1 manifest 与错误 dataset 测试**

```python
def test_mosi_completion_requires_fold1_manifest(tmp_path):
    job = make_job(tmp_path, dataset="CMUMOSI")
    manifest = manifest_for(job, fold=1)
    assert sweep._manifest_matches_job(manifest, job)
    manifest["run"]["fold"] = 5
    assert not sweep._manifest_matches_job(manifest, job)

def test_runner_rejects_unsupported_dataset(tmp_path):
    with pytest.raises(ValueError, match="unsupported dataset"):
        sweep.build_jobs(tmp_path, python="python", dataset="CMUMOSEI")
```

- [ ] **步骤 4：运行 RED 测试**

运行：

```bash
pytest -q tests/test_iemocap6_full_fused_runner.py -k 'mosi or default_dataset or unsupported_dataset'
```

预期：因 `dataset` 参数和 `job.dataset` 尚不存在而 FAIL。

### 任务 2：最小实现 dataset 参数化

**文件：**
- 修改：`scripts/run_iemocap6_full_fused_sweep.py`
- 修改：`tests/test_iemocap6_full_fused_runner.py`

- [ ] **步骤 1：增加显式配置和 job dataset 字段**

```python
SUPPORTED_DATASETS = {"IEMOCAPSix": 5, "CMUMOSI": 1}

@dataclass(frozen=True)
class FullFusedJob:
    dataset: str
    # existing fields remain unchanged

def _expected_fold(dataset: str) -> int:
    try:
        return SUPPORTED_DATASETS[dataset]
    except KeyError as error:
        raise ValueError("unsupported dataset: {}".format(dataset)) from error
```

- [ ] **步骤 2：参数化 command、matrix 和 identity**

`_training_command()` 接收 dataset；仅当 `_expected_fold(dataset) == 5`
时追加 `--fold 5`。`build_jobs()` 新增默认参数 `dataset="IEMOCAPSix"`，每个
job 保存 dataset，identity 使用 `job.dataset`。

- [ ] **步骤 3：参数化 manifest 和 baseline 验证**

`_latest_manifest()` 使用 `run_manifest_fold_*.json`；两条 manifest matcher
同时验证 `job.dataset` 和 `_expected_fold(job.dataset)`。完成错误信息从
`fold5 evidence` 改成 `matching fold evidence`。

- [ ] **步骤 4：增加 CLI**

```python
parser.add_argument(
    "--dataset", choices=tuple(SUPPORTED_DATASETS), default="IEMOCAPSix"
)
```

CLI 将 `args.dataset` 传给 `build_jobs()`。MOSI 正式命令显式提供
`--baseline-root`，不依赖 IEMOCAP 默认路径。

- [ ] **步骤 5：运行 GREEN 和全套测试**

```bash
pytest -q tests/test_iemocap6_full_fused_runner.py
pytest -q tests
```

预期：全部 PASS。

- [ ] **步骤 6：提交实现**

提交必须说明默认 IEMOCAP 行为保持不变、仅支持 CMUMOSI 新路径以及测试证据。

### 任务 3：远程验证和 smoke

**文件：**
- 远程代码：`/data2/yb/paper/GCNet_full_fused_20260823/modality-jepa`
- Smoke：`/data2/yb/experiments/gcnet_mosi_full_fused_smoke_20260824`

- [ ] **步骤 1：同步冻结代码并运行远程测试**

```bash
/data2/yb/reproduction_envs/s0/bin/python3.10 -m pytest -q tests
```

- [ ] **步骤 2：只读预检 MOSI baseline 80/80**

baseline：
`/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/CMUMOSI`。

- [ ] **步骤 3：运行两任务 smoke**

```bash
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset \
/data2/yb/reproduction_envs/s0/bin/python3.10 -u \
scripts/run_iemocap6_full_fused_sweep.py \
  --dataset CMUMOSI \
  --output-root /data2/yb/experiments/gcnet_mosi_full_fused_smoke_20260824 \
  --baseline-root /data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/CMUMOSI \
  --gpus 0 --jobs-per-gpu 2 --rates 0.0,0.4 --seeds 66 --epochs 1 \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python
```

要求：2/2 jobs、2/2 audits、0 failure。

### 任务 4：80-run 正式实验与分析

**文件：**
- 结果：`/data2/yb/experiments/gcnet_mosi_full_fused_10seed_20260824`
- 创建：`docs/experiments/2026-08-24-mosi-full-fused-reconstruction.md`

- [ ] **步骤 1：检查 GPU 所有权后启动正式实验**

```bash
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset \
/data2/yb/reproduction_envs/s0/bin/python3.10 -u \
scripts/run_iemocap6_full_fused_sweep.py \
  --dataset CMUMOSI \
  --output-root /data2/yb/experiments/gcnet_mosi_full_fused_10seed_20260824 \
  --baseline-root /data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/CMUMOSI \
  --gpus 0,1,2 --jobs-per-gpu 3 --epochs 100 \
  --python /data2/yb/reproduction_envs/gcnet-official/bin/python
```

- [ ] **步骤 2：持续监控至完整**

要求 `complete_jobs=80`、`worker_errors=[]`、`paired_audits=80`、
`paired_audit_failures=0`，且 GPU4 未使用。

- [ ] **步骤 3：计算原始配对统计**

每个 missing rate 报告 baseline/Full-Fused mean ± sample SD、delta、wins、
paired Wilcoxon；八个 p 值做 Holm correction。

- [ ] **步骤 4：执行坍塌审计**

分别在每个 `(method, missing-rate)` 十种子组上应用注册严格规则和 MAD-only
规则。过滤表仅删除任一侧被标记的配对，并明确保留原始主表。

- [ ] **步骤 5：记录并提交结果**

报告必须包含环境、参数量、命令、目录、完整性证据、原始表、过滤表、统计
结论和局限性。
