# GCNet 已完成版本目录整理实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 Original、MPFiLM、CP-LECC 和 Sequence-AFF 整理为一个共享运行环境、共享 GCNet 主干和四个薄版本目录，并只发布 IEMOCAPSix fixed fold 5 的已完成证据。

**架构：** 以 Sequence-AFF 完成提交 `60e8dd9` 为代码上界，排除其后的 BiLSTM、ReMasker、G1U/G1S、D0/D1、GenAgg 和 Soft Medoid。`common/gcnet` 保存共享执行代码，`versions/<name>` 保存替换模块与锁定配置，根 `run.py` 把版本配置翻译成现有 GCNet CLI。结果从已提交的 Markdown/JSON 生成紧凑、可追溯的 fold-5 目录，不复制大型训练产物。

**技术栈：** Python 3.8、PyTorch 1.8、PyG 2.0.1、标准库 `argparse/json/subprocess/pathlib`、`unittest`。

---

## 文件职责

- `common/config.py`：共享 GCNet 路径配置，保持官方字段。
- `common/gcnet/*.py`：四个版本共同的数据、mask、图、模型外壳、训练和损失代码。
- `versions/registry.py`：版本名、配置文件和模块路径的唯一注册表。
- `versions/*/variant.py`：各版本独有的模块；Original 为无参数 no-op 描述。
- `versions/*/config.json`：固定该版本相对共享 CLI 的唯一参数差异。
- `run.py`：选择版本、合并锁定参数和用户参数、调用共享训练入口。
- `results/iemocap6/fold5/*`：已完成结果与来源 manifest。
- `provenance/source_map.json`：新路径到历史 branch/commit/path 的绑定。
- `tests/test_organized_layout.py`：目录、版本注册、禁入项和 provenance 测试。
- `tests/test_version_runner.py`：CLI 参数合并、冲突拒绝和 Python 3.8 兼容测试。

### 任务 1：冻结完成代码边界并建立失败的目录契约

**文件：**
- 创建：`tests/test_organized_layout.py`
- 修改：Git 分支基线，仅保留 `60e8dd9` 及本设计/计划提交

- [ ] **步骤 1：将本地整理分支变基到完成代码上界**

运行：

```bash
git rebase --onto 60e8dd9 c677550 release/organized-completed-v1
```

预期：规格和计划提交被保留，`d23d90e` BiLSTM 与 `c677550` 后续工作不再是整理分支祖先。

- [ ] **步骤 2：编写目录契约失败测试**

```python
class OrganizedLayoutTest(unittest.TestCase):
    def test_only_completed_versions_are_published(self):
        root = Path(__file__).resolve().parents[1]
        names = {path.name for path in (root / "versions").iterdir()
                 if path.is_dir() and not path.name.startswith("__")}
        self.assertEqual(
            names,
            {"original", "mpfilm", "cp_lecc", "sequence_aff"},
        )

    def test_unfinished_surfaces_are_absent(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = {
            "remasker", "genagg", "medoid", "g1u", "g1s",
            "dilation", "bilstm_ablation", "pconv",
        }
        published = "\n".join(
            str(path.relative_to(root)).lower()
            for base in (root / "versions", root / "results")
            for path in base.rglob("*")
        )
        for token in forbidden:
            self.assertNotIn(token, published)
```

- [ ] **步骤 3：运行测试确认失败**

运行：

```bash
python -m unittest tests.test_organized_layout -v
```

预期：FAIL，原因是 `versions/` 和 `results/` 尚不存在。

- [ ] **步骤 4：提交红测**

```bash
git add tests/test_organized_layout.py
git commit -m "Define the completed-version publishing boundary"
```

### 任务 2：提取共享 GCNet 主干和四个版本模块

**文件：**
- 移动：`config.py` → `common/config.py`
- 移动：`gcnet/` → `common/gcnet/`
- 移动：`common/gcnet/mpfilm_rgcn.py` → `versions/mpfilm/variant.py`
- 移动：`common/gcnet/cp_lecc_rgcn.py` → `versions/cp_lecc/variant.py`
- 移动：`common/gcnet/sequence_aff.py` → `versions/sequence_aff/variant.py`
- 创建：`common/__init__.py`
- 创建：`common/gcnet/__init__.py`
- 创建：`versions/__init__.py`
- 创建：`versions/original/__init__.py`
- 创建：`versions/original/variant.py`
- 创建：`versions/mpfilm/__init__.py`
- 创建：`versions/cp_lecc/__init__.py`
- 创建：`versions/sequence_aff/__init__.py`
- 修改：`common/gcnet/model.py`

- [ ] **步骤 1：使用 `git mv` 建立目录所有权**

```bash
mkdir -p common versions/{original,mpfilm,cp_lecc,sequence_aff}
git mv config.py common/config.py
git mv gcnet common/gcnet
git mv common/gcnet/mpfilm_rgcn.py versions/mpfilm/variant.py
git mv common/gcnet/cp_lecc_rgcn.py versions/cp_lecc/variant.py
git mv common/gcnet/sequence_aff.py versions/sequence_aff/variant.py
```

- [ ] **步骤 2：修正共享模型导入**

`common/gcnet/model.py` 使用明确版本路径：

```python
from versions.cp_lecc.variant import CompletePreservingLowRankECCConv
from versions.mpfilm.variant import MissingPatternFiLMRGCNConv
from versions.sequence_aff.variant import MaskConditionedSequenceAFF
```

各 variant 对缺失模式编码统一导入：

```python
from common.gcnet.missing_patterns import encode_missing_patterns
```

- [ ] **步骤 3：实现 Original no-op variant**

```python
NAME = "original"
LOCKED_ARGUMENTS = {
    "graph_conv_variant": "original",
    "branch_fusion": "addition",
}

def describe():
    return {
        "name": NAME,
        "adds_parameters": False,
        "replacement": None,
    }
```

- [ ] **步骤 4：增加包初始化文件并运行导入测试**

运行：

```bash
python -m unittest \
  tests.test_mpfilm_rgcn \
  tests.test_cp_lecc_rgcn \
  tests.test_sequence_aff -v
```

预期：三个模块测试全部 PASS。

- [ ] **步骤 5：提交共享边界**

```bash
git add common versions tests
git commit -m "Separate completed variants from the shared GCNet runtime"
```

### 任务 3：实现统一版本注册与运行入口

**文件：**
- 创建：`versions/registry.py`
- 创建：`versions/*/config.json`
- 创建：`run.py`
- 创建：`tests/test_version_runner.py`
- 修改：`common/gcnet/train_gcnet.py`

- [ ] **步骤 1：编写版本解析红测**

```python
class VersionRunnerTest(unittest.TestCase):
    def test_locked_arguments_for_each_version(self):
        self.assertEqual(resolve("original")["graph_conv_variant"], "original")
        self.assertEqual(resolve("mpfilm")["graph_conv_variant"], "full")
        self.assertEqual(resolve("cp_lecc")["graph_conv_variant"], "cp_lecc")
        self.assertEqual(
            resolve("sequence_aff")["branch_fusion"],
            "mask_sequence_aff",
        )

    def test_user_cannot_override_locked_method_switch(self):
        with self.assertRaisesRegex(ValueError, "locked argument"):
            build_command("mpfilm", ["--graph-conv-variant", "original"])
```

- [ ] **步骤 2：运行测试确认失败**

```bash
python -m unittest tests.test_version_runner -v
```

预期：FAIL，`versions.registry` 和 `run.build_command` 尚不存在。

- [ ] **步骤 3：实现注册表和 JSON 配置**

配置只保存版本差异：

```json
{
  "graph_conv_variant": "full",
  "branch_fusion": "addition"
}
```

注册表固定：

```python
VERSION_NAMES = ("original", "mpfilm", "cp_lecc", "sequence_aff")

def resolve(name):
    if name not in VERSION_NAMES:
        raise ValueError("unknown completed version: {!r}".format(name))
    path = Path(__file__).parent / name / "config.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
```

- [ ] **步骤 4：实现 `run.py`**

`build_command()` 将 JSON 键稳定转换成 CLI flag，在用户参数中发现同名锁定 flag 时失败。`main()` 使用当前解释器执行 `common/gcnet/train_gcnet.py`，设置：

```python
env["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT), str(ROOT / "common" / "gcnet"), old_pythonpath]
)
subprocess.call(command, cwd=str(ROOT / "common" / "gcnet"), env=env)
```

- [ ] **步骤 5：运行 CLI 与 Python 3.8 语法测试**

```bash
python -m unittest tests.test_version_runner -v
python run.py --version original --help
python run.py --version mpfilm --help
python run.py --version cp_lecc --help
python run.py --version sequence_aff --help
python -m py_compile run.py versions/registry.py versions/*/variant.py
```

预期：全部 PASS，四个 help 命令退出码均为 0。

- [ ] **步骤 6：提交统一入口**

```bash
git add run.py versions common/gcnet/train_gcnet.py tests/test_version_runner.py
git commit -m "Select completed GCNet variants without editing shared code"
```

### 任务 4：迁移 fixed-fold-5 结果并绑定来源

**文件：**
- 创建：`results/iemocap6/fold5/original/README.md`
- 创建：`results/iemocap6/fold5/original/summary.json`
- 创建：`results/iemocap6/fold5/mpfilm/README.md`
- 移动：`experiments/mpfilm_iemocap6/EDGEWISE_FILM_AB.zh.md` → `results/iemocap6/fold5/mpfilm/RESULTS.zh.md`
- 创建：`results/iemocap6/fold5/cp_lecc/README.md`
- 移动：`experiments/cp_lecc_iemocap6/PROTOCOL_RECOVERY.zh.md` → `results/iemocap6/fold5/cp_lecc/RESULTS.zh.md`
- 创建：`results/iemocap6/fold5/sequence_aff/README.md`
- 移动：`experiments/sequence_aff_iemocap6/results/RESULTS.zh.md` → `results/iemocap6/fold5/sequence_aff/RESULTS.zh.md`
- 移动：`experiments/sequence_aff_iemocap6/results/summary.json` → `results/iemocap6/fold5/sequence_aff/summary.json`
- 创建：`provenance/source_map.json`
- 修改：`tests/test_organized_layout.py`

- [ ] **步骤 1：增加结果完整性红测**

```python
def test_result_manifests_declare_fixed_fold_five(self):
    for name in ("original", "mpfilm", "cp_lecc", "sequence_aff"):
        payload = json.loads(
            (RESULTS / name / "provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["dataset"], "IEMOCAPSix")
        self.assertEqual(payload["fold"], 5)
        self.assertEqual(payload["seeds"], [66, 67, 68, 69, 70])
        self.assertNotEqual(payload["protocol"], "five_fold_cross_validation")
```

- [ ] **步骤 2：运行测试确认失败**

```bash
python -m unittest tests.test_organized_layout -v
```

预期：FAIL，结果目录和 provenance 尚不存在。

- [ ] **步骤 3：移动已提交汇总并生成来源映射**

`provenance/source_map.json` 的每项包含：

```json
{
  "new_path": "results/iemocap6/fold5/sequence_aff/summary.json",
  "source_branch": "feature/mpfilm-rgcn",
  "source_commit": "60e8dd9",
  "source_path": "experiments/sequence_aff_iemocap6/results/summary.json"
}
```

Original compact summary从 Sequence-AFF `summary.json` 的 `tasks[].original` 字段确定性提取，不重新计算或四舍五入任务级指标。

- [ ] **步骤 4：记录不同证据覆盖范围**

- Original：8 rates × 5 seeds inherited control；
- MPFiLM：Linearized 与 Faithful Edge-wise 的 8-rate A/B；
- CP-LECC：只记录实际完成的 `.5/.7` protocol-recovery comparison；
- Sequence AFF：8 rates × 5 seeds paired comparison。

- [ ] **步骤 5：运行结果测试**

```bash
python -m unittest \
  tests.test_organized_layout \
  tests.test_sequence_aff_summary \
  tests.test_cp_lecc_summary -v
```

预期：全部 PASS。

- [ ] **步骤 6：提交结果布局**

```bash
git add results provenance tests experiments
git commit -m "Bind completed fold-five results to their source artifacts"
```

### 任务 5：文档化共享环境和每个版本

**文件：**
- 创建：`README.md`（替换研究入口）
- 创建：`environment/OFFICIAL_ENVIRONMENT.md`
- 创建：`environment/requirements.txt`
- 创建：`versions/*/README.md`
- 修改：`.gitignore`

- [ ] **步骤 1：写根 README**

README 首屏必须包含：四个完成版本表、统一运行命令、fixed fold 5 警告、结果覆盖范围和大型 artifact 排除说明。不得把任何失败方法描述为提升方法。

- [ ] **步骤 2：写共享环境记录**

固定记录：Python 3.8、Torch 1.8、PyG 2.0.1、官方环境解释器来源和复现所需外部 feature 目录；`requirements.txt` 仅记录既有依赖，不引入新包。

- [ ] **步骤 3：写版本 README**

每份 README 包含：替换位置、来源机制、锁定参数、结果状态、失败/局限结论和对应结果链接。

- [ ] **步骤 4：强化 artifact ignore**

忽略：`features/`、`mask_banks/`、`*.pt`、`*.pth`、`*.ckpt`、训练输出目录和 Python cache；保留小型 committed summary JSON。

- [ ] **步骤 5：运行文档和 blob 检查**

```bash
python -m unittest tests.test_organized_layout -v
git ls-files | grep -E '\.(pt|pth|ckpt)$' && exit 1 || true
git diff --check
```

预期：测试 PASS，无禁止 artifact，无空白错误。

- [ ] **步骤 6：提交文档**

```bash
git add README.md environment versions .gitignore tests
git commit -m "Make the completed GCNet evidence the repository entry point"
```

### 任务 6：完成回归验证并发布整理分支

**文件：**
- 修改：仅修复验证发现的整理回归

- [ ] **步骤 1：运行聚焦测试**

```bash
python -m unittest \
  tests.test_organized_layout \
  tests.test_version_runner \
  tests.test_missing_patterns \
  tests.test_mpfilm_rgcn \
  tests.test_model_mpfilm_integration \
  tests.test_cp_lecc_rgcn \
  tests.test_cp_lecc_summary \
  tests.test_sequence_aff \
  tests.test_sequence_aff_integration \
  tests.test_sequence_aff_summary -v
```

预期：0 failures、0 errors。

- [ ] **步骤 2：在官方 biggpu 环境运行非训练兼容门**

仅运行 import、四版本 forward/backward、Original 等价和 CLI help；不运行 1 epoch smoke，不启动正式训练。

```bash
/data2/yb/reproduction_envs/gcnet-official/bin/python -m unittest \
  tests.test_organized_layout \
  tests.test_version_runner \
  tests.test_model_mpfilm_integration \
  tests.test_cp_lecc_rgcn \
  tests.test_sequence_aff_integration -v
```

预期：0 failures、0 errors。

- [ ] **步骤 3：审计 Git 状态与未完成模块**

```bash
git status --short
find versions results -type f | sort
rg -ni 'remasker|genagg|soft.?medoid|g1u|g1s|bilstm_ablation|pconv' versions results
```

预期：工作树干净，最后一条命令无命中。

- [ ] **步骤 4：推送临时发布分支并核验 SHA**

```bash
git push github release/organized-completed-v1
test "$(git rev-parse HEAD)" = \
  "$(gh api repos/YB123-DT/git_gcnet/branches/release/organized-completed-v1 --jq .commit.sha)"
```

- [ ] **步骤 5：切换 GitHub 主入口**

只有上述全部通过后，才把验证提交快进/发布到 `main`。远端历史分支删除另作一次可恢复操作；执行前保留完整本地 branches，并记录被删除 remote ref 与 SHA。不得删除任何本地研究分支。

- [ ] **步骤 6：最终提交验证修复**

```bash
git add -A
git commit -m "Keep the organized GCNet release reproducible and auditable"
```

若步骤 1--3 没有产生修复，则不创建空提交。
