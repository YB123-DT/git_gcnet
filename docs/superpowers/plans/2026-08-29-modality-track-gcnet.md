# Modality-Track GCNet 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 Missing-M3 增加参数共享的三模态图轨道，使 A/T/V 分别经过同一个 GCNet 图核心，再在图后融合。

**架构：** `representation_type=slot` 完整保留现有图前 Slot Fusion；`track` 将 observed Student latent 形成三个稀疏模态轨道，顺序调用同一个 `encode_hidden` 三次，并用 availability-masked post-graph fusion 得到分类和 JEPA 共用 hidden。Predictor、Teacher、loss、mask 与图拓扑不变。

**技术栈：** PyTorch、PyTorch Geometric、pytest、远程 V100 训练环境。

---

## 文件职责

- 修改 `gcnet_missing_m3/model.py`：模态轨道编码、图后融合、模型路径开关。
- 修改 `gcnet_missing_m3/train_gcnet.py`：配置和 CLI 透传。
- 修改 `tests/test_missing_m3.py`：行为、兼容性、梯度与 CLI 回归测试。
- 修改 `experiments/missing_m3_mosi_modality_track_20260829/EXPERIMENT.md`：正式协议、运行证据和结论。

### 任务 1：锁定轨道编码与图后融合行为

- [ ] **步骤 1：编写失败测试**

在 `tests/test_missing_m3.py` 添加测试，直接导入 `ModalityTrackEncoder` 与 `PostGraphTrackFusion`，验证：七种非空 pattern；missing/padding 轨道严格为零；改变 missing feature 不改变输出；融合只读取当前 observed 轨道。

- [ ] **步骤 2：验证红灯**

运行：

```bash
scripts/remote_missing_m3.sh test tests/test_missing_m3.py -k 'modality_track_encoder or post_graph_track_fusion' -q
```

预期：因两个类尚不存在而 FAIL。

- [ ] **步骤 3：最小实现**

在 `gcnet_missing_m3/model.py` 实现：

```python
class ModalityTrackEncoder(nn.Module):
    def forward(self, features, availability, umask):
        # 返回 {A,T,V} 三个 [L,B,d] track 和 observed-only latents
        ...

class PostGraphTrackFusion(nn.Module):
    def forward(self, track_hidden, availability, umask):
        # [a_A h_A; a_T h_T; a_V h_V; e_pattern] -> [L,B,D_h]
        ...
```

Projector 只索引 observed 位置；missing/padding 不调用 projector 且输出为零。

- [ ] **步骤 4：验证绿灯**

运行同一步骤 2 命令，预期新增测试全部 PASS。

### 任务 2：接入共享 GCNet 三次计算路径

- [ ] **步骤 1：编写失败测试**

在 `tests/test_missing_m3.py` 添加：

```python
def test_track_representation_reuses_one_graph_core_for_three_modalities():
    model = MissingM3GraphModel(**_model_arguments(), representation_type="track")
    # spy encode_hidden；断言一次 forward 调用三次，且没有复制图模块
    ...

def test_default_representation_preserves_state_keys_and_single_graph_call():
    # 省略参数与显式 slot 的 key 集相同；forward 只调用一次 encode_hidden
    ...
```

另测 `track` 与 raw-residual、local-context、classification-completion 互斥。

- [ ] **步骤 2：验证红灯**

运行：

```bash
scripts/remote_missing_m3.sh test tests/test_missing_m3.py -k 'track_representation or default_representation' -q
```

预期：构造函数不接受 `representation_type`，测试 FAIL。

- [ ] **步骤 3：最小实现**

修改 `MissingM3GraphModel`：

```python
if representation_type == "track":
    tracks, latents = self.observed_set(features, availability, umask)
    track_hidden = {
        name: self.encode_hidden([track], qmask, umask, seq_lengths)
        for name, track in tracks.items()
    }
    graph_hidden = self.track_fusion(track_hidden, availability, umask)
else:
    node, latents = self.observed_set(features, availability, umask)
    graph_hidden = self.encode_hidden([node], qmask, umask, seq_lengths)
```

只实例化一套父类图参数；default `slot` 不实例化任何轨道参数。

- [ ] **步骤 4：验证绿灯与梯度**

运行新增测试，并验证分类与 JEPA backward 能到达三个 projector、共享 graph core、track fusion 和 predictor。

### 任务 3：CLI、配置与完整回归

- [ ] **步骤 1：编写失败测试**

扩展现有 CLI/config 测试，验证 `--representation-type track` 被传给模型，默认值为 `slot`。

- [ ] **步骤 2：验证红灯**

运行：

```bash
scripts/remote_missing_m3.sh test tests/test_missing_m3.py -k representation_type -q
```

预期：参数未定义，FAIL。

- [ ] **步骤 3：实现 CLI 透传**

在 `MissingM3Config` 尾部增加 `representation_type: str = "slot"`，parser 增加：

```python
parser.add_argument(
    "--representation-type",
    choices=("slot", "track"),
    default="slot",
)
```

模型构造时透传该字段。

- [ ] **步骤 4：完整验证**

运行：

```bash
scripts/remote_missing_m3.sh sync gcnet_missing_m3/model.py gcnet_missing_m3/train_gcnet.py tests/test_missing_m3.py
scripts/remote_missing_m3.sh test tests/test_missing_m3.py tests/test_mosi_text_lora.py tests/test_plci_model.py -q
```

预期：全部 PASS，无新增 warning/error。

### 任务 4：单次 smoke 与正式五种子实验

- [ ] **步骤 1：一次有效 smoke**

用 MOSI fold1、seed66、1 epoch、`--representation-type track --fusion-type slot --mmoe-variant dual-gate` 运行一次；检查 8 个 rate 的 NPZ、finite loss、GPU memory 和 runtime。失败只诊断根因，不重复环境检查。

- [ ] **步骤 2：并行正式运行**

GPU 0/1/2/3/5 各运行 seeds 66/67/68/69/70；保持 Control 的 all-rates-per-batch、mask generation、100 epoch 与 evaluation 不变，不重跑 Control。

- [ ] **步骤 3：结果审计**

从 40 个 NPZ 重算 test weighted-F1，核验每个 seed/rate 的 mask hash 与样本数，汇总：八-rate mean、0.4--0.7 mean、逐 rate delta、逐 seed delta、runtime、peak memory。

- [ ] **步骤 4：记录与提交**

把协议、命令、结果和 PASS/FAIL 判断写入 `experiments/missing_m3_mosi_modality_track_20260829/EXPERIMENT.md`，按 Lore protocol 提交并推送 `github feature/m3-jepa-gcnet`。

