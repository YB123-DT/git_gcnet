# Missing-M3 远程执行唯一入口

所有 Single-View Missing-M3 的测试、同步和训练统一通过：

```bash
scripts/remote_missing_m3.sh
```

禁止再临时搜索 Python、猜测远程目录或直接从任意 cwd 启动。

## 固定合同

| 职责 | 固定值 |
|---|---|
| 主机 | `biggpu` |
| 远程代码根目录 | `/data2/yb/paper/GCNet_TPAMI_single_view_dev` |
| 单元测试 Python | `/data2/yb/reproduction_envs/s0/bin/python` |
| 正式训练 Python | `/data2/yb/reproduction_envs/gcnet-official/bin/python` |
| 禁用 GPU | `4` |

环境变量 `GCNET_REMOTE_HOST`、`GCNET_REMOTE_ROOT`、`GCNET_TEST_PY` 和
`GCNET_TRAIN_PY` 只用于明确迁移，不应在普通实验中临时覆盖。

## 标准顺序

每个新模块只执行一次 preflight：

```bash
scripts/remote_missing_m3.sh preflight
```

只同步本次修改的明确文件：

```bash
scripts/remote_missing_m3.sh sync \
  gcnet_missing_m3/model.py \
  gcnet_missing_m3/train_gcnet.py \
  tests/test_missing_m3.py
```

运行单元测试：

```bash
scripts/remote_missing_m3.sh test \
  -q tests/test_missing_m3.py -k paper_faithful
```

正式环境命令必须显式给 GPU，脚本拒绝 GPU4：

```bash
scripts/remote_missing_m3.sh train 0 \
  -m gcnet_missing_m3.train_gcnet --help
```

## 已确认的重复问题

| 重复问题 | 根因 | 永久规则 |
|---|---|---|
| 本地 `python` 缺少 Torch | 本地 base 不是实验环境 | 不在本地执行 PyTorch tests |
| official Python 缺少 pytest | 正式环境只为复现实验锁依赖 | pytest 只用 `s0` |
| 测试环境与训练环境混用 | 没有职责绑定 | `test`/`train` 子命令固定解释器 |
| 从错误目录 import `model` 失败 | GCNet 依赖仓库 cwd/PYTHONPATH | wrapper 强制 `cd` 到远程根目录 |
| 修改后远端仍运行旧代码 | 手工同步路径不一致 | `sync` 只接收仓库相对路径并保留目标位置 |
| GPU4 随机 SIGKILL | 已知硬件异常 | `train` 明确拒绝 GPU4 |
| 每个模块重新跑 Original | runner 没有继承控制结果 | 新实验记录显式绑定已有 control，不启动 Original |
| 训练完成却因尾部非零退出被判失败 | runner 只看 return code | 先验证结果文件完整性，再决定是否重跑 |
| 动态 mask 与固定 mask 被混比 | 训练器未绑定 mask provenance | 比较前核验 NPZ availability/mask hash |
| 并发读取半写 manifest | 非原子写入或过早扫描 | 只接受原子完成标记/完整 manifest |
| 多个任务在同卡过度占用 CPU | 每进程默认线程数过高 | 正式 runner 固定线程预算与每卡并发上限 |

最后六项属于实验 runner 合同，不应为了每个新模块重新编写或重复验证；新模块只继承
已经通过的 runner、mask 和 control provenance。

