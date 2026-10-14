# MiniMax-H3 在海光 DCU 上的优化：复现包

嵩山杯赛题「基于 InfiniLM 与 InfiniCore 的 MiniMax-H3 首帧视频生成适配与优化」。

本仓库是**可复现的最小集合**：补丁 + 九齿算子源码 + 构建与运行脚本 + 文档。不含任何镜像、权重或二进制，评委在自己的海光环境上按下面步骤即可从上游仓库重建。

成绩（4×K100-AI，gfx928，DTK 25.04.2；astronaut 用例，768×768、124 帧、50 步、seed 42）：

| 版本 | DiT 每步 | `generation.pipeline_total` |
| --- | ---: | ---: |
| 官方 PyTorch 参考实现（`--legacy`） | 20.36 s | 1038.2 s |
| 本实现 | 4.839 s | **246.0 s**（4.22×） |

赛方统一对比脚本 `minimax_h3_compare.py --tasks fl2va` 实测：251.3 s 对 1038.2 s（4.13×），逐帧灰度 SSIM 80.67%。

完整说明见 **[`docs/优化手册-K100-AI.md`](docs/优化手册-K100-AI.md)**（按收益从高到低逐项展开：机制、实测数字、是否逐位一致、对应开关、踩过的坑）。算子层技术报告见 [`docs/技术报告-算子优化.md`](docs/技术报告-算子优化.md)，合规逐项对照见 [`docs/COMPLIANCE-K100.md`](docs/COMPLIANCE-K100.md)。

## 1. 仓库内容

| 路径 | 内容 |
| --- | --- |
| `patches/` | 相对上游各仓库基线提交的全部改动（见下表） |
| `tools/` | 九齿（NineToothed）算子源码与全部诊断/基准工具 |
| `scripts/` | 构建脚本（`build_all.sh`、`build_h3_attention.py`）、相似度评估、用例运行 |
| `scripts/z100/` | 海光 Z100（gfx906）环境的准备与构建脚本 |
| `configs/` | FA 分块配置（`gfx928`、`gfx906`） |
| `runs/` | 增量重编与运行脚本（`rebuild.sh`、`run_plain.sh`、`bench_step.sh`、`prof_step.sh`、`run_compare.sh`） |
| `env/` | 环境变量、首次完整构建、legacy 参考运行、逐步对比等脚本 |
| `docs/` | 优化手册、算子层技术报告、合规对照 |
| `versions.json` | 模型与 Diffusers 的 revision |

`runs/` 和 `env/` 里的脚本默认路径是开发机的 `/root/h3-work`，可用环境变量覆盖：

```bash
export H3_ROOT=/your/path/h3-extreme      # 源码树
export H3_WORK=/your/path/h3-work         # 运行产物（runs/bench/prof/compare）
export H3_MODEL=/your/path/MiniMax-H3     # 模型目录
```

## 2. 上游基线与补丁

三个仓库都是「上游某提交 + 本工作」。请先 clone 到同一个父目录下并切到基线提交，再打补丁：

| 仓库 | 上游基线提交 | 补丁 |
| --- | --- | --- |
| [InfiniCore](https://github.com/InfiniTensor/InfiniCore) | `1ab85ef1` | `patches/InfiniCore.patch` |
| [InfiniLM](https://github.com/InfiniTensor/InfiniLM) | `270feb3e` | `patches/InfiniLM.patch` |
| [ninetoothed](https://github.com/InfiniTensor/ninetoothed) | `897f97b`（v0.25.0） | `patches/ninetoothed.patch` |

```bash
mkdir -p h3-extreme && cd h3-extreme

git clone https://github.com/InfiniTensor/InfiniCore.git
git -C InfiniCore checkout 1ab85ef1
git -C InfiniCore submodule update --init --recursive
git -C InfiniCore apply --3way /path/to/h3-k100-repro/patches/InfiniCore.patch

git clone https://github.com/InfiniTensor/InfiniLM.git
git -C InfiniLM checkout 270feb3e
git -C InfiniLM submodule update --init --recursive
git -C InfiniLM apply --3way /path/to/h3-k100-repro/patches/InfiniLM.patch

git clone https://github.com/InfiniTensor/ninetoothed.git
git -C ninetoothed checkout 897f97b
git -C ninetoothed apply --3way /path/to/h3-k100-repro/patches/ninetoothed.patch

# 本仓库的顶层内容（九齿算子、脚本、配置）直接拷进树根
cp -r /path/to/h3-k100-repro/{tools,scripts,configs} .
```

`patches/toplevel-tracked.patch` 是顶层仓库里被跟踪文件的改动（`tools/h3_pointwise.py`、`scripts/build_h3_attention.py`、`README.md`），仅在你已有 9 月版顶层仓库时需要；全新搭建请直接用上面 `cp -r` 的方式。

## 3. 环境

| 项 | 本工作验证的版本 |
| --- | --- |
| 加速卡 | 4 × K100-AI（gfx928，64 GB），PCIe 互联；2 卡亦可（自动改张量并行 + 分阶段驻留） |
| 驱动 / 工具链 | DTK 25.04.2（`/opt/dtk`） |
| Python / 框架 | Python 3.10，DAS PyTorch 2.5.1、flash-attn 2.6.1+das（容器自带） |
| 主机内存 | 建议 ≥ 300 GB（两份权重的主机缓存约 143 GB；不足会自动退回按需读盘） |
| 本地磁盘 | 约 150 GB 空闲放模型副本 |
| 模型 | `MiniMax/MiniMax-H3`，revision 见 `versions.json` |
| Diffusers | H3 modular pipeline，revision 见 `versions.json` |

换架构时：九齿内核需为新架构重新生成（`python scripts/build_h3_attention.py --archs <arch>`），FA 分块配置用 `tools/tune_attention.py` 重新调优，并新增 `configs/<arch>/attention.json`。

## 4. 构建

```bash
source env/env.sh                 # DTK、venv、INFINI_ROOT、PYTHONPATH
bash env/build.sh                 # 首次完整构建：venv、九齿 AOT 内核、InfiniCore、InfiniLM
# 增量重编（改动源码后）
bash runs/rebuild.sh all          # core|lm|all|nt；nt 会先重新生成九齿内核
```

`env/build.sh` 结束会打印 `BUILD_ALL_DONE`，`runs/rebuild.sh` 成功打印 `REBUILD_OK`。注意**不要**直接 `xmake build`：构建需要 `H3_OFFLINE_BUILD`、`H3_PYBIND11_INCLUDE`、`H3_PYTHON_INCLUDE`、`INFINILM_CXX11_ABI` 这些变量，脚本已设好（并开了 `pipefail` 逐步检查）。

## 5. 运行

所有经过验证的开关都是**代码默认值**（`InfiniLM/python/infinilm/multimodal/_h3_defaults.py` 在 import 时 `setdefault`），因此评测直接调官方入口即可，不必经过任何包装脚本：

```bash
source /opt/dtk/env.sh
export INFINI_ROOT=$H3_ROOT/.infini
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH

python InfiniLM/examples/minimax_h3.py fl2va \
    --model-path $H3_MODEL \
    --text "An astronaut in an orange space suit smiles and slowly waves at the camera." \
    --first-frame samples/astronaut.png --gpus 0,1,2,3 \
    --steps 50 --seed 42 --num-frames 124 --output astronaut.mp4
```

或用脚本（只设库路径，模仿评测环境；结束打印 `EXIT <码>`）：

```bash
cd runs && (nohup ./run_plain.sh r1 > r1.log 2>&1 &)
# 计时表是日志里以空格开头的 "component.name  秒数" 行
tr '\r' '\n' < r1.log | grep -aE "EXIT|^ [a-z]"
```

赛方统一对比脚本（legacy 对 InfiniLM，SSIM + 计时；legacy 约 17–20 分钟）：

```bash
bash runs/run_compare.sh c1 --legacy-memory-fraction 0.85
```

## 6. 验证正确性

本工作的改动除两项显式标注外都与优化前**逐位一致**。复现验证：

```bash
# 1) 捕获单步的真实 DiT 输入（只需一次）
bash env/capture_step.sh

# 2) 单步前向逐位比较：先存基线，再比较
bash runs/bench_step.sh base --save-output $H3_WORK/bench/base.pt
bash runs/bench_step.sh new  --compare     $H3_WORK/bench/base.pt
#    报告里的 video_bitwise_equal / audio_bitwise_equal 应为 true

# 3) 整段视频逐字节比较
cmp $H3_WORK/runs/r1.mp4 $H3_WORK/runs/r2.mp4

# 4) 逐调用哈希（定位偶发不一致）
TRACE=1 ./run_plain.sh t1 > t1.log 2>&1
python tools/compare_traces.py t1.trace.jsonl t2.trace.jsonl

# 5) 各九齿算子对 torch 参考的正确性与性能
python tools/bench_h3_ops.py --seq 8192 --heads 14 --json ops.json
```

为什么不能只看 SSIM：该模型对舍入级扰动极敏感，两种数学等价实现第 0 步输出即有约 3% 相对差，49 步后轨迹分叉，因此整段 SSIM 会随用例在 85%–96% 波动。可靠的做法是上面的逐位比较与逐步 teacher-forcing 对比（`env/run_capture_legacy.sh` + `env/run_compare_steps.sh`）。详见手册「正确性怎么保证」一节。

## 7. 性能剖析

```bash
bash runs/prof_step.sh p1                      # hipprof 单步 kernel 时间线与统计
python tools/prof_timeline.py <db> <dev>       # 按队列统计忙/闲/重叠
python runs/block_timeline.py <db> <dev> 25    # 单个 block 的 kernel 时间线
python tools/pyspy_pipeline_other.py <pyspy>   # 计时组件之外的时间归属
```

## 8. 合规

- 新增或重写的模型计算算子全部用九齿 DSL 编写，经九齿 → Triton → hsaco 生成；无手写 Triton/HIP/CUDA kernel，九齿内部未转调 flash-attn / hipBLASLt 等现成计算 kernel。
- DiT 注意力使用 InfiniCore 的海光 flash attention 适配（赛方 Q&A 明确允许）。
- 未修改权重、计算语义、测试数据与评测程序；Diffusers 源码未改。
- 未做量化、低秩近似、稀疏/窗口注意力、减步数/帧数/分辨率、跨步特征复用、跨用例缓存。

逐项分级对照见 [`docs/COMPLIANCE-K100.md`](docs/COMPLIANCE-K100.md)。
