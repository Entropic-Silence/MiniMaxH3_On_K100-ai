# 算子优化报告（Z100L 单卡）

队名：________　参赛单位：________　成员：________

## 1. 提交范围与环境

提交 8 类 BF16 算子的 NineToothed 实现、DTK 适配代码、复现脚本和原始测试记录。仅完成算子开发与冒烟测试，未部署 MiniMax-H3，未测试模型推理或多卡扩展。

| 项目 | 测试环境 |
|---|---|
| 资源型号 | 海光 Z100L，单卡 |
| 实际设备枚举 | `Z200SM_80`，`gfx906`，64 CU |
| 实际可见显存 | 16368 MiB |
| 软件栈 | DTK 26.04；HIP 6.3.26093；驱动 6.2.31 |
| Python / PyTorch | 3.11.9 / 2.9.0（DTK 镜像） |
| 编译依赖 | NineToothed 源码 0.25.0；Triton 3.8.0；pybind11 2.13.6；xmake 3.0.5 |

型号按 Z100L 归类；驱动枚举名与可见显存采用实测值，不据此推断整卡标称容量。

## 2. 核心改动

- 融合 QK RMSNorm 与 RoPE、RMSNorm 与调制、门控残差与 RMSNorm/调制，减少中间张量和 kernel 启动。独立提供 RoPE、RoPE cache、RMSNorm、门控残差和 SwiGLU。
- 修复该软件栈的 BF16 中间舍入问题，以整数位操作实现 round-to-nearest-even，避免融合路径出现 NaN。SwiGLU 的指数局部使用 FP64 `exp`，随后转回 FP32 并进行 BF16 舍入。
- 使用 DTK 头文件查询真实设备属性，适配 Triton/HIP 主机 ABI；补齐 AOT 启动的 scratch 参数并检查非零 scratch。适配代码不承担设备计算，算子由 NineToothed 生成。

源码见 `tools/h3_pointwise.py`、`tools/h3_gfx906_pointwise.py`、`tools/dtk_triton_compat.py`，以及 `ninetoothed/src/` 中的 AOT 适配。

## 3. 正确性

- InfiniCore 官方 `timestep_embedding` 冒烟：**18/18 通过**。
- 交付目录的一键快速冒烟已在同一赛题 POD 重跑：**8/8 通过**。
- 自定义算子：**52/52 通过**，覆盖 8 类算子、T=17/128/8192、随机种子 0/1，以及两类调制融合的零步长广播。
- 51/52 与 PyTorch 输出逐字节一致。唯一非逐字节一致项为 QK Norm + RoPE（T=8192、seed=1）：最大绝对误差 0.015625，相对 L2 误差 1.7425×10⁻⁵。
- 判定标准：输出有限，`allclose(atol=0.01, rtol=0.01)`，相对 L2 < 0.005；融合算子的残差输出另行核对逐字节一致。

证据：`results/suite_summary.json`、各用例 JSON 和 `logs/official_smoke.log`。

## 4. 单卡性能

BF16、T=8192、seed=0。RMS 隐藏维度 5376，SwiGLU 维度 14336；QK 使用 14 头、头维度 128；独立 RoPE 为 1 头、头维度 128；RoPE cache 输出 `[T,96]`，位置 FP64、频率幂 FP32。

| 算子 | 优化实现 ms | PyTorch eager ms | 倍率 |
|---|---:|---:|---:|
| QK Norm + RoPE | 0.457477 | 5.066651 | 11.08× |
| RoPE | 0.307323 | 0.382624 | 1.25× |
| RoPE cache | 0.257003 | 0.229077 | 0.89× |
| RMSNorm | 0.336203 | 7.230187 | 21.51× |
| RMSNorm + 调制 | 0.625739 | 13.626441 | 21.78× |
| 门控残差 | 0.632064 | 0.802571 | 1.27× |
| 门控残差 + RMSNorm + 调制 | 1.019200 | 14.435001 | 14.16× |
| SwiGLU | 2.570917 | 10.449956 | 4.06× |
| RMSNorm + 调制（广播） | 0.614427 | 13.826472 | 22.50× |
| 门控残差 + RMSNorm + 调制（广播） | 0.883696 | 14.562354 | 16.48× |

测量使用同一 GPU stream 上的事件计时，预热 3 次，每批执行 30 次，取 5 批平均耗时的中位数；不含编译与输入生成。优化实现预分配输出，PyTorch eager 按通常方式分配输出，倍率仅代表此基准条件。RoPE cache 保留为正确性通过的实现，不计为加速成果。

## 5. GitHub 与复现

仓库：[Entropic-Silence/MiniMaxH3_On_K100-ai](https://github.com/Entropic-Silence/MiniMaxH3_On_K100-ai)。

复现源码版本：`bb030a15dfff9dbf48bc6b3573cbb90287905537`。

本次提交目录：[z100l-operators](https://github.com/Entropic-Silence/MiniMaxH3_On_K100-ai/tree/bb030a15dfff9dbf48bc6b3573cbb90287905537/z100l-operators)。报告全部性能数据来自本次单卡测试。

按照赛题教程进入 DTK POD，在可写目录执行：

```bash
git clone https://github.com/Entropic-Silence/MiniMaxH3_On_K100-ai.git
cd MiniMaxH3_On_K100-ai
git checkout bb030a15dfff9dbf48bc6b3573cbb90287905537
bash z100l-operators/scripts/reproduce.sh quick  # 8 类算子快速冒烟
bash z100l-operators/scripts/reproduce.sh full   # 52 组正确性与性能复现
```

脚本复用 POD 的 HIP PyTorch，按需安装 Triton 3.8.0 和 pybind11 2.13.6，不下载模型。离线环境将对应 Python 3.11/x86_64 wheel 放到目录后，以 `DEPS_DIR=/path/to/wheels bash z100l-operators/scripts/reproduce.sh full` 执行。结果写入 `z100l-operators/work/`；通过后输出 `Z100L_OPERATOR_SUITE_OK`，失败返回非零状态。

复现官方冒烟需已安装 xmake 3.0.5，执行 `bash z100l-operators/scripts/reproduce.sh official`；`all` 顺序执行完整算子验证和官方冒烟。随仓库提供 InfiniCore 源码归档（基于 `6e345a97d37312f503e360000fb4d4bf0b6ad28e`，含构建适配），不依赖另外下载其子模块。`SHA256SUMS` 校验提交源码、归档与原始结果。
