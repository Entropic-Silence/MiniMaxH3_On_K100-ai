# K100 ×4 改动的合规对照（依据 PLAN.md 第 1 章三色表）

| 改动 | 所在仓库/文件 | 分类 | 说明 |
|---|---|---|---|
| 九齿 FA：Q 块提到 K/V 循环外 | 主仓库 `tools/h3_attention.py` | 🟢 G3 | 仅 NineToothed DSL 写法变化，经九齿→Triton→hsaco 生成；算术与舍入不变，误差指标与修改前逐项相同 |
| gfx928 FA 配置表 | `configs/gfx928/attention.json` | 🟢 G11 | 构建期离线配置（BM128/BN32/w4/s2），与输入无关 |
| 门控残差（+ RMSNorm/AdaLN）融合算子（默认开，`H3_FUSED_RESIDUAL=0` 关） | 主仓库 `tools/h3_pointwise.py`、`scripts/build_h3_attention.py`；InfiniCore `h3_numerics.*`；InfiniLM `minimax_h3_transformer.cpp` | 🟢 G3 | 九齿 DSL 新算子，经九齿→Triton→hsaco 生成；逐步保留参考的 BF16 舍入（乘积、求和、RMSNorm/AdaLN 各中间量），不是 Y6 的单次舍入融合；单测与完整 CLI 输出均与未融合逐位一致。InfiniCore 侧只有算子接口与启动器 |
| 首次前向同步集合通信 | InfiniCore `collective_sync.*` 与各集合通信算子；InfiniLM `rank_worker.*` | 🟢 G2 | 运行时调度，避免 HIP 惰性代码对象加载与在途 RCCL kernel 死锁；只影响首次前向的执行顺序，不改变计算 |
| all-to-all 通信算子（含指定流版本）；all-reduce 指定流版本 | InfiniCore `ops/distributed/all_to_all.*`、`allreduce.*` | 🟢 G2 | 新增 InfiniCCL 分组 send/recv 组合；纯数据搬运。`allreduce_on_stream_` 只是把已有 InfiniCCL all-reduce 发到指定流 |
| MHA 允许 Q 长度大于 K/V | InfiniCore `mha_ninetoothed.cc` | 🟢 G2 | 接口放宽；kernel 仍为九齿生成，K/V 按真实长度屏蔽 |
| Ulysses 序列并行（可选，`H3_PARALLEL=sp`；默认 TP） | InfiniLM `minimax_h3_transformer.*` | 🟢 G5 | 全注意力、全部步数/帧数/分辨率不变；无近似、无陈旧激活；与 TP 一样是精确并行（数值差异来自 GEMM 划分方式）。多用例 SSIM 平均低于 TP，故不设为默认 |
| 注意力分组与异步通信重叠 | 同上 | 🟢 G5 | 逐头计算不变，输出逐位一致 |
| TP all-reduce 分段重叠（可选，`H3_TP_OVERLAP=C`；默认 1 即关闭） | 同上 | 🟢 G5 | 按 128 行对齐分段计算注意力行/MLP，侧流 all-reduce 与后续段计算重叠；全注意力、步数、分辨率不变。GEMM 的 M 与 RCCL 规约分块改变带来舍入级差异（与 TP/SP 之间同量级），多用例 SSIM 平均低于 TP，故默认关闭 |
| 分阶段驻留 + 主机权重缓存 + 快速加载（含锁页预读、后台解除锁页） | InfiniLM `modeling_utils.py`、`_h3_residency.py`、`minimax_h3.py` | 🟢 G5/G8 | 权重换入换出与宿主调度；读取的是同一 safetensors 字节（与 `safe_open` 逐张量一致）；不缓存任何与输入相关的数据或生成结果 |
| VAE 分片预读并交给 Diffusers 加载 | InfiniLM `_h3_residency.py`（`serve_prefetched_safetensors`） | 🟢 G8 | 仅在非 legacy 分支的 `load_components` 期间，`safetensors.torch.load_file` 对已预读的 VAE 分片返回同一文件字节构成的张量，其他文件照常读取；块结束即恢复原函数。Diffusers 源码未改 |
| VAE 先以 FP32 上卡再转 BF16 | InfiniLM `examples/minimax_h3.py`（非 legacy 分支） | 🟢 G8 | 入口原有 `.to(device, bf16)` 保留；前面加一次仅改设备的 `.to(device)`，BF16 舍入改在卡上做（同为就近偶数舍入），703 个张量与主机端转换逐位一致 |
| 设备内存分配器尺寸策略 | InfiniCore `pinnable_block_allocator.*` | 🟢 G2 | 内存管理 |
| VAE 多卡 tile 并行、解码前清理分配器缓存 | InfiniLM `minimax_h3_vae_parallel.py` | 🟢 G10 | Diffusers VAE 源码未改；tile 几何、重叠、融合公式、时间分块、autocast 均不变，输出与串行逐位一致 |
| 统一入口改动 | InfiniLM `examples/minimax_h3.py` | 🟡 Y3 | 只在非 legacy 分支增加调用：VAE 解码安装、DiT 与 VAE 分片后台预读、VAE 分片交给 Diffusers、VAE 先上卡再转换、加载期预热（计入 `model.warmup`，在 `generation.pipeline_total` 之外）、视频帧后处理安装；CLI、`[timing]` 行格式、legacy 路径均未改；`H3_VAE_PARALLEL=0` 可关闭 VAE 并行，`H3_HOST_CACHE=0` 关闭预读 |

| 代码默认值（2026-10） | InfiniLM `_h3_defaults.py` | 🟢 G8 | 评测直接调官方入口，已验证的开关在 import 时 `setdefault`；可被环境变量覆盖，不改计算 |
| Qwen3-VL 与 DiT 同时常驻、加载期预热（2026-10） | InfiniLM `_h3_residency.py`、`_h3_warmup.py`、`examples/minimax_h3.py`（非 legacy 分支） | 🟢 G5/G8 | 预热在加载阶段用合成输入跑一次 2 步小生成，输出丢弃，不缓存任何与输入有关的数据；正式生成每一步都重新计算 |
| GEMM 走 InfiniCore cuBLASLt/hipBLASLt 路径（2026-10） | InfiniCore `gemm_nvidia.cu`、`nvidia_common.cu` | 🟢 G2 | InfiniCore 算子接口内选厂商 BLAS 的确定性算法（不用 atomics）；计算语义不变。按形状候选表/自动调优仅为诊断，默认不启用 |
| 首次前向启动锁（2026-10） | InfiniCore `collective_sync.*` 及各集合通信入口 | 🟢 G2 | 运行时调度，只影响首次前向的启动顺序 |
| DiT RoPE 表改由 InfiniCore 计算（2026-10-12） | 主仓库 `tools/h3_pointwise.py`（九齿 `h3_rope_cache`）；InfiniCore `h3_numerics.*`（算子接口与启动器）、`float_power`（已有算子）；InfiniLM `minimax_h3.py` | 🟢 G3 | 原先 cos/sin 表在 Python 里用 torch 算；现在频率幂用 InfiniCore `float_power`，逐 token 的 cos/sin 用九齿 kernel，与原写法逐位一致（单步 `--compare` 与整段视频均逐位相同） |
| 位置下标在主机上生成（2026-10-12） | InfiniLM `minimax_h3.py`（Qwen3-VL 3 轴位置、DiT 表行号） | 🟢 G8 | 整数下标改用 numpy 在主机上生成后一次拷贝，路径上不再有 torch 的设备 kernel；结果逐位相同 |
| 视频帧转 PIL 放到设备上（2026-10-12） | InfiniLM `minimax_h3_vae_parallel.py`（`install_video_postprocess`）、`examples/minimax_h3.py`（非 legacy 分支） | 🟢 G10 | 只替换 `video_processor.postprocess_video` 的 PIL 分支实现；同样的 FP32 乘 255、就近偶数舍入、转 uint8，像素逐位相同。Diffusers 源码未改 |
| VAE tile 合批（`H3_VAE_TILE_BATCH=4`，2026-10-12） | InfiniLM `minimax_h3_vae_parallel.py` | 🟢 G10 | VAE 不受合规限制；tile 几何、重叠、融合、时间分块、autocast 均不变，每个 tile 仍独立解码，仅卷积按批执行，差异在 FP16 舍入量级 |
| VAE 副本固定线程、预建 MIOpen 句柄（2026-10-12） | InfiniLM `minimax_h3_vae_parallel.py` | 🟢 G2 | 线程调度，规避 DTK 惰性加载竞态导致的卡死，不改计算 |

未修改：legacy 路径、官方比较脚本、权重文件、Diffusers 源码、步数语义、VAE tiling 参数。没有新增手写 Triton/HIP 计算 kernel；新增的 C++ 代码只做调度、通信、内存管理和张量视图重排。

FA 仅末块 mask 变体（`numerics: fp32_qk_split`）与默认版本存在 1 ulp 级差异，默认不启用。

诊断工具（`tools/bench_dit_step.py`、`tools/ref_dit_step.py`、`tools/run_cases.py`、`tools/capture_legacy_steps.py`、`tools/compare_steps.py`、`tools/bench_read.py`、`tools/bench_h2d.py`、`tools/bench_vae_to.py`、`tools/profile_components_load.py`、`tools/profile_dit_load.py`、`tools/prof_timeline.py`、`logs/k100/*probe*.py`）只用于测量和对比，不进入生成路径。`capture_legacy_steps.py` 通过包装 legacy DiT 的 forward 记录输入输出，运行的是原样 legacy CLI，记录文件标记 `diagnostic_only`，生成路径不读取。
