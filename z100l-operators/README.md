# Z100L 算子提交

[评审报告](REPORT.md)包含环境、核心改动、52 组验证、性能表和复现命令。

在赛题 DTK POD 内执行：

```bash
bash z100l-operators/scripts/reproduce.sh quick
bash z100l-operators/scripts/reproduce.sh full
```

需要 Python 3.11、HIP PyTorch、gfx906 DCU；脚本按需安装 Triton 3.8.0 和 pybind11 2.13.6。离线可设置 `DEPS_DIR` 指向 wheel 目录。`official` 模式另需 xmake 3.0.5，用随附 InfiniCore 源码构建并复现官方 18 项冒烟。仅包含算子，无模型权重。

- `tools/`：算子、DTK ABI 适配与基准。
- `ninetoothed/`：本次实际使用的 NineToothed 源码及许可证。
- `vendor/`：InfiniCore 完整源码归档及其许可证。
- `results/`、`logs/`：实测证据；新结果输出至 `work/`。
- `versions.json`、`SHA256SUMS`：环境版本与内容校验。
