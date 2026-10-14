# MiniMax-H3 算子开发

当前评审提交：[Z100L 单卡算子](z100l-operators/README.md) · [优化报告](z100l-operators/REPORT.md)。

8 类 BF16 算子，52/52 正确性验证通过；InfiniCore 官方冒烟 18/18。提交包含源码、DTK 适配、原始结果和一键复现入口。

在赛题 DTK POD 中：

```bash
git clone https://github.com/Entropic-Silence/MiniMaxH3_On_K100-ai.git
cd MiniMaxH3_On_K100-ai
bash z100l-operators/scripts/reproduce.sh quick
bash z100l-operators/scripts/reproduce.sh full
```

本阶段仅开发算子，无 MiniMax-H3 部署或多卡推理结果。型号按 Z100L 归类，运行时枚举 `Z200SM_80`；详细环境与测量条件见报告。

[历史复现说明](docs/历史复现说明.md)
