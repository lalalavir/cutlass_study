+++
title = "起步单元：框架速览与最小 GEMM"
weight = 1
translationKey = "starter-unit"
slug = "起步单元-框架速览与最小-gemm"
+++

本单元目标：快速建立 CUTLASS 的整体心智模型，并完成一个最小 GEMM 示例。

## 1. 框架速览
- `cutlass/include/`：核心模板与算子定义
- `cutlass/examples/`：可运行示例与教学入口
- `cutlass/tools/`：辅助脚本与工具

## 2. 用 utilities 写最小 GEMM
参考文件：`cutlass/examples/01_cutlass_utilities/cutlass_utilities.cu`

```cpp
#include "cutlass/util/host_tensor.h"

// 伪代码：准备 A/B/C -> 调用 GEMM -> 检查结果
int main() {
  // 1) 用 HostTensor 管理矩阵内存
  // 2) 初始化数据
  // 3) 调用一个基础 GEMM 配置
  // 4) 验证输出
  return 0;
}
```

## 3. 最小实验建议
1. 把 M/N/K 从 `128` 调到 `256`。
2. 对比不同数据类型（如 `float` / `half`）的行为。
3. 记录一次运行时间，作为后续优化基线。
