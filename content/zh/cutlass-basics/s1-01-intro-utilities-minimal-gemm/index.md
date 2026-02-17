---
title: "S1-01 课程导论与张量容器：从 01_cutlass_utilities 写一个最小 GEMM"
slug: "s1-01-课程导论与张量容器-最小-gemm"
translationKey: "s1-01-intro-utilities-minimal-gemm"
weight: 1
date: 2026-02-17T10:00:00+08:00
draft: false
---

这一篇对应教学大纲的 S1-01。目标很直接：先不追求“最快”，先把 CUTLASS 的最小 GEMM 路径完整走通，并理解 utilities 在这个过程中到底帮了什么忙。

## 1. 我们要解决什么问题

手写 CUDA GEMM 时，最容易分散注意力的不是算子本身，而是配套工作：
- 张量内存分配和拷贝
- 初始化输入
- 对照参考实现验证正确性

`examples/01_cutlass_utilities/cutlass_utilities.cu` 的价值是把这些“配套工作”标准化，让你把精力放在 GEMM 配置与数据流上。

## 2. 先认识四个关键工具

这个示例里最值得先掌握的是四个点：

1. `cutlass::HostTensor<>`
作用：同时管理 host/device 两侧内存，并提供 `sync_host()` 等同步接口。

2. `cutlass::reference::device::TensorFillRandomGaussian`
作用：在 device 侧初始化张量，生成可复现实验输入。

3. `cutlass::reference::host::Gemm<>`
作用：在 host 侧计算参考结果，作为 correctness baseline。

4. `cutlass::reference::host::TensorEquals`
作用：比较参考结果与 CUTLASS kernel 输出。

## 3. 最小 GEMM 的主流程（按代码顺序）

可以把这个示例抽象成 6 步：

1. 定义 `Gemm` 类型并准备参数（M/N/K、alpha/beta）
2. 用 `HostTensor` 创建 A/B/C
3. 在 device 侧填充随机输入
4. 调用 `cutlass::gemm::device::Gemm` 执行计算
5. 把数据同步到 host 并调用 host reference GEMM
6. 用 `TensorEquals` 比较结果

如果这 6 步能跑通，你就已经具备了后续所有“换 tile、换数据类型、换 epilogue”的基础框架。

## 4. 先做三组小实验

建议先只改参数，不改模板结构：

1. 维度实验：`M=N=K=128 -> 256`
2. 标量实验：`alpha=1,beta=0` 与 `alpha=1,beta=1`
3. 稳定性实验：固定 seed，多次运行确认结果一致

每次只改一个变量，并记录现象。这会比一次改很多参数更容易建立直觉。

## 5. 本节你应该带走什么

- 你可以不手写复杂内存管理，也能搭出可靠的 GEMM 实验框架。
- utilities 不是“可有可无”的辅助，而是快速迭代和排错的基础设施。
- 后续章节（layout、tile iterator、epilogue）都可以复用这套最小实验骨架。

下一篇会接着从这个骨架出发，进一步拆解数据布局与线程映射的关系。
