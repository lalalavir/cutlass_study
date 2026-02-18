+++
title = "cutlass基础"
weight = 2
translationKey = "cutlass-basics"
slug = "cutlass基础"
+++

这个部分是 CUTLASS C++ 主线基础部分，目标是先把框架“跑起来、看明白、能复用”。

## 这里会介绍什么
- CUTLASS 在 CUDA 生态中的位置，以及 GEMM 主流程
- 常用 utilities 与张量容器的基本用法
- 从最小可运行 GEMM 出发，逐步建立调参与排错直觉
- 为后续 CuTe、融合算子、新架构导读打基础

## 大纲
1. 张量容器
2. 寄存器/共享内存可视化
3. Layout 与线程映射
4. Tile iterator 深入
5. Batched GEMM
6. Split-K 场景
7. Tensor Core 入门（Volta/Turing）
8. Epilogue 与激活融合
9. 双算子融合与工程取舍
10. Grouped 与调度直觉

## 阅读建议
1. 先从本页下方正文目录的第一篇开始
2. 每篇都先跑通示例，再看实现细节
3. 把关键参数变化记录成自己的实验笔记

## 正文目录
1. [01 张量容器:写一个最小GEMM]({{< relref "/cutlass-basics/s1-01-intro-utilities-minimal-gemm/index.md" >}})

后续每增加一个单元，都会在这里同步更新目录入口。

