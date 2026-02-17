---
title: "S1-01 课程导论与张量容器：从 01_cutlass_utilities 写一个最小 GEMM"
slug: "s1-01-课程导论与张量容器-最小-gemm"
translationKey: "s1-01-intro-utilities-minimal-gemm"
weight: 1
date: 2026-02-17T10:00:00+08:00
draft: false
---

这一篇我们用cutlass提供的工具，将最小GEMM路径走通

## 1. 先认识四个关键的类

这个示例里最值得先掌握的是四个类：

1. `cutlass::HostTensor<>`
作用：同时管理 host/device 两侧内存。

2. `cutlass::reference::device::TensorFillRandomGaussian`
作用：在 device 侧初始化张量，生成可复现实验输入。

3. `cutlass::reference::host::Gemm<>`
作用：在 host 侧计算参考结果，作为 correctness baseline。

4. `cutlass::reference::host::TensorEquals`
作用：比较参考结果与 CUTLASS kernel 输出。

>reference这个命名空间主要是参考实现和验证工具，支持host端和device端

## 2. 最小 GEMM 的主流程

首先定义一个host端函数，用来测试矩阵乘法
```cpp
#include "cutlass/cutlass.h" 
#include "cutlass/numeric_types.h" //half_t
#include "cutlass/layout/layout.h" //layout
#include "cutlass/util/host_tensor.h"  //host_tensor
#include "cutlass/util/reference/host/gemm.h"  //Gemm()
#include "cutlass/util/reference/device/tensor_fill.h" //TensorFillRandomGaussian()
cudaError_t TestCutlassGemm(int M,int K,int N,cutlass::half_t alpha, cutlass::half_t beta)
{
	cudaError_t result;

	return result;
}
```
然后我们创建`HostTensor`。`HostTensor`的类模板有两个参数：`Element`和`Layout`。`Element`表示张量元素类型,例如`float`、`half_t`等，`layout`表示逻辑坐标到线性内存地址的映射，包括列主序和行主序。它决定了 `stride`、`TensorCoord` 解释方式，以及后续 `iterator` 如何访问。
```cpp
cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));
cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));
cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M, N));
cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M, N));
```

在这之后，我们会随机填充这三个矩阵。
```cpp
uint64_t seed = 2000;
cutlass::half_t mean = 0.0_hf;
cutlass::half_t stddev = 5.0_hf;

int bits_less_than_one = 0;

cutlass::reference::device::TensorFillRandomGaussian(
	A.device_view(),
	seed,
	mean,
	stddev,
	bits_less_than_one
);

cutlass::reference::device::TensorFillRandomGaussian(
	B.device_view(),
	seed *500,
	mean,
	stddev,
	bits_less_than_one
);
cutlass::reference::device::TensorFillRandomGaussian(
	C_cutlass.device_view(),
	seed * 300,
	mean,
	stddev,
	bits_less_than_one
);

cutlass::device_memory::copy_device_to_device(C_reference.device_data(), C_cutlass.device_data(), C_cutlass.capacity());
```

## 3. 先做三组小实验

建议先只改参数，不改模板结构：

1. 维度实验：`M=N=K=128 -> 256`
2. 标量实验：`alpha=1,beta=0` 与 `alpha=1,beta=1`
3. 稳定性实验：固定 seed，多次运行确认结果一致

每次只改一个变量，并记录现象。这会比一次改很多参数更容易建立直觉。

## 4. 本节你应该带走什么

- 你可以不手写复杂内存管理，也能搭出可靠的 GEMM 实验框架。
- utilities 不是“可有可无”的辅助，而是快速迭代和排错的基础设施。
- 后续章节（layout、tile iterator、epilogue）都可以复用这套最小实验骨架。

下一篇会接着从这个骨架出发，进一步拆解数据布局与线程映射的关系。
