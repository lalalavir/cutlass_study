---
title: "01 导论与张量容器：从 01_cutlass_utilities 写一个最小 GEMM"
slug: "s1-01-课程导论与张量容器-最小-gemm"
translationKey: "s1-01-intro-utilities-minimal-gemm"
weight: 1
date: 2026-02-17T10:00:00+08:00
draft: false
math: true
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

在这之后，我们会随机填充这三个矩阵。下面简单介绍一下`TensorFillRandomGaussian()`这个函数。该函数会把张量里的每个值都初始化为服从N(mean,stddev)的随机值。
第一个参数为张量的view()，第二个为随机数种子，第三个参数控制正态分布的均值，第四个参数控制正态分布的标准差，第五个参数进行精度控制，表示保留多少个小数二进制位，通过$q=\operatorname{round}(x\cdot 2^{\text{bits}})/2^{\text{bits}}$来计算，例如bits=0就表示截断到整数，这个参数主要用来做数值鲁棒性测试。最后一个`exclude_zero`用来表示是否避免生成0。

>TensorRef和TensorView的区别主要在于TensorView会保存extent,能做更多与边界相关的操作。

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

下一步，我们新建一个函数，用来实现基于cutlass的gemm计算
```cpp
#include "cutlass/gemm/device/gemm.h"

cudaError_t cutlass_hgemm_nn(
  int M,
  int N,
  int K,
  cutlass::half_t alpha,
  cutlass::half_t const *A,
  cutlass::layout::ColumnMajor::Stride::Index lda,
  cutlass::half_t const *B,
  cutlass::layout::ColumnMajor::Stride::Index ldb,
  cutlass::half_t beta,
  cutlass::half_t *C,
  cutlass::layout::ColumnMajor::Stride::Index ldc) 
  {

  return cudaSuccess;
}
```

下面我们会使用cutlass提供的`Gemm`类进行矩阵的运算。该类为cutlass的gemm封装器，负责把模板配置成具体kernel，提供统一启动接口。
有两层参数
- 核心参数: A,B C(D)的数据类型和布局
> 注意C和D的数据类型和布局需要一致，因为是实现的加法部分，下面会介绍
- 调优参数: 包括算子类型，目标架构，tile大小，split-k，gather/scatter等，会在之后章节介绍

```cpp
using Gemm = cutlass::gemm::device::Gemm<
	cutlass::half_t,                           // ElementA
	cutlass::layout::ColumnMajor,              // LayoutA
	cutlass::half_t,                           // ElementB
	cutlass::layout::ColumnMajor,              // LayoutB
	cutlass::half_t,                           // ElementOutput
	cutlass::layout::ColumnMajor               // LayoutOutput
>;

```

既然已经有了接口，我们下一步就是使用arguments来配置参数(问题规模+A/B/C/D指针与步长+epilogue参数)
我们矩阵计算的公式为$D=alpha*A*B+beta*C$，在此案例中，由于beta为0，因此可以简单的把D=C输入进去。
```cpp
Gemm gemm_op;

cutlass::Status status = gemm_op({
  {M, N, K},  //problem size
  {A, lda},  
  {B, ldb},  
  {C, ldc},
  {C, ldc},
  {alpha, beta}
	});

if (status != cutlass::Status::kSuccess)
 {
	return cudaErrorUnknown;
}

```

回到我们`TestCutlassGemm()`函数，此时就可以往里面传参数。由于我们的layout是`ColumnMajor`，因此地址偏移为$offset = col * ld + row$，
这边的步长我们就需要是`stride(0)`，表示列(第0维)。
```cpp
result = cutlass_hgemm_nn(
    M,
    N,
    K,
    alpha,
    A.device_data(),
    A.stride(0),
    B.device_data(),
    B.stride(0),
    beta,
    C_cutlass.device_data(),
    C_cutlass.stride(0)
  );

  if (result != cudaSuccess) {
    return result;
  }
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
