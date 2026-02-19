---
title: "02 寄存器与共享内存可视化：从 02_dump_reg_shmem 看数据搬运路径"
slug: "s1-02-reg-shmem-visualization"
translationKey: "s1-02-reg-shmem-visualization"
weight: 2
date: 2026-02-19T10:00:00+08:00
draft: false
---

这一篇我们不追求高性能参数，而是先把 CUTLASS 里的数据搬运路径“看见”。
目标是用 `examples/02_dump_reg_shmem` 看清三件事：
1. 全局内存数据怎样被线程映射读入 fragment（寄存器）
2. fragment 怎样写入 shared memory
3. dump 输出怎样对应回矩阵坐标

## 1. 本节目标与前置知识

你在读这一篇时，建议已经跑通过：
- 本单元上一节（最小 GEMM）

这篇的定位是“调试与建立直觉”，不是“做最终 kernel 调优”。
当你后续读 `Tile Iterator`、`Layout`、`ThreadMap` 时，这篇能提供一个可观察的基准。


## 2. 先看整条数据流

从主机到设备的主流程可以先记成一行：

`HostTensor -> sync_device -> GmemIterator.load(frag) -> SmemIterator.store(frag) -> dump`

对应到代码：
- host 侧构造 `HostTensor<Element, Layout>` 并 `BlockFillSequential`
- kernel 里用 `PredicatedTileIterator` 从 gmem 读到 `Fragment`
- 再用 `RegularTileIterator` 把 `Fragment` 写到 shared memory
- 最后调用 dump API 可视化

## 3. kernel 主流程拆解
首先我们先定义一个矩阵，然后用`BlockFillSequential()`填充一下,该函数按照`layout`选择的布局，以等差数列填充整个矩阵。
```cpp
#include "cutlass/util/reference/host/tensor_fill.h" //BlockFillSequential()

#define EXAMPLE_MATRIX_ROW 64
#define EXAMPLE_MATRIX_COL 32

cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> matrix(
	{EXAMPLE_MATRIX_ROW,EXAMPLE_MATRIX_COL});
cutlass::reference::host::BlockFillSequential(matrix.host_data(), matrix.capacity());

std::cout << "Matrix:\n" << matrix.host_view() << "\n";
matrix.sync_device();
```

接下来我们来介绍一下`ThreadMap`和`TileIterator`。简单来说，`ThreadMap`负责制定策略，`TileIterator`负责实现。
我们来填一个`ThreadMap`的模板。通常来说，我们默认选择`PitchLinearWarpRakedThreadMap`, 因为它兼顾了协同和布局适配。一些其他的`ThreadMap`我们会在之后的章节介绍。
该类模板有四个参数：
- tile形状
- 线程数
- warp 内线程排布
- 每个线程一次访问连续多少个元素(向量化)

```cpp
#include "cutlass/transform/pitch_linear_thread_map.h"

using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
	cutlass::layout::PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, 32,
	cutlass::layout::PitchLinearShape<8, 4>, 8
>;
```

下面我们来填两个`TileIterattor`的模板。对于global memory访问，我们常用`PredicatedTileIterator`，因为其带有边界保护;对于shared memory访问，我们常用`RegularTileIterator`，因为其假定规则访问，效率更高。

`PredicatedTileIterator`的参数如下(默认的参数会在之后章节介绍)
- Shape:当前iterator处理的tile逻辑大小
- Element:元素类型
- Layout:内存布局
- AdvanceRank:++iterator沿哪个逻辑维推进(0或1)。
- ThreadMap:线程到Tile元素的映射规则
我们重点介绍一下AdvanceRank。如果外层我们扫描的是列，那么AdvanceRank=1;如果外层我们扫描的是行，那么AdvanceRank=0。

对于`RegularTileIterator`，参数基本和`PredicatedTileIterator`一致，但需要注意的是，两个的layout表示的不一样。
在`PredicatedTileIterator`中，layout表示张量在全局内存里的逻辑布局。
在`RegularTileIterator`中，layout表示目标内存(通常是shared memory)里的物理排布策略，因此我们用该布局`ColumnMajorTensorOpMultiplicandCongruous<Interleave, Crosswise>`。我们可以这么理解这个布局:逻辑上还是列主序，但是物理上重排了，使得bank conflict更少。

```cpp
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
	cutlass::layout::PitchLinearShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, 32,
	cutlass::layout::PitchLinearShape<8, 4>, 8
>;

using GmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
	cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, cutlass::half_t,
	cutlass::layout::ColumnMajor, 1, ThreadMap
>;

typename GmemIterator::Params params(matrix.layout());

using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
	cutlass::MatrixShape<EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL>, cutlass::half_t,
	cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, 1, ThreadMap
>;

```

接下来我们写一个kernel，用来打印调试信息。

```cpp
template <typename Element, typename GmemIterator, typename SmemIterator>
__global__ void kernel_dump(typename GmemIterator::Params params,
	typename GmemIterator::TensorRef ref)
{
	extern __shared__ Element shared_storage[]; //allocate shared memory

	int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x; //thread_location
}
```

在这个函数内，我们首先构造一个`GmemIterator`并用它来访问矩阵。
GmemIterator的构造函数的参数如下：
- Params:布局参数
- Ref:全局内存起始指针
- Extent: 逻辑矩阵范围
- Thread_id: 线程 id，用于 `ThreadMap` 计算该线程负责的元素起点。
然后我们会声明一个基于该`iterator`对应的寄存器片段类型实例，即`fragment`。这个类型的用途是作为load/store的中间容器。
`dump_fragment()`函数用于打印寄存器里的`fragment`，经常用于回答两个问题:每个线程到底读到了哪些元素; ThreadMap/Iterator 映射是否符合预期。该函数参数可以查看自带的英文注释。

```cpp
GmemIterator gmem_iterator(params, ref.data(),
                             {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL},
                             tb_thread_id);

  typename GmemIterator::Fragment frag;

  frag.clear();
  gmem_iterator.load(frag);

  // Call dump_fragment() with different parameters.
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nAll threads dump all the elements:\n");
  cutlass::debug::dump_fragment(frag);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps all the elements:\n");
  cutlass::debug::dump_fragment(frag,  1);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps first 16 elements:\n");
  cutlass::debug::dump_fragment(frag,  1,  16);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("\nFirst thread dumps first 16 elements with a stride of 8:\n");
  cutlass::debug::dump_fragment(frag,  1,  16,  8);
```

我们会先将`fragment`读取到shared memory中。对此，我们需要构造一个`SmemIterator`。该构造函数接受一个`TensorRef`和`thread_id`。
其中`TensorRef`接受一个指针(共享内存起始地址)和布局，注意这边布局我们采用`packed()`，目的是使该layout生成对应声明`SmemIterator::Layout`的那套layout。
在将`fragment`存储到shared memory后，我们可以用`dump_shmem`打印shared memory里面的内容。

```cpp
  SmemIterator smem_iterator(
      typename SmemIterator::TensorRef(
          {shared_storage, SmemIterator::Layout::packed(
                               {EXAMPLE_MATRIX_ROW, EXAMPLE_MATRIX_COL})}),
      tb_thread_id);

  smem_iterator.store(frag);
  
  if (threadIdx.x == 0 && blockIdx.x == 0) printf("\nDump all the elements:\n");
cutlass::debug::dump_shmem(shared_storage,
	EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

if (threadIdx.x == 0 && blockIdx.x == 0)
	printf("\nDump all the elements with a stride of 8:\n");
cutlass::debug::dump_shmem(
	shared_storage, EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL, 8);
```

最后我们回到`main()`函数中，启动该kernel即可。
```cpp
dim3 grid(1, 1);
dim3 block(32, 1, 1);

int smem_size =
	int(sizeof(cutlass::half_t) * EXAMPLE_MATRIX_ROW * EXAMPLE_MATRIX_COL);

kernel_dump<cutlass::half_t, GmemIterator, SmemIterator>
	<< <grid, block, smem_size, 0 >> > (params, matrix.device_ref());

cudaError_t result = cudaDeviceSynchronize();

if (result != cudaSuccess) 
{
	std::cout << "Failed" << std::endl;
}

return (result == cudaSuccess ? 0 : -1);
```
## 4. 本节你应该学习什么

- 你已经能把 CUTLASS 的数据搬运路径可视化出来
- 你知道 fragment 是线程私有寄存器视图，不是全局矩阵切片本体
- 你知道为什么 gmem 和 smem 常用不同 iterator
- 你有了一套“先观察再调参”的调试流程

## 代码参考

[查看完整 C++ 代码]({{< relref "/cutlass-basics/s1-02-reg-shmem-visualization-code-reference/" >}})
## 参考来源

- [CUTLASS Example 02: dump_reg_shmem.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/02_dump_reg_shmem/dump_reg_shmem.cu)

