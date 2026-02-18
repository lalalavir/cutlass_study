---
title: "代码参考：完整 C++ 实现"
hidden: true
_build:
  list: never
draft: false
---

> 本页代码参考自 CUTLASS 官方 examples：
>
> - [00_basic_gemm/basic_gemm.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu)
> - [01_cutlass_utilities/cutlass_utilities.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/01_cutlass_utilities/cutlass_utilities.cu)
下面粘贴完整 C++ 代码：

```cpp
#include <iostream>
#include <vector>

#include "cuda.h"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/layout.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
cudaError_t cutlass_hgemm_nn(
	int M,
	int N,
	int K,
	cutlass::half_t alpha,
	cutlass::half_t const* A,
	cutlass::layout::ColumnMajor::Stride::Index lda,
	cutlass::half_t const* B,
	cutlass::layout::ColumnMajor::Stride::Index ldb,
	cutlass::half_t beta,
	cutlass::half_t* C,
	cutlass::layout::ColumnMajor::Stride::Index ldc)
{

	// Define the GEMM operation
	using Gemm = cutlass::gemm::device::Gemm<
		cutlass::half_t,                           // ElementA
		cutlass::layout::ColumnMajor,              // LayoutA
		cutlass::half_t,                           // ElementB
		cutlass::layout::ColumnMajor,              // LayoutB
		cutlass::half_t,                           // ElementOutput
		cutlass::layout::ColumnMajor               // LayoutOutput
	>;

	Gemm gemm_op;

	cutlass::Status status = gemm_op({
	  {M, N, K},
	  {A, lda},
	  {B, ldb},
	  {C, ldc},
	  {C, ldc},
	  {alpha, beta}
		});

	if (status != cutlass::Status::kSuccess) {
		return cudaErrorUnknown;
	}

	return cudaSuccess;
}

cudaError_t TestCutlassGemm(int M,int K,int N,cutlass::half_t alpha, cutlass::half_t beta)
{
	cudaError_t result;

	cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));
	cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));
	cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M, N));
	cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M, N));

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

	if (result != cudaSuccess)
	{
		std::cerr << "Error - CUTLASS GEMM launch failed" << std::endl;
		return result;
	}

	A.sync_host();
	B.sync_host();
	C_cutlass.sync_host();
	C_reference.sync_host();

	cutlass::reference::host::Gemm<cutlass::half_t,cutlass::layout::ColumnMajor,
	cutlass::half_t,cutlass::layout::ColumnMajor,cutlass::half_t,cutlass::layout::ColumnMajor,
	cutlass::half_t,cutlass::half_t
	> gemm_ref;

	gemm_ref(
		{ M, N, K },                          // problem size (type: cutlass::gemm::GemmCoord)
		alpha,                              // alpha        (type: cutlass::half_t)
		A.host_ref(),                       // A            (type: TensorRef<half_t, ColumnMajor>)
		B.host_ref(),                       // B            (type: TensorRef<half_t, ColumnMajor>)
		beta,                               // beta         (type: cutlass::half_t)
		C_reference.host_ref()              // C            (type: TensorRef<half_t, ColumnMajor>)
	);

	if (!cutlass::reference::host::TensorEquals(
		C_reference.host_view(),
		C_cutlass.host_view()))
	{
		std::cerr << "Error - CUTLASS GEMM kernel differs from reference" << std::endl;
		result = cudaErrorUnknown;
	}
	
	return result;
}

int main()
{
	
	auto result = TestCutlassGemm(64,32,48,1.0_hf,0.0_hf);

	if (result==cudaSuccess)
	{
		std::cout << "Pass" << std::endl;
	}

	return result == cudaSuccess ? EXIT_SUCCESS :EXIT_FAILURE;
}

```


