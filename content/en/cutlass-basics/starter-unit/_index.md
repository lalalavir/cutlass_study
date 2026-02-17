+++
title = "Starter Unit: Framework Overview and Minimal GEMM"
weight = 1
translationKey = "starter-unit"
slug = "starter-unit-framework-overview-minimal-gemm"
+++

Goal: build a quick mental model of CUTLASS and complete one minimal GEMM example.

## 1. Framework overview
- `cutlass/include/`: core templates and operator definitions
- `cutlass/examples/`: runnable examples and learning entry points
- `cutlass/tools/`: helper scripts and utilities

## 2. Minimal GEMM with utilities
Reference: `cutlass/examples/01_cutlass_utilities/cutlass_utilities.cu`

```cpp
#include "cutlass/util/host_tensor.h"

// Pseudocode: prepare A/B/C -> run GEMM -> verify results
int main() {
  // 1) allocate matrices with HostTensor
  // 2) initialize inputs
  // 3) launch a basic GEMM configuration
  // 4) validate output
  return 0;
}
```

## 3. First experiments
1. Change M/N/K from `128` to `256`.
2. Compare behavior across data types (`float` vs `half`).
3. Record one runtime as your baseline.
