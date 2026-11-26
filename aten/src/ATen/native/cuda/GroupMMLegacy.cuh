#pragma once

#include <c10/cuda/CUDAException.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <ATen/native/cuda/GroupMMCommon.cuh>
#include <algorithm>
#include <type_traits>

namespace at::cuda::detail {

template <
    bool a_row_major,
    bool b_row_major,
    typename StrideA,
    typename StrideB,
    typename StrideOutput>
__global__ void unpack_grouped_gemm_strides(
    const StrideA* packed_a,
    const StrideB* packed_b,
    const StrideOutput* packed_output,
    int64_t* stride_a,
    int64_t* stride_b,
    int64_t* stride_output) {
  const int32_t group = threadIdx.x;
  stride_a[group] = cute::get<a_row_major ? 0 : 1>(packed_a[group]);
  stride_b[group] = cute::get<b_row_major ? 1 : 0>(packed_b[group]);
  stride_output[group] = cute::get<0>(packed_output[group]);
}

template <bool a_row_major, bool b_row_major>
void bf16bf16_grouped_gemm_impl_sm80(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<Tensor>& offs,
    Tensor& out) {
  using Element = cutlass::bfloat16_t;
  using LayoutA = std::conditional_t<
      a_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  using LayoutB = std::conditional_t<
      b_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  using Epilogue =
      cutlass::epilogue::thread::LinearCombination<Element, 1, float, float>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      1,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      1,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      2,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
  using PackedStrideA = std::conditional_t<
      a_row_major,
      cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>,
      cute::Stride<cute::Int<1>, int64_t, cute::Int<0>>>;
  using PackedStrideB = std::conditional_t<
      b_row_major,
      cute::Stride<cute::Int<1>, int64_t, cute::Int<0>>,
      cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;
  using PackedStrideOutput = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

  int32_t rows = mat_a.size(-2);
  int32_t depth = mat_a.size(-1);
  int32_t columns = mat_b.size(-1);
  int32_t group_count;
  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    group_count = offs->size(0);
    depth = -1;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    rows = -1;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    columns = -1;
  } else {
    group_count = mat_a.size(0);
  }
  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");
  if (group_count == 0) {
    return;
  }
  const size_t buffer_size = group_count *
      (3 * sizeof(Element*) + 3 * sizeof(int64_t) + sizeof(PackedStrideA) +
       sizeof(PackedStrideB) + sizeof(PackedStrideOutput) +
       sizeof(cutlass::gemm::GemmCoord));
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto metadata = allocator.allocate(buffer_size);
  auto input_a = reinterpret_cast<Element**>(metadata.get());
  auto input_b = input_a + group_count;
  auto output = input_b + group_count;
  auto stride_a = reinterpret_cast<int64_t*>(output + group_count);
  auto stride_b = stride_a + group_count;
  auto stride_output = stride_b + group_count;
  auto packed_a = reinterpret_cast<PackedStrideA*>(stride_output + group_count);
  auto packed_b = reinterpret_cast<PackedStrideB*>(packed_a + group_count);
  auto packed_output =
      reinterpret_cast<PackedStrideOutput*>(packed_b + group_count);
  auto problems =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(packed_output + group_count);
  auto make_strides = [](IntArrayRef values) {
    Strides result{};
    std::copy(values.begin(), values.end(), result.begin());
    return result;
  };
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  prepare_grouped_gemm_data<<<1, group_count, 0, stream>>>(
      reinterpret_cast<Element*>(mat_a.data_ptr()),
      reinterpret_cast<Element*>(mat_b.data_ptr()),
      reinterpret_cast<Element*>(out.data_ptr()),
      static_cast<float*>(nullptr),
      static_cast<float*>(nullptr),
      input_a,
      input_b,
      output,
      static_cast<float**>(nullptr),
      static_cast<float**>(nullptr),
      problems,
      packed_a,
      packed_b,
      packed_output,
      offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
      rows,
      columns,
      depth,
      make_strides(mat_a.strides()),
      make_strides(mat_b.strides()),
      make_strides(out.strides()),
      make_strides(mat_a.sizes()),
      make_strides(mat_b.sizes()),
      0,
      0,
      a_row_major,
      b_row_major);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (out.numel() == 0) {
    return;
  }
  unpack_grouped_gemm_strides<a_row_major, b_row_major>
      <<<1, group_count, 0, stream>>>(
          packed_a, packed_b, packed_output, stride_a, stride_b, stride_output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int sm_count =
      at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }
  TORCH_CHECK(sm_count > 0, "Expected a positive SM count for grouped GEMM");
  typename Gemm::Arguments arguments{
      problems,
      group_count,
      sm_count * 2,
      {1.0f, 0.0f},
      input_a,
      input_b,
      output,
      output,
      stride_a,
      stride_b,
      stride_output,
      stride_output};
  Gemm gemm;
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "cutlass cannot implement");
  auto workspace = allocator.allocate(Gemm::get_workspace_size(arguments));
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.get(), stream) ==
          cutlass::Status::kSuccess,
      "cutlass cannot initialize");
  auto status = gemm.run(stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass cannot run, error ",
      int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline void dispatch_bf16_grouped_kernel_sm80(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<Tensor>& offs,
    Tensor& out) {
  const bool a_row_major = mat_a.stride(-1) == 1;
  const bool b_row_major = mat_b.stride(-1) == 1;
  if (a_row_major && b_row_major) {
    bf16bf16_grouped_gemm_impl_sm80<true, true>(mat_a, mat_b, offs, out);
  } else if (a_row_major) {
    bf16bf16_grouped_gemm_impl_sm80<true, false>(mat_a, mat_b, offs, out);
  } else if (b_row_major) {
    bf16bf16_grouped_gemm_impl_sm80<false, true>(mat_a, mat_b, offs, out);
  } else {
    bf16bf16_grouped_gemm_impl_sm80<false, false>(mat_a, mat_b, offs, out);
  }
}

}
