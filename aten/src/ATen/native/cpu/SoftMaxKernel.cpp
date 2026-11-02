#include <memory>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cpu/SoftmaxKernel.h>

#include <ATen/native/cpu/LogSoftmaxKernelImpl.h>

#include <algorithm>
#include <atomic>
#include <iterator>
#include <numeric>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

// [Note AVX-SSE transitions] In general we avoid calls into cmath for code
// compiled with AVX/AVX2 This is because of SSE-AVX transitions and a bug in
// Glibc2.23 See https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280
//
// On grainsize: The grainsize is chosen to roughly get GRAIN_SIZE number of
// computations per task. Each task works across dim_size elements. 16 should be
// a very rough approximation of the number of computations per dim_size element
// by counting simple computations (*, +, -) as 1 and exp or log as 4.
//
// We use a chunk size such that it'd fit in L1D.

namespace at::native {
namespace {
template <typename scalar_t>
inline void _vec_log_softmax_lastdim(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  const auto chunk_size = vec_log_softmax_lastdim_chunk_size<scalar_t>(
      at::internal::GRAIN_SIZE,
      outer_size,
      dim_size);
  // Note: grain_size value of 0
  // We don't change the number of OpenMP threads in the OpenMP thread-pool,
  // so some threads do useful work, while others don't.
  // We can simply use grain_size of 0 & rely upon invoke_parallel to distribute
  // work among threads in an equitable manner. We compute CHUNK_SIZE to ensure
  // each thread's computations would be efficient.
  parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
    serial_vec_log_softmax_lastdim_range(
        input_data_base,
        output_data_base,
        dim_size,
        chunk_size,
        begin,
        end);
  });
}

template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_lastdim(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  // See Note: grain_size value of 0
  parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* input_data = input_data_base + i * dim_size;
      scalar_t* output_data = output_data_base + i * dim_size;
      scalar_t max_input = vec::reduce_all<scalar_t>(
          [](Vec& x, Vec& y) { return vec::maximum(x, y); },
          input_data,
          dim_size);
      vec::map(
          [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
          output_data,
          input_data,
          dim_size);
      scalar_t tmp_sum = vec::reduce_all<scalar_t>(
          [](Vec x, Vec y) { return x + y; }, output_data, dim_size);
      tmp_sum = 1 / tmp_sum;
      vec::map(
          [tmp_sum](Vec x) { return x * Vec(tmp_sum); },
          output_data,
          output_data,
          dim_size);
    }
  });
}

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_lastdim(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  // See Note: grain_size value of 0
  parallel_for(0, outer_size, 0, [&](int64_t begin, int64_t end) {
    // thread local temp buffer.
    auto buffer = std::make_unique<float []>(dim_size);
    float* buffer_data = buffer.get();

    for (const auto i : c10::irange(begin, end)) {
      const scalar_t* input_data = input_data_base + i * dim_size;
      scalar_t* output_data = output_data_base + i * dim_size;
      // reduce to max and cache float input data
      fVec max_fvec = fVec(-std::numeric_limits<float>::infinity());
      int64_t d0 = 0;
      for (; d0 < dim_size - (dim_size % Vec::size()); d0 += Vec::size()) {
        Vec data_vec = Vec::loadu(input_data + d0);
        auto [data_fvec0, data_fvec1] = vec::convert_to_float<scalar_t>(data_vec);
        max_fvec = vec::maximum(max_fvec, data_fvec0);
        max_fvec = vec::maximum(max_fvec, data_fvec1);
        data_fvec0.store(buffer_data + d0);
        data_fvec1.store(buffer_data + d0 + fVec::size());
      }
      float max_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return vec::maximum(x, y); }, max_fvec);
      for (; d0 < dim_size; d0++) {
        float data_val = input_data[d0];
        max_val = std::max(max_val, data_val);
        buffer_data[d0] = data_val;
      }

      // map (x - max).exp() and reduce to sum
      fVec sum_fvec = fVec(float(0));
      int64_t d1 = 0;
      for (; d1 < dim_size - (dim_size % fVec::size()); d1 += fVec::size()) {
        fVec data_fvec = (fVec::loadu(buffer_data + d1) - fVec(max_val)).exp();
        sum_fvec += data_fvec;
        data_fvec.store(buffer_data + d1);
      }
      float sum_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, sum_fvec);
      for (; d1 < dim_size; d1++) {
        float data_val = std::exp(buffer_data[d1] - max_val);
        sum_val += data_val;
        buffer_data[d1] = data_val;
      }

      sum_val = 1 / sum_val;
      int64_t d2 = 0;
      for (; d2 < dim_size - (dim_size % Vec::size()); d2 += Vec::size()) {
        fVec out_fvec0 = fVec::loadu(buffer_data + d2) * fVec(sum_val);
        fVec out_fvec1 = fVec::loadu(buffer_data + d2 + fVec::size()) * fVec(sum_val);
        Vec out_vec = vec::convert_from_float<scalar_t>(out_fvec0, out_fvec1);
        out_vec.store(output_data + d2);
      }
      for (; d2 < dim_size; d2++) {
        output_data[d2] = scalar_t(buffer_data[d2] * sum_val);
      }
    }
  });
}

template <typename scalar_t>
inline vec::Vectorized<at::opmath_type<scalar_t>> log_softmax_backward_value(
    vec::Vectorized<at::opmath_type<scalar_t>> gradient,
    vec::Vectorized<at::opmath_type<scalar_t>> output,
    vec::Vectorized<at::opmath_type<scalar_t>> sum,
    vec::Vectorized<at::opmath_type<scalar_t>> other_sum,
    vec::Vectorized<at::opmath_type<scalar_t>> dominant_count) {
  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<opmath_t>;
  auto result = gradient - output.exp() * sum;
  const auto finite_sum = sum.abs() < Vec(std::numeric_limits<opmath_t>::infinity());
  const auto dominant = (output > Vec(opmath_t(-0.69314718055994530942))) &
      (output <= Vec(opmath_t(0))) & finite_sum &
      (other_sum.abs() < Vec(std::numeric_limits<opmath_t>::infinity())) &
      (dominant_count == Vec(opmath_t(1)));
  if (dominant.zero_mask() != (1 << Vec::size()) - 1) {
    auto stable = gradient * (-output.expm1()) - output.exp() * other_sum;
    result = Vec::blendv(result, stable, dominant);
  }
  const auto tiny = (output < Vec(std::log(std::numeric_limits<opmath_t>::min()))) &
      (output > Vec(-std::numeric_limits<opmath_t>::infinity())) & finite_sum;
  if (tiny.zero_mask() != (1 << Vec::size()) - 1) {
    auto root = (output * Vec(opmath_t(0.5))).exp();
    result = Vec::blendv(result, gradient - (sum * root) * root, tiny);
  }
  return result;
}

template <typename scalar_t>
inline at::opmath_type<scalar_t> log_softmax_backward_scalar(
    at::opmath_type<scalar_t> gradient,
    at::opmath_type<scalar_t> output,
    at::opmath_type<scalar_t> sum,
    at::opmath_type<scalar_t> other_sum,
    at::opmath_type<scalar_t> dominant_count) {
  using opmath_t = at::opmath_type<scalar_t>;
  if (std::isfinite(sum) && std::isfinite(other_sum) && dominant_count == 1 &&
      output > opmath_t(-0.69314718055994530942) && output <= 0) {
    return gradient * (-std::expm1(output)) - std::exp(output) * other_sum;
  }
  if (std::isfinite(sum) && std::isfinite(output) &&
      output < std::log(std::numeric_limits<opmath_t>::min())) {
    const auto root = std::exp(output * opmath_t(0.5));
    return gradient - (sum * root) * root;
  }
  return gradient - std::exp(output) * sum;
}

template <typename scalar_t, bool log_softmax>
inline void _vec_host_softmax_backward_lastdim(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<at::opmath_type<scalar_t>>;
  // See Note: grain_size value of 0
  parallel_for(
      0,
      outer_size,
      0,
      [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          scalar_t* grad_input_data = grad_input_data_base + i * dim_size;
          const scalar_t* grad_data = grad_data_base + i * dim_size;
          const scalar_t* output_data = output_data_base + i * dim_size;
          if constexpr (log_softmax) {
            auto sum = vec::reduce_all<scalar_t>(
                [](Vec& x, Vec& y) { return x + y; }, grad_data, dim_size);
            auto dominant_count = vec::map_reduce_all<scalar_t>(
                [](Vec output) {
                  const auto dominant = (output > Vec(-0.69314718055994530942)) &
                      (output <= Vec(0));
                  return Vec::blendv(Vec(0), Vec(1), dominant);
                },
                [](Vec left, Vec right) { return left + right; },
                output_data, dim_size);
            auto other_sum = sum;
            if (dominant_count == 1) {
              other_sum = vec::map2_reduce_all<scalar_t>(
                  [](Vec gradient, Vec output) {
                    const auto dominant = (output > Vec(-0.69314718055994530942)) &
                        (output <= Vec(0));
                    return Vec::blendv(gradient, Vec(0), dominant);
                  },
                  [](Vec left, Vec right) { return left + right; },
                  grad_data, output_data, dim_size);
            }
            vec::map2(
                [sum, other_sum, dominant_count](Vec gradient, Vec output) {
                  return log_softmax_backward_value<scalar_t>(
                      gradient, output, Vec(sum), Vec(other_sum), Vec(dominant_count));
                },
                grad_input_data,
                grad_data,
                output_data,
                dim_size);
          } else {
            auto sum = vec::map2_reduce_all<scalar_t>(
                [](Vec x, Vec y) { return x * y; },
                [](Vec x, Vec y) { return x + y; },
                grad_data,
                output_data,
                dim_size);
            vec::map2(
                [sum](Vec x, Vec y) { return (x - Vec(sum)) * y; },
                grad_input_data,
                grad_data,
                output_data,
                dim_size);
          }
        }
      });
}

template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t outer_stride = dim_size * inner_size;
  int64_t BLOCK_SIZE = 128 * 1024;
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(
      BLOCK_SIZE / dim_size / sizeof(scalar_t), Vec::size());
  MAX_CHUNK_SIZE = MAX_CHUNK_SIZE / Vec::size() * Vec::size();
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, inner_size);
  int64_t num_chunks = divup(inner_size, CHUNK_SIZE);
  // See Note: grain_size value of 0
  parallel_for(
      0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
        // thread local temp buffer that holds vertical sum result
        auto buffer = std::make_unique<scalar_t[]>(CHUNK_SIZE);
        scalar_t* tmp_sum_data = buffer.get();

        for (int64_t i = begin; i < end; i++) {
          int64_t outer_idx = i / num_chunks;
          int64_t k = i % num_chunks;
          int64_t inner_idx_begin = k * CHUNK_SIZE;
          int64_t size = std::min(CHUNK_SIZE, inner_size - inner_idx_begin);

          // init
          Vec zero_vec = Vec(scalar_t(0));
          int64_t d0 = 0;
          for (; d0 < size - (size % Vec::size()); d0 += Vec::size()) {
            zero_vec.store(tmp_sum_data + d0);
          }
          for (; d0 < size; d0++) {
            tmp_sum_data[d0] = scalar_t(0);
          }

          // compute sum of grad_output * output
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            int64_t offset = outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            const scalar_t* grad_output_ptr = grad_output_data_base + offset;
            const scalar_t* output_ptr = output_data_base + offset;

            int64_t d1 = 0;
            for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
              Vec grad_output_vec = Vec::loadu(grad_output_ptr + d1);
              Vec output_vec = Vec::loadu(output_ptr + d1);
              Vec sum_vec = Vec::loadu(tmp_sum_data + d1);
              sum_vec += grad_output_vec * output_vec;
              sum_vec.store(tmp_sum_data + d1);
            }
            for (; d1 < size; d1++) {
              tmp_sum_data[d1] += grad_output_ptr[d1] * output_ptr[d1];
            }
          }

          // compute output * (grad_output - sum)
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            int64_t offset = outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            const scalar_t* grad_output_ptr = grad_output_data_base + offset;
            const scalar_t* output_ptr = output_data_base + offset;
            scalar_t* grad_input_ptr = grad_input_data_base + offset;

            int64_t d2 = 0;
            for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
              Vec grad_output_vec = Vec::loadu(grad_output_ptr + d2);
              Vec output_vec = Vec::loadu(output_ptr + d2);
              Vec sum_vec = Vec::loadu(tmp_sum_data + d2);
              Vec grad_input_vec = output_vec * (grad_output_vec - sum_vec);
              grad_input_vec.store(grad_input_ptr + d2);
            }
            for (; d2 < size; d2++) {
              grad_input_ptr[d2] = output_ptr[d2] * (grad_output_ptr[d2] - tmp_sum_data[d2]);
            }
          }
        }
      });
}

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  int64_t outer_stride = dim_size * inner_size;
  int64_t BLOCK_SIZE = 128 * 1024;
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(
      BLOCK_SIZE / dim_size / sizeof(scalar_t), Vec::size());
  MAX_CHUNK_SIZE = MAX_CHUNK_SIZE / Vec::size() * Vec::size();
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, inner_size);
  int64_t num_chunks = divup(inner_size, CHUNK_SIZE);
  // See Note: grain_size value of 0
  parallel_for(
      0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
        // thread local temp buffer that holds vertical sum result
        auto buffer = std::make_unique<float[]>(CHUNK_SIZE);
        float* tmp_sum_data = buffer.get();

        // thread local buffer that holds grad_output and output data in float32
        auto grad_output_buffer = std::make_unique<float[]>(dim_size * CHUNK_SIZE);
        float* grad_output_buffer_data = grad_output_buffer.get();

        auto output_buffer = std::make_unique<float[]>(dim_size * CHUNK_SIZE);
        float* output_buffer_data = output_buffer.get();

        for (int64_t i = begin; i < end; i++) {
          int64_t outer_idx = i / num_chunks;
          int64_t k = i % num_chunks;
          int64_t inner_idx_begin = k * CHUNK_SIZE;
          int64_t size = std::min(CHUNK_SIZE, inner_size - inner_idx_begin);

          // init
          fVec zero_fvec = fVec(float(0));
          int64_t d0 = 0;
          for (; d0 < size - (size % Vec::size()); d0 += Vec::size()) {
            zero_fvec.store(tmp_sum_data + d0);
            zero_fvec.store(tmp_sum_data + d0 + fVec::size());
          }
          for (; d0 < size; d0++) {
            tmp_sum_data[d0] = float(0);
          }

          // compute sum of grad_output * output
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            int64_t offset = outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            const scalar_t* grad_output_ptr = grad_output_data_base + offset;
            const scalar_t* output_ptr = output_data_base + offset;
            float* grad_output_buffer_ptr =
                grad_output_buffer_data + dim_idx * CHUNK_SIZE;
            float* output_buffer_ptr =
                output_buffer_data + dim_idx * CHUNK_SIZE;

            int64_t d1 = 0;
            for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
              Vec grad_output_vec = Vec::loadu(grad_output_ptr + d1);
              auto [grad_output_fvec0, grad_output_fvec1] =
                  vec::convert_to_float<scalar_t>(grad_output_vec);
              Vec output_vec = Vec::loadu(output_ptr + d1);
              auto [output_fvec0, output_fvec1] =
                  vec::convert_to_float<scalar_t>(output_vec);
              fVec sum_fvec0 = fVec::loadu(tmp_sum_data + d1);
              fVec sum_fvec1 = fVec::loadu(tmp_sum_data + d1 + fVec::size());
              sum_fvec0 += grad_output_fvec0 * output_fvec0;
              sum_fvec1 += grad_output_fvec1 * output_fvec1;
              sum_fvec0.store(tmp_sum_data + d1);
              sum_fvec1.store(tmp_sum_data + d1 + fVec::size());

              // cache the 'converted' float grad_output and output
              grad_output_fvec0.store(grad_output_buffer_ptr + d1);
              grad_output_fvec1.store(
                  grad_output_buffer_ptr + d1 + fVec::size());
              output_fvec0.store(output_buffer_ptr + d1);
              output_fvec1.store(output_buffer_ptr + d1 + fVec::size());
            }
            for (; d1 < size; d1++) {
              float grad_output_val = float(grad_output_ptr[d1]);
              float output_val = float(output_ptr[d1]);
              tmp_sum_data[d1] += grad_output_val * output_val;
              grad_output_buffer_ptr[d1] = grad_output_val;
              output_buffer_ptr[d1] = output_val;
            }
          }

          // compute output * (grad_output - sum)
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            scalar_t* grad_input_ptr = grad_input_data_base +
                outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            float* grad_output_buffer_ptr =
                grad_output_buffer_data + dim_idx * CHUNK_SIZE;
            float* output_buffer_ptr =
                output_buffer_data + dim_idx * CHUNK_SIZE;

            int64_t d2 = 0;
            for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
              fVec sum_fvec0 = fVec::loadu(tmp_sum_data + d2);
              fVec sum_fvec1 = fVec::loadu(tmp_sum_data + d2 + fVec::size());
              fVec grad_output_fvec0 = fVec::loadu(grad_output_buffer_ptr + d2);
              fVec grad_output_fvec1 =
                  fVec::loadu(grad_output_buffer_ptr + d2 + fVec::size());
              fVec output_fvec0 = fVec::loadu(output_buffer_ptr + d2);
              fVec output_fvec1 =
                  fVec::loadu(output_buffer_ptr + d2 + fVec::size());
              fVec grad_input_fvec0 =
                  output_fvec0 * (grad_output_fvec0 - sum_fvec0);
              fVec grad_input_fvec1 =
                  output_fvec1 * (grad_output_fvec1 - sum_fvec1);
              Vec grad_input_vec =
                  vec::convert_from_float<scalar_t>(grad_input_fvec0, grad_input_fvec1);
              grad_input_vec.store(grad_input_ptr + d2);
            }
            for (; d2 < size; d2++) {
              grad_input_ptr[d2] = output_buffer_ptr[d2] * (grad_output_buffer_ptr[d2] - tmp_sum_data[d2]);
            }
          }
        }
      });
}

template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_log_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t outer_stride = dim_size * inner_size;
  int64_t BLOCK_SIZE = 128 * 1024;
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(
      BLOCK_SIZE / dim_size / sizeof(scalar_t), Vec::size());
  MAX_CHUNK_SIZE = MAX_CHUNK_SIZE / Vec::size() * Vec::size();
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, inner_size);
  int64_t num_chunks = divup(inner_size, CHUNK_SIZE);
  // See Note: grain_size value of 0
  parallel_for(
      0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
        // thread local temp buffer that holds vertical sum result
        auto buffer = std::make_unique<scalar_t[]>(CHUNK_SIZE * 3);
        scalar_t* tmp_sum_data = buffer.get();
        scalar_t* tmp_other_sum_data = buffer.get() + CHUNK_SIZE;
        scalar_t* tmp_dominant_count_data = buffer.get() + CHUNK_SIZE * 2;

        for (int64_t i = begin; i < end; i++) {
          int64_t outer_idx = i / num_chunks;
          int64_t k = i % num_chunks;
          int64_t inner_idx_begin = k * CHUNK_SIZE;
          int64_t size = std::min(CHUNK_SIZE, inner_size - inner_idx_begin);

          // init
          Vec zero_vec = Vec(scalar_t(0));
          int64_t d0 = 0;
          for (; d0 < size - (size % Vec::size()); d0 += Vec::size()) {
            zero_vec.store(tmp_sum_data + d0);
            zero_vec.store(tmp_other_sum_data + d0);
            zero_vec.store(tmp_dominant_count_data + d0);
          }
          for (; d0 < size; d0++) {
            tmp_sum_data[d0] = scalar_t(0);
            tmp_other_sum_data[d0] = scalar_t(0);
            tmp_dominant_count_data[d0] = scalar_t(0);
          }

          // compute sum of grad_output
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            const scalar_t* grad_output_ptr = grad_output_data_base +
                outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            const scalar_t* output_ptr = output_data_base +
                outer_idx * outer_stride + dim_idx * inner_size + inner_idx_begin;

            int64_t d1 = 0;
            for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
              Vec grad_output_vec = Vec::loadu(grad_output_ptr + d1);
              Vec sum_vec = Vec::loadu(tmp_sum_data + d1);
              sum_vec += grad_output_vec;
              sum_vec.store(tmp_sum_data + d1);
              Vec output_vec = Vec::loadu(output_ptr + d1);
              Vec other_sum_vec = Vec::loadu(tmp_other_sum_data + d1);
              const auto dominant = (output_vec > Vec(scalar_t(-0.69314718055994530942))) &
                  (output_vec <= Vec(scalar_t(0)));
              other_sum_vec += Vec::blendv(grad_output_vec, Vec(scalar_t(0)), dominant);
              other_sum_vec.store(tmp_other_sum_data + d1);
              Vec dominant_count_vec = Vec::loadu(tmp_dominant_count_data + d1);
              dominant_count_vec += Vec::blendv(Vec(scalar_t(0)), Vec(scalar_t(1)), dominant);
              dominant_count_vec.store(tmp_dominant_count_data + d1);
            }
            for (; d1 < size; d1++) {
              tmp_sum_data[d1] += grad_output_ptr[d1];
              if (!(output_ptr[d1] > scalar_t(-0.69314718055994530942) && output_ptr[d1] <= 0)) {
                tmp_other_sum_data[d1] += grad_output_ptr[d1];
              } else {
                tmp_dominant_count_data[d1] += scalar_t(1);
              }
            }
          }

          // compute grad_output - output.exp() * sum
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            int64_t offset = outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            const scalar_t* grad_output_ptr = grad_output_data_base + offset;
            const scalar_t* output_ptr = output_data_base + offset;
            scalar_t* grad_input_ptr = grad_input_data_base + offset;

            int64_t d2 = 0;
            for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
              Vec grad_output_vec = Vec::loadu(grad_output_ptr + d2);
              Vec output_vec = Vec::loadu(output_ptr + d2);
              Vec sum_vec = Vec::loadu(tmp_sum_data + d2);
              Vec other_sum_vec = Vec::loadu(tmp_other_sum_data + d2);
              Vec grad_input_vec = log_softmax_backward_value<scalar_t>(
                  grad_output_vec, output_vec, sum_vec, other_sum_vec,
                  Vec::loadu(tmp_dominant_count_data + d2));
              grad_input_vec.store(grad_input_ptr + d2);
            }
            for (; d2 < size; d2++) {
              grad_input_ptr[d2] = log_softmax_backward_scalar<scalar_t>(
                  grad_output_ptr[d2], output_ptr[d2], tmp_sum_data[d2],
                  tmp_other_sum_data[d2], tmp_dominant_count_data[d2]);
            }
          }
        }
      });
}

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_log_softmax_backward(
    scalar_t* grad_input_data_base,
    const scalar_t* grad_output_data_base,
    const scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  int64_t outer_stride = dim_size * inner_size;
  int64_t BLOCK_SIZE = 128 * 1024;
  int64_t MAX_CHUNK_SIZE = std::max<int64_t>(
      BLOCK_SIZE / dim_size / sizeof(scalar_t), Vec::size());
  MAX_CHUNK_SIZE = MAX_CHUNK_SIZE / Vec::size() * Vec::size();
  int64_t CHUNK_SIZE = std::min<int64_t>(MAX_CHUNK_SIZE, inner_size);
  int64_t num_chunks = divup(inner_size, CHUNK_SIZE);
  // See Note: grain_size value of 0
  parallel_for(
      0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
        // thread local temp buffer that holds vertical sum result
        auto buffer = std::make_unique<float[]>(CHUNK_SIZE * 3);
        float* tmp_sum_data = buffer.get();
        float* tmp_other_sum_data = buffer.get() + CHUNK_SIZE;
        float* tmp_dominant_count_data = buffer.get() + CHUNK_SIZE * 2;

        // thread local buffer that holds grad_output data in float32
        auto grad_output_buffer = std::make_unique<float[]>(dim_size * CHUNK_SIZE);
        float* grad_output_buffer_data = grad_output_buffer.get();

        for (int64_t i = begin; i < end; i++) {
          int64_t outer_idx = i / num_chunks;
          int64_t k = i % num_chunks;
          int64_t inner_idx_begin = k * CHUNK_SIZE;
          int64_t size = std::min(CHUNK_SIZE, inner_size - inner_idx_begin);

          // init
          fVec zero_fvec = fVec(float(0));
          int64_t d0 = 0;
          for (; d0 < size - (size % Vec::size()); d0 += Vec::size()) {
            zero_fvec.store(tmp_sum_data + d0);
            zero_fvec.store(tmp_sum_data + d0 + fVec::size());
            zero_fvec.store(tmp_other_sum_data + d0);
            zero_fvec.store(tmp_other_sum_data + d0 + fVec::size());
            zero_fvec.store(tmp_dominant_count_data + d0);
            zero_fvec.store(tmp_dominant_count_data + d0 + fVec::size());
          }
          for (; d0 < size; d0++) {
            tmp_sum_data[d0] = float(0);
            tmp_other_sum_data[d0] = float(0);
            tmp_dominant_count_data[d0] = float(0);
          }

          // compute sum of grad_output
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            const scalar_t* grad_output_ptr = grad_output_data_base +
                outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            float* grad_output_buffer_ptr =
                grad_output_buffer_data + dim_idx * CHUNK_SIZE;
            const scalar_t* output_ptr = output_data_base +
                outer_idx * outer_stride + dim_idx * inner_size + inner_idx_begin;

            int64_t d1 = 0;
            for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
              Vec grad_output_vec = Vec::loadu(grad_output_ptr + d1);
              auto [grad_output_fvec0, grad_output_fvec1] =
                  vec::convert_to_float<scalar_t>(grad_output_vec);
              fVec sum_fvec0 = fVec::loadu(tmp_sum_data + d1);
              fVec sum_fvec1 = fVec::loadu(tmp_sum_data + d1 + fVec::size());
              sum_fvec0 += grad_output_fvec0;
              sum_fvec1 += grad_output_fvec1;
              sum_fvec0.store(tmp_sum_data + d1);
              sum_fvec1.store(tmp_sum_data + d1 + fVec::size());
              auto [output_fvec0, output_fvec1] =
                  vec::convert_to_float<scalar_t>(Vec::loadu(output_ptr + d1));
              fVec other_sum_fvec0 = fVec::loadu(tmp_other_sum_data + d1);
              fVec other_sum_fvec1 = fVec::loadu(tmp_other_sum_data + d1 + fVec::size());
              const auto dominant0 = (output_fvec0 > fVec(float(-0.69314718055994530942))) &
                  (output_fvec0 <= fVec(float(0)));
              const auto dominant1 = (output_fvec1 > fVec(float(-0.69314718055994530942))) &
                  (output_fvec1 <= fVec(float(0)));
              other_sum_fvec0 += fVec::blendv(grad_output_fvec0, fVec(float(0)), dominant0);
              other_sum_fvec1 += fVec::blendv(grad_output_fvec1, fVec(float(0)), dominant1);
              other_sum_fvec0.store(tmp_other_sum_data + d1);
              other_sum_fvec1.store(tmp_other_sum_data + d1 + fVec::size());
              auto dominant_count0 = fVec::loadu(tmp_dominant_count_data + d1);
              auto dominant_count1 = fVec::loadu(tmp_dominant_count_data + d1 + fVec::size());
              dominant_count0 += fVec::blendv(fVec(0), fVec(1), dominant0);
              dominant_count1 += fVec::blendv(fVec(0), fVec(1), dominant1);
              dominant_count0.store(tmp_dominant_count_data + d1);
              dominant_count1.store(tmp_dominant_count_data + d1 + fVec::size());

              // cache the 'converted' float grad_output
              grad_output_fvec0.store(grad_output_buffer_ptr + d1);
              grad_output_fvec1.store(
                  grad_output_buffer_ptr + d1 + fVec::size());
            }
            for (; d1 < size; d1++) {
              float grad_output_val = float(grad_output_ptr[d1]);
              tmp_sum_data[d1] += grad_output_val;
              grad_output_buffer_ptr[d1] = grad_output_val;
              const float output_value = float(output_ptr[d1]);
              if (!(output_value > float(-0.69314718055994530942) && output_value <= 0)) {
                tmp_other_sum_data[d1] += grad_output_val;
              } else {
                tmp_dominant_count_data[d1] += float(1);
              }
            }
          }

          // compute grad_output - output.exp() * sum
          for (int64_t dim_idx = 0; dim_idx < dim_size; dim_idx++) {
            int64_t offset = outer_idx * outer_stride + dim_idx * inner_size +
                inner_idx_begin;
            const scalar_t* output_ptr = output_data_base + offset;
            scalar_t* grad_input_ptr = grad_input_data_base + offset;
            float* grad_output_buffer_ptr =
                grad_output_buffer_data + dim_idx * CHUNK_SIZE;

            int64_t d2 = 0;
            for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
              Vec output_vec = Vec::loadu(output_ptr + d2);
              auto [output_fvec0, output_fvec1] =
                  vec::convert_to_float<scalar_t>(output_vec);
              fVec sum_fvec0 = fVec::loadu(tmp_sum_data + d2);
              fVec sum_fvec1 = fVec::loadu(tmp_sum_data + d2 + fVec::size());
              fVec grad_output_fvec0 = fVec::loadu(grad_output_buffer_ptr + d2);
              fVec grad_output_fvec1 =
                  fVec::loadu(grad_output_buffer_ptr + d2 + fVec::size());
              fVec grad_input_fvec0 =
                  log_softmax_backward_value<scalar_t>(grad_output_fvec0, output_fvec0,
                      sum_fvec0, fVec::loadu(tmp_other_sum_data + d2),
                      fVec::loadu(tmp_dominant_count_data + d2));
              fVec grad_input_fvec1 =
                  log_softmax_backward_value<scalar_t>(grad_output_fvec1, output_fvec1,
                      sum_fvec1, fVec::loadu(tmp_other_sum_data + d2 + fVec::size()),
                      fVec::loadu(tmp_dominant_count_data + d2 + fVec::size()));
              Vec grad_input_vec =
                  vec::convert_from_float<scalar_t>(grad_input_fvec0, grad_input_fvec1);
              grad_input_vec.store(grad_input_ptr + d2);
            }
            for (; d2 < size; d2++) {
              grad_input_ptr[d2] = log_softmax_backward_scalar<scalar_t>(
                  grad_output_buffer_ptr[d2], float(output_ptr[d2]), tmp_sum_data[d2],
                  tmp_other_sum_data[d2], tmp_dominant_count_data[d2]);
            }
          }
        }
      });
}

template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_lastdim {
  static void apply(const Tensor& output, const Tensor& input) {
    int64_t outer_size = 1;
    int64_t dim_size = input.size(input.ndimension() - 1);
    for (int64_t i = 0; i < input.ndimension() - 1; ++i)
      outer_size *= input.size(i);
    const scalar_t* input_data_base = input.const_data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();
    if (LogSoftMax) {
      _vec_log_softmax_lastdim(
          input_data_base, output_data_base, outer_size, dim_size);
    } else {
      _vec_softmax_lastdim(
          input_data_base, output_data_base, outer_size, dim_size);
    }
  }
};

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<float>;
  using Vec16 = vec::Vectorized<scalar_t>;
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  int vectorized_step = Vec16().size(); // Currently, we only support BFloat16/Half in this special implementation
  // See Note: grain_size value of 0
  parallel_for(
      0, outer_size * inner_size, 0, [&](int64_t begin, int64_t end) {
        int64_t idx = begin;
        std::vector<float> temp_vec_input(dim_size * vectorized_step);
        std::vector<float> temp_vec_output(dim_size * vectorized_step);
        float* temp_vec_input_data = temp_vec_input.data();
        float* temp_vec_output_data = temp_vec_output.data();
        while (idx < end) {
          int64_t outer_idx = idx / inner_size;
          int64_t inner_idx = idx % inner_size;
          if (((inner_idx + vectorized_step) <= inner_size) && ((idx + vectorized_step) <= end)) {
            // Vectorization
            const scalar_t* input_data =
                input_data_base + outer_idx * outer_stride + inner_idx;
            scalar_t* output_data =
                output_data_base + outer_idx * outer_stride + inner_idx;
            // Step 1: Get max Score
            Vec16 max_vec_bf16 = Vec16::loadu(input_data);
            std::tuple<Vec, Vec> convert_result = vec::convert_to_float<scalar_t>(max_vec_bf16);
            Vec max_vec_o1 = std::get<0>(convert_result);
            Vec max_vec_o2 = std::get<1>(convert_result);
            std::get<0>(convert_result).store(temp_vec_input_data);
            std::get<1>(convert_result).store(temp_vec_input_data + Vec().size());
            for (const auto d : c10::irange(1, dim_size)) {
              Vec16 input_vec_bf16 = Vec16::loadu(input_data + d * dim_stride);
              convert_result = vec::convert_to_float<scalar_t>(input_vec_bf16);
              max_vec_o1 = vec::maximum(max_vec_o1, std::get<0>(convert_result));
              max_vec_o2 = vec::maximum(max_vec_o2, std::get<1>(convert_result));
              std::get<0>(convert_result).store(temp_vec_input_data + d*vectorized_step);
              std::get<1>(convert_result).store(temp_vec_input_data + d*vectorized_step + Vec().size());
            }
            // Step2: Calculate sum
            Vec sum_vec_o1 = Vec(0.0);
            Vec sum_vec_o2 = Vec(0.0);
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec_o1 = Vec::loadu(temp_vec_input_data + d*vectorized_step);
              Vec output_vec_o2 = Vec::loadu(temp_vec_input_data + d*vectorized_step + Vec().size());
              output_vec_o1 = (output_vec_o1 - max_vec_o1).exp();
              output_vec_o2 = (output_vec_o2 - max_vec_o2).exp();
              output_vec_o1.store(temp_vec_output_data + d*vectorized_step);
              output_vec_o2.store(temp_vec_output_data + d*vectorized_step + Vec().size());

              sum_vec_o1 = sum_vec_o1 + output_vec_o1;
              sum_vec_o2 = sum_vec_o2 + output_vec_o2;
            }
            // Step3: Unify
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec_o1 = Vec::loadu(temp_vec_output_data + d*vectorized_step);
              Vec output_vec_o2 = Vec::loadu(temp_vec_output_data + d*vectorized_step + Vec().size());
              output_vec_o1 = output_vec_o1/sum_vec_o1;
              output_vec_o2 = output_vec_o2/sum_vec_o2;
              Vec16 output_vec_bf16 = vec::convert_from_float<scalar_t>(output_vec_o1, output_vec_o2);
              output_vec_bf16.store(output_data + d * dim_stride);
            }
            idx += vectorized_step;
          } else {
            // Tail case(Scalar): it is exactly same logic as host_softmax
            // inside aten/src/ATen/native/SoftMax.cpp. There are 2 kind of
            // cases which will fall through this part:
            // Case 1: For the idx at the end of total chunk for each thread, there are not enough numbers for parallelization.
            // Case 2: For the idx at the end of each inner_size inside thread, there are not enough numbers for parallelization.
            int64_t tail_number = ((idx+vectorized_step) > end) ? /*Case1*/ (end - idx) : /*Case2*/ (inner_size - inner_idx);
            for (const auto i : c10::irange(tail_number)) {
              outer_idx = (idx + i) / inner_size;
              inner_idx = (idx + i) % inner_size;
              const scalar_t* input_data =
                  input_data_base + outer_idx * outer_stride + inner_idx;
              scalar_t* output_data =
                  output_data_base + outer_idx * outer_stride + inner_idx;
              // Step1: Get max score
              float max_input = float(input_data[0]);
              for (const auto d : c10::irange(1, dim_size)) {
                max_input = std::max(max_input, float(input_data[d * dim_stride]));
              }
              // Step2: Calculate the Sum
              float sum_data = 0.0;
              float temp_output_data = 0.0;
              for (const auto d : c10::irange(dim_size)) {
                temp_output_data = std::exp(input_data[d * dim_stride] - max_input);
                sum_data += temp_output_data;
                output_data[d * dim_stride] = scalar_t(temp_output_data);
              }
              // Step3: Unify
              for (const auto d : c10::irange(dim_size)) {
                output_data[d * dim_stride] =
                    scalar_t(float(output_data[d * dim_stride])/sum_data);
              }
            }
            idx += tail_number;
          }
        }
      });
}

template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_softmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  int vectorized_step = Vec().size();
  // See Note: grain_size value of 0
  parallel_for(
      0, outer_size * inner_size, 0, [&](int64_t begin, int64_t end) {
        int64_t idx = begin;
        while (idx < end) {
          int64_t outer_idx = idx / inner_size;
          int64_t inner_idx = idx % inner_size;
          if (((inner_idx + vectorized_step) <= inner_size) && ((idx + vectorized_step) <= end)) {
            // Vectorization
            const scalar_t* input_data =
                input_data_base + outer_idx * outer_stride + inner_idx;
            scalar_t* output_data =
                output_data_base + outer_idx * outer_stride + inner_idx;
            // Step 1: Get max Score
            Vec max_vec = Vec::loadu(input_data);
            for (const auto d : c10::irange(1, dim_size)) {
              Vec input_vec = Vec::loadu(input_data + d * dim_stride);
              max_vec = vec::maximum(max_vec, input_vec);
            }
            // Step2: Calculate sum
            Vec sum_vec = Vec(0.0);
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec =
                  (Vec::loadu(input_data + d * dim_stride) - max_vec).exp();
              output_vec.store(output_data + d * dim_stride);
              sum_vec = sum_vec + output_vec;
            }
            // Step3: Unify
            for (const auto d : c10::irange(dim_size)) {
              Vec output_vec =
                  Vec::loadu(output_data + d * dim_stride) / sum_vec;
              output_vec.store(output_data + d * dim_stride);
            }
            idx += vectorized_step;
          } else {
            // Tail case(Scalar): it is exactly same logic as host_softmax
            // inside aten/src/ATen/native/SoftMax.cpp. There are 2 kind of
            // cases which will fall through this part:
            // Case 1: For the idx at the end of total chunk for each thread, there are not enough numbers for parallelization.
            // Case 2: For the idx at the end of each inner_size inside thread, there are not enough numbers for parallelization.
            int64_t tail_number = ((idx+vectorized_step) > end) ? /*Case1*/ (end - idx) : /*Case2*/ (inner_size - inner_idx);
            for (const auto i : c10::irange(tail_number)) {
              outer_idx = (idx + i) / inner_size;
              inner_idx = (idx + i) % inner_size;
              const scalar_t* input_data =
                  input_data_base + outer_idx * outer_stride + inner_idx;
              scalar_t* output_data =
                  output_data_base + outer_idx * outer_stride + inner_idx;
              // Step1: Get max score
              scalar_t max_input = input_data[0];
              for (const auto d : c10::irange(1, dim_size)) {
                max_input = std::max(max_input, input_data[d * dim_stride]);
              }
              // Step2: Calculate the Sum
              scalar_t sum_data = 0;
              for (const auto d : c10::irange(dim_size)) {
                output_data[d * dim_stride] =
                    std::exp(input_data[d * dim_stride] - max_input);
                sum_data += output_data[d * dim_stride];
              }
              // Step3: Unify
              for (const auto d : c10::irange(dim_size)) {
                output_data[d * dim_stride] =
                    output_data[d * dim_stride]/sum_data;
              }
            }
            idx += tail_number;
          }
        }
      });
}

// NB: fast kernel for log_softmax when dim != -1
// input shape is normalized to {outer_size, dim_size, inner_size}
//
// The algorithm requires to load input tensor 3 times, to increase parallelism
// and cache hit rate, inner_size is blocked as:
//   inner_size: {CHUNK_SIZE, CHUNK_SIZE, ..., Remainder}
//
// Parallel on {outer_size, num_chunks} and do vertical reduction on each block of
// {dim_size, CHUNK_SIZE}, block size (128KB) selected to be L2 hit.
//
template<typename scalar_t>
inline typename std::enable_if_t<std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_logsoftmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  const auto [CHUNK_SIZE_binding, num_chunks_binding] = vec_logsoftmax_chunk_size_and_num_chunks<scalar_t>(
      inner_size, dim_size);
  // Work around "capturing a structured binding is not yet supported in OpenMP".
  const auto CHUNK_SIZE = CHUNK_SIZE_binding;
  const auto num_chunks = num_chunks_binding;

  // See Note: grain_size value of 0
  at::parallel_for(0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
    serial_vec_logsoftmax_range(
        input_data_base,
        output_data_base,
        inner_size,
        CHUNK_SIZE,
        num_chunks,
        dim_size,
        begin,
        end);
  });
}

template<typename scalar_t>
inline typename std::enable_if_t<!std::is_same_v<scalar_t, at::opmath_type<scalar_t>>, void>
_vec_logsoftmax(
    const scalar_t* input_data_base,
    scalar_t* output_data_base,
    int64_t outer_size,
    int64_t inner_size,
    int64_t dim_size) {
  const auto [CHUNK_SIZE_binding, num_chunks_binding] = vec_logsoftmax_chunk_size_and_num_chunks<scalar_t>(
      inner_size, dim_size);
  // Work around "capturing a structured binding is not yet supported in OpenMP".
  const auto CHUNK_SIZE = CHUNK_SIZE_binding;
  const auto num_chunks = num_chunks_binding;

  // See Note: grain_size value of 0
  at::parallel_for(0, outer_size * num_chunks, 0, [&](int64_t begin, int64_t end) {
    serial_vec_logsoftmax_range(
        input_data_base,
        output_data_base,
        inner_size,
        CHUNK_SIZE,
        num_chunks,
        dim_size,
        begin,
        end);
  });
}

template <typename scalar_t, bool LogSoftMax>
struct vec_softmax {
  static void apply(const Tensor& output, const Tensor& input, int64_t dim) {
    int64_t outer_size = 1;
    int64_t dim_size = input.size(dim);
    int64_t inner_size = 1;
    for (const auto i : c10::irange(dim))outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    const scalar_t* input_data_base = input.const_data_ptr<scalar_t>();
    scalar_t* output_data_base = output.data_ptr<scalar_t>();
    if (LogSoftMax) {
      _vec_logsoftmax(
          input_data_base, output_data_base, outer_size, inner_size, dim_size);
    } else {
      _vec_softmax(
          input_data_base, output_data_base, outer_size, inner_size, dim_size);
    }
  }
};

template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_backward_lastdim {
  static void
  apply(const Tensor& grad_input, const Tensor& grad, const Tensor& output) {
    int64_t outer_size = 1;
    int64_t dim_size = grad.size(grad.ndimension() - 1);
    for (int64_t i = 0; i < grad.ndimension() - 1; ++i)
      outer_size *= grad.size(i);
    scalar_t* grad_input_data_base = grad_input.mutable_data_ptr<scalar_t>();
    const scalar_t* grad_data_base = grad.const_data_ptr<scalar_t>();
    const scalar_t* output_data_base = output.const_data_ptr<scalar_t>();
    _vec_host_softmax_backward_lastdim<scalar_t, LogSoftMax>(
        grad_input_data_base,
        grad_data_base,
        output_data_base,
        outer_size,
        dim_size);
  }
};

template <typename scalar_t, bool LogSoftMax>
struct vec_host_softmax_backward {
  static void apply(
      const Tensor& grad_input,
      const Tensor& grad,
      const Tensor& output,
      int64_t dim) {
    int64_t outer_size = 1;
    int64_t dim_size = grad.size(dim);
    int64_t inner_size = 1;
    for (const auto i : c10::irange(dim)) {
      outer_size *= grad.size(i);
    }
    for (int64_t i = dim + 1; i < grad.dim(); ++i) {
      inner_size *= grad.size(i);
    }
    scalar_t* grad_input_data_base = grad_input.mutable_data_ptr<scalar_t>();
    const scalar_t* grad_output_data_base = grad.const_data_ptr<scalar_t>();
    const scalar_t* output_data_base = output.const_data_ptr<scalar_t>();
    if (LogSoftMax) {
      _vec_log_softmax_backward<scalar_t>(
          grad_input_data_base,
          grad_output_data_base,
          output_data_base,
          outer_size,
          inner_size,
          dim_size);
    } else {
      _vec_softmax_backward<scalar_t>(
          grad_input_data_base,
          grad_output_data_base,
          output_data_base,
          outer_size,
          inner_size,
          dim_size);
    }
  }
};

void softmax_lastdim_kernel_impl(
    const Tensor& result,
    const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
      "softmax_lastdim_kernel_impl",
      [&] { vec_host_softmax_lastdim<scalar_t, false>::apply(result, self); });
}

void softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
    "softmax_kernel_impl",
    [&] { vec_softmax<scalar_t, false>::apply(result, self, dim); });
}

void log_softmax_lastdim_kernel_impl(
    const Tensor& result,
    const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
      "log_softmax_lastdim_kernel_impl",
      [&] { vec_host_softmax_lastdim<scalar_t, true>::apply(result, self); });
}

void log_softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
    "log_softmax_kernel_impl",
    [&] { vec_softmax<scalar_t, true>::apply(result, self, dim); });
}

void softmax_backward_lastdim_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, grad.scalar_type(),
      "softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, false>::apply(
            grad_input, grad, output);
      });
}

void log_softmax_backward_lastdim_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, grad.scalar_type(),
      "log_softmax_backward_lastdim_kernel_impl", [&] {
        vec_host_softmax_backward_lastdim<scalar_t, true>::apply(
            grad_input, grad, output);
      });
}

void softmax_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad.scalar_type(),
      "softmax_backward_kernel_impl",
      [&] {
        vec_host_softmax_backward<scalar_t, false>::apply(
            grad_input, grad, output, dim);
      });
}

void log_softmax_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad.scalar_type(),
      "log_softmax_backward_kernel_impl",
      [&] {
        vec_host_softmax_backward<scalar_t, true>::apply(
            grad_input, grad, output, dim);
      });
}

template <typename scalar_t>
void log_softmax_double_backward_kernel_type(TensorIteratorBase& iterator) {
  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<opmath_t>;
  const auto scalar_function =
      [](scalar_t scale, scalar_t incoming, scalar_t output) -> scalar_t {
    const opmath_t factor = opmath_t(scale);
    const opmath_t gradient = opmath_t(incoming);
    const opmath_t value = opmath_t(output);
    if (std::isfinite(factor) && std::isfinite(gradient) && value <= 0) {
      const auto quarter = std::exp(value * opmath_t(0.25));
      const auto product = factor * gradient;
      if (std::isfinite(product) &&
          (product != 0 || factor == 0 || gradient == 0)) {
        return static_cast<scalar_t>(
            (((product * quarter) * quarter) * quarter) * quarter);
      }
      return static_cast<scalar_t>(
          ((factor * quarter) * quarter) * ((gradient * quarter) * quarter));
    }
    return static_cast<scalar_t>(factor * std::exp(value) * gradient);
  };
  const auto vector_function = [](Vec scale, Vec incoming, Vec output) {
    const Vec infinity(std::numeric_limits<opmath_t>::infinity());
    const auto safe = (scale.abs() < infinity) & (incoming.abs() < infinity) &
        (output <= Vec(0));
    const auto quarter = (output * Vec(0.25)).exp();
    const auto product = scale * incoming;
    const auto finite_product = (product.abs() < infinity) &
        ((product != Vec(0)) | (scale == Vec(0)) | (incoming == Vec(0)));
    const auto direct = (((product * quarter) * quarter) * quarter) * quarter;
    if (!(safe & finite_product).zero_mask()) {
      return direct;
    }
    const auto split =
        ((scale * quarter) * quarter) * ((incoming * quarter) * quarter);
    const auto stable = Vec::blendv(split, direct, finite_product);
    if (!safe.zero_mask()) {
      return stable;
    }
    return Vec::blendv(
        scale * output.exp() * incoming,
        stable,
        safe);
  };
  if constexpr (std::is_same_v<scalar_t, c10::Half> ||
                std::is_same_v<scalar_t, c10::BFloat16>) {
    cpu_kernel_vec(
        iterator,
        scalar_function,
        [vector_function](vec::Vectorized<scalar_t> scale,
                          vec::Vectorized<scalar_t> incoming,
                          vec::Vectorized<scalar_t> output) {
          auto [scale0, scale1] = vec::convert_to_float<scalar_t>(scale);
          auto [incoming0, incoming1] = vec::convert_to_float<scalar_t>(incoming);
          auto [output0, output1] = vec::convert_to_float<scalar_t>(output);
          return vec::convert_from_float<scalar_t>(
              vector_function(scale0, incoming0, output0),
              vector_function(scale1, incoming1, output1));
        });
  } else {
    cpu_kernel_vec(iterator, scalar_function, vector_function);
  }
}

bool log_softmax_jvp_lastdim_kernel_impl(
    const Tensor& result,
    const Tensor& tangent,
    const Tensor& output) {
  const int64_t dim_size = output.size(-1);
  const int64_t outer_size = output.numel() / dim_size;
  std::atomic<bool> all_normal{true};
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "log_softmax_jvp_lastdim_cpu", [&] {
    using Vec = vec::Vectorized<scalar_t>;
    const auto* output_data = output.const_data_ptr<scalar_t>();
    const auto* tangent_data = tangent.const_data_ptr<scalar_t>();
    auto* result_data = result.data_ptr<scalar_t>();
    parallel_for(0, outer_size, std::max<int64_t>(1, internal::GRAIN_SIZE / dim_size),
                 [&](int64_t begin, int64_t end) {
      int64_t row = begin;
      if (dim_size <= Vec::size()) {
        alignas(64) scalar_t values_buffer[Vec::size() * Vec::size()];
        alignas(64) scalar_t direction_buffer[Vec::size() * Vec::size()];
        for (; row + Vec::size() <= end; row += Vec::size()) {
          vec::transpose_mxn<scalar_t>(
              output_data + row * dim_size, dim_size, values_buffer,
              Vec::size(), Vec::size(), dim_size);
          vec::transpose_mxn<scalar_t>(
              tangent_data + row * dim_size, dim_size, direction_buffer,
              Vec::size(), Vec::size(), dim_size);
          const auto first_output = Vec::loadu(values_buffer);
          const auto center = Vec::loadu(direction_buffer);
          auto maximum = first_output;
          auto minimum = first_output;
          auto magnitude = center.abs();
          Vec spread(0);
          Vec weighted(0);
          Vec other_probability(0);
          for (int64_t position = 1; position < dim_size; ++position) {
            const auto values = Vec::loadu(values_buffer + position * Vec::size());
            const auto direction = Vec::loadu(direction_buffer + position * Vec::size());
            const auto centered = direction - center;
            const auto probability = values.exp();
            maximum = vec::maximum(maximum, values);
            minimum = vec::minimum(minimum, values);
            magnitude = vec::maximum(magnitude, direction.abs());
            spread = vec::maximum(spread, centered.abs());
            weighted = weighted + centered * probability;
            other_probability = other_probability + probability;
          }
          const auto correction = center * (first_output.expm1().neg() - other_probability);
          const Vec infinity(std::numeric_limits<scalar_t>::infinity());
          const auto valid =
              (maximum <= Vec(-0.69314718055994530942)) &
              (minimum >= Vec(std::log(std::numeric_limits<scalar_t>::min()))) &
              (magnitude <= Vec(std::numeric_limits<scalar_t>::max() / (2 * dim_size))) &
              ((magnitude == Vec(0)) |
               (spread > magnitude * Vec(std::numeric_limits<scalar_t>::epsilon() * (dim_size + 2)))) &
              (weighted.abs() < infinity) & (correction.abs() < infinity);
          if (valid.zero_mask()) {
            all_normal.store(false, std::memory_order_relaxed);
            continue;
          }
          for (int64_t position = 0; position < dim_size; ++position) {
            const auto direction = Vec::loadu(direction_buffer + position * Vec::size());
            ((direction - center) - weighted + correction)
                .store(direction_buffer + position * Vec::size());
          }
          vec::transpose_mxn<scalar_t>(
              direction_buffer, Vec::size(), result_data + row * dim_size,
              dim_size, dim_size, Vec::size());
        }
      }
      for (; row < end; ++row) {
        const auto* values = output_data + row * dim_size;
        const auto* direction = tangent_data + row * dim_size;
        const auto maximum = vec::reduce_all<scalar_t>(
            [](Vec first, Vec second) { return vec::maximum(first, second); },
            values, dim_size);
        const auto minimum = vec::reduce_all<scalar_t>(
            [](Vec first, Vec second) { return vec::minimum(first, second); },
            values, dim_size);
        const auto magnitude = vec::map_reduce_all<scalar_t>(
            [](Vec value) { return value.abs(); },
            [](Vec first, Vec second) { return vec::maximum(first, second); },
            direction, dim_size);
        if (!(maximum <= scalar_t(-0.69314718055994530942) &&
              minimum >= std::log(std::numeric_limits<scalar_t>::min()) &&
              std::isfinite(magnitude) &&
              magnitude <= std::numeric_limits<scalar_t>::max() / (2 * dim_size))) {
          all_normal.store(false, std::memory_order_relaxed);
          continue;
        }
        const auto center = direction[0];
        const auto spread = vec::map_reduce_all<scalar_t>(
            [center](Vec value) { return (value - Vec(center)).abs(); },
            [](Vec first, Vec second) { return vec::maximum(first, second); },
            direction, dim_size);
        if (magnitude != scalar_t(0) &&
            spread <= std::numeric_limits<scalar_t>::epsilon() *
                (dim_size + 2) * magnitude) {
          all_normal.store(false, std::memory_order_relaxed);
          continue;
        }
        Vec weighted_vector(0);
        Vec probability_vector(0);
        int64_t position = 1;
        for (; position + Vec::size() <= dim_size; position += Vec::size()) {
          const auto probability = Vec::loadu(values + position).exp();
          const auto centered = Vec::loadu(direction + position) - Vec(center);
          weighted_vector = weighted_vector + centered * probability;
          probability_vector = probability_vector + probability;
        }
        auto weighted = vec::vec_reduce_all(
            [](Vec& first, Vec& second) { return first + second; },
            weighted_vector);
        auto other_probability = vec::vec_reduce_all(
            [](Vec& first, Vec& second) { return first + second; },
            probability_vector);
        for (; position < dim_size; ++position) {
          const auto probability = std::exp(values[position]);
          weighted += (direction[position] - center) * probability;
          other_probability += probability;
        }
        const auto correction = center * (-std::expm1(values[0]) - other_probability);
        if (!std::isfinite(weighted) || !std::isfinite(correction)) {
          all_normal.store(false, std::memory_order_relaxed);
          continue;
        }
        vec::map<scalar_t>(
            [center, weighted, correction](Vec value) {
              return (value - Vec(center)) - Vec(weighted) + Vec(correction);
            },
            result_data + row * dim_size, direction, dim_size);
      }
    });
  });
  return all_normal.load(std::memory_order_relaxed);
}

void log_softmax_double_backward_kernel_impl(TensorIteratorBase& iterator) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      iterator.common_dtype(),
      "log_softmax_double_backward_cpu",
      [&]() { log_softmax_double_backward_kernel_type<scalar_t>(iterator); });
}

} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(softmax_lastdim_kernel, &softmax_lastdim_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(log_softmax_lastdim_kernel, &log_softmax_lastdim_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(
    softmax_backward_lastdim_kernel,
    &softmax_backward_lastdim_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(
    log_softmax_backward_lastdim_kernel,
    &log_softmax_backward_lastdim_kernel_impl)

ALSO_REGISTER_AVX512_DISPATCH(softmax_kernel, &softmax_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(log_softmax_kernel, &log_softmax_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(log_softmax_double_backward_stub, &log_softmax_double_backward_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(log_softmax_jvp_lastdim_stub, &log_softmax_jvp_lastdim_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(softmax_backward_kernel, &softmax_backward_kernel_impl)
ALSO_REGISTER_AVX512_DISPATCH(
    log_softmax_backward_kernel,
    &log_softmax_backward_kernel_impl)
} // namespace at::native
