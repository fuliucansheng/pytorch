#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <type_traits>

namespace at::native {
namespace {

template <ADAM_MODE mode, typename lr_type>
void fused_adam_tensor_betas_cuda(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const lr_type& lr,
    const at::Tensor& beta1,
    const at::Tensor& beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (params.empty()) {
    return;
  }
  const auto device = params[0].device();
  TORCH_CHECK(params[0].is_cuda(), "Tensor betas require CUDA params");
  TORCH_CHECK(
      beta1.numel() == 1 && beta2.numel() == 1,
      "beta1 and beta2 must each contain one element");
  TORCH_CHECK(
      beta1.device() == device && beta2.device() == device,
      "beta1 and beta2 must be on the same GPU device as the params");
  TORCH_CHECK(
      beta1.is_floating_point() && beta2.is_floating_point(),
      "beta1 and beta2 must have floating point dtype");
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == device,
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == device,
        "found_inf must be on the same GPU device as the params");
  }
  const float* lr_ptr = nullptr;
  double lr_value;
  if constexpr (std::is_same_v<lr_type, at::Tensor>) {
    TORCH_CHECK(lr.numel() == 1, "lr must contain one element");
    if (lr.is_cpu()) {
      lr_value = lr.template item<double>();
    } else {
      TORCH_CHECK(
          lr.device() == device,
          "lr must be on the same GPU device as the params");
      lr_ptr = lr.template const_data_ptr<float>();
      lr_value = 1.0;
    }
  } else {
    lr_value = lr;
  }
  const auto beta1_double = beta1.to(at::kDouble);
  const auto beta2_double = beta2.to(at::kDouble);
  const auto* beta1_ptr = beta1_double.const_data_ptr<double>();
  const auto* beta2_ptr = beta2_double.const_data_ptr<double>();
  const bool is_mixed_precision =
      params[0].scalar_type() != exp_avgs[0].scalar_type();
  if (amsgrad) {
    TORCH_CHECK(
        check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs},
            {},
            false,
            is_mixed_precision),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_cuda_impl_common<mode, true>(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr_ptr,
        lr_value,
        0.0,
        0.0,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf,
        beta1_ptr,
        beta2_ptr);
  } else {
    TORCH_CHECK(
        check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs},
            {},
            false,
            is_mixed_precision),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_cuda_impl_common<mode, false>(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        {},
        state_steps,
        lr_ptr,
        lr_value,
        0.0,
        0.0,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf,
        beta1_ptr,
        beta2_ptr);
  }
}

} // namespace

void _fused_adam_tensor_betas_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const at::Tensor& beta1,
    const at::Tensor& beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  fused_adam_tensor_betas_cuda<ADAM_MODE::ORIGINAL>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      amsgrad,
      maximize,
      grad_scale,
      found_inf);
}

void _fused_adam_tensor_betas_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const at::Tensor& beta1,
    const at::Tensor& beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  fused_adam_tensor_betas_cuda<ADAM_MODE::ORIGINAL>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      amsgrad,
      maximize,
      grad_scale,
      found_inf);
}

void _fused_adamw_tensor_betas_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const at::Tensor& beta1,
    const at::Tensor& beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  fused_adam_tensor_betas_cuda<ADAM_MODE::ADAMW>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      amsgrad,
      maximize,
      grad_scale,
      found_inf);
}

void _fused_adamw_tensor_betas_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const at::Tensor& beta1,
    const at::Tensor& beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  fused_adam_tensor_betas_cuda<ADAM_MODE::ADAMW>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      max_exp_avg_sqs,
      state_steps,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      amsgrad,
      maximize,
      grad_scale,
      found_inf);
}

} // namespace at::native
