# Owner(s): ["module: autograd"]

import math

import torch
import torch.nn.functional as functional
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


class TestLogSoftmaxTailDerivatives(TestCase):
    hw_classification = HardwareClassification.CPU

    @parametrize("dimension", [0, 1])
    def test_ordinary_backward_saved_state(self, device, dimension):
        primal = torch.linspace(-1, 1, 20, device=device, dtype=torch.float64)
        primal = primal.reshape(4, 5).requires_grad_()
        cotangent = torch.linspace(-0.5, 0.75, 20, device=device, dtype=torch.float64)
        cotangent = cotangent.reshape_as(primal)
        recorded = []
        reference_recorded = []

        def capture(tensor):
            recorded.append(
                (tuple(tensor.shape), tensor.numel() * tensor.element_size())
            )
            return tensor

        def capture_reference(tensor):
            reference_recorded.append(
                (tuple(tensor.shape), tensor.numel() * tensor.element_size())
            )
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(capture, lambda tensor: tensor):
            output = primal.log_softmax(dimension)
            first = torch.autograd.grad(output, primal, cotangent, create_graph=True)[0]
        with torch.autograd.graph.saved_tensors_hooks(
            capture_reference, lambda tensor: tensor
        ):
            reference_output = primal.log_softmax(dimension)
            reference_first = torch.ops.aten._log_softmax_backward_data(
                cotangent, reference_output, dimension, primal.dtype
            )
        self.assertEqual(recorded, reference_recorded)
        self.assertEqual(first, reference_first, rtol=4e-13, atol=0)
        direction = torch.linspace(0.25, 1, 20, device=device, dtype=torch.float64)
        direction = direction.reshape_as(primal)
        second = torch.autograd.grad(first, primal, direction)[0]
        reference_second = torch.autograd.grad(reference_first, primal, direction)[0]
        self.assertEqual(second, reference_second, rtol=4e-13, atol=0)

    def check_tiny(self, actual, expected):
        self.assertEqual(actual, expected, rtol=4e-13, atol=2 * math.ulp(0.0))
        self.assertEqual(
            torch.count_nonzero(actual).item(),
            torch.count_nonzero(expected).item(),
        )

    @parametrize("margin", [37.42994775023705, 500.0, 740.0, 1100.0])
    @parametrize("scale", [1.0, 1e308])
    @parametrize("transposed", [False, True])
    def test_normalized_tail_vjp_hvp(self, device, margin, scale, transposed):
        primal = torch.tensor([margin, 0.0], device=device, dtype=torch.float64)
        primal = primal.expand(7, -1).clone()
        if transposed:
            primal = primal.t()
        primal.requires_grad_()
        dimension = 0 if transposed else 1
        cotangent = torch.zeros_like(primal)
        cotangent.select(dimension, 0).fill_(scale)
        tangent = torch.zeros_like(primal)
        tangent.select(dimension, 0).fill_(1)
        output = primal.log_softmax(dimension)
        first = torch.autograd.grad(output, primal, cotangent, create_graph=True)[0]
        second = torch.autograd.grad(first, primal, tangent)[0]
        root = math.exp(-margin / 2)
        denominator = 1 + root * root
        small = ((scale * root) * root) / denominator
        expected = torch.tensor([small, -small], device=device, dtype=torch.float64)
        expected = expected.expand(7, -1)
        if transposed:
            expected = expected.t()
        self.check_tiny(first, expected)
        self.check_tiny(second, -expected / denominator)

    @parametrize("margin", [37.42994775023705, 740.0])
    def test_cross_entropy_tail(self, device, margin):
        primal = torch.tensor(
            [[margin, 0.0]], device=device, dtype=torch.float64, requires_grad=True
        )
        loss = functional.cross_entropy(primal, torch.tensor([0], device=device))
        actual = torch.autograd.grad(loss, primal)[0]
        root = math.exp(-margin / 2)
        small = (root * root) / (1 + root * root)
        expected = torch.tensor([[-small, small]], device=device, dtype=torch.float64)
        self.check_tiny(actual, expected)

    @parametrize("transposed", [False, True])
    def test_finite_excluding_sum_overflow(self, device, transposed):
        primal = torch.tensor(
            [math.log(3), 0.0, 0.0], device=device, dtype=torch.float64
        )
        primal = primal.expand(17, -1).clone()
        gradient = torch.tensor(
            [-1e308, 1e308, 1e308], device=device, dtype=torch.float64
        ).expand_as(primal)
        expected = torch.tensor(
            [-1.6e308, 8e307, 8e307], device=device, dtype=torch.float64
        ).expand_as(primal)
        if transposed:
            primal, gradient, expected = primal.t(), gradient.t(), expected.t()
        primal.requires_grad_()
        dimension = 0 if transposed else 1
        actual = torch.autograd.grad(primal.log_softmax(dimension), primal, gradient)[0]
        self.check_tiny(actual, expected)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("gap", [0.0, 0.00390625])
    @parametrize("transposed", [False, True])
    def test_public_backward_tied_outputs(self, device, dtype, gap, transposed):
        output = torch.tensor([gap, 0.0], device=device, dtype=dtype).log_softmax(0)
        gradient = torch.tensor([1.0, -0.25], device=device, dtype=dtype)
        values = output.tolist()
        expected = torch.tensor(
            [1 - 0.75 * math.exp(values[0]), -0.25 - 0.75 * math.exp(values[1])],
            device=device,
            dtype=dtype,
        )
        output, gradient, expected = (
            value.expand(17, -1) for value in (output, gradient, expected)
        )
        if transposed:
            output, gradient, expected = output.t(), gradient.t(), expected.t()
        dimension = 0 if transposed else 1
        actual = torch.ops.aten._log_softmax_backward_data(
            gradient, output, dimension, dtype
        )
        self.assertEqual(actual, expected, rtol=1e-2, atol=0)
        self.assertEqual(torch.count_nonzero(actual).item(), actual.numel())

    @parametrize("case", ["ambiguous", "rounded_zero"])
    def test_public_backward_independent_output(self, device, case):
        if case == "ambiguous":
            values, scale = [-0.1, -0.2], 1.0
        else:
            values, scale = [0.0, -1100.0], 1e308
        output = torch.tensor(values, device=device, dtype=torch.float64)
        gradient = torch.tensor([scale, 0.0], device=device, dtype=torch.float64)
        root = math.exp(values[1] / 2)
        expected = torch.tensor(
            [scale * -math.expm1(values[0]), -(scale * root) * root],
            device=device,
            dtype=torch.float64,
        )
        actual = torch.ops.aten._log_softmax_backward_data(
            gradient, output, 0, torch.float64
        )
        self.check_tiny(actual, expected)

    @parametrize("margin", [740.0, 1100.0])
    @parametrize("scale", [1.0, 1e308])
    @parametrize("transposed", [False, True])
    def test_long_tail_reduction(self, device, margin, scale, transposed):
        primal = torch.tensor(
            [margin] + [0.0] * 256, device=device, dtype=torch.float64
        )
        primal = primal.expand(5, -1).clone()
        if transposed:
            primal = primal.t()
        primal.requires_grad_()
        dimension = 0 if transposed else 1
        gradient = torch.zeros_like(primal)
        gradient.select(dimension, 0).fill_(scale)
        actual = torch.autograd.grad(primal.log_softmax(dimension), primal, gradient)[0]
        root = math.exp(-margin / 2)
        denominator = 1 + 256 * (root * root)
        small = ((scale * root) * root) / denominator
        dominant = ((scale * root * 256) * root) / denominator
        expected = torch.tensor(
            [dominant] + [-small] * 256, device=device, dtype=torch.float64
        ).expand(5, -1)
        if transposed:
            expected = expected.t()
        self.check_tiny(actual, expected)

    def test_forward_over_reverse_overflow_tangent(self, device):
        primal = torch.tensor(
            [math.log(3), math.log(2)],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        tangent = torch.tensor([1e308, -1e308], device=device, dtype=torch.float64)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(primal, tangent)
            actual = torch.autograd.forward_ad.unpack_dual(dual.log_softmax(0)).tangent
        expected = torch.tensor([8e307, -1.2e308], device=device, dtype=torch.float64)
        self.check_tiny(actual, expected)
        reverse = torch.autograd.grad(actual[0], primal)[0]
        expected_reverse = torch.tensor(
            [-4.8e307, 4.8e307], device=device, dtype=torch.float64
        )
        self.check_tiny(reverse, expected_reverse)

    @parametrize("margin", [37.42994775023705, 740.0, 1100.0])
    @parametrize("tangent_source", ["input", "gradient"])
    def test_forward_tangent_without_reverse_graph(
        self, device, margin, tangent_source
    ):
        primal = torch.tensor(
            [margin, 0.0], device=device, dtype=torch.float64, requires_grad=True
        )
        scale = 1e308
        direction = torch.tensor([1.0, 0.0], device=device, dtype=torch.float64)
        gradient = torch.tensor([scale, 0.0], device=device, dtype=torch.float64)
        with torch.autograd.forward_ad.dual_level():
            if tangent_source == "input":
                argument = torch.autograd.forward_ad.make_dual(primal, direction)
                cotangent = gradient
            else:
                argument = primal
                cotangent = torch.autograd.forward_ad.make_dual(
                    torch.zeros_like(gradient), gradient
                )
            output = argument.log_softmax(0)
            actual = torch.autograd.grad(output, primal, cotangent, create_graph=False)[
                0
            ]
            actual = torch.autograd.forward_ad.unpack_dual(actual).tangent
        root = math.exp(-margin / 2)
        denominator = 1 + root * root
        small = ((scale * root) * root) / denominator
        expected = torch.tensor([small, -small], device=device, dtype=torch.float64)
        if tangent_source == "input":
            expected = -expected / denominator
        self.check_tiny(actual, expected)

    @parametrize("tangent_source", ["gradient", "output", "both"])
    def test_public_backward_forward_tangents(self, device, tangent_source):
        output = torch.tensor([-0.4, -3.0], device=device, dtype=torch.float64)
        gradient = torch.tensor([0.5, -0.25], device=device, dtype=torch.float64)
        gradient_tangent = torch.tensor(
            [1.0, -0.25], device=device, dtype=torch.float64
        )
        output_tangent = torch.tensor([0.125, -0.5], device=device, dtype=torch.float64)
        with torch.autograd.forward_ad.dual_level():
            dual_gradient, dual_output = gradient, output
            if tangent_source != "output":
                dual_gradient = torch.autograd.forward_ad.make_dual(
                    gradient, gradient_tangent
                )
            if tangent_source != "gradient":
                dual_output = torch.autograd.forward_ad.make_dual(
                    output, output_tangent
                )
            result = torch.ops.aten._log_softmax_backward_data(
                dual_gradient, dual_output, 0, torch.float64
            )
            actual = torch.autograd.forward_ad.unpack_dual(result).tangent
        expected = torch.zeros_like(output)
        if tangent_source != "output":
            expected = gradient_tangent - gradient_tangent.sum() * output.exp()
        if tangent_source != "gradient":
            expected = expected - gradient.sum() * output.exp() * output_tangent
        self.check_tiny(actual, expected)


instantiate_device_type_tests(TestLogSoftmaxTailDerivatives, globals(), only_for="cpu")

if __name__ == "__main__":
    run_tests()
