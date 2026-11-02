# Owner(s): ["module: autograd"]

import math
from decimal import Decimal, localcontext

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


def curvature_reference(activation, primal, cotangent, direction):
    with localcontext() as context:
        context.prec = 420
        value = Decimal.from_float(primal)
        exponential = (-abs(value) * (2 if activation == "tanh" else 1)).exp()
        centered = (1 - exponential) / (1 + exponential)
        if value < 0:
            centered = -centered
        variance = exponential / (1 + exponential) ** 2
        curvature = -(8 if activation == "tanh" else 1) * centered * variance
        return float(
            curvature * Decimal.from_float(cotangent) * Decimal.from_float(direction)
        )


class TestActivationOrdinaryRange(TestCase):
    hw_classification = HardwareClassification.CPU

    @dtypes(torch.float32, torch.float64)
    @parametrize("activation", ["sigmoid", "tanh"])
    @parametrize("tracked", [False, True])
    @parametrize("product_edge", ["ordinary", "lower", "upper"])
    @parametrize("layout", ["mixed", "ordinary", "lower", "upper", "scalar", "tail"])
    def test_guard_boundaries_and_mixed_rescue(
        self, device, dtype, activation, tracked, product_edge, layout
    ):
        lower = 1e-100 if dtype == torch.float64 else 1e-10
        upper = 64.0 if dtype == torch.float64 else 16.0
        product_upper = 1e100 if dtype == torch.float64 else 1e10
        edges = torch.tensor([lower, upper], device=device, dtype=dtype)
        below = torch.nextafter(edges, torch.zeros_like(edges))
        above = torch.nextafter(edges, torch.full_like(edges, math.inf))
        ordinary = torch.tensor([0.0, 0.125, 1.0, 2.0], device=device, dtype=dtype)
        tails = torch.tensor([745.0, 1100.0], device=device, dtype=dtype)
        values = torch.cat((below, edges, above, ordinary, tails))
        values = torch.cat((-values.flip(0), values)).requires_grad_()
        if layout == "ordinary":
            values = (
                torch.tensor(
                    [-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0],
                    device=device,
                    dtype=dtype,
                )
                .repeat(2)
                .requires_grad_()
            )
        elif layout in ("lower", "upper"):
            boundary_index = 0 if layout == "lower" else 1
            values = (
                torch.cat(
                    [
                        sign * magnitudes[boundary_index].expand(16)
                        for magnitudes in (below, edges, above)
                        for sign in (-1, 1)
                    ]
                )
                .clone()
                .requires_grad_()
            )
        elif layout in ("scalar", "tail"):
            values = torch.ones(
                1 if layout == "scalar" else 17, device=device, dtype=dtype
            ).requires_grad_()
        product = {"ordinary": 1.0, "lower": lower, "upper": product_upper}[
            product_edge
        ]
        cotangent = torch.ones_like(values)
        direction = torch.full_like(values, product)
        output = getattr(values, activation)()
        first = torch.autograd.grad(output, values, cotangent, create_graph=True)[0]
        actual = torch.autograd.grad(first, values, direction, create_graph=tracked)[0]
        expected = torch.tensor(
            [
                curvature_reference(activation, value, gradient, tangent)
                for value, gradient, tangent in zip(
                    values.detach().cpu().tolist(),
                    cotangent.cpu().tolist(),
                    direction.cpu().tolist(),
                )
            ],
            device=device,
            dtype=dtype,
        )
        subnormal = torch.nextafter(
            torch.zeros((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype),
        ).item()
        self.assertEqual(
            actual,
            expected,
            rtol=4e-13 if dtype == torch.float64 else 4e-5,
            atol=2 * subnormal,
        )
        self.assertEqual(torch.count_nonzero(actual), torch.count_nonzero(expected))


class TestLogSoftmaxDominancePredicate(TestCase):
    hw_classification = HardwareClassification.CPU

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("dimension", [0, 1])
    @parametrize("width", [4, 17, 65])
    @parametrize("multiple", [False, True])
    @parametrize("near_zero", [False, True])
    def test_raw_threshold_neighbours(
        self, device, dtype, dimension, width, multiple, near_zero
    ):
        threshold = torch.tensor(-math.log(2), device=device, dtype=dtype)
        boundaries = (
            torch.nextafter(threshold, torch.full_like(threshold, -math.inf)),
            threshold,
            torch.nextafter(threshold, torch.full_like(threshold, math.inf)),
        )
        zero = torch.zeros((), device=device, dtype=dtype)
        near_zero_value = torch.nextafter(zero, torch.full_like(zero, -math.inf))
        tolerances = {
            torch.float64: 4e-13,
            torch.float32: 4e-5,
            torch.float16: 0.002,
            torch.bfloat16: 0.04,
        }
        for boundary in boundaries:
            output_row = torch.full((width,), -4.0, device=device, dtype=dtype)
            output_row[0] = (near_zero_value if near_zero else zero) if multiple else -1
            output_row[1] = boundary
            gradient_row = torch.arange(width, device=device, dtype=dtype)
            gradient_row[0] = 1e-10
            output = output_row.expand(17, -1).clone()
            gradient = gradient_row.expand_as(output).clone()
            if dimension == 0:
                output = output.t().contiguous()
                gradient = gradient.t().contiguous()
            actual = torch.ops.aten._log_softmax_backward_data(
                gradient, output, dimension, dtype
            )
            reference_output = output.to(torch.float64)
            reference_gradient = gradient.to(torch.float64)
            expected = (
                reference_gradient
                - reference_output.exp()
                * reference_gradient.sum(dimension, keepdim=True)
            ).to(dtype)
            self.assertEqual(actual, expected, rtol=tolerances[dtype], atol=0)
            self.assertEqual(torch.count_nonzero(actual), torch.count_nonzero(expected))
            if dtype == torch.bfloat16 and boundary.item() == -0.69140625:
                self.assertGreater(boundary.item(), -math.log(2))
                if multiple and width == 4:
                    self.assertEqual(
                        actual.select(dimension, 0),
                        torch.full((17,), -6.0, device=device, dtype=dtype),
                    )


class TestLogSoftmaxOrdinaryRange(TestCase):
    hw_classification = HardwareClassification.CPU

    @dtypes(torch.float32, torch.float64)
    @parametrize("tracked", [False, True])
    @parametrize("layout", ["scalar", "lastdim", "nonlastdim"])
    @parametrize(
        "region",
        [
            "ordinary",
            "output_below",
            "output_exact",
            "output_above",
            "product_below",
            "product_lower",
            "product_upper",
            "product_above",
        ],
    )
    def test_public_g2_guard_boundaries(self, device, dtype, tracked, layout, region):
        output_limit = 64.0 if dtype == torch.float64 else 16.0
        product_lower = 1e-100 if dtype == torch.float64 else 1e-10
        product_upper = 1e100 if dtype == torch.float64 else 1e10
        margin = torch.tensor(2.0, device=device, dtype=dtype)
        product = torch.tensor(1.0, device=device, dtype=dtype)
        if region.startswith("output_"):
            margin.fill_(output_limit)
            if region == "output_below":
                margin = torch.nextafter(margin, torch.zeros_like(margin))
            elif region == "output_above":
                margin = torch.nextafter(margin, torch.full_like(margin, math.inf))
        elif region.startswith("product_"):
            margin.zero_()
            product.fill_(
                product_upper
                if region in ("product_upper", "product_above")
                else product_lower
            )
            if region == "product_below":
                product = torch.nextafter(product, torch.zeros_like(product))
            elif region == "product_above":
                product = torch.nextafter(product, torch.full_like(product, math.inf))
        rows = 1 if layout == "scalar" else 17
        primal = (
            torch.stack((torch.zeros_like(margin), -margin)).expand(rows, -1).clone()
        )
        cotangent = torch.zeros_like(primal)
        cotangent[:, 0] = 2 * product
        direction = torch.zeros_like(primal)
        direction[:, 1] = 1
        dimension = 1
        if layout == "nonlastdim":
            primal = primal.t().contiguous()
            cotangent = cotangent.t().contiguous()
            direction = direction.t().contiguous()
            dimension = 0
        primal.requires_grad_()
        output = primal.log_softmax(dimension)
        first = torch.autograd.grad(output, primal, cotangent, create_graph=True)[0]
        actual = torch.autograd.grad(first, primal, direction, create_graph=tracked)[0]
        with localcontext() as context:
            context.prec = 420
            exponential = (-Decimal.from_float(margin.item())).exp()
            scale = Decimal.from_float(cotangent.select(dimension, 0)[0].item())
            curvature = float(scale * exponential / (1 + exponential) ** 2)
        expected = torch.tensor([curvature, -curvature], device=device, dtype=dtype)
        expected = expected.expand(rows, -1).clone()
        if dimension == 0:
            expected = expected.t().contiguous()
        subnormal = torch.nextafter(
            torch.zeros((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype),
        ).item()
        self.assertEqual(
            actual,
            expected,
            rtol=4e-13 if dtype == torch.float64 else 4e-5,
            atol=2 * subnormal,
        )
        self.assertEqual(torch.count_nonzero(actual), torch.count_nonzero(expected))


instantiate_device_type_tests(TestActivationOrdinaryRange, globals())
instantiate_device_type_tests(TestLogSoftmaxOrdinaryRange, globals())
instantiate_device_type_tests(
    TestLogSoftmaxDominancePredicate, globals(), only_for="cpu"
)

if __name__ == "__main__":
    run_tests()
