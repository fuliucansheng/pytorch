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


def product_pair_reference(primal, tangent, dimension):
    if primal.ndim == 0 or primal.numel() == 0:
        return primal.clone(), tangent.clone()
    factors = primal.unbind(dimension)
    directions = tangent.unbind(dimension)
    products = []
    derivatives = []
    for stop in range(1, len(factors) + 1):
        product = torch.ones_like(factors[0])
        for factor in factors[:stop]:
            product = product * factor
        derivative = torch.zeros_like(product)
        for position in range(stop):
            excluded = torch.ones_like(product)
            for other in range(stop):
                if other != position:
                    excluded = excluded * factors[other]
            derivative = derivative + directions[position] * excluded
        products.append(product)
        derivatives.append(derivative)
    return torch.stack(products, dimension), torch.stack(derivatives, dimension)


def decimal_curvature(activation, value):
    exponential = (-abs(value) * (2 if activation == "tanh" else 1)).exp()
    centered = (1 - exponential) / (1 + exponential)
    if value < 0:
        centered = -centered
    variance = exponential / (1 + exponential) ** 2
    if activation == "tanh":
        return -8 * centered * variance
    if activation == "silu":
        return (2 - value * centered) * variance
    return -centered * variance


class TestNumericalDerivativeContinuation(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @dtypes(torch.float32, torch.float64)
    @parametrize("lane_count", [1, 17])
    def test_silu_nested_terminal_backward(self, device, dtype, lane_count):
        pieces = [
            torch.tensor([[-0.7, 0.1], [0.8, 0.0]], device=device, dtype=dtype),
            torch.tensor([[-0.3, 0.2]], device=device, dtype=dtype),
            torch.tensor(
                [[value] * lane_count for value in (-740, -90, -40, 40, 90, 740)],
                device=device,
                dtype=dtype,
            ),
        ]
        inputs = torch.nested.nested_tensor(pieces, requires_grad=True)
        output = torch.nn.functional.silu(inputs)
        (gradient,) = torch.autograd.grad(
            sum(part.sum() for part in output.unbind()), inputs
        )
        tolerance = 4e-13 if dtype == torch.float64 else 4e-5
        for actual, value in zip(gradient.unbind(), pieces):
            with localcontext() as context:
                context.prec = 450
                references = []
                for stored in value.flatten().cpu().tolist():
                    work = Decimal.from_float(stored)
                    exponential = work.exp()
                    sigmoid = exponential / (1 + exponential)
                    references.append(float(sigmoid * (1 + work * (1 - sigmoid))))
            expected = torch.tensor(references, dtype=dtype, device=device).reshape(
                value.shape
            )
            info = torch.finfo(dtype)
            allowed = expected.abs() * tolerance
            subnormal = (expected != 0) & (expected.abs() < info.tiny)
            allowed = torch.where(
                subnormal,
                torch.maximum(
                    allowed, torch.full_like(allowed, 2 * info.tiny * info.eps)
                ),
                allowed,
            )
            self.assertTrue(bool(((actual - expected).abs() <= allowed).all()))
            self.assertTrue(bool((actual[expected != 0] != 0).all()))
        self.assertEqual(
            output.unbind(), [torch.nn.functional.silu(value) for value in pieces]
        )

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("activation", ["sigmoid", "tanh", "silu"])
    @parametrize("lane_count", [1, 17])
    @parametrize("tracked", [False, True])
    def test_activation_polynomial_switch(
        self, device, dtype, activation, lane_count, tracked
    ):
        cutoff = 0.0625 if dtype == torch.float64 else 0.125
        if activation != "tanh":
            cutoff *= 2
        boundary = torch.tensor(cutoff, device=device, dtype=dtype)
        below = torch.nextafter(boundary, torch.zeros_like(boundary))
        above = torch.nextafter(boundary, torch.full_like(boundary, math.inf))
        operation = {
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "silu": torch.nn.functional.silu,
        }[activation]
        tolerance = (
            4e-13
            if dtype == torch.float64
            else 4e-5
            if dtype == torch.float32
            else 0.04
            if dtype == torch.bfloat16
            else 0.005
        )
        for magnitude in (below, boundary, above):
            results = []
            for sign in (-1, 1):
                inputs = torch.full(
                    (lane_count,),
                    float(magnitude) * sign,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                (first,) = torch.autograd.grad(
                    operation(inputs),
                    inputs,
                    torch.full_like(inputs, 0.75),
                    create_graph=True,
                )
                (second,) = torch.autograd.grad(
                    first, inputs, torch.full_like(inputs, 1.25), create_graph=tracked
                )
                with localcontext() as context:
                    context.prec = 100
                    value = Decimal.from_float(float(inputs[0]))
                    reference = (
                        Decimal("0.75")
                        * Decimal("1.25")
                        * decimal_curvature(activation, value)
                    )
                expected = torch.full_like(inputs, float(reference))
                self.assertTrue(bool((second != 0).all()))
                self.assertEqual(second, expected, rtol=tolerance, atol=0)
                if tracked:
                    self.assertIsNotNone(second.grad_fn)
                results.append(second)
            if activation == "silu":
                self.assertTrue(bool((results[0] == results[1]).all()))

    @dtypes(torch.float16, torch.bfloat16)
    def test_log_softmax_reduced_input_connection(self, device, dtype):
        inputs = torch.tensor(
            [1, -0.25, -4], device=device, dtype=dtype, requires_grad=True
        )
        cotangent = torch.tensor([0.75, 0.125, -0.25], device=device, dtype=dtype)
        direction = torch.tensor([0.25, -0.5, 1], device=device, dtype=dtype)
        incoming = torch.tensor([1, 0.25, -0.5], device=device, dtype=dtype)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ):
            output = torch.log_softmax(inputs, 0)
        self.assertTrue(
            any(
                value.dtype == torch.float32
                and value.requires_grad
                and value.grad_fn is not None
                for value in saved
            )
        )
        (first,) = torch.autograd.grad(output, inputs, cotangent, create_graph=True)
        (second,) = torch.autograd.grad(first, inputs, direction, create_graph=True)
        (third,) = torch.autograd.grad(second, inputs, incoming)
        reference_inputs = inputs.detach().to(torch.float64).requires_grad_()
        reference_output = torch.log_softmax(reference_inputs, 0)
        (reference_first,) = torch.autograd.grad(
            reference_output,
            reference_inputs,
            cotangent.to(torch.float64),
            create_graph=True,
        )
        (reference_second,) = torch.autograd.grad(
            reference_first,
            reference_inputs,
            direction.to(torch.float64),
            create_graph=True,
        )
        (reference_third,) = torch.autograd.grad(
            reference_second, reference_inputs, incoming.to(torch.float64)
        )
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.005
        for actual, expected in (
            (first, reference_first),
            (second, reference_second),
            (third, reference_third),
        ):
            expected = expected.to(dtype)
            self.assertTrue(bool((actual[expected != 0] != 0).all()))
            self.assertEqual(actual, expected, rtol=tolerance, atol=0)

        def gradient(values):
            return torch.func.grad(
                lambda value: (torch.log_softmax(value, 0) * cotangent).sum()
            )(values)

        primal, tangent = torch.func.jvp(gradient, (inputs.detach(),), (direction,))
        self.assertEqual(primal, reference_first.to(dtype), rtol=tolerance, atol=0)
        self.assertEqual(tangent, reference_second.to(dtype), rtol=tolerance, atol=0)
        self.assertTrue(bool((tangent[reference_second.to(dtype) != 0] != 0).all()))

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("mode", ["grad", "no_grad", "inference"])
    def test_log_softmax_reduced_input_saved_state(self, device, dtype, mode):
        inputs = torch.tensor(
            [1, -0.25, -4], device=device, dtype=dtype, requires_grad=True
        )
        saved = []
        context = (
            torch.enable_grad()
            if mode == "grad"
            else torch.no_grad()
            if mode == "no_grad"
            else torch.inference_mode()
        )
        with (
            context,
            torch.autograd.graph.saved_tensors_hooks(
                lambda value: saved.append(value) or value, lambda value: value
            ),
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile,
        ):
            output = torch.log_softmax(inputs, 0)
        copies = sum(
            event.count
            for event in profile.key_averages()
            if event.key == "aten::_to_copy"
        )
        if mode == "grad":
            self.assertEqual(len(saved), 2)
            self.assertEqual(copies, 1)
            self.assertTrue(output.requires_grad)
        else:
            self.assertEqual(saved, [])
            self.assertEqual(copies, 0)
            self.assertFalse(output.requires_grad)

    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("mutation", ["input", "output"])
    def test_log_softmax_reduced_input_mutation(self, device, dtype, mutation):
        inputs = torch.tensor(
            [1, -0.25, -4], device=device, dtype=dtype, requires_grad=True
        )
        cotangent = torch.tensor([0.75, 0.125, -0.25], device=device, dtype=dtype)
        original = inputs.detach().to(torch.float64).requires_grad_()
        (reference,) = torch.autograd.grad(
            torch.log_softmax(original, 0), original, cotangent.to(torch.float64)
        )
        output = torch.log_softmax(inputs, 0)
        with torch.no_grad():
            (inputs if mutation == "input" else output).add_(0.25)
        if mutation == "output":
            with self.assertRaisesRegex(
                RuntimeError, "modified by an inplace operation"
            ):
                torch.autograd.grad(output, inputs, cotangent)
        else:
            (actual,) = torch.autograd.grad(output, inputs, cotangent)
            self.assertEqual(
                actual,
                reference.to(dtype),
                rtol=0.04 if dtype == torch.bfloat16 else 0.005,
                atol=0,
            )

    @parametrize("dimension", [0, 1])
    @parametrize("strided", [False, True])
    def test_private_product_scan_complex(self, device, dimension, strided):
        values = torch.tensor(
            [0.5 + 0.25j, 0j, -0.25 + 0.5j, 1 - 0.5j, 0j],
            device=device,
            dtype=torch.complex128,
        )
        directions = torch.tensor(
            [0.25 - 0.5j, 0.75 + 0.25j, -0.5j, 0.5 + 0.75j, 0.25j],
            device=device,
            dtype=values.dtype,
        )
        primal = torch.stack((values, values * (1 + 0.25j)))
        tangent = torch.stack((directions, directions * (0.75 - 0.25j)))
        if strided:
            primal = torch.stack((primal, primal), -1)[..., 0]
            tangent = torch.stack((tangent, tangent), -1)[..., 0]
        if dimension == 0:
            primal = primal.transpose(0, 1)
            tangent = tangent.transpose(0, 1)
        primal.requires_grad_()
        tangent.requires_grad_()

        def operation(value, direction):
            return torch.ops.aten._product_scan_jvp.default(value, direction, dimension)

        def reference(value, direction):
            return product_pair_reference(value, direction, dimension)

        actual = operation(primal, tangent)
        expected = reference(primal, tangent)
        self.assertEqual(actual, expected, rtol=4e-13, atol=0)
        incoming = (
            torch.full_like(primal, 0.75 + 0.5j),
            torch.full_like(primal, -0.25 + 0.75j),
        )
        mixed_direction = (tangent.detach(), primal.detach() + 0.25j)
        for direction in (incoming, mixed_direction, incoming):
            actual = torch.autograd.grad(
                actual, (primal, tangent), direction, create_graph=True
            )
            expected = torch.autograd.grad(
                expected, (primal, tangent), direction, create_graph=True
            )
            self.assertEqual(actual, expected, rtol=4e-13, atol=0)

        def reverse(function, value, direction):
            _, pullback = torch.func.vjp(function, value, direction)
            return pullback(incoming)

        primals = (primal.detach(), tangent.detach())
        actual = torch.func.jvp(
            lambda value, direction: reverse(operation, value, direction),
            primals,
            mixed_direction,
        )
        expected = torch.func.jvp(
            lambda value, direction: reverse(reference, value, direction),
            primals,
            mixed_direction,
        )
        self.assertEqual(actual, expected, rtol=4e-13, atol=0)

        actual = torch.func.jvp(operation, (primal, tangent), mixed_direction)[1]
        expected = torch.func.jvp(reference, (primal, tangent), mixed_direction)[1]
        actual = torch.autograd.grad(actual, (primal, tangent), incoming)
        expected = torch.autograd.grad(expected, (primal, tangent), incoming)
        self.assertEqual(actual, expected, rtol=4e-13, atol=0)

    @parametrize("shape", [(), (0,), (1,), (3, 1)])
    def test_private_product_scan_functional_outputs(self, device, shape):
        primal = torch.full(shape, 0.75, device=device, dtype=torch.float64)
        tangent = torch.full_like(primal, 0.25)

        def operation(value, direction):
            return torch.ops.aten._product_scan_jvp.default(value, direction, -1)

        actual = operation(primal, tangent)
        expected = product_pair_reference(primal, tangent, -1)
        self.assertEqual(actual, expected, rtol=0, atol=0)
        for result in actual:
            self.assertFalse(torch._C._is_alias_of(result, primal))
            self.assertFalse(torch._C._is_alias_of(result, tangent))
        mixed_direction = (tangent, primal)
        self.assertEqual(
            torch.func.jvp(operation, (primal, tangent), mixed_direction),
            torch.func.jvp(
                lambda value, direction: product_pair_reference(value, direction, -1),
                (primal, tangent),
                mixed_direction,
            ),
            rtol=0,
            atol=0,
        )

    @dtypes(torch.float32, torch.float64)
    @parametrize("activation", ["sigmoid", "tanh", "silu"])
    @parametrize("exchanged", [False, True])
    @parametrize("tracked", [False, True])
    def test_activation_finite_scale_product(
        self, device, dtype, activation, exchanged, tracked
    ):
        function = (
            torch.nn.functional.silu
            if activation == "silu"
            else getattr(torch, activation)
        )
        for size in (2, 17):
            scales = (
                [(740.0, 1e-308, 1e308), (1500.0, 1e308, 1e308)]
                if dtype == torch.float64
                else [(100.0, 1e-38, 1e38), (220.0, 1e38, 1e38)]
            )
            for margin, gradient_scale, incoming_scale in scales:
                if exchanged:
                    gradient_scale, incoming_scale = incoming_scale, gradient_scale
                point = margin / 2 if activation == "tanh" else margin
                primal = torch.full(
                    (size,), point, device=device, dtype=dtype, requires_grad=True
                )
                gradient = torch.full_like(primal, gradient_scale)
                incoming = torch.full_like(primal, incoming_scale)
                first = torch.autograd.grad(
                    function(primal), primal, gradient, create_graph=True
                )[0]
                actual = torch.autograd.grad(
                    first, primal, incoming, create_graph=tracked
                )[0]
                with localcontext() as context:
                    context.prec = 850
                    reference = decimal_curvature(
                        activation, Decimal.from_float(float(primal.detach()[0]))
                    )
                    reference *= Decimal.from_float(float(gradient[0]))
                    reference *= Decimal.from_float(float(incoming[0]))
                expected = torch.full_like(primal, float(reference))
                quantum = (
                    torch.nextafter(expected, torch.full_like(expected, math.inf))
                    - expected
                )
                self.assertTrue(bool((actual != 0).all()))
                self.assertEqual(
                    actual,
                    expected,
                    rtol=4e-13 if dtype == torch.float64 else 4e-6,
                    atol=2 * float(quantum.abs().max()),
                )

    @dtypes(torch.float32, torch.float64)
    @parametrize("exchanged", [False, True])
    @parametrize("tracked", [False, True])
    def test_public_log_softmax_backward_finite_scale_product(
        self, device, dtype, exchanged, tracked
    ):
        scales = (
            [(740.0, 1e-308, 1e308), (1500.0, 1e308, 1e308)]
            if dtype == torch.float64
            else [(100.0, 1e-38, 1e38), (220.0, 1e38, 1e38)]
        )
        for size in (2, 17):
            for margin, gradient_scale, incoming_scale in scales:
                if exchanged:
                    gradient_scale, incoming_scale = incoming_scale, gradient_scale
                output = torch.full(
                    (size,), -margin, device=device, dtype=dtype, requires_grad=True
                )
                gradient = torch.zeros_like(output)
                gradient[0] = -gradient_scale
                incoming = torch.zeros_like(output)
                incoming[0] = incoming_scale
                result = torch.ops.aten._log_softmax_backward_data.default(
                    gradient, output, 0, dtype
                )
                actual = torch.autograd.grad(
                    result, output, incoming, create_graph=tracked
                )[0]
                with localcontext() as context:
                    context.prec = 850
                    reference = Decimal.from_float(float(output.detach()[0])).exp()
                    reference *= -Decimal.from_float(float(gradient[0]))
                    reference *= Decimal.from_float(float(incoming[0]))
                expected = torch.zeros_like(actual)
                expected[0] = float(reference)
                quantum = (
                    torch.nextafter(expected[0], torch.full_like(expected[0], math.inf))
                    - expected[0]
                )
                self.assertTrue(bool(actual[0] != 0))
                self.assertEqual(
                    actual,
                    expected,
                    rtol=4e-13 if dtype == torch.float64 else 4e-6,
                    atol=2 * float(quantum.abs()),
                )

    @parametrize("width", [2, 4, 8, 17])
    @parametrize("tangent_source", ["gradient", "output", "both"])
    def test_public_backward_narrow_independent_outputs(
        self, device, width, tangent_source
    ):
        output = torch.full(
            (17, width), -math.log(width), device=device, dtype=torch.float64
        )
        gradient = (
            torch.arange(output.numel(), device=device, dtype=output.dtype).reshape_as(
                output
            )
            / output.numel()
            - 0.25
        )
        direction = torch.cos(gradient * 2)
        tangents = (
            direction if tangent_source != "output" else torch.zeros_like(direction),
            gradient if tangent_source != "gradient" else torch.zeros_like(gradient),
        )

        def operation(cotangent, independent_output):
            return torch.ops.aten._log_softmax_backward_data.default(
                cotangent, independent_output, 1, output.dtype
            )

        def reference(cotangent, independent_output):
            return cotangent - independent_output.exp() * cotangent.sum(1, True)

        self.assertEqual(
            torch.func.jvp(operation, (gradient, output), tangents),
            torch.func.jvp(reference, (gradient, output), tangents),
            rtol=4e-13,
            atol=2e-15,
        )


instantiate_device_type_tests(TestNumericalDerivativeContinuation, globals())

if __name__ == "__main__":
    run_tests()
