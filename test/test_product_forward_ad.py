# Owner(s): ["module: autograd"]

import itertools

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


def cumulative_reference(primal, dimension, dtype=None):
    converted = primal.to(dtype=dtype or primal.dtype)
    if converted.ndim == 0 or converted.numel() == 0:
        return converted.clone()
    factors = converted.unbind(dimension)
    product = torch.ones_like(factors[0])
    outputs = []
    for factor in factors:
        product = product * factor
        outputs.append(product)
    return torch.stack(outputs, dimension)


def scatter_product_reference(primal, source, index, dimension, include_self):
    if primal.ndim == 0:
        return primal * source if include_self else source.clone()
    dimension %= primal.ndim
    positions = list(itertools.product(*(range(size) for size in index.shape)))
    outputs = []
    for destination in itertools.product(*(range(size) for size in primal.shape)):
        matching = [
            position
            for position in positions
            if int(index[position]) == destination[dimension]
            and all(
                position[axis] == destination[axis]
                for axis in range(primal.ndim)
                if axis != dimension
            )
        ]
        product = primal[destination]
        if matching and not include_self:
            product = torch.ones_like(product)
        for position in matching:
            product = product * source[position]
        outputs.append(product)
    if not outputs:
        return primal.clone()
    return torch.stack(outputs).reshape(primal.shape)


def differentiated(function, directions):
    result = function
    for direction in directions:
        previous = result

        def result(*primals, previous=previous, direction=direction):
            return torch.func.jvp(previous, primals, direction)[1]

    return result


class TestProductForwardAD(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("point", [-1.0, 0.0, 1.0, 2.0])
    def test_cumprod_reported_polynomial(self, device, point):
        primal = torch.tensor(point, device=device, dtype=torch.float64)
        weights = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], device=device, dtype=torch.float64
        )

        def inputs(value):
            return torch.stack(
                (value, value + 1, 2 - value, value - 1, 2 * value + 1, value + 3)
            ).reshape(2, 3)

        def operation(value):
            return (inputs(value).cumprod(-1) * weights).sum()

        def reference(value):
            return (cumulative_reference(inputs(value), -1) * weights).sum()

        directions = ((torch.ones_like(primal),),) * 3
        for order in range(4):
            actual = differentiated(operation, directions[:order])(primal)
            expected = differentiated(reference, directions[:order])(primal)
            self.assertEqual(actual, expected, rtol=4e-13, atol=0)
            if point == 1 and order == 2:
                self.assertEqual(actual.item(), 144)

    @parametrize("dimension", [0, 1])
    @parametrize("zero_count", [0, 1, 2])
    @parametrize("strided", [False, True])
    def test_cumprod_mixed_directions(self, device, dimension, zero_count, strided):
        primal = torch.tensor(
            [[1, 2, -1, 3, 2], [2, -1, 3, 1, -2]], device=device, dtype=torch.float64
        )
        if zero_count:
            primal[:, 0] = 0
        if zero_count > 1:
            primal[:, 2] = 0
        if strided:
            primal = primal.repeat_interleave(2, dim=1)[:, ::2]
            self.assertFalse(primal.is_contiguous())
        first = torch.arange(
            1, primal.numel() + 1, device=device, dtype=primal.dtype
        ).reshape_as(primal)
        second = first.flip(1) * -0.5
        third = first.flip(0) * 0.25
        directions = ((first,), (second,), (third,))

        def operation(value):
            return value.cumprod(dimension)

        def reference(value):
            return cumulative_reference(value, dimension)

        for order in range(4):
            self.assertEqual(
                differentiated(operation, directions[:order])(primal),
                differentiated(reference, directions[:order])(primal),
                rtol=4e-13,
                atol=0,
            )
        differentiable = primal.detach().requires_grad_()
        actual = torch.func.jvp(operation, (differentiable,), (first,))[1]
        expected = torch.func.jvp(reference, (differentiable,), (first,))[1]
        self.assertEqual(
            torch.autograd.grad(actual.sum(), differentiable)[0],
            torch.autograd.grad(expected.sum(), differentiable)[0],
            rtol=4e-13,
            atol=0,
        )

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @parametrize("output_dtype", [None, torch.float64])
    def test_cumprod_conversion_scalar_empty(self, device, dtype, output_dtype):
        for values in (0.0, 2.0, [], [0.0, 2.0, 0.0, 1.0]):
            primal = torch.tensor(values, device=device, dtype=dtype)
            direction = torch.ones_like(primal)

            def operation(value):
                return torch.cumprod(value, -1, dtype=output_dtype)

            def reference(value):
                return cumulative_reference(value, -1, output_dtype)

            actual = torch.func.jvp(operation, (primal,), (direction,))
            expected = torch.func.jvp(reference, (primal,), (direction,))
            self.assertEqual(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual[1].dtype, output_dtype or dtype)

    def test_cumprod_complex(self, device):
        primal = torch.tensor(
            [1 + 2j, 0, -2 + 1j, 0], device=device, dtype=torch.complex128
        )
        direction = torch.tensor(
            [2 - 1j, 1 + 2j, -1j, 3 + 1j], device=device, dtype=primal.dtype
        )

        def operation(value):
            return value.cumprod(0)

        def reference(value):
            return cumulative_reference(value, 0)

        for order in range(3):
            directions = ((direction,),) * order
            self.assertEqual(
                differentiated(operation, directions)(primal),
                differentiated(reference, directions)(primal),
                rtol=4e-13,
                atol=0,
            )

    def test_scatter_group_isolation(self, device):
        source = torch.tensor(
            [1e308, 1e308, 3, 4], device=device, dtype=torch.float64, requires_grad=True
        )
        primal = torch.ones(2, device=device, dtype=source.dtype)
        index = torch.tensor([0, 0, 1, 1], device=device)

        def operation(factors):
            return primal.scatter_reduce(0, index, factors, "prod", include_self=False)[
                1
            ]

        def reference(factors):
            return factors[2] * factors[3]

        actual = torch.func.jvp(operation, (source,), (torch.ones_like(source),))[1]
        expected = torch.func.jvp(reference, (source,), (torch.ones_like(source),))[1]
        self.assertEqual(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.item(), 7)
        self.assertEqual(
            torch.autograd.grad(actual, source)[0],
            torch.autograd.grad(expected, source)[0],
            rtol=0,
            atol=0,
        )

    @parametrize("include_self", [False, True])
    def test_scatter_untouched_isolation(self, device, include_self):
        primal = torch.tensor(
            [float("inf"), 3], device=device, dtype=torch.float64, requires_grad=True
        )
        source = torch.tensor(
            [2.0, 4.0], device=device, dtype=primal.dtype, requires_grad=True
        )
        index = torch.tensor([1, 1], device=device)

        def operation(target, factors):
            return target.scatter_reduce(
                0, index, factors, "prod", include_self=include_self
            )

        actual = torch.func.jvp(
            operation,
            (primal, source),
            (torch.ones_like(primal), torch.ones_like(source)),
        )[1]
        expected = torch.tensor(
            [1, 26 if include_self else 6], device=device, dtype=primal.dtype
        )
        self.assertEqual(actual, expected, rtol=0, atol=0)
        gradients = torch.autograd.grad(
            actual[0], (primal, source), allow_unused=True, materialize_grads=True
        )
        self.assertEqual(
            gradients,
            (torch.zeros_like(primal), torch.zeros_like(source)),
            rtol=0,
            atol=0,
        )

    def test_scatter_reported_polynomial(self, device):
        primal = torch.tensor(0.01, device=device, dtype=torch.float64)
        index = torch.tensor([0, 0, 2], device=device)

        def operation(value):
            target = torch.stack((value + 1, 2 * value + 1, value + 3))
            source = torch.stack((value + 2, 3 * value + 1, 2 * value + 3))
            result = target.scatter_reduce(0, index, source, "prod", include_self=True)
            return 2 * result[0] - 3 * result[1] + 5 * result[2]

        def reference(value):
            return 6 * value**3 + 30 * value**2 + 57 * value + 46

        for order in range(4):
            directions = ((torch.ones_like(primal),),) * order
            actual = differentiated(operation, directions)(primal)
            expected = differentiated(reference, directions)(primal)
            self.assertEqual(actual, expected, rtol=4e-13, atol=0)
            if order == 1:
                self.assertEqual(actual.item(), 57.6018, rtol=4e-13, atol=0)

    @parametrize("include_self", [False, True])
    @parametrize("dimension", [0, 1])
    @parametrize("zero_count", [0, 1, 2])
    @parametrize("strided", [False, True])
    def test_scatter_mixed_directions(
        self, device, include_self, dimension, zero_count, strided
    ):
        primal = torch.tensor(
            [[2, -1, 3, 4], [1, 2, -2, 3]], device=device, dtype=torch.float64
        )
        source = torch.tensor(
            [[2, -1, 3, 1, 2], [1, 2, -2, 3, -1]], device=device, dtype=primal.dtype
        )
        index = torch.tensor([[0, 0, 2, 0, 2], [1, 1, 3, 1, 3]], device=device)
        if zero_count:
            source[:, 0] = 0
            primal[0, 2] = 0
        if zero_count > 1:
            source[:, 1] = 0
        if dimension == 0:
            primal, source, index = primal.t(), source.t(), index.t()
        if strided:
            primal = primal.repeat_interleave(2, dim=dimension).index_select(
                dimension, torch.arange(primal.size(dimension), device=device) * 2
            )
            expanded = source.repeat_interleave(2, dim=dimension)
            slices = [slice(None)] * source.ndim
            slices[dimension] = slice(None, None, 2)
            source = expanded[tuple(slices)]
            self.assertFalse(source.is_contiguous())
        first_self = torch.arange(
            1, primal.numel() + 1, device=device, dtype=primal.dtype
        ).reshape_as(primal)
        first_src = torch.arange(
            1, source.numel() + 1, device=device, dtype=source.dtype
        ).reshape_as(source)
        directions = (
            (first_self, first_src),
            (first_self * -0.5, first_src.flip(dimension) * 0.25),
            (first_self.flip(dimension) * 0.5, first_src * -0.25),
        )

        def operation(target, values):
            return target.scatter_reduce(
                dimension, index, values, "prod", include_self=include_self
            )

        def reference(target, values):
            return scatter_product_reference(
                target, values, index, dimension, include_self
            )

        for order in range(4):
            self.assertEqual(
                differentiated(operation, directions[:order])(primal, source),
                differentiated(reference, directions[:order])(primal, source),
                rtol=4e-13,
                atol=0,
            )
        differentiable = (
            primal.detach().requires_grad_(),
            source.detach().requires_grad_(),
        )
        actual = torch.func.jvp(operation, differentiable, directions[0])[1]
        expected = torch.func.jvp(reference, differentiable, directions[0])[1]
        self.assertEqual(
            torch.autograd.grad(
                actual.sum(), differentiable, allow_unused=True, materialize_grads=True
            ),
            torch.autograd.grad(
                expected.sum(),
                differentiable,
                allow_unused=True,
                materialize_grads=True,
            ),
            rtol=4e-13,
            atol=0,
        )

    @parametrize("include_self", [False, True])
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_scatter_empty_scalar_and_source_prefix(self, device, include_self, dtype):
        cases = (
            (2.0, 3.0, 0),
            ([1.0, 2.0, 3.0], [], []),
            ([], [], []),
            ([2.0, 3.0, 4.0], [0.0, 2.0, 9.0], [0, 0]),
        )
        for values, sources, indices in cases:
            primal = torch.tensor(values, device=device, dtype=dtype)
            source = torch.tensor(sources, device=device, dtype=dtype)
            index = torch.tensor(indices, device=device, dtype=torch.long)
            directions = (torch.ones_like(primal), torch.ones_like(source))

            def operation(target, factors):
                return target.scatter_reduce(
                    0, index, factors, "prod", include_self=include_self
                )

            def reference(target, factors):
                return scatter_product_reference(
                    target, factors, index, 0, include_self
                )

            self.assertEqual(
                torch.func.jvp(operation, (primal, source), directions),
                torch.func.jvp(reference, (primal, source), directions),
                rtol=0,
                atol=0,
            )


instantiate_device_type_tests(TestProductForwardAD, globals())


if __name__ == "__main__":
    run_tests()
