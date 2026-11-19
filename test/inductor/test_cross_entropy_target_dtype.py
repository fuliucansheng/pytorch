# Owner(s): ["oncall: cpu inductor"]
import torch
import torch.nn.functional as functional
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
)


class TestCrossEntropyTargetDtype(TestCase):
    hw_classification = HardwareClassification.CPU

    def _check_loss_and_grad(self, function, logits, target, backend):
        expected = function(logits, target)
        actual = torch.compile(function, backend=backend, fullgraph=True)(
            logits, target
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            torch.autograd.grad(actual.sum(), logits),
            torch.autograd.grad(expected.sum(), logits),
        )

    @parametrize("backend", ("aot_eager", "inductor"))
    @parametrize("shape", ((4,), (3, 4), (2, 4, 3), (2, 4, 2, 3)))
    @parametrize(
        "target_dtype", (torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool)
    )
    def test_target_dtype(self, device, backend, shape, target_dtype):
        torch.manual_seed(0)
        logits = torch.randn(
            shape, dtype=torch.float64, device=device, requires_grad=True
        )
        target_shape = shape[:1] + shape[2:] if len(shape) > 1 else ()
        target = torch.randint(4, target_shape, device=device).to(target_dtype)

        def loss_function(logits, target):
            return functional.cross_entropy(logits, target)

        if target_dtype == torch.int64 or (
            target_dtype == torch.uint8 and len(shape) <= 2
        ):
            self._check_loss_and_grad(loss_function, logits, target, backend)
        elif target_dtype == torch.uint8:
            with self.assertRaisesRegex(RuntimeError, "expected scalar type Long"):
                loss_function(logits, target)
            with self.assertRaisesRegex(RuntimeError, r"gather\(\): Expected dtype"):
                torch.compile(loss_function, backend=backend, fullgraph=True)(
                    logits, target
                )
        else:
            message = "expected target dtype to be Long or Byte"
            with self.assertRaisesRegex(RuntimeError, message):
                loss_function(logits, target)
            with self.assertRaisesRegex(RuntimeError, message):
                torch.compile(loss_function, backend=backend, fullgraph=True)(
                    logits, target
                )

    @parametrize("backend", ("aot_eager", "inductor"))
    @parametrize("target_dtype", (torch.int64, torch.uint8))
    @parametrize("reduction", ("none", "sum", "mean"))
    def test_weighted_ignore_index(self, device, backend, target_dtype, reduction):
        logits = torch.randn(
            3, 4, dtype=torch.float64, device=device, requires_grad=True
        )
        ignore_index = 255 if target_dtype == torch.uint8 else -100
        target = torch.tensor([0, ignore_index, 2], dtype=target_dtype, device=device)
        weight = torch.rand(4, dtype=logits.dtype, device=device)

        def loss_function(logits, target):
            return functional.cross_entropy(
                logits,
                target,
                weight=weight,
                reduction=reduction,
                ignore_index=ignore_index,
            )

        self._check_loss_and_grad(loss_function, logits, target, backend)

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_byte_negative_ignore_index_does_not_wrap(self, device, backend):
        logits = torch.randn(
            3, 256, dtype=torch.float64, device=device, requires_grad=True
        )
        target = torch.tensor([255, 1, 2], dtype=torch.uint8, device=device)

        def loss_function(logits, target):
            return functional.cross_entropy(logits, target, ignore_index=-1)

        self._check_loss_and_grad(loss_function, logits, target, backend)

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_probability_target_control(self, device, backend):
        logits = torch.randn(
            3, 4, dtype=torch.float64, device=device, requires_grad=True
        )
        target = torch.randn_like(logits).softmax(dim=1)
        weight = torch.rand(4, dtype=logits.dtype, device=device)

        def loss_function(logits, target):
            return functional.cross_entropy(
                logits, target, weight=weight, label_smoothing=0.1
            )

        self._check_loss_and_grad(loss_function, logits, target, backend)

    @parametrize("backend", ("aot_eager", "inductor"))
    @parametrize("ignore_index", (-257, -100, -1, 0, 127, 128, 255, 256, 511))
    def test_byte_ignore_index_boundaries(self, device, backend, ignore_index):
        logits = torch.randn(
            4, 256, dtype=torch.float64, device=device, requires_grad=True
        )
        target = torch.tensor([0, 127, 128, 255], dtype=torch.uint8, device=device)

        def loss_function(logits, target):
            return functional.cross_entropy(logits, target, ignore_index=ignore_index)

        self._check_loss_and_grad(loss_function, logits, target, backend)

    @parametrize("backend", ("aot_eager", "inductor"))
    @parametrize("target_dtype", (torch.uint8, torch.int64))
    def test_explicit_target_cast_control(self, device, backend, target_dtype):
        torch.manual_seed(1234)
        logits = torch.randn(
            3, 256, dtype=torch.float64, device=device, requires_grad=True
        )
        target = torch.tensor([255, 1, 2], dtype=target_dtype, device=device)

        def loss_function(logits, target):
            return functional.cross_entropy(
                logits, target.to(torch.int64), ignore_index=-1
            )

        self._check_loss_and_grad(loss_function, logits, target, backend)

    @parametrize("backend", ("aot_eager", "inductor"))
    @parametrize("target_dtype", (torch.uint8, torch.int64))
    @parametrize("weighted", (False, True))
    def test_index_target_smoothing_control(
        self, device, backend, target_dtype, weighted
    ):
        logits = torch.randn(
            3, 4, dtype=torch.float64, device=device, requires_grad=True
        )
        target = torch.tensor([0, 1, 2], dtype=target_dtype, device=device)
        weight = torch.rand(4, dtype=logits.dtype, device=device) if weighted else None

        def loss_function(logits, target):
            return functional.cross_entropy(
                logits, target, weight=weight, label_smoothing=0.1
            )

        if weighted and target_dtype == torch.uint8:
            message = r"gather\(\): Expected dtype int32/int64 for index"
            with self.assertRaisesRegex(RuntimeError, message):
                loss_function(logits, target)
            with self.assertRaisesRegex(RuntimeError, message):
                torch.compile(loss_function, backend=backend, fullgraph=True)(
                    logits, target
                )
        else:
            self._check_loss_and_grad(loss_function, logits, target, backend)


instantiate_device_type_tests(TestCrossEntropyTargetDtype, globals(), only_for=("cpu",))


if __name__ == "__main__":
    run_tests()
