# Owner(s): ["module: cuda"]

import os
import subprocess
import sys
import textwrap
import unittest

import torch
from torch.testing._internal.common_cuda import (
    _get_torch_rocm_version,
    TEST_CUDA,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)


# NOTE: this needs to be run in a brand new process

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestCudaDynamoInitialization(TestCase):
    def _run_fresh_python(self, source):
        child_environment = os.environ.copy()
        torch_package_root = os.path.dirname(os.path.dirname(torch.__file__))
        child_python_path = [torch_package_root]
        inherited_python_path = child_environment.get("PYTHONPATH")
        if inherited_python_path:
            child_python_path.append(inherited_python_path)
        child_environment["PYTHONPATH"] = os.pathsep.join(child_python_path)
        completed_process = subprocess.run(
            [
                sys.executable,
                "-s",
                "-B",
                "-c",
                source,
                torch.__file__,
                torch._C.__file__,
            ],
            capture_output=True,
            env=child_environment,
            text=True,
        )
        self.assertEqual(
            completed_process.returncode,
            0,
            msg=(
                f"child stdout:\n{completed_process.stdout}\n"
                f"child stderr:\n{completed_process.stderr}"
            ),
        )
        print(completed_process.stdout, end="")

    def test_cpu_cond_does_not_initialize_cuda(self):
        source = textwrap.dedent(
            """
            import os
            import sys

            import torch

            assert os.path.realpath(torch.__file__) == os.path.realpath(sys.argv[1])
            assert os.path.realpath(torch._C.__file__) == os.path.realpath(sys.argv[2])
            assert not torch.cuda.is_initialized()
            for device_index in range(torch.cuda.device_count()):
                assert not torch._C._cuda_hasPrimaryContext(device_index)

            def true_branch(operand):
                return operand + 1

            def false_branch(operand):
                return operand.clone()

            def conditional(operand, predicate):
                return torch.cond(
                    predicate, true_branch, false_branch, (operand,)
                )

            input_values = torch.arange(
                1, 5, dtype=torch.float32, requires_grad=True
            )
            predicate = torch.tensor(True)
            actual_output = conditional(input_values, predicate)
            torch.testing.assert_close(actual_output, input_values + 1)
            assert not torch.cuda.is_initialized()
            for device_index in range(torch.cuda.device_count()):
                assert not torch._C._cuda_hasPrimaryContext(device_index)
            print(f"CHILD_TORCH={torch.__file__}")
            print(f"CHILD_NATIVE={torch._C.__file__}")
            """
        )
        self._run_fresh_python(source)

    def test_first_cpu_cond_aot_eager_does_not_initialize_cuda(self):
        source = textwrap.dedent(
            """
            import os
            import sys

            import torch
            from torch._dynamo.testing import CompileCounterWithBackend

            assert os.path.realpath(torch.__file__) == os.path.realpath(sys.argv[1])
            assert os.path.realpath(torch._C.__file__) == os.path.realpath(sys.argv[2])
            assert not torch.cuda.is_initialized()
            for device_index in range(torch.cuda.device_count()):
                assert not torch._C._cuda_hasPrimaryContext(device_index)

            def true_branch(operand):
                return operand + 1

            def false_branch(operand):
                return operand.clone()

            def conditional(operand, predicate):
                return torch.cond(
                    predicate, true_branch, false_branch, (operand,)
                )

            oracle_input = torch.arange(
                1, 5, dtype=torch.float32, requires_grad=True
            )
            expected_output = true_branch(oracle_input)
            expected_output.sum().backward()
            expected_gradient = oracle_input.grad.clone()
            assert not torch.cuda.is_initialized()

            backend_counter = CompileCounterWithBackend("aot_eager")
            compiled_input = torch.arange(
                1, 5, dtype=torch.float32, requires_grad=True
            )
            predicate = torch.tensor(True)
            compiled_conditional = torch.compile(
                conditional, backend=backend_counter, fullgraph=True
            )
            actual_output = compiled_conditional(compiled_input, predicate)
            actual_output.sum().backward()

            assert backend_counter.frame_count == 1
            torch.testing.assert_close(actual_output, expected_output)
            torch.testing.assert_close(compiled_input.grad, expected_gradient)
            assert not torch.cuda.is_initialized()
            for device_index in range(torch.cuda.device_count()):
                assert not torch._C._cuda_hasPrimaryContext(device_index)
            print(f"CHILD_TORCH={torch.__file__}")
            print(f"CHILD_NATIVE={torch._C.__file__}")
            """
        )
        self._run_fresh_python(source)

    def test_dynamo_streams_after_cuda_initialization(self):
        source = textwrap.dedent(
            """
            import os
            import sys

            import torch
            from torch._dynamo.testing import CompileCounterWithBackend

            assert os.path.realpath(torch.__file__) == os.path.realpath(sys.argv[1])
            assert os.path.realpath(torch._C.__file__) == os.path.realpath(sys.argv[2])
            assert not torch.cuda.is_initialized()

            def current_stream_computation(host_values):
                active_stream = torch.cuda.current_stream()
                with torch.cuda.stream(active_stream):
                    return host_values.to("cuda") + 1

            def new_stream_computation(host_values):
                created_stream = torch.cuda.Stream()
                with torch.cuda.stream(created_stream):
                    return host_values.to("cuda") + 1

            initial_values = torch.arange(
                4, dtype=torch.float32, device="cuda"
            )
            assert torch.cuda.is_initialized()
            default_stream = torch.cuda.current_stream()
            explicit_stream = torch.cuda.Stream()
            explicit_stream.wait_stream(default_stream)
            with torch.cuda.stream(explicit_stream):
                assert torch.cuda.current_stream() == explicit_stream
                explicit_output = initial_values + 1
            explicit_stream.synchronize()
            expected_output = torch.arange(4, dtype=torch.float32) + 1
            torch.testing.assert_close(explicit_output.cpu(), expected_output)
            assert explicit_output.dtype == torch.float32
            assert torch.cuda.current_stream() == default_stream

            host_values = torch.arange(4, dtype=torch.float32)
            for stream_computation in (
                current_stream_computation,
                new_stream_computation,
            ):
                backend_counter = CompileCounterWithBackend("aot_eager")
                compiled_computation = torch.compile(
                    stream_computation,
                    backend=backend_counter,
                    fullgraph=True,
                )
                actual_output = compiled_computation(host_values)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    actual_output.cpu(), host_values + 1
                )
                assert actual_output.dtype == torch.float32
                assert backend_counter.frame_count == 1
                assert torch.cuda.current_stream() == default_stream
            print(f"CHILD_TORCH={torch.__file__}")
            print(f"CHILD_NATIVE={torch._C.__file__}")
            """
        )
        self._run_fresh_python(source)


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaPrimaryCtx(TestCase):
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in test_cuda_primary_ctx.py must be run in a process "
        "where CUDA contexts are never created. Use either run_test.py or add "
        "--subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        super().setUp()
        for device in range(torch.cuda.device_count()):
            # Ensure context has not been created beforehand
            self.assertFalse(
                torch._C._cuda_hasPrimaryContext(device),
                TestCudaPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG,
            )

    def test_set_device_0(self):
        # In CUDA 12 the behavior of cudaSetDevice has changed. It eagerly creates context on target.
        # The behavior of `torch.cuda.set_device(0)` should also create context on the device 0.
        # Initially, we should not have any context on device 0.
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        torch.cuda.set_device(0)
        if TEST_WITH_ROCM and _get_torch_rocm_version() < (7, 14):
            # hipSetDevice was lazy through ROCm 7.2: the primary context
            # appears on first real use, not on set_device. Probed lazy on 7.2
            # and eager on 7.14; if a version in between proves eager, lower
            # this threshold.
            self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
            torch.empty(1, device="cuda:0")
            self.assertTrue(torch._C._cuda_hasPrimaryContext(0))
        else:
            # Now after the device was set, the context should present in
            # CUDA 12, and in ROCm 7.14+ which matches the eager behavior.
            self.assertTrue(torch._C._cuda_hasPrimaryContext(0))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_str_repr(self):
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        str(x)
        repr(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy(self):
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        y = torch.randn(1, device="cpu")
        y.copy_(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_pin_memory(self):
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        self.assertFalse(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu").pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        self.assertTrue(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.zeros(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.empty(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = x.pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))


if __name__ == "__main__":
    run_tests()
