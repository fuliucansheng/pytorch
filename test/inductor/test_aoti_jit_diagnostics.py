# Owner(s): ["module: inductor"]
import traceback
from unittest import mock

import torch
from torch._inductor import compile_fx, config
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.exc import CppCompileError, InductorError
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase
from torch._inductor.utils import ValueWithLineMap
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


JIT_FAILURE_PREFIX = (
    "When autotune_at_compile_time is False, AOTInductor generates "
    "both JIT code and AOT code. The JIT code failed to run."
)


@instantiate_parametrized_tests
class TestAOTIJitDiagnostics(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def make_graph(self, *, lazy_kernels=True):
        graph = mock.Mock(spec=GraphLowering)
        graph.device_types = {"cuda"}
        graph.aot_mode = True
        graph.operations = []
        graph.wrapper_code = mock.Mock(spec=CppWrapperGpu)
        graph.wrapper_code._lazy_kernel_names = (
            ["triton_kernel"] if lazy_kernels else []
        )
        graph.wrapper_code._aot_output = "aot wrapper"
        graph.codegen.return_value = (ValueWithLineMap("jit wrapper", []), None)
        return graph

    def run_outer_compiler(self, graph):
        gm = torch.fx.symbolic_trace(lambda tensor: (tensor + 1,))
        with (
            config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False}),
            mock.patch.object(
                compile_fx,
                "fx_codegen_and_compile",
                side_effect=lambda *args, **kwargs: (
                    GraphLowering.codegen_with_cpp_wrapper(graph)
                ),
            ),
        ):
            return compile_fx._compile_fx_inner(gm, [torch.ones(1)])

    @config.patch({"triton.autotune_at_compile_time": False})
    @parametrize("boundary", ["jit_companion", "inductor"])
    @parametrize("failure", ["compile", "link", "load", "launch"])
    def test_jit_failure_includes_original_diagnostic(self, boundary, failure):
        errors = {
            "compile": CppCompileError(
                ["cl.exe", "wrapper.cpp"], "error C2039: compiler diagnostic sentinel"
            ),
            "link": CppCompileError(
                ["link.exe", "wrapper.obj", "/OUT:wrapper.pyd"],
                "error LNK2019: unresolved external symbol diagnostic_sentinel",
            ),
            "load": ImportError("DLL load failed: loader diagnostic sentinel"),
            "launch": RuntimeError("CUDA kernel launch diagnostic sentinel"),
        }
        error = errors[failure]
        graph = self.make_graph()

        def raise_jit_failure(*arguments, **keyword_arguments):
            raise error

        graph._run_jit_variant_for_autotune.side_effect = raise_jit_failure
        expected_type = InductorError if boundary == "inductor" else RuntimeError
        with self.assertRaises(expected_type) as caught:
            if boundary == "inductor":
                self.run_outer_compiler(graph)
            else:
                GraphLowering.codegen_with_cpp_wrapper(graph)
        outer = caught.exception
        wrapped = outer.inner_exception if boundary == "inductor" else outer
        self.assertEqual(str(wrapped), JIT_FAILURE_PREFIX + "\n" + str(error))
        self.assertIs(wrapped.__cause__, error)
        self.assertIsNotNone(error.__traceback__)
        origin_frame = list(traceback.walk_tb(error.__traceback__))[-1][0]
        self.assertIs(origin_frame.f_code, raise_jit_failure.__code__)
        self.assertIn(str(error), str(outer))
        if boundary == "inductor":
            self.assertIsNone(outer.__cause__)
            self.assertTrue(outer.__suppress_context__)
            self.assertIs(outer.__context__, wrapped)
        graph._run_jit_variant_for_autotune.assert_called_once()

    @config.patch({"triton.autotune_at_compile_time": False})
    @parametrize("lazy_kernels", [False, True])
    def test_successful_codegen(self, lazy_kernels):
        graph = self.make_graph(lazy_kernels=lazy_kernels)
        with mock.patch(
            "torch._inductor.codegen.cpp_wrapper_gpu.generate_aoti_kernel_config_header",
            return_value="kernel config",
        ):
            wrapper, kernel = GraphLowering.codegen_with_cpp_wrapper(graph)
        self.assertEqual(wrapper.value, "kernel config\naot wrapper")
        self.assertIsNone(kernel)
        self.assertEqual(
            graph._run_jit_variant_for_autotune.call_count, int(lazy_kernels)
        )

    @parametrize("mode", ["cpu", "jit", "compile_time_autotune"])
    def test_sibling_paths_do_not_run_companion(self, mode):
        graph = self.make_graph()
        if mode == "cpu":
            graph.device_types = {"cpu"}
        elif mode == "jit":
            graph.aot_mode = False
        with config.patch(
            {
                "triton.autotune_at_compile_time": mode == "compile_time_autotune",
                "triton.autotune_with_sample_inputs": False,
            }
        ):
            self.assertIs(
                GraphLowering.codegen_with_cpp_wrapper(graph),
                graph.codegen.return_value,
            )
        graph._run_jit_variant_for_autotune.assert_not_called()

    @config.patch({"triton.autotune_at_compile_time": False})
    def test_codegen_failure_is_not_relabelled(self):
        graph = self.make_graph()
        error = ValueError("codegen diagnostic sentinel")
        graph.codegen.side_effect = error
        with self.assertRaises(ValueError) as caught:
            GraphLowering.codegen_with_cpp_wrapper(graph)
        self.assertIs(caught.exception, error)
        graph._run_jit_variant_for_autotune.assert_not_called()

    @config.patch({"triton.autotune_at_compile_time": False})
    @parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_interrupt_is_not_wrapped(self, interrupt):
        graph = self.make_graph()
        error = interrupt("interrupt diagnostic sentinel")
        graph._run_jit_variant_for_autotune.side_effect = error
        with self.assertRaises(interrupt) as caught:
            GraphLowering.codegen_with_cpp_wrapper(graph)
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    run_tests()
