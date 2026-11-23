from unittest import mock

from torch._inductor import config
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.exc import CppCompileError, InductorError
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase
from torch._inductor.utils import ValueWithLineMap
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@instantiate_parametrized_tests
class TestAOTIJitDiagnostics(TestCase):
    def make_graph(self, *, lazy_kernels=True):
        graph = mock.Mock(spec=GraphLowering)
        graph.device_types = {"cuda"}
        graph.aot_mode = True
        graph.operations = []
        graph.wrapper_code = mock.Mock(spec=CppWrapperGpu)
        graph.wrapper_code._lazy_kernel_names = ["triton_kernel"] if lazy_kernels else []
        graph.wrapper_code._aot_output = "aot wrapper"
        graph.codegen.return_value = (ValueWithLineMap("jit wrapper", []), None)
        return graph

    @config.patch({"triton.autotune_at_compile_time": False})
    @parametrize("failure", ["compile", "load", "launch"])
    def test_jit_failure_includes_original_diagnostic(self, failure):
        errors = {
            "compile": CppCompileError(
                ["c++", "wrapper.cpp"], "error: compiler diagnostic sentinel"
            ),
            "load": ImportError("library load diagnostic sentinel"),
            "launch": RuntimeError("kernel launch diagnostic sentinel"),
        }
        error = errors[failure]
        graph = self.make_graph()
        graph._run_jit_variant_for_autotune.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            GraphLowering.codegen_with_cpp_wrapper(graph)
        self.assertIn("The JIT code failed to run.", str(caught.exception))
        self.assertIs(caught.exception.__cause__, error)
        translated = InductorError(caught.exception, None)
        self.assertIs(translated.inner_exception, caught.exception)
        self.assertIn(str(error), str(translated))
        self.assertIn(str(error), str(caught.exception))
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
                GraphLowering.codegen_with_cpp_wrapper(graph), graph.codegen.return_value
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
    def test_interrupt_is_not_wrapped(self):
        graph = self.make_graph()
        graph._run_jit_variant_for_autotune.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            GraphLowering.codegen_with_cpp_wrapper(graph)


if __name__ == "__main__":
    run_tests()
