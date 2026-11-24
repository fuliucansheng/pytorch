# Owner(s): ["module: inductor"]

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor import codecache, config, cpp_builder
from torch._inductor.codegen.cpp_wrapper_cpu import (
    CppWrapperCpu,
    DeferredCpuTritonCallWrapper,
)
from torch._inductor.codegen.cpp_wrapper_gpu import DeferredTritonCallWrapper
from torch._inductor.codegen.cuda.device_op_overrides import CUDADeviceOpOverrides
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    DualIndentedBuffer,
    fresh_inductor_cache,
    IndentedBuffer,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


@instantiate_parametrized_tests
class TestCppWrapperAnnotations(TestCase):
    @parametrize(
        "windows,nvcc,expected_prefix",
        (
            (
                True,
                r"C:\Program Files\CUDA\bin\nvcc.exe",
                [r"C:\Program Files\CUDA\bin\nvcc.exe"],
            ),
            (
                False,
                '"/cuda toolkit/bin/nvcc" --compiler-options=-fPIC',
                ["/cuda toolkit/bin/nvcc", "--compiler-options=-fPIC"],
            ),
        ),
    )
    def test_cuda_fatbin_compiler_path(self, windows, nvcc, expected_prefix):
        gencode = ["arch=compute_80,code=sm_80", "arch=compute_121,code=sm_121"]
        with (
            mock.patch.object(codecache, "_IS_WINDOWS", windows),
            mock.patch.object(
                codecache.cuda_compile_utils,
                "_cuda_multi_arch_gencode_options",
                return_value=gencode,
            ),
        ):
            command = codecache._cuda_fatbin_command(
                "source directory/kernel.ptx",
                "target directory/kernel.fatbin",
                None,
                nvcc,
                None,
                "80",
            )
        self.assertEqual(
            command,
            expected_prefix
            + [
                "-fatbin",
                "source directory/kernel.ptx",
                "-o",
                "target directory/kernel.fatbin",
                "-gencode",
                gencode[0],
                "-gencode",
                gencode[1],
            ],
        )

    @parametrize("windows", (False, True))
    @parametrize("has_sibling", (False, True))
    def test_cuda_fatbinary_sibling(self, windows, has_sibling):
        compiler = str(Path("cuda toolkit") / "bin" / "nvcc")
        expected_sibling = str(
            Path("cuda toolkit") / "bin" / ("fatbinary.exe" if windows else "fatbinary")
        )
        with (
            mock.patch.object(codecache, "_IS_WINDOWS", windows),
            mock.patch.object(codecache.shutil, "which", return_value="path/fatbinary"),
            mock.patch.object(codecache.os.path, "isfile", return_value=has_sibling),
        ):
            binary = codecache._cuda_fatbinary(compiler)
        self.assertEqual(binary, expected_sibling if has_sibling else "path/fatbinary")

    @parametrize("windows", (False, True))
    def test_cuda_fatbinary_sibling_directory_fallback(self, windows):
        compiler = str(Path("toolkit B") / "bin" / "nvcc")
        path_binary = str(
            Path("toolkit A") / "bin" / ("fatbinary.exe" if windows else "fatbinary")
        )
        with (
            mock.patch.object(codecache, "_IS_WINDOWS", windows),
            mock.patch.object(codecache.shutil, "which", return_value=path_binary),
            mock.patch.object(codecache.os.path, "exists", return_value=True),
            mock.patch.object(codecache.os.path, "isfile", return_value=False),
        ):
            self.assertEqual(codecache._cuda_fatbinary(compiler), path_binary)

    @parametrize("windows", (False, True))
    @parametrize("has_sibling", (False, True))
    @parametrize(
        "compiler,resolved_name",
        (("nvcc.exe", "nvcc.exe"), ("nvcc", "nvcc.exe"), ("nvcc", "NVCC.EXE")),
    )
    def test_cuda_fatbinary_bare_compiler(
        self, windows, has_sibling, compiler, resolved_name
    ):
        resolved_compiler = str(Path("toolkit B") / "bin" / resolved_name)
        sibling = str(Path("toolkit B") / "bin" / "fatbinary.exe")
        path_binary = str(Path("toolkit A") / "bin" / "fatbinary.exe")
        resolved = {compiler: resolved_compiler, "fatbinary": path_binary}
        with (
            mock.patch.object(codecache, "_IS_WINDOWS", windows),
            mock.patch.object(
                codecache.shutil, "which", side_effect=lambda name: resolved.get(name)
            ),
            mock.patch.object(codecache.os.path, "isfile", return_value=has_sibling),
        ):
            binary = codecache._cuda_fatbinary(compiler)
        self.assertEqual(binary, sibling if windows and has_sibling else path_binary)

    @parametrize(
        "compiler,resolved_name",
        (("nvcc.exe", "nvcc.exe"), ("nvcc", "nvcc.exe"), ("nvcc", "NVCC.EXE")),
    )
    def test_cuda_fatbinary_bare_compiler_directory_fallback(
        self, compiler, resolved_name
    ):
        resolved_compiler = str(Path("toolkit B") / "bin" / resolved_name)
        path_binary = str(Path("toolkit A") / "bin" / "fatbinary.exe")
        resolved = {compiler: resolved_compiler, "fatbinary": path_binary}
        with (
            mock.patch.object(codecache, "_IS_WINDOWS", True),
            mock.patch.object(
                codecache.shutil, "which", side_effect=lambda name: resolved.get(name)
            ),
            mock.patch.object(codecache.os.path, "exists", return_value=True),
            mock.patch.object(codecache.os.path, "isfile", return_value=False),
        ):
            self.assertEqual(codecache._cuda_fatbinary(compiler), path_binary)

    @parametrize(
        "compiler,resolved_name",
        (
            ("nvcc.cmd", "nvcc.cmd"),
            ("nvcc.bat", "nvcc.bat"),
            ("nvcc", "nvcc.cmd"),
            ("nvcc", "nvcc.bat"),
            ("nvcc", "NVCC.CMD"),
            ("nvcc", "NVCC.BAT"),
        ),
    )
    def test_cuda_fatbinary_bare_wrapper_fallback(self, compiler, resolved_name):
        resolved_compiler = str(Path("wrapper directory") / resolved_name)
        path_binary = str(Path("toolkit A") / "bin" / "fatbinary.exe")
        resolved = {compiler: resolved_compiler, "fatbinary": path_binary}
        with (
            mock.patch.object(codecache, "_IS_WINDOWS", True),
            mock.patch.object(
                codecache.shutil, "which", side_effect=lambda name: resolved.get(name)
            ),
            mock.patch.object(codecache.os.path, "isfile", return_value=True),
        ):
            self.assertEqual(codecache._cuda_fatbinary(compiler), path_binary)

    @parametrize("windows", (False, True))
    @parametrize("compiler", (None, "missing-nvcc"))
    def test_cuda_fatbinary_no_tool(self, windows, compiler):
        with (
            mock.patch.object(codecache, "_IS_WINDOWS", windows),
            mock.patch.object(codecache.shutil, "which", return_value=None),
            mock.patch.object(codecache.os.path, "isfile", return_value=False),
        ):
            self.assertIsNone(codecache._cuda_fatbinary(compiler))

    @parametrize(
        "windows,platform_tag,expected_command",
        (
            (True, "win-arm64", "armasm64 -o target.obj source.asm"),
            (True, "win-amd64", "ml64 source.asm /c /Fo target.obj"),
            (False, "linux-aarch64", "g++ -c source.asm -o target.obj"),
        ),
    )
    def test_asm_target(self, windows, platform_tag, expected_command):
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", windows),
            mock.patch.object(
                cpp_builder.sysconfig, "get_platform", return_value=platform_tag
            ),
            mock.patch.object(cpp_builder, "get_cpp_compiler", return_value="g++"),
            mock.patch.object(cpp_builder, "_is_intel_compiler", return_value=False),
            mock.patch.object(cpp_builder, "run_compile_cmd") as compile_command,
        ):
            cpp_builder.run_asm_build_object("source.asm", "target.obj", "directory")
        compile_command.assert_called_once_with(expected_command, cwd="directory")

    def test_arm64_asm_paths_with_spaces(self):
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(
                cpp_builder.sysconfig, "get_platform", return_value="win-arm64"
            ),
            mock.patch.object(cpp_builder, "run_compile_cmd") as compile_command,
        ):
            cpp_builder.run_asm_build_object(
                "source directory/source file.asm",
                "target directory/target file.obj",
                "build directory",
            )
        command = compile_command.call_args.args[0]
        self.assertEqual(
            cpp_builder.shlex.split(command),
            [
                "armasm64",
                "-o",
                "target directory/target file.obj",
                "source directory/source file.asm",
            ],
        )
        self.assertEqual(compile_command.call_args.kwargs, {"cwd": "build directory"})

    @parametrize("aot_mode", (False, True))
    def test_cpu_triton_signature(self, aot_mode):
        call = DeferredCpuTritonCallWrapper(
            "launch_kernel", "kernel", [torch.float32, int]
        )
        prefix = IndentedBuffer()
        with V.set_graph_handler(SimpleNamespace(aot_mode=aot_mode)):
            call._write_wrapper_signature(prefix, ["tensor", "numel"])
        code = prefix.getvalue()
        self.assertIn("static AOTI_NOINLINE void launch_kernel(", code)
        self.assertIn("typename tensor_type_", code)
        self.assertIn("int64_t numel", code)
        self.assertEqual("kernels_type_& kernels_" in code, aot_mode)

    def test_cpu_triton_runtime_includes(self):
        prefix = IndentedBuffer()
        CppWrapperCpu._write_cpu_triton_runtime_includes(prefix)
        self.assertIn(
            "#include <torch/csrc/inductor/aoti_runtime/utils.h>", prefix.getvalue()
        )

    def test_gpu_triton_signature(self):
        call = DeferredTritonCallWrapper(
            "launch_kernel", "kernel", {}, [torch.float32, int]
        )
        wrapper = SimpleNamespace(device_codegen=CUDADeviceOpOverrides())
        prefix = DualIndentedBuffer()
        call._write_wrapper_signature(prefix, wrapper, ["tensor", "numel"])
        for code in (prefix.getvalue(), prefix.aot.getvalue()):
            self.assertIn("static AOTI_NOINLINE void launch_kernel(", code)
            self.assertIn("typename tensor_type_", code)
            self.assertIn("int64_t numel", code)
        self.assertNotIn("kernels_type_& kernels_", prefix.getvalue())
        self.assertIn("kernels_type_& kernels_", prefix.aot.getvalue())


class TestCppWrapperAnnotationsDevice(TestCase):
    @onlyCUDA
    @skipIfRocm
    @requires_gpu_and_triton
    def test_aoti_cuda_multi_arch_nonembedded_kernel(self, device):
        with config.patch({"aot_inductor.emit_multi_arch_kernel": True}):
            self._check_aoti_gpu_package_with_constant(device, False)

    @requires_gpu_and_triton
    @parametrize("package_constants_in_so", (False, True))
    def test_aoti_gpu_package_without_constant(self, device, package_constants_in_so):
        class Model(torch.nn.Module):
            def forward(self, tensor):
                return torch.relu(tensor)

        with fresh_inductor_cache(), tempfile.TemporaryDirectory() as directory:
            tensor = torch.randn(32, 32, device=device)
            model = Model().eval()
            exported = torch.export.export(model, (tensor,))
            package = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(Path(directory) / "model.pt2"),
                inductor_configs={
                    "aot_inductor.package_constants_in_so": package_constants_in_so,
                    "aot_inductor.embed_kernel_binary": False,
                },
            )
            actual = torch._inductor.aoti_load_package(package)(tensor)
            self.assertEqual(actual, model(tensor))

    @unittest.skipIf(not IS_WINDOWS, "Windows-only CUDA configuration")
    @onlyCUDA
    @skipIfRocm
    @requires_gpu_and_triton
    def test_aoti_windows_multi_arch_without_embedded_payload(self, device):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(32, 32, device=device))

            def forward(self, tensor):
                return tensor @ self.weight

        with (
            fresh_inductor_cache(),
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                codecache,
                "batch_convert_cubins_to_obj",
                wraps=codecache.batch_convert_cubins_to_obj,
            ) as converter,
        ):
            tensor = torch.randn(32, 32, device=device)
            model = Model().eval()
            exported = torch.export.export(model, (tensor,))
            package = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(Path(directory) / "model.pt2"),
                inductor_configs={
                    "aot_inductor.embed_kernel_binary": True,
                    "aot_inductor.emit_multi_arch_kernel": True,
                    "aot_inductor.package_cpp_only": False,
                    "max_autotune_gemm_backends": "ATEN",
                },
            )
            converter.assert_not_called()
            actual = torch._inductor.aoti_load_package(package)(tensor)
            self.assertEqual(actual, model(tensor))

    @unittest.skipIf(not IS_WINDOWS, "Windows-only CUDA configuration")
    @onlyCUDA
    @skipIfRocm
    @requires_gpu_and_triton
    @parametrize("package_cpp_only", (False, True))
    def test_aoti_windows_embedded_multi_arch_rejected(self, device, package_cpp_only):
        with config.patch(
            {
                "aot_inductor.emit_multi_arch_kernel": True,
                "aot_inductor.package_cpp_only": package_cpp_only,
            }
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Embedded multi-architecture CUDA kernels are not supported on Windows"
                r".*aot_inductor\.emit_multi_arch_kernel=False",
            ):
                self._check_aoti_gpu_package_with_constant(device, True)

    @requires_gpu_and_triton
    def test_aoti_gpu_package_with_constant(self, device):
        self._check_aoti_gpu_package_with_constant(device, False)

    @onlyCUDA
    @skipIfRocm
    @requires_gpu_and_triton
    def test_aoti_cuda_package_with_embedded_kernel(self, device):
        self._check_aoti_gpu_package_with_constant(device, True)

    def _check_aoti_gpu_package_with_constant(self, device, embed_kernel_binary):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(32, 32, device=device))

            def forward(self, tensor):
                return torch.relu(tensor + self.weight)

        with fresh_inductor_cache(), tempfile.TemporaryDirectory() as directory:
            tensor = torch.randn(32, 32, device=device)
            model = Model().eval()
            exported = torch.export.export(model, (tensor,))
            package = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(Path(directory) / "model.pt2"),
                inductor_configs={
                    "aot_inductor.embed_kernel_binary": embed_kernel_binary,
                },
            )
            actual = torch._inductor.aoti_load_package(package)(tensor)
            self.assertEqual(actual, model(tensor))


instantiate_device_type_tests(
    TestCppWrapperAnnotationsDevice, globals(), only_for=(GPU_TYPE,), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
