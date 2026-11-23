import os
import shutil
import struct
import sysconfig
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch._inductor import codecache, config, cpp_builder
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
)


@instantiate_parametrized_tests
class TestAOTIConstants(TestCase):
    @parametrize(
        "machine,target",
        (("ARM64", "win-arm64"), ("AMD64", "win-amd64"), ("ARM64", "win-amd64")),
    )
    def test_windows_assembler_command(self, machine, target):
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(cpp_builder.platform, "machine", return_value=machine),
            mock.patch.object(cpp_builder.sysconfig, "get_platform", return_value=target),
            mock.patch.object(cpp_builder, "run_compile_cmd") as compile_command,
        ):
            cpp_builder.run_asm_build_object("constants.asm", "constants.obj", ".")
        if target == "win-arm64":
            expected = "armasm64 -o constants.obj constants.asm"
        else:
            expected = "ml64 constants.asm /c /Fo constants.obj"
        compile_command.assert_called_once_with(expected, cwd=".")

    @parametrize("intel", (False, True))
    def test_non_windows_assembler_command(self, intel):
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", False),
            mock.patch.object(cpp_builder, "get_cpp_compiler", return_value="c++"),
            mock.patch.object(cpp_builder, "_is_intel_compiler", return_value=intel),
            mock.patch.object(cpp_builder, "run_compile_cmd") as compile_command,
        ):
            cpp_builder.run_asm_build_object("constants.S", "constants.o", ".")
        compiler = "gcc" if intel else "c++"
        compile_command.assert_called_once_with(
            f"{compiler} -c constants.S -o constants.o", cwd="."
        )

    @unittest.skipUnless(IS_WINDOWS, "requires Windows COFF assemblers")
    @parametrize("target", ("win-arm64", "win-amd64"))
    def test_windows_constants_object_machine(self, target):
        compiler = "armasm64" if target == "win-arm64" else "ml64"
        if shutil.which(compiler) is None:
            self.skipTest(f"{compiler} is unavailable")
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(cpp_builder.platform, "machine", return_value="ARM64"),
            mock.patch.object(cpp_builder.sysconfig, "get_platform", return_value=target),
        ):
            source = Path(directory) / "constants.asm"
            object_file = source.with_suffix(".obj")
            source.write_text(
                cpp_builder.get_windows_zero_size_constants_asm(), encoding="utf-8"
            )
            cpp_builder.run_asm_build_object(str(source), str(object_file), directory)
            machine_header = struct.unpack("<H", object_file.read_bytes()[:2])[0]
        self.assertEqual(machine_header, 0xAA64 if target == "win-arm64" else 0x8664)


class TestAOTIConstantsIntegration(TestCase):
    @unittest.skipUnless(IS_WINDOWS, "requires native Windows AOTInductor")
    @parametrize("with_constants", (False, True))
    def test_native_aoti_constants(self, device, with_constants):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if with_constants:
                    self.register_buffer("weight", torch.randn(32, 32, device=device))

            def forward(self, tensor):
                if with_constants:
                    return tensor.sin() + self.weight
                return tensor.sin() + tensor.cos()

        machine_headers = []
        build_object = codecache.run_asm_build_object

        def record_object(source, target, directory):
            build_object(source, target, directory)
            machine_headers.append(struct.unpack("<H", Path(target).read_bytes()[:2])[0])

        with (
            tempfile.TemporaryDirectory() as directory,
            fresh_inductor_cache(),
            config.patch({"compile_threads": 1}),
            mock.patch.object(codecache, "run_asm_build_object", side_effect=record_object),
        ):
            tensor = torch.randn(32, 32, device=device)
            model = Model().eval()
            exported = torch.export.export(model, (tensor,))
            package = torch._inductor.aoti_compile_and_package(
                exported, package_path=os.path.join(directory, "model.pt2")
            )
            actual = torch._inductor.aoti_load_package(package)(tensor)
            self.assertEqual(actual, model(tensor))
        expected_machine = (
            0xAA64 if sysconfig.get_platform() == "win-arm64" else 0x8664
        )
        self.assertEqual(machine_headers, [] if with_constants else [expected_machine])


instantiate_device_type_tests(TestAOTIConstantsIntegration, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
