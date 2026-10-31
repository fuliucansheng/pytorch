# Owner(s): ["module: build", "module: windows"]

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestWindowsWheelBuild(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("gpu_arch_type", ("cpu", "cuda", "xpu", None))
    def test_intel_openmp_constraint(self, gpu_arch_type):
        script = (
            Path(__file__).resolve().parent.parent
            / ".ci"
            / "wheel"
            / "windows"
            / "build_install_deps.py"
        )
        specification = importlib.util.spec_from_file_location(
            "windows_wheel_build_dependencies", script
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        dependencies = importlib.util.module_from_spec(specification)
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(sys, "path", sys.path.copy()),
            mock.patch.dict(sys.modules),
        ):
            if gpu_arch_type is None:
                os.environ.pop("GPU_ARCH_TYPE", None)
            else:
                os.environ["GPU_ARCH_TYPE"] = gpu_arch_type
            specification.loader.exec_module(dependencies)
        self.assertIn("mkl-static", dependencies.PIP_PACKAGES)
        self.assertIn("mkl-include", dependencies.PIP_PACKAGES)
        if gpu_arch_type == "cpu":
            self.assertIn("intel-openmp<2026", dependencies.PIP_PACKAGES)
        else:
            self.assertNotIn("intel-openmp<2026", dependencies.PIP_PACKAGES)


instantiate_parametrized_tests(TestWindowsWheelBuild)

if __name__ == "__main__":
    run_tests()
