# Owner(s): ["module: dynamo"]

import os
import subprocess
import sys
import textwrap
import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_NUMPY,
    TestCase,
)


class TestNumpyImports(TestCase):
    def _run_script(self, script, *args):
        # Use the same torch package as the parent, including in source builds.
        torch_root = os.path.dirname(os.path.dirname(torch.__file__))
        script = f"import sys\nsys.path.insert(0, {torch_root!r})\n" + textwrap.dedent(
            script
        )
        result = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", script, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    @parametrize("module", ["torch._numpy", "numpy.fft", "numpy.random"])
    @parametrize("error_type", ["ImportError", "ModuleNotFoundError"])
    def test_import_error_is_preserved(self, module, error_type):
        if module.startswith("numpy.") and not TEST_NUMPY:
            self.skipTest("NumPy is not available")
        self._run_script(
            """
            import builtins
            import importlib
            import types
            import unittest

            import torch

            check = unittest.TestCase()
            module, error_type = sys.argv[1:]
            error = getattr(builtins, error_type)(
                "injected dependency import failure", name="injected_dependency"
            )
            cause = OSError("injected loader failure")
            failure_sites = []
            if module == "torch._numpy":
                original_import = builtins.__import__

                def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
                    if (
                        name == module
                        and globals is not None
                        and globals.get("__name__") == "torch._dynamo.utils"
                    ):
                        failure_sites.append(globals["__name__"])
                        raise error from cause
                    return original_import(name, globals, locals, fromlist, level)

                builtins.__import__ = failing_import
            else:
                import numpy as np

                # Exercise an already loaded submodule, independent of lazy imports.
                dependency = importlib.import_module(module)
                attribute = module.rsplit(".", 1)[1]
                check.assertIs(sys.modules[module], dependency)
                check.assertIs(getattr(np, attribute), dependency)

                class BrokenNumpy(types.ModuleType):
                    def __getattribute__(self, name):
                        caller = sys._getframe(1).f_globals.get("__name__")
                        if name == attribute and caller == "torch._dynamo.utils":
                            failure_sites.append(caller)
                            raise error from cause
                        return super().__getattribute__(name)

                np.__class__ = BrokenNumpy

            with check.assertRaises(ImportError) as caught:
                import torch._dynamo

            check.assertEqual(failure_sites, ["torch._dynamo.utils"])
            check.assertIs(caught.exception, error)
            check.assertIs(caught.exception.__cause__, cause)
            check.assertNotIn("torch._dynamo.utils", sys.modules)
            """,
            module,
            error_type,
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy is not available")
    def test_numpy_module_registry(self):
        self._run_script(
            """
            import unittest

            import numpy as np
            import torch
            import torch._dynamo
            import torch._numpy as tnp
            from torch._dynamo import utils

            check = unittest.TestCase()
            check.assertEqual(
                utils.NP_SUPPORTED_MODULES, (np, np.fft, np.linalg, np.random)
            )
            check.assertEqual(
                utils.NP_TO_TNP_MODULE,
                {np: tnp, np.fft: tnp.fft, np.linalg: tnp.linalg, np.random: tnp.random},
            )
            check.assertEqual(tuple(utils.NP_TO_TNP_MODULE), utils.NP_SUPPORTED_MODULES)
            fn = lambda x: x.sin() + 1
            x = torch.arange(4, dtype=torch.float32)
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            torch.testing.assert_close(compiled(x), fn(x))
            """
        )

    def test_without_numpy(self):
        self._run_script(
            """
            import unittest

            sys.modules["numpy"] = None
            import torch
            import torch._dynamo
            from torch._dynamo import utils

            check = unittest.TestCase()
            check.assertIsNone(utils.np)
            check.assertEqual(utils.NP_SUPPORTED_MODULES, ())
            check.assertEqual(utils.NP_TO_TNP_MODULE, {})
            fn = lambda x: x.sin() + 1
            x = torch.arange(4, dtype=torch.float32)
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            torch.testing.assert_close(compiled(x), fn(x))
            """
        )


instantiate_parametrized_tests(TestNumpyImports)


if __name__ == "__main__":
    run_tests()
