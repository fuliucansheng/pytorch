# Owner(s): ["oncall: cpu inductor"]
import math
import unittest
from unittest.mock import patch

import torch
from torch._inductor import config, cpu_vec_isa
from torch._inductor.autotune_process import CppBenchmarkRequest
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests


class TestCppBmmTemplate(TestCase):
    @unittest.skipUnless(
        cpu_vec_isa.valid_vec_isa_list(), "Requires a supported CPU vector ISA"
    )
    @config.patch(
        max_autotune_gemm_backends="CPP,ATEN",
        autotune_in_subproc=False,
        force_disable_caches=True,
    )
    @parametrize("op", ("bmm", "matmul"))
    @parametrize("mode", ("max-autotune", "max-autotune-no-cudagraphs"))
    def test_bmm_template_export(self, device, op, mode):
        a = torch.randn(4, 8, 8, device=device)
        b = torch.randn(4, 8, 8, device=device)
        fn = getattr(torch, op)
        timings = []
        original_benchmark = CppBenchmarkRequest.benchmark

        def benchmark(request, *args, **kwargs):
            timing = original_benchmark(request, *args, **kwargs)
            timings.append(timing)
            return timing

        with patch.object(CppBenchmarkRequest, "benchmark", benchmark):
            actual = torch.compile(fn, mode=mode, fullgraph=True)(a, b)

        self.assertEqual(actual, fn(a, b))
        self.assertTrue(timings, "The C++ BMM template must be benchmarked")
        for timing in timings:
            self.assertTrue(math.isfinite(timing))


instantiate_device_type_tests(TestCppBmmTemplate, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
