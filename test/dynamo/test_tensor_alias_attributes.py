# Owner(s): ["module: dynamo"]

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


class TensorAliasAttributeTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.compile_evidence = []

    def _compile(self, function, backend, fullgraph=True):
        counter = CompileCounterWithBackend(backend)
        executions = []

        def recording_backend(graph, inputs):
            compiled = counter(graph, inputs)

            def execute(*values):
                result = compiled(*values)
                executions.append(True)
                return result

            return execute

        self.compile_evidence.append((counter, executions))
        return torch.compile(function, backend=recording_backend, fullgraph=fullgraph)

    def _assert_captured(self, calls=1):
        counter, executions = self.compile_evidence[-1]
        self.assertGreater(counter.frame_count, 0)
        self.assertGreaterEqual(len(executions), calls)

    @parametrize("backend", ("eager", "aot_eager"))
    def test_missing_attribute_caught(self, device, backend):
        def function(value):
            alias = value.to(value.dtype)
            try:
                result = alias.foo
            except AttributeError:
                return alias + 1, alias
            return result + alias, alias

        value = torch.arange(4.0, device=device)
        expected = function(value)
        actual = self._compile(function, backend)(value)
        self.assertEqual(actual, expected)
        self.assertIs(actual[1], value)
        self._assert_captured()

    @parametrize("backend", ("eager", "aot_eager"))
    @parametrize("attribute_is_tensor", (False, True))
    def test_present_direct_attribute(self, device, backend, attribute_is_tensor):
        def function(value):
            alias = value.to(value.dtype)
            return alias.foo, alias, alias + 1

        value = torch.arange(4.0, device=device)
        value.foo = value + 7 if attribute_is_tensor else 37
        expected = function(value)
        actual = self._compile(function, backend)(value)
        self.assertEqual(actual, expected)
        self.assertIs(actual[0], value.foo)
        self.assertIs(actual[1], value)
        self._assert_captured()

    @parametrize("backend", ("eager", "aot_eager"))
    def test_attribute_guards_same_callable(self, device, backend):
        def function(value):
            alias = value.to(value.dtype)
            return getattr(alias, "foo", alias + 1), hasattr(alias, "foo"), alias

        value = torch.arange(4.0, device=device)
        compiled = self._compile(function, backend)
        for attribute in (None, value + 2, value + 3, None):
            if attribute is None:
                if hasattr(value, "foo"):
                    del value.foo
            else:
                value.foo = attribute
            expected = function(value)
            actual = compiled(value)
            self.assertEqual(actual, expected)
            self.assertIs(actual[2], value)
            if attribute is not None:
                self.assertIs(actual[0], attribute)
        self._assert_captured(4)
        self.assertGreaterEqual(self.compile_evidence[-1][0].frame_count, 2)

    @parametrize("backend", ("eager", "aot_eager"))
    @parametrize("initially_present", (False, True))
    def test_pending_attribute_write(self, device, backend, initially_present):
        def function(value, attribute):
            value.foo = attribute
            alias = value.to(value.dtype)
            return alias.foo, alias, alias + 1

        value = torch.arange(4.0, device=device)
        reference = value.clone()
        attribute = value + 3
        if initially_present:
            value.foo = value + 2
            reference.foo = reference + 2
        expected = function(reference, attribute)
        actual = self._compile(function, backend)(value, attribute)
        self.assertEqual(actual, expected)
        self.assertIs(actual[0], attribute)
        self.assertIs(actual[1], value)
        self.assertIs(value.foo, attribute)
        self._assert_captured()

    @parametrize("backend", ("eager", "aot_eager"))
    def test_aliased_then_distinct_inputs(self, device, backend):
        def function(left, right, attribute):
            right.foo = attribute
            alias = left.to(left.dtype)
            return alias.foo, right.foo, left + right, alias

        compiled = self._compile(function, backend)
        for same_object in (True, False, False, True):
            left = torch.arange(4.0, device=device)
            left.foo = left + 2
            right = left if same_object else left.clone()
            attribute = left + 3
            expected_attribute = attribute if same_object else left.foo
            actual = compiled(left, right, attribute)
            self.assertIs(actual[0], expected_attribute)
            self.assertIs(actual[1], attribute)
            self.assertEqual(actual[2], left + right)
            self.assertIs(actual[3], left)
            self.assertIs(right.foo, attribute)
        self._assert_captured(4)
        self.assertGreaterEqual(self.compile_evidence[-1][0].frame_count, 2)

    @parametrize("backend", ("eager", "aot_eager"))
    @parametrize("operation", ("contiguous", "inplace"))
    def test_other_identity_aliases(self, device, backend, operation):
        def function(value):
            alias = value.contiguous() if operation == "contiguous" else value.add_(1)
            return alias.foo, alias, alias + 1

        value = torch.arange(4.0, device=device)
        reference = value.clone()
        attribute = value + 2
        value.foo = reference.foo = attribute
        expected = function(reference)
        actual = self._compile(function, backend)(value)
        self.assertEqual(actual, expected)
        self.assertEqual(value, reference)
        self.assertIs(actual[0], attribute)
        self.assertIs(actual[1], value)
        self._assert_captured()

    @parametrize("backend", ("eager", "aot_eager"))
    @parametrize("operation", ("contiguous", "dtype"))
    def test_nonalias_object_controls(self, device, backend, operation):
        def function(value):
            result = (
                value.contiguous()
                if operation == "contiguous"
                else value.to(torch.float64)
            )
            return getattr(result, "foo", result + 1), hasattr(result, "foo"), result

        value = torch.arange(6.0, device=device).reshape(2, 3).transpose(0, 1)
        value.foo = value + 2
        expected = function(value)
        actual = self._compile(function, backend)(value)
        self.assertEqual(actual, expected)
        self.assertIsNot(actual[2], value)
        self.assertFalse(actual[1])
        self.assertEqual(actual[2].dtype, expected[2].dtype)
        self.assertNotEqual(actual[2].data_ptr(), value.data_ptr())
        self._assert_captured()

    @parametrize("backend", ("eager", "aot_eager"))
    def test_caller_graph_break_control(self, device, backend):
        def function(value):
            alias = value.to(value.dtype)
            torch._dynamo.graph_break()
            return getattr(alias, "foo", alias + 1), hasattr(alias, "foo"), alias + 2

        value = torch.arange(4.0, device=device)
        value.foo = value + 3
        expected = function(value)
        actual = self._compile(function, backend, fullgraph=False)(value)
        self.assertEqual(actual, expected)
        self.assertIs(actual[0], value.foo)
        self._assert_captured()


instantiate_device_type_tests(TensorAliasAttributeTests, globals())


if __name__ == "__main__":
    run_tests()
