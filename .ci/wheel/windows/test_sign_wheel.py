#!/usr/bin/env python3
"""Credential-free packaging tests with mocks and a read-only Windows adapter check.

These bootstrap CI tests use only the standard library and do not import torch.
No test creates a certificate, calls a signing service, or verifies SAC.
"""

import base64
import csv
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import build_wheel
import sign_wheel as signing


CONTEXT = {
    "repository": "pytorch/pytorch",
    "sha": "a" * 40,
    "run_id": "123",
    "run_attempt": "1",
}
PUBLISHER = "b" * 64
VENDOR = "c" * 64
EXTERNAL = "torch/lib/vendor.dll"
FILENAME = "torch-2.99.0+cu999-cp313-cp313-win_amd64.whl"
RECORD = "torch-2.99.0.dist-info/RECORD"
POLICY = {
    "schema_version": 1,
    "publisher_sha256": [PUBLISHER],
    "third_party": {EXTERNAL: [VENDOR]},
}


def mock_signature(path, status="Valid", pin=PUBLISHER, timestamped=True):
    return {
        "path": path,
        "status": status,
        "signature_type": "Authenticode",
        "signer_sha256": pin if status == "Valid" else None,
        "timestamped": timestamped,
        "subject": "MOCK certificate; not Authenticode",
    }


class MockAuthenticode:
    def __init__(self):
        self.calls = []
        self.overrides = {}

    def sign(self, path):
        self.calls.append(path.name)
        path.write_bytes(path.read_bytes() + b"\nMOCK-SIGNATURE")

    def verify(self, root, names):
        result = {}
        for name in names:
            signed = (root / name).read_bytes().endswith(b"MOCK-SIGNATURE")
            result[name] = self.overrides.get(
                name,
                mock_signature(name, pin=VENDOR)
                if name == EXTERNAL
                else mock_signature(name, status="Valid" if signed else "NotSigned"),
            )
        return result


class WheelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mock-signing-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheel = self.root / FILENAME
        self.output = self.root / "final"
        self.mock = MockAuthenticode()

    def fixture(
        self, backend="cuda", pyd="_C.cp313-win_amd64.pyd", omit=(), extra=None
    ):
        names = {f"torch/lib/{n}" for n in signing.CPU_DLLS}
        if backend == "cuda":
            names |= {f"torch/lib/{n}" for n in signing.CUDA_DLLS}
        names |= {f"torch/{pyd}", EXTERNAL}
        payload = {
            name: b"MZ-mock-native:" + name.encode() for name in names - set(omit)
        }
        payload.update(
            {
                "torch/__init__.py": b"# mock fixture; not a PyTorch build\n",
                "torch-2.99.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp313-cp313-win_amd64\n",
                "torch-2.99.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: torch\nVersion: 2.99.0\n",
            }
        )
        payload.update(extra or {})
        self.write_wheel(payload)
        return payload

    def write_wheel(self, payload):
        rows = []
        with zipfile.ZipFile(self.wheel, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, data in payload.items():
                z.writestr(name, data)
                sha256, size = signing.digest(io.BytesIO(data))
                encoded = (
                    base64.urlsafe_b64encode(bytes.fromhex(sha256))
                    .rstrip(b"=")
                    .decode()
                )
                rows.append([name, f"sha256={encoded}", str(size)])
            rows.append([RECORD, "", ""])
            text = io.StringIO(newline="")
            csv.writer(text, lineterminator="\n").writerows(rows)
            z.writestr(RECORD, text.getvalue())

    def sign(self, backend="cuda", policy=None, callback=None):
        return signing.sign_wheel(
            self.wheel,
            self.output,
            CONTEXT,
            POLICY if policy is None else policy,
            backend,
            callback or self.mock.sign,
            verify=self.mock.verify,
        )

    def assert_no_published_wheel(self):
        self.assertFalse(self.output.exists() and list(self.output.glob("*.whl")))

    def test_cuda_nine_dlls_and_pyd_all_records_and_preservation(self):
        original = self.fixture()
        before = signing.file_identity(self.wheel)
        report = self.sign()
        self.assertEqual(len(self.mock.calls), 10)
        self.assertTrue(report["all_native_valid"])
        self.assertEqual(report["mode"], "signed")
        final = signing.check_handoff(
            self.output, CONTEXT, True, True, self.mock.verify
        )
        self.assertNotEqual(signing.file_identity(final), before)
        self.assertEqual(signing.file_identity(self.wheel), before)
        with zipfile.ZipFile(final) as z:
            for name, data in original.items():
                if signing.owner(name) != "pytorch":
                    self.assertEqual(z.read(name), data)
            record = z.read(RECORD).decode()
        self.assertIn("torch/__init__.py,sha256=", record)
        self.assertIn("_C.cp313-win_amd64.pyd,sha256=", record)
        self.assertEqual(len(report["native"]), 11)

    def test_cpu_wheels_do_not_need_cuda_and_pyd_abi_is_not_hardcoded(self):
        for pyd in ("_C.pyd", "_C.cp312-win_amd64.pyd", "_C.cp313t-win_amd64.pyd"):
            with self.subTest(pyd=pyd):
                self.fixture("cpu", pyd)
                self.sign("cpu")
                self.assertIn(pyd, self.mock.calls)
                shutil.rmtree(self.output)
                self.mock.calls.clear()

    def test_already_valid_first_party_is_preserved(self):
        payload = self.fixture()
        payload["torch/lib/c10.dll"] += b"\nMOCK-SIGNATURE"
        self.write_wheel(payload)
        report = self.sign()
        self.assertNotIn("c10.dll", self.mock.calls)
        item = next(n for n in report["native"] if n["path"] == "torch/lib/c10.dll")
        self.assertEqual(item["action"], "preserved")
        self.assertEqual(item["input_sha256"], item["sha256"])

    def test_unconfigured_unsigned_is_explicit_and_never_accepted_as_signed(self):
        self.fixture()
        report = signing.publish_unsigned(self.wheel, self.output, CONTEXT)
        self.assertEqual(report["signing_status"], "not-configured")
        self.assertFalse(report["all_native_valid"])
        signing.check_handoff(self.output, CONTEXT)
        with self.assertRaisesRegex(signing.WheelError, "Required signing is missing"):
            signing.check_handoff(self.output, CONTEXT, True)

    def test_missing_policy_fails_before_any_signing(self):
        self.fixture()
        with self.assertRaisesRegex(signing.WheelError, "policy schema"):
            self.sign(policy={})
        self.assertEqual(self.mock.calls, [])
        self.assert_no_published_wheel()

    def test_missing_pyd_and_cuda_targets_are_rejected(self):
        for missing in ("torch/_C.cp313-win_amd64.pyd", "torch/lib/caffe2_nvrtc.dll"):
            with self.subTest(missing=missing):
                self.fixture(omit=[missing])
                with self.assertRaisesRegex(signing.WheelError, "Missing required"):
                    self.sign()
                self.assert_no_published_wheel()

    def test_extra_pyd_is_scanned_and_cannot_be_silently_supported(self):
        self.fixture(extra={"torch/plugins/new.pyd": b"MZ-mock"})
        with self.assertRaisesRegex(signing.WheelError, "Unclassified"):
            self.sign()
        self.assertEqual(self.mock.calls, [])

    def test_unknown_valid_dll_is_not_implicitly_approved(self):
        unknown = "torch/lib/new.dll"
        self.fixture(extra={unknown: b"MZ-mock"})
        self.mock.overrides[unknown] = mock_signature(unknown, pin=VENDOR)
        with self.assertRaisesRegex(signing.WheelError, "Unclassified"):
            self.sign()
        self.assert_no_published_wheel()

    def test_unsigned_third_party_is_rejected_without_overwriting_vendor_file(self):
        self.fixture()
        self.mock.overrides[EXTERNAL] = mock_signature(EXTERNAL, "NotSigned")
        with self.assertRaisesRegex(signing.WheelError, "third-party"):
            self.sign()
        self.assertEqual(self.mock.calls, [])
        self.assert_no_published_wheel()

    def test_invalid_or_unexpected_existing_signatures_are_rejected(self):
        self.fixture()
        name = "torch/lib/c10.dll"
        for value in (
            mock_signature(name, "HashMismatch"),
            mock_signature(name, pin=VENDOR),
            mock_signature(name, timestamped=False),
            {**mock_signature(name), "signature_type": "Catalog"},
        ):
            with self.subTest(signature=value):
                self.mock.overrides[name] = value
                with self.assertRaisesRegex(
                    signing.WheelError, "Unexpected or invalid"
                ):
                    self.sign()
                self.assertEqual(self.mock.calls, [])
                self.assert_no_published_wheel()

    def test_zero_exit_signer_without_valid_signature_fails_closed(self):
        self.fixture()
        with self.assertRaisesRegex(
            signing.WheelError, "Required signature verification failed"
        ):
            self.sign(callback=lambda path: None)
        self.assert_no_published_wheel()

    def test_signing_failure_never_publishes_partial_or_unsigned_output(self):
        self.fixture()
        original = signing.file_identity(self.wheel)

        def fail_after_one(path):
            self.mock.sign(path)
            raise subprocess.CalledProcessError(1, ["MOCK signer"])

        with self.assertRaises(subprocess.CalledProcessError):
            self.sign(callback=fail_after_one)
        self.assertEqual(signing.file_identity(self.wheel), original)
        self.assert_no_published_wheel()
        self.assertEqual(list(self.root.glob("wheel-sign-*")), [])

    def test_unrelated_file_change_is_rejected(self):
        self.fixture()

        def damage_vendor(path):
            self.mock.sign(path)
            native_root = (
                path.parents[2] if path.parent.name == "lib" else path.parents[1]
            )
            (native_root / EXTERNAL).write_bytes(b"modified vendor")

        with self.assertRaisesRegex(signing.WheelError, "unrelated file"):
            self.sign(callback=damage_vendor)
        self.assert_no_published_wheel()

    def test_invalid_record_is_not_silently_repaired(self):
        self.fixture()
        with zipfile.ZipFile(self.wheel) as z:
            payload = {name: z.read(name) for name in z.namelist()}
        payload["torch/__init__.py"] += b"tampered"
        with zipfile.ZipFile(self.wheel, "w") as z:
            for name, data in payload.items():
                z.writestr(name, data)
        with self.assertRaisesRegex(signing.WheelError, "RECORD mismatch"):
            self.sign()
        self.assertEqual(self.mock.calls, [])
        self.assert_no_published_wheel()

    def test_unsafe_archive_paths_links_duplicates_and_collisions_are_rejected(self):
        cases = [
            "../escape.dll",
            "torch/../escape.dll",
            "C:/escape.dll",
            "torch\\escape.dll",
            "torch/file.dll:ads",
            "torch/NUL.dll",
            "torch/trailing. /file.dll",
            "/absolute.dll",
        ]
        for name in cases:
            with self.subTest(path=name):
                self.fixture(extra={name: b"bad"})
                with self.assertRaises(signing.WheelError):
                    signing.inspect_wheel(self.wheel)
        for attack in ("symlink", "duplicate", "case-collision", "parent-collision"):
            with self.subTest(attack=attack):
                self.fixture()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(self.wheel, "a") as z:
                        if attack == "symlink":
                            item = zipfile.ZipInfo("torch/link.dll")
                            item.create_system = 3
                            item.external_attr = (stat.S_IFLNK | 0o777) << 16
                            z.writestr(item, "../../escape")
                        elif attack == "duplicate":
                            z.writestr(EXTERNAL, b"duplicate")
                        elif attack == "case-collision":
                            z.writestr(EXTERNAL.upper(), b"duplicate")
                        else:
                            z.writestr("torch/lib", b"parent file")
                with self.assertRaises(signing.WheelError):
                    signing.inspect_wheel(self.wheel)

    def test_handoff_rejects_stale_run_unsigned_or_altered_artifacts(self):
        self.fixture()
        self.sign()
        with self.assertRaisesRegex(signing.WheelError, "different source/run/attempt"):
            signing.check_handoff(self.output, {**CONTEXT, "run_attempt": "2"}, True)
        (self.output / "old.whl").write_bytes(b"stale")
        with self.assertRaisesRegex(signing.WheelError, "Unexpected"):
            signing.check_handoff(self.output, CONTEXT, True)
        (self.output / "old.whl").unlink()
        receipt = self.output / signing.RECEIPT
        report = json.loads(receipt.read_text(encoding="utf-8"))
        report["wheel"]["sha256"] = "0" * 64
        receipt.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(signing.WheelError, "digest mismatch"):
            signing.check_handoff(self.output, CONTEXT, True)

    def test_stale_output_is_refused_before_build_or_sign(self):
        self.fixture()
        self.output.mkdir()
        (self.output / "old.whl").write_bytes(b"old")
        with patch.object(build_wheel.subprocess, "run") as build:
            with self.assertRaisesRegex(signing.WheelError, "nonempty"):
                build_wheel.build_wheel(self.output)
            build.assert_not_called()
        with self.assertRaisesRegex(signing.WheelError, "nonempty"):
            self.sign()
        self.assertEqual(self.mock.calls, [])
        self.assertEqual((self.output / "old.whl").read_bytes(), b"old")

    def test_failed_build_with_partial_wheel_never_reaches_final_directory(self):
        def partial_build(command, **kwargs):
            staging = Path(command[command.index("--outdir") + 1])
            (staging / FILENAME).write_bytes(b"partial build")
            raise subprocess.CalledProcessError(1, command)

        with patch.object(build_wheel.subprocess, "run", side_effect=partial_build):
            with self.assertRaises(subprocess.CalledProcessError):
                build_wheel.build_wheel(self.output)
        self.assert_no_published_wheel()
        self.assertEqual(list(self.root.glob("wheel-build-*")), [])

    def test_successful_build_hands_off_exact_wheel_before_optional_signing(self):
        self.fixture("cpu")

        def mock_build(command, **kwargs):
            shutil.copyfile(self.wheel, Path(command[-1]) / self.wheel.name)

        with patch.object(build_wheel.subprocess, "run", side_effect=mock_build):
            with patch.object(build_wheel, "context_from_env", return_value=CONTEXT):
                build_wheel.build_wheel(self.output)
        final = signing.check_handoff(self.output, CONTEXT)
        self.assertEqual(
            signing.file_identity(final), signing.file_identity(self.wheel)
        )

    def test_failed_required_signing_cannot_fall_back_to_unsigned_handoff(self):
        self.fixture()
        unsigned = self.root / "unsigned-build"
        signing.publish_unsigned(self.wheel, unsigned, CONTEXT)
        source = signing.check_handoff(unsigned, CONTEXT)
        before = signing.file_identity(source)

        def failed_signer(path):
            self.mock.sign(path)
            raise subprocess.CalledProcessError(1, ["MOCK signer"])

        with self.assertRaises(subprocess.CalledProcessError):
            signing.sign_wheel(
                source,
                self.output,
                CONTEXT,
                POLICY,
                "cuda",
                failed_signer,
                verify=self.mock.verify,
            )
        self.assert_no_published_wheel()
        self.assertEqual(signing.file_identity(source), before)
        with self.assertRaisesRegex(signing.WheelError, "Required signing is missing"):
            signing.check_handoff(unsigned, CONTEXT, require_signed=True)
        with self.assertRaisesRegex(signing.WheelError, "Missing or invalid"):
            signing.check_handoff(self.output, CONTEXT, require_signed=True)

    def test_downloaded_signature_failure_rejects_signed_handoff(self):
        self.fixture()
        self.sign()
        name = "torch/lib/c10.dll"
        self.mock.overrides[name] = mock_signature(name, "HashMismatch")
        with self.assertRaisesRegex(signing.WheelError, "Downloaded Authenticode"):
            signing.check_handoff(self.output, CONTEXT, True, True, self.mock.verify)

    def test_release_context_denies_pr_fork_unprotected_and_ciflow(self):
        env = {
            "GITHUB_REPOSITORY": "pytorch/pytorch",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/nightly",
            "GITHUB_REF_PROTECTED": "true",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(signing.require_release_context(), CONTEXT)
        for field, value in (
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_EVENT_NAME", "pull_request_target"),
            ("GITHUB_REPOSITORY", "fork/pytorch"),
            ("GITHUB_REF_PROTECTED", "false"),
            ("GITHUB_REF", "refs/tags/ciflow/binaries/123"),
            ("GITHUB_REF", "refs/heads/main"),
        ):
            with self.subTest(field=field, value=value):
                with patch.dict(os.environ, {**env, field: value}, clear=True):
                    with self.assertRaises(signing.WheelError):
                        signing.require_release_context()

    def test_cli_missing_configuration_fails_without_a_final_artifact(self):
        self.fixture()
        unsigned = self.root / "unsigned"
        signing.publish_unsigned(self.wheel, unsigned, CONTEXT)
        env = {
            **os.environ,
            "GITHUB_REPOSITORY": CONTEXT["repository"],
            "GITHUB_SHA": CONTEXT["sha"],
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_REF_PROTECTED": "true",
            "GITHUB_REF": "refs/heads/nightly",
            "GITHUB_EVENT_NAME": "push",
        }
        env.pop("WINDOWS_WHEEL_SIGNING_POLICY", None)
        env.pop("WINDOWS_WHEEL_SIGNING_COMMAND", None)
        result = subprocess.run(
            [
                sys.executable,
                str(signing.HERE / "sign_wheel.py"),
                "sign",
                str(unsigned),
                str(self.output),
                "--backend",
                "cuda",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicit reviewed signing policy", result.stderr)
        self.assert_no_published_wheel()


@unittest.skipUnless(
    os.name == "nt", "Real read-only Authenticode adapter requires Windows"
)
class RealAdapterTests(unittest.TestCase):
    def test_existing_signed_executable_and_unsigned_file(self):
        # Use the Windows-supplied executable; never create a certificate or sign.
        powershell = shutil.which("powershell.exe")
        self.assertIsNotNone(powershell)
        with tempfile.TemporaryDirectory(prefix="real-authenticode-test-") as temp:
            root = Path(temp)
            signed = root / "existing signed.exe"
            unsigned = root / "unsigned fixture.ps1"
            shutil.copyfile(powershell, signed)
            unsigned.write_text("# Unsigned test fixture.\n", encoding="utf-8")
            before = {p.name: signing.file_identity(p) for p in (signed, unsigned)}
            rows = signing.verify_native(root, [signed.name, unsigned.name])
            self.assertEqual(set(rows), {signed.name, unsigned.name})
            self.assertEqual(rows[signed.name]["status"], "Valid")
            self.assertRegex(rows[signed.name]["signer_sha256"], r"^[a-f0-9]{64}$")
            self.assertEqual(rows[unsigned.name]["status"], "NotSigned")
            self.assertIsNone(rows[unsigned.name]["signer_sha256"])
            self.assertIsNone(rows[unsigned.name]["subject"])
            self.assertFalse(rows[unsigned.name]["timestamped"])
            self.assertEqual(
                signing.verify_native(root, [unsigned.name]),
                {unsigned.name: rows[unsigned.name]},
            )
            self.assertEqual(signing.verify_native(root, []), {})
            self.assertEqual(
                before, {p.name: signing.file_identity(p) for p in (signed, unsigned)}
            )


if __name__ == "__main__":
    unittest.main()
