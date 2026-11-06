"""Pure archive tests with MOCK signatures, not signing or SAC compatibility tests."""

import base64
import copy
import csv
import hashlib
import importlib.util
import io
import stat
import subprocess
import tempfile
import warnings
import zipfile
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


SPEC = importlib.util.spec_from_file_location(
    "wheel_authenticode", Path(__file__).with_name("authenticode.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load the standalone packaging helper")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
PROFILE = "1.3.6.1.4.1.311.97.10.20.30.40"
TARGET = "pkg/target.dll"
VENDOR = "pkg/vendor.dll"
RECORD = "pkg-1.0.dist-info/RECORD"
PE = b"MZ" + bytes(58) + (64).to_bytes(4, "little") + b"PE\0\0fixture"
MOCK_SIGNING_MARKER = b"MOCK_SIGNED_BYTES_NOT_AUTHENTICODE"


def mock_valid(profile=None):
    return {
        "status": "Valid",
        "signature_type": "Authenticode",
        "signer": {"subject": "MOCK signer", "ekus": [profile] if profile else []},
        "timestamp": {"subject": "MOCK timestamp"},
        "signtool_exit_code": 0,
    }


def mock_inspect(path):
    if path.name == "vendor.dll":
        return mock_valid()
    if path.read_bytes().endswith(MOCK_SIGNING_MARKER):
        return mock_valid(PROFILE)
    return {"status": "NotSigned", "signer": None, "timestamp": None, "signtool_exit_code": 1}


class WheelAuthenticodeTests(TestCase):
    def setUp(self):
        super().setUp()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequence = 0

    def fixture(self, extras=(), inspector=mock_inspect, prepare=True):
        self.sequence += 1
        directory = self.root / f"case-{self.sequence}"
        directory.mkdir()
        wheel = directory / "pkg-1.0-cp313-cp313-win_amd64.whl"
        members = [
            (TARGET, PE),
            (VENDOR, PE + b"MOCK_VENDOR_BYTES"),
            ("pkg/data.txt", b"original data"),
            ("pkg-1.0.dist-info/METADATA", b"Name: pkg\nVersion: 1.0\n"),
            ("pkg-1.0.dist-info/WHEEL", b"Wheel-Version: 1.0\n"),
            (RECORD, b"original fixture RECORD\n"),
        ]
        with warnings.catch_warnings(), zipfile.ZipFile(wheel, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            for name, payload in members + list(extras):
                archive.writestr(name, payload)
        review = {
            "wheel_sha256": TOOL.digest_file(wheel)["sha256"],
            "reviewed_by": "MOCK release reviewer",
            "profile_eku": PROFILE,
            "profile_eku_provenance": "MOCK approved mapping; no real service identity",
            "targets": [{"path": TARGET, "owner": "PyTorch", "source_provenance": "https://example.invalid/reviewed-source"}],
        }
        workspace = directory / "prepared"
        receipt = TOOL.prepare(wheel, review, workspace, inspector) if prepare else None
        return directory, wheel, review, workspace, receipt

    def mark_mock_signed(self, workspace):
        with (workspace / "tree" / TARGET).open("ab") as stream:
            stream.write(MOCK_SIGNING_MARKER)

    def test_record_vendor_and_input_bytes(self):
        directory, wheel, review, workspace, receipt = self.fixture()
        original_digest = TOOL.digest_file(wheel)
        self.mark_mock_signed(workspace)
        output = directory / "signed.whl"
        TOOL.repack(workspace, receipt["manifest_sha256"], output, mock_inspect)
        self.assertEqual(TOOL.digest_file(wheel), original_digest)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read(VENDOR), PE + b"MOCK_VENDOR_BYTES")
            rows = list(csv.reader(io.StringIO(archive.read(RECORD).decode("utf-8"))))
            self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
            self.assertEqual(len(rows), len(archive.namelist()))
            for name, encoded, size in rows:
                if name == RECORD:
                    self.assertEqual((encoded, size), ("", ""))
                else:
                    payload = archive.read(name)
                    expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
                    self.assertEqual(encoded, "sha256=" + expected)
                    self.assertEqual(size, str(len(payload)))
                    self.assertNotIn("=", encoded.removeprefix("sha256="))

    @parametrize(
        "variant",
        [
            {"status": "NotSigned"}, {"status": "HashMismatch"},
            {"timestamp": None}, {"signtool_exit_code": 1}, {"signtool_exit_code": 2},
            {"signature_type": "Catalog"},
            {"signer": {"subject": "Expected display name", "ekus": [TOOL.GENERIC_EKU]}},
            {"signer": {"ekus": [TOOL.PROFILE_PREFIX]}},
            {"signer": {"ekus": [PROFILE + ".1"]}},
        ],
    )
    def test_target_signing_and_verification_failures(self, variant):
        directory, wheel, review, workspace, receipt = self.fixture()
        self.mark_mock_signed(workspace)
        signature = {**mock_valid(PROFILE), **variant}

        def inspector(path):
            return copy.deepcopy(signature) if path.name == "target.dll" else mock_inspect(path)

        output = directory / "rejected.whl"
        with self.assertRaises(ValueError):
            TOOL.repack(workspace, receipt["manifest_sha256"], output, inspector)
        self.assertFalse(output.exists())

    @parametrize("action", ["change_vendor", "change_data", "change_record", "add", "delete", "add_directory"])
    def test_non_target_mutations_and_file_set(self, action):
        directory, wheel, review, workspace, receipt = self.fixture()
        self.mark_mock_signed(workspace)
        tree = workspace / "tree"
        if action == "change_vendor":
            (tree / VENDOR).write_bytes(PE + b"changed")
        elif action == "change_data":
            (tree / "pkg/data.txt").write_bytes(b"changed")
        elif action == "change_record":
            (tree / RECORD).write_bytes(b"changed")
        elif action == "add":
            (tree / "pkg/new.txt").write_bytes(b"unexpected")
        elif action == "delete":
            (tree / "pkg/data.txt").unlink()
        else:
            (tree / "unexpected_directory").mkdir()
        with self.assertRaises(ValueError):
            TOOL.repack(workspace, receipt["manifest_sha256"], directory / "out.whl", mock_inspect)

    def test_unsigned_dependencies_and_extra_pe_remain_unresolved(self):
        directory, wheel, review, workspace, receipt = self.fixture(
            extras=[("pkg/unknown.bin", PE), ("pkg/unsigned.dll", PE)]
        )
        self.assertEqual({item["path"] for item in receipt["unresolved"]}, {"pkg/unknown.bin", "pkg/unsigned.dll"})
        self.mark_mock_signed(workspace)
        with self.assertRaises(ValueError):
            TOOL.repack(workspace, receipt["manifest_sha256"], directory / "out.whl", mock_inspect)

    def test_already_signed_target_is_not_overwritten(self):
        directory, wheel, review, workspace, receipt = self.fixture(prepare=False)
        review["targets"][0]["path"] = VENDOR
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)

    @parametrize(
        "field,value",
        [
            ("wheel_sha256", "0" * 64), ("reviewed_by", ""),
            ("profile_eku", TOOL.GENERIC_EKU), ("profile_eku", TOOL.PROFILE_PREFIX),
            ("profile_eku_provenance", ""),
            ("targets", [{"path": TARGET, "owner": "vendor", "source_provenance": "https://example.invalid"}]),
        ],
    )
    def test_review_requires_exact_wheel_ownership_and_full_profile(self, field, value):
        directory, wheel, review, workspace, receipt = self.fixture(prepare=False)
        review[field] = value
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)
        self.assertFalse(workspace.exists())

    @parametrize(
        "name",
        [
            "../escape", "/absolute", "C:/drive", "//host/share", "pkg\\backslash",
            "pkg/file:stream", "pkg/CON.txt", "pkg/con .txt", "pkg/trailing.", "pkg/trailing ",
            "pkg/PROGRA~1", "pkg/\u00e9.txt", "pkg//empty", "pkg/../parent", "pkg/./dot",
            "PKG/new.txt", TARGET, "pkg", "pkg/vendor.dll/child", "pkg/dir//",
        ],
    )
    def test_bad_member_names_and_collisions(self, name):
        directory, wheel, review, workspace, receipt = self.fixture(extras=[(name, b"bad")], prepare=False)
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)
        self.assertFalse(workspace.exists())

    @parametrize("mode", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK])
    def test_links_and_special_files(self, mode):
        member = zipfile.ZipInfo("pkg/link")
        member.create_system = 3
        member.external_attr = (mode | 0o777) << 16
        directory, wheel, review, workspace, receipt = self.fixture(extras=[(member, b"destination")], prepare=False)
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)

    @parametrize("suffix", [".jws", ".p7s"])
    def test_stale_record_signatures(self, suffix):
        directory, wheel, review, workspace, receipt = self.fixture(extras=[(RECORD + suffix, b"stale")], prepare=False)
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)

    def test_manifest_digest_and_no_clobber(self):
        directory, wheel, review, workspace, receipt = self.fixture()
        self.mark_mock_signed(workspace)
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)
        for output in (wheel, workspace / "output.whl"):
            with self.assertRaises(ValueError):
                TOOL.repack(workspace, receipt["manifest_sha256"], output, mock_inspect)
        with self.assertRaises(ValueError):
            TOOL.repack(workspace, "0" * 64, directory / "out.whl", mock_inspect)
        manifest_path = workspace / "manifest.json"
        manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaises(ValueError):
            TOOL.repack(workspace, receipt["manifest_sha256"], directory / "out.whl", mock_inspect)

    @parametrize("name,mode", [("pkg/ambiguous/", stat.S_IFREG), ("pkg/ambiguous", stat.S_IFDIR)])
    def test_inconsistent_directory_member_modes(self, name, mode):
        member = zipfile.ZipInfo(name)
        member.create_system = 3
        member.external_attr = (mode | 0o755) << 16
        directory, wheel, review, workspace, receipt = self.fixture(
            extras=[(member, b"")], prepare=False
        )
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)
        self.assertFalse(workspace.exists())

    @parametrize("attributes", [0x10, 0x400])
    def test_dos_directory_and_reparse_attributes(self, attributes):
        member = zipfile.ZipInfo("pkg/ambiguous")
        member.external_attr = attributes
        directory, wheel, review, workspace, receipt = self.fixture(
            extras=[(member, b"")], prepare=False
        )
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, workspace, mock_inspect)

    def test_zip64_size_and_count_boundaries(self):
        directory, wheel, review, workspace, receipt = self.fixture()
        self.mark_mock_signed(workspace)
        output = directory / "zip64.whl"
        with mock.patch.object(zipfile, "ZIP64_LIMIT", 64), mock.patch.object(zipfile, "ZIP_FILECOUNT_LIMIT", 2):
            TOOL.repack(workspace, receipt["manifest_sha256"], output, mock_inspect)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue(any(member.extract_version >= 45 for member in archive.infolist()))
        self.assertIn(b"PK\x06\x06", output.read_bytes())

    @parametrize("failure", [FileNotFoundError("MOCK missing PowerShell"), subprocess.CalledProcessError(1, "MOCK verifier")])
    def test_verifier_command_errors_are_not_swallowed(self, failure):
        with mock.patch.object(TOOL.sys, "platform", "win32"), mock.patch.object(TOOL.subprocess, "run", side_effect=failure):
            with self.assertRaises(type(failure)):
                TOOL.inspect_windows(self.root / "native.dll", self.root / "signtool.exe")

    def test_explicit_directories_survive_without_record_rows(self):
        directory, wheel, review, workspace, receipt = self.fixture(
            extras=[("pkg/empty/", b"")]
        )
        self.mark_mock_signed(workspace)
        output = directory / "directories.whl"
        TOOL.repack(workspace, receipt["manifest_sha256"], output, mock_inspect)
        with zipfile.ZipFile(output) as archive:
            self.assertTrue(archive.getinfo("pkg/empty/").is_dir())
            rows = list(csv.reader(io.StringIO(archive.read(RECORD).decode("utf-8"))))
            self.assertNotIn("pkg/empty/", {row[0] for row in rows})

    def test_symlinked_workspace_ancestor_is_rejected(self):
        directory, wheel, review, workspace, receipt = self.fixture(prepare=False)
        alias = directory / "alias"
        try:
            alias.symlink_to(directory, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Creating a task-local symlink is unavailable")
        with self.assertRaises(ValueError):
            TOOL.prepare(wheel, review, alias / "prepared", mock_inspect)

    def test_changed_streamed_bytes_remove_partial_output(self):
        directory, wheel, review, workspace, receipt = self.fixture()
        self.mark_mock_signed(workspace)
        original_zip_info = TOOL.zip_info

        def mutating_zip_info(name, original):
            with (workspace / "tree" / name).open("ab") as stream:
                stream.write(b"MOCK concurrent write during repack")
            return original_zip_info(name, original)

        output = directory / "partial.whl"
        with mock.patch.object(TOOL, "zip_info", side_effect=mutating_zip_info):
            with self.assertRaises(ValueError):
                TOOL.repack(workspace, receipt["manifest_sha256"], output, mock_inspect)
        self.assertFalse(output.exists())

    def test_signature_inspection_cannot_mutate_verified_bytes(self):
        directory, wheel, review, workspace, receipt = self.fixture()
        self.mark_mock_signed(workspace)

        def mutating_inspector(path):
            signature = mock_inspect(path)
            with path.open("ab") as stream:
                stream.write(b"unexpected concurrent edit")
            return signature

        with self.assertRaises(ValueError):
            TOOL.repack(workspace, receipt["manifest_sha256"], directory / "out.whl", mutating_inspector)


instantiate_parametrized_tests(WheelAuthenticodeTests)


if __name__ == "__main__":
    run_tests()
