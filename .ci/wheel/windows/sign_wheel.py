#!/usr/bin/env python3
"""Finalize Windows wheels without exposing a signing identity to build jobs."""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unicodedata
import zipfile
from pathlib import Path


CPU_DLLS = {
    "c10.dll",
    "shm.dll",
    "torch.dll",
    "torch_cpu.dll",
    "torch_global_deps.dll",
    "torch_python.dll",
}
CUDA_DLLS = {"c10_cuda.dll", "caffe2_nvrtc.dll", "torch_cuda.dll"}
FIRST_PARTY = {f"torch/lib/{name}" for name in CPU_DLLS | CUDA_DLLS}
CORE_PYD = re.compile(r"torch/_C(?:\.[A-Za-z0-9_-]+)?\.pyd\Z")
NATIVE_SUFFIXES = {".dll", ".pyd", ".exe"}
RECEIPT = "wheel-handoff.json"
MAX_EXPANDED_BYTES = 16 * 1024**3
MAX_MEMBERS = 100000
HERE = Path(__file__).resolve().parent


class WheelError(ValueError):
    pass


def digest(stream):
    h = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
        size += len(block)
    return h.hexdigest(), size


def file_identity(path):
    with path.open("rb") as stream:
        sha256, size = digest(stream)
    return {"sha256": sha256, "bytes": size}


def safe_name(name):
    parts = name.rstrip("/").split("/")
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1, 10)
    }
    if (
        not name
        or "\\" in name
        or unicodedata.normalize("NFC", name) != name
        or any(
            part in ("", ".", "..")
            or part.endswith((".", " "))
            or part.split(".")[0].casefold() in reserved
            or any(ord(c) < 32 or c in ':<>"|?*' for c in part)
            for part in parts
        )
    ):
        raise WheelError(f"Unsafe archive path: {name!r}")
    return name.rstrip("/").casefold()


def wheel_members(archive):
    entries = archive.infolist()
    if (
        len(entries) > MAX_MEMBERS
        or sum(i.file_size for i in entries) > MAX_EXPANDED_BYTES
    ):
        raise WheelError("Wheel exceeds the member or expanded-size limit")
    seen = {}
    files = {}
    for entry in entries:
        name = entry.filename
        key = safe_name(name)
        kind = stat.S_IFMT(entry.external_attr >> 16)
        if (
            name != entry.orig_filename
            or key in seen
            or entry.flag_bits & 1
            or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
            or (kind == stat.S_IFDIR and not entry.is_dir())
        ):
            raise WheelError(f"Duplicate, encrypted or special archive entry: {name!r}")
        seen[key] = entry.is_dir()
        if not entry.is_dir():
            files[name] = entry
    for key in seen:
        parents = key.split("/")[:-1]
        for i in range(1, len(parents) + 1):
            if seen.get("/".join(parents[:i])) is False:
                raise WheelError(f"File/directory collision: {key}")
    records = [
        name for name in files if re.fullmatch(r"torch-[^/]+\.dist-info/RECORD", name)
    ]
    if len(records) != 1 or any(
        name.endswith(("/RECORD.jws", "/RECORD.p7s")) for name in files
    ):
        raise WheelError("Expected one torch RECORD and no wheel-level signatures")
    return files, records[0]


def owner(name):
    return (
        "pytorch" if name in FIRST_PARTY or CORE_PYD.fullmatch(name) else "unclassified"
    )


def inspect_wheel(wheel):
    if not re.fullmatch(r"torch-[A-Za-z0-9_.+!-]+-win_amd64\.whl", wheel.name):
        raise WheelError("Expected a torch Windows x64 wheel")
    with wheel.open("rb") as raw, zipfile.ZipFile(raw) as archive:
        files, record = wheel_members(archive)
        if files[record].file_size > 32 * 1024**2:
            raise WheelError("RECORD exceeds the size limit")
        rows = {}
        for row in csv.reader(io.StringIO(archive.read(record).decode("utf-8"))):
            if len(row) != 3 or row[0] in rows:
                raise WheelError("Malformed or duplicate RECORD entry")
            rows[row[0]] = row[1:]
        if set(rows) != set(files) or rows[record] != ["", ""]:
            raise WheelError("RECORD must cover every file exactly once")
        identities = {}
        native = []
        for name in files:
            with archive.open(name) as stream:
                sha256, size = digest(stream)
            identities[name] = {"sha256": sha256, "bytes": size}
            if name != record:
                encoded = (
                    base64.urlsafe_b64encode(bytes.fromhex(sha256))
                    .rstrip(b"=")
                    .decode()
                )
                if rows[name] != [f"sha256={encoded}", str(size)]:
                    raise WheelError(f"RECORD mismatch: {name}")
            if Path(name).suffix.lower() in NATIVE_SUFFIXES:
                native.append({"path": name, **identities[name], "owner": owner(name)})
        if not native:
            raise WheelError("Wheel contains no native files")
        raw.seek(0)
        sha256, size = digest(raw)
    return {
        "wheel": {"name": wheel.name, "sha256": sha256, "bytes": size},
        "files": identities,
        "record": record,
        "native": native,
    }


def extract_wheel(wheel, root):
    with zipfile.ZipFile(wheel) as archive:
        files, _ = wheel_members(archive)
        for name in files:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)


def repack_wheel(original, root, output):
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(output, "w") as dst:
        files, record = wheel_members(src)
        rows = []
        for name, entry in files.items():
            if name == record:
                continue
            with (root / name).open("rb") as stream:
                sha256, size = digest(stream)
            encoded = (
                base64.urlsafe_b64encode(bytes.fromhex(sha256)).rstrip(b"=").decode()
            )
            rows.append([name, f"sha256={encoded}", str(size)])
            info = copy.copy(entry)
            info.file_size = size
            with (
                (root / name).open("rb") as stream,
                dst.open(info, "w", force_zip64=True) as out,
            ):
                shutil.copyfileobj(stream, out, 1024 * 1024)
        rows.append([record, "", ""])
        text = io.StringIO(newline="")
        csv.writer(text, lineterminator="\n").writerows(rows)
        dst.writestr(copy.copy(files[record]), text.getvalue().encode("utf-8"))


def context_from_env():
    return {
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "sha": os.environ.get("GITHUB_SHA", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
    }


def require_release_context():
    context = context_from_env()
    ref = os.environ.get("GITHUB_REF", "")
    if (
        context["repository"] != "pytorch/pytorch"
        or os.environ.get("GITHUB_EVENT_NAME") != "push"
        or os.environ.get("GITHUB_REF_PROTECTED") != "true"
        or not (
            ref == "refs/heads/nightly"
            or re.fullmatch(r"refs/tags/v\d+\.\d+\.\d+(?:-rc\d+)?", ref)
        )
        or not re.fullmatch(r"[a-f0-9]{40}", context["sha"])
        or not context["run_id"].isdigit()
        or not context["run_attempt"].isdigit()
    ):
        raise WheelError("Signing requires an official protected nightly/release push")
    return context


def prepare_output(output):
    if output.is_symlink() or (
        output.exists()
        and (
            not output.is_dir()
            or any(output.iterdir())
            or getattr(output.lstat(), "st_file_attributes", 0) & 1024
        )
    ):
        raise WheelError(
            f"Refusing a nonempty or redirected output directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)


def publish(stage, output):
    prepare_output(output)
    if output.exists():
        output.rmdir()
    stage.replace(output)


def write_receipt(stage, inspection, context, mode, native, **extra):
    report = {
        "schema_version": 1,
        "mode": mode,
        "context": context,
        "wheel": inspection["wheel"],
        "native": native,
        **extra,
    }
    (stage / RECEIPT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def publish_unsigned(wheel, output, context):
    prepare_output(output)
    inspection = inspect_wheel(wheel)
    with tempfile.TemporaryDirectory(prefix="wheel-stage-", dir=output.parent) as temp:
        stage = Path(temp) / "final"
        stage.mkdir()
        shutil.copyfile(wheel, stage / wheel.name)
        if file_identity(stage / wheel.name) != {
            k: inspection["wheel"][k] for k in ("sha256", "bytes")
        }:
            raise WheelError("Wheel changed during staging")
        report = write_receipt(
            stage,
            inspection,
            context,
            "unsigned",
            inspection["native"],
            all_native_valid=False,
            signing_status="not-configured",
        )
        publish(stage, output)
    return report


def verify_native(root, names):
    if os.name != "nt":
        raise WheelError("Authenticode verification requires Windows")
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        raise WheelError("powershell.exe is required")
    with tempfile.TemporaryDirectory(prefix="authenticode-") as temp:
        manifest = Path(temp) / "files.json"
        result = Path(temp) / "signatures.json"
        manifest.write_text(
            json.dumps(
                [{"path": name, "file": str((root / name).resolve())} for name in names]
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(HERE / "verify_authenticode.ps1"),
                "-Manifest",
                str(manifest),
                "-Output",
                str(result),
            ],
            check=True,
        )
        rows = json.loads(result.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != len(names):
        raise WheelError("Incomplete Authenticode result")
    results = {row["path"]: row for row in rows}
    if set(results) != set(names):
        raise WheelError("Duplicate or unexpected Authenticode result")
    return results


def validate_policy(policy):
    if (
        set(policy) != {"schema_version", "publisher_sha256", "third_party"}
        or policy["schema_version"] != 1
    ):
        raise WheelError("Unknown signing policy schema")
    pins = policy["publisher_sha256"]
    third_party = policy["third_party"]
    if not isinstance(third_party, dict):
        raise WheelError(
            "third_party must map exact paths to signer certificate SHA-256 values"
        )
    for name in third_party:
        safe_name(name)
        if owner(name) == "pytorch" or Path(name).suffix.lower() not in NATIVE_SUFFIXES:
            raise WheelError(f"Invalid third-party policy path: {name}")
    for values in [pins, *third_party.values()]:
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
                for value in values
            )
        ):
            raise WheelError(
                "Policy requires nonempty lowercase certificate SHA-256 lists"
            )


def valid_signature(signature, pins, timestamp=False):
    return (
        signature.get("status") == "Valid"
        and signature.get("signature_type") == "Authenticode"
        and signature.get("signer_sha256") in pins
        and (not timestamp or signature.get("timestamped") is True)
    )


def required_targets(native, backend):
    if backend not in ("cpu", "cuda"):
        raise WheelError("Signing currently supports reviewed CPU and CUDA wheels only")
    names = {item["path"] for item in native}
    expected = {f"torch/lib/{name}" for name in CPU_DLLS}
    if backend == "cuda":
        expected |= {f"torch/lib/{name}" for name in CUDA_DLLS}
    if (
        not expected <= names
        or sum(bool(CORE_PYD.fullmatch(name)) for name in names) != 1
    ):
        raise WheelError("Missing required PyTorch DLLs or exactly one _C PYD")
    if backend == "cpu" and any(f"torch/lib/{name}" in names for name in CUDA_DLLS):
        raise WheelError("CUDA DLLs in a CPU signing request")


def signer_command_from_env():
    try:
        command = json.loads(os.environ["WINDOWS_WHEEL_SIGNING_COMMAND"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise WheelError(
            "A maintainer-provisioned signing command is required"
        ) from exc
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in command)
        or not Path(command[0]).is_absolute()
        or Path(command[0]).suffix.lower() != ".exe"
        or not Path(command[0]).is_file()
    ):
        raise WheelError(
            "Signing command must be an argv list beginning with an absolute executable"
        )
    return command


def sign_wheel(
    wheel, output, context, policy, backend, sign_file, verify=verify_native
):
    prepare_output(output)
    validate_policy(policy)
    inspection = inspect_wheel(wheel)
    required_targets(inspection["native"], backend)
    names = [item["path"] for item in inspection["native"]]
    with tempfile.TemporaryDirectory(prefix="wheel-sign-", dir=output.parent) as temp:
        root = Path(temp) / "unpacked"
        root.mkdir()
        extract_wheel(wheel, root)
        before = verify(root, names)
        pending = set()
        for item in inspection["native"]:
            name = item["path"]
            signature = before[name]
            if item["owner"] == "pytorch":
                if signature["status"] == "NotSigned":
                    pending.add(name)
                elif not valid_signature(
                    signature, policy["publisher_sha256"], timestamp=True
                ):
                    raise WheelError(f"Unexpected or invalid PyTorch signature: {name}")
            elif name not in policy["third_party"] or not valid_signature(
                signature, policy["third_party"][name]
            ):
                raise WheelError(
                    f"Unclassified or invalid third-party signature: {name}"
                )
        # Validate the entire input before giving any file to the signing service.
        for name in sorted(pending):
            sign_file(root / name)
        actual_files = {
            p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
        }
        if actual_files != set(inspection["files"]) or any(
            p.is_symlink() for p in root.rglob("*")
        ):
            raise WheelError("Signing changed the file set or introduced a link")
        for name, identity in inspection["files"].items():
            if name not in pending and file_identity(root / name) != identity:
                raise WheelError(f"Signing changed an unrelated file: {name}")
        after = verify(root, names)
        native = []
        for item in inspection["native"]:
            name = item["path"]
            first_party = item["owner"] == "pytorch"
            pins = (
                policy["publisher_sha256"]
                if first_party
                else policy["third_party"][name]
            )
            if not valid_signature(after[name], pins, timestamp=first_party):
                raise WheelError(f"Required signature verification failed: {name}")
            native.append(
                {
                    "path": name,
                    **file_identity(root / name),
                    "owner": "pytorch" if first_party else "reviewed-third-party",
                    "signature": after[name],
                    "input_sha256": item["sha256"],
                    "action": "signed" if name in pending else "preserved",
                }
            )
        stage = Path(temp) / "final"
        stage.mkdir()
        repack_wheel(wheel, root, stage / wheel.name)
        final = inspect_wheel(stage / wheel.name)
        if {n["path"]: n["sha256"] for n in native} != {
            n["path"]: n["sha256"] for n in final["native"]
        }:
            raise WheelError("Repacked native bytes differ from verified bytes")
        report = write_receipt(
            stage,
            final,
            context,
            "signed",
            native,
            all_native_valid=True,
            backend=backend,
            input_wheel=inspection["wheel"],
            policy_sha256=hashlib.sha256(
                json.dumps(policy, sort_keys=True).encode()
            ).hexdigest(),
        )
        publish(stage, output)
    return report


def check_handoff(
    directory, context, require_signed=False, authenticode=False, verify=verify_native
):
    receipt = directory / RECEIPT
    if (
        receipt.is_symlink()
        or not receipt.is_file()
        or receipt.stat().st_size > 16 * 1024**2
    ):
        raise WheelError("Missing or invalid wheel handoff receipt")
    report = json.loads(receipt.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("context") != context:
        raise WheelError("Wheel handoff belongs to a different source/run/attempt")
    name = report["wheel"]["name"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise WheelError("Unsafe wheel handoff name")
    wheel = directory / name
    if {p.name for p in directory.iterdir()} != {name, RECEIPT} or wheel.is_symlink():
        raise WheelError("Unexpected or redirected files in final artifact directory")
    actual = inspect_wheel(wheel)
    if actual["wheel"] != report["wheel"]:
        raise WheelError("Wheel handoff digest mismatch")
    native = report["native"]
    if len(native) != len(actual["native"]) or {
        n["path"]: (n["sha256"], n["bytes"]) for n in native
    } != {n["path"]: (n["sha256"], n["bytes"]) for n in actual["native"]}:
        raise WheelError("Wheel handoff native inventory mismatch")
    if report.get("mode") not in ("signed", "unsigned"):
        raise WheelError("Unknown handoff mode")
    if require_signed:
        if report["mode"] != "signed" or report.get("all_native_valid") is not True:
            raise WheelError("Required signing is missing; unsigned is not signed")
        required_targets(actual["native"], report.get("backend"))
        for item in native:
            signature = item.get("signature", {})
            pin = signature.get("signer_sha256", "")
            if (
                not re.fullmatch(r"[a-f0-9]{64}", pin)
                or not valid_signature(
                    signature, [pin], timestamp=owner(item["path"]) == "pytorch"
                )
                or item.get("owner") not in ("pytorch", "reviewed-third-party")
            ):
                raise WheelError(
                    f"Missing verified signature in handoff: {item['path']}"
                )
        if authenticode:
            with tempfile.TemporaryDirectory(prefix="wheel-recheck-") as temp:
                root = Path(temp)
                extract_wheel(wheel, root)
                signatures = verify(root, [n["path"] for n in native])
                for item in native:
                    if not valid_signature(
                        signatures[item["path"]],
                        [item["signature"]["signer_sha256"]],
                        timestamp=owner(item["path"]) == "pytorch",
                    ):
                        raise WheelError(
                            f"Downloaded Authenticode verification failed: {item['path']}"
                        )
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("check-release")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("wheel", type=Path)
    check = commands.add_parser("check")
    check.add_argument("directory", type=Path)
    check.add_argument("--require-signed", action="store_true")
    check.add_argument("--authenticode", action="store_true")
    sign = commands.add_parser("sign")
    sign.add_argument("directory", type=Path)
    sign.add_argument("output", type=Path)
    sign.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.operation == "check-release":
        require_release_context()
    elif args.operation == "inspect":
        result = inspect_wheel(args.wheel)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    elif args.operation == "check":
        wheel = check_handoff(
            args.directory, context_from_env(), args.require_signed, args.authenticode
        )
        print(
            f"Verified final wheel handoff: {wheel.name}; required signing: {args.require_signed}"
        )
    else:
        context = require_release_context()
        wheel = check_handoff(args.directory, context)
        if (
            json.loads((args.directory / RECEIPT).read_text(encoding="utf-8"))["mode"]
            != "unsigned"
        ):
            raise WheelError("Expected the unsigned build artifact")
        try:
            policy = json.loads(os.environ["WINDOWS_WHEEL_SIGNING_POLICY"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise WheelError("An explicit reviewed signing policy is required") from exc
        command = signer_command_from_env()
        sign_wheel(
            wheel,
            args.output,
            context,
            policy,
            args.backend,
            lambda path: subprocess.run([*command, str(path.resolve())], check=True),
        )
        print(
            "Signed and verified PyTorch native files; third-party signatures preserved"
        )


if __name__ == "__main__":
    main()
