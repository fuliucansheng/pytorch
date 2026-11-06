"""Opt-in, offline wheel preparation and post-signing verification."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


PROFILE_PREFIX = "1.3.6.1.4.1.311.97."
GENERIC_EKU = PROFILE_PREFIX + "1.0"
CHUNK_SIZE = 1024 * 1024


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "size": size}


def checked_path(path):
    path = Path(os.path.abspath(path))
    for ancestor in (*reversed(path.parents), path):
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError(f"Symlink/reparse point is not allowed: {ancestor.name}")
    return path


def validate_name(name):
    if (
        not name
        or not name.isascii()
        or any(
            ord(character) < 32 or ord(character) == 127 or character in '\\:<>"|?*~'
            for character in name
        )
    ):
        raise ValueError(f"Unsafe or ambiguous Windows path: {name!r}")
    parts = name.split("/")
    for component in parts:
        stem = component.split(".", 1)[0].rstrip(" ").upper()
        if (
            not component
            or component in {".", ".."}
            or component.endswith((".", " "))
            or stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
            or re.fullmatch(r"(?:COM|LPT)[1-9]", stem)
        ):
            raise ValueError(f"Unsafe or reserved Windows path: {name!r}")
    return parts


def archive_layout(archive):
    files = {}
    directories = set()
    nodes = {}
    explicit = set()
    for member in archive.infolist():
        name = member.filename.rstrip("/") if member.is_dir() else member.filename
        parts = validate_name(name)
        if member.orig_filename != member.filename or member.filename.endswith("//"):
            raise ValueError("ZIP name has ambiguous terminators")
        mode = stat.S_IFMT(member.external_attr >> 16)
        if (
            mode not in {0, stat.S_IFREG, stat.S_IFDIR}
            or member.external_attr & 0x400
            or (mode == stat.S_IFDIR and not member.is_dir())
            or (mode == stat.S_IFREG and member.is_dir())
            or (member.external_attr & 0x10 and not member.is_dir())
            or member.flag_bits & 1
            or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        ):
            raise ValueError(f"Unsupported ZIP member type: {name}")
        if name.casefold() in explicit:
            raise ValueError(f"Duplicate ZIP member: {name}")
        explicit.add(name.casefold())
        for index in range(1, len(parts) + 1):
            node = "/".join(parts[:index])
            kind = "directory" if index < len(parts) or member.is_dir() else "file"
            key = node.casefold()
            if key in nodes and nodes[key] != (node, kind):
                raise ValueError(f"Case collision or file/directory conflict: {name}")
            nodes[key] = (node, kind)
            if kind == "directory":
                directories.add(node)
        if not member.is_dir():
            files[name] = member
    records = [name for name in files if re.fullmatch(r"[^/]+\.dist-info/RECORD", name)]
    if len(records) != 1:
        raise ValueError("Expected exactly one top-level wheel RECORD")
    record = records[0]
    metadata_directory = record.rsplit("/", 1)[0]
    for leaf in ("WHEEL", "METADATA"):
        if f"{metadata_directory}/{leaf}" not in files:
            raise ValueError(f"Missing wheel metadata: {leaf}")
    if any(name.endswith(("/RECORD.jws", "/RECORD.p7s")) for name in files):
        raise ValueError(
            "Stale RECORD signatures must be removed upstream, not silently copied"
        )
    return files, directories, record


def native_kind(path):
    with path.open("rb") as stream:
        header = stream.read(64)
        mz_header = header.startswith(b"MZ")
        pe_header = False
        if mz_header and len(header) == 64:
            stream.seek(int.from_bytes(header[60:64], "little"))
            pe_header = stream.read(4) == b"PE\0\0"
    return mz_header or path.suffix.lower() in {".dll", ".pyd", ".exe"}, pe_header


def inspect_windows(path, signtool):
    if sys.platform != "win32":
        raise ValueError("Real Authenticode inspection requires Windows")
    script = Path(__file__).with_suffix(".ps1").read_text(encoding="utf-8-sig")
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        base64.b64encode(script.encode("utf-16le")).decode("ascii"),
    ]
    completed = subprocess.run(
        command,
        input=json.dumps({"path": str(path), "signtool": str(signtool)}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(completed.stdout)


def verification_error(signature, profile=None):
    if signature.get("status") != "Valid":
        return "Missing or invalid Authenticode signature"
    if signature.get("signature_type") != "Authenticode":
        return "An embedded Authenticode signature is required"
    if signature.get("signtool_exit_code") != 0:
        return "SignTool verification failed or produced a warning"
    if not signature.get("timestamp"):
        return "Missing verified timestamp"
    signer = signature.get("signer")
    if not signer:
        return "Missing signer certificate"
    if profile is not None and profile not in signer.get("ekus", []):
        return "Signer lacks the full reviewed certificate-profile EKU"
    return None


def validate_review(review, wheel_digest):
    if review.get("wheel_sha256") != wheel_digest:
        raise ValueError("Review does not bind this exact input wheel")
    for field in ("reviewed_by", "profile_eku_provenance"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"Missing explicit reviewer input: {field}")
    profile = review.get("profile_eku", "")
    if not re.fullmatch(r"1\.3\.6\.1\.4\.1\.311\.97\.(?:[0-9]+\.)+[0-9]+", profile):
        raise ValueError("A full Public Trust certificate-profile EKU is required")
    if profile == GENERIC_EKU:
        raise ValueError("The generic Public Trust EKU does not identify a profile")
    targets = review.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Explicit individual reviewed targets are required")
    names = set()
    for target in targets:
        validate_name(target["path"])
        if (
            target.get("owner") != "PyTorch"
            or not target.get("source_provenance", "").startswith("https://")
            or target["path"] in names
        ):
            raise ValueError(
                "Targets need unique paths, PyTorch ownership and source provenance"
            )
        names.add(target["path"])
    return names


def snapshot(tree):
    files = set()
    directories = set()
    checked_path(tree)
    for current, subdirectories, filenames in os.walk(tree, followlinks=False):
        for leaf in subdirectories + filenames:
            path = checked_path(Path(current) / leaf)
            relative = path.relative_to(tree).as_posix()
            validate_name(relative)
            if path.is_dir():
                directories.add(relative)
            elif path.is_file():
                files.add(relative)
            else:
                raise ValueError(f"Unexpected filesystem object: {relative}")
    return files, directories


def prepare(wheel, review, workspace, inspect):
    wheel = checked_path(wheel)
    workspace = checked_path(workspace)
    if workspace.exists() or not workspace.parent.is_dir():
        raise ValueError(
            "Preparation requires a new workspace in an existing directory"
        )
    wheel_identity = digest_file(wheel)
    targets = validate_review(review, wheel_identity["sha256"])
    with zipfile.ZipFile(wheel) as archive:
        members, directories, record = archive_layout(archive)
        if not targets <= members.keys():
            raise ValueError("A reviewed target is absent from the input wheel")
        workspace.mkdir()
        tree = workspace / "tree"
        tree.mkdir()
        for directory in sorted(directories):
            (tree / directory).mkdir(exist_ok=True)
        originals = {}
        native = {}
        unresolved = []
        for name, member in sorted(members.items()):
            path = tree / name
            with archive.open(member) as source, path.open("xb") as destination:
                shutil.copyfileobj(source, destination, CHUNK_SIZE)
            originals[name] = {
                **digest_file(path),
                "date_time": list(member.date_time),
                "external_attr": member.external_attr,
            }
            candidate, is_pe = native_kind(path)
            if not candidate:
                if name in targets:
                    raise ValueError(f"Reviewed target is not a native binary: {name}")
                continue
            signature = inspect(path)
            native[name] = {"is_pe": is_pe, "signature": signature}
            if name in targets:
                if (
                    not is_pe
                    or signature.get("status") != "NotSigned"
                    or signature.get("signer")
                ):
                    raise ValueError(
                        f"Only unsigned reviewed PyTorch PE targets may change: {name}"
                    )
            else:
                problem = verification_error(signature)
                if not is_pe or problem:
                    unresolved.append(
                        {"path": name, "reason": problem or "Not a supported PE file"}
                    )
        manifest = {
            "schema": 1,
            "wheel": {"filename": wheel.name, **wheel_identity},
            "review": review,
            "record": record,
            "files": originals,
            "directories": sorted(directories),
            "archive_directories": [
                member.filename for member in archive.infolist() if member.is_dir()
            ],
            "native": native,
            "unresolved": unresolved,
        }
    if digest_file(wheel) != wheel_identity:
        raise ValueError("Input wheel changed during preparation")
    manifest_path = workspace / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {
        "manifest_sha256": digest_file(manifest_path)["sha256"],
        "unresolved": unresolved,
    }


def checked_workspace(workspace, expected_digest):
    workspace = checked_path(workspace)
    if {path.name for path in workspace.iterdir()} != {"manifest.json", "tree"}:
        raise ValueError("Unexpected workspace objects")
    manifest_path = checked_path(workspace / "manifest.json")
    if digest_file(manifest_path)["sha256"] != expected_digest:
        raise ValueError("Manifest digest differs from the retained preparation digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != 1:
        raise ValueError("Unsupported manifest schema")
    validate_review(manifest["review"], manifest["wheel"]["sha256"])
    for name in manifest["files"]:
        validate_name(name)
    tree = checked_path(workspace / "tree")
    if snapshot(tree) != (set(manifest["files"]), set(manifest["directories"])):
        raise ValueError("Extracted file/directory set differs from preparation")
    return workspace, tree, manifest


def zip_info(name, original):
    member = zipfile.ZipInfo(name, tuple(original["date_time"]))
    member.external_attr = original["external_attr"]
    member.compress_type = zipfile.ZIP_DEFLATED
    return member


def repack(workspace, expected_digest, output, inspect):
    workspace, tree, manifest = checked_workspace(workspace, expected_digest)
    output = checked_path(output)
    if output.exists() or workspace == output or workspace in output.parents:
        raise ValueError("Output must be new and outside the preparation workspace")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    targets = {target["path"] for target in manifest["review"]["targets"]}
    verified = {}
    signatures = {}
    for name, original in sorted(manifest["files"].items()):
        path = checked_path(tree / name)
        identity = digest_file(path)
        if name not in targets and identity != {
            field: original[field] for field in ("sha256", "size")
        }:
            raise ValueError(f"Non-target bytes changed: {name}")
        if name in manifest["native"]:
            if not native_kind(path)[1]:
                raise ValueError(f"Native file is no longer a PE binary: {name}")
            signature = inspect(path)
            profile = manifest["review"]["profile_eku"] if name in targets else None
            problem = verification_error(signature, profile)
            if problem:
                raise ValueError(f"{name}: {problem}")
            if digest_file(path) != identity:
                raise ValueError(f"Bytes changed during signature inspection: {name}")
            signatures[name] = signature
        verified[name] = identity
    if snapshot(tree) != (set(verified), set(manifest["directories"])):
        raise ValueError("Workspace changed during verification")
    complete = False
    with output.open("xb") as raw:
        try:
            with zipfile.ZipFile(raw, "w", allowZip64=True) as archive:
                for directory in manifest["archive_directories"]:
                    archive.writestr(directory, b"")
                rows = []
                for name, original in sorted(manifest["files"].items()):
                    if name == manifest["record"]:
                        continue
                    digest = hashlib.sha256()
                    size = 0
                    with (
                        checked_path(tree / name).open("rb") as source,
                        archive.open(
                            zip_info(name, original), "w", force_zip64=True
                        ) as destination,
                    ):
                        for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                            destination.write(block)
                            digest.update(block)
                            size += len(block)
                    if {"sha256": digest.hexdigest(), "size": size} != verified[name]:
                        raise ValueError(f"Bytes changed during repack: {name}")
                    encoded = (
                        base64.urlsafe_b64encode(digest.digest())
                        .rstrip(b"=")
                        .decode("ascii")
                    )
                    rows.append((name, "sha256=" + encoded, str(size)))
                rows.append((manifest["record"], "", ""))
                record_text = io.StringIO(newline="")
                csv.writer(record_text, lineterminator="\n").writerows(rows)
                with archive.open(
                    zip_info(manifest["record"], manifest["files"][manifest["record"]]),
                    "w",
                    force_zip64=True,
                ) as destination:
                    destination.write(record_text.getvalue().encode("utf-8"))
            if snapshot(tree) != (set(verified), set(manifest["directories"])):
                raise ValueError("Workspace file set changed during repack")
            complete = True
        finally:
            if not complete:
                raw.close()
                output.unlink()
    return {
        "output": output.name,
        **digest_file(output),
        "verified_signatures": signatures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--wheel", type=Path, required=True)
    preparation.add_argument("--review", type=Path, required=True)
    preparation.add_argument("--workspace", type=Path, required=True)
    preparation.add_argument("--signtool", type=Path, required=True)
    verification = commands.add_parser("repack")
    verification.add_argument("--workspace", type=Path, required=True)
    verification.add_argument("--manifest-sha256", required=True)
    verification.add_argument("--output", type=Path, required=True)
    verification.add_argument("--signtool", type=Path, required=True)
    arguments = parser.parse_args()
    signtool = checked_path(arguments.signtool)
    if not signtool.is_file():
        parser.error("The explicitly selected installed SignTool is missing")

    def inspector(path):
        return inspect_windows(path, signtool)

    try:
        if arguments.command == "prepare":
            review = json.loads(arguments.review.read_text(encoding="utf-8"))
            result = prepare(arguments.wheel, review, arguments.workspace, inspector)
        else:
            result = repack(
                arguments.workspace,
                arguments.manifest_sha256,
                arguments.output,
                inspector,
            )
    except (
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
