# Opt-in Windows wheel Authenticode groundwork

This standalone tool prepares a wheel for an **external** signing step and verifies
it before repacking. It is not called by normal builds or publication. It does not
sign files, provision Azure resources, change Windows security, or establish Smart
App Control (SAC) compatibility. A release owner must review the code and inputs
before use. There is no signing-service adapter or credential handling here.

## Prerequisites and trust inputs

- An existing supported Python interpreter (the helper uses only the stdlib).
  Tests use PyTorch's `torch.testing._internal.common_utils` test runner and need
  an existing PyTorch test environment.
- Windows PowerShell with `Get-AuthenticodeSignature`, and an installed Microsoft
  Windows SDK `signtool.exe` selected by its explicit absolute path.
- A wheel from a trusted source, an exact SHA-256, and human-reviewed individual
  PyTorch-owned PE targets with build-source provenance. Directory membership and
  an NVIDIA-oriented filename do not prove ownership. In particular,
  `caffe2_nvrtc` is a PyTorch build target in `caffe2/CMakeLists.txt`.
- The **full trusted certificate-profile-specific durable EKU** for the approved
  Azure Artifact Signing Public Trust profile, with a reviewed mapping to that
  profile. `authenticode-review.example.json` is intentionally invalid until a
  human supplies these inputs; it is not a signing approval or complete target list.

[Microsoft certificate management documentation](https://learn.microsoft.com/en-us/azure/artifact-signing/concept-certificate-management)
describes the generic Public Trust EKU `1.3.6.1.4.1.311.97.1.0` and a durable,
profile-specific EKU beginning with `1.3.6.1.4.1.311.97.` followed by unique
components. This tool compares the **entire reviewed OID**, not a prefix or display
subject. The generic EKU is rejected. Leaf thumbprints are not durable across daily
certificate renewal. The public documentation's example OID is not a PyTorch
profile and must not be copied as an approval.

The binary alone cannot tell the tool which Azure resource the reviewer intended.
Release administrators must establish the profile-to-full-EKU mapping through a
trusted channel. The review file is an explicitly scoped verifier input, not a
receipt framework. No approved real profile mapping is bundled. Already-signed
vendor files do **not** have to carry the PyTorch profile EKU.

## Workflow

Run outside any source/build tree used for torch imports. Select an existing
interpreter and SignTool; do not install missing tools silently. Example commands
from the repository root (replace the input wheel, review, and installed SDK path):

```powershell
$signtool = (Get-Command signtool.exe -ErrorAction Stop).Source
python -I .ci/wheel/windows/authenticode.py prepare --wheel inputs/torch.whl --review review.json --workspace prepared --signtool $signtool
```

If the installed SDK SignTool is not on PATH, set `$signtool` to its verified
absolute executable path instead. Do not install tools or change the global PATH.

Preparation validates the complete archive before extraction, creates a **new**
workspace, and writes `tree/` plus `manifest.json`. Retain the printed
`manifest_sha256` independently of the mutable signing workspace. The manifest
binds the wheel identity, review, complete original file hashes/sizes, directory
set, native inventory, and unresolved dependencies. Do not edit the manifest.

Only individually reviewed, initially unsigned PyTorch PE files can be signing
targets. Every other original file, including vendor binaries and the original
RECORD, must remain byte-identical until repack. Existing signed targets are
rejected to avoid overwriting a vendor signature accidentally. Unknown/unsigned
non-target native dependencies are reported as unresolved. Resolve them with the
upstream/release owner and prepare a newly reviewed input wheel; do not broaden
targets or sign every file to make verification pass.

Perform external signing **only** on the exact manifest targets under `tree/`,
using the approved identity and timestamp service. Microsoft provides the
[Artifact Signing GitHub Action](https://github.com/Azure/artifact-signing-action).
Its actual release integration and identity authorization are outside this change.
Do not use a directory-wide glob that includes vendor binaries. Keep external
signing logs outside the workspace; unexpected workspace/tree objects are rejected.

```powershell
python -I .ci/wheel/windows/authenticode.py repack --workspace prepared --manifest-sha256 RETAINED_PREPARATION_DIGEST --output signed/torch.whl --signtool $signtool
```

The output parent must exist. The output must be new, outside the workspace, and
separate from the original wheel. Repack verifies every native file with Windows
trust APIs and `signtool verify /pa /all /v /tw`; missing tools, process failures,
nonzero exit codes (including timestamp-warning exit 2), invalid/missing embedded
signatures, missing/invalid timestamps, or wrong target profile fail closed.
`Get-AuthenticodeSignature` supplies the signer and timestamp certificates;
SignTool performs trust/timestamp verification. A timestamp certificate need not
be unexpired today if its historical timestamp is valid. Vendor verification uses
its own trust chain, without requiring the PyTorch profile. Output JSON includes
the verification diagnostics; preserve it using an explicit UTF-8 writer.

The PowerShell companion is ASCII source passed as an encoded inline command, not
loaded as a `.ps1` file. No execution-policy change is requested. JSON input is
passed over stdin, not interpolated into shell code. Trust/revocation/network
failures are errors, never silently downgraded to an unsigned pass.

Filesystem traversal errors abort verification with the original error and path;
an unreadable directory cannot be treated as an empty directory.
Repack rechecks the exact file/directory set and non-target hashes, binds signature
checks to file digests, and hashes bytes again while streaming them to ZIP64-capable
output. RECORD is regenerated with SHA-256 URL-safe base64 without padding, byte
sizes, and an empty self row. Stale `RECORD.jws`/`RECORD.p7s` are explicitly rejected
at preparation, not copied or silently dropped. Existing directories are retained.
Partial output created by a failed repack is removed; failed preparation workspaces
remain for diagnostics and must not be reused.

For conservative Windows extraction, names are ASCII only. Traversal, absolute,
drive/UNC, ADS, reserved DOS, trailing-dot/space, short-name aliases (`~`), control
characters, backslashes, duplicate/case-colliding names (including parent
directories), links/reparse points, special files, encryption and unsupported ZIP
compression are rejected. Workspaces/ancestors/output ancestors must not be links
or junctions. Keep the workspace private and quiescent: these accidental-mutation
guards are not a sandbox against a privileged concurrent filesystem attacker.

## Focused tests and limits

```text
python -I .ci/wheel/windows/test_authenticode.py -v
```

This command imports the helper by exact file path and uses the PyTorch test
runner from the selected environment; isolated mode avoids importing unbuilt
source torch. Its synthetic PE payloads, signer/profile data and timestamps are **mocks**;
no certificate is created and no mock is SAC evidence. Tests cover unchanged
vendors, failed signing/verification/timestamps/profile, mutated/additional/deleted
files, unreadable directory scans, unsafe ZIPs, RECORD, clobber/digest guards, and
ZIP64 size/count limits lowered
for small fixtures. Real Windows trust checks and an isolated import under actual
SAC **On** are separate required release validation. SAC Off/Evaluation, or a valid
signature alone, cannot establish compatibility.

Use repository-supported `spin` commands for repository lint. Do not substitute
tool installation or claim unexecuted tests passed when prerequisites are missing.
