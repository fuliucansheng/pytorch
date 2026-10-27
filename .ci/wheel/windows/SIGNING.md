# Windows wheel signing integration

This is an opt-in interface for a PyTorch-managed signing service, not a configured
service or a claim that existing wheels are signed. It addresses the packaging
part of [#199711](https://github.com/pytorch/pytorch/issues/199711). SAC enforcement,
Docling reproduction and real release signing still require separate validation.

## Artifact flow

The x64 build helper builds in a temporary directory, validates the complete
wheel RECORD, and atomically exposes one wheel plus `wheel-handoff.json`.
It rejects nonempty output directories instead of reusing stale wheels.
The receipt explicitly says `mode: unsigned` and `signing_status: not-configured`.

For build names selected for signing, the Windows workflow uploads this as
`<build_name>-unsigned`. The protected signing job downloads that exact artifact
from the same workflow run, checks its wheel hash and source/run/attempt, signs
the extracted first-party files, verifies them, regenerates every RECORD row and
checks the repacked wheel. Only then does it create `<build_name>-signed`.

The existing Windows smoke test downloads the final artifact and checks its
hash, complete RECORD and (when signing is required) Authenticode again before
installation. The upload job downloads the same immutable artifact name and
checks the receipt, wheel hash and RECORD before publishing. The receipt is a
handoff integrity check, not an independently signed provenance attestation:
its trust depends on the protected producer and GitHub's immutable, same-run
artifact. It is not accepted from arbitrary runs or URLs.

Build artifact upload now requires success. Wheel publication explicitly
requires wheel-test success. Signing or verification failure never falls back
to the unsigned artifact. Failure while building/signing leaves intermediates
in temporary storage and does not create a final artifact. Existing nonempty
directories are left untouched for diagnosis and rejected; failure guards
prevent them being published. Libtorch extraction uses the tested final source
wheel; this does not add a separate libtorch signing/release acceptance claim.

## Scope and policy

First-party coverage is the six CPU DLLs `c10`, `shm`, `torch`, `torch_cpu`,
`torch_global_deps`, `torch_python`; CUDA additionally requires `c10_cuda`,
`caffe2_nvrtc`, `torch_cuda`. All are under `torch/lib/`.
Exactly one `torch/_C[.<ABI>].pyd` is required. The scan is not pinned to cp313:
all DLL, PYD and EXE members are inventoried. CPU wheels do not need CUDA DLLs.

The CLI supports reviewed x64 CPU/CUDA configurations. ARM64 and non-Windows
pipelines retain their existing build path. New first-party targets, XPU and
other variants require explicit ownership review before opting them in.
No matrix entry is removed.

The required signing mode fails closed on unknown native paths, invalid or
missing required signatures, wrong certificate identities, missing first-party
timestamps, unexpected catalog-only signatures, malformed archives, or bad
RECORDs. A file does not become an approved third-party component merely because
it has a valid signature. Each third-party path must have an explicit allowed
certificate identity. Already valid approved signatures are preserved byte for
byte. Existing invalid/unexpected first-party signatures are rejected, not
silently overwritten.

In particular, this change does not exempt, replace, delete or sign
`nvToolsExt64_1.dll`, `uv.dll` or `zlibwapi.dll`. Their source/licensing and
signing decision is separate. If an opted-in wheel still includes these unsigned
files, preflight refuses it before calling the signer. Resolve those dependencies
before enabling that build. There is deliberately no "allow unsigned" policy
escape hatch.

## Maintainer configuration

These are **new proposed configuration names**, not evidence that any such
variables, environment, runner, service or identity currently exists.

1. Provision an ephemeral Windows x64 signing runner with the labels
   `self-hosted, Windows, X64, windows-wheel-signing`, Python 3.10+ and Windows
   PowerShell. Do not share it with untrusted PR build jobs.
2. Create and protect the `windows-wheel-signing` environment. Permit only
   protected nightly/release refs. Configure the signing provider's identity
   policy to bind the official repository, protected ref, environment and
   approved workflow identity. The code checks the context again; environment
   and provider protections must remain an independent boundary.
3. Provision a reviewed, immutable signing adapter on that runner. It receives
   one absolute PE filename as its final argv element. It must sign in place
   using the authorized publisher identity, SHA-256 and a trusted timestamp,
   return nonzero on any failure, and avoid changing any other file. The helper
   independently verifies the result. Authentication belongs in the adapter's
   managed identity/provider integration, never command-line secrets or source.
4. Set environment variable definitions through GitHub environment **variables**:
   `WINDOWS_WHEEL_SIGNING_COMMAND` is a JSON argv array whose first item is an
   absolute existing `.exe` path. Shell strings and `.cmd/.bat` launchers are
   rejected. Remaining fixed arguments may identify the reviewed adapter.
   `WINDOWS_WHEEL_SIGNING_POLICY` is the JSON policy below.
5. After real end-to-end testing, set repository variable
   `WINDOWS_WHEEL_SIGNING_BUILD_NAMES` to a JSON array of the exact reviewed
   matrix build names to enable. An absent value means `[]`: builds remain
   explicitly unsigned. Invalid configuration of an opted-in build fails.
   Enabling happens only for official nightly or release-tag pushes. PR/ciflow
   jobs never request production signing.

Policy shape (placeholders must be replaced with actual reviewed values):

```json
{
  "schema_version": 1,
  "publisher_sha256": ["<lowercase SHA-256 of the authorized leaf certificate DER>"],
  "third_party": {
    "torch/lib/<reviewed-vendor-file>.dll": ["<lowercase vendor certificate DER SHA-256>"]
  }
}
```

All fingerprints are 64 lowercase hex characters, not the usual SHA-1
`Thumbprint`. Windows must report a valid embedded Authenticode signature;
first-party signatures must also have a timestamp. The release owner must
establish public trust, ownership, service access and certificate rotation.
A custom trust store on one runner does not establish trust on customers'
machines. No certificate or trust-store installation is performed by this code.

The signing job alone requests `id-token: write`; the helper PR tests have
`contents: read`, no signing environment and no signing credentials.
No `pull_request_target` path is added. There is no workflow dispatch input for
an arbitrary source commit, external run or artifact URL.

## Tests and remaining validation

Run the standard-library bootstrap tests without building or importing torch:

```sh
python -B .ci/wheel/windows/test_sign_wheel.py -v
```

The packaging tests use explicitly **mock** signer and certificate results. They exercise
CPU/CUDA target coverage, alternate PYD ABI names, third-party preservation,
signature failures, configuration absence, archive traversal/collisions,
full RECORD validation/repacking and failure-path handoff. They do not measure
SAC behavior. A separate Windows-only read-only adapter regression verifies an
existing system-signed executable and an unsigned file through the real
Python/PowerShell adapter, including zero/one/two-file output shapes and null
certificate fields. It creates no certificate and does not sign files. The PR
workflow runs the suite on Windows and Linux.

Run project lint through `spin lint` and regenerate workflows through
`spin regenerate-github-workflows` after editing the Jinja template.
Do not hand-edit a generated workflow.

Before enabling a build, maintainers still need real service integration,
reviewed third-party policy, install/import and GPU smoke tests using the final
signed wheel, release/upload verification, and SAC enforcement testing on an
already-authorized machine. A valid signature alone is not a SAC acceptance
result. Any later metadata repackaging (for example PyPI promotion in test-infra)
must preserve PE bytes and recheck full RECORD/signatures; a repacked wheel has a
new digest and cannot reuse this receipt. That separate promotion path has not
been executed or modified by this patch.
