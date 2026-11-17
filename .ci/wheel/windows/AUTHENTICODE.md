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

## Azure Public Trust and GitHub OIDC (documentation only)

The following is an integration outline, **not an enabled release workflow**.
It uses Microsoft's existing Action for external signing; neither helper logs in
to Azure or handles credentials. No adapter, Azure SDK authentication, resource
provisioning, or build/publication hook is added.

Release administrators must configure these prerequisites outside this repository:

- An Artifact Signing account with completed identity validation and an approved
  **Public Trust** certificate profile, not Public Trust Test or Private Trust.
  Assign the signing service principal the
  [Artifact Signing Certificate Profile Signer role](https://learn.microsoft.com/en-us/azure/artifact-signing/concept-resources-roles#supported-roles)
  scoped to the target certificate profile, rather than the whole subscription.
- A Microsoft Entra application/service principal with a federated identity
  credential. For GitHub's default environment subject format, use issuer
  `https://token.actions.githubusercontent.com`, audience
  `api://AzureADTokenExchange`, and subject
  `repo:OWNER/REPOSITORY:environment:windows-wheel-signing`. Replace
  `OWNER/REPOSITORY` with the exact authorized repository, not a wildcard or
  another fork. The environment name must match the signing job. See
  [Azure's OIDC setup](https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure-openid-connect).
  No client secret or exported signing private key is needed.
- A preconfigured `windows-wheel-signing` GitHub environment with required human
  reviewers, prevention of self-review, and deployment restrictions to approved
  protected release refs. A future workflow must be manually dispatched from a
  trusted ref and use a trusted Windows runner. The `environment:` line alone
  does not create these protections. Do not grant untrusted PR runs signing
  access, execute PR-supplied code, or sign unreviewed PR artifacts in this job.
- Environment secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and
  `AZURE_SUBSCRIPTION_ID` for `azure/login`, plus environment variables
  `ARTIFACT_SIGNING_ENDPOINT`, `ARTIFACT_SIGNING_ACCOUNT`, and
  `ARTIFACT_SIGNING_PROFILE` configured through GitHub environment **variables**
  (`vars` below). Copy the endpoint from the account's Azure region and the
  account/profile names from the approved resources; these are not bundled values.
  Independently establish the profile's full `profile_eku` and provenance in the
  review JSON as described above.

### Prepare, external signing, and repack handoff

First run `prepare` as shown above against the exact approved wheel and review.
Retain its stdout JSON, including `manifest_sha256`, in an independently controlled
release record **outside the signing workspace** before allowing signing. Stage
only the resulting `manifest.json` and `tree/` in `prepared/`. Supply the retained
digest from that record, not a newly computed digest of the transferred manifest.
An unresolved dependency is a release blocker, not permission to add vendor targets.

The YAML below is a **job excerpt**, intentionally without a trigger, checkout,
input transfer, or publication step. It starts after preparation. Before use,
provide a trusted helper checkout at a reviewed immutable revision, the prepared
workspace, an existing Python interpreter and installed SDK SignTool, and an
existing output parent directory. Replace all `REPLACE_...` values; the digest is
generated by `prepare`, while the tool/output paths are operator-selected inputs.
No Azure login, signing, or Windows verification has been exercised by this example.

```yaml
jobs:
  sign:
    runs-on: windows-2022
    environment: windows-wheel-signing
    permissions:
      id-token: write
      contents: read
    env:
      SIGNING_WORKSPACE: ${{ github.workspace }}\prepared
      RETAINED_MANIFEST_SHA256: REPLACE_FROM_INDEPENDENT_PREPARATION_RECORD
      SIGNTOOL: REPLACE_WITH_INSTALLED_SIGNTOOL_ABSOLUTE_PATH
      OUTPUT_WHEEL: REPLACE_WITH_NEW_OUTPUT_ABSOLUTE_WHEEL_PATH
    steps:
      - name: Bind exact signing paths to the retained manifest
        id: targets
        shell: pwsh
        run: |
          $ErrorActionPreference = 'Stop'
          $path = Join-Path $env:SIGNING_WORKSPACE 'manifest.json'
          $digest = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
          if ($env:RETAINED_MANIFEST_SHA256 -notmatch '^[0-9a-f]{64}$' -or
              $digest -ine $env:RETAINED_MANIFEST_SHA256) {
            throw 'Manifest differs from independently retained preparation digest'
          }
          $manifest = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
          if ($manifest.schema -ne 1 -or @($manifest.unresolved).Count -ne 0) {
            throw 'Require schema 1 with no unresolved native dependencies'
          }
          $files = @($manifest.review.targets | ForEach-Object {
            $name = $_.path
            $file = [IO.Path]::GetFullPath((Join-Path "$env:SIGNING_WORKSPACE\tree" $name))
            if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ine
                $manifest.files.$name.sha256) {
              throw "Signing target bytes differ from preparation: $name"
            }
            $file
          })
          if ($files.Count -eq 0 -or ($files -match ',')) {
            throw 'Require explicit manifest targets with no comma-delimited paths'
          }
          @('files<<MANIFEST_TARGETS'; $files; 'MANIFEST_TARGETS') |
            Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append

      - name: Azure login with GitHub OIDC
        uses: azure/login@a641126d1b8aa4d1fa005f4f92df94a3a4c4c906 # v3.1.0
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
          audience: api://AzureADTokenExchange

      - name: Sign only reviewed PyTorch targets
        uses: Azure/artifact-signing-action@c7ab2a863ab5f9a846ddb8265964877ef296ee82 # v2.0.0
        with:
          endpoint: ${{ vars.ARTIFACT_SIGNING_ENDPOINT }}
          signing-account-name: ${{ vars.ARTIFACT_SIGNING_ACCOUNT }}
          certificate-profile-name: ${{ vars.ARTIFACT_SIGNING_PROFILE }}
          files: ${{ steps.targets.outputs.files }}
          file-digest: SHA256
          timestamp-rfc3161: http://timestamp.acs.microsoft.com
          timestamp-digest: SHA256
          append-signature: false
          cache-dependencies: false
          exclude-environment-credential: true
          exclude-workload-identity-credential: true
          exclude-managed-identity-credential: true
          exclude-shared-token-cache-credential: true
          exclude-visual-studio-credential: true
          exclude-visual-studio-code-credential: true
          exclude-azure-cli-credential: false
          exclude-azure-powershell-credential: true
          exclude-azure-developer-cli-credential: true
          exclude-interactive-browser-credential: true

      - name: Verify and repack without publishing
        shell: pwsh
        run: |
          $ErrorActionPreference = 'Stop'
          $result = & python -I .ci/wheel/windows/authenticode.py repack `
            --workspace $env:SIGNING_WORKSPACE `
            --manifest-sha256 $env:RETAINED_MANIFEST_SHA256 `
            --output $env:OUTPUT_WHEEL --signtool $env:SIGNTOOL
          if ($LASTEXITCODE -ne 0) { throw 'Repack verification failed' }
          $result | Out-File "$env:RUNNER_TEMP\repack-result.json" -Encoding utf8
```

The pinned Action's
[README](https://github.com/Azure/artifact-signing-action/blob/c7ab2a863ab5f9a846ddb8265964877ef296ee82/README.md),
[OIDC guide](https://github.com/Azure/artifact-signing-action/blob/c7ab2a863ab5f9a846ddb8265964877ef296ee82/docs/OIDC.md),
and [input definitions](https://github.com/Azure/artifact-signing-action/blob/c7ab2a863ab5f9a846ddb8265964877ef296ee82/action.yml)
define this contract; the login pin is
[v3.1.0](https://github.com/Azure/login/blob/a641126d1b8aa4d1fa005f4f92df94a3a4c4c906/action.yml).
Re-review immutable pins and dependencies before adopting or updating an integration.
The signing Action uses the Azure CLI session established by `azure/login`;
excluding its separate `WorkloadIdentityCredential` does not disable that OIDC login.

`files` is generated only from schema 1's `manifest.review.targets[*].path`, under
`tree/`, after checking the independently retained digest and original target
hashes. Commas are rejected because the Action also treats them as list separators.
Do not substitute `files-folder`, a glob, or the complete native inventory.
Already-signed vendors must remain byte-identical; keep logs and other sidecars
outside `prepared/`.
`certificate-profile-name` selects an Azure resource: it is **not** the full
profile-specific EKU, a subject-name identity check, or a replacement for the
review file's trusted profile-to-EKU mapping. Repack still verifies that entire
EKU and all native signatures/timestamps, preserves vendor bytes, and regenerates
RECORD before any separate release approval or publication.

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
