# Windows PyTorch issue processing registry

This durable registry lives on `msbeacon/windows-issue-registry` in the personal fork `fuliucansheng/pytorch`. It coordinates native-Windows issue work across sessions. It is not a code branch, proof of reproduction, or a claim that upstream has accepted a fix.

## History first, on every run

1. Freeze and retain the inclusive discovery window and exact query. Consider only currently open upstream issues created within that window with actual reported native-Windows failures. A platform mentioned in disabled-test bot instructions, Linux under WSL2 alone, or a tensor's sliding windows is not native-Windows failure evidence.
2. Read this stable branch before assigning or reproducing anything. Consult `issues/NUMBER.json`, `reports/NUMBER.md`, prior `runs/*/manifest.json`, existing investigation reports, and available issue timeline references. Do not equate missing local files with an absence of processing history.
3. Check all open **and closed** pull requests in the personal fork, following pagination, and relevant upstream PR references/search results. An existing investigation, candidate fix, or PR is prior processing even if not merged. Never create a duplicate investigation, owner assignment, or PR. Search limitations must remain explicit; unreferenced private sessions are not observable.
4. Skip processed, terminal, active, or documented Blocked records owned by another assignment. Skip unknown ownership rather than overwriting it. A documented Blocked issue remains skipped until meaningful new evidence or an explicit resume request is retained. A previously processed issue is not automatically reopened because another run starts.

## Atomic bootstrap and claim

If the registry branch is absent, read the current personal `main` HEAD and atomically `POST /repos/fuliucansheng/pytorch/git/refs` with `ref=refs/heads/msbeacon/windows-issue-registry` and that SHA. If another worker already created it, re-read and accept the existing branch. Never reset, delete, force-update, or force-push an existing ref. Do not change personal `main` or existing code branches.

Only the designated issue owner may claim its issue. Create `windows-triage/issues/NUMBER.json` using `PUT /repos/fuliucansheng/pytorch/contents/windows-triage/issues/NUMBER.json`, `branch=msbeacon/windows-issue-registry`, base64 UTF-8 JSON content, and **no `sha` field**. This is create-if-absent, not an upsert. Do not reproduce or fix before the durable claim exists.

On HTTP 409/422, re-read the path and branch. Never overwrite another claim. Only a matching owner assignment may update its own existing claim, supplying the freshly observed file SHA. Use bounded compare-and-swap retries; re-read after each conflict. A missing, unfamiliar, processed, or differently owned record is not update authority. Preserve timestamps and issue/report/PR links. An HTTP success alone does not establish a validated fix.

## Records and status semantics

Owner records identify the upstream issue URL and number, assignment ID, worker identity, status, `processed`, observation/update timestamps, report path, evidence, and PR URL if one actually exists. Owners update only their own `issues/NUMBER.json` and `reports/NUMBER.md` in the registry.

Allowed owner statuses are: **Pending**, **Assigned**, **Investigating**, **Fixed**, **Validated**, **PR Created**, **Resolved Without Code Change**, and **Blocked**. Preserve unknown execution states as Pending/not-yet-observed with observation timestamps. A dispatch request is not a durable claim, started investigation, successful test, fix, or PR.

Set `processed=true` only when terminal investigation/resolution documentation is durable. A documented final Blocked outcome is also skip-on-future-run; it is not success. Fixed or Validated alone is not permission to assume the PR/documentation stage is complete. Retain unsuccessful reproduction attempts and missing evidence honestly.

Records with `record_type=preexisting_processing` and `processed=true` are history-only skip markers. They retain a skip reason and source URL; they do not assert that this run reproduced, fixed, validated, merged, or resolved the issue. Create these markers only when absent and never overwrite an existing owner/history record.

## Branch and repository separation

- Upstream `pytorch/pytorch` is read-only: no comments, issues, PRs, or branch changes.
- The only writable repository is `fuliucansheng/pytorch`. Do not modify its `main` or existing code branches.
- This registry branch contains processing documentation only. Issue owners keep any newly authorized code/fix branches separate, based on the appropriate personal-main source rather than registry commits. Any required PR targets the personal fork, never upstream, and remains one issue per PR.
- Native-Windows reproduction and regression evidence belong to the assigned Windows owner, not to the coordination worker.

## Run a11c95d4-20261118T112856Z

The inclusive frozen window is `2026-11-16T11:28:56Z` through `2026-11-18T11:28:56Z`. Query: `repo:pytorch/pytorch is:issue created:2026-11-16T11:28:56Z..2026-11-18T11:28:56Z`.

The initial scan found 79 issues (74 open, five closed); six reported native-Windows failures. History-only skips are #205291 (existing upstream draft PR #205407) and #205495 (preexisting AI-assisted investigation/candidate fix in the issue body). Neither is reinvestigated by this run.

The Orchestrator has already requested these owners; this table does not assert that their claims or tests succeeded:

| Issue | Requested owner | Assignment |
|---|---|---|
| [205275](https://github.com/pytorch/pytorch/issues/205275) | DESKTOP-3O9Q5CE | a11c95d4-20261118-205275 |
| [205276](https://github.com/pytorch/pytorch/issues/205276) | DESKTOP-5IP0SN3 | a11c95d4-20261118-205276 |
| [205292](https://github.com/pytorch/pytorch/issues/205292) | DESKTOP-QBOG748 | a11c95d4-20261118-205292 |
| [205293](https://github.com/pytorch/pytorch/issues/205293) | ProArt5090 | a11c95d4-20261118-205293 |

Read `runs/a11c95d4-20261118T112856Z/manifest.json` for the audit, source URLs, observed statuses, timestamps, exclusions, and limitations. Statuses are point-in-time observations, not live monitoring. The Orchestrator consolidates the separate owners' actual results after asynchronous completion; this registry bootstrap is not their final resolution report.
