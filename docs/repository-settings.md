# Repository settings for public access

The reference is [ai-lab-tech/triton-control](https://github.com/ai-lab-tech/triton-control).
The settings below were checked through GitHub's public API on 2026-10-30.
They are a concrete preparation plan for `ai-lab-tech/triton-hf-builder`.
This document does not change GitHub settings or make the repository public.

## Repository features

| Setting | Observed on triton-control | Planned for triton-hf-builder |
| --- | --- | --- |
| Default branch | `main` | `main` |
| Issues | Enabled | Enabled |
| Discussions | Enabled | Enabled |
| Projects | Enabled | Enabled |
| Wiki | Disabled | Disabled |
| Template repository | Disabled | Disabled |
| Web commit sign-off | Not required | Not required |
| Homepage | `https://ailabtech.net/` | `https://pypi.org/project/triton-hf-builder/` |

Use the builder's own description and topics rather than copying the control
plane's Kubernetes and web-UI description:

- Description: `Build Triton model repositories from local Hugging Face checkpoints`.
- Topics: `python`, `triton`, `nvidia-triton`, `huggingface`, `model-serving`,
  `inference-server`, `model-repository`, `pypi`.

## Main branch rules

The reference has an active repository ruleset named `Protected main branch`,
targeting `refs/heads/main`. Its publicly visible rules are:

- Prevent branch deletion.
- Prevent non-fast-forward updates, including force pushes.
- Require a pull request with one approving review.
- Dismiss stale approvals when new commits are pushed.
- Do not require code-owner review or approval of the last push.
- Do not require every review thread to be resolved.
- Permit merge, squash and rebase in the ruleset.

The visible ruleset does not list required CI status checks. The builder's
existing build and Python test jobs continue to run for pull requests, and the
release pipeline continues to require successful checks.

The ruleset is available at
[triton-control main rules](https://github.com/ai-lab-tech/triton-control/rules/17212820).
Administrator and integration bypass actors cannot be confirmed from the
unauthenticated response. Check them with authenticated access before applying
the corresponding rule to the builder. Do not infer repository-level merge
method switches from the merge methods allowed by this ruleset.

## Community files

The builder now has equivalent project-specific community documents:

- `CONTRIBUTING.md`: contributions through PRs and maintainer review.
- `CODE_OF_CONDUCT.md`: participation rules and private conduct reporting.
- `SECURITY.md`: private vulnerability reporting and supported fixes.
- Bug-report, feature-request and PR templates.
- `.github/ISSUE_TEMPLATE/config.yml`: blank issues disabled; a private security
  reporting contact link, with the fallback described in `SECURITY.md`.

## Settings still requiring authenticated verification

These live administration settings have not been copied or verified on the
private builder repository:

- Repository feature toggles and topics listed above.
- Ruleset configuration and bypass actors.
- Repository-level merge preferences and automatic branch deletion.
- GitHub Actions default permissions and external-contributor approval policy.
- Repository access, teams, environments and release restrictions.
- Security analysis and secret protection settings.
- Historical Actions logs and artifacts that become visible with the public
  repository.

The public API reports that private vulnerability reporting is **disabled** on
`triton-control`. This is an observed setting, not evidence that reporting is
enabled on the builder. The builder's live state still needs an authenticated
read; `SECURITY.md` provides a fallback while private reporting is unavailable.

Dependabot version-update PRs remain disabled in the builder, as requested.
Do not re-enable them when copying other security settings. Preserve the
builder's existing `pypi` environment, Trusted Publisher and `PUBLISH_PYPI`
repository variable; they are specific to this package's publishing pipeline.

Keep the builder private while applying and verifying its configuration.
Changing visibility to public is a separate final action after reviewing the
configuration and repository history.
