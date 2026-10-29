# Security policy

Do not put vulnerabilities, credentials or private checkpoints in public issues.
Use [GitHub private vulnerability reporting](https://github.com/ai-lab-tech/triton-hf-builder/security/advisories/new)
when enabled. Otherwise contact a repository maintainer privately through their
GitHub profile before sharing details. No response SLA is promised. Fixes target
the current development line and latest release; older backports are not guaranteed.

Include the package version, affected workflow, impact and a minimal reproduction
with secrets removed.

## Trust boundaries

The builder reads metadata and copies checkpoint files; it does not load weights
or execute custom model Python. This does not certify that a checkpoint is safe
to load. Serving runs third-party engines and loads serialization formats that
may execute code. Use trusted models, dependencies and images in an isolated
serving environment. Generated adapters disable `trust_remote_code`.

Checkpoint assets, prompts, responses and local paths can appear in generated
repositories or serving-test reports. Review them before sharing. Do not place
credentials in model metadata. The opt-in test runner controls Docker and installs
requirements; only run it on an appropriate development host.
