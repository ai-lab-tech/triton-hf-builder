# Releases

The package is currently beta. A release packages generation support; it does
not certify every model, engine, GPU or CUDA combination. Record real serving
evidence separately, including image digests, model revisions and failed cases.

## Prepare

Commit and push the code you want to release, then create and push an unused tag:

```bash
git tag -a v0.1.5 -m "Release v0.1.5"
git push origin v0.1.5
```

The tag sets the version automatically. Once checks pass, GitHub Actions
publishes the package to PyPI and creates a GitHub Release, provided publishing
is enabled as described below. Use a new version for each release.

## Enable PyPI once ready

Publication is disabled unless the GitHub repository variable `PUBLISH_PYPI`
equals `true`. Configure these external settings before enabling it:

- Choose an available PyPI project name matching the package metadata.
- Register a PyPI Trusted Publisher for owner `ai-lab-tech`, repository
  `triton-hf-builder`, workflow `tests.yml`, environment `pypi`.
- Create the GitHub `pypi` environment and configure release reviewers and tag
  restrictions. These settings are not created by the repository files.
- Enable private vulnerability reporting and branch protection as appropriate.

The publish job runs only on a pushed `v*` tag, after build and wheel tests pass.
It uses OIDC; no persistent PyPI token is needed. Follow the
[PyPI Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
Do not overwrite a published version; publish a new version for corrections.

After PyPI publication succeeds, the release job creates a GitHub Release for
the same existing tag, generates release notes, and attaches the tested wheel
and source archive. It uses GitHub's automatic workflow token; no additional
secret is needed. A failed build, test, or PyPI upload prevents the GitHub Release
from being created. The release job does not run when PyPI publishing is disabled.

Pushing a tag alone does not create a GitHub Release outside this workflow.
Tags pushed before the release job was added do not gain a release retroactively.

## code-server integration

This repository produces the Python package. `triton-control` owns the extension,
code-server Docker image and Helm deployment. Its image pipeline should install
a pinned builder release in a virtual environment and configure
`tritonControlDeploy.hfBuilderPython` to that interpreter. Test the installed CLI
before publishing that image to Docker Hub. Helm then deploys the immutable image
version or digest. Do not install the builder on every pod startup.

Before PyPI publication, the downstream image build can consume a reviewed wheel
artifact. The builder is not required in Triton serving images.
