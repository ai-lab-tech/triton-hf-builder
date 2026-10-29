"""Fetch only official GitHub repositories; pin every document to a commit."""

import hashlib
import json
import os
import re
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOSITORIES = {"triton-inference-server/server", "vllm-project/vllm", "NVIDIA/TensorRT-LLM"}


class MissingSource(Exception):
    """The exact release or document is not published upstream."""


class GitHubSources:
    def __init__(self):
        self.commits = {}
        self.documents = {}

    def _read(self, url, *, api=False):
        headers = {"User-Agent": "triton-hf-builder-compatibility"}
        if api and os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                content = response.read(4 * 1024 * 1024 + 1)
        except HTTPError as error:
            if error.code == 404:
                raise MissingSource(url) from error
            raise
        if len(content) > 4 * 1024 * 1024:
            raise ValueError("Upstream document exceeds size limit")
        return content

    def document(self, repository, ref, path):
        if repository not in REPOSITORIES or ".." in path.split("/"):
            raise ValueError("Unapproved upstream source")
        key = (repository, ref, path if ref == "main" else "")
        if key not in self.commits:
            if ref == "main":
                response = self._read(
                    f"https://api.github.com/repos/{repository}/commits?sha=main&per_page=1&path={quote(path, safe='')}",
                    api=True,
                )
                commit = json.loads(response)[0]["sha"]
            else:
                response = self._read(
                    f"https://api.github.com/repos/{repository}/commits/{quote(ref, safe='')}",
                    api=True,
                )
                commit = json.loads(response)["sha"]
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("Invalid upstream commit")
            self.commits[key] = commit
        commit = self.commits[key]
        key = (repository, commit, path)
        if key not in self.documents:
            raw = self._read(f"https://raw.githubusercontent.com/{repository}/{commit}/{path}")
            self.documents[key] = (
                raw.decode("utf-8"),
                {
                    "url": f"https://github.com/{repository}/blob/{commit}/{path}",
                    "ref": ref,
                    "commit": commit,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
            )
        return self.documents[key]
