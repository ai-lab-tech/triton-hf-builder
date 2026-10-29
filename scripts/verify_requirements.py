"""Verify the generated requirement specifications in the actual serving image."""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement


def verify(path):
    content = path.read_bytes()
    packages = []
    for line in content.decode("utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        # A missing package or an incompatible version fails the image build.
        version = importlib.metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"{requirement}: installed version is {version}")
        packages.append({"requirement": str(requirement), "version": version})
    return {"sha256": hashlib.sha256(content).hexdigest(), "packages": packages}


if __name__ == "__main__":
    result = verify(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
