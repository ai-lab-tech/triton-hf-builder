"""Validate release versions without importing inference engines."""

import argparse
import ast
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def check_version(source: str, tag: str | None = None) -> str:
    tree = ast.parse(source)
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Define exactly one literal __version__ string")
    value = values[0]
    version = Version(value)
    if str(version) != value or version.local is not None or version.is_devrelease:
        raise ValueError("Use a canonical public release version without local/dev suffixes")
    if tag is not None and tag != f"v{value}":
        raise ValueError(f"Release tag must be v{value}, got {tag!r}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    try:
        version = check_version(
            (ROOT / "src/triton_hf_builder/_version.py").read_text(encoding="utf-8"), args.tag
        )
    except (ValueError, SyntaxError) as error:
        parser.error(str(error))
    print(f"Release version: {version}")


if __name__ == "__main__":
    main()
