"""Set CI release versions from tags and verify source or installed package versions."""

import argparse
import ast
from importlib import metadata
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def validate_version(value: str, tag: str | None = None) -> str:
    version = Version(value)
    if str(version) != value or version.local is not None or version.is_devrelease:
        raise ValueError("Use a canonical public release version without local/dev suffixes")
    if tag is not None and tag != f"v{value}":
        raise ValueError(f"Release tag must be v{value}, got {tag!r}")
    return value


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
    return validate_version(values[0], tag)


def write_version_from_tag(path: Path, tag: str) -> str:
    if not isinstance(tag, str) or not tag.startswith("v"):
        raise ValueError("Release tags must start with 'v', for example v0.1.5")
    version = validate_version(tag[1:], tag)
    path.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    return version


def check_installed_version(tag: str | None = None) -> str:
    from triton_hf_builder import __version__

    version = validate_version(metadata.version("triton-hf-builder"), tag)
    if __version__ != version:
        raise ValueError(
            f"Runtime version {__version__!r} differs from package metadata {version!r}"
        )
    return version


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Set the build version from --tag")
    mode.add_argument(
        "--installed", action="store_true", help="Check the installed package version"
    )
    args = parser.parse_args(argv)
    if args.write and args.tag is None:
        parser.error("--write requires --tag")
    path = ROOT / "src/triton_hf_builder/_version.py"
    try:
        if args.write:
            version = write_version_from_tag(path, args.tag)
        elif args.installed:
            version = check_installed_version(args.tag)
        else:
            version = check_version(path.read_text(encoding="utf-8"), args.tag)
    except (ValueError, SyntaxError, OSError, metadata.PackageNotFoundError) as error:
        parser.error(str(error))
    print(f"Release version: {version}")


if __name__ == "__main__":
    main()
