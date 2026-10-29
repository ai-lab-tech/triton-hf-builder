"""Stable JSON CLI boundary for editor and deployment integrations."""

import argparse
import json
import signal
import sys

from ._version import __version__
from .builder import generate_repository
from .inspector import TASKS, inspect_model
from .runtimes import RUNTIMES


def main(argv=None):
    parser = argparse.ArgumentParser(prog="triton-hf")
    parser.add_argument("--version", action="version", version=f"triton-hf-builder {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read local metadata without loading weights")
    inspect.add_argument("source")
    build = commands.add_parser("build", help="Generate a new Triton model repository")
    build.add_argument("source")
    build.add_argument("--output", required=True)
    build.add_argument("--name", required=True)
    build.add_argument("--runtime", choices=RUNTIMES, required=True)
    build.add_argument(
        "--image", required=True, help="Target serving image with explicit tag/digest"
    )
    build.add_argument("--task", choices=TASKS)
    build.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    build.add_argument("--requirement", action="append", default=[], dest="requirements")
    build.add_argument("--tensor-parallel-size", type=int, default=1)
    build.add_argument("--max-new-tokens", type=int, default=128)
    build.add_argument("--max-model-len", type=int)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    # The editor cancels with SIGTERM. Unwind the temporary-directory context so
    # a cancelled multi-GB checkpoint copy does not leave assets behind.
    previous_handler = signal.signal(signal.SIGTERM, _cancel)
    try:
        result = inspect_model(**args) if command == "inspect" else generate_repository(**args)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(json.dumps({"error": "Generation cancelled"}), file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _cancel(_signal, _frame):
    raise KeyboardInterrupt
