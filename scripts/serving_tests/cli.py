"""Download checkpoints, install each generated requirements file and test real Triton serving."""

import argparse
import importlib.util
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .checkpoints import CASES
from .runner import run_case
from .support import command, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", action="append", choices=("transformers", "vllm", "tensorrt-llm", "all")
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--work-dir", type=Path, default=Path(".e2e-artifacts"))
    parser.add_argument(
        "--gpu-device", default="0", help="Single GPU index exposed to test containers"
    )
    parser.add_argument("--triton-version", default="26.06")
    parser.add_argument("--python-base-image", help="Base image for the Transformers test image")
    parser.add_argument(
        "--transformers-image", help="Custom base image; each test still installs its requirements"
    )
    parser.add_argument("--vllm-image")
    parser.add_argument("--trtllm-image")
    parser.add_argument("--torch-version", default="2.9.0")
    parser.add_argument("--transformers-version", default="4.57.1")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--inference-timeout", type=int, default=300)
    parser.add_argument("--build-timeout", type=int, default=3600)
    parser.add_argument("--keep-containers", action="store_true")
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="REPO_ID=COMMIT",
        help="Pin a model revision; resolved commit SHAs are always saved",
    )
    args = parser.parse_args(argv)
    args.runtime = list(dict.fromkeys(args.runtime or ["transformers"]))
    args.case = list(dict.fromkeys(args.case)) if args.case else None
    if "all" in args.runtime:
        args.runtime = ["transformers", "vllm", "tensorrt-llm"]
    if args.device != "gpu" and any(runtime != "transformers" for runtime in args.runtime):
        parser.error("vLLM and TensorRT-LLM require --device gpu")
    if (
        any(runtime != "transformers" for runtime in args.runtime)
        and args.case
        and args.case != ["generation"]
    ):
        parser.error(
            "Use --case generation with engine runtimes, or omit --case for their defaults"
        )
    if not args.gpu_device.isdigit():
        parser.error("--gpu-device must be a single numeric GPU index")
    if min(args.startup_timeout, args.inference_timeout, args.build_timeout) < 1:
        parser.error("Timeouts must be positive")
    args.python_base_image = (
        args.python_base_image or f"nvcr.io/nvidia/tritonserver:{args.triton_version}-py3"
    )
    args.vllm_image = (
        args.vllm_image or f"nvcr.io/nvidia/tritonserver:{args.triton_version}-vllm-python-py3"
    )
    args.trtllm_image = (
        args.trtllm_image or f"nvcr.io/nvidia/tritonserver:{args.triton_version}-trtllm-python-py3"
    )
    return args


def main(argv=None):
    args = parse_args(argv)
    work_dir = args.work_dir.expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    run_dir = work_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    print(f"Artifacts: {run_dir}", flush=True)
    report = {"run_id": run_id, "status": "running", "results": []}
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    try:
        revisions = {}
        for value in args.revision:
            repo_id, revision = value.split("=", 1)
            if not repo_id or not revision:
                raise ValueError("Revision must have the form REPO_ID=COMMIT")
            revisions[repo_id] = revision
        if importlib.util.find_spec("huggingface_hub") is None:
            raise RuntimeError(
                "Missing huggingface_hub. Install with this Python: python -m pip install -e '.[e2e]', inside a virtual environment."
            )
        command(["docker", "info"], run_dir / "preflight.log", timeout=30)
        cache = work_dir / "hf-cache"
        cache.mkdir(exist_ok=True)
        for runtime in args.runtime:
            cases = args.case or (list(CASES) if runtime == "transformers" else ["generation"])
            base_image = {
                "transformers": args.transformers_image or args.python_base_image,
                "vllm": args.vllm_image,
                "tensorrt-llm": args.trtllm_image,
            }[runtime]
            for case in cases:
                print(f"RUN {runtime}/{case}", flush=True)
                result = {"runtime": runtime, "case": case, "status": "running"}
                report["results"].append(result)
                write_json(report_path, report)
                run_case(args, runtime, case, base_image, run_dir, cache, revisions, result=result)
                write_json(report_path, report)
                print(f"{result['status'].upper()} {runtime}/{case}", flush=True)
                if result.get("error"):
                    print(result["error"], file=sys.stderr, flush=True)
        failed = any(result["status"] != "passed" for result in report["results"])
        report["status"] = "failed" if failed or not report["results"] else "passed"
        return 1 if report["status"] == "failed" else 0
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Interrupted by user")
        return 130
    except Exception as error:
        report.update(status="failed", error=str(error))
        (run_dir / "failure.txt").write_text(traceback.format_exc())
        print(str(error), file=sys.stderr)
        return 1
    finally:
        write_json(report_path, report)
        print(f"Report: {report_path}", flush=True)
