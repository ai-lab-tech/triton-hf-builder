#!/usr/bin/env python3
"""Download real HF checkpoints, generate repositories, and test Docker Triton serving.

This is opt-in integration testing. It is never imported by the unit-test suite.
Run from an environment containing ``triton-hf-builder[e2e]``. All model loading
and reference inference happen inside serving containers, not in this process.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CASES = {
    "generation": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "task": "text-generation",
        "prompts": ["The capital of France is"],
    },
    "classification": {
        "repo_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "task": "text-classification",
        "prompts": ["I absolutely loved this movie.", "This movie was awful and boring."],
        "labels": ["POSITIVE", "NEGATIVE"],
    },
    "text2text": {
        "repo_id": "google/flan-t5-small",
        "task": "text2text-generation",
        "prompts": ["translate English to German: The house is wonderful."],
    },
    "features": {
        "repo_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "task": "feature-extraction",
        "prompts": ["A small serving test."],
    },
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command(arguments, log, *, capture=False, timeout=1800):
    """Run argument arrays, saving stdout/stderr and preserving useful failures."""
    with log.open("a", encoding="utf-8") as output:
        output.write("\nCOMMAND " + json.dumps([str(arg) for arg in arguments]) + "\n")
        output.flush()
        result = subprocess.run(
            [str(arg) for arg in arguments],
            stdout=subprocess.PIPE if capture else output,
            stderr=output,
            text=True,
            timeout=timeout,
            check=False,
        )
        if capture:
            output.write(result.stdout)
    if result.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Command exited {result.returncode}; see {log}\n{tail}")
    return result.stdout if capture else ""


def request_json(base_url, route, payload=None, *, timeout=180):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url + route,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
            return response.status, json.loads(data) if data else None
    except HTTPError as error:
        data = error.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(data)
        except ValueError:
            pass
        return error.code, data


def require(condition, message):
    # Explicit checks are active even when Python is invoked with -O.
    if not condition:
        raise AssertionError(message)


def download_checkpoint(repo_id, cache, revisions):
    from huggingface_hub import HfApi, snapshot_download

    revision = revisions.get(repo_id, "main")
    info = HfApi().model_info(repo_id, revision=revision)
    revisions.setdefault(repo_id, info.sha)
    names = [item.rfilename for item in info.siblings]
    safe_weights = [name for name in names if "/" not in name and name.endswith(".safetensors")]
    weights = safe_weights or [
        name
        for name in names
        if "/" not in name and name.startswith("pytorch_model") and name.endswith(".bin")
    ]
    if not weights:
        raise ValueError(f"No supported root checkpoint weights in {repo_id}@{info.sha}")
    suffixes = {".json", ".txt", ".model", ".tiktoken", ".jinja", ".jinja2"}
    metadata = [
        name
        for name in names
        if ("/" not in name or name.startswith("chat_templates/"))
        and Path(name).suffix in suffixes
        and not (safe_weights and name.endswith(".bin.index.json"))
        and not (not safe_weights and name.endswith(".safetensors.index.json"))
    ]
    path = snapshot_download(
        repo_id, revision=info.sha, cache_dir=str(cache), allow_patterns=[*weights, *metadata]
    )
    return Path(path), info.sha


def image_details(image, log):
    details = json.loads(command(["docker", "image", "inspect", image], log, capture=True))[0]
    return {"reference": image, "id": details["Id"], "digests": details.get("RepoDigests", [])}


def build_serving_image(args, runtime, repository, image, base_image, directory, log):
    context = directory / "image-build"
    context.mkdir()
    shutil.copy2(repository / "requirements.txt", context / "requirements.txt")
    shutil.copy2(
        Path(__file__).with_name("verify_requirements.py"), context / "verify_requirements.py"
    )
    index_option = '--extra-index-url "${TORCH_INDEX_URL}" ' if runtime == "transformers" else ""
    (context / "Dockerfile").write_text(
        "ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.06-py3\nFROM ${BASE_IMAGE}\n"
        "RUN --mount=type=cache,id=triton-hf-builder-pip,target=/root/.cache/pip,sharing=locked "
        "python3 -m pip --timeout 180 --retries 10 install --no-deps --ignore-installed "
        '--target /opt/triton-hf-builder-pip "pip==25.2"\n'
        "ARG TORCH_INDEX_URL\n"
        "COPY requirements.txt /tmp/e2e-requirements.txt\n"
        "COPY verify_requirements.py /tmp/e2e-verify-requirements.py\n"
        "ARG E2E_CASE_ID\n"
        "RUN --mount=type=cache,id=triton-hf-builder-pip,target=/root/.cache/pip,sharing=locked "
        'test -n "${E2E_CASE_ID}" && '
        "PYTHONPATH=/opt/triton-hf-builder-pip "
        "python3 -m pip --timeout 180 --retries 10 --resume-retries 10 install "
        "--report /tmp/e2e-install-report.json " + index_option + "-r /tmp/e2e-requirements.txt && "
        "PYTHONPATH=/opt/triton-hf-builder-pip python3 /tmp/e2e-verify-requirements.py "
        "/tmp/e2e-requirements.txt /tmp/e2e-requirements-verified.json\n"
    )
    index = "https://download.pytorch.org/whl/" + ("cpu" if args.device == "cpu" else "cu130")
    command(
        [
            "docker",
            "build",
            "--tag",
            image,
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--build-arg",
            f"TORCH_INDEX_URL={index}",
            "--build-arg",
            f"E2E_CASE_ID={directory.parent.name}/{runtime}/{directory.name}",
            context,
        ],
        log,
        timeout=args.build_timeout,
    )


def wait_ready(container, model_name, base_url, log, timeout):
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            server, _ = request_json(base_url, "/v2/health/ready", timeout=2)
            model, _ = request_json(base_url, f"/v2/models/{model_name}/ready", timeout=2)
            if server == model == 200:
                return
            last_error = f"server={server}, model={model}"
        except (OSError, URLError) as error:
            last_error = str(error)
        state = command(
            ["docker", "inspect", "--format", "{{.State.Running}}", container], log, capture=True
        ).strip()
        if state != "true":
            raise RuntimeError("Triton exited before becoming ready; see triton.log")
        time.sleep(2)
    raise TimeoutError(f"Triton/model not ready after {timeout}s ({last_error}); see triton.log")


def python_inference(base_url, model_name, prompt, timeout):
    payload = {
        "inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [prompt]}]
    }
    status, response = request_json(
        base_url, f"/v2/models/{model_name}/infer", payload, timeout=timeout
    )
    require(status == 200, f"Inference returned HTTP {status}: {response}")
    outputs = {output["name"]: output for output in response.get("outputs", [])}
    require("text_output" in outputs, f"Missing text_output: {response}")
    output = outputs["text_output"]
    require(output["datatype"] == "BYTES" and output["shape"] == [1], "Wrong output contract")
    require(len(output["data"]) == 1 and isinstance(output["data"][0], str), "Invalid text output")
    return output["data"][0], {"request": payload, "response": response}


def close_values(actual, expected):
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(close_values(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(close_values(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-5)
        )
    return actual == expected


def check_inference(args, case, runtime, model_name, container, base_url, directory, log):
    observations = []
    for index, prompt in enumerate(case["prompts"]):
        # Arrange
        reference = None
        if runtime == "transformers":
            reference = json.loads(
                command(
                    [
                        "docker",
                        "exec",
                        container,
                        "python3",
                        "/e2e/reference_inference.py",
                        "--model",
                        f"/models/{model_name}/1/model",
                        "--task",
                        case["task"],
                        "--prompt",
                        prompt,
                        "--device",
                        args.device,
                        "--max-new-tokens",
                        "16",
                    ],
                    log,
                    capture=True,
                    timeout=args.inference_timeout,
                )
            )

        # Act
        if runtime == "vllm":
            payload = {
                "text_input": prompt,
                "stream": False,
                "exclude_input_in_output": True,
                "parameters": {"temperature": 0.0, "max_tokens": 16},
            }
            status, response = request_json(
                base_url,
                f"/v2/models/{model_name}/generate",
                payload,
                timeout=args.inference_timeout,
            )
            record = {"request": payload, "status": status, "response": response}
            actual = response.get("text_output") if isinstance(response, dict) else None
            if isinstance(actual, list) and len(actual) == 1:
                actual = actual[0]
        else:
            actual, record = python_inference(base_url, model_name, prompt, args.inference_timeout)
            status = 200
        # Persist responses even when an assertion below fails.
        record["reference"] = reference
        write_json(directory / f"inference-{index}.json", record)

        # Assert
        require(status == 200, f"Inference returned HTTP {status}: {record}")
        require(isinstance(actual, str), "Completion is not a string")
        if case["task"] in ("text-generation", "text2text-generation"):
            require(bool(actual.strip()), "Model returned an empty completion")
        if reference is not None:
            actual_value = (
                json.loads(actual)
                if case["task"] in ("text-classification", "feature-extraction")
                else actual
            )
            require(
                close_values(actual_value, reference["output"]),
                "Triton output differs from direct Transformers inference; see inference JSON",
            )
        if case.get("labels"):
            classification = json.loads(actual)[0]
            require(classification["label"] == case["labels"][index], "Wrong sentiment label")
            require(classification["score"] > 0.9, "Low sentiment confidence")
        observations.append({"prompt": prompt, "reference_comparison": reference is not None})

    if runtime != "vllm":
        # Arrange
        malformed = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [2],
                    "datatype": "BYTES",
                    "data": ["first", "second"],
                }
            ]
        }

        # Act
        status, response = request_json(
            base_url,
            f"/v2/models/{model_name}/infer",
            malformed,
            timeout=args.inference_timeout,
        )
        write_json(directory / "invalid-request.json", {"status": status, "response": response})

        # Assert
        require(400 <= status < 500, f"Malformed request was not rejected: {status}")
    return observations


def run_case(args, runtime, case_name, base_image, run_dir, cache, revisions):
    directory = run_dir / f"{runtime}-{case_name}"
    directory.mkdir()
    log = directory / "commands.log"
    result = {"runtime": runtime, "case": case_name, "status": "failed"}
    container = None
    started = time.monotonic()
    image = f"triton-hf-builder-e2e:{run_dir.name.lower()}-{runtime}-{case_name}"
    try:
        # Arrange: download exact checkpoint, build artifacts, prepare serving image.
        case = CASES[case_name]
        source, revision = download_checkpoint(case["repo_id"], cache, revisions)
        result.update(repo_id=case["repo_id"], revision=revision)
        model_name = f"hf_{case_name}"
        repository = directory / "repository"
        arguments = [
            sys.executable,
            "-m",
            "triton_hf_builder",
            "build",
            str(source),
            "--output",
            str(repository),
            "--name",
            model_name,
            "--runtime",
            runtime,
            "--task",
            case["task"],
            "--device",
            args.device,
            "--image",
            image,
        ]
        if runtime == "vllm":
            arguments += ["--max-model-len", "512"]
        else:
            arguments += ["--max-new-tokens", "16"]
        if runtime == "transformers":
            torch_version = args.torch_version
            if "+" not in torch_version:
                torch_version += "+cpu" if args.device == "cpu" else "+cu130"
            for requirement in (
                f"torch=={torch_version}",
                f"transformers=={args.transformers_version}",
                "numpy==1.26.4",
                "safetensors==0.6.2",
                "sentencepiece==0.2.1",
                "protobuf==5.29.5",
            ):
                arguments += ["--requirement", requirement]
        command(arguments, log)
        requirements_path = repository / "requirements.txt"
        result["requirements"] = {
            "file": str(requirements_path),
            "sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
            "status": "installing",
        }
        # Every case installs its own generated file, including engine extras.
        # E2E_CASE_ID invalidates this installation layer, while downloads are cached.
        build_serving_image(args, runtime, repository, image, base_image, directory, log)
        result["requirements"]["status"] = "installed_and_verified"
        result["base_image"] = image_details(base_image, log)
        result["image"] = image_details(image, log)
        result["repository"] = str(repository)
        container = "hf-builder-e2e-" + uuid.uuid4().hex[:12]
        docker_args = [
            "docker",
            "run",
            "--detach",
            "--name",
            container,
            "--label",
            f"triton-hf-builder.e2e={run_dir.name}",
            "--publish",
            "127.0.0.1::8000",
            "--shm-size",
            "2g",
            "--volume",
            f"{repository}:/models",
            "--volume",
            f"{Path(__file__).resolve().parent}:/e2e:ro",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "TOKENIZERS_PARALLELISM=false",
            "--env",
            "OMP_NUM_THREADS=2",
        ]
        if args.device == "gpu":
            docker_args += ["--gpus", f"device={args.gpu_device}"]
        startup = ""
        if runtime == "vllm":
            startup = "python3 /models/prepare_repository.py && "
        startup += "exec tritonserver --model-repository=/models --exit-on-error=true"
        docker_args += ["--entrypoint", "/bin/bash", image, "-lc", startup]

        # Act: start a real server, wait for model readiness, then invoke real inference.
        command(docker_args, log)
        for filename in ("e2e-install-report.json", "e2e-requirements-verified.json"):
            command(["docker", "cp", f"{container}:/tmp/{filename}", directory / filename], log)
        verification = json.loads((directory / "e2e-requirements-verified.json").read_text())
        require(
            verification["sha256"] == result["requirements"]["sha256"],
            "Serving image installed a different requirements.txt",
        )
        result["requirements"]["packages"] = verification["packages"]
        bindings = json.loads(
            command(
                ["docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", container],
                log,
                capture=True,
            )
        )
        port = bindings["8000/tcp"][0]["HostPort"]
        base_url = f"http://127.0.0.1:{port}"
        result.update(container=container, url=base_url)
        wait_ready(container, model_name, base_url, log, args.startup_timeout)
        _, metadata = request_json(base_url, f"/v2/models/{model_name}")
        write_json(directory / "model-metadata.json", metadata)
        environment = command(
            [
                "docker",
                "exec",
                container,
                "python3",
                "/e2e/reference_inference.py",
                "--environment",
            ],
            log,
            capture=True,
        )
        result["environment"] = json.loads(environment)
        observations = check_inference(
            args, case, runtime, model_name, container, base_url, directory, log
        )

        # Assert: success requires serving and inference checks, not just generated files.
        require(bool(observations), "No inference requests were verified")
        result.update(status="passed", observations=observations)
        manifest_path = repository / "serving.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["validation"]["serving"] = "passed"
        manifest["serving_evidence"] = {
            "report": str(directory / "result.json"),
            "revision": revision,
            "image_id": result["image"]["id"],
        }
        write_json(manifest_path, manifest)
    except KeyboardInterrupt:
        result.update(status="interrupted", error="Interrupted by user")
        raise
    except Exception as error:
        if result.get("requirements", {}).get("status") == "installing":
            result["requirements"]["status"] = "failed"
        result["error"] = str(error)
        (directory / "failure.txt").write_text(traceback.format_exc())
    finally:
        if container is not None:
            try:
                command(["docker", "logs", container], directory / "triton.log", timeout=30)
            except Exception as error:
                result["log_collection_error"] = str(error)
            if not args.keep_containers:
                try:
                    command(["docker", "rm", "--force", container], log, timeout=60)
                except Exception as error:
                    result["cleanup_error"] = str(error)
                    result["status"] = "failed"
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(directory / "result.json", result)
    return result


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
                result = run_case(args, runtime, case, base_image, run_dir, cache, revisions)
                report["results"].append(result)
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


if __name__ == "__main__":
    raise SystemExit(main())
