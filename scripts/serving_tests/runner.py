import hashlib
import json
import sys
import time
import traceback
import uuid
from pathlib import Path

from .checkpoints import CASES, download_checkpoint
from .docker import base_image_details, build_serving_image, image_details, wait_ready
from .inference import check_inference
from .support import command, request_json, require, write_json


def run_case(args, runtime, case_name, base_image, run_dir, cache, revisions, *, result=None):
    directory = run_dir / f"{runtime}-{case_name}"
    directory.mkdir()
    log = directory / "commands.log"
    if result is None:
        result = {"runtime": runtime, "case": case_name}
    result["status"] = "failed"
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
        result["base_image"] = base_image_details(base_image, log)
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
            f"{Path(__file__).resolve().parents[1]}:/e2e:ro",
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
        status, metadata = request_json(base_url, f"/v2/models/{model_name}")
        write_json(directory / "model-metadata.json", metadata)
        require(status == 200, f"Model metadata returned HTTP {status}: {metadata}")
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
        if result["status"] == "passed":
            try:
                manifest_path = repository / "serving.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["validation"]["serving"] = "passed"
                manifest["serving_evidence"] = {
                    "report": str(directory / "result.json"),
                    "revision": revision,
                    "image_id": result["image"]["id"],
                }
                write_json(manifest_path, manifest)
            except Exception as error:
                result.update(status="failed", error=f"Cannot save serving evidence: {error}")
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(directory / "result.json", result)
    return result
