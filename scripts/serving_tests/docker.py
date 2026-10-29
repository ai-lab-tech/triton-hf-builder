import json
import shutil
import time
from pathlib import Path
from urllib.error import URLError

from .support import command, request_json


def image_details(image, log):
    details = json.loads(command(["docker", "image", "inspect", image], log, capture=True))[0]
    return {"reference": image, "id": details["Id"], "digests": details.get("RepoDigests", [])}


def base_image_details(image, log):
    """BuildKit may cache FROM layers without registering a local image tag.

    A missing base tag is not a build or serving failure. Keep the requested
    reference and explicitly leave its identity unresolved. The derived serving
    image is still inspected strictly by the runner.
    """
    try:
        return image_details(image, log)
    except RuntimeError as error:
        if f"No such image: {image}" not in str(error):
            raise
        return {
            "reference": image,
            "id": None,
            "digests": [],
            "inspection_status": "not_in_local_store",
        }


def build_serving_image(args, runtime, repository, image, base_image, directory, log):
    context = directory / "image-build"
    context.mkdir()
    shutil.copy2(repository / "requirements.txt", context / "requirements.txt")
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "verify_requirements.py",
        context / "verify_requirements.py",
    )
    index_option = '--extra-index-url "${TORCH_INDEX_URL}" ' if runtime == "transformers" else ""
    (context / "Dockerfile").write_text(
        Path(__file__)
        .with_name("Dockerfile.serving")
        .read_text(encoding="utf-8")
        .replace("@INDEX_OPTION@", index_option),
        encoding="utf-8",
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
