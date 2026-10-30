# SPDX-License-Identifier: Apache-2.0
"""Send one Triton HTTP request to the CPU text-generation demo."""

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def infer(base_url: str, prompt: str) -> str:
    payload = {
        "inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [prompt]}],
        "outputs": [{"name": "text_output"}],
    }
    request = Request(
        base_url.rstrip("/") + "/v2/models/demo/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    outputs = [output for output in result.get("outputs", []) if output["name"] == "text_output"]
    if len(outputs) != 1 or len(outputs[0].get("data", [])) != 1:
        raise ValueError("Triton returned an unexpected text_output response")
    completion = outputs[0]["data"][0]
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("Triton returned an empty completion")
    return completion


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args(argv)
    try:
        completion = infer(args.url, args.prompt)
    except (HTTPError, URLError, OSError, ValueError) as error:
        parser.exit(1, f"Inference failed: {error}\n")
    print(
        json.dumps({"prompt": args.prompt, "completion": completion}, indent=2, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
