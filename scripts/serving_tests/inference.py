import json
import math

from .support import command, request_json, require, write_json


def python_inference(base_url, model_name, prompt, timeout, record_path, reference=None):
    payload = {
        "inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [prompt]}]
    }
    status, response = request_json(
        base_url, f"/v2/models/{model_name}/infer", payload, timeout=timeout
    )
    record = {"request": payload, "status": status, "response": response, "reference": reference}
    write_json(record_path, record)
    require(status == 200, f"Inference returned HTTP {status}: {response}")
    outputs = {output["name"]: output for output in response.get("outputs", [])}
    require("text_output" in outputs, f"Missing text_output: {response}")
    output = outputs["text_output"]
    require(output["datatype"] == "BYTES" and output["shape"] == [1], "Wrong output contract")
    require(len(output["data"]) == 1 and isinstance(output["data"][0], str), "Invalid text output")
    return output["data"][0], record


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
            and not isinstance(actual, bool)
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
            actual, record = python_inference(
                base_url,
                model_name,
                prompt,
                args.inference_timeout,
                directory / f"inference-{index}.json",
                reference,
            )
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
