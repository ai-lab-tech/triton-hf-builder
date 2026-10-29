"""Run an independent real Transformers reference inside the serving container."""

import argparse
import importlib.metadata
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--task")
    parser.add_argument("--prompt")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.environment:
        versions = {}
        for package in ("torch", "transformers", "numpy", "vllm", "tensorrt-llm", "sentencepiece"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        print(json.dumps({"python": sys.version, "packages": versions}))
        return

    # Arrange: load the same real checkpoint directly, independently of model.py.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import pipeline

    predictor = pipeline(
        task=args.task,
        model=args.model,
        tokenizer=args.model,
        framework="pt",
        device=0 if args.device == "gpu" else -1,
        trust_remote_code=False,
    )
    options = {}
    if args.task in ("text-generation", "text2text-generation"):
        options.update(max_new_tokens=args.max_new_tokens, do_sample=False)
    if args.task == "text-generation":
        options["return_full_text"] = False

    # Act
    result = predictor(args.prompt, **options)
    output = (
        result[0]["generated_text"]
        if args.task in ("text-generation", "text2text-generation")
        else result
    )

    # Assert: the host runner compares this independent result to Triton inference.
    print(json.dumps({"output": output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
