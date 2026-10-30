# SPDX-License-Identifier: Apache-2.0
"""Standalone Triton Python adapter. No dependency on triton-hf-builder."""

import json
import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import pipeline

        version = Path(args["model_repository"]) / args["model_version"]
        self.settings = json.loads((version / "settings.json").read_text())
        self.task = self.settings["task"]
        device = (
            int(args["model_instance_device_id"]) if args["model_instance_kind"] == "GPU" else -1
        )
        self.pipeline = pipeline(
            task=self.task,
            model=str(version / "model"),
            tokenizer=str(version / "model"),
            framework="pt",
            device=device,
            trust_remote_code=False,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
                if tensor is None or tensor.as_numpy().shape != (1,):
                    raise ValueError("text_input must contain exactly one UTF-8 string, shape [1]")
                value = tensor.as_numpy()[0]
                if not isinstance(value, (bytes, str)):
                    raise ValueError("text_input must be a UTF-8 string")
                prompt = value.decode("utf-8") if isinstance(value, bytes) else value
                kwargs = {}
                if self.task in ("text-generation", "text2text-generation"):
                    kwargs = {"max_new_tokens": self.settings["max_new_tokens"], "do_sample": False}
                if self.task == "text-generation":
                    kwargs["return_full_text"] = False
                result = self.pipeline(prompt, **kwargs)
                text = (
                    result[0]["generated_text"]
                    if self.task in ("text-generation", "text2text-generation")
                    else json.dumps(result)
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "text_output", np.array([text.encode("utf-8")], dtype=object)
                            )
                        ]
                    )
                )
            except Exception as error:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error))))
        return responses

    def finalize(self):
        self.pipeline = None
