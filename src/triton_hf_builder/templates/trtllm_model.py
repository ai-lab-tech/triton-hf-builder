"""Standalone TensorRT-LLM public LLM API adapter using local HF weights."""

import json
import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from tensorrt_llm import LLM, SamplingParams

        version = Path(args["model_repository"]) / args["model_version"]
        settings = json.loads((version / "settings.json").read_text())
        self.sampling = SamplingParams(max_tokens=settings["max_new_tokens"], temperature=0.0)
        self.llm = LLM(
            model=str(version / "model"),
            backend="pytorch",
            trust_remote_code=False,
            tensor_parallel_size=settings["tensor_parallel_size"],
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
                outputs = self.llm.generate([prompt], sampling_params=self.sampling)
                if not outputs or not outputs[0].outputs:
                    raise RuntimeError("LLM returned no completion")
                text = outputs[0].outputs[0].text
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
        shutdown = getattr(self.llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.llm = None
