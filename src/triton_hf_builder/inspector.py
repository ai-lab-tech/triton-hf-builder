"""Read local checkpoint metadata. Never import or execute model code."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

TASKS = ("text-generation", "text2text-generation", "text-classification", "feature-extraction")


@dataclass(frozen=True)
class ModelInfo:
    source: Path
    model_type: str
    architectures: tuple[str, ...]
    inferred_task: str | None
    weight_files: tuple[str, ...]
    tokenizer_present: bool
    custom_code: bool
    quantization: str | None
    transformers_version: str | None
    warnings: tuple[str, ...]
    multimodal: bool = False

    def to_dict(self):
        return {**asdict(self), "source": str(self.source)}


def read_json(path: Path) -> dict:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError(f"Metadata exceeds 8 MiB: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid JSON metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def inspect_model(source: str | Path) -> ModelInfo:
    source = Path(source).expanduser().resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("Source must be a local HF directory containing config.json")
    config = read_json(source / "config.json")
    architectures = config.get("architectures")
    if architectures is None:
        architectures = []
    if not isinstance(architectures, list) or not all(isinstance(a, str) for a in architectures):
        raise ValueError("config.json architectures must be a list of strings")
    weights = set()
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        weights.update(
            p.name for p in source.glob(pattern) if p.is_file() and not p.name.startswith(".")
        )
    for pattern in ("*.safetensors.index.json", "pytorch_model*.bin.index.json"):
        for index in source.glob(pattern):
            weight_map = read_json(index).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Missing weight_map in {index.name}")
            for shard in weight_map.values():
                if not isinstance(shard, str):
                    raise ValueError(f"Invalid shard name in {index.name}")
                parts = PurePosixPath(shard)
                if (
                    parts.is_absolute()
                    or PureWindowsPath(shard).drive
                    or any(part.startswith(".") for part in parts.parts)
                    or "\\" in shard
                    or "\x00" in shard
                ):
                    raise ValueError(f"Unsafe shard name in {index.name}")
                suffix = (
                    ".safetensors" if index.name.endswith(".safetensors.index.json") else ".bin"
                )
                if parts.suffix != suffix:
                    raise ValueError(f"Checkpoint shard must end with {suffix}: {shard}")
                if not (source / shard).is_file():
                    raise ValueError(f"Missing checkpoint shard: {shard}")
                weights.add(shard)
    if not weights:
        raise ValueError("No HF safetensors or PyTorch checkpoint weights found")

    tokenizer_present = any(
        (source / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt", "vocab.json")
    )
    tokenizer_config = (
        read_json(source / "tokenizer_config.json")
        if (source / "tokenizer_config.json").is_file()
        else {}
    )
    inferred = None
    if any(a.endswith("ForCausalLM") or a == "GPT2LMHeadModel" for a in architectures):
        inferred = "text-generation"
    elif any(a.endswith(("ForConditionalGeneration", "ForSeq2SeqLM")) for a in architectures):
        # Multimodal conditional generation also uses this suffix. Require explicit task
        # unless this is an encoder-decoder text checkpoint without modality metadata.
        if config.get("is_encoder_decoder") and not any(
            key in config for key in ("vision_config", "audio_config", "encoder")
        ):
            inferred = "text2text-generation"
    elif any(a.endswith("ForSequenceClassification") for a in architectures):
        inferred = "text-classification"
    quantization = config.get("quantization_config")
    if quantization is None:
        quantization = {}
    if not isinstance(quantization, dict):
        raise ValueError("quantization_config must be an object")
    warnings = []
    custom_code = bool(config.get("auto_map") or tokenizer_config.get("auto_map"))
    if custom_code:
        warnings.append(
            "Custom code metadata present; generated adapters disable trust_remote_code"
        )
    if not tokenizer_present:
        warnings.append(
            "No recognized tokenizer vocabulary found; provide a complete local tokenizer"
        )
    if quantization:
        warnings.append("Quantized checkpoint: verify method, hardware and runtime support")
    if inferred is None:
        warnings.append("Task cannot be inferred safely; specify --task")
    return ModelInfo(
        source=source,
        model_type=str(config.get("model_type", "unknown")),
        architectures=tuple(architectures),
        inferred_task=inferred,
        weight_files=tuple(sorted(weights)),
        tokenizer_present=tokenizer_present,
        custom_code=custom_code,
        quantization=quantization.get("quant_method"),
        transformers_version=config.get("transformers_version"),
        warnings=tuple(warnings),
        multimodal=any(key in config for key in ("vision_config", "audio_config")),
    )
