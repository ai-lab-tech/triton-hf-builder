"""Resolve native vLLM asset paths after mounting this repository for serving.

Run this standalone script in the serving container, before starting Triton.
triton-control's synchronized repository deployment does this rewriting itself.
"""

import json
from pathlib import Path


def main():
    repository = Path(__file__).resolve().parent
    manifest = json.loads((repository / "serving.json").read_text())
    name = manifest["model_name"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid model name in serving.json")
    version = repository / name / "1"
    assets = version / "model"
    if not assets.is_dir():
        raise ValueError("Missing local model assets")
    settings = json.loads((version / "model.json").read_text())
    settings.update(model=str(assets), tokenizer=str(assets))
    (version / "model.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
