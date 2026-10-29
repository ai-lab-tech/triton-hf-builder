"""Create Triton configuration using the official client's protobuf type."""

from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2


def parse_model_config(content: str) -> model_config_pb2.ModelConfig:
    """Reject malformed text, unknown fields and invalid protobuf field types."""
    try:
        return text_format.Parse(content, model_config_pb2.ModelConfig())
    except text_format.ParseError as error:
        raise ValueError(f"Invalid config.pbtxt: {error}") from error


def render_model_config(name: str, backend: str, instance_kind: str, *, python_io: bool) -> str:
    config = model_config_pb2.ModelConfig(name=name, backend=backend, max_batch_size=0)
    config.instance_group.add(
        count=1,
        kind=model_config_pb2.ModelInstanceGroup.Kind.Value(instance_kind),
    )
    if python_io:
        config.input.add(name="text_input", data_type=model_config_pb2.TYPE_STRING, dims=[1])
        config.output.add(name="text_output", data_type=model_config_pb2.TYPE_STRING, dims=[1])
    content = text_format.MessageToString(config, as_utf8=True)
    parse_model_config(content)
    return content
