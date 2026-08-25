"""No-op descriptor for the official GCNet computation path."""

NAME = "original"
LOCKED_ARGUMENTS = {
    "graph_conv_variant": "original",
    "branch_fusion": "addition",
}


def describe():
    return {
        "name": NAME,
        "adds_parameters": False,
        "replacement": None,
    }
