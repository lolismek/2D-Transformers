"""Reader registry: depth-combination architectures that own the model's readout.

Add a new architecture by implementing a BaseReader subclass in its own module and
registering it here. `build_reader(config)` returns None for config.reader == "none"
(the stock nanochat top-state readout), keeping the baseline path byte-identical.
"""

from nanochat.readers.base import BaseReader
from nanochat.readers.vertical import VerticalReader

REGISTRY = {
    "vertical": VerticalReader,
}


def build_reader(config):
    name = getattr(config, "reader", "none")
    if name in (None, "none", ""):
        return None
    if name not in REGISTRY:
        raise ValueError(f"unknown reader {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](config)
