"""road_health - camera health monitoring for road / ITS video feeds.

Classical computer vision (OpenCV + NumPy only). No deep learning.
"""

__version__ = "0.1.0"

from road_health.config import Config, load_config  # noqa: E402,F401
from road_health.errors import (  # noqa: E402,F401
    ConfigError,
    CorruptInputError,
    InputNotFoundError,
    RoadHealthError,
    UnsupportedInputError,
)

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "RoadHealthError",
    "InputNotFoundError",
    "UnsupportedInputError",
    "CorruptInputError",
    "ConfigError",
]
