"""Typed errors.

The CLI maps these to distinct exit codes so that an operator or a scheduler
can distinguish "your input was bad" from "the tool crashed".
"""


class RoadHealthError(Exception):
    """Base class for all expected, user-facing failures."""

    exit_code = 1


class InputNotFoundError(RoadHealthError):
    exit_code = 2


class UnsupportedInputError(RoadHealthError):
    exit_code = 3


class CorruptInputError(RoadHealthError):
    exit_code = 4


class ConfigError(RoadHealthError):
    exit_code = 5


class BaselineError(RoadHealthError):
    exit_code = 6
