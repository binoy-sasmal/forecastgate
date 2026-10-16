"""Failure types. Each carries the process exit code the CLI reports for it."""

from __future__ import annotations


class ForecastgateError(Exception):
    """An expected, explained failure. Anything else is a bug and exits 1 with a traceback."""

    exit_code = 2


class DataContractError(ForecastgateError):
    """Input data failed a hash check or a data contract."""

    exit_code = 3


class PredictionContractError(ForecastgateError):
    """Predictions failed the prediction contract; nothing was published."""

    exit_code = 4


class PublishRefusedError(ForecastgateError):
    """An artifact with the same identity exists with different content."""

    exit_code = 5


class LockHeldError(ForecastgateError):
    """Another publisher holds the publication lock."""

    exit_code = 6
