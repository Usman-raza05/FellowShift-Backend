import logging
from contextvars import ContextVar
from typing import Any
from uuid import uuid4


request_correlation_id: ContextVar[str | None] = ContextVar(
    "verification_request_correlation_id",
    default=None,
)


def new_correlation_id() -> str:
    return str(uuid4())


def log_event(
    logger: logging.Logger,
    level: int,
    prefix: str,
    event: str,
    **fields: Any,
) -> None:
    correlation_id = request_correlation_id.get()
    message = prefix
    if correlation_id:
        message += f" request_id={correlation_id}"
    message += f" {event}"
    if fields:
        message += " | " + " ".join(
            f"{name}={value}" for name, value in fields.items()
        )
    logger.log(level, message)


def log_exception(
    logger: logging.Logger,
    prefix: str,
    event: str,
    error: Exception,
    **fields: Any,
) -> None:
    fields["error_type"] = type(error).__name__
    fields["error_category"] = fields.get("error_category", "UNEXPECTED_EXCEPTION")
    correlation_id = request_correlation_id.get()
    message = prefix
    if correlation_id:
        message += f" request_id={correlation_id}"
    message += f" {event}"
    message += " | " + " ".join(
        f"{name}={value}" for name, value in fields.items()
    )

    safe_error = RuntimeError("sanitized exception")
    safe_error.__traceback__ = error.__traceback__
    try:
        raise safe_error from None
    except RuntimeError:
        logger.exception(message)
