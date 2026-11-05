import math
import os
from pathlib import Path


def get_service_token() -> str:
    return os.getenv("FELLOWSHIFT_SERVICE_TOKEN", "")


def get_ai_request_timeout_seconds() -> float:
    value = os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "65")
    try:
        timeout = float(value)
    except ValueError:
        raise RuntimeError("AI_REQUEST_TIMEOUT_SECONDS must be numeric.") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("AI_REQUEST_TIMEOUT_SECONDS must be a positive finite number.")
    return timeout


def get_provider_timeout_ms() -> int:
    value = os.getenv("AI_PROVIDER_TIMEOUT_SECONDS", "25")
    try:
        configured_timeout = int(value)
    except ValueError:
        raise RuntimeError("AI_PROVIDER_TIMEOUT_SECONDS must be an integer.") from None
    request_timeout = get_ai_request_timeout_seconds()
    if configured_timeout <= 0:
        raise RuntimeError("AI_PROVIDER_TIMEOUT_SECONDS must be greater than zero.")
    fallback_safe_timeout = max(1, int(request_timeout * 0.45))
    return max(1, min(configured_timeout, fallback_safe_timeout, 60)) * 1000


def get_model_name(environment_variable: str, default: str) -> str:
    value = os.getenv(environment_variable, default).strip()
    if not value:
        raise RuntimeError(f"{environment_variable} must not be empty.")
    return value


def get_idempotency_database_path() -> Path:
    configured_path = os.getenv("IDEMPOTENCY_DB_PATH")
    if configured_path:
        return Path(configured_path)
    return Path(".data") / "idempotency.sqlite3"
