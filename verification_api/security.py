import hmac
import logging

from fastapi import HTTPException, Request, status

from verification_api.config import get_service_token
from verification_api.diagnostics import log_event


logger = logging.getLogger(__name__)


async def require_service_auth(request: Request) -> None:
    log_event(
        logger,
        logging.INFO,
        "[AUTH]",
        "Authentication started",
    )
    configured_token = get_service_token()
    if len(configured_token) < 32:
        log_event(
            logger,
            logging.ERROR,
            "[AUTH]",
            "Authentication failed",
            error_category="AUTH_NOT_CONFIGURED",
            http_status=503,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "AUTH_NOT_CONFIGURED", "message": "Service authentication is not configured."},
        )

    authorization = request.headers.get("authorization", "")
    scheme, separator, supplied_token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied_token, configured_token)
    ):
        log_event(
            logger,
            logging.WARNING,
            "[AUTH]",
            "Authentication failed",
            error_category="UNAUTHORIZED",
            http_status=401,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "UNAUTHORIZED", "message": "A valid bearer token is required."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    log_event(
        logger,
        logging.INFO,
        "[AUTH]",
        "Authentication succeeded",
    )
