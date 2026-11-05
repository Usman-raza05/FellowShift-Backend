import hmac

from fastapi import HTTPException, Request, status

from verification_api.config import get_service_token


async def require_service_auth(request: Request) -> None:
    configured_token = get_service_token()
    if len(configured_token) < 32:
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "UNAUTHORIZED", "message": "A valid bearer token is required."},
            headers={"WWW-Authenticate": "Bearer"},
        )
