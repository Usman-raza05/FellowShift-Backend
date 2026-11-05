from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


MAX_DOCUMENTS = 5
MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_IMAGE_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


async def read_and_validate_document(
    upload: UploadFile,
) -> tuple[bytes, str]:
    declared_type = (upload.content_type or "").lower()
    if declared_type and declared_type not in SUPPORTED_IMAGE_TYPES.values():
        raise HTTPException(
            status_code=415,
            detail={
                "error": "UNSUPPORTED_FILE_TYPE",
                "message": "Supported document formats are JPEG, PNG, and WebP images.",
            },
        )

    content = await upload.read(MAX_DOCUMENT_SIZE_BYTES + 1)
    if not content:
        raise HTTPException(
            status_code=400,
            detail={"error": "EMPTY_DOCUMENT", "message": "A document is empty."},
        )
    if len(content) > MAX_DOCUMENT_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "DOCUMENT_TOO_LARGE", "message": "Each document must be 10 MiB or smaller."},
        )

    try:
        with Image.open(BytesIO(content)) as image:
            image_format = image.format
            width, height = image.size
            if image_format not in SUPPORTED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail={
                        "error": "UNSUPPORTED_FILE_TYPE",
                        "message": "Supported document formats are JPEG, PNG, and WebP images.",
                    },
                )
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "IMAGE_DIMENSIONS_TOO_LARGE",
                        "message": "The document image exceeds the maximum pixel dimensions.",
                    },
                )
            detected_type = SUPPORTED_IMAGE_TYPES[image_format]
            if declared_type and declared_type != detected_type:
                raise HTTPException(
                    status_code=415,
                    detail={
                        "error": "FILE_TYPE_MISMATCH",
                        "message": "The declared file type does not match the document content.",
                    },
                )
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_DOCUMENT", "message": "The document is not a readable image."},
        ) from None

    return content, SUPPORTED_IMAGE_TYPES[image_format]
