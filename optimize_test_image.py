from PIL import Image
import os

INPUT_PATH = "test_documents/income_certificate.png"
OUTPUT_PATH = "test_documents/income_certificate_optimized.jpg"

MAX_WIDTH = 1400
JPEG_QUALITY = 85


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


if not os.path.exists(INPUT_PATH):
    print(f"❌ File not found: {INPUT_PATH}")
    exit()


original_size = os.path.getsize(INPUT_PATH)

print("=" * 60)
print("        TEST DOCUMENT IMAGE OPTIMIZER")
print("=" * 60)

print(f"\nOriginal image : {INPUT_PATH}")
print(f"Original size  : {format_size(original_size)}")

image = Image.open(INPUT_PATH)

print(f"Original dimensions: {image.width} × {image.height}")

# Keep aspect ratio
if image.width > MAX_WIDTH:
    new_height = int(image.height * (MAX_WIDTH / image.width))
    image = image.resize(
        (MAX_WIDTH, new_height),
        Image.Resampling.LANCZOS
    )

# Convert to RGB because JPEG doesn't support RGBA
if image.mode != "RGB":
    image = image.convert("RGB")

image.save(
    OUTPUT_PATH,
    "JPEG",
    quality=JPEG_QUALITY,
    optimize=True
)

optimized_size = os.path.getsize(OUTPUT_PATH)

reduction = (1 - optimized_size / original_size) * 100

print(f"\nOptimized image: {OUTPUT_PATH}")
print(f"New dimensions : {image.width} × {image.height}")
print(f"Optimized size : {format_size(optimized_size)}")
print(f"Size reduction : {reduction:.2f}%")

print("\n" + "=" * 60)
print("✅ Optimization complete")
print("=" * 60)