from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFont


class SyntheticDocument(TypedDict):
    filename: str
    identifier: str
    title: str
    reference: str
    fields: tuple[tuple[str, str], ...]


OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic_documents"
DISCLAIMER = "SYNTHETIC TEST DOCUMENT — NOT A VALID GOVERNMENT CERTIFICATE"
STUDENT_NAME = "Aarav Sample"
STUDENT_ID = "SYNTH-STU-0001"
ADDRESS = "42 Example Lane, Sample Nagar, Demo State 000000"

DOCUMENTS: tuple[SyntheticDocument, ...] = (
    {
        "filename": "synthetic_income_certificate.png",
        "identifier": "INCOME_CERT",
        "title": "Income Certificate",
        "reference": "SYNTH-INCOME-2026-001",
        "fields": (
            ("Student", STUDENT_NAME),
            ("Student ID", STUDENT_ID),
            ("Fictional annual family income", "INR 120,000"),
            ("Assessment period", "2026-2027 (synthetic)"),
            ("Issuer", "None — software test fixture"),
        ),
    },
    {
        "filename": "synthetic_category_certificate.png",
        "identifier": "ST_CASTE_CERT",
        "title": "Category Certificate",
        "reference": "SYNTH-CATEGORY-2026-001",
        "fields": (
            ("Student", STUDENT_NAME),
            ("Student ID", STUDENT_ID),
            ("Fictional category", "TEST CATEGORY — NOT A REAL COMMUNITY"),
            ("Purpose", "Scholarship software test only"),
            ("Issuer", "None — software test fixture"),
        ),
    },
    {
        "filename": "synthetic_academic_marksheet.png",
        "identifier": "MARKSHEET",
        "title": "Academic Marksheet",
        "reference": "SYNTH-MARKS-2026-001",
        "fields": (
            ("Student", STUDENT_NAME),
            ("Student ID", STUDENT_ID),
            ("Institution", "Sample Demo College (fictional)"),
            ("Examination", "Synthetic Board Demo Exam"),
            ("Fictional marks", "428 / 500  |  85.6%"),
        ),
    },
    {
        "filename": "synthetic_domicile_certificate.png",
        "identifier": "DOMICILE_CERT",
        "title": "Domicile Certificate",
        "reference": "SYNTH-DOMICILE-2026-001",
        "fields": (
            ("Student", STUDENT_NAME),
            ("Student ID", STUDENT_ID),
            ("Fictional address", ADDRESS),
            ("Residency period", "Synthetic test data only"),
            ("Issuer", "None — software test fixture"),
        ),
    },
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_names = (
        ("arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("arial.ttf", "DejaVuSans.ttf")
    )
    font_paths = (
        *(Path("C:/Windows/Fonts") / name for name in font_names),
        *(Path("/usr/share/fonts/truetype/dejavu") / name for name in font_names),
    )
    for font_path in font_paths:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _render_document(document: SyntheticDocument) -> Image.Image:
    image = Image.new("RGB", (1400, 1000), "#f4f6fa")
    draw = ImageDraw.Draw(image)
    regular = _font(28)
    bold = _font(30, bold=True)
    heading = _font(48, bold=True)
    warning = _font(25, bold=True)
    small = _font(22)

    draw.rectangle((0, 0, 1400, 120), fill="#a8212b")
    draw.text((55, 37), DISCLAIMER, fill="white", font=warning)
    draw.text((70, 165), "FICTIONAL SOFTWARE TEST FIXTURE", fill="#a8212b", font=bold)
    draw.text((70, 220), str(document["title"]), fill="#17243a", font=heading)
    draw.text(
        (70, 300),
        f"Backend document identifier: {document['identifier']}",
        fill="#31415c",
        font=regular,
    )
    draw.text(
        (70, 360),
        f"Reference: {document['reference']}     |     No official issuer or authority",
        fill="#526078",
        font=small,
    )
    draw.line((70, 410, 1330, 410), fill="#c6ceda", width=3)

    y = 455
    for label, value in document["fields"]:
        draw.text((85, y), str(label), fill="#526078", font=small)
        draw.text((480, y - 3), str(value), fill="#17243a", font=regular)
        y += 78

    draw.rounded_rectangle((70, 850, 1330, 925), radius=16, fill="#ffe5e5")
    draw.text((95, 872), DISCLAIMER, fill="#8d1720", font=warning)
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    destinations = [OUTPUT_DIR / document["filename"] for document in DOCUMENTS]
    existing = [path.name for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing synthetic documents: "
            + ", ".join(existing)
        )

    for document, destination in zip(DOCUMENTS, destinations):
        image = _render_document(document)
        with destination.open("xb") as output:
            image.save(output, format="PNG")
        print(f"{destination.relative_to(OUTPUT_DIR.parent)} | {document['identifier']}")


if __name__ == "__main__":
    main()
