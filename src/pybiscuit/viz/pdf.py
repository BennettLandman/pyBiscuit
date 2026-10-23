"""Pure-Python minimal PDF generator and merger (B12.3).

Provides a lightweight, zero-dependency PDF document writer that supports:
- Arbitrary MediaBox dimensions per page (e.g. 8.5 x {10.5, 8.7, 9.2, 11.7} in)
- Vector graphics (lines, rectangles, margins/frames)
- Typography (Helvetica, Helvetica-Bold) with text alignment and URI links
- Raster image embedding (RGB/RGBA numpy arrays or raw RGB) via FlateDecode
- Multi-page PDF concatenation preserving individual page MediaBox geometries
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np

# Helvetica standard character widths (1/1000th of font size)
_HELVETICA_WIDTHS: dict[str, int] = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    "0": 556, "1": 556, "2": 556, "3": 556, "4": 556, "5": 556, "6": 556, "7": 556,
    "8": 556, "9": 556, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556,
    "@": 1015, "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778,
    "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556,
    "`": 333, "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556, "o": 556,
    "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556, "v": 500, "w": 722,
    "x": 500, "y": 500, "z": 500, "{": 334, "|": 260, "}": 334, "~": 584,
}


def text_width(text: str, font_size: float) -> float:
    """Estimate string width in points for Helvetica font."""
    w = sum(_HELVETICA_WIDTHS.get(c, 556) for c in text)
    return (w / 1000.0) * font_size


def _escape_pdf_string(s: str) -> str:
    """Escape special characters in a PDF literal string."""
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfPage:
    """Single page in a PdfDocument."""

    def __init__(self, width: float, height: float) -> None:
        self.width = float(width)
        self.height = float(height)
        self.commands: list[str] = []
        self.images: list[dict] = []
        self.annotations: list[dict] = []

    def draw_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        stroke: bool = True,
        fill: bool = False,
        stroke_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fill_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        line_width: float = 1.0,
    ) -> None:
        op = "b" if (stroke and fill) else ("s" if stroke else ("f" if fill else "n"))
        self.commands.append("q")
        self.commands.append(f"{line_width:.2f} w")
        if stroke:
            r, g, b = stroke_color
            self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
        if fill:
            r, g, b = fill_color
            self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        self.commands.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")
        self.commands.append("Q")

    def draw_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        line_width: float = 1.0,
    ) -> None:
        self.commands.append("q")
        self.commands.append(f"{line_width:.2f} w")
        self.commands.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG")
        self.commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")
        self.commands.append("Q")

    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        *,
        font: str = "Helvetica",
        size: float = 10.0,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        align: str = "left",
        link: str | None = None,
    ) -> None:
        w = text_width(text, size)
        tx = x
        if align == "center":
            tx = x - w / 2.0
        elif align == "right":
            tx = x - w

        font_tag = "/F2" if "bold" in font.lower() else "/F1"
        escaped = _escape_pdf_string(text)
        self.commands.append("BT")
        self.commands.append(f"{font_tag} {size:.2f} Tf")
        self.commands.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg")
        self.commands.append(f"{tx:.2f} {y:.2f} Td")
        self.commands.append(f"({escaped}) Tj")
        self.commands.append("ET")

        if link:
            # Annotation rectangle [llx, lly, urx, ury]
            rect = [tx, y - 2, tx + w, y + size + 2]
            self.annotations.append({"rect": rect, "uri": link})

    def draw_image(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        """Embed an RGB or RGBA image at (x, y) with display width w and height h."""
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)

        img_idx = len(self.images) + 1
        img_name = f"Im{img_idx}"
        img_h, img_w = arr.shape[:2]
        raw_rgb = arr.tobytes()
        compressed = zlib.compress(raw_rgb)

        self.images.append({
            "name": img_name,
            "width": img_w,
            "height": img_h,
            "data": compressed,
        })
        self.commands.append("q")
        self.commands.append(f"{w:.2f} 0 0 {h:.2f} {x:.2f} {y:.2f} cm")
        self.commands.append(f"/{img_name} Do")
        self.commands.append("Q")


class PdfDocument:
    """PDF Document builder."""

    def __init__(self) -> None:
        self.pages: list[PdfPage] = []

    def add_page(self, width: float, height: float) -> PdfPage:
        page = PdfPage(width, height)
        self.pages.append(page)
        return page

    def write(self, target: str | Path | BinaryIO) -> None:
        """Serialize document to a PDF file or binary stream."""
        if isinstance(target, (str, Path)):
            with open(target, "wb") as f:
                self._write_to_stream(f)
        else:
            self._write_to_stream(target)

    def _write_to_stream(self, out: BinaryIO) -> None:
        offsets: list[int] = []
        objects: list[bytes] = []

        def add_obj(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        # 1: Catalog
        catalog_obj_id = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")

        # 2: Pages container (placeholder, resolved after page objects)
        pages_obj_id = add_obj(b"")

        # 3: Helvetica
        font_f1_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        # 4: Helvetica-Bold
        font_f2_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        page_ids: list[int] = []

        for page in self.pages:
            # Create image XObjects
            xobj_entries = []
            for img in page.images:
                w_img, h_img = img["width"], img["height"]
                dlen = len(img["data"])
                xobj_body = (
                    f"<< /Type /XObject /Subtype /Image /Width {w_img} /Height {h_img} "
                    f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
                    f"/Filter /FlateDecode /Length {dlen} >>\n"
                    f"stream\n"
                ).encode() + img["data"] + b"\nendstream"
                img_id = add_obj(xobj_body)
                xobj_entries.append(f"/{img['name']} {img_id} 0 R")

            # Create annotation objects
            annot_refs = []
            for a in page.annotations:
                r = a["rect"]
                uri = _escape_pdf_string(a["uri"])
                r_str = f"[{r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}]"
                annot_body = (
                    f"<< /Type /Annot /Subtype /Link /Rect {r_str} "
                    f"/Border [0 0 0] /A << /S /URI /URI ({uri}) >> >>".encode()
                )
                aid = add_obj(annot_body)
                annot_refs.append(f"{aid} 0 R")

            # Page content stream
            stream_data = "\n".join(page.commands).encode("latin1")
            compressed_stream = zlib.compress(stream_data)
            content_body = (
                f"<< /Length {len(compressed_stream)} /Filter /FlateDecode >>\nstream\n".encode()
                + compressed_stream
                + b"\nendstream"
            )
            content_id = add_obj(content_body)

            # Resources dict
            xobj_dict = f"<< {' '.join(xobj_entries)} >>" if xobj_entries else "<<"
            res_dict = (
                f"<< /Font << /F1 {font_f1_id} 0 R /F2 {font_f2_id} 0 R >> "
                f"/XObject {xobj_dict} >> >>"
            )

            # Page object
            annots_str = f"/Annots [{' '.join(annot_refs)}] " if annot_refs else ""
            page_body = (
                f"<< /Type /Page /Parent {pages_obj_id} 0 R "
                f"/MediaBox [0 0 {page.width:.2f} {page.height:.2f}] "
                f"/Contents {content_id} 0 R "
                f"{annots_str}"
                f"/Resources {res_dict} >>".encode()
            )
            pid = add_obj(page_body)
            page_ids.append(pid)

        # Update Pages object
        kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects[pages_obj_id - 1] = (
            f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>".encode()
        )

        # Write output bytes with cross-reference table
        out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        pos = len(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        for i, obj in enumerate(objects):
            offsets.append(pos)
            obj_bytes = f"{i + 1} 0 obj\n".encode() + obj + b"\nendobj\n"
            out.write(obj_bytes)
            pos += len(obj_bytes)

        xref_pos = pos
        out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
        for off in offsets:
            out.write(f"{off:010d} 00000 n \n".encode())

        out.write(
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj_id} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n".encode()
        )

    @classmethod
    def merge_pdf_files(cls, input_paths: Sequence[str | Path], output_path: str | Path) -> Path:
        """Merge multiple PDF files into one output PDF preserving each page's MediaBox."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        pages_data = []
        for p in input_paths:
            path = Path(p)
            if not path.is_file():
                continue
            data = path.read_bytes()
            mb_pat = rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]"
            mb_match = re.search(mb_pat, data)
            if mb_match:
                w = float(mb_match.group(3))
                h = float(mb_match.group(4))
            else:
                w, h = 612.0, 792.0

            # Extract content stream(s)
            stream_match = re.search(rb"/Contents\s+(\d+)\s+0\s+R", data)
            content_stream = b""
            if stream_match:
                cid = int(stream_match.group(1))
                c_pat = (
                    rf"\b{cid}\s+0\s+obj\s*<<(.*?)>>\s*stream[\r\n]+(.*?)[\r\n]+endstream".encode()
                )
                obj_m = re.search(c_pat, data, re.DOTALL)
                if obj_m:
                    dict_str = obj_m.group(1)
                    raw = obj_m.group(2)
                    if b"/FlateDecode" in dict_str:
                        try:
                            content_stream = zlib.decompress(raw)
                        except Exception:
                            content_stream = raw
                    else:
                        content_stream = raw

            # Extract any XObjects / Images
            images = []
            img_pat = (
                rb"<<\s*/Type\s*/XObject\s*/Subtype\s*/Image\s*/Width\s+(\d+)\s+/Height\s+(\d+)"
                rb".*?/Length\s+(\d+)\s*>>\s*stream[\r\n]+(.*?)[\r\n]+endstream"
            )
            for img_m in re.finditer(img_pat, data, re.DOTALL):
                iw = int(img_m.group(1))
                ih = int(img_m.group(2))
                idata = img_m.group(4)
                images.append((iw, ih, idata))

            pages_data.append((w, h, content_stream, images))

        doc = cls()
        for w, h, cstream, imgs in pages_data:
            page = doc.add_page(w, h)
            if cstream:
                page.commands.append(cstream.decode("latin1", errors="ignore"))
            for iw, ih, idata in imgs:
                img_name = f"Im{len(page.images) + 1}"
                page.images.append({
                    "name": img_name,
                    "width": iw,
                    "height": ih,
                    "data": idata,
                })

        doc.write(out_p)
        return out_p
