"""
Utilidades para extraer texto desde archivos (txt, pdf, docx).
"""
from pathlib import Path
from typing import Optional


def extract_text_from_bytes(data: bytes, filename: Optional[str] = None) -> str:
    """Detecta el tipo por extensión en `filename` (si se da) y extrae texto.
    Si no hay filename intenta decodificar como UTF-8 y, si falla, devuelve texto ignorando errores.
    """
    import io

    # Helper: safe text decode
    def _decode(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except Exception:
            return b.decode(errors="ignore")

    if filename:
        suffix = Path(filename).suffix.lower()
    else:
        suffix = None

    # PDF
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except Exception:
            return _decode(data)
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            txts = []
            for page in doc:
                txts.append(page.get_text())
            return "\n".join(txts)
        except Exception:
            return _decode(data)

    # DOCX
    if suffix == ".docx":
        try:
            import docx
        except Exception:
            return _decode(data)
        try:
            f = io.BytesIO(data)
            doc = docx.Document(f)
            paras = [p.text for p in doc.paragraphs]
            return "\n".join(paras)
        except Exception:
            return _decode(data)

    # TXT or unknown: try to decode
    return _decode(data)
