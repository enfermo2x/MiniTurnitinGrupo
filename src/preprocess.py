"""Módulo 1 — Preprocesamiento de texto en español.

Limpia y normaliza texto. Cuando `spaCy` no está disponible se aplica
un preprocesado seguro y sencillo para mantener la funcionalidad.
"""
import re
from typing import List

try:
    import spacy
    nlp = spacy.load("es_core_news_sm")
except Exception:
    nlp = None

# Patrón unificado: elimina todo lo que no sea letra Unicode ni espacio
_CLEAN = re.compile(r'[^\w\s]', flags=re.UNICODE)


def _chunk_text_by_chars(text: str, max_chars: int) -> List[str]:
    """Divide `text` en fragmentos de hasta `max_chars` sin cortar palabras.
    Devuelve una lista de fragmentos.
    """
    words = text.split()
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        wlen = len(w) + 1
        if cur_len + wlen > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += wlen
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def preprocess(text: str) -> str:
    """Limpia, tokeniza y lematiza texto en español.

    Para textos muy largos evita pasar todo el texto a `nlp` de una vez —
    lo divide en fragmentos manejables usando `_chunk_text_by_chars`.
    También desactiva `parser` y `ner` durante este preprocesado para ahorrar memoria.
    """
    if not text:
        return ""

    # Normalizar y limpiar
    text = _CLEAN.sub('', text.lower())

    # Determinar tamaño máximo de fragmento: usar nlp.max_length si está disponible
    try:
        nlp_max = int(getattr(nlp, "max_length", 1000000))
    except Exception:
        nlp_max = 1000000

    # Reservar un margen para evitar alcanzar justo el límite
    chunk_size = min(100000, max(20000, nlp_max - 1000))

    if len(text) > chunk_size:
        fragments = _chunk_text_by_chars(text, chunk_size)
    else:
        fragments = [text]

    # Si spaCy no está disponible, devolver una versión limpia y tokenizada
    if nlp is None:
        tokens: List[str] = []
        for frag in fragments:
            toks = re.findall(r'\w+', frag.lower())
            tokens.extend([t for t in toks if len(t) > 2])
        return " ".join(tokens)

    lemmas: List[str] = []
    # Desactivar componentes pesados que no necesitamos (parser, ner)
    with nlp.select_pipes(disable=["parser", "ner"]):
        for frag in fragments:
            doc = nlp(frag)
            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    lemmas.append(token.lemma_)

    return " ".join(lemmas)
