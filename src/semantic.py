"""
Módulo 3 — Similitud semántica.
Detecta paráfrasis que la similitud léxica no ve,
usando embeddings multilingües de Sentence-BERT.
"""
import numpy as np
from typing import Optional, Any
from text_utils import safe_sent_tokenize

# Modelo por defecto — puede ajustarse si se desea usar otro embedding
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_MODEL: Optional[Any] = None


def get_model(name: Optional[str] = None) -> Any:
    """Carga el `SentenceTransformer` de forma perezosa y cacheada.

    Esto evita cargar el modelo en import time (útil para UI y tests ligeros).
    """
    global _MODEL
    if _MODEL is None:
        mname = name or _MODEL_NAME
        # Importar lazy para evitar cargar transformers en import time
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(mname)
    return _MODEL


def semantic_similarity(doc1: str, doc2: str) -> float:
    """Similitud semántica usando embeddings multilingües (producto punto normalizado).

    Devuelve 0.0 si alguno de los textos está vacío.
    """
    if not doc1 or not doc2:
        return 0.0
    model = get_model()
    emb1 = model.encode(doc1, normalize_embeddings=True)
    emb2 = model.encode(doc2, normalize_embeddings=True)
    return float(np.dot(emb1, emb2))


def find_similar_fragments(
    doc1: str,
    doc2: str,
    threshold: float = 0.82,
) -> list[dict]:
    """
    Compara oraciones de doc1 contra oraciones de doc2.
    Retorna pares con similitud semántica >= threshold, ordenados desc.
    """
    if not doc1 or not doc2:
        return []

    sents1 = safe_sent_tokenize(doc1, language="spanish")
    sents2 = safe_sent_tokenize(doc2, language="spanish")

    if not sents1 or not sents2:
        return []

    model = get_model()
    embs1 = model.encode(sents1, normalize_embeddings=True)
    embs2 = model.encode(sents2, normalize_embeddings=True)

    results = []
    for s1, e1 in zip(sents1, embs1):
        for s2, e2 in zip(sents2, embs2):
            score = float(np.dot(e1, e2))
            if score >= threshold:
                results.append({
                    "frag_doc1": s1,
                    "frag_doc2": s2,
                    "score": round(score, 3),
                })

    # Si no encontramos fragmentos y los textos son cortos o con pocas oraciones,
    # intentar una segunda pasada dividiendo por comas/; para capturar coincidencias.
    if not results:
        import re as _re
        parts1 = []
        for s in sents1:
            parts1.extend([p.strip() for p in _re.split(r'[;,:\u2014\-]\s*', s) if p.strip()])
        parts2 = []
        for s in sents2:
            parts2.extend([p.strip() for p in _re.split(r'[;,:\u2014\-]\s*', s) if p.strip()])

        if parts1 and parts2:
            embs1p = model.encode(parts1, normalize_embeddings=True)
            embs2p = model.encode(parts2, normalize_embeddings=True)
            for s1, e1 in zip(parts1, embs1p):
                for s2, e2 in zip(parts2, embs2p):
                    score = float(np.dot(e1, e2))
                    if score >= threshold:
                        results.append({
                            "frag_doc1": s1,
                            "frag_doc2": s2,
                            "score": round(score, 3),
                        })

    return sorted(results, key=lambda x: -x["score"])
