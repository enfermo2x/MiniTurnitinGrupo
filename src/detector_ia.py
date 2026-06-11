"""
Módulo 4 — Detección de texto generado por IA.
Usa heurísticas estadísticas: burstiness, riqueza léxica y longitud de oraciones.
No requiere ningún modelo de HuggingFace (evita problemas de tokenizador en Windows).
"""
import math
import re
import numpy as np
from text_utils import safe_sent_tokenize


def burstiness(text: str) -> float:
    """Coeficiente de variación de longitud de oraciones. IA = bajo, humano = alto."""
    if not text or not isinstance(text, str):
        return 0.0
    sents = safe_sent_tokenize(text, language="spanish")
    lengths = [len(s.split()) for s in sents]
    if len(lengths) < 2:
        # Intentar dividir la única oración en cláusulas por comas/;/: si hay varias
        if lengths:
            parts = [p.strip() for p in re.split(r'[;,:\u2014\-]\s*', sents[0]) if p.strip()]
            if len(parts) >= 2:
                lengths = [len(p.split()) for p in parts]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths) / (np.mean(lengths) + 1e-9))


def lexical_richness(text: str) -> float:
    """Type-Token Ratio: IA tiende a repetir más palabras → TTR más bajo."""
    if not text or not isinstance(text, str):
        return 0.0
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def avg_sentence_length(text: str) -> float:
    """IA tiende a generar oraciones de longitud uniforme y moderada (15-25 palabras)."""
    if not text or not isinstance(text, str):
        return 0.0
    sents = safe_sent_tokenize(text, language="spanish")
    if not sents:
        return 0.0
    return float(np.mean([len(s.split()) for s in sents]))


def ai_probability(text: str) -> dict:
    """
    Estima probabilidad de texto generado por IA usando heurísticas estadísticas.
    Score en [0, 1]: alto → probable IA, bajo → probable humano.
    """
    bst = burstiness(text)
    ttr = lexical_richness(text)
    asl = avg_sentence_length(text)

    # Burstiness baja → más IA
    bst_score = 1.0 / (1.0 + bst)
    # TTR baja → más IA (poca variedad léxica)
    ttr_score = 1.0 - ttr
    # Longitud promedio entre 15-25 palabras → sospechoso de IA
    asl_score = math.exp(-0.5 * ((asl - 20) / 8) ** 2)

    score = round(0.4 * bst_score + 0.35 * ttr_score + 0.25 * asl_score, 3)

    return {
        "perplexity": "N/A (modo heurístico)",
        "burstiness": round(bst, 3),
        "ai_probability": score,
        "label": "Probable IA" if score > 0.6 else "Probable humano",
    }