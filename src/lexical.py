"""
Módulo 2 — Similitud léxica.
Implementa TF-IDF + coseno y MinHash para detección de near-duplicates.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash


def tfidf_similarity(doc1: str, doc2: str) -> float:
    """Similitud léxica clásica: TF-IDF + coseno con n-gramas (1–3)."""
    if not doc1 or not doc2:
        return 0.0
    try:
        vec = TfidfVectorizer(ngram_range=(1, 3))
        matrix = vec.fit_transform([doc1, doc2])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return 0.0


def minhash_similarity(doc1: str, doc2: str) -> float:
    """Similitud aproximada rápida usando MinHash (detección de near-duplicates)."""
    def to_minhash(text: str) -> MinHash:
        tokens = [w for w in text.split() if w]
        m = MinHash(num_perm=128)
        for word in tokens:
            m.update(word.encode("utf8"))
        return m, len(tokens)

    m1, n1 = to_minhash(doc1)
    m2, n2 = to_minhash(doc2)
    if n1 == 0 or n2 == 0:
        return 0.0
    try:
        return m1.jaccard(m2)
    except Exception:
        return 0.0
