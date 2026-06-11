"""
Motor principal — fusión de scores.
Orquesta preprocesamiento, similitud léxica, semántica y detección de IA.
"""
import sys
from pathlib import Path

# Permite ejecutar main.py directamente desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess  import preprocess
from lexical     import tfidf_similarity, minhash_similarity
from semantic    import semantic_similarity, find_similar_fragments
from detector_ia import ai_probability


def analyze(doc1: str, doc2: str, check_ia: bool = True) -> dict:
    """
    Analiza la similitud entre dos documentos de texto.

    Args:
        doc1:     Texto del documento A (trabajo del alumno).
        doc2:     Texto del documento B (referencia / corpus).
        check_ia: Si True, ejecuta el detector de texto generado por IA.
                  Es costoso (carga un LM ~500 MB), desactívalo si no hace falta.

    Returns:
        Diccionario con scores léxico, MinHash, semántico, final,
        nivel de alerta, fragmentos similares y (opcionalmente) análisis IA.
    """
    p1, p2 = preprocess(doc1), preprocess(doc2)

    lex = tfidf_similarity(p1, p2)
    mh  = minhash_similarity(p1, p2)
    sem = semantic_similarity(p1, p2)

    # Ponderación: semántico tiene más peso (captura paráfrasis)
    final = round(0.25 * lex + 0.15 * mh + 0.60 * sem, 3)

    result = {
        "score_lexico":         round(lex, 3),
        "score_minhash":        round(mh,  3),
        "score_semantico":      round(sem, 3),
        "score_final":          final,
        "nivel": "Alto" if final > 0.75 else "Medio" if final > 0.45 else "Bajo",
        "fragmentos_similares": find_similar_fragments(doc1, doc2),
    }

    if check_ia:
        result["analisis_ia"] = {
            "doc_a": ai_probability(doc1),
            "doc_b": ai_probability(doc2),
        }

    return result


# ── Uso desde línea de comandos ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json
    from report import generate_report

    parser = argparse.ArgumentParser(description="Mini-Turnitin — análisis de similitud")
    parser.add_argument("doc_a", help="Ruta al archivo del documento A")
    parser.add_argument("doc_b", help="Ruta al archivo del documento B")
    parser.add_argument("--no-ia", action="store_true", help="Omitir detección de texto IA")
    parser.add_argument("--reporte", default="reporte.html", help="Ruta del reporte HTML de salida")
    parser.add_argument("--corpus", default=None, help="Directorio corpus para comparación (opcional)")
    parser.add_argument("--top", type=int, default=5, help="Número de matches del corpus a mostrar")
    parser.add_argument("--gpt2", action="store_true", help="Intentar calcular perplexity con GPT-2 (opcional)")
    args = parser.parse_args()

    text_a = Path(args.doc_a).read_text(encoding="utf-8")
    text_b = Path(args.doc_b).read_text(encoding="utf-8")

    print("Analizando…")
    result = analyze(text_a, text_b, check_ia=not args.no_ia)

    print(json.dumps(result, ensure_ascii=False, indent=2,
                     default=lambda o: str(o)))

    # Pasar los textos a generate_report para permitir comparaciones con corpus y GPT-2
    report_path = generate_report(
        result,
        text_a,
        text_b,
        args.reporte,
        corpus_dir=args.corpus,
        top_k=args.top,
        use_gpt2=args.gpt2,
    )

    print(f"\nReporte guardado en: {report_path}")
