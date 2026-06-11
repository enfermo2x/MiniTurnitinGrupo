"""
Módulo 5 — Generación de reporte HTML y utilidades adicionales.
Mejoras:
- Soporte opcional de perplexidad con GPT-2 (si está instalado).
- Comparación de un documento frente a un directorio `corpus`.
- Plantilla HTML más completa con sección de comparaciones.
"""
from __future__ import annotations

import datetime
import html
from pathlib import Path
from typing import List, Optional

# Importar utilidades del proyecto
from preprocess import preprocess
from lexical import tfidf_similarity, minhash_similarity
from semantic import semantic_similarity, find_similar_fragments
from detector_ia import ai_probability


# Plantilla principal (ligero rediseño)
_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <title>Reporte Mini-Turnitin</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 980px;
            margin: 2rem auto; color: #1a1a1a; background: #f7f9fb; }}
    header {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; }}
    h1   {{ color: #0f172a; border-bottom: 2px solid #e6eef8; padding-bottom:.6rem; }}
    h2   {{ color: #0b2340; margin-top: 1.6rem; }}
    .badge {{ display:inline-block; padding:.25rem .6rem; border-radius:6px;
              font-weight:700; font-size:.9rem; }}
    .alto   {{ background:#ffe6e6; color:#a41e1e; }}
    .medio  {{ background:#fff7e6; color:#8a5a00; }}
    .bajo   {{ background:#e9fbf1; color:#0a5f3a; }}
    .scores {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }}
    .score-box {{ background:#fff; border:1px solid #e6eef8; border-radius:8px;
                  padding:.75rem 1.2rem; min-width:140px; text-align:center; box-shadow:0 2px 6px rgba(16,24,40,.04); }}
    .score-box .val {{ font-size:1.6rem; font-weight:700; color:#0b2440 }}
    .score-box .lbl {{ font-size:.78rem; color:#6b7280; }}
    .fragment {{ background:#fff; border-left:4px solid #ef4444;
                 margin:.6rem 0; padding:.6rem 1rem; border-radius:0 8px 8px 0; }}
    .fragment .sim {{ font-size:.82rem; color:#ef4444; font-weight:700; }}
    .fragment .label {{ font-size:.78rem; color:#6b7280; margin-top:.3rem; }}
    .ia-box {{ background:#fff; border:1px solid #e6eef8; border-radius:8px;
               padding:.5rem .9rem; margin:.5rem 0; }}
    table.matches {{ width:100%; border-collapse:collapse; margin-top:.6rem; }}
    table.matches th, table.matches td {{ text-align:left; padding:.45rem .6rem; border-bottom:1px solid #eef2f7; font-size:.9rem; }}
    footer {{ margin-top:2.4rem; font-size:.78rem; color:#6b7280; text-align:center; }}
    a.file-link {{ color:#0b61ff; text-decoration:none }}
  </style>
</head>
<body>
  <header>
    <h1>Reporte de similitud — Mini-Turnitin</h1>
    <div><small>{timestamp}</small></div>
  </header>

  <h2>Resumen</h2>
  <span class="badge {nivel_cls}">{nivel} — {score_final}%</span>

  <div class="scores">
    <div class="score-box">
      <div class="val">{score_lexico}%</div>
      <div class="lbl">Léxico (TF-IDF)</div>
    </div>
    <div class="score-box">
      <div class="val">{score_minhash}%</div>
      <div class="lbl">MinHash</div>
    </div>
    <div class="score-box">
      <div class="val">{score_semantico}%</div>
      <div class="lbl">Semántico</div>
    </div>
  </div>

  {ia_section}

  <h2>Fragmentos con alta similitud ({n_frags})</h2>
  {fragments_html}

  {corpus_section}

  <footer>Mini-Turnitin · Curso PLN · Escuela de Ciencia de Datos</footer>
</body>
</html>
"""


_IA_SECTION = """\
  <h2>Análisis de texto IA</h2>
  {ia_rows}
"""


_IA_ROW = """\
  <div class="ia-box">
    <strong>{doc_label} — {label}</strong>
    &nbsp;|&nbsp; Perplejidad: {perplexity}
    &nbsp;|&nbsp; Variabilidad: {burstiness}
    &nbsp;|&nbsp; Score IA: {ai_probability}
  </div>
"""


_FRAGMENT = """\
  <div class="fragment">
    <div class="sim">Similitud: {score}%</div>
    <div class="label">Doc A</div>
    <p>{frag_doc1}</p>
    <div class="label">Doc B</div>
    <p>{frag_doc2}</p>
  </div>
"""


_CORPUS_SECTION = """\
  <h2>Comparación con corpus (top {top_k})</h2>
  {matches_table}
"""


def _format_fragments(frags: list[dict]) -> str:
    if not frags:
        return "<p>No se encontraron fragmentos con similitud alta.</p>"
    return "\n".join(
        _FRAGMENT.format(
            score=round(f["score"] * 100, 1),
            frag_doc1=html.escape(f.get("frag_doc1", "")),
            frag_doc2=html.escape(f.get("frag_doc2", "")),
        )
        for f in frags
    )


def _format_ia_section(ia_data: dict, use_gpt2_perplexity: bool = False) -> str:
    if not ia_data:
        return ""
    ia_rows = "\n".join(
        _IA_ROW.format(doc_label=lbl, **data)
        for lbl, data in [("Doc A", ia_data["doc_a"]), ("Doc B", ia_data["doc_b"])]
    )
    return _IA_SECTION.format(ia_rows=ia_rows)


def compute_gpt2_perplexity(text: str) -> Optional[float]:
    """Intento seguro de calcular perplexity usando GPT-2 (si está disponible).
    Devuelve `None` si no está instalado o si falla la evaluación.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        max_len = model.config.n_positions

        nlls = []
        # Procesar en ventanas para textos largos
        stride = max_len
        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + 1 - max_len, 0)
            end_loc = min(i + stride, input_ids.size(1))
            input_ids_section = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_section.clone()
            # Todos los tokens excepto los últimos se esconden como objetivo
            with torch.no_grad():
                outputs = model(input_ids_section, labels=target_ids)
                # outputs.loss ya es la media negativa por token
                neg_log_likelihood = outputs.loss * input_ids_section.size(1)
            nlls.append(neg_log_likelihood)

        import math

        ppl = float(torch.exp(torch.stack(nlls).sum() / input_ids.size(1)).item())
        return round(ppl, 3)
    except Exception:
        return None


def compare_with_corpus(
    doc_text: str, corpus_dir: str | Path, top_k: int = 5, check_ia: bool = False
) -> list[dict]:
    """Compara `doc_text` contra todos los archivos de texto en `corpus_dir`.

    Retorna una lista con los mejores `top_k` matches con métricas y fragmentos.
    """
    corpus_dir = Path(corpus_dir)
    candidates = []
    for p in sorted(corpus_dir.rglob("*.txt")):
        try:
            other = p.read_text(encoding="utf-8")
        except Exception:
            continue

        p1, p2 = preprocess(doc_text), preprocess(other)
        lex = tfidf_similarity(p1, p2)
        mh = minhash_similarity(p1, p2)
        sem = semantic_similarity(p1, p2)
        final = round(0.25 * lex + 0.15 * mh + 0.60 * sem, 3)

        fragments = find_similar_fragments(doc_text, other)
        match = {
            "file": str(p.relative_to(Path.cwd())),
            "score_final": final,
            "score_lexico": round(lex, 3),
            "score_minhash": round(mh, 3),
            "score_semantico": round(sem, 3),
            "fragmentos": fragments[:3],
        }
        if check_ia:
            match["analisis_ia"] = ai_probability(other)

        candidates.append(match)

    candidates.sort(key=lambda x: -x["score_final"])  # desc
    return candidates[:top_k]


def generate_report(
    result: dict,
    doc_a: Optional[str] = None,
    doc_b: Optional[str] = None,
    output_path: str | Path = "reporte.html",
    corpus_dir: Optional[str | Path] = None,
    top_k: int = 5,
    use_gpt2: bool = False,
) -> Path:
    """Genera un archivo HTML con el reporte del análisis.

    Args:
        result: diccionario devuelto por `main.analyze()` o estructura equivalente.
        doc_a: texto original del documento A (opcional, necesario para comparar con corpus).
        doc_b: texto original del documento B (opcional, solo para información).
        output_path: ruta del archivo de salida
        corpus_dir: si se pasa, se comparará `doc_a` contra ese directorio
        top_k: cuántos matches del corpus mostrar
        use_gpt2: si True, intenta calcular perplexity con GPT-2 (si está disponible)

    Returns:
        Path del archivo generado
    """
    output_path = Path(output_path)
    nivel = result.get("nivel", "Desconocido")
    nivel_cls = nivel.lower()

    # Fragmentos del análisis principal
    frags = result.get("fragmentos_similares", [])
    fragments_html = _format_fragments(frags)

    # Sección IA (usa lo que venga en result)
    ia_section = _format_ia_section(result.get("analisis_ia"))

    # Comparación con corpus (si se solicita)
    if corpus_dir and doc_a:
        matches = compare_with_corpus(doc_a, corpus_dir, top_k=top_k, check_ia=False)
        if matches:
            rows = [
                "<tr><th>Archivo</th><th>Score final</th><th>Léxico</th><th>MinHash</th><th>Semántico</th></tr>"
            ]
            for m in matches:
                rows.append(
                    f"<tr>"
                    f"<td><a class=\"file-link\" href=\"{html.escape(m['file'])}\">{html.escape(m['file'])}</a></td>"
                    f"<td>{round(m['score_final']*100,1)}%</td>"
                    f"<td>{round(m['score_lexico']*100,1)}%</td>"
                    f"<td>{round(m['score_minhash']*100,1)}%</td>"
                    f"<td>{round(m['score_semantico']*100,1)}%</td>"
                    f"</tr>"
                )
            matches_table = f"<table class=\"matches\">{''.join(rows)}</table>"
        else:
            matches_table = "<p>No se encontraron coincidencias en el corpus.</p>"
        corpus_section = _CORPUS_SECTION.format(top_k=top_k, matches_table=matches_table)
    else:
        corpus_section = ""

    # Si se pide GPT-2, intentar calcular y añadir a la sección IA
    if use_gpt2 and doc_a and doc_b:
        gpta = compute_gpt2_perplexity(doc_a)
        gptb = compute_gpt2_perplexity(doc_b)
        # Añadir/actualizar result['analisis_ia'] para render
        anal = result.setdefault("analisis_ia", {"doc_a": {}, "doc_b": {}})
        anal["doc_a"]["perplexity"] = gpta or "N/D"
        anal["doc_b"]["perplexity"] = gptb or "N/D"
        ia_section = _format_ia_section(anal)

    html_content = _TEMPLATE.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        nivel=nivel,
        nivel_cls=nivel_cls,
        score_final=round(result.get("score_final", 0) * 100, 1),
        score_lexico=round(result.get("score_lexico", 0) * 100, 1),
        score_minhash=round(result.get("score_minhash", 0) * 100, 1),
        score_semantico=round(result.get("score_semantico", 0) * 100, 1),
        n_frags=len(frags),
        fragments_html=fragments_html,
        ia_section=ia_section,
        corpus_section=corpus_section,
    )

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
