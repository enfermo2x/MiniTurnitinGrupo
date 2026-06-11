"""
Interfaz web con Streamlit para el Mini-Turnitin.
Ejecutar con: streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from main import analyze
from report import generate_report
from compat import (
    has_spacy,
    has_sentence_transformers,
    has_torch,
    has_transformers,
    has_gpt2 as _has_gpt2,
)
from typing import Optional


@st.cache_resource
def _load_embedding_model(name: Optional[str] = None):
    # Carga perezosa y cacheada del modelo de embeddings (si está disponible)
    from semantic import get_model

    return get_model(name)
from report import compare_with_corpus

st.set_page_config(
    page_title="Mini-Turnitin PLN",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Mini-Turnitin — Curso PLN · Escuela de Ciencia de Datos")
st.caption("Detección de similitud léxica, semántica y texto generado por IA.")

# ----------------- Entorno y estado ---------------------------------
spacy_ok = has_spacy()
st_ok = has_sentence_transformers()
torch_ok = has_torch()
transformers_ok = has_transformers()
gpt2_ok = _has_gpt2()

st.sidebar.header("Estado del entorno")
st.sidebar.markdown(f"- spaCy: {'✅' if spacy_ok else '❌'}")
st.sidebar.markdown(f"- Sentence-Transformers: {'✅' if st_ok else '❌'}")
st.sidebar.markdown(f"- PyTorch: {'✅' if torch_ok else '❌'}")
st.sidebar.markdown(f"- Transformers: {'✅' if transformers_ok else '❌'}")
st.sidebar.markdown(f"- GPT-2 disponible: {'✅' if gpt2_ok else '❌'}")

if st_ok:
    if st.sidebar.button("Precargar embeddings"):
        with st.spinner("Precargando modelo de embeddings (puede tardar)..."):
            try:
                _load_embedding_model()
                st.sidebar.success("Modelo de embeddings pre-cargado")
            except Exception as e:
                st.sidebar.error(f"Error cargando modelo: {e}")

if not spacy_ok:
    st.info("spaCy no detectado: se usará un preprocesado alternativo (más ligero). Si deseas mejor lematización instala `es_core_news_sm`.")


# ── Entradas ────────────────────────────────────────────────────────────────
mode = st.radio("Modo de análisis", (
    "Comparar dos documentos",
    "Comparar contra corpus",
    "Detectar IA únicamente",
))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Documento A — trabajo del alumno")
    type_a = st.radio("Tipo A", ("Texto", "Archivo"), horizontal=True, key="type_a")
    if type_a == "Texto":
        doc1 = st.text_area("", height=220, key="doc1",
                             placeholder="Pega aquí el texto del trabajo a evaluar…")
    else:
        up_a = st.file_uploader("Subir archivo A (txt/pdf/docx)", type=["txt","pdf","docx"], key="up_a")
        doc1 = None
        if up_a is not None:
            from file_utils import extract_text_from_bytes
            doc1 = extract_text_from_bytes(up_a.getvalue(), filename=up_a.name)

with col2:
    st.subheader("Documento B — referencia / corpus")
    type_b = st.radio("Tipo B", ("Texto", "Archivo"), horizontal=True, key="type_b")
    if type_b == "Texto":
        doc2 = st.text_area("", height=220, key="doc2",
                             placeholder="Pega aquí el texto de referencia…")
    else:
        up_b = st.file_uploader("Subir archivo B (txt/pdf/docx)", type=["txt","pdf","docx"], key="up_b")
        doc2 = None
        if up_b is not None:
            from file_utils import extract_text_from_bytes
            doc2 = extract_text_from_bytes(up_b.getvalue(), filename=up_b.name)

check_ia = st.checkbox("Incluir análisis de texto IA (más lento)", value=True)
if gpt2_ok:
    use_gpt2 = st.checkbox("Usar GPT-2 para perplexity (si está instalado)", value=False)
else:
    st.caption("GPT-2 no disponible (transformers/torch no instalados). Perplexity será 'N/D'.")
    use_gpt2 = False

# Opciones adicionales para comparación contra corpus
corpus_dir = None
top_k = 5
if mode == "Comparar contra corpus":
    corpus_dir = st.text_input("Ruta al directorio corpus (ej: data/corpus)")
    top_k = st.number_input("Top K matches a mostrar", min_value=1, max_value=50, value=5)

# Botón principal
can_run = False
if mode == "Comparar dos documentos":
    can_run = bool(doc1 and doc2)
elif mode == "Comparar contra corpus":
    can_run = bool(doc1 and corpus_dir)
elif mode == "Detectar IA únicamente":
    can_run = bool(doc1)

run = st.button("Analizar similitud", type="primary", disabled=not can_run)

# ── Resultados ───────────────────────────────────────────────────────────────
if run:
    # Validar precondiciones según modo
    if mode == "Comparar dos documentos" and not (doc1 and doc2):
        st.error("Debe proporcionar ambos documentos (A y B).")
    elif mode == "Comparar contra corpus" and not (doc1 and corpus_dir):
        st.error("Debe proporcionar el Documento A y la ruta al corpus.")
    else:
        # Ejecutar según modo
        if mode == "Comparar contra corpus":
            with st.spinner("Comparando contra corpus…"):
                matches = compare_with_corpus(doc1, corpus_dir, top_k=top_k, check_ia=check_ia)
            st.subheader(f"Top {len(matches)} coincidencias en corpus")
            if matches:
                for m in matches:
                    st.markdown(f"**{m['file']}** — Score: {m['score_final']*100:.1f}%")
                    st.markdown(f"Léxico: {m['score_lexico']*100:.1f}% — MinHash: {m['score_minhash']*100:.1f}% — Semántico: {m['score_semantico']*100:.1f}%")
                    if m.get("fragmentos"):
                        for f in m.get("fragmentos", [])[:2]:
                            with st.expander(f"Fragment (score {f['score']*100:.1f}%):"):
                                fa, fb = st.columns(2)
                                fa.markdown(f"**Doc A**\n\n{f.get('frag_doc1','')}")
                                fb.markdown(f"**Doc B**\n\n{f.get('frag_doc2','')}")
            else:
                st.info("No se encontraron coincidencias en el corpus.")
            # Generar reporte con sección corpus si se quiere descargar
            report_path = Path("reporte_temp.html")
            generate_report({}, doc_a=doc1, doc_b=None, output_path=report_path, corpus_dir=corpus_dir, top_k=top_k, use_gpt2=use_gpt2)
            st.download_button(
                label="⬇️ Descargar reporte HTML (corpus)",
                data=report_path.read_bytes(),
                file_name="reporte_mini_turnitin_corpus.html",
                mime="text/html",
            )
        else:
            # Comparar dos documentos o detectar IA únicamente
            if mode == "Detectar IA únicamente":
                # Analizar A vs documento vacío para obtener IA solo en A
                with st.spinner("Analizando texto IA…"):
                    result = analyze(doc1, "", check_ia=check_ia)
            else:
                with st.spinner("Calculando similitud…"):
                    result = analyze(doc1, doc2, check_ia=check_ia)
    # Mostrar resultados solo si no estuvimos en modo 'Comparar contra corpus'
    if mode != "Comparar contra corpus":
        nivel = result["nivel"]
        color = {"Alto": "red", "Medio": "orange", "Bajo": "green"}.get(nivel, "gray")

        st.markdown(f"### Score final: :{color}[**{result['score_final']*100:.1f}%** — {nivel}]")

        c3, c4, c5 = st.columns(3)
        c3.metric("Léxico (TF-IDF)", f"{result['score_lexico']*100:.1f}%")
        c4.metric("MinHash",          f"{result['score_minhash']*100:.1f}%")
        c5.metric("Semántico",        f"{result['score_semantico']*100:.1f}%")

        # Análisis IA
        if check_ia and "analisis_ia" in result:
            st.subheader("Análisis de texto IA")
            for doc_label, ia in [("Doc A", result["analisis_ia"]["doc_a"]),
                                   ("Doc B", result["analisis_ia"].get("doc_b", {}))]:
                ia_color = "red" if ia.get("label") == "Probable IA" else "green"
                st.markdown(
                    f"**{doc_label} — :{ia_color}[{ia.get('label','N/D')}]**  "
                    f"| Perplejidad: `{ia.get('perplexity','N/D')}`  "
                    f"| Variabilidad: `{ia.get('burstiness','N/D')}`  "
                    f"| Score IA: `{ia.get('ai_probability','N/D')}`"
                )

        # Fragmentos similares
        frags = result.get("fragmentos_similares", [])
        st.subheader(f"Fragmentos con alta similitud ({len(frags)} encontrados)")
        if frags:
            for frag in frags[:10]:
                with st.expander(f"Similitud: {frag['score']*100:.1f}%"):
                    fa, fb = st.columns(2)
                    fa.markdown(f"**Doc A**\n\n{frag['frag_doc1']}")
                    fb.markdown(f"**Doc B**\n\n{frag['frag_doc2']}")
        else:
            st.info("No se encontraron fragmentos con similitud alta (umbral: 82%).")

        # Botón de reporte HTML
        st.divider()
        report_path = Path("reporte_temp.html")
        generate_report(result, doc_a=doc1, doc_b=(doc2 or ""), output_path=report_path, corpus_dir=(corpus_dir or None), top_k=top_k, use_gpt2=use_gpt2)
        st.download_button(
            label="⬇️ Descargar reporte HTML",
            data=report_path.read_bytes(),
            file_name="reporte_mini_turnitin.html",
            mime="text/html",
        )
