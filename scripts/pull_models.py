#!/usr/bin/env python3
"""Script sencillo para predescargar pesos del modelo de embeddings usado.

Ejecuta: `python scripts/pull_models.py` para forzar la descarga de
`paraphrase-multilingual-MiniLM-L12-v2` y evitar la latencia en la primera
ejecución de la app.
"""
from sentence_transformers import SentenceTransformer


def main():
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Descargando modelo de embeddings: {model_name} ...")
    SentenceTransformer(model_name)
    print("Descarga completada.")


if __name__ == "__main__":
    main()
