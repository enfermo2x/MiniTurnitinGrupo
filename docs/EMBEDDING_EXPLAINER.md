# Explicación técnica del embedding usado en Mini-Turnitin

Este proyecto utiliza `sentence-transformers` con el modelo por defecto:

- `paraphrase-multilingual-MiniLM-L12-v2`

Resumen técnico que puedes presentar en la exposición:

- **Tipo de arquitectura:** MiniLM (distilled transformer) adaptado a Sentence-BERT.
- **Capas / tamaño:** el sufijo `L12` indica que la arquitectura central está basada
  en 12 capas Transformer (modelo ligero y eficiente). El modelo fue entrenado
  para obtener embeddings de oraciones útiles en tareas de paraphrase y similitud.
- **Pooling:** `sentence-transformers` aplica una etapa de pooling (media o `mean pooling`)
  sobre las representaciones de tokens para obtener un vector por oración/documento.
- **Backend / librerías:** el pipeline usa `sentence-transformers` sobre HuggingFace
  Transformers; el backend por defecto es PyTorch (`torch` está en `requirements.txt`).
- **No es Word2Vec:** no se usan embeddings estáticos como Word2Vec o GloVe; se emplea
  una red Transformer contextual que produce vectores por oración.
- **Uso de SciPy:** `scipy` no es necesaria en el pipeline principal; las similitudes se
  calculan con producto punto en NumPy y funciones de `sentence-transformers`.

Puntos clave para la exposición (sugeridos):

- Menciona el nombre exacto del modelo `paraphrase-multilingual-MiniLM-L12-v2`.
- Indica que usas `torch` y `sentence-transformers` para cargar el modelo y generar embeddings.
- Explica brevemente qué es un Transformer (capas de atención), que el modelo es una
  versión pequeña/distilled adecuada para producción docente (menor memoria y latencia).
- Señala que el embedding toma la secuencia de tokens y aplica mean-pooling para
  obtener un vector fijo por oración/documento; la similitud se calcula por producto
  punto (coseno si los embeddings se normalizan).
