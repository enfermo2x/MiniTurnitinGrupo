# Informe de mejoras — Mini-Turnitin

Fecha: 2026-06-10

Resumen de objetivos cumplidos:

- Auditar el código y dependencias del proyecto.
- Robustecer el preprocesado para entornos sin spaCy.
- Mejorar la gestión de embeddings: carga perezosa y protecciones contra entradas vacías.
- Añadir Dockerfile y `.dockerignore` para despliegue por contenedor.
- Generar documentación técnica para la exposición centrada en el embedding.
- Incluir instrucciones claras para subir a GitHub y desplegar en Docker / Streamlit Cloud.

Cambios realizados (archivos modificados / añadidos):

- `src/semantic.py`: carga del modelo `SentenceTransformer` ahora es perezosa
  (`get_model()`), funciones más robustas ante entradas vacías y explicación del
  modelo por defecto (`paraphrase-multilingual-MiniLM-L12-v2`).
- `src/preprocess.py`: `spaCy` es opcional; si no está instalado se usa un
  preprocesado sencillo (regex) que permite ejecutar la app sin modelos grandes.
- `Dockerfile`, `.dockerignore`: archivos para crear una imagen de Docker que
  lance `streamlit run app.py` en producción/local.
- `docs/EMBEDDING_EXPLAINER.md`: documento específico para la exposición del
  embedding (arquitectura, capas, librerías usadas).
- `docs/DEPLOY_AND_GITHUB.md`: comandos y pasos para subir el repo a GitHub y
  desplegar en Docker / Streamlit Cloud.
- `docs/FULL_REPORT.md`: este documento.

Detalles técnicos relevantes (para la exposición):

- Embedding usado: `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers).
  - Arquitectura: MiniLM (modelo transformer distilado). El sufijo `L12`
    indica el diseño sobre 12 capas transformer en su encoder.
  - Pooling: mean-pooling sobre las representaciones de tokens para obtener
    un vector por oración/documento.
  - Backend: PyTorch (`torch`) a través de HuggingFace / sentence-transformers.
  - No se usan Word2Vec ni embeddings estáticos; el modelo es contextual.
  - SciPy no es necesario en el pipeline principal; se usan NumPy y operaciones
    de producto punto para calcular similitudes.

Recomendaciones para la presentación (qué decir cuando te pregunten):

- "Usamos `sentence-transformers` con el modelo `paraphrase-multilingual-MiniLM-L12-v2`.
  Es un transformer distilado con 12 capas; producimos embeddings por sentencia
  con pooling y calculamos la similitud por producto punto."
- "El backend es PyTorch (sí usamos `torch`) — la inferencia se realiza con
  `model.encode()` y los embeddings están normalizados antes de medir la similitud."
- "No usamos Word2Vec; los embeddings son contextualizados y capturan significado
  más allá de coincidencias léxicas."

Pasos siguientes que puedo ejecutar por ti (elige uno):

1. Preparar un repo Git local y ayudarte a ejecutar `gh repo create` y `git push`.
2. Construir y probar localmente la imagen Docker (`docker build`) en este equipo.
3. Añadir una opción en la UI de `app.py` para seleccionar el modelo de embedding.
4. Crear un script para predescargar los pesos del modelo y evitar la descarga en la primera ejecución.

Si quieres que haga alguna de estas acciones, dime cuál y la ejecuto.
