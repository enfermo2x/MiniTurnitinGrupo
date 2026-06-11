# Instrucciones para subir a GitHub y desplegar (Docker / Streamlit Cloud)

Pasos mínimos para poner el proyecto en GitHub y desplegarlo:

1. Inicializar repo local y subir a GitHub:

```bash
git init
git add .
git commit -m "Initial mini-turnitin improvements and deployment files"
gh repo create <tu-usuario>/<tu-repo> --public --source=. --remote=origin
git push -u origin main
```

Reemplaza `<tu-usuario>/<tu-repo>` por tu usuario/repositorio GitHub. Si no usas `gh`, crea
el repositorio en la web y añade el `remote` manualmente.

2. Desplegar con Docker (localmente o en un servicio como Docker Hub / Cloud Run):

```bash
# Construir imagen
docker build -t mini-turnitin:latest .

# Ejecutar localmente
docker run -p 8501:8501 mini-turnitin:latest
```

3. Desplegar en Streamlit Cloud:

- Crea el repo en GitHub (paso 1) y añade `requirements.txt` (ya existe).
- En Streamlit Cloud, conecta tu cuenta de GitHub y selecciona el repo; configura el comando
  de arranque como `streamlit run app.py` si lo piden.

4. Notas sobre recursos y modelos:

- El modelo de Sentence-Transformers descargará pesos la primera vez que se ejecute
  (requiere internet). En entornos limitados considera usar un modelo aún más pequeño
  o predescargar los pesos.
  - Para predescargar los pesos en una máquina con internet ejecuta:

```bash
python scripts/pull_models.py
```

- Para la detección de IA (opcional) el proyecto usa heurísticas locales y no requiere
  modelos grandes. Si activas GPT-2 para perplexity, necesitarás más memoria.
