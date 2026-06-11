FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para algunas librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando por defecto: lanzar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
