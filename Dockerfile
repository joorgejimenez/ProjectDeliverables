# Usar una imagen base de Python 3.8 basada en Debian
FROM python:3.8-slim-buster

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libopenblas-dev \
    libpng-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear un entorno virtual para las instalaciones de pip
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar TensorFlow, Transformers y Sentence Transformers
 RUN pip install --no-cache-dir --upgrade pip && \
 pip install --no-cache-dir lxml transformers==4.33.3 rdflib sentence-transformers==2.2.2 scikit-learn beautifulsoup4 torch

# Copiar el directorio del cliente GROBID y el archivo de requisitos al contenedor
COPY grobid_client_python /grobid_client_python
COPY requirements.txt /grobid_client_python

# Cambiar al directorio grobid_client_python y instalar dependencias adicionales
WORKDIR /grobid_client_python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos adicionales si son necesarios
COPY Grobid.py input_pdfs /grobid_client_python/

# Hacer el script ejecutable
RUN chmod +x /grobid_client_python/Grobid.py

# Configurar el directorio de trabajo para cualquier comando RUN, CMD, ENTRYPOINT posterior
WORKDIR /grobid_client_python

# Comando para ejecutar el script de Python cuando el contenedor inicie
ENTRYPOINT ["python", "Grobid.py"]