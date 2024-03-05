# Usar una imagen base de Alpine Linux por ser ligera
FROM alpine:latest

# Instalar Python, Pip y dependencias necesarias para la compilación
RUN apk add --no-cache python3 py3-pip \
    && apk add --no-cache --virtual .build-deps build-base python3-dev \
    && apk add --no-cache libxml2-dev libxslt-dev freetype-dev \
       pkgconfig g++ libpng-dev openblas-dev

# Instalar dependencias adicionales para matplotlib
RUN apk add --no-cache jpeg-dev zlib-dev libjpeg
# Copiar la carpeta grobid_client_python al contenedor
# Asegúrate de que esta carpeta esté en el contexto de construcción de Docker, junto al Dockerfile
COPY grobid_client_python /grobid_client_python

# Cambiar al directorio grobid_client_python
WORKDIR /grobid_client_python

# Instalar las dependencias del cliente GROBID de Python
# Si tu proyecto necesita un archivo requirements.txt para las dependencias de Python, asegúrate de incluirlo en la carpeta grobid_client_python
# Si el proyecto no tiene dependencias fuera de setup.py, ejecuta directamente el script de instalación
RUN python3 setup.py install


# Crear un entorno virtual para las instalaciones de pip
RUN python3 -m venv /opt/venv

# Activar el entorno virtual para las instalaciones siguientes
ENV PATH="/opt/venv/bin:$PATH"

# Instalar matplotlib, wordcloud, y lxml con pip dentro del entorno virtual
RUN pip install matplotlib wordcloud lxml
# Instalar las dependencias del cliente GROBID de Python
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Limpiar las dependencias de compilación no necesarias después de la instalación
RUN apk del .build-deps

# Copiar el script de Python específico (si es diferente de lo que ya está en grobid_client_python)
# Esta línea es opcional si ya tienes tu script en la carpeta grobid_client_python antes de copiarla
COPY Grobid.py /grobid_client_python/Grobid.py
COPY input_pdfs /grobid_client_python/input_pdfs
# Hacer el script ejecutable, ajusta el nombre del script si es necesario
RUN chmod +x /grobid_client_python/Grobid.py

# Configurar el directorio de trabajo para cualquier comando RUN, CMD, ENTRYPOINT posterior
WORKDIR /grobid_client_python

# Comando para ejecutar el script de Python cuando el contenedor inicie
ENTRYPOINT ["python3", "Grobid.py"]
