# Documento de Rationale para el Proyecto de Procesamiento de TEI

## Contexto

El proyecto se centra en la extracción, procesamiento y enriquecimiento de información contenida en archivos TEI (Text Encoding Initiative). La finalidad es transformar estos datos en un grafo de conocimiento que puede ser utilizado para análisis posteriores, utilizando técnicas de Procesamiento de Lenguaje Natural (PLN) y aprendizaje automático.

## Alternativas Consideradas

1. **Procesamiento manual de los archivos TEI**:
    - Ventajas: Control total sobre la precisión de la extracción.
    - Desventajas: Requiere mucho tiempo y es propenso a errores humanos.

2. **Utilización de herramientas de extracción existentes**:
    - Ventajas: Rapidez en la implementación.
    - Desventajas: Puede no ajustarse a las necesidades específicas del proyecto.

3. **Desarrollo de un pipeline automatizado**:
    - Ventajas: Eficiencia en el procesamiento y capacidad de manejar grandes volúmenes de datos.
    - Desventajas: Requiere conocimiento técnico y desarrollo inicial.

## Decisión Tomada

Desarrollar un pipeline automatizado utilizando Python y varias bibliotecas de PLN y aprendizaje automático para extraer, procesar y enriquecer la información de los archivos TEI.

## Justificación

- **Eficiencia**: Un pipeline automatizado permite procesar grandes cantidades de archivos TEI de manera eficiente y consistente.
- **Flexibilidad**: Permite ajustes y mejoras continuas en los métodos de extracción y procesamiento.
- **Enriquecimiento**: La capacidad de integrar datos externos (por ejemplo, Wikidata, ROR) en el grafo de conocimiento añade valor significativo.

## Implementación

### Configuración y Dependencias

Se utilizan varias bibliotecas y herramientas para la implementación:
- `nltk`: Para la lematización y manejo de stopwords.
- `transformers`: Para la extracción de entidades nombradas (NER).
- `sentence-transformers`: Para generar embeddings de oraciones.
- `scikit-learn`: Para el cálculo de similitudes, modelado de tópicos y clustering.
- `rdflib`: Para la creación y manejo del grafo de conocimiento RDF.
- `requests`: Para la obtención de datos externos (Wikidata, ROR).

### Pasos de Implementación

1. **Configuración del Logging**: Para monitorear el flujo del proceso y registrar mensajes informativos y de error.
2. **Definición de Namespaces**: Utilizados para estructurar el grafo RDF.
3. **Inicialización del Modelo NER**: Para la identificación de entidades nombradas en el texto.
4. **Funciones de Limpieza y Preprocesamiento**:
    - Limpieza del texto y eliminación de puntuación.
    - Lematización y eliminación de stopwords.

### Extracción de Información de Archivos TEI

Se implementan funciones específicas para extraer diferentes secciones de los archivos TEI:
- **Título**
- **Resumen**
- **Agradecimientos**
- **Autores**

### Generación de Embeddings y Comparación de Similitudes

- **Embeddings de Oraciones**: Utilizando `sentence-transformers`.
- **Cálculo de Similitudes**: Utilizando `cosine_similarity` de `scikit-learn`.

### Modelado de Tópicos y Clustering

- **Modelado de Tópicos**: Utilizando `LatentDirichletAllocation` para identificar temas dentro de los resúmenes.
- **Clustering**: Agrupación de resúmenes similares utilizando `AgglomerativeClustering`.

### Enriquecimiento del Grafo RDF

- **Integración de Datos Externos**: Obtención y adición de datos de Wikidata y ROR al grafo RDF.
- **Creación del Grafo RDF**: Estructuración y almacenamiento del grafo en formato Turtle.

### Función Principal

La función `process_tei_documents` gestiona el flujo general del proceso:
1. Procesa los archivos TEI en el directorio de entrada.
2. Extrae la información necesaria.
3. Genera similitudes y realiza el modelado de tópicos.
4. Enriquecer y almacenar el grafo de conocimiento resultante.

## Implicaciones

- **Ventajas**:
    - Procesamiento automatizado y eficiente de grandes volúmenes de datos.
    - Enriquecimiento del grafo de conocimiento con datos externos.
    - Estructuración clara y reutilizable del proceso.

- **Desventajas**:
    - Complejidad en la configuración inicial y dependencias.
    - Necesidad de manejo de errores y excepciones durante la obtención de datos externos.

## Revisión y Validación

El pipeline automatizado ha sido validado mediante pruebas con diferentes archivos TEI, mostrando una alta precisión en la extracción y procesamiento de la información. El grafo de conocimiento resultante ha sido enriquecido con éxito con datos externos, proporcionando una base sólida para análisis posteriores.

Este conjunto de decisiones y justificaciones proporciona una guía clara sobre el "por qué" y el "cómo" detrás de la implementación del proyecto, facilitando su comprensión y mantenimiento futuro.
