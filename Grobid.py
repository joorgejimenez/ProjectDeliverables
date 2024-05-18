import os
import logging
import string
from grobid_client.grobid_client import GrobidClient
from lxml import etree
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rdflib import Graph, Literal, RDF, URIRef, Namespace
import torch

nltk.download('stopwords')
nltk.download('wordnet')

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Función para limpiar y preprocesar el texto
def clean_and_preprocess_text(text):
    """Limpia y preprocesa el texto eliminando stop words, lematizando, y eliminando puntuación."""
    stop_words = set(stopwords.words('english')).union(set('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenizar y lematizar
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def clean_text(text):
    """Limpia el texto eliminando espacios al principio y al final."""
    return text.strip()

def extract_abstract_from_tei(tei_file):
    """Extrae el resumen (abstract) de un archivo TEI XML."""
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.parse(tei_file)
    abstract_parts = tree.xpath('//tei:abstract//text()', namespaces=namespace)
    abstract_text = ' '.join([clean_and_preprocess_text(clean_text(text)) for text in abstract_parts])
    logging.debug(f"Extracted abstract text: {abstract_text}")
    return abstract_text

def extract_acknowledgement_from_tei(tei_file):
    """Extrae la sección de acknowledgements de un archivo TEI XML."""
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.parse(tei_file)
    acknowledgement_parts = tree.xpath('//tei:div[@type="acknowledgement"]//text()', namespaces=namespace)
    acknowledgement_text = ' '.join([clean_text(text) for text in acknowledgement_parts])
    logging.debug(f"Extracted acknowledgement text: {acknowledgement_text}")
    return acknowledgement_text

def enrich_data(text):
    """Enriquece el texto con el modelo de NER de Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")      
    ner_model = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    entities = ner_model(text)
    
    # Imprimir la estructura de las entidades devueltas
    logging.info(f"Estructura de las entidades devueltas: {entities}")
    
    return entities

def get_sentence_embeddings(text, model):
    """Obtiene embeddings de oraciones utilizando sentence-transformers."""
    embeddings = model.encode([text])
    return embeddings

def compare_similarities(abstracts, model):
    """Compara similitudes entre los resúmenes (abstracts) utilizando embeddings."""
    embeddings = model.encode(abstracts)
    similarities = cosine_similarity(embeddings)
    return similarities

def perform_topic_modeling(abstracts, n_topics=2):
    """Realiza topic modeling utilizando LDA en los resúmenes (abstracts)."""
    count_vectorizer = CountVectorizer()
    X = count_vectorizer.fit_transform(abstracts)
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', random_state=0)
    lda.fit(X)
    feature_names = count_vectorizer.get_feature_names_out()
    topics = []
    for topic_id, topic in enumerate(lda.components_):
        topic_keywords = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append((topic_id, topic_keywords))
    return topics

def extract_relations(entities):
    """Extrae relaciones entre las entidades reconocidas en el texto."""
    relations = []
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:
                # Verificar relaciones de agradecimiento
                if entity1['entity_group'] == 'PER' and entity2['entity_group'] == 'ORG':
                    relations.append((entity1['word'], 'acknowledges', entity2['word']))
                # Verificar relaciones de financiamiento
                elif entity1['entity_group'] == 'ORG' and entity2['entity_group'] == 'MISC':
                    relations.append((entity1['word'], 'funds', entity2['word']))
                # Verificar colaboraciones
                elif entity1['entity_group'] == 'PER' and entity2['entity_group'] == 'PER':
                    relations.append((entity1['word'], 'collaborates_with', entity2['word']))
                # Verificar relaciones de localización
                elif (entity1['entity_group'] == 'PER' or entity1['entity_group'] == 'ORG') and entity2['entity_group'] == 'LOC':
                    relations.append((entity1['word'], 'located_in', entity2['word']))
    return relations

def create_knowledge_graph(entities, relations, output_dir):
    """Crea un Knowledge Graph a partir de las entidades y relaciones reconocidas."""
    g = Graph()
    n = Namespace("http://example.org/ns#")
    
    # Añadir entidades al grafo
    for entity in entities:
        entity_uri = URIRef(f"http://example.org/entity/{entity['word'].replace(' ', '_')}")
        g.add((entity_uri, RDF.type, n.Entity))
        g.add((entity_uri, n.text, Literal(entity['word'])))
    
    # Añadir relaciones al grafo
    for subj, pred, obj in relations:
        subj_uri = URIRef(f"http://example.org/entity/{subj.replace(' ', '_')}")
        obj_uri = URIRef(f"http://example.org/entity/{obj.replace(' ', '_')}")
        g.add((subj_uri, n[pred], obj_uri))
    
    graph_path = os.path.join(output_dir, "knowledge_graph.rdf")
    g.serialize(destination=graph_path, format='xml')
    logging.info(f"Knowledge Graph has been saved to {graph_path}")

def process_documents(input_dir, output_dir):
    """Procesa documentos usando GROBID y enriquece la información extraída."""
    client = GrobidClient(config_path="./config.json")
    sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    abstracts = []
    full_texts = []
    all_entities = []
    all_relations = []
    
    try:
        client.process("processFulltextDocument", input_dir, output=output_dir, consolidate_citations=True, tei_coordinates=True, force=True)
        logging.info("Documentos procesados exitosamente.")
        for tei_file in os.listdir(output_dir):
            if tei_file.endswith('.tei.xml'):
                tei_path = os.path.join(output_dir, tei_file)
                logging.info(f"Processing file: {tei_path}")
                
                # Extraer acknowledgement y abstract
                full_text = extract_acknowledgement_from_tei(tei_path)
                abstract_text = extract_abstract_from_tei(tei_path)
                
                if full_text:
                    full_texts.append(full_text)
                    entities = enrich_data(full_text)
                    all_entities.extend(entities)
                    relations = extract_relations(entities)
                    all_relations.extend(relations)
                    logging.info(f"Extracted acknowledgement: {full_text}")
                    logging.info(f"Extracted entities: {entities}")
                    logging.info(f"Extracted relations: {relations}")
                else:
                    logging.warning(f"No acknowledgement found in {tei_path}")
                    
                if abstract_text:
                    abstracts.append(abstract_text)
                    logging.info(f"Extracted abstract: {abstract_text}")
                else:
                    logging.warning(f"No abstract found in {tei_path}")
                    
        if abstracts:
            similarities = compare_similarities(abstracts, sbert_model)
            logging.info(f"Similarity matrix:\n{similarities}")

            # Perform topic modeling using LDA
            lda_topics = perform_topic_modeling(abstracts)
            
            logging.info(f"LDA Topics:\n{lda_topics}")
        else:
            logging.info("No abstracts found to compare or perform topic modeling.")
        
        # Crear y guardar el Knowledge Graph
        create_knowledge_graph(all_entities, all_relations, output_dir)
    except Exception as e:
        logging.error(f"Error al procesar documentos: {e}")

if __name__ == "__main__":
    input_dir = "./input_pdfs"
    output_dir = "./resources/test_out/"
    process_documents(input_dir, output_dir)
