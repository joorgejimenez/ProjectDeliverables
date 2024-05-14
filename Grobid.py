import os
import logging
from grobid_client.grobid_client import GrobidClient
from lxml import etree
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bs4 import BeautifulSoup
import re

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """Limpia el texto eliminando caracteres especiales y espacios extra."""
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_abstract_from_tei(tei_file):
    """Extrae el resumen (abstract) de un archivo TEI XML."""
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.parse(tei_file)
    abstract_parts = tree.xpath('//tei:abstract//text()', namespaces=namespace)
    abstract_text = ' '.join([clean_text(text) for text in abstract_parts])
    return abstract_text

def extract_full_text_from_tei(tei_file):
    """Extrae el texto completo de un archivo TEI XML usando BeautifulSoup."""
    with open(tei_file, 'r', encoding='utf-8') as file:
        tei_content = file.read()
    soup = BeautifulSoup(tei_content, 'xml')
    text_elements = soup.find_all('p')
    text_content = " ".join([element.get_text() for element in text_elements])
    return clean_text(text_content)

def enrich_data(text):
    """Enriquece el texto con el modelo de NER de Hugging Face."""
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = ner_model(text)
    return entities

def extract_relations(text, entities):
    """Extrae relaciones entre las entidades identificadas usando un modelo de RE de Hugging Face."""
    re_model = pipeline("ner", model="allenai/scibert_scivocab_cased")
    relations = []
    for i, entity in enumerate(entities):
        for j, other_entity in enumerate(entities):
            if i != j:
                relation_input = f"{text[entity['start']:entity['end']]} [SEP] {text[other_entity['start']:other_entity['end']]}"
                relation = re_model(relation_input)
                if relation:
                    relations.append((entity, other_entity, relation))
    return relations

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
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)

    feature_names = count_vectorizer.get_feature_names_out()
    topics = []
    for topic_id, topic in enumerate(lda.components_):
        topic_keywords = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append((topic_id, topic_keywords))
    return topics

# Las siguientes funciones están comentadas ya que no son necesarias para la comparación de similitud

# def create_knowledge_graph(entities, cluster_labels, output_dir):
#     """Crea un Knowledge Graph a partir de las entidades reconocidas."""
#     g = Graph()
#     n = Namespace("http://example.org/ns#")
#     for index, entity in enumerate(entities):
#         entity_uri = URIRef(f"http://example.org/entity/{entity['word'].replace(' ', '_')}")
#         g.add((entity_uri, RDF.type, n.Entity))
#         g.add((entity_uri, n.text, Literal(entity['word'])))
#         cluster_uri = URIRef(f"http://example.org/cluster/{cluster_labels[index]}")
#         g.add((entity_uri, n.isPartOfCluster, cluster_uri))
#     graph_path = os.path.join(output_dir, "knowledge_graph.rdf")
#     g.serialize(destination=graph_path, format='xml')
#     logging.info(f"Knowledge Graph has been saved to {graph_path}")

# def get_sentence_embeddings(text):
#     """Obtiene embeddings de oraciones utilizando sentence-transformers."""
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode([text])  # Lista de textos para procesar
#     return embeddings

# def cluster_embeddings(embeddings, num_clusters=5):
#     """Aplica clustering a los embeddings."""
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
#     return kmeans.labels_

def process_documents(input_dir, output_dir):
    """Procesa documentos usando GROBID y enriquece la información extraída."""
    client = GrobidClient(config_path="./config.json")
    sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    abstracts = []
    full_texts = []
    try:
        client.process("processFulltextDocument", input_dir, output=output_dir, consolidate_citations=True, tei_coordinates=True, force=True)
        logging.info("Documentos procesados exitosamente.")
        for tei_file in os.listdir(output_dir):
            if tei_file.endswith('.tei.xml'):
                tei_path = os.path.join(output_dir, tei_file)
                full_text = extract_full_text_from_tei(tei_path)
                abstract_text = extract_abstract_from_tei(tei_path)
                if full_text:
                    full_texts.append(full_text)
                    entities = enrich_data(full_text)
                    relations = extract_relations(full_text, entities)
                    logging.info(f"Entities: {entities}")
                    logging.info(f"Relations: {relations}")
                if abstract_text:
                    abstracts.append(abstract_text)
                    
        if abstracts:
            similarities = compare_similarities(abstracts, sbert_model)
            logging.info(f"Similarity matrix:\n{similarities}")

            topics = perform_topic_modeling(abstracts)
            for topic_id, keywords in topics:
                logging.info(f"Topic {topic_id}: {' '.join(keywords)}")
        else:
            logging.info("No abstracts found to compare or perform topic modeling.")
    except Exception as e:
        logging.error(f"Error al procesar documentos: {e}")

if __name__ == "__main__":
    input_dir = "./input_pdfs"
    output_dir = "./resources/test_out/"
    process_documents(input_dir, output_dir)
