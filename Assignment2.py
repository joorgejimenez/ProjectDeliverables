import os
import logging
import string
import requests
import time
from lxml import etree
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
from rdflib import Graph, Literal, RDF, URIRef, Namespace
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definición de los namespaces
EX = Namespace("http://example.org/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
DC = Namespace("http://purl.org/dc/elements/1.1/")
PROV = Namespace("http://www.w3.org/ns/prov#")

# Inicialización del modelo de NER
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Función para limpiar y preprocesar el texto
def clean_and_preprocess_text(text):
    stop_words = set(stopwords.words('english')).union(set('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def clean_text(text):
    return text.strip()

def extract_title_from_tei(tei_file):
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.parse(tei_file)
    title_parts = tree.xpath('//tei:titleStmt/tei:title//text()', namespaces=namespace)
    title_text = ' '.join([clean_text(text) for text in title_parts])
    logging.debug(f"Extracted title text: {title_text}")
    return title_text

def extract_abstract_from_tei(tei_file):
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
    ack_text = ' '.join([clean_text(text) for text in acknowledgement_parts])
    logging.debug(f"Extracted acknowledgement text: {ack_text}")
    return ack_text

def extract_authors_from_tei(tei_file):
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.parse(tei_file)
    authors = []

    # Extracting authors from the analytic section
    author_elements = tree.xpath('//tei:sourceDesc//tei:biblStruct//tei:analytic/tei:author/tei:persName', namespaces=namespace)
    for author in author_elements:
        forename = author.xpath('tei:forename/text()', namespaces=namespace)
        surname = author.xpath('tei:surname/text()', namespaces=namespace)
        if forename and surname:
            full_name = f"{clean_text(forename[0])} {clean_text(surname[0])}"
            authors.append(full_name)

    logging.info(f"Extracted authors: {authors}")
    return authors

def get_sentence_embeddings(text, model):
    embeddings = model.encode([text])
    return embeddings

def compare_similarities(abstracts, model):
    embeddings = model.encode(abstracts)
    similarities = cosine_similarity(embeddings)
    return similarities

def perform_topic_modeling(abstracts, n_topics=3):
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


def fetch_ror_organization_info(org_name):
    url = f"https://api.ror.org/organizations?query={org_name}"
    logging.debug(f"Fetching ROR info for organization: {org_name} with URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from ROR for organization {org_name}: {e}")
        return None

def fetch_wikidata_info(entity_name):
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'language': 'en',
        'format': 'json',
        'search': entity_name
    }
    logging.debug(f"Fetching Wikidata info for entity: {entity_name} with URL: {url}")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get('search', [])
        if results:
            entity_id = results[0].get('id')
            return entity_id
        else:
            logging.warning(f"No results found for {entity_name} in Wikidata")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Wikidata for entity {entity_name}: {e}")
        return None

def enrich_graph_with_external_data(g, authors, organizations):
    for org in organizations:
        ror_info = fetch_ror_organization_info(org)
        if ror_info and ror_info['items']:
            org_uri = EX[org.replace(' ', '_')]
            org_data = ror_info['items'][0]
            g.add((org_uri, EX.ror, Literal(org_data['id'])))
            if 'links' in org_data and org_data['links']:
                g.add((org_uri, FOAF.homepage, Literal(org_data['links'][0])))
            if 'name' in org_data:
                g.add((org_uri, FOAF.name, Literal(org_data['name'])))
            if 'acronyms' in org_data and org_data['acronyms']:
                g.add((org_uri, EX.acronym, Literal(org_data['acronyms'][0])))
            if 'types' in org_data and org_data['types']:
                g.add((org_uri, EX.type, Literal(org_data['types'][0])))
            time.sleep(1)  # Esperar 1 segundo entre solicitudes


    for author in authors:
        wikidata_id = fetch_wikidata_info(author)
        if wikidata_id:
            author_uri = EX[author.replace(' ', '_')]
            wikidata_uri = URIRef(f"https://www.wikidata.org/entity/{wikidata_id}")
            g.add((author_uri, EX.wikidata, wikidata_uri))
            g.add((wikidata_uri, RDF.type, FOAF.Person))
            g.add((wikidata_uri, FOAF.name, Literal(author)))
            time.sleep(1)  # Esperar 1 segundo entre solicitudes

def create_kg(titles, abstracts, acknowledgements, authors, similarities, topics):
    g = Graph()
    g.bind("ex", EX)
    g.bind("foaf", FOAF)
    g.bind("dc", DC)
    g.bind("prov", PROV)

    # Añadir abstracts como nodos de Papers
    for idx, (title, abstract, author_list) in enumerate(zip(titles, abstracts, authors)):
        paper_uri = EX[f"paper{idx}"]
        g.add((paper_uri, RDF.type, EX.Paper))
        g.add((paper_uri, DC.title, Literal(title)))
        g.add((paper_uri, DC.abstract, Literal(abstract)))

        for author in author_list:
            author_uri = EX[author.replace(' ', '_')]
            g.add((paper_uri, DC.creator, author_uri))
            g.add((author_uri, RDF.type, FOAF.Person))
            g.add((author_uri, FOAF.name, Literal(author)))

        # entities = ner(abstract)
        # logging.debug(f"Extracted entities from abstract {idx}: {entities}")
        # for entity in entities:
        #     if entity['entity_group'] == 'ORG':
        #         org_uri = EX[entity['word'].replace(' ', '_')]
        #         g.add((paper_uri, DC.creator, org_uri))
        #         g.add((org_uri, RDF.type, FOAF.Organization))
        #         g.add((org_uri, FOAF.name, Literal(entity['word'])))
        #     elif entity['entity_group'] == 'PER':
        #         person_uri = EX[entity['word'].replace(' ', '_')]
        #         g.add((paper_uri, DC.creator, person_uri))
        #         g.add((person_uri, RDF.type, FOAF.Person))
        #         g.add((person_uri, FOAF.name, Literal(entity['word'])))

    # Añadir acknowledgements a los nodos de Papers
    for idx, (paper_uri, ack_text) in enumerate(zip(g.subjects(RDF.type, EX.Paper), acknowledgements)):
        entities = ner(ack_text)
        logging.info(f"Extracted entities from acknowledgement {idx}: {entities}")
        for entity in entities:
            if entity['entity_group'] == 'ORG':
                org_uri = EX[entity['word'].replace(' ', '_')]
                g.add((paper_uri, EX.acknowledges, org_uri))
                g.add((org_uri, RDF.type, FOAF.Organization))
                g.add((org_uri, FOAF.name, Literal(entity['word'])))
            elif entity['entity_group'] == 'PER':
                person_uri = EX[entity['word'].replace(' ', '_')]
                g.add((paper_uri, EX.acknowledges, person_uri))
                g.add((person_uri, RDF.type, FOAF.Person))
                g.add((person_uri, FOAF.name, Literal(entity['word'])))

    # Añadir temas como nodos de Topics
    for topic_id, keywords in topics:
        topic_uri = EX[f"topic{topic_id}"]
        g.add((topic_uri, RDF.type, EX.Topic))
        for keyword in keywords:
            g.add((topic_uri, DC.subject, Literal(keyword)))

    # Asignar documentos a los temas basados en el modelado de tópicos (LDA)
    count_vectorizer = CountVectorizer()
    X = count_vectorizer.fit_transform(abstracts)
    lda = LatentDirichletAllocation(n_components=len(topics), max_iter=10, learning_method='online', random_state=0)
    lda.fit(X)
    doc_topic_distributions = lda.transform(X)

    for idx, paper_uri in enumerate(g.subjects(RDF.type, EX.Paper)):
        topic_id = doc_topic_distributions[idx].argmax()
        topic_uri = EX[f"topic{topic_id}"]
        g.add((paper_uri, EX.belongs_to_topic, topic_uri))

    # Relacionar documentos entre sí según similitudes
    for idx, paper_uri in enumerate(g.subjects(RDF.type, EX.Paper)):
        for other_idx, sim in enumerate(similarities[idx]):
            if sim > 0.8 and idx != other_idx:  # Umbral de similitud
                other_paper_uri = EX[f"paper{other_idx}"]
                g.add((paper_uri, EX.similar_to, other_paper_uri))

    return g


# Incorporar la extracción de autores en el proceso general
def process_tei_documents(input_dir, output_dir):
    sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    titles = []
    abstracts = []
    acknowledgements = []
    authors = []  # Añadimos una lista para almacenar autores

    try:
        for tei_file in os.listdir(input_dir):
            if tei_file.endswith('.tei.xml'):
                tei_path = os.path.join(input_dir, tei_file)
                logging.info(f"Processing file: {tei_path}")
                
                title_text = extract_title_from_tei(tei_path)
                abstract_text = extract_abstract_from_tei(tei_path)
                ack_text = extract_acknowledgement_from_tei(tei_path)
                authors_text = extract_authors_from_tei(tei_path)  # Extraer autores

                if title_text:
                    titles.append(title_text)
                    logging.info(f"Extracted title: {title_text}")
                else:
                    logging.warning(f"No title found in {tei_path}")

                if abstract_text:
                    abstracts.append(abstract_text)
                    logging.info(f"Extracted abstract: {abstract_text}")
                else:
                    logging.warning(f"No abstract found in {tei_path}")

                if ack_text:
                    acknowledgements.append(ack_text)
                    logging.info(f"Extracted acknowledgements: {ack_text}")
                else:
                    logging.warning(f"No acknowledgements found in {tei_path}")

                if authors_text:
                    authors.append(authors_text)
                    logging.info(f"Extracted authors: {authors_text}")
                else:
                    logging.warning(f"No authors found in {tei_path}")

        if abstracts:
            similarities = compare_similarities(abstracts, sbert_model)
            logging.info(f"Similarity matrix:\n{similarities}")

            clustering = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='complete')
            labels = clustering.fit_predict(similarities)
            logging.info(f"Clustering:\n{labels}")

            lda_topics = perform_topic_modeling(abstracts)
            logging.info(f"LDA Topics:\n{lda_topics}")

            kg = create_kg(titles, abstracts, acknowledgements, authors, similarities, lda_topics)
            enrich_graph_with_external_data(kg, [author for sublist in authors for author in sublist],
                                            list(set([entity['word'] for ack in acknowledgements for entity in ner(ack) if entity['entity_group'] == 'ORG'])))
            output_kg_file = os.path.join(output_dir, 'knowledge_graph.ttl')
            kg.serialize(destination=output_kg_file, format='turtle')
        else:
            logging.info("No abstracts found to compare or perform topic modeling.")

    except Exception as e:
        logging.error(f"Error al procesar documentos: {e}", exc_info=True)

if __name__ == "__main__":
    input_dir = "./resources/test_out/"
    output_dir = "./resources/test_out/"
    process_tei_documents(input_dir, output_dir)