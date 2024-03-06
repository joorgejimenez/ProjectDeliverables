import json
import os
import logging
from grobid_client.grobid_client import GrobidClient
from lxml import etree
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re  # Importado para la limpieza de texto

# Configura el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_info_from_tei(tei_file):
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.parse(tei_file)
    
    # Ajustando la consulta XPath para coincidir con tu estructura de abstract
    abstract_text = " ".join(tree.xpath('//tei:profileDesc/tei:abstract//tei:p/text()', namespaces=namespace))
    
    figures = tree.xpath('//tei:figure', namespaces=namespace)
    num_figures = len(figures)
    
    links = tree.xpath('//tei:ptr/@target', namespaces=namespace)

    return abstract_text, num_figures, links


def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Elimina caracteres especiales
    text = re.sub(r'\s+', ' ', text).strip()  # Elimina espacios extra
    return text

def generate_wordcloud(text, output_filename):
    cleaned_text = clean_text(text)  # Limpia el texto
    if not cleaned_text:
        logging.warning(f"No se puede generar una nube de palabras para {output_filename} porque el texto está vacío.")
        return
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(output_filename)
    plt.close()

def process_documents(input_dir, output_dir):
    logging.info("Iniciando el procesamiento de documentos...")
    client = GrobidClient(config_path="./config.json")
    try:
        client.process("processFulltextDocument", input_dir, output=output_dir, consolidate_citations=True, tei_coordinates=True, force=True)
        logging.info("Documentos procesados exitosamente.")
    except Exception as e:
        logging.error(f"Error al procesar documentos: {e}")

if __name__ == "__main__":
    input_dir = "./input_pdfs"
    output_dir = "./resources/test_out/"
    process_documents(input_dir, output_dir)
    
    figures_per_article, all_links = [], []
    for tei_file in os.listdir(output_dir):
        if tei_file.endswith('.tei.xml'):
            tei_path = os.path.join(output_dir, tei_file)
            logging.info(f"Texto del abstract antes de la limpieza: '{tei_path}'")
            abstract_text, num_figures, links = extract_info_from_tei(tei_path)
            logging.info(f"Texto del abstract antes de la limpieza: '{abstract_text}'")
            wc_output_filename = os.path.join(output_dir, f"{tei_file}_wordcloud.png")
            generate_wordcloud(abstract_text, wc_output_filename)
            figures_per_article.append(num_figures)
            all_links.extend(links)
    
    # Visualización y almacenamiento de datos
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(figures_per_article)), figures_per_article)
    plt.xlabel('Artículos')
    plt.ylabel('Número de figuras')
    plt.title('Número de figuras por artículo')
    plt.savefig(os.path.join(output_dir, "figures_per_article.png"))
    plt.close()
    
    links_filename = os.path.join(output_dir, "extracted_links.txt")
    with open(links_filename, "w") as f:
        for link in all_links:
            f.write(f"{link}\n")
