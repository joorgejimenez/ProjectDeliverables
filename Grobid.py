import json
import os
from grobid_client.grobid_client import GrobidClient
from lxml import etree
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def process_documents(input_dir, output_dir):
    client = GrobidClient(config_path="./config.json")
    client.process("processFulltextDocument", input_dir, output=output_dir, consolidate_citations=True, tei_coordinates=True, force=True)

def extract_abstracts(xml_files_dir):
    abstract_texts = []
    for filename in os.listdir(xml_files_dir):
        if filename.endswith(".xml"):
            path = os.path.join(xml_files_dir, filename)
            tree = etree.parse(path)
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            abstract = tree.xpath('//tei:abstract/tei:p', namespaces=namespaces)
            abstract_text = " ".join([p.text for p in abstract if p.text])
            abstract_texts.append(abstract_text)
    return abstract_texts

def create_wordcloud(texts):
    full_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400).generate(full_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def count_figures(xml_files_dir):
    figures_per_article = []
    for filename in os.listdir(xml_files_dir):
        if filename.endswith(".xml"):
            path = os.path.join(xml_files_dir, filename)
            tree = etree.parse(path)
            figures = tree.xpath('//tei:figure', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
            figures_per_article.append((filename, len(figures)))
    return figures_per_article

def visualize_figures_count(figures_count):
    articles = [item[0] for item in figures_count]
    counts = [item[1] for item in figures_count]
    plt.bar(articles, counts)
    plt.xlabel('Articles')
    plt.ylabel('Number of Figures')
    plt.xticks(rotation=45)
    plt.show()

def extract_links(xml_files_dir):
    links_per_article = {}
    for filename in os.listdir(xml_files_dir):
        if filename.endswith(".xml"):
            path = os.path.join(xml_files_dir, filename)
            tree = etree.parse(path)
            links = tree.xpath('//tei:ref[@type="url"]', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
            links_per_article[filename] = [link.text for link in links if link.text]
    return links_per_article

if __name__ == "__main__":
    input_dir = "./input_pdfs"
    output_dir = "./resources/test_out/"
    process_documents(input_dir, output_dir)
    abstracts = extract_abstracts(output_dir)
    create_wordcloud(abstracts)
    figures_count = count_figures(output_dir)
    visualize_figures_count(figures_count)
    links = extract_links(output_dir)
    print(links)