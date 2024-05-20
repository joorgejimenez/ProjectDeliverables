import os
import logging
import string
from grobid_client.grobid_client import GrobidClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def process_documents(input_dir, output_dir):
    """Procesa documentos usando GROBID y enriquece la información extraída."""
    client = GrobidClient(config_path="./config.json")

    
    try:
        client.process("processFulltextDocument", input_dir, output=output_dir, consolidate_citations=True, tei_coordinates=True, force=True)
        logging.info("Documentos procesados exitosamente.")
        for tei_file in os.listdir(output_dir):
            if tei_file.endswith('.tei.xml'):
                tei_path = os.path.join(output_dir, tei_file)
                logging.info(f"Processing file: {tei_path}")
                

    except Exception as e:
        logging.error(f"Error al procesar documentos: {e}")

if __name__ == "__main__":
    input_dir = "./input_pdfs"
    output_dir = "./resources/test_out/"
    process_documents(input_dir, output_dir)






