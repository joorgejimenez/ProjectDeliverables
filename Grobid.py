import json
import os
from grobid_client.grobid_client import GrobidClient
    
if __name__ == "__main__":
    client = GrobidClient(config_path="./config.json")
    client.process("processFulltextDocument", "./resources/test_pdf", output="./resources/test_out/", consolidate_citations=True, tei_coordinates=True, force=True)
# Agregando el código de verificación aquí
    output_dir = "./resources/test_out/"
    if os.listdir(output_dir):
        print("Archivos encontrados en el directorio de salida:")
        for filename in os.listdir(output_dir):
            if filename.endswith(".xml"):
                print(f"Archivo XML encontrado: {filename}")
            else:
                print(f"Archivo no XML encontrado: {filename}")
    else:
        print("No se encontraron archivos en el directorio de salida. Por favor, verifica que el proceso de GROBID se haya completado correctamente.")