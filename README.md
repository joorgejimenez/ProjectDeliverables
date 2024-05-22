# Project Deliverables: Artificial Intelligence and Open Science in Research Software Engineering


## Getting Started

Follow these steps to clone the necessary repository, but we skip the Grobid Analysis part(already covered in 1st deliverable), so we already share in the repo the tei.xml as an input in the resources/test_out dir. Anyways if you want to change papers and do the Grobid analysis see Apendix.  

### 1. Clone the  Repository


```bash
git clone https://github.com/joorgejimenez/ProjectDeliverables.git
```

### 2. Build and Run the Client

Build the Docker container with the following command:

```bash
docker build -t grobid2 .
```


After building the container, run it using:

```bash
docker run --rm -it -v ${PWD}/input_pdfs:/grobid_client_python/input_pdfs -v ${PWD}/resources/test_out:/grobid_client_python/resources/test_out grobid2
```
You will need to wait a little in order to fully complete both commands due to the API requests and the downloads of several packages in the container. 

## 3: Ejecutar el comando Docker para lanzar el servidor SPARQL

Primero, ejecuta el siguiente comando para iniciar el contenedor que contiene el servidor SPARQL:

```bash
docker run -it --rm -p 3030:3030 stain/jena-fuseki
```
## 4: Consultas SPARQL

En el script consultas_sparql tiene los ejemplos de la demo si desea usarlos


### 5. Citing This Repository

If our work assists or inspires your research, consider citing us:

[![DOI](https://zenodo.org/badge/753741900.svg)](https://zenodo.org/badge/latestdoi/753741900)



### APENDIX(GROBID) 
## Getting Started
First you need to uncomment in the Dockerfile the last command and comment the ENTRYPOINT ["python", "Assignment2.py"] command in order to do both tasks sequentally.           
### 1. Clone the GROBID Client Repository

First, clone the `grobid_client_python` repository by `kermitt2` into your local machine using the following command:

```bash
git clone https://github.com/kermitt2/grobid_client_python.git
```

### 2. Edit the Configuration

Navigate into the cloned repository and edit the `config.json` file. You will need to change the `grobid_server` URL to the following:

```json
{
  "grobid_server": "http://host.docker.internal:8070"
}
```

### 3. Setting up the GROBID Server with Docker

With Docker Desktop running, execute the following commands in your terminal to pull the GROBID Docker image and run the server:

```bash
docker pull lfoppiano/grobid:0.7.2
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2              


