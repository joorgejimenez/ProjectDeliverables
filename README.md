# Project Deliverables: Artificial Intelligence and Open Science in Research Software Engineering

This repository guides you through the setup of a research environment that leverages the power of Artificial Intelligence with the GROBID tool, focusing on Open Science principles. GROBID (GeneRation Of BIbliographic Data) leverages machine learning to provide document analysis and metadata extraction.

## Getting Started

Follow these steps to clone the necessary repository, set up the GROBID client, and run the GROBID server using Docker.

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
```

### 4. Build and Run the Client

To interact with the GROBID server, navigate to the directory of the cloned `grobid_client_python` repository. Build the Docker container for the GROBID client with the following command:

```bash
docker build -t grobid .
```
If in this point you get an error of this kind
```bash
 => ERROR [internal] load metadata for docker.io/library/alpine:latest                                                                                                                                   0.5s 
------
 > [internal] load metadata for docker.io/library/alpine:latest:
------
Dockerfile:2
--------------------
   1 |     # Usar una imagen base de Alpine Linux por ser ligera
   2 | >>> FROM alpine:latest
   3 |
   4 |     # Instalar Python, Pip y dependencias necesarias para la compilación
--------------------
ERROR: failed to solve: alpine:latest: error getting credentials - err: exec: "docker-credential-desktop": executable file not found in %PATH%, out: ``
```
You may edit the `config.json` of your `.docker/` file directory and remove the `s` from `credsStore` to `credStore`

After building the container, run it using:

```bash
docker run --rm -it grobid
```

### 5. Citing This Repository

If our work assists or inspires your research, consider citing us:

[![DOI](https://zenodo.org/badge/753741900.svg)](https://zenodo.org/badge/latestdoi/753741900)
             
              


