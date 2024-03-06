# Analyzing Open Access PDFs with Docker and GROBID

This guide provides instructions on how to analyze 10 open access PDFs to extract abstracts, generate word clouds, visualize the number of figures per article, and compile a list of links using a Docker container equipped with GROBID (GeneRation Of BIbliographic Data).

## Prerequisites

- Docker must be installed on your system.
- You should have a directory named `input_pdfs` containing at least 10 PDFs for analysis.
- Ensure you have a Docker image named `grobid2` that is set up with GROBID and necessary scripts.

## Steps to Analyze PDFs

### Step 1: Organize Your PDFs

Prepare a directory in your working space with the PDF files to be analyzed:

your-working-directory/   
│   
├── input_pdfs/   
│ ├── document1.pdf    
│ ├── document2.pdf    
│ ...    
│ └── document10.pdf    


### Step 2: Prepare the Output Directory

Create an output directory where the analysis results will be saved:

your-working-directory/    
│     
├── resources/    
│ └── test_out/   

### Step 3: Execute the Docker Container

Navigate to your working directory in the terminal and run the following Docker command. This mounts your `input_pdfs` and `resources/test_out` directories to the corresponding locations inside the Docker container, allowing GROBID to process the PDFs and save the output.

```sh
docker run --rm -it -v ${PWD}/input_pdfs:/grobid_client_python/input_pdfs -v ${PWD}/resources/test_out:/grobid_client_python/resources/test_out grobid2
```
### Step 4: Accessing the Results

Once the container has processed the PDFs, you'll find the outputs in the `resources/test_out` directory. The results include:

- Word cloud images visualizing the abstract's keywords for each PDF.
- Images showing the visualization of the number of figures in each article.
- A `extracted_links.txt` file listing all the links found within the analyzed PDFs.

## Conclusion

This process allows for an efficient analysis of open access PDFs, leveraging Docker and GROBID to extract meaningful data, such as abstract content, figures, and embedded links. By following these steps, researchers and enthusiasts can gain insights into the content of multiple PDFs quickly and visually.
