# ProjectDeliverables
Github repository for Artifical Intelligence and Open Science In Research Software Engineering 

First you need to git clone the git repository by kermitt2 into this git with the command line --> git clone https://github.com/kermitt2/grobid_client_python.git   
Second you need to edit the file config.json from the recently cloned git repository and change the line to "grobid_server: "http://host.docker.internal:8070","   
Third you need to open Docker Desktop and run th grobid server with this 2 commands: docker pull lfoppiano/grobid:0.7.2   
                                                                                    docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2   
And finally you need to run the client with this 2 commands: docker build -t grobid .   
                                                             docker run --rm -it grobid    



             [![DOI](https://zenodo.org/badge/{753741900}.svg)](https://zenodo.org/badge/latestdoi/{753741900})  
             
              


