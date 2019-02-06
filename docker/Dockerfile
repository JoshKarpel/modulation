FROM continuumio/miniconda3:latest

# update & upgrade
RUN apt-get -y update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN conda install -y python=3.7 \
                     numpy \
                     scipy \
                     matplotlib \
                     cython \
                     paramiko \
                     psutil \
                     tqdm \
                     cloudpickle=0.6.1 \
    && \
    conda update -y --all && \
    conda clean -y --all

ARG CACHEBUST=1

# my packages
RUN pip install htcondor==8.8.2b1 \
                git+https://github.com/JoshKarpel/htmap.git \
                git+https://github.com/JoshKarpel/simulacra.git \
                git+https://github.com/JoshKarpel/modulation.git

# pre-compile cython code via pyximport
RUN python3 -c "import modulation"
