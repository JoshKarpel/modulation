FROM continuumio/miniconda3:latest

# update & upgrade
RUN apt-get -y update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN conda install -y python=3.6 \
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
RUN pip install git+https://github.com/JoshKarpel/htmap.git@dockery-map-management && \
    pip install git+https://github.com/JoshKarpel/simulacra.git && \
    pip install git+https://github.com/JoshKarpel/modulation.git

# pre-compile cython code via pyximport
RUN python3 -c "import modulation"
