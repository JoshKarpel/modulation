FROM continuumio/miniconda3

# update & upgrade
RUN apt-get -y update && \
    apt-get -y upgrade

# get gcc
RUN apt-get install -y build-essential

# install python dependencies
RUN conda install -y python=3.6 \
                     numpy \
                     scipy \
                     matplotlib \
                     cython \
                     paramiko \
                     psutil \
                     tqdm \
                     cloudpickle \
    && \
    conda update -y --all && \
    conda clean -y --all

# my custom packages
RUN pip install git+https://github.com/JoshKarpel/simulacra.git

COPY .. .
RUN pip install .

# pre-compile cython code via pyximport
RUN python3 -c "import modulation"
