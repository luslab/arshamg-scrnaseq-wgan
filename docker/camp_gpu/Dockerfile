ARG BASE_CONTAINER=tensorflow/tensorflow:latest-gpu-py3
FROM $BASE_CONTAINER

LABEL authors="Chris Cheshire" \
      description="Docker image for running tensorflow on GPU CAMP"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

# Reload and update conda
# Conda update base fixes the "module named conda not found" issue
RUN source ~/.bashrc
RUN conda update -n base conda
RUN conda update --all

# Install scanpy
RUN conda install seaborn scikit-learn statsmodels numba pytables
RUN conda install -c conda-forge python-igraph leidenalg

# Switch to bioinformatics channels
#RUN conda config --add channels defaults && \
#    conda config --add channels bioconda && \
#    conda config --add channels conda-forge;

# Install pip packages
RUN pip install --upgrade pip
RUN pip install scanpy
RUN pip install tensorflow

# Install open cv
RUN apt-get update --fix-missing && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install opencv-python