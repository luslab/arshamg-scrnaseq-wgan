ARG BASE_CONTAINER=jupyter/datascience-notebook
FROM $BASE_CONTAINER

LABEL authors="Chris Cheshire" \
      description="Docker image containing all requirements for the scRNA-Seq WGAN project"

# Configure bioconda
RUN conda config --add channels defaults && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge;

# R packages
RUN conda install --quiet --yes \
    'r-biocmanager=1.30.*' \
    'bioconductor-multtest=2.42.*' \
    'r-seurat=3.1.*' \
    && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN pip install tensorflow~=2.1.0 && \
    pip install biomart && \
    pip install gtfparse~=1.2.0 && \
    pip install jupyter_contrib_nbextensions

RUN jupyter contrib nbextension install --user

# https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/spellchecker/README.html
RUN jupyter nbextension enable spell-checker

RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
