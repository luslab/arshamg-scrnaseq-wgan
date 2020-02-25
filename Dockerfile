ARG BASE_CONTAINER=jupyter/datascience-notebook
FROM $BASE_CONTAINER

LABEL authors="Chris Cheshire" \
      description="Docker image containing all requirements for the scRNA-Seq WGAN project"

RUN pip install tensorflow~=2.1.0 && \
    pip install biomart && \
    pip install gtfparse~=1.2.0

RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER