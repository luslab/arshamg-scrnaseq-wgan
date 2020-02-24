FROM luslab/bioinf-base:latest
LABEL authors="Chris Cheshire" \
      description="Docker image containing all requirements for the scRNA-Seq WGAN project"

# Install the conda environment
COPY environment.yml /
RUN conda env create -f /environment.yml && conda clean -a

# Add conda installation dir to PATH (instead of doing 'conda activate')
ENV PATH /opt/conda/envs/scrnaseq-wgan/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name scrnaseq-wgan > scrnaseq-wgan.yml