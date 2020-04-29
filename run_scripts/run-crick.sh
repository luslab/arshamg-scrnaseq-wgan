#!/bin/sh

export TERM=xterm

## LOAD REQUIRED MODULES
ml purge
ml Nextflow/20.01.0
ml Singularity/3.4.2
ml Graphviz

## UPDATE PIPLINE
nextflow pull luslab/arshamg-scrnaseq-wgan

## RUN PIPELINE
nextflow run luslab/arshamg-scrnaseq-wgan \
  -profile crick \
  --epochs 2 \
  --writeFreq 1