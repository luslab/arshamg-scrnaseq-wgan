# scRNAseq-WGAN-GP

<img align="right" width="250" height="250" src="/images/github.png?raw=true">

Wasserstein Generative Adversarial Network for analysing scRNAseq data.

This repo contains a Jupyter Notebook with a minimal version of the WGAN-GP algorithm.

Related manuscript is on [bioRxiv](https://www.biorxiv.org/content/early/2018/02/08/262501)

## Usage

Dependencies: Tensorflow, NumPy, Pandas, scikit-learn



Clone the repo, ensure you have git lfs installed and perform git lfs pull for the CSV files
```
python ./scripts/WGAN-GP_minimal.py
```


Watch discriminator/generator loss convergence:
```
tensorboard --logdir=./summaries/
```

`docker run --rm -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes -v "$PWD":/home/jovyan/work luslab/scrnaseq-wgan`

`tensorboard --logdir logs --host 0.0.0.0`
