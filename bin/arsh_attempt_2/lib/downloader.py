#!/usr/bin/env python
# coding: utf-8

import os.path
from os import path
import gzip
import urllib.request

class Downloader:
    gtf_path_gz = 'Mus_musculus.GRCm38.99.gtf.gz'
    gtf_path =  'Mus_musculus.GRCm38.99.gtf'
    tpm_yang_path_gz = 'GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz'
    tpm_yang_path = 'GSE90848_Ana6_basal_hair_bulb_TPM.txt'
    tpm_yang_path2_gz = 'GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz'
    tpm_yang_path2 = 'GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt'
    tpm_joost_path_gz = 'GSE67602_Joost_et_al_expression.txt.gz'
    tpm_joost_path = 'GSE67602_Joost_et_al_expression.txt'
    tpm_ghahramani_path_gz = 'GSE99989_NCA_BCatenin_TPM_matrix.csv.gz'
    tpm_ghahramani_path = 'GSE99989_NCA_BCatenin_TPM_matrix.csv'
    gene_tsv_path = 'gene_names.tsv'
    
    def downloadFile(self, url, file_path_gz, file_path):
        if path.exists(file_path) is False: 
            urllib.request.urlretrieve(url, file_path_gz)
            input = gzip.GzipFile(file_path_gz, 'rb')
            s = input.read()
            input.close()
            output = open(file_path, 'wb')
            output.write(s)
            output.close()

    def downloadArshamData(self):
        self.downloadFile('ftp://ftp.ensembl.org/pub/release-99/gtf/mus_musculus/Mus_musculus.GRCm38.99.gtf.gz', self.gtf_path_gz, self.gtf_path)
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz', self.tpm_yang_path_gz, self.tpm_yang_path)
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz', self.tpm_yang_path2_gz, self.tpm_yang_path2)
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/GSE67602/suppl/GSE67602_Joost_et_al_expression.txt.gz', self.tpm_joost_path_gz, self.tpm_joost_path)
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE99nnn/GSE99989/suppl/GSE99989_NCA_BCatenin_TPM_matrix.csv.gz', self.tpm_ghahramani_path_gz, self.tpm_ghahramani_path)
