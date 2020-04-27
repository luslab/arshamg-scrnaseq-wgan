#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libs
import os.path
from os import path
import gzip
import urllib.request
from biomart import BiomartServer


# In[2]:


def download_file(url, file_path_gz, file_path):
    if path.exists(file_path) is False: 
        urllib.request.urlretrieve(url, file_path_gz)
        input = gzip.GzipFile(file_path_gz, 'rb')
        s = input.read()
        input.close()
        output = open(file_path, 'wb')
        output.write(s)
        output.close()


# In[3]:


# Setup
data_path = '../data'
gtf_path_gz = data_path + '/Mus_musculus.GRCm38.99.gtf.gz'
gtf_path = data_path + '/Mus_musculus.GRCm38.99.gtf'
tpm_yang_path_gz = data_path + '/GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz'
tpm_yang_path = data_path + '/GSE90848_Ana6_basal_hair_bulb_TPM.txt'
tpm_yang_path2_gz = data_path + '/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz'
tpm_yang_path2 = data_path + '/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt'
tpm_joost_path_gz = data_path + '/GSE67602_Joost_et_al_expression.txt.gz'
tpm_joost_path = data_path + '/GSE67602_Joost_et_al_expression.txt'
tpm_ghahramani_path_gz = data_path + '/GSE99989_NCA_BCatenin_TPM_matrix.csv.gz'
tpm_ghahramani_path = data_path + '/GSE99989_NCA_BCatenin_TPM_matrix.csv'
gene_tsv_path = data_path + '/gene_names.tsv'


# In[4]:


# Download initial datasets and decompress
download_file('ftp://ftp.ensembl.org/pub/release-99/gtf/mus_musculus/Mus_musculus.GRCm38.99.gtf.gz', gtf_path_gz, gtf_path)
download_file('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz', tpm_yang_path_gz, tpm_yang_path)
download_file('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz', tpm_yang_path2_gz, tpm_yang_path2)
download_file('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/GSE67602/suppl/GSE67602_Joost_et_al_expression.txt.gz', tpm_joost_path_gz, tpm_joost_path)
download_file('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE99nnn/GSE99989/suppl/GSE99989_NCA_BCatenin_TPM_matrix.csv.gz', tpm_ghahramani_path_gz, tpm_ghahramani_path)


# In[12]:


# Get refseq gene ids
if path.exists(gene_tsv_path) is False: 
    server = BiomartServer( "http://www.ensembl.org:80/biomart/martservice" )
    mouse_dataset = server.datasets['mmusculus_gene_ensembl']
    search_response = mouse_dataset.search({
      'attributes': [ 'ensembl_gene_id', 'external_gene_name' ]
    })

    with open(gene_tsv_path, 'wb') as output:
        output.write(search_response.raw.data)

