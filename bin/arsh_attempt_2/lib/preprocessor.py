#!/usr/bin/env python
# coding: utf-8

import os
from os import path
import gzip
import urllib.request

import numpy as np
import pandas as pd
from biomart import BiomartServer
from gtfparse import read_gtf

class Preprocessor:
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
    gtf_csv_path = 'Mus_musculus.GRCm38.99.csv'
    gene_length_tsv_path = 'gene_lengths.tsv'

    def __init__(self, logger):
        self.logger = logger
    
    def downloadFile(self, url, file_path_gz, file_path):
        if path.exists(file_path) is False: 
            urllib.request.urlretrieve(url, file_path_gz)
            input = gzip.GzipFile(file_path_gz, 'rb')
            s = input.read()
            input.close()
            output = open(file_path, 'wb')
            output.write(s)
            output.close()

    def print_df_rowcol(self, pretext, df):
        template = pretext + ': {} rows x {} cols'
        print(template.format(str(df.shape[0]), str(df.shape[1])))

    def downloadArshamData(self):
        self.logger.info('Downloading mouse Mus_musculus.GRCm38.99')
        self.downloadFile('ftp://ftp.ensembl.org/pub/release-99/gtf/mus_musculus/Mus_musculus.GRCm38.99.gtf.gz', self.gtf_path_gz, self.gtf_path)
        self.logger.info('Downloading GSE90848_Ana6_basal_hair_bulb_TPM')
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz', self.tpm_yang_path_gz, self.tpm_yang_path)
        self.logger.info('Downloading GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM')
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz', self.tpm_yang_path2_gz, self.tpm_yang_path2)
        self.logger.info('Downloading GSE67602_Joost_et_al_expression')
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/GSE67602/suppl/GSE67602_Joost_et_al_expression.txt.gz', self.tpm_joost_path_gz, self.tpm_joost_path)
        self.logger.info('Downloading GSE99989_NCA_BCatenin_TPM_matrix')
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE99nnn/GSE99989/suppl/GSE99989_NCA_BCatenin_TPM_matrix.csv.gz', self.tpm_ghahramani_path_gz, self.tpm_ghahramani_path)

        self.logger.info('Downloading Gene id->name conversion table from biomart')

        # Get refseq gene ids
        if path.exists(self.gene_tsv_path) is False: 
            server = BiomartServer( "http://www.ensembl.org:80/biomart/martservice" )
            mouse_dataset = server.datasets['mmusculus_gene_ensembl']
            search_response = mouse_dataset.search({
                'attributes': [ 'ensembl_gene_id', 'external_gene_name' ]
            })

            with open(self.gene_tsv_path, 'wb') as output:
                output.write(search_response.raw.data)

    def findTranscriptLengths(self):
        # Check we need to do anything
        if path.exists(self.gene_length_tsv_path) is True:
            self.logger.info('Gene length file already exists')
            return

        # Convert gtf to csv
        if path.exists(self.gtf_csv_path) is False:
            self.logger.info('Converting gtf to csv')
            df_gtf = read_gtf(self.gtf_path)
            df_gtf.to_csv(self.gtf_csv_path)

        # Load gtf file
        self.logger.info('Calculating transcript lengths')
        df_gtf = pd.read_csv(self.gtf_csv_path)
        self.print_df_rowcol('Loaded GTF', df_gtf)

        # Calculate feature lengths
        df_gtf.insert(6,"feature_len", df_gtf.end - df_gtf.start)

        self.logger.info('Filtering for transcripts and group by gene id')

        # Filter GTF data for exons, 3' and 5' UTRs
        df_gtf = df_gtf[(df_gtf.feature=='exon') | (df_gtf.feature=='three_prime_utr') | (df_gtf.feature=='five_prime_utr')]
        self.print_df_rowcol('Filtered GTF for exons and UTR', df_gtf)

        # Aggregate the feature lengths to find a transcript length for each gene
        df_gtf_transcript_len = pd.DataFrame(df_gtf.groupby(['gene_id']).sum()['feature_len'])

        # Write to file
        df_gtf_transcript_len.to_csv(self.gene_length_tsv_path)