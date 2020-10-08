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
import scanpy as sc
import scipy.sparse as sp_sparse

class Preprocessor:
    figure_dir = './figures/'
    gtf_path_gz = 'Mus_musculus.GRCm38.99.gtf.gz'
    gtf_path =  'Mus_musculus.GRCm38.99.gtf'
    tpm_yang_path_gz = 'GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz'
    tpm_yang_path = 'GSE90848_Ana6_basal_hair_bulb_TPM.txt'
    tpm_yang_path2_gz = 'GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz'
    tpm_yang_path2 = 'GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt'
    fpkm_joost_path_gz = 'GSE67602_Joost_et_al_expression.txt.gz'
    fpkm_joost_path = 'GSE67602_Joost_et_al_expression.txt'
    tpm_ghahramani_path_gz = 'GSE99989_NCA_BCatenin_TPM_matrix.csv.gz'
    tpm_ghahramani_path = 'GSE99989_NCA_BCatenin_TPM_matrix.csv'
    gene_tsv_path = 'gene_names.tsv'
    gtf_csv_path = 'Mus_musculus.GRCm38.99.csv'
    gene_length_tsv_path = 'gene_lengths.tsv'
    tpm_combined_path = 'tpm_combined.csv'
    h5ad_combined_path = 'tpm_combined.h5ad'
    all_preprocessed_path = 'all_preprocessed.h5ad'

    def __init__(self, logger):
        self.logger = logger
        
        sc.settings.autosave=True
        sc.settings.autoshow=False
        sc.settings.figdir=self.figure_dir
        sc.settings.verbosity=3
        sc.set_figure_params(format='png')
    
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
        self.logger.info(template.format(str(df.shape[0]), str(df.shape[1])))

    def downloadArshamData(self):
        self.logger.info('Downloading mouse Mus_musculus.GRCm38.99')
        self.downloadFile('ftp://ftp.ensembl.org/pub/release-99/gtf/mus_musculus/Mus_musculus.GRCm38.99.gtf.gz', self.gtf_path_gz, self.gtf_path)
        self.logger.info('Downloading GSE90848_Ana6_basal_hair_bulb_TPM')
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz', self.tpm_yang_path_gz, self.tpm_yang_path)
        self.logger.info('Downloading GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM')
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90848/suppl/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz', self.tpm_yang_path2_gz, self.tpm_yang_path2)
        self.logger.info('Downloading GSE67602_Joost_et_al_expression')
        self.downloadFile('ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/GSE67602/suppl/GSE67602_Joost_et_al_expression.txt.gz', self.fpkm_joost_path_gz, self.fpkm_joost_path)
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
        df_gtf = pd.read_csv(self.gtf_csv_path, low_memory=False)
        self.print_df_rowcol('Loaded GTF', df_gtf)

        # Calculate feature lengths
        df_gtf.insert(6,"feature_len", df_gtf.end - df_gtf.start)

        self.logger.info('Filtering for transcripts and group by gene id')

        # Filter GTF data for exons, 3' and 5' UTRs
        df_gtf = df_gtf[(df_gtf.feature=='exon') | (df_gtf.feature=='three_prime_utr') | (df_gtf.feature=='five_prime_utr')]
        self.print_df_rowcol('Filtered GTF for exons and UTR', df_gtf)

        # Aggregate the feature lengths to find a transcript length for each gene
        df_gtf_transcript_len = pd.DataFrame(df_gtf.groupby(['gene_id']).sum()['feature_len'])

        # Convert to kilobase
        df_gtf_transcript_len = df_gtf_transcript_len / 1000

        # Write to file
        df_gtf_transcript_len.to_csv(self.gene_length_tsv_path)

    def preprocessRnaData(self):

        if path.exists(self.tpm_combined_path) is True:
            self.logger.info('Preprocessed file already exists')
            return

        # Get gene names
        self.logger.info('Loading gene names')
        df_gene_names = pd.read_csv(self.gene_tsv_path, sep='\t', header=None)
        df_gene_names.columns = ['gene_id', "gene_name"]
        df_gene_names.index = df_gene_names.gene_id
        df_gene_names = df_gene_names.drop('gene_id', axis=1)
        self.print_df_rowcol('Loaded gene names', df_gene_names)

        # Create reverse lookup
        self.logger.info('Creating name to id lookup')
        df_gene_name2id = pd.read_csv(self.gene_tsv_path, sep='\t', header=None)
        df_gene_name2id.columns = ['gene_id', "gene_name"]
        df_gene_name2id.index = df_gene_name2id.gene_name
        df_gene_name2id = df_gene_name2id.drop('gene_name', axis=1)

        # Load transcript lengths
        self.logger.info('Loading transcript lengths')
        df_gtf_transcript_len = pd.read_csv(self.gene_length_tsv_path)
        df_gtf_transcript_len.index = df_gtf_transcript_len.gene_id
        df_gtf_transcript_len = df_gtf_transcript_len.drop('gene_id', axis=1)

        # *****************************************************************************
        # YANG 1
        # *****************************************************************************

        # Load in data
        self.logger.info('Loading and processing Yang1 dataset')
        df_tpm_1 = pd.read_csv(self.tpm_yang_path, sep='\t')
        self.print_df_rowcol('Loaded Yang1', df_tpm_1)

        # Clean the yang1 dataset to get the gene names, call `.shape` to check we havent filtered anything
        # Split gene ids on _ and load into a new data frame and set the columns
        split_data = pd.DataFrame(df_tpm_1.Gene_id.str.split("_", expand=True))
        split_data.columns = ["Gene_id", "Gene_name", "Gene_name2"]

        # Fill in the NA's with blank strings
        split_data["Gene_name"] = split_data.Gene_name.fillna('')
        split_data["Gene_name2"] = split_data.Gene_name2.fillna('')

        # Concatenate the strings that have split more than once back to their standard for e.g GENEID_GENENAME_SOMEMORENAME
        split_data["Gene_name"] = split_data.apply(lambda x: x.Gene_name if x.Gene_name2 == '' else x.Gene_name + '_' + x.Gene_name2, axis=1)
        self.print_df_rowcol('Filter check (check the first number the same as the number above)', split_data)

        # Write the gene names back into the main dataset, print out dataset to check we do indeed have gene names where they were available
        # Insert the columns back into the main data array
        df_tpm_1["Gene_id"] = split_data.Gene_id
        df_tpm_1.insert(1,"Gene_name", split_data.Gene_name)

        # Set gene name to index after checking it is unique
        df_tpm_1.index = df_tpm_1['Gene_id']
        df_tpm_1 = df_tpm_1.drop('Gene_id', axis=1)
        df_tpm_1 = df_tpm_1.drop('Gene_name', axis=1)

        # Prefix for dataset
        df_tpm_1 = df_tpm_1.add_prefix('yang1_')

        # *****************************************************************************
        # YANG 2
        # *****************************************************************************

        # Load in Yang2
        self.logger.info('Loading and processing Yang2 dataset')
        df_tpm_2 = pd.read_csv(self.tpm_yang_path2, sep='\t')
        self.print_df_rowcol('Loaded Yang2', df_tpm_2)

        # Split gene ids on _ and load into a new data frame and set the columns
        split_data = pd.DataFrame(df_tpm_2.Gene_id.str.split("_", expand=True))
        split_data.columns = ["Gene_id", "Gene_name", "Gene_name2"]

        # Fill in the NA's with blank strings
        split_data["Gene_name"] = split_data.Gene_name.fillna('')
        split_data["Gene_name2"] = split_data.Gene_name2.fillna('')

        # Concatenate the strings that have split more than once back to their standard for e.g GENEID_GENENAME_SOMEMORENAME
        split_data["Gene_name"] = split_data.apply(lambda x: x.Gene_name if x.Gene_name2 == '' else x.Gene_name + '_' + x.Gene_name2, axis=1)
        self.print_df_rowcol('Filter check (check the first number the same as the number above)', split_data)

        # Insert the columns back into the main data array
        df_tpm_2["Gene_id"] = split_data.Gene_id
        df_tpm_2.insert(1,"Gene_name", split_data.Gene_name)

        # Set gene name to index after checking it is unique
        df_tpm_2.index = df_tpm_2['Gene_id']
        df_tpm_2 = df_tpm_2.drop('Gene_id', axis=1)
        df_tpm_2 = df_tpm_2.drop('Gene_name', axis=1)

        # Prefix for dataset
        df_tpm_2 = df_tpm_2.add_prefix('yang2_')

        # *****************************************************************************
        # Joost / Kasper
        # *****************************************************************************

        # Load data
        self.logger.info('Loading and processing Joost dataset')
        df_rpk = pd.read_csv(self.fpkm_joost_path, sep='\t')
        self.print_df_rowcol('Loaded Kasper', df_rpk)

        # Convert from FPKM (Fragments Per Kilobase Million) to TPM (Transcripts Per Kilobase Million)

        # Rename gene/cell column
        df_rpk.rename( columns={'Gene\Cell':'gene_name'}, inplace=True)
        df_rpk.index = df_rpk['gene_name']

        # Insert gene_id into data
        df_rpk_merged = df_rpk
        df_rpk_merged = df_rpk_merged.drop('gene_name', axis=1)
        df_rpk_merged = df_gene_name2id.join(df_rpk_merged, lsuffix='', rsuffix='', how='inner')
        df_rpk_merged.index = df_rpk_merged.gene_id
        df_rpk_merged = df_rpk_merged.drop('gene_id', axis=1)
        self.print_df_rowcol('Shape after merging gene ids', df_rpk_merged)


        # Get length of each gene into data frame
        df_rpk_merged_len = df_gtf_transcript_len.join(df_rpk_merged, lsuffix='', rsuffix='', how='inner')
        self.print_df_rowcol('Shape after merging gene feature_len', df_rpk_merged_len)

        # Sum the read counts per sample
        df_scaling_factor = pd.DataFrame(df_rpk_merged_len.sum(axis=0) / 1000000)
        df_scaling_factor.columns = ['scaling_factor']
        df_scaling_factor = df_scaling_factor.drop('feature_len')

        # Divide the read counts by the length of each gene in kilobases. This gives you reads per kilobase (RPK)
        df_rpk_merged_len = df_rpk_merged_len.iloc[:,1:].div(df_rpk_merged_len.feature_len, axis=0)

        # Divide the RPK values by the “per million” scaling factor. This gives you TPM.
        df_tpm_3 = df_rpk_merged_len.div(df_scaling_factor.scaling_factor, axis=1)

        # Prefix for dataset
        df_tpm_3 = df_tpm_3.add_prefix('jhoos_')

        # *****************************************************************************
        # Ghahramani
        # *****************************************************************************

        # Load data
        self.logger.info('Loading and processing Ghahramani dataset')
        df_tpm_4 = pd.read_csv(self.tpm_ghahramani_path, sep=',')
        self.print_df_rowcol('Loaded Ghahramani', df_tpm_4)

        # Rename gene name column and set to index
        df_tpm_4.rename( columns={'Unnamed: 0':'gene_name'}, inplace=True)
        df_tpm_4.index = df_tpm_4['gene_name']

        # Insert gene into data
        df_tpm_4 = df_tpm_4.drop('gene_name', axis=1)
        df_tpm_4 = df_gene_name2id.join(df_tpm_4, lsuffix='', rsuffix='', how='inner')
        df_tpm_4.index = df_tpm_4.gene_id
        df_tpm_4 = df_tpm_4.drop('gene_id', axis=1)
        self.print_df_rowcol('Shape after merging gene ids', df_tpm_4)

        # Prefix for dataset
        df_tpm_4 = df_tpm_4.add_prefix('ghahr_')

        # *****************************************************************************
        # Merge
        # *****************************************************************************

        self.logger.info('Merging datasets')

        # Create merged dataset from all subsets
        df_tpm_combined = df_tpm_1.join(df_tpm_2, lsuffix='', rsuffix='_other', how='inner')
        df_tpm_combined = df_tpm_combined.join(df_tpm_3, lsuffix='', rsuffix='_other', how='inner')
        df_tpm_combined = df_tpm_combined.join(df_tpm_4, lsuffix='', rsuffix='_other', how='inner')

        self.print_df_rowcol('Shape after merging', df_tpm_combined)

        # Convert to real gene names
        #df_tpm_combined = df_gene_names.join(df_tpm_combined, lsuffix='', rsuffix='', how='inner')
        #df_tpm_combined.index = df_tpm_combined.gene_name
        #df_tpm_combined = df_tpm_combined.drop('gene_name', axis=1)
        #self.print_df_rowcol('Shape after switching to gene names', df_tpm_combined)

        # Write to file
        df_tpm_combined.to_csv(self.tpm_combined_path)

    def annotateScData(self):
        if path.exists(self.h5ad_combined_path) is True:
            self.logger.info('Preprocessed file already exists')
            return
            
        self.logger.info('Annotating dataset')

        # Load up as anndata object
        sc_data = sc.read_csv(self.tpm_combined_path)

        self.logger.info('Annotating data set')

        # Split out dataset name
        sc_data.var['dataset'] = sc_data.var_names.str.split('_').str.get(0)

        self.logger.info('Annotating gene names')

        # Load gene name lookup
        df_gene_names = pd.read_csv(self.gene_tsv_path, sep='\t', header=None)
        df_gene_names.columns = ['gene_id', "gene_name"]
        df_gene_names.index = df_gene_names.gene_id
        df_gene_names = df_gene_names.drop('gene_id', axis=1)
        self.print_df_rowcol('Loaded gene names', df_gene_names)

        # Set gene names into meta
        sc_data.obs = sc_data.obs.join(df_gene_names, lsuffix='', rsuffix='', how='left')

        # Write to file
        sc_data.write(filename=self.h5ad_combined_path)

    def preprocessScData(self):
        if path.exists(self.all_preprocessed_path) is True:
            self.logger.info('Preprocessed file already exists')
            return

        self.logger.info('Preprocessing single cell data')
        sc_raw = sc.read(self.h5ad_combined_path)
        sc_raw = sc_raw.transpose()

        if sp_sparse.issparse(sc_raw.X):
            self.logger.info('Data is sparse...')
            sc_raw.X = sc_raw.X.toarray()

        # Log the data matrix (log2(TPM+1))
        sc.pp.log1p(sc_raw, base=2)

        # Filter cells
        sc.pp.filter_cells(sc_raw, min_genes=1000)
        print("Cells remaining: " + str(sc_raw.n_obs))

        # Filter gene
        sc.pp.filter_genes(sc_raw, min_cells=500)
        print("Genes remaining: " + str(len(sc_raw.var.index)))

        # Convert to log
        # Total count normalise the data

        # Write to file
        sc_raw.write(filename=self.all_preprocessed_path)

    def realDataAnalysis(self):
        #sc.logging.print_header()

        # Load data
        sc_pp = sc.read(self.all_preprocessed_path)

        # Top 20 highest expressed genes
        #sc.pl.highest_expr_genes(sc_pp, n_top=20, gene_symbols='gene_name', )

        #There are no MT genes in the merged dataset

        sc.tl.tsne(sc_pp)
        sc.pl.tsne(sc_pp, color=['dataset'])