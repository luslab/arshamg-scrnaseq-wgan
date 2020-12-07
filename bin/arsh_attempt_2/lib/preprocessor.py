#!/usr/bin/env python
# coding: utf-8

import os
from os import path
import pathlib
import random
import gzip
import urllib.request
import collections
from collections import Counter, namedtuple

import numpy as np
import pandas as pd
from natsort import natsorted
from biomart import BiomartServer
from gtfparse import read_gtf
import scanpy as sc
import scipy.sparse as sp_sparse
import tensorflow as tf

SEED = 0

class Preprocessor:
    data_dir = 'data'
    figure_dir = 'figures/'
    tf_dir = 'tf/'
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

    params_pre_cluster_res = 0.15
    params_pre_test_cells = 0
    params_pre_valid_cells = 500
    params_train_split_files = 2
    params_pre_scale = 20000

    sct = collections.namedtuple('sc', ('barcode', 'count_no', 'genes_no', 'dset', 'cluster'))

    def __init__(self, logger):
        self.logger = logger
        
        sc.settings.autosave=True
        sc.settings.autoshow=False
        sc.settings.figdir=path.join(self.data_dir, self.figure_dir)
        sc.settings.verbosity=3
        sc.set_figure_params(format='png')

    def create_directories(self):
        pathlib.Path(path.join(self.data_dir, self.figure_dir)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(path.join(self.data_dir, self.tf_dir)).mkdir(parents=True, exist_ok=True)
    
    def downloadFile(self, url, file_path_gz, file_path):
        if path.exists(path.join(self.data_dir, file_path)) is False: 
            urllib.request.urlretrieve(url, path.join(self.data_dir, file_path_gz))
            input = gzip.GzipFile(path.join(self.data_dir, file_path_gz), 'rb')
            s = input.read()
            input.close()
            output = open(path.join(self.data_dir, file_path), 'wb')
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

            with open(path.join(self.data_dir, self.gene_tsv_path), 'wb') as output:
                output.write(search_response.raw.data)

    def findTranscriptLengths(self):
        # Check we need to do anything
        if path.exists(path.join(self.data_dir,self.gene_length_tsv_path)) is True:
            self.logger.info('Gene length file already exists')
            return

        # Convert gtf to csv
        if path.exists(path.join(self.data_dir,self.gtf_csv_path)) is False:
            self.logger.info('Converting gtf to csv')
            df_gtf = read_gtf(path.join(self.data_dir,self.gtf_path))
            df_gtf.to_csv(path.join(self.data_dir,self.gtf_csv_path))

        # Load gtf file
        self.logger.info('Calculating transcript lengths')
        df_gtf = pd.read_csv(path.join(self.data_dir,self.gtf_csv_path), low_memory=False)
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
        df_gtf_transcript_len.to_csv(path.join(self.data_dir,self.gene_length_tsv_path))

    def preprocessRnaData(self):

        if path.exists(path.join(self.data_dir,self.tpm_combined_path)) is True:
            self.logger.info('Preprocessed file already exists')
            return

        # Get gene names
        self.logger.info('Loading gene names')
        df_gene_names = pd.read_csv(path.join(self.data_dir,self.gene_tsv_path), sep='\t', header=None)
        df_gene_names.columns = ['gene_id', "gene_name"]
        df_gene_names.index = df_gene_names.gene_id
        df_gene_names = df_gene_names.drop('gene_id', axis=1)
        self.print_df_rowcol('Loaded gene names', df_gene_names)

        # Create reverse lookup
        self.logger.info('Creating name to id lookup')
        df_gene_name2id = pd.read_csv(path.join(self.data_dir,self.gene_tsv_path), sep='\t', header=None)
        df_gene_name2id.columns = ['gene_id', "gene_name"]
        df_gene_name2id.index = df_gene_name2id.gene_name
        df_gene_name2id = df_gene_name2id.drop('gene_name', axis=1)

        # Load transcript lengths
        self.logger.info('Loading transcript lengths')
        df_gtf_transcript_len = pd.read_csv(path.join(self.data_dir,self.gene_length_tsv_path))
        df_gtf_transcript_len.index = df_gtf_transcript_len.gene_id
        df_gtf_transcript_len = df_gtf_transcript_len.drop('gene_id', axis=1)

        # *****************************************************************************
        # YANG 1
        # *****************************************************************************

        # Load in data
        self.logger.info('Loading and processing Yang1 dataset')
        df_tpm_1 = pd.read_csv(path.join(self.data_dir,self.tpm_yang_path), sep='\t')
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
        df_tpm_2 = pd.read_csv(path.join(self.data_dir,self.tpm_yang_path2), sep='\t')
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
        df_rpk = pd.read_csv(path.join(self.data_dir, self.fpkm_joost_path), sep='\t')
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
        df_tpm_4 = pd.read_csv(path.join(self.data_dir, self.tpm_ghahramani_path), sep=',')
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
        df_tpm_combined.to_csv(path.join(self.data_dir, self.tpm_combined_path))

    def annotateScData(self):
        if path.exists(path.join(self.data_dir, self.h5ad_combined_path)) is True:
            self.logger.info('Preprocessed file already exists')
            return
            
        self.logger.info('Annotating dataset')

        # Load up as anndata object
        sc_data = sc.read_csv(path.join(self.data_dir, self.tpm_combined_path))

        self.logger.info('Annotating data set')

        # Split out dataset name
        sc_data.var['dataset'] = sc_data.var_names.str.split('_').str.get(0)

        self.logger.info('Annotating gene names')

        # Load gene name lookup
        df_gene_names = pd.read_csv(path.join(self.data_dir, self.gene_tsv_path), sep='\t', header=None)
        df_gene_names.columns = ['gene_id', "gene_name"]
        df_gene_names.index = df_gene_names.gene_id
        df_gene_names = df_gene_names.drop('gene_id', axis=1)
        self.print_df_rowcol('Loaded gene names', df_gene_names)

        # Set gene names into meta
        sc_data.obs = sc_data.obs.join(df_gene_names, lsuffix='', rsuffix='', how='left')

        # Write to file
        sc_data.write(filename=path.join(self.data_dir, self.h5ad_combined_path))

    def preprocessScData(self):
        #if path.exists(self.all_preprocessed_path) is True:
        #    self.logger.info('Preprocessed file already exists')
        #    return

        self.logger.info('Preprocessing single cell data')
        sc_raw = sc.read(path.join(self.data_dir, self.h5ad_combined_path))
        sc_raw = sc_raw.transpose()

        if sp_sparse.issparse(sc_raw.X):
            self.logger.info('Data is sparse...')
            sc_raw.X = sc_raw.X.toarray()

        # TODO: Use pca to choose correct number of components

        sc.tl.pca(sc_raw, n_comps=50) # Get pca of this?
        sc.pl.pca(sc_raw, color=['dataset'], save='_pre_b4_zheng17.png')
        clustered = sc_raw.copy()
        sc.pp.recipe_zheng17(clustered) # some kind of expression normalisation and selection
        sc.tl.pca(clustered, n_comps=50) # Get pca of this?
        sc.pl.pca(clustered, color=['dataset'], save='_pre_post_zheng17.png')

        # Log the data matrix (log2(TPM+1))
        sc.pp.log1p(sc_raw, base=2)

        # Filter cells
        sc.pp.filter_cells(sc_raw, min_genes=1000)
        self.logger.info("Cells remaining: " + str(sc_raw.n_obs))

        # Filter gene
        sc.pp.filter_genes(sc_raw, min_cells=500)
        self.logger.info("Genes remaining: " + str(len(sc_raw.var.index)))

        # Clustering
        sc.pp.neighbors(clustered, n_pcs=50)
        sc.tl.louvain(clustered, resolution=self.params_pre_cluster_res)

        # Add clusters to the raw data and plot
        sc_raw.obs['cluster'] = clustered.obs['louvain']
        sc.tl.tsne(sc_raw)
        sc.pl.tsne(sc_raw, color=['cluster','dataset'], save='_pre_louvain.png')

        # Calc cluster ratios
        cells_per_cluster = Counter(sc_raw.obs['cluster'])
        cluster_ratios = dict()
        for key, value in cells_per_cluster.items():
            cluster_ratios[key] = value / sc_raw.shape[0]
        clusters_no = len(cells_per_cluster)
        self.logger.info("Clustering of the raw data is done to %d clusters." % clusters_no)
        self.logger.info(cluster_ratios)

        #print(sc_raw.X.sum(axis=0).max())

        # TODO: Total count normalise the data?
        sc.pp.normalize_per_cell(sc_raw, counts_per_cell_after=self.params_pre_scale)

        # Setup random
        random.seed(SEED)
        np.random.seed(SEED)

        # Calc how many validation cells there should be per cluster
        valid_cells_per_cluster = {
            key: int(value * self.params_pre_valid_cells)
            for key, value in cluster_ratios.items()}
        self.logger.info("Valid cells per cluster")
        self.logger.info(valid_cells_per_cluster)

        # Calc how many test cells there should be per cluster
        test_cells_per_cluster = {
            key: int(value * self.params_pre_test_cells)
            for key, value in cluster_ratios.items()}
        self.logger.info("Test cells per cluster")
        self.logger.info(test_cells_per_cluster)

        # Create a obs column for the split of the train/test valid data
        # Initialise to train and set it as a categorical column
        split_col = np.repeat('train', sc_raw.shape[0])
        unique_groups = np.asarray(['valid', 'test', 'train'])
        sc_raw.obs['split'] = pd.Categorical(values=split_col, categories=natsorted(unique_groups))

        # For each cluster
        self.logger.info("Assigning to train/valid/test based on clusters")
        for key in valid_cells_per_cluster:
            # Get all the cells that match the cluster id
            indices = sc_raw.obs[sc_raw.obs['cluster'] == str(key)].index

            # Randomly assign a number of cells to the test and valid sets
            # based on the clustering
            test_valid_indices = np.random.choice(
                indices, valid_cells_per_cluster[key] +
                test_cells_per_cluster[key], replace=False)

            test_indices = test_valid_indices[0:test_cells_per_cluster[key]]
            valid_indices = test_valid_indices[test_cells_per_cluster[key]:]

            # assign the test/training split
            for i in test_indices:
                #sc_raw.obs.set_value(i, 'split', 'test')
                sc_raw.obs.at[i, 'split'] = 'test'

            for i in valid_indices:
                #sc_raw.obs.set_value(i, 'split', 'valid')
                sc_raw.obs.at[i, 'split'] = 'valid'
    
        # TODO: move to umap
        # Show plot of training split
        self.logger.info("Data split results")
        print(sc_raw.obs['split'].value_counts())
        sc.pl.tsne(sc_raw, color=['dataset','cluster','split'], save='_pre_split.png')

        # Write to file
        self.logger.info("Write processed dataset")
        sc_raw.write(filename=path.join(self.data_dir, self.all_preprocessed_path))

    def realDataAnalysis(self):
        #sc.logging.print_header()

        # Load data
        sc_pp = sc.read(path.join(self.data_dir, self.all_preprocessed_path))

        # Top 20 highest expressed genes
        #sc.pl.highest_expr_genes(sc_pp, n_top=20, gene_symbols='gene_name', )

        #There are no MT genes in the merged dataset

        sc.tl.tsne(sc_pp)
        sc.pl.tsne(sc_pp, color=['dataset'])

    def createTfRecords(self):
        self.logger.info('Creating tensor records')

        # Create paths for files
        train_filenames = [os.path.join(self.data_dir, self.tf_dir, 'train-%s.tfrecords' % i)
                                for i in range(self.params_train_split_files)]
        valid_filename = os.path.join(self.data_dir, "tf", 'validate.tfrecords')
        test_filename = os.path.join(self.data_dir, "tf", 'test.tfrecords')

        # Load single cell data
        sc_raw = sc.read(os.path.join(self.data_dir, self.all_preprocessed_path))
        cell_count = sc_raw.shape[0]
        gene_count = sc_raw.shape[1]
        self.logger.info("Cells number is %d , with %d genes per cell." % (cell_count, gene_count))

        # Create TF Writers
        opt = tf.io.TFRecordOptions(compression_type='GZIP')
        valid = tf.io.TFRecordWriter(valid_filename, opt)
        test = tf.io.TFRecordWriter(test_filename, opt)
        train = [tf.io.TFRecordWriter(filename, opt) for filename in train_filenames]

        #cat = sc_raw.obs['cluster'].cat.categories
        count = 0
        for line in sc_raw:
            # Doesnt stop for some reason
            if count == cell_count:
                break
            count += 1

            dset = line.obs['split'][0]

            # metadata from line
            scmd = self.sct(barcode=line.obs_names[0],
                count_no=int(np.sum(line.X)),
                genes_no=line.obs['n_genes'][0],
                dset=dset,
                cluster=line.obs['cluster'][0])

            sc_genes = line.X
            d = scmd

            flat = sc_genes.flatten()
            idx = np.nonzero(flat)[0]
            vals = flat[idx]

            feat_map = {}
            feat_map['indices'] = tf.train.Feature(int64_list=tf.train.Int64List(value=idx))
            feat_map['values'] = tf.train.Feature(float_list=tf.train.FloatList(value=vals))
            feat_map['barcode'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(d.barcode)]))
            feat_map['genes_no'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[d.genes_no]))
            feat_map['count_no'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[d.count_no]))

            # add hot encoding for classification problems
            #feat_map['cluster_1hot'] = tf.train.Feature(
            #    int64_list=tf.train.Int64List(value=[int(c == cat) for c in cat]))
            #feat_map['cluster_int'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(cat)]))

            example = tf.train.Example(features=tf.train.Features(feature=feat_map))
            example_str = example.SerializeToString()

            if d.dset == 'test':
                test.write(example_str)
            elif d.dset == 'train':
                train[random.randint(0, self.params_train_split_files - 1)].write(example_str)
            elif d.dset == 'valid':
                valid.write(example_str)
            else:
                raise ValueError("invalid dataset: %s" % d.dset)

        # Close the file writers
        test.close()
        valid.close()
        for f in train:
            f.close()

        self.logger.info("Tensor write complete - wrote %d examples" % count)