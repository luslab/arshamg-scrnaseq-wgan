#!/usr/bin/env python
# coding: utf-8

# Import libs
import sys
import os.path
from os import path
import gzip
import pandas as pd
import numpy as np
import urllib.request
from gtfparse import read_gtf
from biomart import BiomartServer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def print_rowcol(pretext, df):
    template = pretext + ': {} rows x {} cols'
    print(template.format(str(df.shape[0]), str(df.shape[1])))

#gtf_path = data_path + '/Mus_musculus.GRCm38.99.gtf'
#tpm_yang_path2 = data_path + '/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt'
#tpm_yang_path = data_path + '/GSE90848_Ana6_basal_hair_bulb_TPM.txt'
#tpm_ghahramani_path = data_path + '/GSE99989_NCA_BCatenin_TPM_matrix.csv'
#tpm_joost_path = data_path + '/GSE67602_Joost_et_al_expression.txt'
#gene_tsv_path = data_path + '/gene_names.tsv'
#gtf_data_path = data_path + '/gtf.txt'*/

gtf_path = sys.argv[1]
tpm_yang_path2 = sys.argv[2]
tpm_yang_path = sys.argv[3]
tpm_ghahramani_path = sys.argv[4]
tpm_joost_path = sys.argv[5]
gene_tsv_path = sys.argv[6]
gtf_data_path = sys.argv[7]

# Set the main thresholds for the pre-processing
# - min_num_genes_in_cell_exp - is the min number of genes that need to have expression > **min_ltpm_exp** for it to be considered a valid cell
# - min_num_cells_for_gene_exp - is the min number of cells that need to have an expression for that gene > **min_ltpm_exp** for it to be considered a valid gene

# Thresholds
min_num_genes_in_cell_exp = 1000
min_num_cells_for_gene_exp = 500
min_ltpm_exp = 1

# Train/Test
test_data_size = 500

# ## Load and process gene data
# 
# Load GTF data and show some data
if path.exists(gtf_data_path) is False:
    df_gtf = read_gtf(gtf_path)
    df_gtf.to_csv(gtf_data_path)

df_gtf = pd.read_csv(gtf_data_path)
print_rowcol('Loaded GTF', df_gtf)

# Calculate feature lengths
df_gtf.insert(6,"feature_len", df_gtf.end - df_gtf.start)

# Filter GTF data for exons, 3' and 5' UTRs
df_gtf = df_gtf[(df_gtf.feature=='exon') | (df_gtf.feature=='three_prime_utr') | (df_gtf.feature=='five_prime_utr')]
print_rowcol('Filtered GTF for exons and UTR', df_gtf)

# Aggregate the feature lengths to find a transcript length for each gene
df_gtf_transcript_len = pd.DataFrame(df_gtf.groupby(['gene_id']).sum()['feature_len'])

del df_gtf

df_gtf_transcript_len = df_gtf_transcript_len / 1000

df_gene_names = pd.read_csv(gene_tsv_path, sep='\t', header=None)
df_gene_names.columns = ['gene_id', "gene_name"]
df_gene_names.index = df_gene_names.gene_id
df_gene_names = df_gene_names.drop('gene_id', axis=1)
print_rowcol('Loaded gene names', df_gene_names)

df_gene_name2id = pd.read_csv(gene_tsv_path, sep='\t', header=None)
df_gene_name2id.columns = ['gene_id', "gene_name"]
df_gene_name2id.index = df_gene_name2id.gene_name
df_gene_name2id = df_gene_name2id.drop('gene_name', axis=1)
print_rowcol('Loaded gene names', df_gene_name2id)

# ## Load in RNA-Seq data
# 
# ### Yang
# 
# Read the yang1 data set and look at the data

df_tpm_1 = pd.read_csv(tpm_yang_path, sep='\t')
print_rowcol('Loaded Yang1', df_tpm_1)

# Clean the yang1 dataset to get the gene names, call `.shape` to check we havent filtered anything
# Split gene ids on _ and load into a new data frame and set the columns
split_data = pd.DataFrame(df_tpm_1.Gene_id.str.split("_", expand=True))
split_data.columns = ["Gene_id", "Gene_name", "Gene_name2"]

# Fill in the NA's with blank strings
split_data["Gene_name"] = split_data.Gene_name.fillna('')
split_data["Gene_name2"] = split_data.Gene_name2.fillna('')

# Concatinate the strings that have split more than once back to their standard for e.g GENEID_GENENAME_SOMEMORENAME
split_data["Gene_name"] = split_data.apply(lambda x: x.Gene_name if x.Gene_name2 == '' else x.Gene_name + '_' + x.Gene_name2, axis=1)
print_rowcol('Filter check', split_data)


# Write the gene names back into the main dataset, print out dataset to check we do indeed have gene names where they were available
# Insert the columns back into the main data array
df_tpm_1["Gene_id"] = split_data.Gene_id
df_tpm_1.insert(1,"Gene_name", split_data.Gene_name)


# Set gene name to index after checking it is unique
df_tpm_1.index = df_tpm_1['Gene_id']
df_tpm_1 = df_tpm_1.drop('Gene_id', axis=1)
df_tpm_1 = df_tpm_1.drop('Gene_name', axis=1)

# Load in Yang2 
df_tpm_2 = pd.read_csv(tpm_yang_path2, sep='\t')
print_rowcol('Loaded Yang2', df_tpm_2)

# Split gene ids on _ and load into a new data frame and set the columns
split_data = pd.DataFrame(df_tpm_2.Gene_id.str.split("_", expand=True))
split_data.columns = ["Gene_id", "Gene_name", "Gene_name2"]

# Fill in the NA's with blank strings
split_data["Gene_name"] = split_data.Gene_name.fillna('')
split_data["Gene_name2"] = split_data.Gene_name2.fillna('')

# Concatinate the strings that have split more than once back to their standard for e.g GENEID_GENENAME_SOMEMORENAME
split_data["Gene_name"] = split_data.apply(lambda x: x.Gene_name if x.Gene_name2 == '' else x.Gene_name + '_' + x.Gene_name2, axis=1)
print_rowcol('Filter check', split_data)


# Insert the columns back into the main data array
df_tpm_2["Gene_id"] = split_data.Gene_id
df_tpm_2.insert(1,"Gene_name", split_data.Gene_name)


# Set gene name to index after checking it is unique
df_tpm_2.index = df_tpm_2['Gene_id']
df_tpm_2 = df_tpm_2.drop('Gene_id', axis=1)
df_tpm_2 = df_tpm_2.drop('Gene_name', axis=1)

# ## Joost / Kasper
# 
# Load in data

df_rpk = pd.read_csv(tpm_joost_path, sep='\t')
print_rowcol('Loaded Kasper', df_rpk)

# ### Convert from FPKM (Fragments Per Kilobase Million) to TPM (Transcripts Per Kilobase Million)
# 
# Rename gene/cell column
df_rpk.rename( columns={'Gene\Cell':'gene_name'}, inplace=True)


# Set gene to index after checking unique
df_rpk.index = df_rpk['gene_name']

# Insert gene_id into data
df_rpk_merged = df_rpk
df_rpk_merged = df_rpk_merged.drop('gene_name', axis=1)
df_rpk_merged = df_gene_name2id.join(df_rpk_merged, lsuffix='', rsuffix='', how='inner')
df_rpk_merged.index = df_rpk_merged.gene_id
df_rpk_merged = df_rpk_merged.drop('gene_id', axis=1)
print_rowcol('Shape after merging gene ids', df_rpk_merged)


# Get length of each gene into data frame
df_rpk_merged_len = df_gtf_transcript_len.join(df_rpk_merged, lsuffix='', rsuffix='', how='inner')
print_rowcol('Shape after merging gene feature_len', df_rpk_merged_len)

# Sum the read counts per sample
df_scaling_factor = pd.DataFrame(df_rpk_merged_len.sum(axis=0) / 1000000)
df_scaling_factor.columns = ['scaling_factor']
df_scaling_factor = df_scaling_factor.drop('feature_len')


# Divide the read counts by the length of each gene in kilobases. This gives you reads per kilobase (RPK)
df_rpk_merged_len = df_rpk_merged_len.iloc[:,1:].div(df_rpk_merged_len.feature_len, axis=0)

# Divide the RPK values by the “per million” scaling factor. This gives you TPM.
df_tpm_3 = df_rpk_merged_len.div(df_scaling_factor.scaling_factor, axis=1)


# ## Ghahramani

# Load in dataset
df_tpm_4 = pd.read_csv(tpm_ghahramani_path, sep=',')
print_rowcol('Loaded Ghahramani', df_tpm_4)


# Rename gene name column
df_tpm_4.rename( columns={'Unnamed: 0':'gene_name'}, inplace=True)

# Set gene to index after checking unique
df_tpm_4.index = df_tpm_4['gene_name']

# Insert gene into data
df_tpm_4 = df_tpm_4.drop('gene_name', axis=1)
df_tpm_4 = df_gene_name2id.join(df_tpm_4, lsuffix='', rsuffix='', how='inner')
df_tpm_4.index = df_tpm_4.gene_id
df_tpm_4 = df_tpm_4.drop('gene_id', axis=1)
print_rowcol('Shape after merging gene ids', df_tpm_4)


# ## Create merged dataset
# 
# Create merged dataset from all subsets
df_tpm_combined = df_tpm_1.join(df_tpm_2, lsuffix='', rsuffix='_other', how='inner')
df_tpm_combined = df_tpm_combined.join(df_tpm_3, lsuffix='', rsuffix='_other', how='inner')
df_tpm_combined = df_tpm_combined.join(df_tpm_4, lsuffix='', rsuffix='_other', how='inner')

# Create `log2(TPM+1)` dataset
df_ltpm_combined = np.log2(df_tpm_combined + 1)

# Convert to real gene names
df_ltpm_combined_named = df_ltpm_combined
df_ltpm_combined_named = df_gene_names.join(df_ltpm_combined, lsuffix='', rsuffix='', how='inner')
df_ltpm_combined_named.index = df_ltpm_combined_named.gene_name
df_ltpm_combined_named = df_ltpm_combined_named.drop('gene_name', axis=1)
print_rowcol('Shape after switching to gene names', df_ltpm_combined_named)


# Filter for genes and cells which match filtering criteria. Call `.shape` to check what was filtered
df_expression_filt_mask = df_ltpm_combined_named > min_ltpm_exp
df_ltpm_combined_genefilt = df_ltpm_combined_named[df_expression_filt_mask.sum(axis=1) > min_num_cells_for_gene_exp]

df_ltpm_combined_genecellfilt = df_ltpm_combined_genefilt     .T[(df_ltpm_combined_genefilt > min_ltpm_exp).sum(axis=0) > min_num_genes_in_cell_exp]
df_ltpm_combined_genecellfilt = df_ltpm_combined_genecellfilt.T

# ## Normalise data
# 
# Check max values
#data_max = df_ltpm_combined_genecellfilt.max()
#data_max = data_max.max()
#print(data_max)


# Normalise data
#np_data = df_ltpm_combined_genecellfilt.T.values
#scaler = MinMaxScaler()
#print(scaler.fit(np_data))

# Check which dimension we are fitting to - if we are fitting to gene expression then should be equal to number of genes
#print(scaler.data_max_.shape)

#np_data_norm = np.transpose(scaler.transform(np_data))

#df_ltpm_combined_norm = pd.DataFrame(np_data_norm)
#df_ltpm_combined_norm.columns = df_ltpm_combined_genecellfilt.columns
#df_ltpm_combined_norm.index = df_ltpm_combined_genecellfilt.index


# Check new max
#data_max = df_ltpm_combined_norm.max()
#data_max = data_max.max()
#print(data_max)

# ## Split train and test data sets
# 
# Randomly select test and training data
train_features, test_features = train_test_split(df_ltpm_combined_genecellfilt.T, test_size=test_data_size)
train_features = train_features.T
test_features = test_features.T

print_rowcol('Created training dataset', train_features)
print_rowcol('Created test dataset', test_features)

# Create unnormalised split dataset from the same choices
#df_ltpm_combined_train = df_ltpm_combined_genecellfilt.T[df_ltpm_combined_genecellfilt.T.index.isin(list(train_features.T.index.values))]
#df_ltpm_combined_train = df_ltpm_combined_train.T
#print_rowcol('Created training unnorm dataset', df_ltpm_combined_train)

#df_ltpm_combined_test = df_ltpm_combined_genecellfilt.T[df_ltpm_combined_genecellfilt.T.index.isin(list(test_features.T.index.values))]
#df_ltpm_combined_test = df_ltpm_combined_test.T
#print_rowcol('Created test unnorm dataset', df_ltpm_combined_test)


# ## Write data to file

# Get the column and row names as a list
train_df_column_names = pd.DataFrame(list(train_features.columns.values))
train_df_row_names = pd.DataFrame(list(train_features.index.values))
test_df_column_names = pd.DataFrame(list(test_features.columns.values))
test_df_row_names = pd.DataFrame(list(test_features.index.values))

print(train_df_column_names.shape)
print(train_df_row_names.shape)
print(test_df_column_names.shape)
print(test_df_row_names.shape)

print(train_features.max().max())
print(test_features.max().max())

# Write the data to file
train_features.to_csv('tpm_combined.csv', index=False, header=False)
train_df_column_names.to_csv('tpm_combined_cols.csv', index=False, header=False)
train_df_row_names.to_csv('tpm_combined_rows.csv', index=False, header=False)

test_features.to_csv('tpm_combined_test.csv', index=False, header=False)
test_df_column_names.to_csv('tpm_combined_cols_test.csv', index=False, header=False)
test_df_row_names.to_csv('tpm_combined_rows_test.csv', index=False, header=False)

train_features.to_csv('tpm_combined_wheader.csv')
test_features.to_csv('tpm_combined_test_wheader.csv')

#df_ltpm_combined_train.to_csv('tpm_combined_train_nonorm.csv')
#df_ltpm_combined_test.to_csv('tpm_combined_test_nonorm.csv')