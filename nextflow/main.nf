#!/usr/bin/env nextflow
/*
========================================================================================
                         luslab/scrnaseq-gann
========================================================================================
----------------------------------------------------------------------------------------
*/

// Define DSL2
nextflow.preview.dsl=2

/* Module inclusions 
--------------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------------*/
/* Params
--------------------------------------------------------------------------------------*/

//params.input = "$baseDir/test/data/metadata.csv"
// params.input = "metadata.csv"

//params.reads = "$baseDir/test/data/reads/*.fq.gz"
//params.bowtie_index = "$baseDir/test/data/small_rna_bowtie"
//params.star_index = "$baseDir/test/data/reduced_star_index"
//params.genome_fai = "$baseDir/test/data/GRCh38.primary_assembly.genome_chr6_34000000_35000000.fa.fai"
//params.results = "$baseDir/test/data/results"

/*------------------------------------------------------------------------------------*/
/* Processes
--------------------------------------------------------------------------------------*/

/*gtf_path_gz = 'Mus_musculus.GRCm38.99.gtf.gz'
tpm_yang_path_gz = 'GSE90848_Ana6_basal_hair_bulb_TPM.txt.gz''
tpm_yang_path2_gz = 'GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt.gz'
tpm_joost_path_gz = 'GSE67602_Joost_et_al_expression.txt.gz'
tpm_ghahramani_path_gz = 'GSE99989_NCA_BCatenin_TPM_matrix.csv.gz'
tpm_ghahramani_path = 'GSE99989_NCA_BCatenin_TPM_matrix.csv'
gene_tsv_path = 'gene_names.tsv'

gtf_path = data_path + '/Mus_musculus.GRCm38.99.gtf'
tpm_yang_path = data_path + '/GSE90848_Ana6_basal_hair_bulb_TPM.txt'
tpm_yang_path2 = data_path + '/GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt'
tpm_joost_path = data_path + '/GSE67602_Joost_et_al_expression.txt'
tpm_ghahramani_path = data_path + '/GSE99989_NCA_BCatenin_TPM_matrix.csv'
gene_tsv_path = data_path + '/gene_names.tsv'
gtf_data_path = data_path + '/gtf.txt'*/

process download {
  publishDir "${params.outdir}/data",
    mode: "copy", overwrite: true

    output:
      path "*.{txt,tsv,csv,gtf}"

    script:
    """
    python $baseDir/bin/download-data.py
    """
}

/*------------------------------------------------------------------------------------*/

// Run workflow
workflow {

    // Download data
    download().collect().view()
}


workflow.onComplete {
    log.info "\nPipeline complete!\n"
}