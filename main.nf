#!/usr/bin/env nextflow
/*
========================================================================================
                         luslab/scrnaseq-gann
========================================================================================
----------------------------------------------------------------------------------------
*/

// nextflow main.nf -profile docker --downloadDataDir "/Users/cheshic/dev/repos/arshamg-scrnaseq-wgan/data/" --prepareDataDir "/Users/cheshic/dev/repos/arshamg-scrnaseq-wgan/data/"

// Define DSL2
nextflow.preview.dsl=2

/* Module inclusions 
--------------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------------*/
/* Params
--------------------------------------------------------------------------------------*/

params.epochs = 10
params.writeFreq = 10

staticDownloadPaths = [
  "${params.downloadDataDir}GSE67602_Joost_et_al_expression.txt",
  "${params.downloadDataDir}GSE90848_Ana6_basal_hair_bulb_TPM.txt",
  "${params.downloadDataDir}GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt",
  "${params.downloadDataDir}GSE99989_NCA_BCatenin_TPM_matrix.csv",
  "${params.downloadDataDir}Mus_musculus.GRCm38.99.gtf",
  "${params.downloadDataDir}gene_names.tsv",
  "${params.downloadDataDir}gtf.txt"
]

staticPreparePaths = [
  "${params.prepareDataDir}tpm_combined.csv",
  "${params.prepareDataDir}tpm_combined_cols.csv",
  "${params.prepareDataDir}tpm_combined_cols_test.csv",
  "${params.prepareDataDir}tpm_combined_rows.csv",
  "${params.prepareDataDir}tpm_combined_rows_test.csv",
  "${params.prepareDataDir}tpm_combined_test.csv",
  "${params.prepareDataDir}tpm_combined_test_nonorm.csv",
  "${params.prepareDataDir}tpm_combined_train_nonorm.csv"
]

/*------------------------------------------------------------------------------------*/
/* Processes
--------------------------------------------------------------------------------------*/

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

process prepareData {
  publishDir "${params.outdir}/processed",
    mode: "copy", overwrite: true

    input:
      val(files)

    output:
      path "*.csv"

    shell:
      arg1 = files[4]
      arg2 = files[2]
      arg3 = files[1]
      arg4 = files[3]
      arg5 = files[0]
      arg6 = files[5]
      arg7 = 'gtf.txt'

      """
      python $baseDir/bin/prepare-data.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7
      """
}

process runGann {
  publishDir "${params.outdir}/gann",
    mode: "copy", overwrite: true

    input:
      val(files)

    output:
      path "gen/*.csv"

    shell:
      arg1 = files[0]
      arg2 = files[1]
      arg3 = files[2]
      arg4 = files[3]
      arg5 = files[4]
      arg6 = files[5]
      arg7 = files[6]
      arg8 = files[7]

      """
      mkdir gen logs
      python $baseDir/bin/gann.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 ${params.epochs} ${params.writeFreq}
      """
}

/*------------------------------------------------------------------------------------*/

// Run workflow
workflow {
    // Create channels for static data
    if(params.downloadDataDir) {
      Channel
      .from(staticDownloadPaths)
      .set {ch_static_download}
    }

    if(params.prepareDataDir) {
      Channel
      .from(staticPreparePaths)
      .set {ch_static_prepare}
    }

    // Download data
    if(!params.downloadDataDir) {
      download()
    }

    // Prepare data
    if(!params.downloadDataDir) {
      prepareData( download.out.collect() )
    }
    else if (!params.prepareDataDir) {
      prepareData( ch_static_download.collect() )
    }

    // run gann
    if(!params.prepareDataDir) {
      runGann( prepareData.out.collect() )
    }
    else {
      runGann( ch_static_prepare.collect() )
    }
}

workflow.onComplete {
    log.info "\nPipeline complete!\n"
}