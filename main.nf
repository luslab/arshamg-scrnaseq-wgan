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

params.epochs = 2
params.writeFreq = 1

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
      echo test
      mkdir gen logs
      python $baseDir/bin/gann.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 ${params.epochs} ${params.writeFreq}
      """
}

/*------------------------------------------------------------------------------------*/

// Run workflow
workflow {

    // Download data
    download()

    // Prepare data
    prepareData( download.out.collect() )

    // run gann
    runGann( prepareData.out.collect() )
}

workflow.onComplete {
    log.info "\nPipeline complete!\n"
}