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

params.epochs = 1
params.writefreq = 1

/*------------------------------------------------------------------------------------*/
/* Processes
--------------------------------------------------------------------------------------*/

process runGann {
  publishDir "${params.outdir}/gann",
    mode: "copy", overwrite: true

    output:
      path("pbmc/*")

    shell:
      """
      mkdir pbmc pbmc/figures pbmc/logs pbmc/gen_profiles
      python $baseDir/bin/scgan/main.py --pbmc_train --data_path ${params.datadir} --training_output pbmc --epochs ${params.epochs} --write_freq ${params.writefreq}
      """
}

/*------------------------------------------------------------------------------------*/

// Run workflow
workflow {
    // run gann
    runGann()
}