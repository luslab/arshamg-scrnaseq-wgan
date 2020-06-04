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
params.datadir = ''

/*------------------------------------------------------------------------------------*/
/* Processes
--------------------------------------------------------------------------------------*/

process runGann {
  publishDir "${params.outdir}/gann",
    mode: "copy", overwrite: true

    input:
      path(datadir)

    output:
      path("pbmc_output/*")

    shell:
      """
      mkdir pbmc_output pbmc_output/figures pbmc_output/logs pbmc_output/gen_profiles
      python $baseDir/bin/scgan/main.py --pbmc_train --data_path $datadir --training_output pbmc_output --epochs ${params.epochs} --write_freq ${params.writefreq}
      """
}

/*------------------------------------------------------------------------------------*/

// Run workflow
workflow {

  Channel
    .from(params.datadir)
    .set {ch_data}

    // run gann
    runGann(ch_data)
}