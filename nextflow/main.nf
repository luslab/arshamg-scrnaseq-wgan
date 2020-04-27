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

// Run workflow
workflow {

    // Create channels for indices
    ch_bowtieIndex = Channel.fromPath( params.bowtie_index )
    ch_starIndex = Channel.fromPath( params.star_index )
    ch_genomeFai = Channel.fromPath( params.genome_fai )

    // Get fastq paths 
    metadata( params.input )
}


workflow.onComplete {
    log.info "\nPipeline complete!\n"
}