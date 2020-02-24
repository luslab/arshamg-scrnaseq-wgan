library(biomaRt)

# Data paths
GTF_path = "../../scRNAseq/scRNAseq_runII/mouse_bt2_ref/Mus_musculus.GRCm38.86.gtf"
joost_data_path = "GSE67602_Joost_et_al_expression.txt"
ghahramani_data_path = "combined_tpm_expression_name.csv"
yang_data_path_1 = "GSE90848_Ana6_basal_hair_bulb_TPM.txt"
yang_data_path_3 = "GSE90848_Tel_Ana1_Ana2_bulge_HG_basal_HB_TPM.txt"

# Thresholds - set below as desired
exp_threshold = 6
num_cells_above_thresh = 10
min_num_genes_exp = 500

kasper_data = read.table(joost_data_path, header=T, row.names=1)

gtf_data = read.table(GTF_path, stringsAsFactor=F, skip=5, sep="\t")
gtf_data$gene_id = sub(".*gene_id *(.*?) *;.*", "\\1", gtf_data$V9)

gtf_data_filt = gtf_data[gtf_data$V3 %in% c("exon", "three_prime_utr", "five_prime_utr"),]
gtf_data_filt$feat_len = gtf_data_filt$V5 - gtf_data_filt$V4

convertIDs_order_engi_name <- function(engi) {
	refseq<-as.character(engi)
	ensembl <- useMart("ENSEMBL_MART_ENSEMBL",dataset="mmusculus_gene_ensembl", host="may2012.archive.ensembl.org")	
	result<-getBM(filters="ensembl_gene_id", attributes=c("ensembl_gene_id","external_gene_id"), values=engi, mart=ensembl)
	result<-result[match(engi, result$ensembl_gene_id),]
	return(result)
}

convert_gtf_engis = convertIDs_order_engi_name(gtf_data_filt$gene_id)

gtf_data_filt$gene_name = convert_gtf_engis$external_gene_id

gene_eff_len = aggregate(gtf_data_filt$feat_len, by=list(gtf_data_filt$gene_name), sum)

raw_to_gtf_match = match(rownames(kasper_data), gene_eff_len$Group.1)

count_div_length = kasper_data / (gene_eff_len[raw_to_gtf_match,]$x/1000)

scaling_factor = colSums(count_div_length, na.rm=T) / 1000000

tpm_combined = t(t(count_div_length) / scaling_factor)

tpm_kasper = tpm_combined[rowSums(is.na(tpm_combined)) < 1400,]

#load arsham data
arsham_data = read.csv(ghahramani_data_path, stringsAsFactor=F, row.names=1)

#load Fuchs data
yang_1 = read.table(yang_data_path_1, header=T, row.names=1)
yang_1_rownames = unlist(lapply(strsplit(rownames(yang_1),"_"), function(x) x[2]))
yang_1 = yang_1[!is.na(yang_1_rownames),]
yang_1_rownames = yang_1_rownames[!is.na(yang_1_rownames)]
rownames(yang_1) = make.unique(yang_1_rownames)

yang_3 = read.table(yang_data_path_3, header=T, row.names=1)
yang_3_rownames = unlist(lapply(strsplit(rownames(yang_3),"_"), function(x) x[2]))
yang_3 = yang_3[!is.na(yang_3_rownames),]
yang_3_rownames = yang_3_rownames[!is.na(yang_3_rownames)]
rownames(yang_3) = make.unique(yang_3_rownames)

genes_common_to_all = Reduce(intersect, list(rownames(tpm_kasper) ,rownames(arsham_data), rownames(yang_1), rownames(yang_3) ))

kasper_data_red = tpm_kasper[genes_common_to_all,]
arsham_data_red = arsham_data[genes_common_to_all,]
yang_1_red = yang_1[genes_common_to_all,]
yang_3_red = yang_3[genes_common_to_all,]

combined_tpm = cbind(kasper_data_red, arsham_data_red, yang_1_red, yang_3_red)

write.csv(combined_tpm, "four_datasets_combined_TPM_clean.csv", quote=F, row.names=F)

combined_ltpm = log2(combined_tpm + 1)

write.csv(combined_ltpm, "four_datasets_combined_lTPM_clean.csv", quote=F, row.names=F)

combined_ltpm_red = combined_ltpm[rowSums(combined_ltpm > exp_threshold) > num_cells_above_thresh,]
write.csv(combined_ltpm_red, "four_datasets_combined_lTPM_red_clean.csv", quote=F, row.names=F)

combined_ltpm_red_small = combined_ltpm[rowSums(combined_ltpm > 1) > min_num_genes_exp,]
combined_ltpm_red_small = combined_ltpm_red_small[,sample(ncol(combined_ltpm_red_small))]

combined_ltpm_red_small = combined_ltpm_red_small[,colSums(combined_ltpm_red_small > 0) > 1000]

write.csv(combined_ltpm_red_small, "four_datasets_combined_lTPM_red_small_clean.csv", quote=F, row.names=F)

write.table(rownames(combined_ltpm_red_small), "rownames_four_datasets_combined_lTPM_red_small_clean.csv", quote=F, row.names=F, col.names=F, sep=',')
write.table(colnames(combined_ltpm_red_small), "colnames_four_datasets_combined_lTPM_red_small_clean.csv", quote=F, row.names=F, col.names=F, sep=',')


