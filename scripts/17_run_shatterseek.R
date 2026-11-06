# Step 17. ShatterSeek chromosome-level statistics for every tumor
suppressMessages({ library(data.table); library(ShatterSeek); library(parallel) })

root <- Sys.getenv("GCPS_ROOT", getwd())
out  <- file.path(Sys.getenv("GCPS_OUT", file.path(root, "results")), "16_chromothripsis")
jobs <- as.integer(Sys.getenv("GCPS_JOBS", "8"))

sv  <- fread(file.path(out, "shatterseek_sv_input.csv"),
             colClasses = list(character = c("chrom1", "chrom2", "strand1", "strand2", "SVtype")))
cnv <- fread(file.path(out, "shatterseek_cnv_input.csv"), colClasses = list(character = "chrom"))
samples <- sort(unique(c(sv$sample_id, cnv$sample_id)))

run_sample <- function(id) {
  s <- sv[sample_id == id]; c <- cnv[sample_id == id]
  if (nrow(s) == 0 || nrow(c) == 0) return(NULL)
  tryCatch({
    sv_obj  <- SVs(chrom1 = s$chrom1, pos1 = s$pos1, chrom2 = s$chrom2, pos2 = s$pos2,
                   SVtype = s$SVtype, strand1 = s$strand1, strand2 = s$strand2)
    cnv_obj <- CNVsegs(chrom = c$chrom, start = c$start, end = c$end, total_cn = c$total_cn)
    res <- shatterseek(SV.sample = sv_obj, seg.sample = cnv_obj, min.Size = 1, genome = "hg38")
    cbind(sample_id = id, res@chromSummary)
  }, error = function(e) NULL)
}

summaries <- rbindlist(mclapply(samples, run_sample, mc.cores = jobs), fill = TRUE)
fwrite(summaries, file.path(out, "shatterseek_chromsummary.csv"))
cat(length(unique(summaries$sample_id)), "of", length(samples), "tumors processed\n")
