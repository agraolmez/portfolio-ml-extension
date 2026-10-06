# Execute the pinned functions themselves against the complete fresh history.
root <- normalizePath(commandArgs(trailingOnly=TRUE)[1])
.libPaths(c(file.path(root, "env/step2/R-library"), .libPaths()))
suppressPackageStartupMessages({library(data.table); library(dplyr); library(lubridate); library(stringr)})
setDTthreads(2)
out <- file.path(root, "output/step3_author_conventions_20260911")
inputs <- file.path(out, "r_inputs")
wanted <- c("long_horizon_ret", "size_screen_fun", "investment_universe", "addition_deletion_fun")
for (expr in parse(file.path(root, "references/author_code/0 - General functions.R"))) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) && as.character(expr[[2]]) %in% wanted) eval(expr)
}
monthly <- fread(file.path(inputs, "monthly.csv")); monthly[, eom := as.Date(eom)]
actual_returns <- long_horizon_ret(monthly, h=12, impute="zero")
fwrite(actual_returns, file.path(out, "r_actual_future_returns.csv"))
chars <- fread(file.path(inputs, "screen_input.csv")); chars[, eom := as.Date(eom)]
chars <- chars[!is.na(me) & !is.na(tr_ld0_r0) & !is.na(tr_ld1_r0) & !is.na(dolvol) & dolvol>0 & !is.na(sic) & raw_feature_count>=57]
chars[, valid_data := TRUE]
setorder(chars, id, eom)
chars[, eom_lag := shift(eom, n=12, type="lag"), by=id]
chars[, month_diff := interval(eom_lag,eom) %/% months(1)]
chars[, valid_data := valid_data & month_diff==12 & !is.na(month_diff)]
size_screen_fun(chars, type="perc_low50_high100_min50")
flags <- copy(chars[, .(id,eom,valid_data,valid_size)])
addition_deletion_fun(chars, addition_n=12, deletion_n=12)
flags <- chars[, .(id,eom,valid)][flags,on=.(id,eom)]
fwrite(flags, file.path(out, "r_actual_screens.csv"))
raw <- fread(file.path(inputs, "raw_rank_samples.csv"), colClasses="character")
raw[, id := as.numeric(id)]
features <- setdiff(names(raw),c("id","eom"))
# Exact hexadecimal literals preserve raw adjacent-double ECDF ties across tools.
raw[, (features) := lapply(.SD,as.numeric), .SDcols=features]
for (f in features) {
  raw[, zero := get(f)==0]
  raw[!is.na(get(f)), (f) := ecdf(get(f))(get(f)), by=eom]
  raw[zero==TRUE, (f) := 0][, zero := NULL]
  raw[is.na(get(f)), (f) := .5]
}
fwrite(raw,file.path(out,"r_actual_rank_samples.csv"))
writeLines(capture.output(sessionInfo()),file.path(out,"r_session.txt"))
