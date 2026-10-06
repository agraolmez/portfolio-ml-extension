# Independent R lm check on the captured historical design matrices.
args <- commandArgs(trailingOnly=TRUE)
root <- if (length(args)) args[[1]] else "."
p <- file.path(root, "output", "step3_20260911")
x <- read.csv(file.path(p, "full_factor_regression_R_comparison_inputs.csv"), check.names=FALSE)
terms <- setdiff(names(x), c("dlycaldt", "permno", "ret_exc"))
form <- as.formula(paste("ret_exc~-1+", paste(terms, collapse="+")))
result <- lapply(split(x, x$dlycaldt), function(g) {
  b <- coef(lm(form, data=g))
  data.frame(date=g$dlycaldt[1], term=names(b), estimate=as.numeric(b))
})
write.csv(do.call(rbind,result), file.path(p,"full_factor_regression_R_comparison.csv"), row.names=FALSE)
