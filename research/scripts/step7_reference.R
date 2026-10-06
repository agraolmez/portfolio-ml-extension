# Independent sampled conditional refits. The released source uses solve(A+l*I,r).
root <- normalizePath(commandArgs(trailingOnly=TRUE)[1])
library(jsonlite)
out <- file.path(root,'output/step7_20260912')
stopifnot(!file.exists(file.path(out,'reference_status.json')))
models <- fromJSON(file.path(out,'annual_models.json'))
results <- list(); counter <- 0
for (gi in 0:1) {
  folder <- file.path(root,'output/step6_20260911/reference/fit',paste0('g',gi))
  dates <- fromJSON(file.path(folder,'dates.json')); nm <- length(dates); p <- 513
  con <- file(file.path(folder,'denom.bin'),'rb')
  # Columns are chronological months; each column has one row-major matrix.
  a <- matrix(readBin(con,numeric(),n=nm*p*p,size=8,endian='little'),nrow=p*p);close(con)
  con <- file(file.path(folder,'rtilde.bin'),'rb')
  r <- matrix(readBin(con,numeric(),n=nm*p,size=8,endian='little'),nrow=p);close(con)
  stopifnot(all(is.finite(a)),all(is.finite(r)))
  for (j in which(models$g_index==gi)) {
    row <- models[j,]; n <- row$training_months
    stopifnot(dates[n]==row$training_cutoff,dates[n+1]==row$first_decision)
    for (length in c(12,24)) {
      ref <- file.path(out,'reference',paste0('block',length))
      counts <- as.matrix(read.csv(file.path(ref,paste0(row$fit_year,'_counts.csv')),header=FALSE))
      python <- as.matrix(read.csv(file.path(ref,paste0(row$fit_year,'_coefficients.csv')),header=FALSE))
      stopifnot(nrow(counts)==3,ncol(counts)==n,all(rowSums(counts)==n))
      fullcounts <- matrix(0,nrow=3,ncol=nm);fullcounts[,seq_len(n)] <- counts
      ma <- a %*% t(fullcounts)/n; mr <- r %*% t(fullcounts)/n
      for (b in 1:3) {
        matrix_a <- matrix(ma[,b],nrow=p,byrow=TRUE)
        coef <- drop(solve(matrix_a+row$penalty*diag(p),mr[,b]))
        err <- max(abs(coef-python[,b]))
        pass <- all(abs(coef-python[,b]) <= 1e-12+1e-9*abs(python[,b]))
        stopifnot(pass)
        counter<-counter+1;results[[counter]]<-data.frame(fit_year=row$fit_year,block_months=length,
          replica=c(1,50,100)[b],max_abs_error=err,passed=pass)
      }
    }
    cat('Independent R sampled replicas:',row$fit_year,'both block lengths.\n');flush.console()
  }
  rm(a,r);gc()
}
result<-do.call(rbind,results)
write.csv(result,file.path(out,'reference_comparison.csv'),row.names=FALSE)
write_json(list(status='PASS',fits=nrow(result),max_abs_error=max(result$max_abs_error),
  scope='Replicas 1,50,100 for each of 40 annual models and both block lengths; 240 sampled independent refits',
  source='Previously accepted Step 6 chronological moment exports; R rebuilds weighted means and uses general solve'),
  file.path(out,'reference_status.json'),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,'reference_session.txt'))
