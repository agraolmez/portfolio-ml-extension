# Independent index reconstruction and full return/interval arithmetic in base R.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
out <- file.path(root, 'output/step10_20260913')
dest <- file.path(out, 'reference')
if (dir.exists(dest)) stop('Reference destination already exists; preserve attempt')
dir.create(dest)
methods <- c('R1','B1','B2','B3','A0','A1','A2')
variants <- c('primary','horizon36','blocks24')
periods <- c('selection','evaluation','robustness')
B <- 10000L
interval_rows <- list(); observed_rows <- list(); k <- 0L; j <- 0L
index_checks <- 0L
score <- function(x) 12 * (mean(x) - 5 * mean((x-mean(x))^2))
for (period in periods) {
  n <- if (period=='robustness') 60L else 120L
  for (L in c(12L,24L)) {
    tag <- paste0(period,'_L',L)
    read_int <- function(kind) {
      con <- file(file.path(out,'reference_inputs',paste0(tag,'_',kind,'.bin')),'rb')
      on.exit(close(con))
      x <- readBin(con,integer(),n*B,size=4,endian='little')
      stopifnot(length(x)==n*B)
      matrix(x,nrow=n,ncol=B)
    }
    idx <- read_int('indices'); starts <- read_int('starts'); restart <- read_int('restart')
    stopifnot(all(starts>=0L & starts<n), all(restart[1,]==1L))
    replay <- matrix(0L,nrow=n,ncol=B); replay[1,] <- starts[1,]
    for (t in 2:n) {
      continued <- replay[t-1,]+1L; continued[continued==n] <- 0L
      replay[t,] <- ifelse(restart[t,]==1L,starts[t,],continued)
    }
    stopifnot(identical(idx,replay)); index_checks <- index_checks+1L
    for (variant in variants) {
      d <- read.csv(file.path(out,'inputs',paste0(period,'_',variant,'_returns.csv')),check.names=FALSE)
      stopifnot(nrow(d)==n,all(methods %in% names(d)))
      x <- as.matrix(d[,methods]); stopifnot(all(is.finite(x)))
      u <- matrix(0,nrow=B,ncol=length(methods)); observed <- numeric(length(methods))
      for (m in seq_along(methods)) {
        y <- matrix(x[,m][as.vector(idx)+1L],nrow=n,ncol=B)
        means <- colMeans(y)
        centered <- sweep(y,2,means,'-')
        u[,m] <- 12*(means-5*colMeans(centered^2))
        observed[m] <- score(x[,m])
        j <- j+1L
        observed_rows[[j]] <- data.frame(period,variant,block_months=L,method=methods[m],utility=observed[m])
      }
      con <- file(file.path(dest,paste0(period,'_',variant,'_L',L,'_utility.bin')),'wb')
      writeBin(as.double(u),con,size=8,endian='little'); close(con)
      pairs <- combn(seq_along(methods),2)
      for (pair in seq_len(ncol(pairs))) {
        a<-pairs[1,pair]; b<-pairs[2,pair]; estimate<-observed[a]-observed[b]
        qs<-quantile(u[,a]-u[,b],c(.025,.975),type=7,names=FALSE)
        k<-k+1L; interval_rows[[k]]<-data.frame(period,variant,block_months=L,contrast=paste0(methods[a],'-',methods[b]),
          family=1L,estimate,lower=2*estimate-qs[2],upper=2*estimate-qs[1])
      }
      # The frozen selection is independently verified in Python; all variants selected R1.
      for (b_name in c('A0','R1','C*')) {
        a<-match('A1',methods); b<-match(if (b_name=='C*') 'R1' else b_name,methods)
        estimate<-observed[a]-observed[b]
        qs<-quantile(u[,a]-u[,b],c(.05/6,1-.05/6),type=7,names=FALSE)
        k<-k+1L; interval_rows[[k]]<-data.frame(period,variant,block_months=L,contrast=paste0('A1-',b_name),
          family=3L,estimate,lower=2*estimate-qs[2],upper=2*estimate-qs[1])
      }
      cat('Recomputed',period,variant,L,':',B,'complete utility vectors\n')
    }
  }
}
options(digits=17)
write.csv(do.call(rbind,interval_rows),file.path(dest,'intervals.csv'),row.names=FALSE)
write.csv(do.call(rbind,observed_rows),file.path(dest,'observed.csv'),row.names=FALSE)
writeLines(paste('All',index_checks,'complete index arrays reconstructed; 18 utility arrays and 432 intervals.'),file.path(dest,'completion.txt'))
