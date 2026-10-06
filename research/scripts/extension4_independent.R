args <- commandArgs(trailingOnly=TRUE)
folder <- args[[1]]
u <- function(x) 12 * (mean(x) - 5 * mean((x - mean(x))^2))
if (length(args) == 1) {
  x <- as.matrix(read.csv(file.path(folder, 'returns.csv'), header=FALSE))
  idx <- as.matrix(read.csv(file.path(folder, 'indices.csv'), header=FALSE))
  if (any(idx < 1 | idx > nrow(x)) || ncol(idx) != nrow(x)) stop('Invalid paired month indices')
  observed <- apply(x, 2, u)
  draws <- t(vapply(seq_len(nrow(idx)), function(i) apply(x[idx[i,],,drop=FALSE], 2, u), numeric(ncol(x))))
  write.table(draws, file.path(folder, 'R_draws.csv'), sep=',', row.names=FALSE, col.names=FALSE)
  write.table(t(observed), file.path(folder, 'R_observed.csv'), sep=',', row.names=FALSE, col.names=FALSE)
} else {
  mapping <- as.matrix(read.csv(file.path(folder, 'contrast_columns.csv'), header=FALSE))
  draws <- as.matrix(read.csv(file.path(folder, 'R_draws.csv'), header=FALSE))
  points <- as.numeric(read.csv(file.path(folder, 'R_observed.csv'), header=FALSE)[1,])
  alpha <- as.numeric(args[[3]]); family <- as.numeric(args[[4]])
  n <- ncol(mapping)/2
  result <- t(vapply(seq_len(nrow(mapping)), function(i) {
    a <- mapping[i,seq_len(n)]; b <- mapping[i,n+seq_len(n)]
    value <- mean(points[a]-points[b]); delta <- rowMeans(draws[,a,drop=FALSE]-draws[,b,drop=FALSE])
    bounds <- quantile(delta, c(alpha/(2*family),1-alpha/(2*family)), type=7, names=FALSE)
    c(value,2*value-bounds[2],2*value-bounds[1])
  }, numeric(3)))
  write.table(result,file.path(folder,'R_intervals.csv'),sep=',',row.names=FALSE,col.names=FALSE)
}
cat('PASS: independent R calculations completed\n')
