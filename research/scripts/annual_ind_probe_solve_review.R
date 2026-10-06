args <- commandArgs(trailingOnly=TRUE)
folder <- args[[1]]
q <- as.integer(args[[2]])
p <- as.integer(args[[3]])
ridge <- as.numeric(args[[4]])
read_double <- function(name, n) {
  con <- file(file.path(folder,name), 'rb')
  on.exit(close(con))
  x <- readBin(con, 'double', n=n, size=8, endian='little')
  if(length(x)!=n || any(!is.finite(x))) stop('Invalid transfer')
  x
}
curvature <- matrix(read_double('curvature.bin',q*q), nrow=q)
score <- read_double('score.bin',q)
answers <- sapply(c(100,10,1), function(multiplier) {
  penalty <- c(rep(ridge,p),rep(multiplier*ridge,q-p))
  solve(curvature+diag(penalty),score)
})
write.table(answers,file=file.path(folder,'R_coefficients.csv'),sep=',',row.names=FALSE,col.names=FALSE)
cat('PASS: independent R solves completed\n')
