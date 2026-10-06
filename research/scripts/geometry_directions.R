args<-commandArgs(trailingOnly=TRUE)
out<-args[1];dir.create(out,recursive=TRUE,showWarnings=FALSE)
for(s in c(1,2,3)) for(gi in 0:1){
 g<-exp(-3+gi)
 set.seed(s); W<-t(MASS::mvrnorm(n=256,mu=rep(0,115),Sigma=g*diag(115)))
 set.seed(1000+s); add<-t(MASS::mvrnorm(n=256,mu=rep(0,115),Sigma=g*diag(115)))
 writeBin(as.double(t(cbind(W,add))),file.path(out,paste0('seed',s,'_g',gi,'.bin')),size=8,endian='little')
}
writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
