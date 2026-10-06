a <- commandArgs(TRUE); mode <- a[1]; d <- a[2]
getbin <- function(name,n) readBin(file.path(d,name),what='double',n=n,size=8,endian='little')
if (mode=='system') {
  p <- scan(file.path(d,'dimensions.txt'),quiet=TRUE); n<-p[1]; w<-p[2]; ridge<-p[3]
  A<-matrix(getbin('A.bin',n*n),n,n); b<-getbin('b.bin',n); actual<-matrix(getbin('beta.bin',n*3),n,3)
  errors<-numeric(3)
  for (j in 1:3) {
    penalty<-c(rep(ridge,w),rep(ridge*c(100,10,1)[j],n-w))
    expected<-solve((A+t(A))/2+diag(penalty),b)
    errors[j]<-max(abs(expected-actual[,j]))
    if(any(!is.finite(expected)) || any(abs(expected-actual[,j])>1e-11+1e-10*abs(expected))) stop('Independent coefficient mismatch')
  }
  write.table(errors,file.path(d,'errors.txt'),row.names=FALSE,col.names=FALSE)
} else if (mode=='statistics') {
  x<-read.csv(file.path(d,'monthly.csv')); output<-list(); k<-1
  for(window in list(c(1981,2020),c(2020,2025),c(2021,2025))) {
    z<-x[x$year>=window[1] & x$year<=window[2],]
    b<-z[z$method=='B1',]; i<-z[z$method=='IND',]; b<-b[order(b$return_eom),];i<-i[order(i$return_eom),]
    stopifnot(identical(b$return_eom,i$return_eom))
    flow<-function(v) v$gross_excess-v$impact_cost-5*(v$gross_excess-mean(v$gross_excess))^2
    utility<-function(v) 12*(mean(v$gross_excess-v$impact_cost)-5*var(v$gross_excess))
    delta<-flow(i)-flow(b); zz<-mean(delta)/(sd(delta)/sqrt(length(delta)))
    output[[k]]<-data.frame(first=window[1],last=window[2],months=nrow(b),baseline=utility(b),IND=utility(i),z=zz,p=pnorm(zz,lower.tail=FALSE)); k<-k+1
  }
  write.csv(do.call(rbind,output),file.path(d,'statistics.csv'),row.names=FALSE)
} else stop('Unknown verification mode')
