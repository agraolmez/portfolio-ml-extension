# Independent R implementation: weighted ridge via augmented QR, including tuning.
args <- commandArgs(trailingOnly=TRUE)
root <- args[[1]]
.libPaths(c(file.path(root,'env/step2/R-library'),.libPaths()))
library(jsonlite)
out <- file.path(root,'output/step8_20260913/reference')
inputs <- fromJSON(file.path(out,'inputs.json'),simplifyVector=FALSE)
fit_qr <- function(x,y,w,penalty) {
  x<-as.matrix(x);y<-as.matrix(y);w<-w/sum(w)
  center<-colSums(x*w); sc<-sqrt(colSums(sweep(x,2,center)^2*w)); constant<-sc<=1e-12;sc[constant]<-1
  z<-sweep(sweep(x,2,center),2,sc,'/');z[,constant]<-0
  design<-cbind(1,z)
  regularizer<-cbind(0,sqrt(penalty)*diag(ncol(x)))
  co<-qr.solve(rbind(design*sqrt(w),regularizer),rbind(y*sqrt(w),matrix(0,ncol(x),ncol(y))),tol=1e-12)
  list(mean=center,scale=sc,constant=constant,co=co)
}
pred <- function(f,x) {
 z<-sweep(sweep(as.matrix(x),2,f$mean),2,f$scale,'/');z[,f$constant]<-0;cbind(1,z)%*%f$co
}
python_pred <- function(f,x) {
 z<-sweep(sweep(as.matrix(x),2,unlist(f$mean)),2,unlist(f$scale),'/');z[,unlist(f$constant)]<-0
 co<-do.call(rbind,lapply(f$coefficients,unlist));sweep(z%*%co,2,unlist(f$intercept),'+')
}
records<-list();idx<-1
for (item in inputs) {
 d<-read.csv(item$csv,check.names=FALSE); expected<-fromJSON(item$fit,simplifyVector=FALSE)
 for (method in c('A0','A1')) {
  p<-if(method=='A0')4 else 6;x<-as.matrix(d[paste0('x',0:(p-1))]);y<-as.matrix(d[c('b','c')]);w<-d$weight
  tr<-d$inner_train==1;va<-d$validation==1;wn<-w[tr]/sum(w[tr]);center<-colSums(y[tr,,drop=FALSE]*wn)
  variance<-pmax(colSums(sweep(y[tr,,drop=FALSE],2,center)^2*wn),1e-12)
  penalties<-c(.01,.1,1,10)
  losses<-sapply(penalties,function(penalty) {
    f<-fit_qr(x[tr,,drop=FALSE],y[tr,,drop=FALSE],w[tr],penalty)
    mean(colSums((pred(f,x[va,,drop=FALSE])-y[va,,drop=FALSE])^2*(w[va]/sum(w[va])))/variance)
  })
  best<-min(losses);chosen<-max(penalties[losses<=best+1e-12*(1+abs(best))])
  f<-fit_qr(x,y,w,chosen);ep<-python_pred(expected$models[[method]],x);actual<-pred(f,x)
  err<-max(abs(actual-ep));relative<-err/max(1,max(abs(ep)))
  loss_error<-max(abs(losses-sapply(expected$models[[method]]$candidates,function(v)v$loss)))/max(1,max(abs(losses)))
  stopifnot(chosen==expected$models[[method]]$penalty,relative<1e-9,loss_error<1e-9)
  records[[idx]]<-data.frame(variant=item$variant,year=item$year,method=method,replica=0,penalty=chosen,max_abs_error=err,relative_error=relative,relative_loss_error=loss_error);idx<-idx+1
 }
 x<-as.matrix(d[paste0('x',0:5)]);y<-matrix(d$b,ncol=1)
 for (replica in c(1,50,100)) {
  f<-fit_qr(x,y,d$weight*d[[paste0('count',replica)]],expected$models$A1$penalty)
  ep<-python_pred(expected$bootstrap[[replica]],x);err<-max(abs(pred(f,x)-ep));relative<-err/max(1,max(abs(ep)))
  stopifnot(relative<1e-9)
  records[[idx]]<-data.frame(variant=item$variant,year=item$year,method='A2_b_replica',replica=replica,penalty=expected$models$A1$penalty,max_abs_error=err,relative_error=relative,relative_loss_error=NA);idx<-idx+1
 }
}
result<-do.call(rbind,records);write.csv(result,file.path(out,'comparison.csv'),row.names=FALSE)
write_json(list(status='PASS',annual_variants=length(inputs),full_target_pairs=2*length(inputs),sampled_bootstrap_fits=3*length(inputs),max_relative_error=max(result$relative_error),max_abs_error=max(result$max_abs_error)),file.path(out,'summary.json'),auto_unbox=TRUE,pretty=TRUE,digits=17)
cat('Independent R weighted augmented-QR checks passed:',nrow(result),'fits\n')
