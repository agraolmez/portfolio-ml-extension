# Independent R mean/population-variance calculation for all saved bootstrap draws.
args<-commandArgs(trailingOnly=TRUE); folder<-args[1]
max_error<-0; checked<-0; interval_error<-0; interval_count<-0
intervals<-read.csv(file.path(folder,"primary_intervals.csv"),check.names=FALSE)
for(period in c('evaluation','exposed_robustness')){
 x<-as.matrix(read.csv(file.path(folder,paste0(period,'_returns.csv')),check.names=FALSE));n<-nrow(x)
 for(block in c(12,24)){
  prefix<-file.path(folder,paste0(period,'_L',block));raw<-readBin(paste0(prefix,'_indices.bin'),integer(),n=10000*n,size=4,endian='little')
  if(length(raw)!=10000*n)stop('Incomplete bootstrap index file')
  idx<-matrix(raw,nrow=10000,byrow=TRUE)+1L
  expected<-as.matrix(read.csv(paste0(prefix,'_utility_python.csv'),header=FALSE))
  actual<-matrix(NA_real_,nrow=10000,ncol=ncol(x))
  for(b in 1:10000){z<-x[idx[b,],,drop=FALSE];mu<-colMeans(z);v<-colMeans(sweep(z,2,mu,'-')^2);actual[b,]<-12*(mu-5*v)}
  err<-max(abs(actual-expected));if(!is.finite(err)||err>1e-11)stop('Independent R utility mismatch')
  max_error<-max(max_error,err);checked<-checked+length(actual)
  mu<-colMeans(x);v<-colMeans(sweep(x,2,mu,'-')^2);point<-12*(mu-5*v)
  for(method in c('fixed512','risk_only','return_prediction','joint','joint_refit','fixed1024')){
   ia<-match(paste0('s',1:3,'_full'),colnames(x));ib<-match(paste0('s',1:3,'_',method),colnames(x))
   observed<-mean(point[ia]-point[ib]);draws<-rowMeans(actual[,ia,drop=FALSE]-actual[,ib,drop=FALSE])
   qq<-quantile(draws,probs=c(.05/12,1-.05/12),type=7,names=FALSE);bound<-2*observed-rev(qq)
   row<-intervals[intervals$period==period & intervals$block_months==block & intervals$contrast==paste0('full-',method),]
   if(nrow(row)!=1)stop('Missing unique interval to verify')
   ce<-max(abs(c(observed,bound)-c(row$mean_seed_delta,row$adjusted_lower,row$adjusted_upper)))
   if(!is.finite(ce)||ce>1e-11)stop('Independent R contrast/interval mismatch')
   interval_error<-max(interval_error,ce);interval_count<-interval_count+1
  }
 }
}
# No JSON package dependency is needed for this scalar receipt.
cat(sprintf('{"status":"PASS","bootstrap_utilities_checked":%d,"maximum_absolute_error":%.17g,"contrast_intervals_checked":%d,"interval_maximum_error":%.17g,"calculation":"R column means and explicit squared deviations; paired seed-average differences; type-7 basic adjusted intervals"}\n',checked,max_error,interval_count,interval_error),file=file.path(folder,'R_verification.json'))
cat('Independent R utility checks passed:',checked,'maximum error',max_error,'\n')
