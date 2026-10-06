root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'))
out<-file.path(root,'output/step4_20260911/reference')
meta<-read_json(file.path(out,'historical_metadata.json'),simplifyVector=TRUE)
data_tc<-fread(file.path(out,'historical_data.csv'));data_tc[,eom:=as.Date(eom)];data_tc[,valid:=as.logical(valid)];setorder(data_tc,eom,id)
risk_free<-fread(file.path(out,'historical_rates.csv'))[,.(eom=as.Date(eom),rf=rf_french)]
wealth<-unique(data_tc[,.(eom,wealth=wealth_usd,mu_ld0,mu_ld1)])
cov_list<-list();lambda_list<-list()
for(j in seq_len(nrow(meta$risk))){
 z<-meta$risk[j,];d<-z$date;folder<-file.path(out,'risk',d)
 ids<-as.character(read_json(file.path(folder,'ids.json'),simplifyVector=TRUE))
 X<-read_matrix(file.path(folder,'loadings.bin'),z$n,z$k);rownames(X)<-ids
 iv<-drop(read_matrix(file.path(folder,'specific.bin'),z$n,1));names(iv)<-ids
 cov_list[[d]]<-list(fct_load=X,fct_cov=read_matrix(file.path(folder,'factor.bin'),z$k,z$k),ivol_vec=iv)
 tmp<-data_tc[eom==as.Date(d)&valid==TRUE];lv<-tmp$lambda_diagonal;names(lv)<-as.character(tmp$id);lambda_list[[d]]<-lv
}
for(d in meta$moment_dates){
 cat('Actual source moment:',d,'\n');flush.console()
 subset<-data_tc[eom>=as.Date(d)+1-months(13)-1 & eom<=as.Date(d)]
 ids<-as.character(data_tc[eom==as.Date(d)&valid==TRUE,id])
 sigma<-create_cov(cov_list[[d]],ids)
 m<-m_func(wealth[eom==as.Date(d),wealth],mu=pf_set$mu,rf=risk_free[eom==as.Date(d),rf],sigma_gam=sigma*pf_set$gamma_rel,gam=pf_set$gamma_rel,lambda=create_lambda(lambda_list[[d]],ids),iter=10)
 write_matrix(m,file.path(out,paste0('r_m_',d,'.bin')))
 for(gi in 0:1){
  x<-pfml_input_fun(copy(subset),cov_list,lambda_list,gamma_rel=pf_set$gamma_rel,wealth=wealth,mu=pf_set$mu,dates=as.Date(d),lb=pf_set$lb_hor,scale=TRUE,risk_free=risk_free,features=features,rff_feat=TRUE,seed=settings$seed_no,p_max=max(settings$pf_ml$p_vec),g=settings$pf_ml$g_vec[gi+1],add_orig=FALSE,iter=10,balanced=FALSE)
  nm<-pfml_feat_fun(max(settings$pf_ml$p_vec),FALSE);prefix<-file.path(out,paste0('r_moment_g',gi,'_',d,'_'))
  for(field in c('r_tilde','denom','risk','tc')){
   a<-x$reals[[d]][[field]]
   if(field=='r_tilde')a<-matrix(a[nm],nrow=1) else a<-a[nm,nm]
   write_matrix(a,paste0(prefix,field,'.bin'))
  }
  write_matrix(x$signal_t[[d]][,nm],paste0(prefix,'signal_t.bin'))
 }
}
dates<-as.Date(meta$mechanics_dates);data<-data_tc[valid==TRUE&eom%in%dates]
data[,`:=`(lambda=lambda_diagonal,pred_ld1=0)]
aims<-fread(file.path(out,'mechanics_aims.csv'));aims[,eom:=as.Date(eom)]
w<-pfml_w(data,dates,cov_list,lambda_list,gamma_rel=pf_set$gamma_rel,iter=10,risk_free=risk_free,wealth=wealth,mu=pf_set$mu,aims=aims)
fwrite(w,file.path(out,'r_mechanics_weights.csv'))
pf<-pf_ts_fun(w,data,wealth,gam=pf_set$gamma_rel);fwrite(pf,file.path(out,'r_mechanics_returns.csv'))
write_json(list(status='ACTUAL_PINNED_R_EXECUTED',moment_dates=meta$moment_dates,mechanics_dates=meta$mechanics_dates,mechanics_positions=nrow(w)),file.path(out,'historical_status.json'),auto_unbox=TRUE,pretty=TRUE)
cat('Historical R checks complete.\n')
