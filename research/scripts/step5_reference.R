root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'))
out<-file.path(root,'output/step5_20260911/reference');prior<-file.path(root,'output/step4_20260911/reference')
meta<-read_json(file.path(out,'metadata.json'),simplifyVector=TRUE);dates<-as.Date(meta$decision_dates)
data_tc<-fread(file.path(out,'data.csv'));data_tc[,eom:=as.Date(eom)];data_tc[,valid:=as.logical(valid)];setorder(data_tc,eom,id)
risk_free<-fread(file.path(out,'rates.csv'))[,.(eom=as.Date(eom),rf=rf_french)]
wealth<-unique(data_tc[,.(eom,wealth=wealth_usd,mu_ld0,mu_ld1)])
cov_list<-list();lambda_list<-list()
for(j in seq_len(nrow(meta$risk))){
 z<-meta$risk[j,];d<-z$date;folder<-file.path(out,'risk',d);ids<-as.character(read_json(file.path(folder,'ids.json'),simplifyVector=TRUE))
 X<-read_matrix(file.path(folder,'loadings.bin'),z$n,z$k);rownames(X)<-ids
 iv<-drop(read_matrix(file.path(folder,'specific.bin'),z$n,1));names(iv)<-ids
 cov_list[[d]]<-list(fct_load=X,fct_cov=read_matrix(file.path(folder,'factor.bin'),z$k,z$k),ivol_vec=iv)
 tmp<-data_tc[eom==as.Date(d)&valid==TRUE];v<-tmp$lambda_diagonal;names(v)<-as.character(tmp$id);lambda_list[[d]]<-v
}
validation_all<-fread(file.path(prior,'r_validation.csv'));validation_all[,`:=`(eom=as.Date(eom),eom_ret=as.Date(eom_ret))]
# Preserve exact original grid values after decimal CSV transport; integer index was saved by source.
validation_all[,l:=settings$pf_ml$l_vec[lambda_index+1L]]
stopifnot(!anyNA(match(validation_all$l,settings$pf_ml$l_vec)))
selection<-fread(file.path(prior,'r_annual_selection.csv'));selection[,eom_ret:=as.Date(eom_ret)]
aim_by_g<-list()
for(gi in 0:1){
 cat('Rebuilding all 24 actual source input months for bandwidth',gi,'\n');flush.console()
 nm<-pfml_feat_fun(512,FALSE)
 cached<-gi==0 && file.exists(file.path(root,'output/step5_20260911/reference_cache_receipt.json'))
 if(cached){
  cat('Reusing the 24 complete source input blocks from documented first attempt.\n')
  signals<-lapply(as.character(dates),function(d){
   n<-nrow(data_tc[eom==as.Date(d)&valid==TRUE]);a<-read_matrix(file.path(out,paste0('r_g',gi,'_',d,'_signal.bin')),n,513);colnames(a)<-nm;a
  });names(signals)<-as.character(dates);x<-list(signal_t=signals)
 }else{
 x<-pfml_input_fun(copy(data_tc),cov_list,lambda_list,gamma_rel=pf_set$gamma_rel,wealth=wealth,mu=pf_set$mu,dates=dates,lb=pf_set$lb_hor,scale=TRUE,risk_free=risk_free,features=features,rff_feat=TRUE,seed=settings$seed_no,p_max=max(settings$pf_ml$p_vec),g=settings$pf_ml$g_vec[gi+1],add_orig=FALSE,iter=10,balanced=FALSE)
 for(d in as.character(dates)){
  prefix<-file.path(out,paste0('r_g',gi,'_',d,'_'))
  write_matrix(x$signal_t[[d]][,nm],paste0(prefix,'signal.bin'))
  write_matrix(x$reals[[d]]$denom[nm,nm],paste0(prefix,'denom.bin'))
  write_matrix(matrix(x$reals[[d]]$r_tilde[nm],nrow=1),paste0(prefix,'rtilde.bin'))
 }
 }
 coefs<-list()
 for(yr in 1981:1982){
  coefs[[as.character(yr)]]<-list()
  for(p in settings$pf_ml$p_vec){
   a<-read_matrix(file.path(prior,'fit',paste0('g',gi),paste0('r_coef_',yr,'_p',p,'.bin')),p+1,length(settings$pf_ml$l_vec))
   block<-lapply(seq_along(settings$pf_ml$l_vec),function(j)matrix(a[,j],ncol=1));names(block)<-settings$pf_ml$l_vec
   coefs[[as.character(yr)]][[as.character(p)]]<-block
  }
 }
 aim_by_g[[gi+1]]<-pfml_aims_fun(x,validation=validation_all[g_index==gi],data_tc=data_tc,hp_coef=coefs,hp_years=1981:1982,dates_oos=dates,l_vec=settings$pf_ml$l_vec,orig_feat=FALSE)
 rm(x,coefs);gc()
}
aims<-rbindlist(lapply(as.character(dates),function(d){
 ret_year<-year(as.Date(d)+1+months(1)-1);choice<-selection[year(eom_ret)==ret_year-1];stopifnot(nrow(choice)==1)
 answer<-copy(aim_by_g[[choice$g_index+1]][[d]]$aim_pf);answer[,`:=`(g_index=choice$g_index,p=choice$p,lambda_index=choice$lambda_index)];answer
}));fwrite(aims,file.path(out,'r_aims.csv'))
data<-data_tc[valid==TRUE&eom%in%dates];data[,`:=`(lambda=lambda_diagonal,pred_ld1=0)]
cat('Executing full 24-month source holdings recursion.\n');flush.console()
w<-pfml_w(data,dates,cov_list,lambda_list,gamma_rel=pf_set$gamma_rel,iter=10,risk_free=risk_free,wealth=wealth,mu=pf_set$mu,aims=aims[,.(id,eom,w_aim)])
fwrite(w,file.path(out,'r_weights.csv'))
fwrite(pf_ts_fun(w,data,wealth,gam=pf_set$gamma_rel),file.path(out,'r_performance.csv'))
writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
write_json(list(status='ACTUAL_PINNED_R_PILOT_COMPLETE',months=length(dates),positions=nrow(w),bandwidths=2,aim_source='Actual R input builder, prior independently executed R coefficients/selection and actual R aims function',inputs='Identical accepted R1 economic inputs; not original paper-vintage equality'),file.path(out,'status.json'),auto_unbox=TRUE,pretty=TRUE)
cat('Actual R pilot complete.\n')
