root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'))
out<-file.path(root,'output/step6_20260911/reference')
stopifnot(!file.exists(file.path(out,'path_status.json')))
meta<-read_json(file.path(out,'metadata.json'),simplifyVector=TRUE)
dates<-as.Date(meta$decision_dates);sample_dates<-as.Date(meta$sample_input_dates)
data_tc<-fread(file.path(out,'data.csv'));data_tc[,eom:=as.Date(eom)];data_tc[,valid:=as.logical(valid)];setorder(data_tc,eom,id)
risk_free<-fread(file.path(out,'rates.csv'))[,.(eom=as.Date(eom),rf=rf_french)]
wealth<-unique(data_tc[,.(eom,wealth=wealth_usd,mu_ld0,mu_ld1)])
cov_list<-list();lambda_list<-list()
for(j in seq_len(nrow(meta$risk))){
 z<-meta$risk[j,];d<-z$date;folder<-file.path(out,'risk',d)
 ids<-as.character(read_json(file.path(folder,'ids.json'),simplifyVector=TRUE))
 x<-read_matrix(file.path(folder,'loadings.bin'),z$n,z$k);rownames(x)<-ids
 iv<-drop(read_matrix(file.path(folder,'specific.bin'),z$n,1));names(iv)<-ids
 cov_list[[d]]<-list(fct_load=x,fct_cov=read_matrix(file.path(folder,'factor.bin'),z$k,z$k),ivol_vec=iv)
 tmp<-data_tc[eom==as.Date(d)&valid==TRUE];v<-tmp$lambda_diagonal;names(v)<-as.character(tmp$id);lambda_list[[d]]<-v
}
validation<-fread(file.path(out,'r_validation.csv'));validation[,`:=`(eom=as.Date(eom),eom_ret=as.Date(eom_ret))]
# Restore original doubles by saved integer grid index, avoiding decimal equality failures.
validation[,l:=settings$pf_ml$l_vec[lambda_index+1L]]
selection<-fread(file.path(out,'r_annual_selection.csv'));selection[,eom_ret:=as.Date(eom_ret)]
nm<-pfml_feat_fun(512,FALSE);aim_by_g<-list()
dir.create(file.path(out,'sample_inputs'),showWarnings=FALSE)
for(gi in 0:1){
 blocks<-list()
 for(yr in 1981:2020){
  yr_dates<-dates[year(dates+1)==yr]
  samples<-sample_dates[year(sample_dates+1)==yr]
  prefix<-paste0('r_g',gi,'_')
  # Bounded year-sized calls preserve every selected month's 12-month lookback.
  # This is an independent source rebuild at predeclared dates, not full-period input replication.
  lower<-min(samples)+1-months(13)-1
  sub<-copy(data_tc[eom>=lower&eom<=max(samples)])
  inputs<-pfml_input_fun(sub,cov_list,lambda_list,gamma_rel=pf_set$gamma_rel,wealth=wealth,mu=pf_set$mu,
   dates=samples,lb=pf_set$lb_hor,scale=TRUE,risk_free=risk_free,features=features,rff_feat=TRUE,
   seed=settings$seed_no,p_max=max(settings$pf_ml$p_vec),g=settings$pf_ml$g_vec[gi+1],
   add_orig=FALSE,iter=10,balanced=FALSE)
  for(d in as.character(samples)){
   pre<-file.path(out,'sample_inputs',paste0(prefix,d,'_'))
   write_matrix(inputs$signal_t[[d]][,nm],paste0(pre,'signal.bin'))
   write_matrix(inputs$reals[[d]]$denom[nm,nm],paste0(pre,'denom.bin'))
   write_matrix(matrix(inputs$reals[[d]]$r_tilde[nm],nrow=1),paste0(pre,'rtilde.bin'))
  }
  rm(inputs,sub);gc()
  signals<-lapply(as.character(yr_dates),function(d){
   n<-nrow(data_tc[eom==as.Date(d)&valid==TRUE])
   x<-read_matrix(file.path(out,'signals',paste0('g',gi),paste0(d,'.bin')),n,513);colnames(x)<-nm;x
  });names(signals)<-as.character(yr_dates)
  coefs<-list();coefs[[as.character(yr)]]<-list()
  for(p in settings$pf_ml$p_vec){
   x<-read_matrix(file.path(out,'fit',paste0('g',gi),paste0('r_coef_',yr,'_p',p,'.bin')),p+1,length(settings$pf_ml$l_vec))
   block<-lapply(seq_along(settings$pf_ml$l_vec),function(j)matrix(x[,j],ncol=1));names(block)<-settings$pf_ml$l_vec
   coefs[[as.character(yr)]][[as.character(p)]]<-block
  }
  result<-pfml_aims_fun(list(signal_t=signals),validation=validation[g_index==gi],data_tc=data_tc,
   hp_coef=coefs,hp_years=yr,dates_oos=yr_dates,l_vec=settings$pf_ml$l_vec,orig_feat=FALSE)
  blocks[[as.character(yr)]]<-rbindlist(lapply(result,function(z)z$aim_pf))
  cat('Source inputs and annual aims:',gi,yr,'\n');flush.console()
  rm(signals,coefs,result);gc()
 }
 aim_by_g[[gi+1]]<-rbindlist(blocks)
}
aims<-rbindlist(lapply(as.character(dates),function(d){
 yr<-year(as.Date(d)+1);choice<-selection[year(eom_ret)==yr-1];stopifnot(nrow(choice)==1)
 answer<-copy(aim_by_g[[choice$g_index+1]][eom==as.Date(d)])
 answer[,`:=`(g_index=choice$g_index,p=choice$p,lambda_index=choice$lambda_index)];answer
}));fwrite(aims,file.path(out,'r_aims.csv'))
data<-data_tc[valid==TRUE&eom%in%dates,.(id,eom,me,ret_ld1,tr_ld1,lambda=lambda_diagonal,pred_ld1=0)]
rm(data_tc,aim_by_g,blocks,validation);gc()
cat('Executing complete 480-month original R holdings recursion.\n');flush.console()
w<-pfml_w(data,dates,cov_list,lambda_list,gamma_rel=pf_set$gamma_rel,iter=10,
 risk_free=risk_free,wealth=wealth,mu=pf_set$mu,aims=aims[,.(id,eom,w_aim)])
fwrite(w,file.path(out,'r_weights.csv'))
fwrite(pf_ts_fun(w,data,wealth,gam=pf_set$gamma_rel),file.path(out,'r_performance.csv'))
writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
write_json(list(status='ACTUAL_R_FULL_PATH_COMPLETE',months=length(dates),positions=nrow(w),
 sample_input_months=length(sample_dates),bandwidths=2,
 scope='Full original R fits/aims/holdings on identical R1 inputs; independent input rebuild sampled at 81 predeclared dates'),
 file.path(out,'path_status.json'),auto_unbox=TRUE,pretty=TRUE)
cat('Actual R full baseline path complete.\n')
