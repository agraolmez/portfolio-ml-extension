root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'))
out<-file.path(root,'output/step6_20260911/reference');nfeat<-513
stopifnot(!file.exists(file.path(out,'fit_status.json')))
all_validation<-list()
for(gi in 0:1){
 folder<-file.path(out,'fit',paste0('g',gi));dates<-read_json(file.path(folder,'dates.json'),simplifyVector=TRUE)
 nm<-pfml_feat_fun(512,FALSE);a<-file(file.path(folder,'denom.bin'),'rb');r<-file(file.path(folder,'rtilde.bin'),'rb')
 reals<-lapply(dates,function(d){
  ma<-matrix(readBin(a,numeric(),n=nfeat*nfeat,size=8,endian='little'),nrow=nfeat,byrow=TRUE);dimnames(ma)<-list(nm,nm)
  vr<-readBin(r,numeric(),n=nfeat,size=8,endian='little');names(vr)<-nm
  list(r_tilde=vr,denom=ma)
 });close(a);close(r);names(reals)<-dates
 input<-list(reals=reals);years<-1971:2020
 cat('Actual R source coefficients, bandwidth',gi,'all 50 fitting years.\n');flush.console()
 coefs<-pfml_search_coef(input,settings$pf_ml$p_vec,settings$pf_ml$l_vec,years,FALSE)
 for(yr in years)for(p in settings$pf_ml$p_vec)write_matrix(do.call(cbind,coefs[[as.character(yr)]][[as.character(p)]]),file.path(folder,paste0('r_coef_',yr,'_p',p,'.bin')))
 cat('Actual R source validation, bandwidth',gi,'all 600 validation return months.\n');flush.console()
 v<-pfml_hp_reals_fun(input,coefs,settings$pf_ml$p_vec,settings$pf_ml$l_vec,years,FALSE);v[,g_index:=gi]
 v[,lambda_index:=match(l,settings$pf_ml$l_vec)-1L];all_validation[[gi+1]]<-v
 rm(input,reals,coefs);gc()
}
v<-rbindlist(all_validation);fwrite(v,file.path(out,'r_validation.csv'))
v[,rank:=frank(-cum_obj),by=eom_ret]
chosen<-v[month(eom_ret)==12&rank==1]
stopifnot(nrow(chosen)==50)
fwrite(chosen,file.path(out,'r_annual_selection.csv'))
write_json(list(status='ACTUAL_R_FULL_GRID_COMPLETE',first_year=1971,last_year=2020,
 coefficient_fits=50*2*4*101,validation_rows=nrow(v),selected_years=nrow(chosen),
 input_scope='Identical Python-prepared R1 moments; moment-source agreement separately sampled'),
 file.path(out,'fit_status.json'),auto_unbox=TRUE,pretty=TRUE)
cat('Actual R full-grid annual fits complete.\n')
