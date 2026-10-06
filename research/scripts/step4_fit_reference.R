root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'))
out<-file.path(root,'output/step4_20260911/reference');nfeat<-513
all_validation<-list()
for(gi in 0:1){
 folder<-file.path(out,'fit',paste0('g',gi));dates<-read_json(file.path(folder,'dates.json'),simplifyVector=TRUE)
 nm<-pfml_feat_fun(512,FALSE);a<-file(file.path(folder,'denom.bin'),'rb');r<-file(file.path(folder,'rtilde.bin'),'rb')
 reals<-lapply(dates,function(d){
  matrix_a<-matrix(readBin(a,numeric(),n=nfeat*nfeat,size=8,endian='little'),nrow=nfeat,byrow=TRUE);dimnames(matrix_a)<-list(nm,nm)
  vector_r<-readBin(r,numeric(),n=nfeat,size=8,endian='little');names(vector_r)<-nm
  list(r_tilde=vector_r,denom=matrix_a)
 });close(a);close(r);names(reals)<-dates
 input<-list(reals=reals);years<-1971:1982
 cat('Actual R coefficient path for g',gi,'\n');flush.console()
 coefs<-pfml_search_coef(input,settings$pf_ml$p_vec,settings$pf_ml$l_vec,years,FALSE)
 for(year in years)for(p in settings$pf_ml$p_vec)write_matrix(do.call(cbind,coefs[[as.character(year)]][[as.character(p)]]),file.path(folder,paste0('r_coef_',year,'_p',p,'.bin')))
 v<-pfml_hp_reals_fun(input,coefs,settings$pf_ml$p_vec,settings$pf_ml$l_vec,years,FALSE);v[,g_index:=gi]
 v[,lambda_index:=match(l,settings$pf_ml$l_vec)-1L];all_validation[[gi+1]]<-v
 rm(input,reals,coefs);gc()
}
v<-rbindlist(all_validation);fwrite(v,file.path(out,'r_validation.csv'))
v[,rank:=frank(-cum_obj),by=eom_ret]
fwrite(v[month(eom_ret)==12&rank==1],file.path(out,'r_annual_selection.csv'))
cat('Actual R full-grid annual fits and selection complete.\n')
