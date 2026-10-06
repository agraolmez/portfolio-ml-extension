root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'))
out<-file.path(root,'output/step9_20260913/extension_reference');nfeat<-513
stopifnot(!file.exists(file.path(out,'summary.json')))
models<-fromJSON(file.path(out,'conditional_models.json'));all_validation<-list();replica_checks<-list()
for(gi in 0:1){
 folder<-file.path(out,'fit',paste0('g',gi));nm<-pfml_feat_fun(512,FALSE);reals<-list()
 for(src in c(file.path(root,'output/step6_20260911/reference/fit',paste0('g',gi)),folder)){
  dates<-read_json(file.path(src,'dates.json'),simplifyVector=TRUE)
  a<-file(file.path(src,'denom.bin'),'rb');r<-file(file.path(src,'rtilde.bin'),'rb')
  for(d in dates){
   ma<-matrix(readBin(a,numeric(),n=nfeat*nfeat,size=8,endian='little'),nrow=nfeat,byrow=TRUE);dimnames(ma)<-list(nm,nm)
   vr<-readBin(r,numeric(),n=nfeat,size=8,endian='little');names(vr)<-nm;reals[[d]]<-list(r_tilde=vr,denom=ma)
  };close(a);close(r)
 }
 input<-list(reals=reals);years<-2021:2025
 coefs<-pfml_search_coef(input,settings$pf_ml$p_vec,settings$pf_ml$l_vec,years,FALSE)
 for(yr in years)for(p in settings$pf_ml$p_vec)write_matrix(do.call(cbind,coefs[[as.character(yr)]][[as.character(p)]]),file.path(folder,paste0('r_coef_',yr,'_p',p,'.bin')))
 v<-pfml_hp_reals_fun(input,coefs,settings$pf_ml$p_vec,settings$pf_ml$l_vec,years,FALSE);v[,g_index:=gi];v[,lambda_index:=match(l,settings$pf_ml$l_vec)-1L];all_validation[[gi+1]]<-v
 for(j in which(models$g_index==gi)){
  row<-models[j,];n<-row$training_months;length<-row$block_months;ref<-file.path(out,'conditional',paste0('block',length))
  counts<-as.matrix(read.csv(file.path(ref,paste0(row$fit_year,'_counts.csv')),header=FALSE));python<-as.matrix(read.csv(file.path(ref,paste0(row$fit_year,'_coefficients.csv')),header=FALSE))
  idx<-fromJSON(file.path(ref,paste0(row$fit_year,'_indices.json')))+1L
  stopifnot(names(reals)[n]==row$training_cutoff,all(rowSums(counts)==n))
  for(b in 1:3){
   aa<-matrix(0,length(idx),length(idx));rr<-numeric(length(idx))
   for(t in which(counts[b,]>0)){aa<-aa+counts[b,t]*reals[[t]]$denom[idx,idx];rr<-rr+counts[b,t]*reals[[t]]$r_tilde[idx]}
   penalty<-settings$pf_ml$l_vec[row$lambda_index+1L];coef<-drop(solve(aa/n+penalty*diag(length(idx)),rr/n));err<-max(abs(coef-python[,b]))
   stopifnot(all(abs(coef-python[,b])<=1e-12+1e-9*abs(python[,b])))
   replica_checks[[length(replica_checks)+1]]<-data.frame(year=row$fit_year,block=length,replica=c(1,50,100)[b],max_abs_error=err)
  }
 };rm(input,reals,coefs);gc();cat('Source grid and conditional checks:',gi,'\n');flush.console()
}
newv<-rbindlist(all_validation);fwrite(newv,file.path(out,'r_new_validation.csv'))
old<-fread(file.path(root,'output/step6_20260911/reference/r_validation.csv'));v<-rbindlist(list(old,newv),use.names=TRUE,fill=TRUE)
setorder(v,g_index,p,lambda_index,eom_ret);v[,cum_obj:=cumsum(obj)/seq_len(.N),by=.(g_index,p,lambda_index)];v[,rank:=frank(-cum_obj),by=eom_ret]
chosen<-v[month(as.Date(eom_ret))==12&rank==1];stopifnot(nrow(chosen)==55);fwrite(chosen,file.path(out,'r_annual_selection.csv'))
fwrite(rbindlist(replica_checks),file.path(out,'conditional_checks.csv'))
meta<-read_json(file.path(out,'metadata.json'),simplifyVector=TRUE);sample_dates<-as.Date(meta$sample_input_dates)
data_tc<-fread(file.path(out,'data.csv'));data_tc[,eom:=as.Date(eom)];data_tc[,valid:=as.logical(valid)];setorder(data_tc,eom,id)
risk_free<-fread(file.path(out,'rates.csv'))[,.(eom=as.Date(eom),rf=rf_french)];wealth<-unique(data_tc[,.(eom,wealth=wealth_usd,mu_ld0,mu_ld1)])
cov_list<-list();lambda_list<-list()
for(j in seq_len(nrow(meta$risk))){
 z<-meta$risk[j,];d<-z$date;folder<-file.path(out,'risk',d);ids<-as.character(read_json(file.path(folder,'ids.json'),simplifyVector=TRUE))
 x<-read_matrix(file.path(folder,'loadings.bin'),z$n,z$k);rownames(x)<-ids;iv<-drop(read_matrix(file.path(folder,'specific.bin'),z$n,1));names(iv)<-ids
 cov_list[[d]]<-list(fct_load=x,fct_cov=read_matrix(file.path(folder,'factor.bin'),z$k,z$k),ivol_vec=iv)
 tmp<-data_tc[eom==as.Date(d)&valid==TRUE];v<-tmp$lambda_diagonal;names(v)<-as.character(tmp$id);lambda_list[[d]]<-v
}
dir.create(file.path(out,'sample_inputs'))
for(gi in 0:1)for(yr in 2021:2025){
 samples<-sample_dates[year(sample_dates+1)==yr];lower<-min(samples)+1-months(13)-1;sub<-copy(data_tc[eom>=lower&eom<=max(samples)])
 inputs<-pfml_input_fun(sub,cov_list,lambda_list,gamma_rel=pf_set$gamma_rel,wealth=wealth,mu=pf_set$mu,dates=samples,lb=pf_set$lb_hor,scale=TRUE,risk_free=risk_free,features=features,rff_feat=TRUE,seed=settings$seed_no,p_max=max(settings$pf_ml$p_vec),g=settings$pf_ml$g_vec[gi+1],add_orig=FALSE,iter=10,balanced=FALSE)
 for(d in as.character(samples)){
  pre<-file.path(out,'sample_inputs',paste0('r_g',gi,'_',d,'_'));write_matrix(inputs$signal_t[[d]][,nm],paste0(pre,'signal.bin'));write_matrix(inputs$reals[[d]]$denom[nm,nm],paste0(pre,'denom.bin'));write_matrix(matrix(inputs$reals[[d]]$r_tilde[nm],nrow=1),paste0(pre,'rtilde.bin'))
 };rm(inputs,sub);gc();cat('Source moments:',gi,yr,'\n');flush.console()
}
sys.source(file.path(root,'scripts/step9_speed_reference.R'),envir=new.env(parent=globalenv()))
write_json(list(status='PASS',base_coefficient_fits=4040,new_validation_rows=nrow(newv),annual_selections=nrow(chosen),conditional_refits=length(replica_checks),moment_dates=11,bandwidths=2),file.path(out,'summary.json'),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,'R_session.txt'))
