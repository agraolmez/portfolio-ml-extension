# Independent current-state features and full portfolio recursions in R.
root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);.libPaths(c(file.path(root,'env/step2/R-library'),.libPaths()))
library(jsonlite);out<-file.path(root,'output/step9_20260913/path_reference');stopifnot(!file.exists(file.path(out,'summary.json')))
metadata<-fromJSON(file.path(out,'metadata.json'));variants<-c('primary','horizon36','blocks24');choices<-list();all_errors<-list();all_monthly<-list()
readmat<-function(path,n,p){con<-file(path,'rb');on.exit(close(con));matrix(readBin(con,numeric(),n=n*p,size=8,endian='little'),nrow=n,byrow=TRUE)}
test<-function(name,a,b,atol=1e-12,rtol=1e-9){
 stopifnot(length(a)==length(b),all(is.finite(a)),all(is.finite(b)))
 err<-max(abs(a-b));pass<-all(abs(a-b)<=atol+rtol*abs(b))
 all_errors[[length(all_errors)+1]]<<-data.frame(check=name,max_abs_error=err,passed=pass)
 if(!pass)stop(paste('Comparison failed',name,err))
}
predict<-function(model,x){
 z<-(x-unlist(model$mean))/unlist(model$scale);z[unlist(model$constant)]<-0
 drop(z%*%do.call(rbind,lapply(model$coefficients,unlist))+unlist(model$intercept))
}
utility<-function(x)12*(mean(x)-5*mean((x-mean(x))^2))
select_controls<-function(df){
 scores<-tapply(df$net_excess,df$method,utility);speeds<-c(.5,.75,1,1.25,1.5);rhos<-c(.25,.5,.75)
 best<-max(scores[paste0('B1_',speeds)]);eligible<-speeds[scores[paste0('B1_',speeds)]>=best-1e-8];speed<-eligible[order(abs(eligible-1),eligible)][1]
 best<-max(scores[paste0('B3_',rhos)]);rho<-min(rhos[scores[paste0('B3_',rhos)]>=best-1e-8])
 candidates<-c('R1',paste0('B1_',speed),'B2',paste0('B3_',rho));best<-max(scores[candidates]);chosen<-candidates[scores[candidates]>=best-1e-8][1]
 list(B1_kappa=unname(speed),B3_rho=unname(rho),C_star=strsplit(chosen,'_',fixed=TRUE)[[1]][1])
}
for(period in c('selection','evaluation','robustness')){
 rows<-metadata[metadata$period==period,];spec<-fromJSON(file.path(out,paste0(period,'_columns.json')));ns<-nrow(spec);holdings<-NULL;stored<-vector('list',ns);prev_ids<-prev_growth<-NULL;monthly<-list();fits<-list()
 # For later blocks independently validate the Python specification against R's selection.
 if(period!='selection')for(variant in variants){
  stopifnot(spec$parameter[spec$variant==variant&spec$policy=='B1']==choices[[variant]]$B1_kappa,spec$parameter[spec$variant==variant&spec$policy=='B3']==choices[[variant]]$B3_rho)
 }
 for(t in seq_len(nrow(rows))){
  info<-rows[t,];folder<-file.path(out,'months',info$date);d<-read.csv(file.path(folder,'current.csv'));ids<-d$id;n<-length(ids);m<-readmat(file.path(folder,'m.bin'),n,n)
  loadings<-readmat(file.path(folder,'loadings.bin'),n,info$k);fc<-readmat(file.path(folder,'factor.bin'),info$k,info$k);specific<-drop(readmat(file.path(folder,'specific.bin'),n,1));wealth<-info$wealth
  risk<-function(x){if(is.null(dim(x)))drop(loadings%*%(fc%*%(t(loadings)%*%x)))+specific*x else loadings%*%(fc%*%(t(loadings)%*%x))+specific*x}
  trans<-m*sqrt(d$impact)/rep(sqrt(d$impact),each=n);ev<-eigen(trans,symmetric=TRUE,only.values=TRUE)$values;cap<-min(1.5,1/(1-min(ev)))
  stopifnot(min(ev)>=0,max(ev)<=1);test(paste(info$date,'matrix cap'),cap,info$kappa_max)
  means<-list();numerators<-list()
  for(block in c(12,24)){
   replicas<-readmat(file.path(folder,paste0('replicas',block,'.bin')),n,100);center<-rowMeans(replicas);centered<-replicas-center;trades<-centered-m%*%centered
   numerator<-sum(trades*risk(trades))/100;test(paste(info$date,'direct replica variance',block),numerator,info[[paste0('numerator',block)]],atol=1e-10,rtol=1e-8)
   means[[as.character(block)]]<-center;numerators[[as.character(block)]]<-numerator
  }
  if(is.null(holdings))holdings<-matrix(rep(drop(readmat(file.path(out,paste0(period,'_initial.bin')),n,1)),ns),n,ns)
  targets<-trades<-weights<-matrix(0,n,ns)
  for(j in seq_len(ns)){
   s<-spec[j,];h<-holdings[,j];target<-d$aim;kappa<-1;block<-if(s$variant=='blocks24')'24' else '12';b<-craw<-chat<-u<-NA_real_;outofrange<-FALSE
   features<-rep(NA_real_,6)
   if(s$policy=='B1')kappa<-max(.5,min(cap,s$parameter))
   if(s$policy=='B2')target<-means[[block]]
   if(s$policy=='B3'){
    if(is.null(stored[[j]]))target<-d$aim else {
     ix<-match(ids,prev_ids);mapped<-d$aim;survivor<-!is.na(ix);mapped[survivor]<-(stored[[j]]*prev_growth)[ix[survivor]];target<-(1-s$parameter)*d$aim+s$parameter*mapped
    };stored[[j]]<-target
   }
   if(s$policy%in%c('A0','A1','A2')){
    q<-drop((d$aim-h)-m%*%(d$aim-h));qbar<-drop((means[[block]]-h)-m%*%(means[[block]]-h));qr<-sum(q*risk(q));hr<-sum(h*risk(h));ir<-wealth*sum(d$impact*q*q);den<-sum(qbar*risk(qbar))
    align<-if(qr<=1e-12||hr<=1e-12)0 else sum(q*risk(h))/sqrt(qr*hr);ld<-log1p(numerators[[block]]/max(den,1e-12));features<-c(log(max(qr,1e-12)),log(max(ir,1e-12)),log(max(hr,1e-12)),align,ld,align*ld)
    year<-as.integer(substr(info$date,1,4))+(substr(info$date,6,7)=='12');key<-paste(s$variant,year)
    if(is.null(fits[[key]]))fits[[key]]<-fromJSON(file.path(root,'data/prepared/step9_v1/learner/fits',s$variant,paste0(year,'.json')),simplifyVector=FALSE)
    fit<-fits[[key]];stopifnot(fit$status=='FITTED',info$date>=fit$first_decision,info$date<=fit$last_decision,10*qr+ir>1e-12)
    model<-fit$models[[if(s$policy=='A0')'A0' else 'A1']];xx<-features[seq_along(model$mean)];bc<-predict(model,xx);b<-bc[1];craw<-bc[2];chat<-max(1,craw);u<-0
    if(s$policy=='A2'){stopifnot(length(fit$bootstrap)==100,fit$uncertainty_status=='FITTED');u<-sd(sapply(fit$bootstrap,function(replica)predict(replica,xx)[1]))}
    numerator<-if(s$policy=='A2')sign(b)*max(abs(b)-u,0) else b;kappa<-1+max(-.5,min(cap-1,numerator/(chat+1)))
    outofrange<-any(xx<unlist(model$minimum)|xx>unlist(model$maximum))
   }
   q<-drop(kappa*((target-h)-m%*%(target-h)));w<-h+q;targets[,j]<-target;trades[,j]<-q;weights[,j]<-w
   dollars<-wealth*q;cost<-sum(d$impact*dollars^2)/2/wealth;gross<-sum(w*d$returns)
   monthly[[length(monthly)+1]]<-data.frame(period=period,variant=s$variant,method=s$method,decision_eom=info$date,gross_excess=gross,impact_cost=cost,net_excess=gross-cost,turnover=sum(abs(q)),gross_exposure=sum(abs(w)),net_exposure=sum(w),modeled_annual_volatility=sqrt(12*sum(w*risk(w))),max_absolute_weight=max(abs(w)),absolute_weight_hhi=sum(w^2)/sum(abs(w))^2,kappa=kappa,b_hat=b,c_raw=craw,c_hat=chat,u=u,out_of_range=outofrange,log_trade_risk=features[1],log_trade_impact=features[2],log_holdings_risk=features[3],risk_alignment=features[4],log1p_aim_dispersion=features[5],alignment_times_log1p_dispersion=features[6])
  }
  drift<-weights*d$growth
  for(name in c('pretrade','target','trade','weight','drifted_weight')){
   actual<-switch(name,pretrade=holdings,target=targets,trade=trades,weight=weights,drifted_weight=drift)
   test(paste(period,info$date,name),actual,readmat(file.path(folder,paste0('python_',name,'.bin')),n,ns),atol=1e-8,rtol=1e-6)
  }
  nextids<-fromJSON(file.path(folder,'next_ids.json'));ix<-match(nextids,ids);holdings<-matrix(0,length(nextids),ns);survivor<-!is.na(ix);holdings[survivor,]<-drift[ix[survivor],,drop=FALSE]
  prev_ids<-ids;prev_growth<-d$growth
  if(t%%24==1){cat('Independent full path:',period,t,'/',nrow(rows),'\n');flush.console()}
 }
 result<-do.call(rbind,monthly);write.csv(result,file.path(out,paste0('r_',period,'_monthly.csv')),row.names=FALSE)
 python<-read.csv(file.path(out,paste0('python_',period,'_monthly.csv')));keys<-c('period','variant','method','decision_eom');combined<-merge(result,python,by=keys,suffixes=c('_r','_py'));stopifnot(nrow(combined)==nrow(result))
 for(key in setdiff(names(result),keys)){
  a<-combined[[paste0(key,'_r')]];b<-combined[[paste0(key,'_py')]];if(key=='out_of_range'&&is.character(b))b<-tolower(b)=='true';ok<-!is.na(a)&!is.na(b);stopifnot(all(is.na(a)==is.na(b)))
  test(paste(period,'all monthly',key),as.numeric(a[ok]),as.numeric(b[ok]),atol=1e-10,rtol=1e-8)
 }
 if(period=='selection'){
  for(variant in variants)choices[[variant]]<-select_controls(result[result$variant==variant,])
  write_json(choices,file.path(out,'r_frozen_selection.json'),auto_unbox=TRUE,pretty=TRUE,digits=17)
  py<-fromJSON(file.path(root,'output/step9_20260913/frozen_selection.json'),simplifyVector=FALSE)$choices
  for(variant in variants)stopifnot(choices[[variant]]$B1_kappa==py[[variant]]$B1_kappa,choices[[variant]]$B3_rho==py[[variant]]$B3_rho,choices[[variant]]$C_star==py[[variant]]$C_star)
 }
 all_monthly[[period]]<-result
}
errors<-do.call(rbind,all_errors);write.csv(errors,file.path(out,'comparison_errors.csv'),row.names=FALSE)
write_json(list(status='PASS',periods=3,variants=3,paths=c(selection=39,evaluation=21,robustness=21),monthly_records=sum(sapply(all_monthly,nrow)),checks=nrow(errors),max_abs_error=max(errors$max_abs_error),independent_selection=TRUE,complete_own_holdings_recursion=TRUE),file.path(out,'summary.json'),auto_unbox=TRUE,pretty=TRUE,digits=17)
capture.output(sessionInfo(),file=file.path(out,'R_session.txt'))
