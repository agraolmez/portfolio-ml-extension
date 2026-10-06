root<-normalizePath(commandArgs(trailingOnly=TRUE)[1]);source(file.path(root,'baseline/reference_helpers.R'))
load_functions(file.path(author,'1 - Prepare Data.R'),'wealth_func')
out<-file.path(root,'output/step4_20260911/reference');x<-fread(file.path(out,'wealth_inputs.csv'));x[,eom:=as.Date(eom)]
# Both variants get identical audited total returns. Isolate the recurrence difference,
# without introducing the original French/CRSP subtraction-rate difference as a second change.
risk_free<-x[,.(eom,rf=rf_crsp_t30)];market<-x[,.(eom_ret=eom,mkt_vw_exc=published_excess)]
w<-wealth_func(1e10,as.Date('2020-12-31'),market,risk_free)
fwrite(w,file.path(out,'r_uncorrected_wealth.csv'))
