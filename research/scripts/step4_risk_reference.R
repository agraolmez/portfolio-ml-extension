root <- normalizePath(commandArgs(trailingOnly=TRUE)[1])
source(file.path(root,"baseline/reference_helpers.R"))
out <- file.path(root,"output/step4_20260911/reference")
cfg <- settings$cov_set
f <- fread(file.path(out,"factor_returns.csv")); f[,date:=as.Date(date)]
terms <- setdiff(names(f),"date")
u <- fread(file.path(out,"risk_universe.csv"));u[,eom:=as.Date(eom)]
dates <- sort(unique(u$eom))
wc <- (.5^(1/cfg$hl_cor))^(cfg$obs:1)
wv <- (.5^(1/cfg$hl_var))^(cfg$obs:1)
covs <- lapply(dates,function(d){
  block <- tail(f[date<=d],cfg$obs);x<-as.matrix(block[,..terms]);n<-nrow(x)
  correlation <- cov.wt(x,wt=tail(wc,n),center=TRUE,cor=TRUE,method="unbiased")$cor
  sd <- sqrt(diag(cov.wt(x,wt=tail(wv,n),center=TRUE,method="unbiased")$cov))
  (diag(sd)%*%correlation%*%diag(sd))*21
})
con<-file(file.path(out,"r_factor_covariances.bin"),"wb")
for(x in covs)writeBin(as.double(t(x)),con,size=8,endian="little")
close(con)
res <- fread(file.path(out,"residuals.csv"));res[,date:=as.Date(date)];setorder(res,id,date)
res[,res_vol:=ewma_c(residual,lambda=.5^(1/cfg$hl_stock_var),start=cfg$initial_var_obs),by=id]
calendar <- data.table(date=f$date,td_252d=dplyr::lag(f$date,252))
res <- calendar[res,on="date"]
res[,date_200d:=shift(date,200),by=id]
res <- res[date_200d>=td_252d & !is.na(res_vol)]
res[,eom:=ceiling_date(date,"month")-1]
res[,last_date:=max(date),by=.(id,eom)]
monthly <- res[date==last_date,.(id,eom,res_vol)]
answer <- monthly[u,on=.(id,eom)]
answer[,med_res_vol:=median(res_vol,na.rm=TRUE),by=.(size_grp,eom)]
answer[,med_all:=median(res_vol,na.rm=TRUE),by=eom]
answer[is.na(med_res_vol),med_res_vol:=med_all]
answer[,source_missing:=is.na(res_vol)]
answer[is.na(res_vol),res_vol:=med_res_vol]
answer[,specific_variance:=res_vol^2*21]
fwrite(answer[,.(id,eom,res_vol,specific_variance,source_missing)],file.path(out,"r_specific_risk.csv"))
# Actual author MASS random-feature draws with the prescribed seed and p_max.
for(i in seq_along(settings$pf_ml$g_vec)){
  set.seed(settings$seed_no)
  draw<-rff(matrix(0,nrow=2,ncol=length(features)),p=max(settings$pf_ml$p_vec),g=settings$pf_ml$g_vec[i])$W
  write_matrix(draw,file.path(out,paste0("rff_g",i-1,".bin")))
}
writeLines(capture.output(sessionInfo()),file.path(out,"R_session.txt"))
write_json(list(covariance_months=length(dates),specific_stock_months=nrow(answer),specific_missing=sum(answer$source_missing),status="SOURCE_R_EXECUTED"),file.path(out,"risk_reference_status.json"),auto_unbox=TRUE,pretty=TRUE)
cat("Actual R risk construction and random-feature draws complete.\n")
