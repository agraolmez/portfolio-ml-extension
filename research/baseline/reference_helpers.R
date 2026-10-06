# Load complete pinned function definitions, without running the author driver.
.libPaths(c(file.path(root,"env/step2/R-library"),.libPaths()))
Sys.setenv(R_MAKEVARS_USER=file.path(root,"env/step2/Makevars"))
suppressPackageStartupMessages({library(data.table);library(dplyr);library(lubridate);library(purrr);library(expm);library(jsonlite)})
setDTthreads(2)
load_functions <- function(path,names) {
  for(e in parse(path)) {
    if(is.call(e)&&identical(e[[1]],as.name("<-"))&&is.symbol(e[[2]])&&as.character(e[[2]])%in%names) eval(e,envir=.GlobalEnv)
  }
}
author <- file.path(root,"references/author_code")
load_functions(file.path(author,"0 - General functions.R"),c("create_cov","create_lambda","initial_weights_new","pf_ts_fun"))
load_functions(file.path(author,"0 - Return prediction functions.R"),"rff")
load_functions(file.path(author,"0 - Portfolio choice functions.R"),c("m_func","pfml_input_fun","pfml_feat_fun","denom_sum_fun","pfml_search_coef","pfml_hp_reals_fun","pfml_aims_fun","pfml_w"))
Rcpp::sourceCpp(file.path(root,"output/step2_20260910/author_support/ewma.cpp"))
Rcpp::sourceCpp(file.path(root,"output/step2_20260910/author_support/sqrtm_cpp.cpp"))
spec <- read_json(file.path(root,"output/step2_20260910/source_settings.json"),simplifyVector=TRUE)
features <- spec$features; settings <- spec$settings; pf_set <- spec$pf_set
write_matrix <- function(a,path) writeBin(as.double(t(as.matrix(a))),path,size=8,endian="little")
read_matrix <- function(path,n,p) matrix(readBin(path,numeric(),n=n*p,size=8,endian="little"),nrow=n,byrow=TRUE)
