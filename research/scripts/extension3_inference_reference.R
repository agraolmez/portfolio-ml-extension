# Independent R verification of supplied Extension-3 inference artifacts.
# Uses no research database and fits no model. All case settings are explicit.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop('Provide exactly one new verification input folder')
folder <- normalizePath(args[1], mustWork=TRUE)
receipt_path <- file.path(folder, 'R_verification.json')
if (file.exists(receipt_path)) stop('Verification receipt already exists; do not overwrite')
safe_file <- function(name) {
  if (length(name) != 1L || is.na(name) || !grepl('^[A-Za-z0-9_.-]+$', name)
      || name %in% c('.', '..')) stop('Invalid verification input filename')
  path <- file.path(folder, name)
  if (!file.exists(path)) stop(paste('Missing verification input', name))
  path
}
cases <- read.csv(safe_file('verification_cases.csv'), check.names=FALSE, stringsAsFactors=FALSE)
contrasts <- read.csv(safe_file('contrasts.csv'), check.names=FALSE, stringsAsFactors=FALSE)
if (nrow(cases) < 1L || anyDuplicated(cases$case)) stop('Distinct verification cases required')
if (nrow(contrasts) != 4L || anyDuplicated(contrasts) || any(contrasts$left == contrasts$right)) {
  stop('Four distinct paired contrasts are required')
}
utility <- function(z) {
  mu <- colMeans(z)
  variance <- colMeans(sweep(z, 2, mu, '-')^2)
  12 * (mu - 5 * variance)
}
utility_error <- 0.; interval_error <- 0.; utilities_checked <- 0L; intervals_checked <- 0L
for (c in seq_len(nrow(cases))) {
  spec <- cases[c, ]
  b <- as.integer(spec$replicas); n <- as.integer(spec$months)
  if (b < 1L || n < 2L || b != spec$replicas || n != spec$months
      || spec$family != 4L || !is.finite(spec$alpha) || spec$alpha <= 0 || spec$alpha >= 1) {
    stop('Invalid explicit inference case settings')
  }
  x <- as.matrix(read.csv(safe_file(spec$returns_csv), header=FALSE, check.names=FALSE))
  expected <- as.matrix(read.csv(safe_file(spec$utilities_csv), header=FALSE, check.names=FALSE))
  columns <- read.csv(safe_file(spec$columns_csv), check.names=FALSE, stringsAsFactors=FALSE)
  intervals <- read.csv(safe_file(spec$intervals_csv), check.names=FALSE, stringsAsFactors=FALSE)
  if (nrow(x) != n || nrow(expected) != b || ncol(expected) != ncol(x)
      || !all(is.finite(x)) || !all(is.finite(expected))
      || nrow(columns) != ncol(x) || any(columns$position != seq_len(ncol(x)))
      || anyDuplicated(columns[c('seed', 'method')])
      || !identical(sort(unique(columns$seed)), 1:3)) stop('Misaligned numerical matrices or column identities')
  for (method in unique(columns$method)) {
    if (!identical(sort(columns$seed[columns$method == method]), 1:3)) stop('Each method needs three separate seeds')
  }
  path <- safe_file(spec$indices_bin)
  if (file.info(path)$size != 4 * b * n) stop('Incomplete or overlong saved bootstrap index file')
  raw <- readBin(path, integer(), n=b*n, size=4, endian='little')
  if (length(raw) != b*n || anyNA(raw) || any(raw < 0L) || any(raw >= n)) stop('Bootstrap index out of range')
  indices <- matrix(raw, nrow=b, byrow=TRUE) + 1L
  actual <- matrix(NA_real_, nrow=b, ncol=ncol(x))
  for (replica in seq_len(b)) actual[replica, ] <- utility(x[indices[replica, ], , drop=FALSE])
  error <- max(abs(actual-expected))
  if (!is.finite(error) || error > 1e-11) stop('Independent R bootstrap utility mismatch')
  utility_error <- max(utility_error, error)
  utilities_checked <- utilities_checked + length(actual)
  point <- utility(x)
  if (nrow(intervals) != 4L * 4L) stop('Every contrast needs mean-seed and three seed diagnostic intervals')
  for (pair in seq_len(nrow(contrasts))) {
    left <- contrasts$left[pair]; right <- contrasts$right[pair]
    ia <- vapply(1:3, function(seed) which(columns$seed == seed & columns$method == left), integer(1))
    ib <- vapply(1:3, function(seed) which(columns$seed == seed & columns$method == right), integer(1))
    seed_points <- point[ia] - point[ib]
    seed_draws <- actual[, ia, drop=FALSE] - actual[, ib, drop=FALSE]
    for (aggregation in 0:3) {
      if (aggregation == 0L) {
        observed <- mean(seed_points); draws <- rowMeans(seed_draws)
        matched <- intervals$left == left & intervals$right == right & intervals$aggregation == 'mean_seed' & is.na(intervals$seed)
      } else {
        observed <- seed_points[aggregation]; draws <- seed_draws[, aggregation]
        matched <- intervals$left == left & intervals$right == right & intervals$aggregation == 'individual_seed' & !is.na(intervals$seed) & intervals$seed == aggregation
      }
      row <- intervals[which(matched), ]
      if (nrow(row) != 1L || row$family != 4L || row$replicas != b
          || abs(row$confidence - (1-spec$alpha)) > 1e-15) stop('Missing or misconfigured unique reported interval')
      adjusted_q <- quantile(draws, probs=c(spec$alpha/(2*4), 1-spec$alpha/(2*4)), type=7, names=FALSE)
      ordinary_q <- quantile(draws, probs=c(spec$alpha/2, 1-spec$alpha/2), type=7, names=FALSE)
      adjusted <- 2*observed - rev(adjusted_q)
      ordinary <- 2*observed - rev(ordinary_q)
      actual_row <- c(observed, adjusted, ordinary)
      expected_row <- c(row$estimate, row$adjusted_lower, row$adjusted_upper, row$ordinary_lower, row$ordinary_upper)
      error <- max(abs(actual_row-expected_row))
      if (!is.finite(error) || error > 1e-11) stop('Independent R paired contrast or basic interval mismatch')
      interval_error <- max(interval_error, error)
      intervals_checked <- intervals_checked + 1L
    }
  }
}
receipt <- sprintf(paste0('{"status":"PASS","cases_checked":%d,',
  '"bootstrap_utilities_checked":%d,"maximum_utility_error":%.17g,',
  '"contrast_intervals_checked":%d,"maximum_interval_error":%.17g,',
  '"comparison_family":4,"model_seeds_resampled":false,"historical_fit_run":false,',
  '"originality_or_performance_claim":false,',
  '"calculation":"Independent R population variance, paired seed contrasts, seed-average statistics, type-7 basic intervals"}\n'),
  nrow(cases), utilities_checked, utility_error, intervals_checked, interval_error)
connection <- file(receipt_path, open='wx')
writeLines(receipt, connection, sep='')
close(connection)
cat('Independent R verification passed:', utilities_checked, 'utilities and', intervals_checked, 'intervals\n')
