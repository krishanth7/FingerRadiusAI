# conformance_test.R - Check the R port matches Python exactly.
#
# Run:  Rscript ports/r/conformance_test.R
#
# jsonlite is the only dependency and it is only needed to read the fixture;
# the port itself has none.

suppressWarnings(suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    cat("R: SKIP - jsonlite is not installed (install.packages('jsonlite'))\n")
    quit(status = 0)
  }
  library(jsonlite)
}))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "ports/r"
source(file.path(here, "fingerradius.R"))

fixture_path <- file.path(here, "..", "fixtures", "conformance.json")
fixture <- fromJSON(fixture_path, simplifyVector = FALSE)

TOL <- 1e-6
failures <- character(0)
checks <- 0L

for (item in fixture$gestures) {
  landmarks <- matrix(unlist(item$landmarks), ncol = 2, byrow = TRUE)
  expected <- item$expected
  actual <- fr_classify(landmarks)
  checks <- checks + 1L

  if (!identical(actual$gesture, expected$gesture)) {
    failures <- c(failures, sprintf("%s: gesture %s != %s",
                                    item$id, actual$gesture, expected$gesture))
  }
  if (actual$extended_count != expected$extended_count) {
    failures <- c(failures, sprintf("%s: extended_count %d != %d",
                                    item$id, actual$extended_count,
                                    expected$extended_count))
  }
  for (finger in names(expected$fingers)) {
    want <- isTRUE(expected$fingers[[finger]])
    got <- isTRUE(actual$fingers[[finger]])
    if (got != want) {
      failures <- c(failures, sprintf("%s: finger %s %s != %s",
                                      item$id, finger, got, want))
    }
  }

  radii <- fr_radii(landmarks)
  for (pair in names(expected$radii)) {
    checks <- checks + 1L
    delta <- abs(radii[[pair]] - expected$radii[[pair]])
    if (delta > TOL) {
      failures <- c(failures, sprintf("%s: radius %s delta %.3e",
                                      item$id, pair, delta))
    }
  }
}

kal <- fixture$kalman
filter <- fr_kalman(process_noise = kal$process_noise,
                    measurement_noise = kal$measurement_noise)
worst <- 0
for (i in seq_along(kal$input)) {
  got <- filter$update(kal$input[[i]])
  want <- kal$expected[[i]]
  checks <- checks + 1L
  delta <- abs(got - want)
  worst <- max(worst, delta)
  if (delta > TOL) {
    failures <- c(failures, sprintf("kalman[%d]: delta %.3e", i, delta))
  }
}

if (length(failures) == 0L) {
  cat(sprintf("r: PASS - %d checks against the Python fixture (Kalman worst %.3e)\n",
              checks, worst))
  quit(status = 0)
} else {
  cat(sprintf("r: FAIL - %d of %d checks\n", length(failures), checks))
  for (f in head(failures, 10)) cat("  ", f, "\n")
  quit(status = 1)
}
