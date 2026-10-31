#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
project <- normalizePath(file.path(dirname(script), ".."))
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}
# Activate isolation before restore so matching global packages cannot hide an empty project library.
renv::activate(project = project)
renv::load(project = project, quiet = TRUE)
project_library <- renv::paths$library(project = project)
dir.create(project_library, recursive = TRUE, showWarnings = FALSE)
renv::restore(project = project, library = project_library,
              lockfile = file.path(project, "renv.lock"), prompt = FALSE)
lock <- renv::lockfile_read(file.path(project, "renv.lock"))
installed <- installed.packages(lib.loc = project_library)
recommended <- installed.packages(lib.loc = .libPaths(), priority = "recommended")
for (name in names(lock$Packages)) {
  expected <- lock$Packages[[name]]$Version
  available <- if (name %in% rownames(installed)) installed else recommended
  if (!name %in% rownames(available) || available[name, "Version"] != expected) {
    stop("Activated project does not resolve the locked version of ", name)
  }
}
