# Run the packaged analysis program from a source checkout.
script_arg <- grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[1])))
source(file.path(script_dir, "../../src/pace_abeta/analysis/memory_analysis.R"), chdir=FALSE)
