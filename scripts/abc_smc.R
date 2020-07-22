library(reticulate)
use_condaenv("wildcats_summer_env")
sim <- import("sim.model")
param_vec = c(1, 100, 100, 100, 40,0.1,20,0.1,400, 400, 30000, 0.1, 5000, 3000, 20000, 3000, 20000, 5000000, 1.8e-8, 6e-8)
names(param_vec) <- as.character(1:length(param_vec))


param_vec
sim$run_sim(param_vec)

