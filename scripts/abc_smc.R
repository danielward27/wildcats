library(reticulate)
priors = py_load_object("../output/priors.pkl")

# Get params

priors[["div_time"]]$target$pdf(40000)



list(priors)

priors$bottleneck_strength_domestic$target$rvs(100L)

iterate(priors$iteritems(), print)

library(tidyverse)
library(corrplot)
# Plot prior compared to posterior kde plot

plot_densities = function(posterior, pdf, obs=NULL){
  # Plots the marginal posterior densities (kernal density estimation)
  # against the prior pdfs.
  # posterior: df with cols parameter, value and weights
  # pdf: df of densities with cols parameter, x, value
  # obs: (pseudo) observed parameter values in df with cols pameter, 
  p = ggplot() +
    facet_wrap(~parameter, scales = "free") +
    geom_density(data=posterior, aes(x=value, weight=weights, colour="steelblue4"), ) +
    geom_line(data=pdf, aes(x, value, colour="red"),
              inherit.aes = FALSE) +
    guides(col=guide_legend("Distribution")) +
    scale_color_hue(labels = c("Prior", "Posterior")) +
    theme_bw()
  if (!is.null(obs)){
    p = p + geom_vline(data=obs, aes(xintercept=value))
  }
  return(p)
}


smc_res = read_csv("../output/smc_posterior.csv")
smc_res_long = smc_res %>% pivot_longer(-"weights", names_to="parameter")

smc_pdf = read_csv("../output/prior_pdf.csv")

y_obs = c(
  3000.0, 10000.0, 3000.0, 
  3000.0, 20.0, 20000.0,
  1000.0, 20.0, 0.01, 0.1,
  0.01, 100.0, 100.0, 100.0,
  100.0, 100.0
)

y_obs = data.frame(parameter=sort(unique(smc_res_long$parameter)), value=y_obs)

p = plot_densities(smc_res_long, smc_pdf, y_obs)
ggsave("../plots/marginal_posterior_densities.png",
       width = 12, height = 7)

smc_res %>%
  select(-"weights") %>%
  cor() %>%
  corrplot(type = "lower", method = "square", tl.col = "black")





#---- Imports ----
library(reticulate)
library(EasyABC)
library(glue)
library(tictoc)









use_condaenv("wildcats_summer_env")
sim <- import("sim.model")

run_sim_parallel <- function(x) {
  # Small wrapper function to run the simulation from R.
  # EasyABC has no way to export variables or functions to clusters,
  # So we have to do things a bit weirdly.
  
  #---- Imports ----
  library(reticulate)
  library(EasyABC)
  library(glue)
  library(tictoc)
  use_condaenv("wildcats_summer_env")
  sim <- import("sim.model")
  
  #---- Helper functions and variables ----
  param_names = names(x)
  
  Xn <- function(param){
    # Returns "Xn" string, where n is replaced with index of parameter
    idx = which(param_names==param)
    Xn = paste0("X", idx)
    return(Xn)
  }
  
  param_names = c(
    "random_seed", "pop_size_domestic_1", "pop_size_wild_1", "pop_size_captive", "captive_time",
    "mig_rate_captive", "mig_length_wild", "mig_rate_wild", "pop_size_domestic_2", "pop_size_wild_2",
    "bottleneck_time_domestic", "bottleneck_strength_domestic", "bottleneck_time_wild", "bottleneck_strength_wild",
    "mig_length_post_split", "mig_rate_post_split", "div_time", "seq_length", "recombination_rate", "mutation_rate"
  )
  
  #---- Run simulation ----
  names(x) = param_names
  param_list <- lapply(split(x, names(x)), unname)  # Named list (gets converted to python dict)
  sum_stats <- sim$run_sim(param_list)
  sum_stats <- unlist(sum_stats)
  
  return(sum_stats)
}

print_prior_summary <- function(prior_list){
  function_mappings = c("normal"=rnorm, "unif"=runif, "lognormal"=rlnorm)
  for (i in 1:length(prior)){
    param = names(prior[i])
    
    dist_name = prior[[i]][1]
    dist_args = prior[[i]][-1]
    dist_args = as.list(c(100, as.numeric(dist_args)))
    dist_func = as.function(function_mappings[[dist_name]])
    sample = do.call(dist_func, dist_args)
    
    print(param)
    print(dist_name)
    print(summary(sample))
    cat("\n")
  }
}

#---- Example "pod" ----

example_param_vec = c(1, 100, 100, 100, 40,0.1,20,0.1,400, 400, 30000, 0.1,
                      5000, 3000, 10000, 3000, 20000, 5000000, 1.8e-8, 6e-8)
example_target = run_sim_parallel(example_param_vec)

#---- Prior ----
prior=list(
  "pop_size_domestic_1" = c("unif", 100, 1000), ## Changed
  "pop_size_wild_1" = c("lognormal", 8, 0.4),
  "pop_size_captive" = c("lognormal", 4.5, 0.3),
  "captive_time" = c("lognormal", 3, 0.7),
  "mig_rate_captive" = c("lognormal", -4, 1),
  "mig_length_wild" = c("lognormal", 3, 0.7),
  "mig_rate_wild" = c("lognormal", -3, 1),
  "pop_size_domestic_2" =  c("unif", 1000, 2000), ## Changed
  "pop_size_wild_2" = c("lognormal", 8.8, 0.2),
  "bottleneck_time_domestic" = c("normal", 3500, 600),
  "bottleneck_strength_domestic" = c("unif", 0, 40000),
  "bottleneck_time_wild" = c("normal", 3500, 600),
  "bottleneck_strength_wild" = c("unif", 0, 40000),
  "mig_length_post_split" = c("unif", 0, 10000),
  "mig_rate_post_split" = c("unif", 0, 0.1),
  "div_time" = c("normal", 40000, 5000),
  "seq_length" = c("unif", 10e6, 10e6),
  "recombination_rate" = c("unif", 1.8e-8, 1.8e-8),
  "mutation_rate" = c("unif", 6e-8, 6e-8)
  )

print_prior_summary(prior)

prior = unname(prior)

#---- Run rejection algorithm ----
set.seed(4)
tic("Sequential - 30 cores:")
ABC_beaumont <- ABC_sequential(method = "Beaumont", model=run_sim_parallel, prior=prior, nb_simul=15,
                               summary_stat_target=example_target, use_seed = TRUE,
                               n_cluster = 20, tolerance_tab=c(1.5e15, 1.4e15))
print(toc(log=TRUE))

tic("Sequential - 20 cores:")
ABC_beaumont <- ABC_sequential(method = "Beaumont", model=run_sim_parallel, prior=prior, nb_simul=15,
                               summary_stat_target=example_target, use_seed = TRUE,
                               n_cluster = 10, tolerance_tab=c(1.5e15, 1.4e15))
print(toc(log=TRUE))


tic("Sequential - 10 cores:")
ABC_beaumont <- ABC_sequential(method = "Beaumont", model=run_sim_parallel, prior=prior, nb_simul=15,
                          summary_stat_target=example_target, use_seed = TRUE,
                          n_cluster = 5, tolerance_tab=c(1.5e15, 1.4e15))
print(toc(log=TRUE))




# prior_test = glue('{Xn("div_time")} > {Xn("mig_rate_post_split")}')
#glue('{Xn("mig_rate_post_split")} > 0')
#Xn("mig_rate_post_split")

#glue('{Xn("div_time")} > {Xn("mig_rate_post_split")}')

#Xn("bottleneck_time_wild")
#glue('{Xn("div_time")} > 50')


