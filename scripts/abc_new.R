#### Imports ####
library(tidyverse)
library(abc)
library(feather)
library(reticulate)
library(corrplot)

#### Read in datasets ####
prior = read_feather("../output/rejection/priors.feather")
prior$random_seed <- 0:(nrow(prior)-1)

sum_stats = read.csv("../output/summary_stats.csv")

observed_df <- read_csv("../output/observed_summary_stats_e2.csv")
observed <- as.numeric(as.vector(observed_df))
names(observed) <- colnames(observed_df)


sprintf("Prior rows: %s sum_stats rows: %s", nrow(prior), nrow(sum_stats))

prior = prior[prior$random_seed %in% sum_stats$random_seed, ]
sum_stats = sum_stats[sum_stats$random_seed %in% prior$random_seed, ]

stopifnot(all(prior["random_seed"] == sum_stats["random_seed"]))

table(is.na(sum_stats))

prior = dplyr::select(prior, -random_seed)
sum_stats = dplyr::select(sum_stats, -random_seed)

start_time <- Sys.time()
tol = 0.01
res = cv4abc(param = data.frame(prior), sumstat = data.frame(sum_stats),
   nval = 200, method = "ridge", tols = tol, statistic = "mean")
end_time <- Sys.time()

start_time-end_time

y_hat = data.frame(res$estim) %>%
  set_names(~ str_remove(., paste0("tol", as.character(tol), "."))) %>%
  mutate(pod_index = 1:nrow(.)) %>%
  pivot_longer(cols = colnames(.)[colnames(.) != "pod_index"],
               names_to = "parameter", values_to = "predicted")

y = data.frame(res$true) %>%
  mutate(pod_index = 1:nrow(.)) %>%
  pivot_longer(cols = colnames(.)[colnames(.) != "pod_index"],
               names_to = "parameter", values_to = "pseudo_observed")

df = full_join(y, y_hat, by=c("parameter", "pod_index"))



### CORPLOT



remove_outliers = function(x,threshold=2, na.rm = TRUE, ...) {
  # threshold: float denoting how many iqr above and below the upper and lower quartiles to accept
  qnt = quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H = threshold*IQR(x, na.rm = na.rm)
  y = x
  y[x < (qnt[1] - H)] = NA
  y[x > (qnt[2] + H)] = NA
  y
}

df_no_outliers = df %>% group_by(parameter) %>%
  mutate(pseudo_observed = remove_outliers(pseudo_observed, threshold = 10),
         predicted = remove_outliers(predicted, threshold = 10))
  
p1 = df_no_outliers %>%
  ggplot(aes(x=pseudo_observed, y=predicted)) +
  facet_wrap(~parameter, scales = "free") +
  geom_point(size=0.7) +
  geom_abline(colour="red")

p1

ggsave("../plots/goodness_of_fit/rejection_cv4abc_ridge.png", p1,
       height = 8, width = 12, units = "in")

# cv looks ok. Now lets use the observed data.
abc_res = abc(target = observed, param = data.frame(prior),
          sumstat = data.frame(sum_stats), 
          method = "ridge", tol = tol, statistic = "mean")


posterior_df <- as.data.frame(abc_res$unadj.values) %>%
  pivot_longer(everything(), values_to="parameter_value",
               names_to = "parameter")

ggplot() +
  geom_density(data = posterior_df, aes(x = parameter_value)) +
  facet_wrap(~parameter, scales = "free")


# Add priors to plot
use_condaenv(condaenv = 'wildcats_env', required=TRUE)
priors <- py_load_object("../output/priors.pkl")

param_names <- names(priors)

n = 1000
x_vec <- c()
density_vec <- c()
parameter <- c()

for (i in 1:length(priors)){
  prior <- priors[[i]]
  name <- names(priors)[i]
  min_x <- prior$target$ppf(0.001)
  max_x <- prior$target$ppf(0.999)
  x <- seq(min_x, max_x, length.out = n)
  density <- prior$target$pdf(x)
  
  x_vec <- c(x_vec, x)
  density_vec <- c(density_vec, density)
  parameter <- c(parameter, rep(name, n))
}

prior_density_df <- tibble(x_vec, density_vec, parameter)


ggplot() +
  geom_density(data = posterior_df, aes(x = parameter_value)) +
  geom_line(data = prior_density_df, aes(x = x_vec, y=density_vec), colour="red") +
  facet_wrap(~parameter, scales = "free") +
  theme_bw()

ggsave("../plots/posterior_v_prior_ridge.png",
       height = 8, width = 12, units = "in")

cor(abc_res$unadj.values) %>%
  corrplot()







corr = df %>%
  dplyr::select(-pseudo_observed) %>%
  pivot_wider(id_cols = pod_index, names_from = parameter,
              values_from = c("predicted")) %>%
  dplyr::select(-pod_index) %>%
  cor()

corrplot::corrplot(corr, method = "square", tl.col = "black")
