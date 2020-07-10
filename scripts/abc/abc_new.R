#### Imports ####
library(tidyverse)
library(abc)

#### Read in datasets ####
prior = read.csv("../../output/prior.csv", )
sum_stats = read.csv("../../output/summary_stats.csv")

sprintf("Prior rows: %s sum_stats rows: %s", nrow(prior), nrow(sum_stats))
shared_seeds = intersect(prior$random_seed, sum_stats$random_seed)
prior$migration_length_1 = prior$migration_length_1

prior = prior[prior$random_seed %in% sum_stats$random_seed, ]
sum_stats = sum_stats[sum_stats$random_seed %in% prior$random_seed, ]

stopifnot(all(prior["random_seed"] == sum_stats["random_seed"]))

table(prior$migration_length_1)

prior = dplyr::select(prior, -random_seed)
sum_stats = dplyr::select(sum_stats, -random_seed)

start_time <- Sys.time()
tol = 0.01
res = cv4abc(param = data.frame(prior), sumstat = data.frame(sum_stats),
       nval = 500, method = "ridge", tols = tol)
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


remove_outliers = function(x,threshold=2, na.rm = TRUE, ...) {
  # threshold: float denoting how many iqr above and below the upper and lower quartiles to accept
  qnt = quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H = threshold*IQR(x, na.rm = na.rm)
  y = x
  y[x < (qnt[1] - H)] = NA
  y[x > (qnt[2] + H)] = NA
  y
}
df

df_no_outliers = df %>% group_by(parameter) %>%
  mutate(pseudo_observed = remove_outliers(pseudo_observed, threshold = 6),
         predicted = remove_outliers(predicted, threshold = 6))
  

p = df_no_outliers %>%
  ggplot(aes(x=pseudo_observed, y=predicted)) +
  facet_wrap(~parameter, scales = "free") +
  geom_point(size=0.7) +
  geom_abline(colour="red")

p


#ggsave("../../plots/goodness_of_fit/with_migration_1_added_and0001_GOF.png", p,
       height = 8, width = 12, units = "in")
