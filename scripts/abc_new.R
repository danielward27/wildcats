#### Imports ####
library(tidyverse)
library(abc)
library(feather)
library(reticulate)
library(corrplot)


#### Key hyperparameters
method = "neuralnet"  # match abc function "method"
tol = 0.01

#### Functions
remove_cols <- function(X, idxs){
  print("Removing columns:")
  print(colnames(X)[idxs])
  X <- X[ , -idxs]
  X
}

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


#### Some summary stats don't have many unique values. Filter these out.
threshold = 0.75  # Percent threshold

remove_idxs <- c()
for (i in 1:ncol(sum_stats)){
  col <- sum_stats[,i]
  if (length(unique(col)) / nrow(sum_stats) < threshold){
    remove_idxs <- c(remove_idxs, i)
  }
}

sum_stats <- remove_cols(sum_stats, remove_idxs)
observed <- observed[-remove_idxs]

#### Filter out some highly correlated sum stats
correlated_vars <- which(cor(sum_stats) > 0.95, arr.ind = TRUE) %>%
  data.frame() %>%
  filter(row != col)

# Randomly remove variables to limit extreme collinearity
remove_idxs = c()
set.seed(1)
while (nrow(correlated_vars) > 0){
  idx <- sample(unique(correlated_vars$row), 1)
  correlated_vars <-  filter(correlated_vars, row != idx, col != idx)
  remove_idxs = c(remove_idxs, idx)
}

sum_stats <- remove_cols(sum_stats, remove_idxs)
observed <- observed[-remove_idxs]

# Log transform and standardize summary statistics
for (i in 1:ncol(sum_stats)){
  smallest <- min(sum_stats[ , i])
  if (smallest < 0){
    sum_stats[ , i] <- sum_stats[ , i] - smallest
    observed[i] <- observed[i] - smallest
  }

  # Log
  sum_stats[ , i] <- log(sum_stats[ , i] + 1e-10)
  observed[i] <- log(observed[i] + 1e-10)

  # Scale
  mean_ <- mean(sum_stats[ , i])
  sd_ <- sd(sum_stats[ , i])

  sum_stats[ , i] <- (sum_stats[ , i] - mean_) / sd_
  observed[i] <- (observed[i] - mean_) / sd_
}

#### Run cross-validation
res = cv4abc(param = data.frame(prior), sumstat = data.frame(sum_stats),
               nval = 50, method = method, tols = tol, statistic = "median",
               transf = "log")

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


df_no_outliers = df %>% group_by(parameter) %>%
  mutate(pseudo_observed = remove_outliers(pseudo_observed, threshold = 10),
         predicted = remove_outliers(predicted, threshold = 10))

p1 = df %>%
  ggplot(aes(x=pseudo_observed, y=predicted)) +
  facet_wrap(~parameter, scales = "free") +
  geom_point(size=0.7) +
  geom_abline(colour="red")

p1

ggsave(sprintf("../plots/rejection_cv4abc_%s_%s.png", method, tol), p1,
       height = 8, width = 12, units = "in")


# cv looks ok. Now lets use the observed data.
abc_res = abc(target = observed, param = data.frame(prior),
              sumstat = data.frame(sum_stats),
              method = method, tol = tol, transf = "log")

all.equal(abc_res$adj.values, abc_res$unadj.values)



all.equal(target = data.frame(abc_res$adj.values), current = data.frame(abc_res$unadj.values))


if (method == "rejection"){
  posterior_values = abc_res$unadj.values
} else {
  posterior_values = abc_res$adj.values
}


posterior_df <- as.data.frame(posterior_values) %>%
  pivot_longer(everything(), values_to="parameter_value",
               names_to = "parameter")

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

ggsave(sprintf("../plots/posterior_v_prior_%s_%s.png", method, tol),
       height = 8, width = 12, units = "in")

png(height = 6, width=6, units = "in", res = 300,
    file = sprintf("../plots/rejection_corrplot_%s_%s.png", method, tol))
cor(posterior_values) %>%
  corrplot()
dev.off()


write_csv(as.data.frame(posterior_values),
          sprintf("../output/rejection/posterior_%s_%s.csv", method, tol))
