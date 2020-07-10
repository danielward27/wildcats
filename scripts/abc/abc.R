# Quite a big script, ctrl + shift + O gives an overview.

#### Imports ####
library(tidyverse)
library(data.table)
library(abc)
library(feather)
library(hydroGOF)
require(abc.data)
library(patchwork)
library(viridis)
library(corrplot)

#### Functions ####
abc_df = function(res, adjusted_strategy, unadjusted_strategy, prior_values=NA){
  # Converts abc function output values to tidy dataframe.
  # res: results from abc function
  # adjusted_strategy: String to specify the adjusted values in strategy column
  # unadjusted_strategy: String to specify the unadjusted values in strategy column
  # prior_values: parameters in the train set (if wanted to be included)
  unadj_values = data.frame(res$unadj.values)
  adj_values = data.frame(res$adj.values)
  pivot_cols = colnames(adj_values)
  adj_values$strategy = adjusted_strategy
  unadj_values$strategy = unadjusted_strategy
  
  if (!is.na(prior_values)){
    prior_values$strategy = "prior"
    df_list = list(prior_values, adj_values, unadj_values)
  }
  else {
    df_list = list(adj_values, unadj_values)
  }
  
  df_list = lapply(df_list, function(df){
    df = pivot_longer(df, cols = pivot_cols, names_to = "parameter")})
  df = rbindlist(df_list)
  return(df)
}

density_plotter = function(df, pod_prior, bandwidths){
  # Function plots density plots.
  # df: tidy formate df with cols strategy, parameter and value. Faceted by parameter, stacked plots by strategy.
  # bandwidths: vector of length parameter
  # pod_prior: single row df of pod parameters
  parameters = unique(df$parameter)
  colours = c(viridis(6)[1], viridis(6)[3], viridis(6)[5])
  for (i in 1:length(parameters)){
    param = parameters[i]
    bw = bandwidths[i]
    p_to_add = ggplot(filter(df, parameter == param),
                      aes(x=value, color=strategy,
                          fill=strategy)) + 
      ggtitle(param) +
      theme(plot.title = element_text(size=10, hjust = 0.5),
            axis.title = element_blank()) +
      geom_density(alpha=0.2, bw = bw) +
      scale_color_manual(values = colours) +
      scale_fill_manual(values = colours) +
      geom_vline(xintercept = as.double(pod_prior[param]))
    if (param == "pop_size_domestic_1"){
      p = p_to_add
    }
    else {
      p = p + p_to_add
    }
  }
  p = p +  plot_layout(ncol = 3) +
    plot_layout(guides = "collect")
  return(p)
}


# Rejection with linear (ridge) adjustment
cross_val = function(train_prior, train_sum_stats, test_prior,
                     test_sum_stats, tol, method="loclinear", transf="none",
                     adjusted_strategy, unadjusted_strategy){
  # ABC on hold out validation set. Stores posterior means.
  # adjusted_strategy: String to specify the adjusted values in strategy column
  # unadjusted_strategy: String to specify the unadjusted values in strategy column
  df_list = list()
  names = list(colnames(train_prior), colnames(train_sum_stats))
  for (i in 1:nrow(test_prior)){
    pod_sum_stats = test_sum_stats[i,]
    pod_prior = test_prior[i,]
    
    res = abc(target=pod_sum_stats, param=train_prior,
              sumstat=train_sum_stats, tol=tol,
              method=method, names=names, trans="none")
    
    res_df = abc_df(res, adjusted_strategy, unadjusted_strategy)
    mean_est = res_df %>% group_by(strategy, parameter) %>% summarise(predicted=mean(value))
    actual = pivot_longer(pod_prior, cols = colnames(pod_prior), names_to = "parameter", values_to = "pseudo_observed")
    results = left_join(mean_est, actual, by = "parameter")
    results$pod_index = i
    df_list[[i]] = results
    if (i == nrow(test_prior)/2){
      print("Wooaahh, we're half way there")
    }
  }
  df = rbindlist(df_list)
  return(df)
}

remove_outliers = function(x,threshold=2, na.rm = TRUE, ...) {
  # threshold: float denoting how many iqr above and below the upper and lower quartiles to accept
  qnt = quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H = threshold*IQR(x, na.rm = na.rm)
  y = x
  y[x < (qnt[1] - H)] = NA
  y[x > (qnt[2] + H)] = NA
  y
}

goodness_of_fit_plotter = function(df, ncol=3, title=TRUE){
  # df: dataframe with cols parameter, predicted, and pseudo_observed
  # outlier_threshold: Removes outliers to improve plotting
  parameters = sort(unique(df$parameter))
  for (i in 1:length(parameters)){
    param = parameters[i]
    df2 = df %>% filter(parameter == param)

    p_to_add = ggplot(df2, aes(predicted, pseudo_observed)) +
      theme(plot.title = element_text(size=8, hjust = 0.5),
            axis.title = element_blank()) +
      geom_point(size=0.5) +
      geom_abline(color="red")
    
    if (title){
      p_to_add = p_to_add + ggtitle(param)
    }
    
    # So we can force x and y onto same axis scales
    x_range = ggplot_build(p_to_add)$layout$panel_params[[1]]$x.range
    y_range = ggplot_build(p_to_add)$layout$panel_params[[1]]$y.range
    
    all_limits = c(x_range, y_range)
    limits = c(min(all_limits), max(all_limits))
    p_to_add = p_to_add +
      coord_cartesian(
        xlim = limits,
        ylim = limits)
    if (i==1){
      p = p_to_add
    }
    else {
      p = p + p_to_add
    }
  }
  p = p + plot_layout(ncol = ncol)
  return(p)
}

rsq = function(x, y){
  cor(x, y)^2
}

#### Read in datasets ####
train_prior = read.csv("../output/train_prior.csv")
test_prior = read.csv("../output/test_prior.csv")
train_sum_stats = read.csv("../output/train_sum_stats.csv")
test_sum_stats = read.csv("../output/test_sum_stats.csv")
train_lin_proj_sum_stats = read.csv("../output/projection/linear_regression_train_sum_stats_projection.csv")
test_lin_proj_sum_stats = read.csv("../output/projection/linear_regression_test_sum_stats_projection.csv")
train_rf_proj_sum_stats = read.csv("../output/projection/random_forest_train_sum_stats_projection.csv")
test_rf_proj_sum_stats = read.csv("../output/projection/random_forest_test_sum_stats_projection.csv")


#### plot the priors ####
p = train_prior %>%
  pivot_longer(cols = colnames(train_prior), names_to = c("parameter")) %>%
  group_by(parameter) %>%
  mutate(value = remove_outliers(value, threshold = 8)) %>% # Just so scales are sensible
  ggplot(aes(x=value)) +
  geom_density(fill="black", color="black", alpha=0.5) +
  facet_wrap(~parameter, ncol = 3, nrow = 5,  scales = "free") +
  theme(panel.spacing = unit(1.5, "lines")) +
  xlab("")
p
ggsave("../plots/prior_histograms.png", p, width = 9, height = 9, units = "in", dpi = 300 )




p

ggsave("../plots/prior_histograms.png", p, width = 9, height = 9,
       units = "in", dpi = 300 )







#### Cross-validation ####
# Choosing not to use cv4abc function as I can't choose pods, and I need
# to avoid overfitting/data leakage in the case of regression projection.

# Local linear adjustment (Beaumont et al., 2002)
cv_rej_ridge_adj = cross_val(train_prior, train_sum_stats, test_prior,
                             test_sum_stats, tol = 0.005, method = "ridge",
                             adjusted_strategy = "Rejection + ridge adj",
                             unadjusted_strategy = "Simple rejection")

# Neural net projection (Blum and Francois, 2010)
cv_nnet_proj = cross_val(train_prior, train_sum_stats, test_prior,
                         test_sum_stats, tol = 0.005, transf=c("log"), method = "neuralnet",
                         adjusted_strategy = "Neural network",
                         unadjusted_strategy = "Simple rejection")

# Linear projection and linear adjustment
cv_lin_proj_lin_adj = cross_val(train_prior, train_lin_proj_sum_stats, test_prior, 
                                test_lin_proj_sum_stats, tol=0.005, method = "loclinear",
                                adjusted_strategy = "Linear reg + rejection + local linear adj",
                                unadjusted_strategy = "Linear reg + rejection")

# Random forest projection and linear adjustment
cv_rf_proj_lin_adj = cross_val(train_prior, train_rf_proj_sum_stats,
                               test_prior, test_rf_proj_sum_stats, tol=0.005,
                               adjusted_strategy = "Random forest reg + rejection + local linear adj",
                               unadjusted_strategy = "Random forest reg + rejection")

df = bind_rows(list(cv_rej_ridge_adj,
                    filter(cv_nnet_proj, strategy == "Neural network"), # Avoid duplicate unadjusted
                    cv_lin_proj_lin_adj,
                    cv_rf_proj_lin_adj,
                    filter(cv_rf_proj_nnet_adj, strategy=="neuralnet"))
)

# Add "no abc" approach to the dataframe (i.e. just posterior means from regression projection)
pivot_cols = colnames(test_prior)
psuedo_observed = test_prior %>%
  mutate(pod_index=1:nrow(test_prior)) %>%
  pivot_longer(cols = pivot_cols, names_to = "parameter", values_to = "pseudo_observed")

linear_proj = test_lin_proj_sum_stats %>%
  mutate(pod_index = 1:nrow(test_prior)) %>%
  pivot_longer(cols=pivot_cols, names_to = "parameter", values_to = "predicted")

rf_proj = test_rf_proj_sum_stats %>%
  mutate(pod_index=1:nrow(test_prior)) %>%
  pivot_longer(cols=pivot_cols, names_to = "parameter", values_to = "predicted")

linear_proj=left_join(psuedo_observed, linear_proj, by=c("parameter", "pod_index"))
linear_proj$strategy = "Linear reg (no rejection)"
rf_proj=left_join(psuedo_observed, rf_proj, by=c("parameter", "pod_index"))
rf_proj$strategy = "Random forest reg (no rejection)"
no_abc = bind_rows(linear_proj, rf_proj)
rm(psuedo_observed, linear_proj, rf_proj)  # Clean up namespace

df = bind_rows(df, no_abc)
df$parameter = factor(df$parameter, levels = sort(unique(df$parameter)))

# There is a couple PODs that gave odd results: I'll drop these.
df = df %>% group_by(parameter, projection_strategy, adjustment_strategy) %>%
  mutate(predicted = remove_outliers(predicted, threshold = 10))

df = drop_na(df)

# write_csv(df, "../output/abc_cross_validation_results.csv")

df = read_csv("../output/abc_cross_validation_results.csv")

####  Plot cross-validation results #####
# Plot for each strategy
strategies = unique(df$strategy)

for (strat in strategies){
  plot_df = df %>% filter(strategy == strat)
  p = goodness_of_fit_plotter(plot_df) + plot_annotation(title = strat)
  file = paste0("../plots/goodness_of_fit/", strat, ".png")
  ggsave(file, p, width = 8, height = 8)
}

# Let's make a plot focussing in on the variables which worked nicely
params_that_worked = c("pop_size_domestic_1", "pop_size_wild_1", "captive_time")
df2 = df %>% filter(parameter %in% params_that_worked)

p=df2 %>% filter(strategy == strategies[1]) %>% goodness_of_fit_plotter(title = FALSE)
for (strat in strategies[c(2:length(strategies))]){
  p = p /
    df2 %>% filter(strategy == strat) %>%
    goodness_of_fit_plotter(title = FALSE)
}
ggsave("../plots/goodness_of_fit/pop_size_and_captive_time.png", p, width = 8, height = 10)


#### Calculate goodness of fit statistics #####
# Root mean square errors and R-squared
gof_stats = df %>% group_by(parameter, strategy) %>%
  summarise(rmse = rmse(predicted, pseudo_observed),
            rsq = rsq(predicted, pseudo_observed))

gof_stats %>% filter(parameter == "pop_size_domestic_1")

gof_stats %>%   # Wide format better for putting on A4 page
  pivot_wider(names_from = parameter, values_from = rmse, id_cols = strategy) %>%
  write_csv("../output/rmse.csv")

#### Plot R2 of for every strategy ####
# First change level order so it plots nicer (parameters with higher r2 on left)
param_levels = gof_stats %>% group_by(parameter) %>%
  summarise(mean=mean(rsq)) %>%
  mutate(rank=rank(mean)) %>%
  arrange(desc(rank)) %>%
  pull(parameter)

gof_stats$parameter = factor(gof_stats$parameter, levels = param_levels)

strategy_levels = gof_stats %>% group_by(strategy) %>%
  summarise(mean=mean(rsq)) %>%
  mutate(rank=rank(mean)) %>%
  arrange(desc(rank)) %>%
  pull(strategy)

gof_stats$strategy = factor(gof_stats$strategy, levels = strategy_levels)

p = ggplot(gof_stats, aes(x=parameter, y=rsq, group=strategy, col=strategy)) +
  geom_point() +
  geom_line() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(x = "Parameter", y="R-squared") +
  scale_colour_viridis_d() +
  theme(legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right")
ggsave("../plots/goodness_of_fit/rsq_by_strategy.png", p, width = 6.5, height = 8, dpi = 600)

# plot the same on a log 10 scale
p = p + scale_y_continuous(trans = "log10") +
  theme(legend.position = c(.0, .0),
        legend.justification = c("left", "bottom"),
        legend.box.just = "left",
        legend.background = element_blank())
p
ggsave("../plots/goodness_of_fit/log_10_rsq_by_strategy.png", p, width = 6.5, height = 8, dpi = 600)


#### One POD analysis ####
pod_sum_stats = test_sum_stats[7,]
pod_prior = test_prior[7,]
names = list(colnames(train_prior), colnames(train_sum_stats))

res = abc(target=pod_sum_stats, param=train_prior,
          sumstat=train_sum_stats, tol=0.005,
          method="ridge", names=names)

res_df = abc_df(res, train_prior)

res_df = res_df %>% group_by(parameter) %>%
  mutate(value = remove_outliers(value, threshold = 5)) # Remove outliers so it plots a bit better

bandwidths = c(400, 300, 5, 15, 5, 0.01, 1000, 300, 2000, 1000, 100, 1000, 0.01, 1000, 5000)
p = density_plotter(res_df, pod_prior, bandwidths)
ggsave("../plots/pod_posterior_example.png", p, height = 8, width = 10)



# Maybe we can combine the rejection and ridge regression and random forest results
# to improve MSE for pop_size_domestic_1. Random forest works better
# at lower pop sizes, whereas rejection and ridge regression work better at higher pop sizes.
df = read_csv("../output/abc_cross_validation_results.csv")
unique(df$strategy)
strats = c("Random forest reg (no rejection)", "Rejection + ridge adj")
df = df %>% filter(parameter == "pop_size_domestic_1",
                   strategy %in% strats) %>%
  dplyr::select(-projection_strategy, -adjustment_strategy) %>%
  mutate(strategy = case_when(
    strategy == "Rejection + ridge adj" ~ "rej_ridge",
    strategy == "Random forest reg (no rejection)" ~ "rf")
    ) %>%
  pivot_wider(id_cols = pod_index, names_from = strategy,
              values_from = c(predicted, pseudo_observed)) %>%
  dplyr::select(-pseudo_observed_rf) %>%
  rename(pseudo_observed = pseudo_observed_rej_ridge)

df = drop_na(df)

normalise = function(x){
  (x-min(x))/(max(x)-min(x))
}

df = df %>% mutate(normalised_pop_size = normalise((predicted_rej_ridge + predicted_rf)/2))

df = df %>%
  mutate(new_pred = predicted_rf*(1-normalised_pop_size) +
           normalised_pop_size*predicted_rej_ridge)

rmse(df$new_pred, df$pseudo_observed)

#### Rejection and ridge adjustment with specific sum stats ####
# Try choosing features with rf. Take union of 8 most important for each param.
importances = read_csv("../output/projection/random_forest_importances.csv")

importances = importances %>%
  pivot_longer(cols = colnames(dplyr::select(importances, -summary_stats)), names_to = "parameter")

importances = importances %>%
  group_by(parameter) %>%
  mutate(rank = rank(-value)) # - makes rank descending

imp_stats = importances %>%
  filter(rank < 8) %>%
  pull(summary_stats) %>%
  unique()

# Local linear adjustment (Beaumont et al., 2002)
cv_imp_stats_ridge = cross_val(train_prior, train_sum_stats[, imp_stats], test_prior,
                             test_sum_stats[, imp_stats], tol = 0.005, method = "ridge",
                             adjusted_strategy = "Important stats: rejection + ridge adj",
                             unadjusted_strategy = "Simple rejection")

imp_stats_rsq = cv_imp_stats_ridge %>% 
  filter(strategy == "Important stats: rejection + ridge adj") %>%
  group_by(parameter) %>%
  summarise(rsq = rsq(predicted, pseudo_observed))
  
all_stats_rsq = df %>% 
  filter(strategy == "Rejection + ridge adj") %>%
  group_by(parameter) %>%
  summarise(rsq = rsq(predicted, pseudo_observed))

all_stats_rsq$sum_stats = "all"
imp_stats_rsq$sum_stats = "top_25_important"

rsqs = rbind(all_stats_rsq, imp_stats_rsq)

param_levels = rsqs %>% group_by(parameter) %>%
  summarise(mean=mean(rsq)) %>%
  mutate(rank=rank(mean)) %>%
  arrange(desc(rank)) %>%
  pull(parameter)

rsqs$parameter = factor(rsqs$parameter, levels = param_levels)

p = ggplot(rsqs, aes(x=parameter, y=rsq, group=sum_stats, col=sum_stats)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(trans = "log10") +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(x = "Parameter", y="R-squared (log10 scale)") +
  scale_colour_viridis_d() +
  theme(legend.position = c(.95, .95),
        legend.justification = c("right", "top"),
        legend.box.just = "right")
ggsave("../plots/goodness_of_fit/rsq_important_subset.png", p)

#### Calculate correlation between sum stats ####
png(filename="../plots/sum_stat_corrplot.png", width = 15, height = 15, units = "in", res = 200)

train_sum_stats %>%
  as.matrix() %>%
  cor() %>%
  corrplot(method = "square", order = "FPC", tl.col = "black")

dev.off()



params = c("pop_size_domestic_1", "pop_size_wild_1")

df %>%
  filter(strategy == "Rejection + ridge adj",
         parameter %in% params) %>%
  ggplot(aes(predicted, pseudo_observed)) +
  geom_point(size=0.3) +
  facet_wrap(~parameter, ncol = 3, scales = "free") +
  geom_abline(col = "red") +
  ggsave("C:/Users/Danie/OneDrive/Documents/Year_4_work/Professional Development/gof_plot.png", width = 5.5, height = 2.5, dpi = 600)
