#---- Imports ----
library(reticulate)
library(tidyverse)
library(corrplot)

#---- Functions ----
# Plot prior compared to posterior kde

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


#---- Load in data ----
priors = py_load_object("../output/priors.pkl")
posterior = py_load_object("../output/smc_posterior.pkl")
param_names = posterior$parameter_names

#---- Format data for plotting ----

# Prior pdf evaluations
pdf_evaluations = tibble()
for (param in param_names){
  prior = priors[[param]]$target
  xlims = prior$ppf(c(0.001, 0.999))
  x = seq(xlims[1], xlims[2], length.out = 1000)
  value = prior$pdf(x)
  df = tibble(x, value)
  df$parameter = param
  pdf_evaluations = bind_rows(pdf_evaluations, df)
}

posterior_df = data.frame(posterior$samples)

# Scale up samples from sampling distribution to values used in simulator
for (param in param_names){
  posterior_df[param] = priors[[param]]$scale_up_samples(posterior_df[param])
}

posterior_df$weights = posterior$weights
posterior_df_long = posterior_df %>% pivot_longer(-"weights", names_to="parameter")

#---- Plots ----
# Density plot
p = plot_densities(posterior_df_long, pdf_evaluations)
p

ggsave("../plots/marginal_posterior_densities.png",
       width = 12, height = 7)

# corrplot
par(xpd=TRUE)
png(height=8, width=6, units = "in", res=600, file="../plots/smc_posterior_corrplot.png")

posterior_df %>%
  select(-"weights") %>%
  cor() %>%
  corrplot(type = "lower", method = "square", tl.col = "black", tl.cex = 0.7,
           mar = c(0, 0, 0, 2))

dev.off()

posterior$summary(all=TRUE)
quantile(posterior$discrepancies, 0.01)  # The distances get pretty tight!

