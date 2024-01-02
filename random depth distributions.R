
#ChatGPT prompt: Animals were collected at each meter from the surface to 1000m, and the abundance distribution fits a multimodal curve with 3 peaks. Create the example data for this distribution, fit the curve, and extract the parameter values

# Install and load necessary packages
if (!requireNamespace("mixtools", quietly = TRUE)) {
  install.packages("mixtools")
}
library(mixtools)

vals <- data.frame(species_no = numeric(),mu1 = numeric(),mu2=numeric(),mu3 = numeric(),sigma1=numeric(),sigma2=numeric(),sigma3=numeric(),lambda1=numeric(),lambda2=numeric(),lambda3=numeric())

library(dplyr)
for (i in 1:232){
  # Set a seed for reproducibility
  
  # Create synthetic data with three peaks
  depth <- seq(1, 1000, by = 1)
  avg1 <- runif(1,0.1,200)
  avg2 <- runif(1,150.1,300)
  avg3 <- runif(1,300.1,500)
  avg4 <- runif(1,400.1,600)
  avg5 <- runif(1,500.1,700)
  avg6 <- runif(1,700.1,900)
  
  abundance <- c(
    #rnorm(round(runif(1,100,500),0), mean = avg1, sd = runif(1,1,100)),
    #rnorm(round(runif(1,100,500),0), mean = avg2, sd = runif(1,1,100)),
    rnorm(round(runif(1,100,500),0), mean = avg3, sd = runif(1,1,100)),
    rnorm(round(runif(1,100,500),0), mean = avg4, sd = runif(1,1,100)),
    rnorm(round(runif(1,100,500),0), mean = avg5, sd = runif(1,1,100)),
    rnorm(round(runif(1,100,500),0), mean = avg6, sd = runif(1,1,100)))
  
  depth <- seq(1,length(abundance))
  
  # Combine data into a dataframe
  data <- data.frame(depth = depth, abundance = abundance)
  
  # Plot the synthetic data
  plot(data$depth, data$abundance, pch = 16, col = "blue",
       xlab = "Depth", ylab = "Abundance", main = "Synthetic Data with 3 Peaks")
  
  # Fit a Gaussian Mixture Model (GMM) with three components
  fit <- normalmixEM(data$abundance, k = 3)
 
  vals = vals %>% add_row(species_no = i,mu1 = fit$mu[1],mu2=fit$mu[2],mu3 = fit$mu[3],sigma1=fit$sigma[1],sigma2=fit$sigma[2],sigma3=fit$sigma[3],lambda1=fit$lambda[1],lambda2=fit$lambda[2],lambda3=fit$lambda[3]) 
}
setwd("D:/SwimmingIndividuals/Adapted")
write.csv(vals,"Random Depth Distributions_Day.csv")



#Night
vals <- data.frame(species_no = numeric(),mu1 = numeric(),mu2=numeric(),mu3 = numeric(),sigma1=numeric(),sigma2=numeric(),sigma3=numeric(),lambda1=numeric(),lambda2=numeric(),lambda3=numeric())

library(dplyr)
for (i in 1:232){
  # Set a seed for reproducibility
  
  # Create synthetic data with three peaks
  depth <- seq(1, 1000, by = 1)
  avg1 <- runif(1,0.1,75)
  avg2 <- runif(1,50.1,125)
  avg3 <- runif(1,100.1,175)
  avg4 <- runif(1,150.1,250)
  avg5 <- runif(1,500.1,700)
  avg6 <- runif(1,700.1,900)
  
  abundance <- c(
    rnorm(round(runif(1,100,500),0), mean = avg1, sd = runif(1,1,25)),
    rnorm(round(runif(1,100,500),0), mean = avg2, sd = runif(1,1,25))
    #rnorm(round(runif(1,100,500),0), mean = avg3, sd = runif(1,1,25)),
    #rnorm(round(runif(1,100,500),0), mean = avg4, sd = runif(1,1,25))
    #rnorm(round(runif(1,100,500),0), mean = avg5, sd = runif(1,1,100)),
    #rnorm(round(runif(1,100,500),0), mean = avg6, sd = runif(1,1,100)))
  )
  depth <- seq(1,length(abundance))
  
  # Combine data into a dataframe
  data <- data.frame(depth = depth, abundance = abundance)
  
  # Plot the synthetic data
  plot(data$depth, data$abundance, pch = 16, col = "blue",
       xlab = "Depth", ylab = "Abundance", main = "Synthetic Data with 3 Peaks")
  
  # Fit a Gaussian Mixture Model (GMM) with three components
  fit <- normalmixEM(data$abundance, k = 3)
  
  vals = vals %>% add_row(species_no = i,mu1 = fit$mu[1],mu2=fit$mu[2],mu3 = fit$mu[3],sigma1=fit$sigma[1],sigma2=fit$sigma[2],sigma3=fit$sigma[3],lambda1=fit$lambda[1],lambda2=fit$lambda[2],lambda3=fit$lambda[3]) 
}

setwd("D:/SwimmingIndividuals/Adapted")
write.csv(vals,"Random Depth Distributions_Night.csv")
# Print the summary of the fit
summary(fit)


# Display the estimated parameters
library(ggplot2)
ggplot(data, aes(x = abundance)) +
  geom_density(fill = "lightblue", color = "black", alpha = 0.7) +
  stat_function(fun = function(x) {
    fit$lambda[1] * dnorm(x, mean = fit$mu[1], sd = fit$sigma[1]) +
      fit$lambda[2] * dnorm(x, mean = fit$mu[2], sd = fit$sigma[2]) +
      fit$lambda[3] * dnorm(x, mean = fit$mu[3], sd = fit$sigma[3])
  }, color = "red", linewidth = 1) +
  labs(title = "Histogram with Fitted Mixture Model Curve",
       x = "Abundance",
       y = "Frequency") +
  theme_minimal()









## Resample Distribution

gaussmix <- function(n,m1,m2,m3,s1,s2,s3,alpha) {
  I <- runif(n)<alpha
  I2 <- runif(n)<alpha
  rnorm(n,mean=ifelse(I,m1,ifelse(I2,m2,m3)),sd=ifelse(I,s1,ifelse(I2,s2,s3)))
}
s <- gaussmix(1000000,fit$mu[1],fit$mu[2],fit$mu[3],fit$sigma[1],fit$sigma[2],fit$sigma[3],0.5)

abs(s)

hist(abs(s))

