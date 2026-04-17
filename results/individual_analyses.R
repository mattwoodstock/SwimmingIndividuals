# ==============================================================================
# SwimmingIndividuals: Individual-Level Time Series Analysis
# ==============================================================================
# Developer: Matt Woodstock
# Description: Processes individual-level tracking data to generate 
# manuscript-quality plots detailing population, growth, and spatial dynamics.
# ==============================================================================

# --- 1. Load Libraries ---
# Install missing packages if necessary
required_packages <- c("ggplot2", "dplyr", "tidyr", "raster", "marmap", 
                       "gganimate", "patchwork", "viridis", "purrr")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(ggplot2)
library(dplyr)
library(tidyr)
library(raster)
library(marmap)
library(gganimate)
library(patchwork) # For combining multiple plots into one manuscript figure
library(viridis)   # Colorblind-safe palettes
library(purrr)     # For fast data loading

# --- 2. Configuration & Fast Data Loading ---
indir <- "../examples/Mackerel/results/Individual"
# Fallback to current directory if ./Individual doesn't exist (for testing)
if(!dir.exists(indir)) indir <- "." 

# Find all files matching the pattern
files <- list.files(indir, pattern = "IndividualResults_1-.*\\.csv", full.names = TRUE)

if(length(files) == 0) {
  stop("No IndividualResults files found. Check your directory path.")
}

message(paste("Loading", length(files), "timesteps..."))

# Use map_df (much faster than rbind in a loop for thousands of files)
all_dat <- files %>%
  map_df(~ {
    # Extract timestep number strictly from the filename
    ts <- as.numeric(gsub(".*_1-([0-9]+)\\.csv", "\\1", basename(.x)))
    read.csv(.x) %>% mutate(Time = ts)
  }) %>%
  arrange(Time, Individual)

message("Data loading complete!")

# Define standard manuscript theme
theme_ms <- theme_classic(base_size = 16) +
  theme(plot.title = element_text(face = "bold"),
        strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "bold"))

# ==============================================================================
# PLOT 1: POPULATION DEMOGRAPHICS (Biomass & Abundance)
# ==============================================================================
# Shows the emergent stability of the population over the simulation.

pop_summary <- all_dat %>%
  group_by(Time) %>%
  summarize(
    Total_Biomass_T = sum(Biomass, na.rm = TRUE) / 1000, # Assuming kg to Metric Tons
    Total_Abundance = sum(Abundance, na.rm = TRUE)
  )

p1a <- ggplot(pop_summary, aes(x = Time, y = Total_Biomass_T)) +
  geom_line(color = "#D55E00", size = 1.2) +
  labs(title = "A) Population Biomass", x = "Timestep", y = "Total Biomass (Tons)") +
  theme_ms

p1b <- ggplot(pop_summary, aes(x = Time, y = Total_Abundance)) +
  geom_line(color = "#0072B2", size = 1.2) +
  labs(title = "B) Population Abundance", x = "Timestep", y = "Total Individuals") +
  theme_ms

plot_demographics <- p1a / p1b # Stack using patchwork
ggsave("Figure_1_Demographics.png", plot_demographics, width = 8, height = 8, dpi = 300)

# ==============================================================================
# PLOT 2: BIOENERGETIC BALANCE (Cost vs. Intake)
# ==============================================================================
# Shows if the population is starving or accumulating surplus energy over time.

energy_summary <- all_dat %>%
  group_by(Time) %>%
  summarize(
    Mean_Ration_E = mean(Ration_e, na.rm = TRUE),
    Mean_Cost = mean(Cost, na.rm = TRUE),
    Mean_Fullness = mean(Fullness, na.rm = TRUE)
  ) %>%
  pivot_longer(cols = c(Mean_Ration_E, Mean_Cost), 
               names_to = "Metric", values_to = "Joules")

plot_energy <- ggplot(energy_summary, aes(x = Time, y = Joules, color = Metric)) +
  geom_line(size = 1.2) +
  scale_color_manual(values = c("Mean_Cost" = "red", "Mean_Ration_E" = "darkgreen"),
                     labels = c("Metabolic Cost", "Energy Consumed")) +
  labs(title = "Population Bioenergetic Balance",
       x = "Timestep", y = "Energy (Joules / Timestep)") +
  theme_ms +
  theme(legend.position = "bottom", legend.title = element_blank())

ggsave("Figure_2_Bioenergetics.png", plot_energy, width = 8, height = 5, dpi = 300)

# ==============================================================================
# PLOT 3: EMERGENT GROWTH CURVES (Length at Age by Cohort)
# ==============================================================================
# Validates whether the agents grow realistically based on their energy surplus.

growth_summary <- all_dat %>%
  group_by(Generation, Age) %>%
  summarize(
    Mean_Length = mean(Length, na.rm = TRUE),
    SD_Length = sd(Length, na.rm = TRUE),
    N = n(),
    .groups = 'drop'
  ) %>%
  filter(N > 5) # Filter out noise from dying cohorts

plot_growth <- ggplot(growth_summary, aes(x = Age, y = Mean_Length, color = as.factor(Generation))) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = Mean_Length - SD_Length, ymax = Mean_Length + SD_Length, fill = as.factor(Generation)), 
              alpha = 0.2, color = NA) +
  scale_color_viridis_d(name = "Cohort (Gen)") +
  scale_fill_viridis_d(name = "Cohort (Gen)") +
  labs(title = "Emergent Growth Trajectories",
       x = "Age (Timesteps)", y = "Mean Length (mm)") +
  theme_ms

ggsave("Figure_3_Growth.png", plot_growth, width = 8, height = 5, dpi = 300)

# ==============================================================================
# PLOT 4: VERTICAL HABITAT UTILIZATION (Depth over time)
# ==============================================================================

depth_summary <- all_dat %>%
  group_by(Time) %>%
  summarize(
    Mean_Z = mean(Z, na.rm = TRUE),
    Q25 = quantile(Z, 0.25, na.rm = TRUE),
    Q75 = quantile(Z, 0.75, na.rm = TRUE)
  )

plot_depth <- ggplot(depth_summary, aes(x = Time, y = Mean_Z)) +
  geom_ribbon(aes(ymin = Q25, ymax = Q75), fill = "navy", alpha = 0.3) +
  geom_line(color = "navy", size = 1) +
  scale_y_reverse() + # Reverse Y axis so 0 (surface) is at the top
  labs(title = "Vertical Depth Distribution over Time",
       subtitle = "Line = Mean Depth, Shaded = Interquartile Range (25th-75th)",
       x = "Timestep", y = "Depth (Z)") +
  theme_ms

ggsave("Figure_4_Vertical_Depth.png", plot_depth, width = 8, height = 5, dpi = 300)

# ==============================================================================
# PLOT 5: STATIC SPATIAL DENSITY (For Manuscript Print)
# ==============================================================================
# Animations are great for presentations, but papers need a static heatmap.
# Note: marmap queries can fail if offline, so we use a robust density plot.

# Grab boundaries from data
lon_bounds <- range(all_dat$X, na.rm = TRUE)
lat_bounds <- range(all_dat$Y, na.rm = TRUE)

# Try fetching bathymetry, but don't crash if offline
bathy_df <- NULL
tryCatch({
  message("Fetching NOAA Bathymetry for background...")
  bathy_data <- getNOAA.bathy(lon1 = floor(lon_bounds[1])-1, lon2 = ceiling(lon_bounds[2])+1, 
                              lat1 = floor(lat_bounds[1])-1, lat2 = ceiling(lat_bounds[2])+1, 
                              resolution = 4)
  bathy_df <- fortify.bathy(bathy_data)
}, error = function(e) {
  message("Could not fetch bathymetry (offline?). Proceeding without it.")
})

plot_spatial <- ggplot()

# Add bathymetry background if available
if(!is.null(bathy_df)) {
  plot_spatial <- plot_spatial + 
    geom_contour(data = bathy_df, aes(x = x, y = y, z = z), 
                 breaks = c(-50, -100, -200, -500), color = "gray80", size = 0.3)
}

# Overlay 2D density of where agents spent their time
plot_spatial <- plot_spatial +
  stat_density_2d(data = all_dat, aes(x = X, y = Y, fill = ..level..), 
                  geom = "polygon", alpha = 0.7, bins = 15) +
  scale_fill_viridis_c(option = "magma", name = "Agent Density") +
  coord_fixed(xlim = lon_bounds, ylim = lat_bounds) +
  labs(title = "Aggregated Spatial Distribution",
       subtitle = "Density of agent locations across all timesteps",
       x = "Longitude", y = "Latitude") +
  theme_ms

ggsave("Figure_5_Spatial_Density.png", plot_spatial, width = 8, height = 6, dpi = 300)

# ==============================================================================
# PLOT 6: SPATIAL ANIMATION (For Presentations)
# ==============================================================================
message("Generating animation... this may take a few minutes depending on data size.")

# Subsample data to speed up animation rendering if there are many timesteps
anim_dat <- all_dat %>% filter(Time %% max(1, floor(max(Time)/50)) == 0)

anim_plot <- ggplot()
if(!is.null(bathy_df)) {
  anim_plot <- anim_plot + 
    geom_raster(data = bathy_df %>% filter(z < 0), aes(x = x, y = y, fill = z)) +
    scale_fill_gradientn(colors = c("navy", "blue", "lightblue", "white"),
                         name = "Depth (m)")
}

anim_plot <- anim_plot +
  geom_point(data = anim_dat, aes(x = X, y = Y, size = Length),
             color = "red", alpha = 0.6, inherit.aes = FALSE) +
  coord_fixed(xlim = lon_bounds, ylim = lat_bounds) +
  theme_minimal() +
  labs(title = "King Mackerel Locations - Timestep: {frame_time}",
       x = "Longitude", y = "Latitude", size = "Length (mm)") +
  transition_time(Time) +
  ease_aes('linear')

# Save as GIF
anim_save("Figure_6_Animation.gif", animation = anim_plot, fps = 10, width = 800, height = 600)

message("All analyses complete! Check your working directory for the PNG and GIF files.")