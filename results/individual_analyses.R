#### Example individual analyses for a SwimmingIndividuals model
####### Developer: Matt Woodstock


## Load Libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(raster)
library(marmap)
library(gganimate)

indir = "./Individual"
testdir = "./Test"
files = list.files(indir)
test_files = list.files(testdir)

#Load all data (May need to break up if there are a lot of iterations) ----
all_dat = c()
pb = txtProgressBar(min=0,max=n_distinct(files),style=3)
for (file in 1:n_distinct(files)){
  scen_dat = read.csv(paste0(indir,"/IndividualResults_1-",file,".csv")) %>% mutate(Time = file)
  all_dat = rbind(all_dat,scen_dat)
  setTxtProgressBar(pb,file) #Progress will slow as the df gets larger
}

head(all_dat)

#Location Analyses ----
bathy_data <- getNOAA.bathy(lon1 = -98, lon2 = -80, lat1 = 23, lat2 = 31, resolution = 1)
bathy_df <- expand.grid(
  lon = as.numeric(attr(bathy_data, "dimnames")[[1]]),
  lat = as.numeric(attr(bathy_data, "dimnames")[[2]])
)

bathy_df$depth <- as.vector(-1 * bathy_data)
coords = all_dat %>% filter(Time == 1)

bathy_df$depth[bathy_df$depth < 0] <- NA
head(bathy_df)

p <- ggplot(bathy_df, aes(x = lon, y = lat, fill = depth)) +
  geom_raster(na.rm = TRUE) +
  geom_point(data = all_dat, aes(x = X, y = Y),
             color = "red", size = 2, inherit.aes = FALSE) +
  scale_fill_gradientn(colors = c("white", "lightblue", "blue", "navy"),
                       na.value = "black",
                       name = "Depth (m)") +
  coord_fixed() +
  theme_minimal() +
  labs(title = "King Mackerel Locations - Day: {frame_time}") +
  transition_time(Time)

# Animate and preview
anim <- animate(p, nframes = max(all_dat$Time), fps = 10, width = 800, height = 600, 
                renderer = gifski_renderer())

# Save the animation as a .gif file
anim_save("animal_locations.gif", animation = anim)

## Growth Analyses ---
df_relative <- all_dat %>%
  group_by(Species, Individual) %>%
  mutate(initial_length = Length[which.min(Time)],  # Or `Time == 1` if that's explicit
         relative_length = Length / initial_length)


new_dat = df_relative %>% group_by(Time) %>% summarize(Rel = median(relative_length))

ggplot(new_dat) + geom_point(aes(x=Time,y=Rel))+
  theme_classic()
