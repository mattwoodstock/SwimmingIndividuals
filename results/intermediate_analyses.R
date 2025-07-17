library(ggplot2)
library(dplyr)
library(tidyr)

# Inputs
indir = "./inputs_for_plotting"

# Reproduction trend
repro = read.csv(paste0(indir,"/reproduction.csv")) %>% 
  pivot_longer(cols = 2:ncol(repro),names_to = "Month")

repro$Month = factor(repro$Month,levels=c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"))

repro_plot = ggplot(repro,aes(x=Month,y=value,fill=value))+
  geom_bar(stat="identity",position = position_dodge())+
  theme_classic()+
  labs(title="Reproduction Seasonality",y="Seasonal Spawning Index")

filename = "reproduction_seasonality.png"
ggsave(filename,repro_plot,height=unit(3,"in"),width=unit(7,"in"))
