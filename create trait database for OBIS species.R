library(rfishbase)
library(dplyr)
library(tidyverse)
setwd("D:/SwimmingIndividuals/Adapted")

species <- read.csv("Species Traits.csv")


dat <- data.frame("SpeciesLong" = character(), "SpeciesShort" = character(), "Habitat" = character(), "Taxon" = character(),	"Abundance" = numeric(),	"Assimilation_eff" = numeric(),	"LF_dist" = character(),	"LF_dist1"	= numeric(), "LF_dist2" = numeric(),	"LWR_a" = numeric(),	"LWR_b" = numeric(),	"VBG_K" = numeric(),	"VBG_LOO" = numeric(),	"VBG_t0"=numeric(),	"Mat_age" = numeric(),	"Tmax" = numeric(),	"M_type" = character(),	"M_const" = numeric(),	"Sex_rat" = numeric(),	"Swim_velo" = numeric())

pb <- txtProgressBar(min=0,max=nrow(species),style=3)
for (i in 1:nrow(species)){
  fish <- fb_tbl("species", "fishbase")
  sub_fish <- fish[fish$Genus %in% species$Genus[i] & fish$Species %in% species$Species[i],]
  
  long_name <- paste(species$Genus[i],species$Species[i],sep="_")
  short_name <- paste("sp",i,sep="")
  
  if (nrow(sub_fish) > 0){

    ## Habitat
    eco <- ecology(species_list = paste(species$Genus[i]," ",species$Species[i],sep=""))
    if (nrow(eco) > 0){
      hab <- apply(eco, 1, function(x) paste(colnames(eco)[which(x == -1)], collapse = ", "))
    } else {
      hab <- NA
    }
    
    #Growth
    growth <- rfishbase::popgrowth(species_list = paste(species$Genus[i]," ",species$Species[i],sep=""))
    
    if (nrow(growth) > 0){
      k <- mean(na.omit(growth$K))
      linf <- mean(na.omit(growth$Loo))
      t0 <- mean(na.omit(growth$to))
      tmax <- mean(na.omit(growth$tmax))
      condition <- mean(na.omit(growth$c))
      power <- mean(na.omit(growth$b))
      age_mat <- mean(na.omit(growth$tm))
    } else {
      k <- NA
      linf <- NA
      t0 <- NA
      tmax <- NA
      condition <- NA
      power <- NA
      age_mat <- NA
    }
    
    
    
    swim <- rfishbase::speed(species_list = paste(species$Genus[i]," ",species$Species[i],sep=""))
    
    if (nrow(swim) > 0){
      
      swspeed <- mean(na.omit(swim$SpeedLS))
      
    } else {
      swspeed <- NA
    } 
      
      
      dat <- dat %>% add_row("SpeciesLong" = long_name, "SpeciesShort" = short_name, "Habitat" = hab, "Taxon" = "",	"Abundance" = 10,	"Assimilation_eff" = 0.8,	"LF_dist" = "constant",	"LF_dist1"	= 0, "LF_dist2" = 10,	"LWR_a" = condition,	"LWR_b" = power,	"VBG_K" = k,	"VBG_LOO" = linf,	"VBG_t0"= t0,	"Mat_age" = age_mat,	"Tmax" = tmax,	"M_type" = "constant",	"M_const" = 0,	"Sex_rat" = 0.5,	"Swim_velo" = swspeed)
      
  } else {
    sealife <- fb_tbl("species", "sealifebase")
    sub_sea <- sealife[sealife$Genus %in% species$Genus[i] & sealife$Species %in% species$Species[i],] %>% as.data.frame()
    
    if (nrow(sub_sea) > 0){
      
      spec_code <- sub_sea$SpecCode[1]
    
      ## Habitat
      eco <- fb_tbl("ecology", "sealifebase") %>% filter(SpecCode %in% spec_code)
      if (nrow(eco) > 0){
        hab <- apply(eco, 1, function(x) paste(colnames(eco)[which(x == 1)], collapse = ", "))
      } else {
        hab <- NA
      }
      
      #Growth
      
      growth <- fb_tbl("popgrowth", "sealifebase") %>% filter(SpecCode %in% spec_code)
      
      if (nrow(growth) > 0){
        k <- mean(na.omit(growth$K))
        linf <- mean(na.omit(growth$Loo))
        t0 <- mean(na.omit(growth$to))
        tmax <- mean(na.omit(growth$tmax))
        condition <- mean(na.omit(growth$c))
        power <- mean(na.omit(growth$b))
        age_mat <- mean(na.omit(growth$tm))
      } else {
        k <- NA
        linf <- NA
        t0 <- NA
        tmax <- NA
        condition <- NA
        power <- NA
        age_mat <- NA
      }
        
        dat <- dat %>% add_row("SpeciesLong" = long_name, "SpeciesShort" = short_name, "Habitat" = hab, "Taxon" = "",	"Abundance" = 10,	"Assimilation_eff" = 0.8,	"LF_dist" = "constant",	"LF_dist1"	= 0, "LF_dist2" = 10,	"LWR_a" = condition,	"LWR_b" = power,	"VBG_K" = k,	"VBG_LOO" = linf,	"VBG_t0"= t0,	"Mat_age" = age_mat,	"Tmax" = tmax,	"M_type" = "constant",	"M_const" = 0,	"Sex_rat" = 0.5,	"Swim_velo" = NA)
        
        
      } else {
        hab <- NA
        k <- NA
        linf <- NA
        t0 <- NA
        tmax <- NA
        condition <- NA
        power <- NA
        age_mat <- NA
        swspeed <- NA
        
        
          dat <- dat %>% add_row("SpeciesLong" = long_name, "SpeciesShort" = short_name, "Habitat" = hab, "Taxon" = "",	"Abundance" = 10,	"Assimilation_eff" = 0.8,	"LF_dist" = "constant",	"LF_dist1"	= 0, "LF_dist2" = 10,	"LWR_a" = condition,	"LWR_b" = power,	"VBG_K" = k,	"VBG_LOO" = linf,	"VBG_t0"= t0,	"Mat_age" = age_mat,	"Tmax" = tmax,	"M_type" = "constant",	"M_const" = 0,	"Sex_rat" = 0.5,	"Swim_velo" = NA)
          ## All NAs
      
      }
    
  }
  setTxtProgressBar(pb,i)
}

write.csv(dat,"All Traits.csv")

