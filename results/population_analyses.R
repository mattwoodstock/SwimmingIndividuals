#### Example Population analyses for a SwimmingIndividuals model
####### Developer: Matt Woodstock

## Load Libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(raster)
library(marmap)
library(gganimate)
library(hdf5r)

indir = "./Population"
fishdir = "./Fishery"
files = list.files(indir)
fish_files = list.files(fishdir)


## Natural mortality maps
file <- H5File$new("Instantaneous_Mort_1-1.jld", mode = "r")

# Read the dataset
M <- file[["M"]][]
file$close_all()

## Fishing mortality maps
