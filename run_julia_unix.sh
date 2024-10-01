#!/bin/bash
#SBATCH --partition=compute         # Queue selection
#SBATCH --job-name=HD       # Job name
#SBATCH --mail-type=ALL             # Mail events (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matthew.woodstock@whoi.edu  # Where to send mail
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=20	# Nubmer of CPU cores per task
#SBATCH --mem=100gb                   # Job memory request
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=serial_job_%j.err  # Standard error
#SBATCH --output=serial_job_%j.out  # Standard output
 
date
 
module load julia                  # Load the julia module
 
echo "Running julia script for SwimmingIndividuals.jl"
 
julia /vortexfs1/scratch/matthew.woodstock/SwimmingIndividuals/GoM/Scenarios/Daily/Historical/model.jl
 
date
