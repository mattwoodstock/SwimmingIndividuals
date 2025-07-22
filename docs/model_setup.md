## Model Setup & Initialization

This is the main entry point for running a SwimmingIndividuals simulation. This script is responsible for loading all necessary packages and source code files, reading input data from CSVs, setting up the model parameters, and launching the simulation loop.

### Overview

The script performs the following key tasks in order:

1.  **Package Loading:** It loads all required Julia packages, including `PlanktonIndividuals.jl`, `CUDA.jl` for GPU support, and various data handling and analysis libraries.
2.  **Include Source Files:** It includes all the custom Julia source code files from the `src/` directory, making functions like `generate_environment!`, `TimeStep!`, etc., available.
3.  **Load Input Data:** It reads the `files.csv` manifest to get the paths to all other input data files, such as agent traits, environmental preferences, and fishery regulations.
4.  **Set Parameters:** It parses the main parameters file to configure the simulation, including the number of species, the number of timesteps, and the computational architecture (CPU or GPU).
5.  **Initialize Model Components:** It calls the high-level functions to generate the environment, calculate habitat capacities, and create the initial populations of both focal species and resource grids.
6.  **Run Simulation:** It assembles the final `MarineModel` and `MarineSimulation` objects and passes them to the `runSI` function to start the time-stepping loop.

### Full Code Example

```julia
# --- 1. Load All Necessary Packages ---
using PlanktonIndividuals, Distributions, Random, CSV, DataFrames, StructArrays,Statistics,Dates,Optim,LinearAlgebra, Tables, CUDA, LoopVectorization, NCDatasets,StaticArrays,Interpolations, DelimitedFiles, StatsBase,Plots, Distributions, NearestNeighbors, QuadGK,Printf, HDF5, NCDatasets

#using Profile, ProfileView, Cthulhu, BenchmarkTools #Only use for benchmarking purposes
using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions
using KernelAbstractions: @kernel, @index
using CUDA: @atomic, atomic_cas!, atomic_sub!

# --- 2. Include All Model Source Code Files ---
# This makes the custom functions defined in the /src directory available to this script.
include("src/utilities.jl")
include("src/create.jl")
include("src/environment.jl")
include("src/simulation.jl")
include("src/output.jl")
include("src/behavior.jl")
include("src/movement.jl")
include("src/predation.jl")
include("src/mortality.jl")
include("src/fisheries.jl")
include("src/energy.jl")
include("src/timestep.jl")
include("src/analysis.jl")
include("src/update.jl")

# --- 3. Load Input Data from CSV Files ---
# The `files.csv` acts as a manifest, pointing to all other input files.
files = CSV.read("inputs/files.csv",DataFrame) 

# Load the specific data files using the paths from the manifest
trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1],DataFrame)))) 
resource_trait = CSV.read(files[files.File .== "resource_trait",:Destination][1],DataFrame)
params = CSV.read(files[files.File .== "params",:Destination][1],DataFrame) 
grid = CSV.read(files[files.File .== "grid",:Destination][1],DataFrame) 
fisheries = CSV.read(files[files.File .== "fisheries",:Destination][1],DataFrame)
envi_file = files[files.File .== "environment",:Destination][1]

# --- 4. Parse and Set Up Simulation Parameters ---
# Extract key parameters from the loaded `params` DataFrame
Nsp = parse(Int64,params[params.Name .== "numspec", :Value][1])         # Number of focal species
Nresource = parse(Int64,params[params.Name .== "numresource", :Value][1]) # Number of resource species
output_dt = parse(Int64,params[params.Name .== "output_dt", :Value][1])   # Frequency of output (in timesteps)
spinup = parse(Int64,params[params.Name .== "spinup", :Value][1])        # Number of timesteps for model burn-in
plt_diags = parse(Int64,params[params.Name .== "plt_diags", :Value][1])   # Boolean (1 or 0) to control diagnostic plots

# Set the maximum number of agents per species. This is critical for memory allocation.
maxN = 500000
# Determine the computational architecture (CPU or GPU) from the parameters file
arch_str = params[params.Name .== "architecture", :Value][1]

# Logic to select the architecture and provide user feedback
if arch_str == "GPU"
    if CUDA.functional()
        arch = GPU()
        println("✅ Architecture successfully set to GPU.")
    else
        @warn "GPU specified but CUDA is not functional. Falling back to CPU."
        arch = CPU()
    end
elseif arch_str == "CPU"
    arch = CPU()
    println("✅ Architecture successfully set to CPU.")
else
    @warn "Architecture '$arch_str' not recognized. Defaulting to CPU."
    arch = CPU()
end

# Initialize model time and timestep parameters
t = 0.0 # Initial time
n_iteration = parse(Int,params[params.Name .== "nts", :Value][1]) # Total number of timesteps to run
dt = parse(Int,params[params.Name .== "model_dt", :Value][1])   # Duration of a single timestep in minutes
n_iters = parse(Int16,params[params.Name .== "n_iter", :Value][1]) # Number of replicate simulation runs

# --- 5. Initialize Model Components ---
# Create the environment object from the NetCDF file
envi = generate_environment!(arch, envi_file,plt_diags)

# Create the depth information struct
depths = generate_depths(files)

# Calculate the habitat capacity maps for all species and months
capacities = initial_habitat_capacity(envi,Nsp,Nresource,files,arch,plt_diags)

# --- 6. Main Simulation Loop ---
# This loop runs multiple replicate simulations (controlled by n_iters)
for iter in 1:n_iters
    # Get the initial biomass targets for each focal species
    B = trait[:Biomass][1:Nsp]

    # Create the focal species agents and place them in the environment
    inds = generate_individuals(trait, arch, Nsp, B, maxN,depths,capacities,dt,envi)

    # Initialize the resource biomass grids
    resources = initialize_resources(resource_trait,Nsp,Nresource,depths,capacities,arch)

    # Load the fishery regulations and data
    fishery = load_fisheries(fisheries)

    # Initialize vectors to track population-level statistics
    init_abund = fill(0,Nsp)
    bioms = fill(0.0,Nsp)

    for sp in 1:Nsp
        init_abund[sp] = sum(inds.animals[sp].data.abundance)
        bioms[sp] = sum(inds.animals[sp].data.biomass_school)
    end

    # Assemble the complete model object containing the full simulation state
    model = MarineModel(arch,envi,depths, fishery, t, 0,dt, inds,resources,resource_trait, capacities,maxN, Nsp,Nresource,init_abund,bioms,init_abund, files, output_dt,spinup)

    # Create the output arrays for storing results
    outputs = generate_outputs(model)

    # Create the simulation object that bundles the model and run parameters
    sim = MarineSimulation(model, dt, n_iteration,iter,outputs)

    # Launch the simulation
    runSI(sim)
end
```