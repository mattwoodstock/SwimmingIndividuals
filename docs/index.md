SwimmingIndividuals.jl v1.0 User Guide
This user guide provides a comprehensive overview of the SwimmingIndividuals.jl model, detailing the setup, execution, and the purpose of its various components.

Page 1: Main Model Script (model.jl)
This is the main entry point for running a SwimmingIndividuals simulation. This script is responsible for loading all necessary packages and source code files, reading input data from CSVs, setting up the model parameters, and launching the simulation loop.

Overview of model.jl
The script performs the following key tasks in order:

Package Loading: It loads all required Julia packages, including PlanktonIndividuals.jl, CUDA.jl for GPU support, and various data handling and analysis libraries.

Include Source Files: It includes all the custom Julia source code files from the src/ directory, making functions like generate_environment!, TimeStep!, etc., available.

Load Input Data: It reads the files.csv manifest to get the paths to all other input data files, such as agent traits, environmental preferences, and fishery regulations.

Set Parameters: It parses the main parameters file to configure the simulation, including the number of species, the number of timesteps, and the computational architecture (CPU or GPU).

Initialize Model Components: It calls the high-level functions to generate the environment, calculate habitat capacities, and create the initial populations of both focal species and resource grids.

Run Simulation: It assembles the final MarineModel and MarineSimulation objects and passes them to the runSI function to start the time-stepping loop.

Full Code Example
using PlanktonIndividuals, Distributions, Random, CSV, DataFrames, StructArrays,Statistics,Dates,Optim,LinearAlgebra, Tables, CUDA, LoopVectorization, NCDatasets,StaticArrays,Interpolations, DelimitedFiles, StatsBase,Plots, Distributions, NearestNeighbors, QuadGK,Printf, HDF5, NCDatasets

#using Profile, ProfileView, Cthulhu, BenchmarkTools #Only use for benchmarking purposes
using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions
using KernelAbstractions: @kernel, @index
using CUDA: @atomic, atomic_cas!, atomic_sub!

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

## Load in necessary databases
files = CSV.read("inputs/files.csv",DataFrame) #All files needed in the model. Collected like this so that this can be passed through without passing each individual dataframe. 

trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1],DataFrame)))) #Database of IBM species traits 
resource_trait = CSV.read(files[files.File .== "resource_trait",:Destination][1],DataFrame)
params = CSV.read(files[files.File .== "params",:Destination][1],DataFrame) #Database of state variables
grid = CSV.read(files[files.File .== "grid",:Destination][1],DataFrame) #Database of grid variables
fisheries = CSV.read(files[files.File .== "fisheries",:Destination][1],DataFrame)
envi_file = files[files.File .== "environment",:Destination][1]

## Convert values to match proper structure
Nsp = parse(Int64,params[params.Name .== "numspec", :Value][1]) #Number of IBM species
Nresource = parse(Int64,params[params.Name .== "numresource", :Value][1]) #Number of IBM species
output_dt = parse(Int64,params[params.Name .== "output_dt", :Value][1]) #Number of pooled species/groups
spinup = parse(Int64,params[params.Name .== "spinup", :Value][1]) #Number of timesteps for a burn-in
plt_diags = parse(Int64,params[params.Name .== "plt_diags", :Value][1]) #Number of timesteps for a burn-in

maxN = 500000  # Placeholder where the maximum number of individuals created is simply the maximum abundance
arch_str = params[params.Name .== "architecture", :Value][1] #Architecture to use.

if arch_str == "GPU"
    if CUDA.functional()
        arch = GPU()
        println("✅ Architecture successfully set to GPU.")
    else
        # Fallback to CPU if CUDA is not available or functional
        @warn "GPU specified but CUDA is not functional. Falling back to CPU."
        arch = CPU()
    end
elseif arch_str == "CPU"
    arch = CPU()
    println("✅ Architecture successfully set to CPU.")
else
    # Default to CPU if the setting is unrecognized
    @warn "Architecture '$arch_str' not recognized. Defaulting to CPU."
    arch = CPU()
end

t = 0.0 #Initial time

n_iteration = parse(Int,params[params.Name .== "nts", :Value][1]) #Number of model iterations (i.e., timesteps) to run
dt = parse(Int,params[params.Name .== "model_dt", :Value][1]) #minutes per time step. Keep this at one.
n_iters = parse(Int16,params[params.Name .== "n_iter", :Value][1]) #Number of iterations to run

#Create environment struct
envi = generate_environment!(arch, envi_file,plt_diags)

#Create Depth Struct and Carry Through Grid
depths = generate_depths(files)

capacities = initial_habitat_capacity(envi,Nsp,Nresource,files,arch,plt_diags) #3D array (x,y,spec)

for iter in 1:n_iters
    B = trait[:Biomass][1:Nsp] #Vector of IBM abundances for all species

    ## Create individuals
    inds = generate_individuals(trait, arch, Nsp, B, maxN,depths,capacities,dt,envi)

    resources = initialize_resources(resource_trait,Nsp,Nresource,depths,capacities,arch)

    #Load in fisheries data
    fishery = load_fisheries(fisheries)

    init_abund = fill(0,Nsp)
    bioms = fill(0.0,Nsp)

    for sp in 1:Nsp
        init_abund[sp] = sum(inds.animals[sp].data.abundance)
        bioms[sp] = sum(inds.animals[sp].data.biomass_school)
    end

    ## Create model object
    model = MarineModel(arch,envi,depths, fishery, t, 0,dt, inds,resources,resource_trait, capacities,maxN, Nsp,Nresource,init_abund,bioms,init_abund, files, output_dt,spinup)

    ##Set up outputs in the simulation
    outputs = generate_outputs(model)

    # Set up simulation parameters
    sim = MarineSimulation(model, dt, n_iteration,iter,outputs)

    # Run model. Currently this is very condensed, but I kept the code for when we work with environmental factors
    runSI(sim)
    #reset_run
