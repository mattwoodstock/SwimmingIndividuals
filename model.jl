using PlanktonIndividuals, Distributions, Random, CSV, DataFrames, StructArrays, JLD2,Statistics,Dates,Optim,LinearAlgebra, Tables, CUDA, LoopVectorization, NCDatasets,StaticArrays,Interpolations, DelimitedFiles, StatsBase,Plots, Distributions
using Profile, ProfileView, Cthulhu #Only use for benchmarking purposes
using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions: @kernel, @index

include("src/utilities.jl")
include("src/create.jl")
include("src/environment.jl")
include("src/simulation.jl")
include("src/update.jl")
include("src/output.jl")
include("src/behavior.jl")
include("src/movement.jl")
include("src/predation.jl")
include("src/mortality.jl")
include("src/fisheries.jl")
include("src/energy.jl")
include("src/timestep.jl")
include("src/analysis.jl")

## Load in necessary databases
files = CSV.read("files.csv",DataFrame) #All files needed in the model. Collected like this so that this can be passed through without passing each individual dataframe. 

trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1],DataFrame)))) #Database of IBM species traits 
pool_trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "nonfocal_trait",:Destination][1],DataFrame)))) #Database of pooled species traits
state = CSV.read(files[files.File .== "state",:Destination][1],DataFrame) #Database of state variables
grid = CSV.read(files[files.File .== "grid",:Destination][1],DataFrame) #Database of grid variables
fisheries = CSV.read(files[files.File .== "fisheries",:Destination][1],DataFrame)

## Convert values to match proper structure
Nsp = parse(Int64,state[state.Name .== "numspec", :Value][1]) #Number of IBM species
Npool = parse(Int64,state[state.Name .== "numpool", :Value][1]) #Number of pooled species/groups
output_dt = parse(Int64,state[state.Name .== "output_dt", :Value][1]) #Number of pooled species/groups
spinup = parse(Int64,state[state.Name .== "spinup", :Value][1]) #Number of pooled species/groups

Nall = Nsp + Npool #Number of all groups

maxN = 1 # Placeholder where the maximum number of individuals created is simply the maximum abundance
arch = CPU() #Architecture to use.
t = 0.0 #Initial time
n_iteration = parse(Int,state[state.Name .== "nts", :Value][1]) #Number of model iterations (i.e., timesteps) to run
dt = parse(Int,state[state.Name .== "model_dt", :Value][1]) #minutes per time step. Keep this at one.
n_iters = parse(Int16,state[state.Name .== "n_iter", :Value][1]) #Number of iterations to run

## Create Output grid
latres = grid[grid.Name .== "latres", :Value][1]
lonres = grid[grid.Name .== "lonres", :Value][1]
depthres = grid[grid.Name .== "depthres", :Value][1]

latmin = grid[grid.Name .== "latmin", :Value][1]
latmax = grid[grid.Name .== "latmax", :Value][1]
lonmin = grid[grid.Name .== "lonmin", :Value][1]
lonmax = grid[grid.Name .== "lonmax", :Value][1]
depthmax = grid[grid.Name .== "depthmax", :Value][1]

n_lat = Int(round((latmax - latmin) / latres))
n_lon = Int(round((lonmax - lonmin) / lonres))
n_depth = Int(round(depthmax / depthres))

g = RectilinearGrid(size = (n_lat, n_lon, n_depth),landmask = nothing,x = (latmin, latmax),y = (lonmin, lonmax),z = (0.0, -depthmax))

maxdepth = grid[grid.Name .== "depthmax", :Value][1]
depthres = grid[grid.Name .== "depthres", :Value][1]
lonmax = grid[grid.Name .== "lonmax", :Value][1]
lonmin = grid[grid.Name .== "lonmin", :Value][1]
latmax = grid[grid.Name .== "latmax", :Value][1]
latmin = grid[grid.Name .== "latmin", :Value][1]
lonres = grid[grid.Name .== "lonres", :Value][1]
latres = grid[grid.Name .== "latres", :Value][1]

cell_size = ((latmax-latmin)/latres) * ((lonmax-lonmin)/lonres) * (maxdepth/depthres) #cubic meters of water in each grid cell

#Create environment struct
envi = generate_environment!()

#Create Depth Struct and Carry Through Grid
depths = generate_depths(files)

capacities = initial_habitat_capacity(envi,Nsp,files) #3D array (x,y,spec)
pool_capacities = pool_habitat_capacity(envi,Npool,files)
for iter in 1:n_iters
    B = trait[:Biomass][1:Nsp] #Vector of IBM abundances for all species
  
    ## Create individuals
    ### Focal species
    inds = generate_individuals(trait, arch, Nsp, B, maxN,depths,capacities,grid)

    ### Nonfocal species/groups
    pooled = generate_pools(arch, pool_trait, Npool, g::AbstractGrid,maxN,dt,pool_capacities,grid,depths)

    #Load in fisheries data
    fishery = load_fisheries(fisheries)

    ## Create model object
    init_abund = fill(0,Nsp)
    

    model = MarineModel(arch,envi,depths, fishery, t, 0,dt, inds, pooled, capacities,pool_capacities,maxN, Nsp, Npool, B,init_abund, g, files, output_dt, cell_size,spinup)

    for i in 1:Nsp
        model.abund[i] = length(model.individuals.animals[i].data.x)
    end

    println(model.abund)

    ##Set up outputs in the simulation
    outputs = generate_outputs(model, Nall, n_iteration, output_dt)

    # Set up simulation parameters
    sim = MarineSimulation(model, dt, n_iteration,iter,outputs)

    # Run model. Currently this is very condensed, but I kept the code for when we work with environmental factors
    update!(sim)
    #reset_run(sim)
    
end