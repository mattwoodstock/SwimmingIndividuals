using PlanktonIndividuals, Distributions, Random, CSV, DataFrames, StructArrays, JLD2,Statistics,Dates,Optim,LinearAlgebra, Tables, CUDA, LoopVectorization, NCDatasets
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
include("src/energy.jl")
include("src/timestep.jl")
include("src/analysis.jl")

## Load in necessary databases
files = CSV.read("files.csv",DataFrame) #All files needed in the model. Collected like this so that this can be passed through without passing each individual dataframe. 

trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1],DataFrame)))) #Database of IBM species traits 
pool_trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "nonfocal_trait",:Destination][1],DataFrame)))) #Database of pooled species traits
state = CSV.read(files[files.File .== "state",:Destination][1],DataFrame) #Database of state variables
grid = CSV.read(files[files.File .== "grid",:Destination][1],DataFrame) #Database of grid variables

## Convert values to match proper structure
Nsp = parse(Int64,state[state.Name .== "numspec", :Value][1]) #Number of IBM species
Npool = parse(Int64,state[state.Name .== "numpool", :Value][1]) #Number of pooled species/groups
output_dt = parse(Int64,state[state.Name .== "output_dt", :Value][1]) #Number of pooled species/groups
spinup = parse(Int64,state[state.Name .== "spinup", :Value][1]) #Number of pooled species/groups

Nall = Nsp + Npool #Number of all groups

maxN = 1 # Placeholder where the maximum number of individuals created is simply the maximum abundance
arch = CPU() #Architecture to use.
t = 0.0 #Initial time
n_iteration = parse(Int64,state[state.Name .== "nts", :Value][1]) #Number of model iterations (i.e., timesteps) to run
dt = 60.0 #minutes per time step. Keep this at one.
n_iters = 365

## Create Output grid
g = RectilinearGrid(size=(grid[grid.Name.=="latres",:Value][1],grid[grid.Name.=="lonres",:Value][1],grid[grid.Name.=="depthres",:Value][1]), landmask = nothing, x = (grid[grid.Name.=="latmin",:Value][1], grid[grid.Name.=="latmax",:Value][1]), y = (grid[grid.Name.=="lonmin",:Value][1],grid[grid.Name.=="lonmax",:Value][1]), z = (0,-1*grid[grid.Name.=="depthmax",:Value][1]))
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
envi = generate_environment(files)

for iter in 1:n_iters
    B = trait[:Biomass][1:Nsp] #Vector of IBM abundances for all species

    ## Create individuals
    ### Focal species
    inds = generate_individuals(trait, arch, Nsp, B, maxN, g::AbstractGrid, files)

    ### Nonfocal species/groups
    pooled = generate_pools(arch, pool_trait, Npool, g::AbstractGrid, files,maxN,dt,envi)
    ## Create model object
    init_abund = fill(0,Nsp)

    model = MarineModel(arch,envi, t, 0,dt, inds, pooled,maxN, Nsp, Npool, B,init_abund, g, files, output_dt, cell_size,spinup)

    for i in 1:Nsp
        model.abund[i] = length(model.individuals.animals[i].data.x)
    end

    ##Set up outputs in the simulation
    outputs = generate_outputs(model, Nall, n_iteration, output_dt)

    # Set up simulation parameters
    sim = MarineSimulation(model, dt, n_iteration,iter,outputs)

    # Run model. Currently this is very condensed, but I kept the code for when we work with environmental factors
    update!(sim)

    #reset_run(sim)
end

#Full results
#results!(model,outputs)

#Create plots
#food_web_plot(outputs,model,dt)
#plot_depths(sim)