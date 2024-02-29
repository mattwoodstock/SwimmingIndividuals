using PlanktonIndividuals, Distributions, Random, CSV, DataFrames, StructArrays, JLD2,Plots,Statistics,Dates,Optim,LinearAlgebra, Tables
using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions: @kernel, @index


include("utilities.jl")
include("create.jl")
include("particles.jl")
include("environment.jl")
include("diagnostics.jl")
include("simulation.jl")
include("update.jl")
include("output.jl")
include("behavior.jl")
include("movement.jl")
include("predation.jl")
include("mortality.jl")
include("energy.jl")
include("timestep.jl")
include("plotting.jl")
include("analysis.jl")

start = now()
## Load in necessary databases

files = CSV.read("files.csv",DataFrame) #All files needed in the model. Collected like this so that this can be passed through without passing each individual dataframe. 

trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1],DataFrame)))) #Database of IBM species traits 

pool_trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "nonfocal_trait",:Destination][1],DataFrame)))) #Database of pooled species traits
state = CSV.read(files[files.File .== "state",:Destination][1],DataFrame) #Database of state variables
grid = CSV.read(files[files.File .== "grid",:Destination][1],DataFrame) #Database of grid variables
#tracer = Dict(pairs(eachcol(CSV.read(files[files.File .== "tracers",:Destination][1],DataFrame)))) #Database of state variables for particles

#=
envi = #Database of environmental parameters. Likely .nc files, but could be another function
=#
## Convert values to match proper structure
Nsp = parse(Int64,state[state.Name .== "numspec", :Value][1]) #Number of IBM species
Npool = parse(Int64,state[state.Name .== "numpool", :Value][1]) #Number of pooled species/groups
MaxParticle = parse(Int64,state[state.Name .== "maxparticle", :Value][1]) #Number of pooled species/groups

Nall = Nsp + Npool #Number of all groups
N = trait[:Abundance] #Vector of IBM abundances for all species
maxN = maximum(N) # Placeholder where the maximum number of individuals created is simply the maximum abundance
arch = CPU() #Architecure to use. Currently the only one that works. Will want to add a GPU() component
t = 0.0 #Intial time
n_iteration = parse(Int64,state[state.Name .== "nts", :Value][1]) #Number of model iterations (i.e., timesteps) to run
dt = 1.0 #minutes per time step. Keep this at one.
dimension = parse(Int64,state[state.Name .== "dimensions", :Value][1])

## Create Output grid
g = RectilinearGrid(size=(grid[grid.Name.=="latres",:Value][1],grid[grid.Name.=="lonres",:Value][1],grid[grid.Name.=="depthres",:Value][1]), landmask = nothing, x = (grid[grid.Name.=="latmin",:Value][1], grid[grid.Name.=="latmax",:Value][1]), y = (grid[grid.Name.=="lonmin",:Value][1],grid[grid.Name.=="lonmax",:Value][1]), z = (0,-1*grid[grid.Name.=="depthmax",:Value][1]))

## Create individuals
### Focal species
inds = generate_individuals(trait, arch, Nsp, N, maxN, g::AbstractGrid,"z_distributions_night.csv")

### Nonfocal species/groups
pooled = generate_pools(arch, pool_trait, Npool, g::AbstractGrid,"z_pool_distributions_night.csv",grid)
### Create tracers
#tracers = generate_tracer(tracer, arch)

## Create model object
model = MarineModel(arch, t, 0, inds, pooled, Nsp, Npool, N, g, dimension,files)

## Create environmental parameters for now
temp = generate_temperature(grid[grid.Name.=="depthmax",:Value][1])

##Set up outputs in the simulation
outputs = generate_outputs(model,Nall,grid[grid.Name.=="depthres",:Value][1], n_iteration)

#May want a function to combine all necessary outputs
#Could then combine all outputs into one struct to pass through. Similar to @individuals.

## Set up diagnostics (Rework once model runs)
#diags = MarineDiagnostics(model; tracer=(:PAR, :NH4, :NO3, :DOC), plankton = (:num, :graz, :mort, :dvid, :PS, :BS, :Chl), iteration_interval = 1)

# Set up simulation parameters
sim = simulation(model,dt,n_iteration,temp,outputs)

# Set up output writer
#sim.output_writer = MarineOutputWriter(save_plankton=true)

# Run model. Currently this is very condensed, but I kept the code for when we work with environmental factors
update!(sim)

stop = now()

println(stop-start)
#Full results
#results!(model,outputs)

#Create plots
#food_web_plot(outputs,model,dt)
#plot_depths(sim)

#= 
x,y,reshaped_z = reshape_for_heatmap(sim.depth_dens)

println(x)
println(y)
println(reshaped_z)
Plot = heatmap(x,y,reshaped_z,c=:blues,xlabel="Time (N timesteps: $(Int(dt)) resolution)",ylabel="Depth (m)",title="Depth by Time",yflip=true)
display(Plot)
=#

#println("Works") #Tells me model works