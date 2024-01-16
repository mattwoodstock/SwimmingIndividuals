using PlanktonIndividuals, Distributions, Random, CSV, DataFrames, StructArrays, JLD2,Plots,Statistics,Dates, BenchmarkTools, Profile, ProfileView

using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions: @kernel, @index


include("utilities.jl")
include("create.jl")
include("environment.jl")
include("diagnostics.jl")
include("simulation.jl")
include("update.jl")
include("output.jl")
include("movement.jl")
include("predation.jl")
include("mortality.jl")
include("energy.jl")
include("timestep.jl")
include("plotting.jl")


## Load in necessary databases
cd("D:/SwimmingIndividuals/Adapted")
trait = Dict(pairs(eachcol(CSV.read("traits.csv",DataFrame)))) #Database of IBM species traits 
pool_trait = Dict(pairs(eachcol(CSV.read("pooled_traits.csv",DataFrame)))) #Database of pooled species traits
state = CSV.read("state.csv",DataFrame) #Database of state variables
grid = CSV.read("grid.csv",DataFrame) #Database of grid variables

#=
envi = #Database of environmental parameters. Likely .nc files, but could be another function
=#

## Convert values to match proper structure
Nsp = parse(Int64,state[state.Name .== "numspec", :Value][1]) #Number of IBM species
Npool = parse(Int64,state[state.Name .== "numpool", :Value][1]) #Number of pooled species/groups
Nall = Nsp + Npool #Number of all groups
N = trait[:Abundance] #Vector of IBM abundances for all species
maxN = maximum(N) # Placeholder where the maximum number of individuals created is simply the maximum abundance
arch = CPU() #Architecure to use. Currently the only one that works. Will want to add a GPU() component
t = 0.0 #Intial time
n_iteration = parse(Int64,state[state.Name .== "nts", :Value][1]) #Number of model iterations (i.e., timesteps) to run
dt = 20.0 #minutes per time step


## Create Output grid
g = RectilinearGrid(size=(grid[grid.Name.=="latres",:Value][1],grid[grid.Name.=="lonres",:Value][1],grid[grid.Name.=="depthres",:Value][1]), landmask = nothing, x = (grid[grid.Name.=="latmin",:Value][1], grid[grid.Name.=="latmax",:Value][1]), y = (grid[grid.Name.=="lonmin",:Value][1],grid[grid.Name.=="lonmax",:Value][1]), z = (0,-1*grid[grid.Name.=="depthmax",:Value][1]))

## Create individuals
###IBM species
inds = generate_individuals(trait, arch, Nsp, N, maxN, g::AbstractGrid,"z_distributions_night.csv")
###Pooled species/groups
pooled = generate_pools(arch, pool_trait, Npool, g::AbstractGrid,"z_pool_distributions_night.csv",grid)

## Create model object
model = MarineModel(arch, t, 0, inds, pooled, Nsp, Npool, N, g, 1)

## Create environmental parameters for now
temp = generate_temperature()

#input = MarineInput(temp)

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

#Full results
results!(model,outputs)

#Create plots
food_web_plot(outputs,model,dt)

#= 
x,y,reshaped_z = reshape_for_heatmap(sim.depth_dens)

println(x)
println(y)
println(reshaped_z)
Plot = heatmap(x,y,reshaped_z,c=:blues,xlabel="Time (N timesteps: $(Int(dt)) resolution)",ylabel="Depth (m)",title="Depth by Time",yflip=true)
display(Plot)
=#

println("Works") #Tells me model works