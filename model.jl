using PlanktonIndividuals, Random, CSV, DataFrames, StructArrays, JLD2

using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions: @kernel, @index


include("utilities.jl")
include("create.jl")
include("diagnostics.jl")
include("simulation.jl")
include("update.jl")
include("output.jl")
include("movement.jl")
include("timestep.jl")


## Load in necessary databases
cd("D:/SwimmingIndividuals/Adapted")
trait = Dict(pairs(eachcol(CSV.read("traits.csv",DataFrame))))
state = CSV.read("state.csv",DataFrame)
grid = CSV.read("grid.csv",DataFrame)

## Convert values to match proper structure
Nsp = parse(Int64,state[state.Name .== "numspec", :Value][1])
N = trait[:Abundance]
maxN = maximum(N) # Placeholder where the maximum number of individuals created is simply the maximum abundance
arch = CPU() #Architecure to use
t = 0.0 #Time in seconds (Will need to adjust)
n_iteration = parse(Int64,state[state.Name .== "nts", :Value][1]) #Number of iterations
dt = 10.0 #minutes per time step



## Create Output grid
g = RectilinearGrid(size=(grid[grid.Name.=="latres",:Value][1],grid[grid.Name.=="lonres",:Value][1],grid[grid.Name.=="depthres",:Value][1]), landmask = nothing, x = (grid[grid.Name.=="latmin",:Value][1], grid[grid.Name.=="latmax",:Value][1]), y = (grid[grid.Name.=="lonmin",:Value][1],grid[grid.Name.=="lonmax",:Value][1]), z = (0,-1*grid[grid.Name.=="depthmax",:Value][1]))

## Create individuals
inds = generate_individuals(trait, arch, Nsp, N, maxN, g::AbstractGrid)

## Create model object
model = MarineModel(arch, t, 1, inds, Nsp, N, g)

## Set up diagnostics (Rework once model runs)
#diags = MarineDiagnostics(model; tracer=(:PAR, :NH4, :NO3, :DOC), plankton = (:num, :graz, :mort, :dvid, :PS, :BS, :Chl), iteration_interval = 1)

# Set up simulation parameters
sim = simulation(model, ΔT = dt, iterations = n_iteration)

# Set up output writer
sim.output_writer = MarineOutputWriter(save_plankton=true)

# Run model. Currently this is very condensed, but I kept the code for when we work with environmental factors
update!(sim)


println("Works")
#println(inds.animals.sp1.data.z) #This is how you call specific values for each individual


#println(fieldnames(typeof(model.individuals.animals))) #Vector of species names
#println(test2.data.mig_status) #Gather vector of current values