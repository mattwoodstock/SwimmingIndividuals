using PlanktonIndividuals, Random, CSV, DataFrames, StructArrays

using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions: @kernel, @index


include("utilities.jl")
include("create.jl")
include("simulation.jl")
include("timestep.jl")
include("diagnostics.jl")
include("output.jl")
include("update.jl")

## Load in necessary databases
cd("D:/SwimmingIndividuals/Adapted")
trait = Dict(pairs(eachcol(CSV.read("traits.csv",DataFrame))))
state = CSV.read("state.csv",DataFrame)
grid = CSV.read("grid.csv",DataFrame)

## Convert values to match proper structure
Nsp = parse(Int64,state[state.Name .== "numspec", :Value][1])
N = trait[:Abundance]
maxN = parse(Int64,state[state.Name .== "maxindividuals", :Value][1])
arch = CPU() #Architecure to use
t = 1 #Time in seconds (Will need to adjust)
n_iteration = 1 #Number of iterations I think
dt = 60 #seconds per time step



## Create Output grid
g = RectilinearGrid(size=(grid[grid.Name.=="latres",:Value][1],grid[grid.Name.=="lonres",:Value][1],grid[grid.Name.=="depthres",:Value][1]), landmask = nothing, x = (grid[grid.Name.=="latmin",:Value][1], grid[grid.Name.=="latmax",:Value][1]), y = (grid[grid.Name.=="lonmin",:Value][1],grid[grid.Name.=="lonmax",:Value][1]), z = (0,-1*grid[grid.Name.=="depthmax",:Value][1]))

## Create individuals
inds = generate_individuals(trait, arch, Nsp, N, maxN, g::AbstractGrid)

## Create model object
model = MarineModel(arch, t, iteration, inds, g)

## Set up diagnostics (Rework once model runs)
#diags = MarineDiagnostics(model; tracer=(:PAR, :NH4, :NO3, :DOC),
#                                   plankton = (:num, :graz, :mort, :dvid, :PS, :BS, :Chl),
#                                   iteration_interval = 1)

# Set up simulation parameters
sim = PlanktonSimulation(model, ΔT = dt,iterations = n_iteration)

# Set up output writer
sim.output_writer = PlanktonOutputWriter(save_plankton=true)

# Run model. Currently this is very condensed, but I kept the code for when we work with environmental factors
update!(sim)



println("Works")
println(inds.animals.sp1.data.z) #This is how you call specific values for each individual