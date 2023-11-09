## Set Main Working Directory 
cd("D:/SwimmingIndividuals")

export plankton_advection!
export plankton_diffusion!
export plankton_update!
export generate_individuals, individuals
export find_inds!, find_NPT!, acc_counts!, calc_par!

using CUDA
using StructArrays
using Random
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type
using PlanktonIndividuals.Grids
using PlanktonIndividuals.Diagnostics

using PlanktonIndividuals: AbstractMode, CarbonMode, QuotaMode, MacroMolecularMode

using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using ProgressBars

include("initialize.jl")
include("grid.jl")
include("output.jl")
include("movement.jl")

# Initialize Model 

## Make dataframe of output grid cells 
g_frame = grid_dataframe(OutputGrid,"grid.csv")

## Make Individuals
inds = generate_individuals(trait,parse(Int,state[state.Name .=="numspec",:Value][1]),g_frame,"grid.csv")

## Assign necessary state variables
nts = parse(Int,state[state.Name.=="nts",:Value][1])
nind = nrow(inds)
# Run Model

for ts in 500:510

    ### *** Resample inds dataframe so that there is a different order each minute 
    inds[!,"Order"] = [1:1:nind;]

    for ind in 50:50
        this_ind = inds[inds.Order .== ind,:] #Subset individual

        this_trait = trait[trait[!,"SpeciesLong"] .== this_ind[!,"Species"][1],:] #Subset trait database

        ## Animal movement at time step
        this_ind.x[1],this_ind.y[1] = horizontal_movement(this_ind,this_trait,"grid.csv")
        this_ind.z[1],this_ind.move[1],this_ind.target_z[1],this_ind.dive_interval[1],this_ind.surface_interval[1] = vertical_movement(this_ind,this_trait,"grid.csv",ts)

        ## Predation/energy Component 

        ## Reproduction Component 

        ## Additional Mortality Component

    println(this_ind.z[1])
    end # End individual
end # End model run


# Produce Outputs
#density = biomass_density(inds,g_frame)
println("Works")