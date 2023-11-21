##### structs for individuals
mutable struct plankton
    data::AbstractArray
    p::NamedTuple
end

struct individuals
    animals::NamedTuple
end

mutable struct MarineModel
    arch::Architecture          # architecture on which models will run
    t::Float64                  # time in minute
    iteration::Int64            # model interation
    individuals::individuals    # initial individuals generated
    n_species::Int64            # Number of species
    ninds::Vector{Int}          # Total number of individuals in the model
    grid::AbstractGrid          # grid information
    #timestepper::timestepper    # Add back in once environmental parameters get involved
end

@kernel function mask_individuals_kernel!(plank, g::AbstractGrid)
    i = @index(Global)
    @inbounds xi = unsafe_trunc(Int, (plank.x[i]+1)) + g.Hx 
    @inbounds yi = unsafe_trunc(Int, (plank.y[i]+1)) + g.Hy
    @inbounds zi = unsafe_trunc(Int, (plank.z[i]+1)) + g.Hz
end
function mask_individuals!(plank, g::AbstractGrid, N, arch)
    kernel! = mask_individuals_kernel!(device(arch), 256, (N,))
    kernel!(plank, g)
    return nothing
end