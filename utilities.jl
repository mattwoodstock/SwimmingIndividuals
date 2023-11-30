##### structs for individuals
mutable struct plankton
    data::AbstractArray
    p::NamedTuple
end

struct individuals
    animals::NamedTuple
end

mutable struct groups
    density::AbstractArray
    characters::NamedTuple
end

struct pools
    pool::NamedTuple
end

mutable struct MarineModel
    arch::Architecture          # architecture on which models will run
    t::Float64                  # time in minute
    iteration::Int64            # model interation
    individuals::individuals    # initial individuals generated
    #pools::pools              # Characteristics of pooled species
    n_species::Int64            # Number of species
    ninds::Vector{Int}          # Total number of individuals in the model
    grid::AbstractGrid          # grid information
    dimension::Int64            # Dimensionality of the model. Either 1 or 3.
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

#Resample from a gaussian mixture model
function gaussmix(n,m1, m2, m3, s1, s2, s3, l1, l2)
    I = rand(n) .< l1
    I2 = rand(n) .< l1 .+ l2
    z = [rand(I[i] ? Normal(m1, s1) : (I2[i] ? Normal(m2, s2) : Normal(m3, s3))) for i in 1:n]
    return z
end

#Convert degrees to radians
function deg2rad(deg)
    return deg * π / 180
end

#Derive a NewtonRaphson equation
function NewtonRaphson(f, fp, x0, tol=1e-5, maxIter = 1000)
    x = x0
    fx = f(x0)
    iter = 0
    while abs(fx) > tol && iter < maxIter
        x = x  - fx/fp(x)   # Iteration
        fx = f(x)           # Precompute f(x)
        iter += 1
    end
    return x
end


function multimodal_distribution(x, means, stds, weights)
    if length(means) != length(stds) != length(weights) || length(means) < 1
        error("Invalid input: The lengths of means, stds, and weights should be equal and greater than 0.")
    end
    
    pdf_values = [weights[i] * pdf(Normal(means[i], stds[i]), x) for i in 1:length(means)]
    return sum(pdf_values)
end