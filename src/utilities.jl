##### structs for individuals
mutable struct plankton
    data::AbstractArray
    p::NamedTuple
end

struct individuals
    animals::NamedTuple
end

##### structs for pools
mutable struct groups
    density::AbstractArray
    characters::NamedTuple
end

struct pools
    pool::NamedTuple
end

##### Model struct
mutable struct MarineModel
    arch::Architecture          # architecture on which models will run
    t::Float64                  # time in minute
    iteration::Int64            # model interation
    individuals::individuals    # initial individuals generated
    pools::pools              # Characteristics of pooled species
    n_species::Int64            # Number of IBM species
    n_pool::Int64               # Number of pooled species
    ninds::Vector{Int}          # Total number of individuals in the model
    grid::AbstractGrid          # grid information
    files::DataFrame            #Files to call later in model
    output_dt::Int64
    cell_size::Int64            #Cubic meters of each grid cell
    #timestepper::timestepper    # Add back in once environmental parameters get involved
end

#####Functions from PlanktonIndividuals that have been placed here.
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

#Derive a NewtonRaphson equation
function newton_raphson(f,fp)
    # Initial guess for r
    r_prev = 1.0
    
    # Define tolerance for convergence
    tolerance = 1e-6
    
    # Perform Newton-Raphson iteration
    while true
        f_val = f(r_prev)
        df_val = fp(r_prev)
        r_next = r_prev - f_val / df_val
        if abs(r_next - r_prev) < tolerance
            return r_next
        end
        r_prev = r_next
    end
end

#Create a multimodal distribution. May not need to be used in the model and should probably be used a priori.
function multimodal_distribution(x, means, stds, weights)
    if length(means) != length(stds) != length(weights) || length(means) < 1
        error("Invalid input: The lengths of means, stds, and weights should be equal and greater than 0.")
    end
    
    pdf_values = [weights[i] * pdf(Normal(means[i], stds[i]), x) for i in 1:length(means)]
    return sum(pdf_values)
end

function logistic(x, k, c)
    return 1 / (1 + exp(k*(x-c)))
end
