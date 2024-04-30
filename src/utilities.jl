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

mutable struct eDNA
    data::AbstractArray
    state::NamedTuple
end

struct particles
    eDNA::eDNA
end

##### Model struct
mutable struct MarineModel
    arch::Architecture          # architecture on which models will run
    t::Float64                  # time in minute
    iteration::Int64            # model interation
    individuals::individuals    # initial individuals generated
    pools::pools              # Characteristics of pooled species
    #parts::particles            # Particle characteristics (e.g., eDNA)
    n_species::Int64            # Number of IBM species
    n_pool::Int64               # Number of pooled species
    ninds::Vector{Int}          # Total number of individuals in the model
    grid::AbstractGrid          # grid information
    files::DataFrame            #Files to call later in model
    output_dt::Int64
    cell_size::Int64            #Cubic meters of each grid cell
    spinup::Int64               #Number of timesteps in a spinup
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

#Convert degrees to radians
function deg2rad(deg)
    return deg * π / 180
end

#Derive a NewtonRaphson equation
function newton_raphson(f, fp)
    # Initial guess for r
    r_prev = 1.0
    
    # Define tolerance for convergence
    tolerance = 1e-6
    
    # Perform Newton-Raphson iteration
    while true
        f_val = f(r_prev)
        df_val = fp(r_prev)
        r_next = r_prev .- f_val ./ df_val
        difference = abs.(r_next .- r_prev)
        if all(difference .< tolerance)
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

function set_z_bin!(model, grid_file)
    # Read grid data outside the loop since it doesn't change
    grid = CSV.read(grid_file, DataFrame)
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    z_interval = maxdepth / depthres
    z_seq = 0:z_interval:maxdepth

    for i in 1:length(model.data.length)
        # Find the index where the animal's z value is greater than or equal to z_seq
        j = findlast(z_seq .<= model.data.z[i])

        # Handle the case where the animal is below the specified maximum depth
        if j > depthres
            j = depthres
        end

        model.data.pool_z[i] = j
    end
end

function sample_normal(minimum_value, maximum_value; num_samples = 1000, std=0.1)
    # Calculate the mean as halfway between the minimum and maximum values
    mean_value = (minimum_value + maximum_value) / 2.0
    
    # Generate an array of samples from a normal distribution
    samples = rand(Normal(mean_value, std), num_samples)
    return samples
end

function degrees_to_meters(lat, lon)
    # Earth's radius in meters
    R = 6371000.0

    # Convert degrees to radians
    lat_rad = lat * π / 180
    lon_rad = lon * π / 180

    # Calculate the coordinates in meters
    lat_meters = R * lat_rad
    lon_meters = R * lon_rad * cos(lat_rad)

    return lat_meters, lon_meters
end

function meters_to_degrees(lat_meters, lon_meters)
    # Earth's radius in meters
    R = 6371000.0

    # Convert meters to radians
    lat_rad = lat_meters / R
    lon_rad = lon_meters / (R * cos(lat_rad))

    # Convert radians to degrees
    lat_deg = lat_rad * 180 / π
    lon_deg = lon_rad * 180 / π

    return lat_deg, lon_deg
end

function logistic(x, k, c)
    return 1 ./ (1 .+ exp.(k.*(x.-c)))
end