##### structs for individuals
mutable struct plankton
    data::AbstractArray
    p::NamedTuple
end

struct individuals
    animals::NamedTuple
end

##### structs for pools
mutable struct patch
    data::AbstractArray
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

struct PredatorInfo
    x::Float64
    y::Float64
    z::Float64
    Distance::Float64
end

struct PreyInfo
    Type::Int
    Sp::Int
    Ind::Int
    x::Float64
    y::Float64
    z::Float64
    Biomass::Float64
    Length::Float64
    Inds::Float64
    Distance::Float64
end

mutable struct AllPreds
    preds::NamedTuple
end

mutable struct AllPreys
    preys::NamedTuple
end

mutable struct MarineEnvironment
    temp::Array                 #TemperatureArray
    temp_z::Vector{Float64}     #Z values for temperaturearray
    chl::Array
end

##### Model struct
mutable struct MarineModel
    arch::Architecture          # architecture on which models will run
    environment::MarineEnvironment
    t::Float64                  # time in minute
    iteration::Int64            # model interation
    dt::Float64                 # Patch Resolution
    individuals::individuals    # initial individuals generated
    pools::pools              # Characteristics of pooled species
    #parts::particles            # Particle characteristics (e.g., eDNA)
    ninds::Int64
    n_species::Int64            # Number of IBM species
    n_pool::Int64               # Number of pooled species
    bioms::Vector{Float64}          # Total number of individuals in the model
    abund::Vector{Int64}        #Abundance of animals
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
function newton_raphson(f, fp, r_init=1.0, max_iteration=100, tolerance=1e-6)
    r_prev = r_init
    iter = 1
    
    while iter <= max_iteration
        f_val = f(r_prev)
        df_val = fp(r_prev)
        r_next = r_prev .- f_val ./ df_val
        difference = abs.(r_next .- r_prev)
        
        if all(difference .< tolerance)
            break
        end
        
        r_prev = r_next
        iter += 1
    end
    
    return r_prev
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

function safe_intersect(sets::Vector{Set{Int}})
    common_indices = sets[1]
    for s in sets[2:end]
        common_indices = intersect(common_indices, s)
        if isempty(common_indices)
            return Set{Int}()
        end
    end
    return common_indices
end

function trilinear_interpolation_irregular_z(temp_grid, xs, ys, zs, z_vals)
    num_individuals = length(xs)
    temperatures = Vector{Float64}(undef, num_individuals)

    # Get grid dimensions
    x_size, y_size, z_size = size(temp_grid)

    # Calculate indices
    x1 = clamp.(floor.(Int, xs), 1, x_size - 1)
    x2 = clamp.(x1 .+ 1, 1, x_size)

    y1 = clamp.(floor.(Int, ys), 1, y_size - 1)
    y2 = clamp.(y1 .+ 1, 1, y_size)

    # Prepare arrays to hold z1 and z2 indices
    z1_indices = Int[]
    z2_indices = Int[]

    # Find z indices and corresponding values
    for z in zs
        z1 = findfirst(z_val -> z_val >= z, z_vals)
        if isnothing(z1)
            push!(z1_indices, z_size - 1)
            push!(z2_indices, z_size)
        else
            z1 = clamp(z1, 1, z_size - 1)
            push!(z1_indices, z1)
            push!(z2_indices, clamp(z1 + 1, 1, z_size))
        end
    end

    z1_values = z_vals[z1_indices]
    z2_values = z_vals[z2_indices]

    # Calculate the relative positions along the z-axis
    zd = (zs .- z1_values) ./ (z2_values .- z1_values)

    # Retrieve the values from the grid using the computed indices
    for i in 1:num_individuals
        xi, yi, zi1, zi2 = x1[i], y1[i], z1_indices[i], z2_indices[i]
        
        c000 = temp_grid[xi, yi, zi1]
        c001 = temp_grid[xi, yi, zi2]
        c010 = temp_grid[xi, y2[i], zi1]
        c011 = temp_grid[xi, y2[i], zi2]
        c100 = temp_grid[x2[i], yi, zi1]
        c101 = temp_grid[x2[i], yi, zi2]
        c110 = temp_grid[x2[i], y2[i], zi1]
        c111 = temp_grid[x2[i], y2[i], zi2]

        # Compute the interpolated value for each z slice
        c00 = c000 * (1 - zd[i]) + c100 * zd[i]
        c01 = c001 * (1 - zd[i]) + c101 * zd[i]
        c10 = c010 * (1 - zd[i]) + c110 * zd[i]
        c11 = c011 * (1 - zd[i]) + c111 * zd[i]

        # Interpolate in x and y
        xd = xs[i] - x1[i] + 1
        yd = ys[i] - y1[i] + 1

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        temperatures[i] = c0 * (1 - xd) + c1 * xd
    end

    return temperatures
end

function sphere_volume(length::Float64, num_individuals)::Float64
    # Calculate the total volume occupied by the individuals
    volume_individual = π * length^3 / 6
    total_volume = num_individuals * volume_individual
    
    # Calculate the radius of the sphere containing all individuals
    radius_cubed = total_volume * 3 / (4 * π)
    radius = radius_cubed^(1/3)
    
    # Calculate the volume of the sphere
    sphere_volume = 4/3 * π * radius^3
    return sphere_volume
end