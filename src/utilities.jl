##### structs for individuals
mutable struct plankton
    data::AbstractArray
    p::NamedTuple
end

struct individuals
    animals::NamedTuple
end

mutable struct resource
    sp::Int
    ind::Int
    x::Float64
    y::Float64
    z::Float64
    pool_x::Int
    pool_y::Int
    pool_z::Int
    biomass::Float64
    capacity::Float64
end

mutable struct PreyInfo
    Predator::Int
    Sp::Int
    Ind::Int
    Type::Int
    Length::Float64
    Biomass::Float64
    Distance::Float64
end

mutable struct ResourcePrey
    Type::Int64
    Sp::Int64
    Ind::Int64
    Length::Float64       # mm
    Biomass::Float64    # g
    a::Float64             # length-weight coefficient
    b::Float64             # length-weight exponent
end

struct Selectivity
    species::String
    L50::Float64         # length at 50% selectivity
    slope::Float64       # steepness of logistic curve
end

struct HabitatPoint
    x::Int
    y::Int
    value::Float64
end

mutable struct Fishery
    name::String
    target_species::Vector{String}
    bycatch_species::Vector{String}
    selectivities::Dict{String, Selectivity}
    quota::Float64
    cumulative_catch::Float64
    cumulative_inds::Int64
    season::Tuple{Int, Int}                     # (start_day, end_day)
    area::Tuple{Tuple{Float64, Float64},        # x bounds
                Tuple{Float64, Float64},        # y bounds
                Tuple{Float64, Float64}}        # z bounds
    slot_limit::Tuple{Float64, Float64}         # (min_len, max_len)
    bag_limit::Int64                            # Daily Bag Limit for Fishery
end

mutable struct MarineEnvironment
    bathymetry::Array           #Bathymetry grid
    temp::Array                 #TemperatureArray
    salt::Array                 #SalinityArray
    chl::Array                  #CHL array
    ts::Int                     #Environmental time step
end

mutable struct MarineDepths
    focal_day::DataFrame
    focal_night::DataFrame 
    patch_day::DataFrame
    patch_night::DataFrame
    grid::DataFrame
end

##### Model struct
mutable struct MarineModel
    arch::Architecture          # architecture on which models will run
    environment::MarineEnvironment #Environment variables
    depths::MarineDepths        #Depth Profiles for all species
    fishing::Vector{Fishery}
    t::Float64                  # time in minute
    iteration::Int64            # model interation
    dt::Float64                 # Temporal Resolution
    individuals::individuals    # initial individuals generated
    resources::Vector{resource}         # resource characteristics
    resource_trait::DataFrame
    capacities::Array
    ninds::Int64
    n_species::Int64            # Number of species
    n_resource::Int64
    abund::Vector{Int64}
    bioms::Vector{Float64}
    init_abund::Vector{Int64}   # Initial Abundance of animals
    files::DataFrame            #Files to call later in model
    output_dt::Int64
    spinup::Int64               #Number of timesteps in a spinup
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

#Create a multimodal distribution. May not need to be used in the model and should probably be used a priori.
function multimodal_distribution(x, means, stds, weights)
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

# Function to get target_z based on distribution
function get_target_z(sp, dist)
    return gaussmix(1, dist[sp, "mu1"], dist[sp, "mu2"], dist[sp, "mu3"], dist[sp, "sigma1"], dist[sp, "sigma2"], dist[sp, "sigma3"], dist[sp, "lambda1"], dist[sp, "lambda2"])[1]
end

function generate_depths(files)
    focal_file_day = files[files.File .== "focal_z_dist_day", :Destination][1]
    focal_file_night = files[files.File .== "focal_z_dist_night", :Destination][1]

    focal_day = CSV.read(focal_file_day, DataFrame)
    focal_night = CSV.read(focal_file_night, DataFrame)

    resource_file_day = files[files.File .== "resource_z_dist_day", :Destination][1]
    resource_file_day = files[files.File .== "resource_z_dist_night", :Destination][1]

    resource_day = CSV.read(resource_file_day, DataFrame)
    resource_night = CSV.read(resource_file_day, DataFrame)

    grid_file = files[files.File .== "grid",:Destination][1]
    grid = CSV.read(grid_file, DataFrame)

    MarineDepths(focal_day,focal_night,resource_day,resource_night,grid)
end

function load_ascii_raster(file_path::String)
    open(file_path, "r") do f
        # Read header lines
        header = Dict{String, Float64}()
        for _ in 1:6
            line = readline(f)
            key, val = split(line)
            header[key] = parse(Float64, val)
        end
        
        # Load the remaining values as an array
        data = readdlm(f)
        return data
    end
end

function lognormal_params_from_maxsize(max_size::Int)
    median = 1/3 * max_size
    percentile = 0.95

    μ = log(median)
    z = quantile(Normal(0, 1), percentile)  # z-score for given percentile (e.g., 1.645 for 95%)
    
    # Solve for σ using: log(max_size) = μ + z * σ
    σ = (log(max_size) - μ) / z
    
    return μ, σ
end

function atan2(y::Float64, x::Float64)
    if x > 0
        return atan(y / x)
    elseif x < 0 && y >= 0
        return atan(y / x) + π
    elseif x < 0 && y < 0
        return atan(y / x) - π
    elseif x == 0 && y > 0
        return π / 2
    elseif x == 0 && y < 0
        return -π / 2
    else
        return 0.0  # undefined case (x == 0 && y == 0), return 0 by convention
    end
end

# Haversine distance in meters
function haversine(lat1, lon1, lat2, lon2)
    R = 6371000.0  # Earth radius in meters

    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1)
    Δλ = deg2rad(lon2 - lon1)
    a = sin(Δφ/2)^2 + cos(φ1) * cos(φ2) * sin(Δλ/2)^2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R * c
end

function knn_haversine(tree, query_point,z_prey, k,sp,ind,type,biomass,lengths,max_dist)
    # query_point is in (lat, lon)
    lat_query, lon_query,z_query = query_point

    # Preallocate arrays for distances and indices
    n = size(tree.data, 1)
    distances = Float64[]
    spec = Int64[]
    individual = Int64[]
    types = Int64[]
    biomasses=Float64[]
    sizes = Float64[]

    # Calculate distances and store indices
    for i in 1:n
        lat_prey, lon_prey = tree.data[i][1], tree.data[i][2]
        horizontal_dist = haversine(lat_query, lon_query, lat_prey, lon_prey)
        depth_diff = abs(z_query - z_prey[i])
        dist = sqrt(horizontal_dist ^ 2 + depth_diff ^ 2)

        if dist <= max_dist

            push!(distances, dist)
            push!(spec, sp[i])
            push!(individual, ind[i])
            push!(types,type[i])
            push!(biomasses,biomass[i])
            push!(sizes,lengths[i])
        end
    end

    if length(individual) > 0
        # Combine distances and indices into a tuple array
        distance_index_pairs = [(distances[i],spec[i],types[i],individual[i],biomasses[i],sizes[i]) for i in 1:length(individual)]

        # Sort the pairs by distance
        sort!(distance_index_pairs, by = x -> x[1])

        # Extract the first k entries
        k = min(k, length(distance_index_pairs))
        nearest_neighbors = distance_index_pairs[1:k]

        # Return indices and distances
        return [neighbor[4] for neighbor in nearest_neighbors], [neighbor[2] for neighbor in nearest_neighbors],[neighbor[3] for neighbor in nearest_neighbors], [neighbor[5] for neighbor in nearest_neighbors], [neighbor[1] for neighbor in nearest_neighbors],[neighbor[6] for neighbor in nearest_neighbors]
    else
        return individual, spec,types,biomasses, distances, sizes
    end
end

function logistic(x, k, c)
    return 1 ./ (1 .+ exp.(k.*(x.-c)))
end

function update_prey_distances(model,sp,ind,prey_list,max_dist,type)
    if type == 1
        animal_data = model.individuals.animals[sp].data
        px, py, pz = animal_data.x[ind], animal_data.y[ind], animal_data.z[ind]
    else
        px = getfield(model.resources[ind], :x)
        py = getfield(model.resources[ind], :y)
        pz = getfield(model.resources[ind], :z)
    end
    updated_prey = PreyInfo[]
    for prey in prey_list
        if prey.Type == 1
            prey_data = model.individuals.animals[prey.Sp].data
            if prey_data.alive[prey.Ind] == 0
                continue
            end
            prey_x =  prey_data.x[prey.Ind]
            prey_y =  - prey_data.y[prey.Ind]
            prey_z = prey_data.z[prey.Ind]
        else
            resource = model.resources[prey.Ind[1]]
            if resource.biomass <= 0
                continue
            end
            prey_x = resource.x
            prey_y = resource.y
            prey_z = resource.z
        end
        horizontal_dist = haversine(py, px, prey_y, prey_x)
        depth_diff = abs(pz -prey_z)
        dist = sqrt(horizontal_dist ^ 2 + depth_diff ^ 2)

        if dist <= max_dist
            prey.Distance = dist
            push!(updated_prey, prey)
        end
    end
    sort!(updated_prey, by = x -> x.Distance)
    if length(updated_prey) > 0
        return updated_prey[1]
    else
        return nothing
    end
end