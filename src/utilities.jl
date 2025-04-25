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

struct PredatorInfo
    Prey::Int
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

struct PreyInfo
    Predator::Int
    Type::Int
    Sp::Int
    Ind::Int
    x::Float64
    y::Float64
    z::Float64
    Biomass::Float64
    Energy::Float64
    Length::Float64
    Inds::Float64
    Distance::Float64
end

struct Selectivity
    species::String
    L50::Float64         # length at 50% selectivity
    slope::Float64       # steepness of logistic curve
end

mutable struct Fishery
    name::String
    target_species::Vector{String}
    bycatch_species::Vector{String}
    selectivities::Dict{String, Selectivity}
    quota::Float64
    cumulative_catch::Float64
    season::Tuple{Int, Int}                     # (start_day, end_day)
    area::Tuple{Tuple{Float64, Float64},        # x bounds
                Tuple{Float64, Float64},        # y bounds
                Tuple{Float64, Float64}}        # z bounds
    slot_limit::Tuple{Float64, Float64}         # (min_len, max_len)
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
    dt::Float64                 # Patch Resolution
    individuals::individuals    # initial individuals generated
    pools::pools              # Characteristics of pooled species
    capacities::Array
    pool_capacities::Array
    ninds::Int64
    n_species::Int64            # Number of IBM species
    n_pool::Int64               # Number of pooled species
    bioms::Vector{Float64}          # Total number of individuals in the model
    abund::Vector{Int64}        #Abundance of animals
    grid::AbstractGrid          # grid information
    files::DataFrame            #Files to call later in model
    output_dt::Int64
    cell_size::Float64            #Cubic meters of each grid cell
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

# Function to get target_z based on distribution
function get_target_z(sp, dist)
    return gaussmix(1, dist[sp, "mu1"], dist[sp, "mu2"], dist[sp, "mu3"], dist[sp, "sigma1"], dist[sp, "sigma2"], dist[sp, "sigma3"], dist[sp, "lambda1"], dist[sp, "lambda2"])[1]
end

function add_prey(model, prey_type, sp_data, prey_data, ind, this_pred, indices, abundances, sp, detection, max_num, n_preys, max_dist)
    # Precompute haversine for horizontal distances once
    horiz_dist = haversine.(sp_data.y[ind], sp_data.x[ind], prey_data.x[indices], prey_data.y[indices])
    dz = sp_data.z[ind] .- prey_data.z[indices]
    dist = sqrt.(horiz_dist.^2 .+ dz.^2)

    # Filter indices within detection range
    within_detection = findall(dist .<= detection[this_pred])

    # Early exit if no prey is within detection range
    if isempty(within_detection)
        return PreyInfo[]
    end

    # Filter by maximum distance
    valid_indices = findall(dist .<= max_dist)
    
    if (n_preys + length(valid_indices)) > max_num
        # Sort and get top `n_add` indices based on distance
        n_add = Int(max_num - (n_preys + length(valid_indices)))
        dists_subset = dist[valid_indices]
        idx_subset = partialsortperm(dists_subset, 1:n_add)

        # Remaining valid indices after partial sort
        remaining = valid_indices[idx_subset]
    else
        remaining = valid_indices
    end

    # Pre-allocate array for prey_infos
    prey_infos = Vector{PreyInfo}(undef, length(remaining))

    for i in 1:length(remaining)
        prey_idx = remaining[i]

        if prey_type == 1
            energy = prey_data.biomass[indices[prey_idx]] * model.individuals.animals[sp].p.Energy_density[2][sp]
            sp_data.landscape[ind] += energy
            prey_infos[i] = PreyInfo(ind, prey_type, sp, indices[prey_idx], prey_data.x[indices[prey_idx]], prey_data.y[indices[prey_idx]], 
                                     prey_data.z[indices[prey_idx]], prey_data.biomass[indices[prey_idx]], energy, 
                                     prey_data.length[indices[prey_idx]], abundances, dist[prey_idx])
        else
            energy = prey_data.biomass[indices[prey_idx]] * model.pools.pool[sp].characters.Energy_density[2][sp]
            sp_data.landscape[ind] += energy
            prey_infos[i] = PreyInfo(ind, prey_type, sp, indices[prey_idx], prey_data.x[indices[prey_idx]], prey_data.y[indices[prey_idx]], 
                                     prey_data.z[indices[prey_idx]], prey_data.biomass[indices[prey_idx]], energy, 
                                     prey_data.length[indices[prey_idx]], abundances[indices[prey_idx]], dist[prey_idx])
        end
    end

    return prey_infos
end

function add_pred(pred_type,sp_data, pred_data, ind,this_pred, indices, abundances, sp,detection)
    dx = sp_data.x[ind] .- pred_data.x[indices]
    dy = sp_data.y[ind] .- pred_data.y[indices]
    dz = sp_data.z[ind] .- pred_data.z[indices]
    dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)

    within_detection = findall(dist .<= detection[this_pred])
    #println(within_detection)
    pred_infos = PredatorInfo[]  # Initialize a vector to store prey info for this individual
    for i in within_detection
        if pred_type == 1
            pred_infos = vcat(pred_infos, PredatorInfo(ind,pred_type, sp, indices[i], pred_data.x[indices[i]], pred_data.y[indices[i]], pred_data.z[indices[i]], pred_data.biomass[indices[i]], pred_data.length[indices[i]], abundances, dist[i]))

        else
            pred_infos = vcat(pred_infos, PredatorInfo(ind,pred_type, sp, indices[i], pred_data.x[indices[i]], pred_data.y[indices[i]], pred_data.z[indices[i]], pred_data.biomass[indices[i]], pred_data.length[indices[i]], abundances[indices[i]], dist[i]))
        end
    end
    return pred_infos
end

function generate_depths(files)
    focal_file_day = files[files.File .== "focal_z_dist_day", :Destination][1]
    focal_file_night = files[files.File .== "focal_z_dist_night", :Destination][1]

    focal_day = CSV.read(focal_file_day, DataFrame)
    focal_night = CSV.read(focal_file_night, DataFrame)

    patch_file_day = files[files.File .== "nonfocal_z_dist_day", :Destination][1]
    patch_file_night = files[files.File .== "nonfocal_z_dist_night", :Destination][1]

    patch_day = CSV.read(patch_file_day, DataFrame)
    patch_night = CSV.read(patch_file_night, DataFrame)

    grid_file = files[files.File .== "grid",:Destination][1]
    grid = CSV.read(grid_file, DataFrame)

    MarineDepths(focal_day,focal_night,patch_day,patch_night,grid)
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

function lognormal_params_from_maxsize(max_size::Float64)
    median = 1/3 * max_size
    percentile = 0.95

    μ = log(median)
    z = quantile(Normal(0, 1), percentile)  # z-score for given percentile (e.g., 1.645 for 95%)
    
    # Solve for σ using: log(max_size) = μ + z * σ
    σ = (log(max_size) - μ) / z
    
    return μ, σ
end

function find_path(capacity::Matrix{Float64}, start::Tuple{Int,Int}, goal::Tuple{Int,Int})
    flipped_cap = reverse(capacity,dims=1)
    
    open_set = [start]
    came_from = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    visited = Set{Tuple{Int,Int}}()
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    while !isempty(open_set)
        current = popfirst!(open_set)
        if current == goal
            # reconstruct path
            path = [current]
            while current in keys(came_from)
                current = came_from[current]
                pushfirst!(path, current)
            end
            return path
        end
        push!(visited, current)
        for (dy, dx) in directions
            ny, nx = current[1] + dy, current[2] + dx
            if 1 ≤ nx ≤ size(flipped_cap,2) && 1 ≤ ny ≤ size(flipped_cap,1)
                neighbor = (ny, nx)
                if flipped_cap[ny, nx] > 0 && !(neighbor in visited) && !(neighbor in open_set)
                    push!(open_set, neighbor)
                    came_from[neighbor] = current
                end
            end
        end
    end
    return []  # no path found
end

function reachable_point(current_pos::Tuple{Float64, Float64}, path::Vector{Tuple{Int, Int}},max_distance::Float64, latmin::Float64, lonmin::Float64,cell_size::Float64, nrows::Int, ncols::Int)

    # Convert grid cell (i, j) to lat/lon, add randomness within cell
    function grid_to_coords(cell::Tuple{Int, Int})
        i, j = cell
        lon = lonmin + (j - 1) * cell_size + rand() * cell_size
        lat = latmin + (i - 1) * cell_size + rand() * cell_size
        return (lat, lon)
    end

    total_distance = 0.0
    prev_lat, prev_lon = current_pos

    for i in 1:length(path)
        lat, lon = grid_to_coords(path[i])
        d = haversine(prev_lat, prev_lon, lat, lon)
        total_distance += d

        if total_distance > max_distance
            excess = total_distance - max_distance
            frac = 1 - (excess / d)



            interp_lat = prev_lat + frac * (lat - prev_lat)
            interp_lon = prev_lon + frac * (lon - prev_lon)

            # Compute the grid cell index of the interpolated point
            grid_x = clamp(Int(floor((interp_lon - lonmin) / cell_size) + 1), 1, ncols)
            grid_y = clamp(Int(floor((interp_lat - latmin) / cell_size) + 1), 1, nrows)

            # Add randomness within the cell of the interpolated point
            rand_lat = latmin + (grid_y - 1) * cell_size + rand() * cell_size
            rand_lon = lonmin + (grid_x - 1) * cell_size + rand() * cell_size

            return (rand_lat, rand_lon, grid_y, grid_x)
        end

        prev_lat, prev_lon = lat, lon
    end

    # If max distance not reached, return final randomized point within the last cell
    final_cell = path[end]
    final_lat = latmin + (final_cell[1] - 1) * cell_size + rand() * cell_size
    final_lon = lonmin + (final_cell[2] - 1) * cell_size + rand() * cell_size

    return (final_lat, final_lon, first(final_cell),last(final_cell))
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

function nearest_suitable_habitat(habitat::Matrix{Float64},start_latlon::Tuple{Float64, Float64},start_pool::Tuple{Int,Int},max_distance_m::Float64,latmin::Float64,lonmin::Float64,cellsize_deg::Float64)
    R = 6371000.0  # Earth radius in meters

    # Grid indexing and helpers
    get_neighbors(r, c, nrows, ncols) = [
        (r+dr, c+dc) for (dr, dc) in
        ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
        if 1 ≤ r+dr ≤ nrows && 1 ≤ c+dc ≤ ncols
    ]

    random_point_in_cell(cell::Tuple{Int, Int}) = begin
        row, col = cell
        lat = latmin + (row - 1 + rand()) * cellsize_deg
        lon = lonmin + (col - 1 + rand()) * cellsize_deg
        (lat, lon)
    end

    nrows, ncols = size(habitat)

    # 2. BFS to find nearest suitable cell
    visited = falses(nrows, ncols)
    queue = [start_pool]
    visited[start_pool...] = true
    came_from = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()

    goal_cell = nothing
    while !isempty(queue)
        current = popfirst!(queue)
        if habitat[nrows - first(current)+1,last(current)] > 0
            goal_cell = current
            break
        end
        for neighbor in get_neighbors(current[1], current[2], nrows, ncols)
            if !visited[neighbor...]
                visited[neighbor...] = true
                push!(queue, neighbor)
                came_from[neighbor] = current
            end
        end
    end

    if isnothing(goal_cell)
        return
    end

    # Random location in goal cell
    goal_latlon = random_point_in_cell(goal_cell)

    dist_m = haversine(start_latlon[1], start_latlon[2], goal_latlon[1], goal_latlon[2])

    if dist_m <= max_distance_m
        new_latlon = goal_latlon
    else
        # Move toward the goal with proper scaling
        φ1 = deg2rad(start_latlon[1])
        λ1 = deg2rad(start_latlon[2])
        φ2 = deg2rad(goal_latlon[1])
        λ2 = deg2rad(goal_latlon[2])

        Δφ = φ2 - φ1
        Δλ = λ2 - λ1
        θ = atan2(sin(Δλ) * cos(φ2), cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(Δλ))
        d_frac = max_distance_m / dist_m

        δ = dist_m * d_frac / R
        new_φ = asin(sin(φ1) * cos(δ) + cos(φ1) * sin(δ) * cos(θ))
        new_λ = λ1 + atan2(sin(θ) * sin(δ) * cos(φ1), cos(δ) - sin(φ1) * sin(new_φ))

        new_latlon = (rad2deg(new_φ), rad2deg(new_λ))
    end

    # Convert new location to grid cell index
    new_row = floor(Int, (new_latlon[1] - latmin) / cellsize_deg) + 1
    new_col = floor(Int, (new_latlon[2] - lonmin) / cellsize_deg) + 1

    return new_latlon[1], new_latlon[2], new_col, new_row
end
