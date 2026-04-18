# ===================================================================
# Pathfinding and Movement Helpers (CPU-based)
# ===================================================================

"""
    find_path(capacity, start, goal)

Calculates a path from start to goal using the A* algorithm and linear indexing.
Optimized to minimize node exploration and dictionary overhead.
"""
function find_path(capacity::Matrix{Float32}, start::Tuple{Int,Int}, goal::Tuple{Int,Int})
    lonres, latres = size(capacity)
    
    # --- Option 2: Linear Indexing ---
    @inline to_idx(x, y) = (y - 1) * lonres + x
    @inline from_idx(i) = (mod1(i, lonres), div(i - 1, lonres) + 1)
    
    start_idx = to_idx(start...)
    goal_idx = to_idx(goal...)
    
    # --- Option 1: A* Algorithm ---
    # Costs from start to node
    g_score = Dict{Int, Float32}()
    g_score[start_idx] = 0.0f0
    
    # Heuristic: Euclidean distance to goal
    heuristic(idx) = begin
        curr_x, curr_y = from_idx(idx)
        goal_x, goal_y = from_idx(goal_idx)
        return sqrt(Float32((curr_x - goal_x)^2 + (curr_y - goal_y)^2))
    end
    
    # Estimated total cost (g + heuristic)
    f_score = Dict{Int, Float32}()
    f_score[start_idx] = heuristic(start_idx)
    
    # open_set contains nodes to explore. We find the node with lowest f_score.
    open_set = [start_idx]
    came_from = Dict{Int, Int}()
    
    # Search directions (including diagonals)
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    
    while !isempty(open_set)
        # Find index with lowest f_score (Priority selection)
        best_f = Inf32
        current = -1
        current_in_open_idx = -1
        
        for (i, node) in enumerate(open_set)
            f = get(f_score, node, Inf32)
            if f < best_f
                best_f = f
                current = node
                current_in_open_idx = i
            end
        end
        
        # Path Found: Reconstruct and return
        if current == goal_idx
            path = Tuple{Int, Int}[]
            curr = current
            while curr != start_idx
                pushfirst!(path, from_idx(curr))
                curr = came_from[curr]
            end
            pushfirst!(path, start)
            return path
        end
        
        deleteat!(open_set, current_in_open_idx)
        
        cx, cy = from_idx(current)
        for (dx, dy) in directions
            nx, ny = cx + dx, cy + dy
            
            if 1 <= nx <= lonres && 1 <= ny <= latres
                if capacity[nx, ny] > 0
                    neighbor = to_idx(nx, ny)
                    # Cost: 1.0 for cardinal, ~1.41 for diagonal
                    step_cost = (dx == 0 || dy == 0) ? 1.0f0 : 1.414f0
                    tentative_g = g_score[current] + step_cost
                    
                    if tentative_g < get(g_score, neighbor, Inf32)
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor)
                        if !(neighbor in open_set)
                            push!(open_set, neighbor)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

"""
    reachable_point(...)

Calculates the furthest point an agent can reach along a path given a distance budget.
Uses a consistent random offset for the journey to provide a realistic 'drunken walk'.
"""
function reachable_point(rng::AbstractRNG, current_pos::Tuple{Float32, Float32}, path::Vector{Tuple{Int, Int}}, max_distance::Float64, latmax::Float64, lonmin::Float64, cell_size::Float64, lonres::Int, latres::Int)
    
    # Generate a consistent random offset for this specific journey.
    rand_dx = rand(rng)
    rand_dy = rand(rng)

    function grid_to_coords(cell::Tuple{Int, Int})
        lon_idx, lat_idx = cell
        lon = lonmin + (lon_idx - 1 + rand_dx) * cell_size
        flipped_lat_idx = (latres - lat_idx) + 1
        lat = latmax - (flipped_lat_idx - 1 + rand_dy) * cell_size
        return (lat, lon)
    end

    total_distance = 0.0
    prev_lat, prev_lon = current_pos

    # Start at i = 2 to eliminate vibration at the start of the path.
    for i in 2:length(path)
        lat, lon = grid_to_coords(path[i])
        d = haversine(prev_lat, prev_lon, lat, lon)
        
        if total_distance + d > max_distance
            epsilon = 1e-6 
            if d > epsilon
                remaining_dist = max_distance - total_distance
                frac = remaining_dist / d

                interp_lat = prev_lat + frac * (lat - prev_lat)
                interp_lon = prev_lon + frac * (lon - prev_lon)
            
                grid_x = clamp(Int(floor((interp_lon - lonmin) / cell_size) + 1), 1, lonres)
                grid_y = clamp(latres - Int(floor((latmax - interp_lat) / cell_size)), 1, latres)

                return (interp_lat, interp_lon, grid_y, grid_x)
            else
                grid_x = clamp(Int(floor((prev_lon - lonmin) / cell_size) + 1), 1, lonres)
                grid_y = clamp(latres - Int(floor((latmax - prev_lat) / cell_size)), 1, latres)
                return (prev_lat, prev_lon, grid_y, grid_x)
            end
        end
        
        total_distance += d
        prev_lat, prev_lon = lat, lon
    end

    final_cell = path[end]
    final_lat, final_lon = grid_to_coords(final_cell)
    return (final_lat, final_lon, final_cell[2], final_cell[1])
end

function nearest_suitable_habitat(rng::AbstractRNG, habitat::Matrix{<:Real}, start_latlon::Tuple{<:Real, <:Real}, start_pool::Tuple{<:Integer,<:Integer}, max_distance_m::Real, latmax::Real, lonmin::Real, cellsize_deg::Real)
    R = 6371000.0
    lonres, latres = size(habitat)
    
    get_neighbors(x, y) = [(x+dx, y+dy) for (dx, dy) in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)) if 1<=x+dx<=lonres && 1<=y+dy<=latres]
    random_point_in_cell(cell) = (latmax - ((latres - cell[2] + 1) - 1 + rand(rng)) * cellsize_deg, lonmin + (cell[1] - 1 + rand(rng)) * cellsize_deg)

    visited = falses(lonres, latres)
    queue = [start_pool]
    visited[start_pool...] = true
    
    goal_cell = nothing
    while !isempty(queue)
        current = popfirst!(queue)
        if habitat[current...] > 0
            goal_cell = current
            break
        end
        for neighbor in get_neighbors(current[1], current[2])
            if !visited[neighbor...]
                visited[neighbor...] = true
                push!(queue, neighbor)
            end
        end
    end

    isnothing(goal_cell) && return nothing

    goal_latlon = random_point_in_cell(goal_cell)
    dist_m = haversine(start_latlon[1], start_latlon[2], goal_latlon[1], goal_latlon[2])

    if dist_m <= max_distance_m
        new_latlon = goal_latlon
    else
        φ1, λ1 = deg2rad(start_latlon[1]), deg2rad(start_latlon[2])
        φ2, λ2 = deg2rad(goal_latlon[1]), deg2rad(goal_latlon[2])
        Δλ = λ2 - λ1
        θ = atan(sin(Δλ) * cos(φ2), cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(Δλ))
        δ = max_distance_m / R
        new_φ = asin(sin(φ1) * cos(δ) + cos(φ1) * sin(δ) * cos(θ))
        new_λ = λ1 + atan(sin(θ) * sin(δ) * cos(φ1), cos(δ) - sin(φ1) * sin(new_φ))
        new_latlon = (rad2deg(new_φ), rad2deg(new_λ))
    end

    new_lon_idx = clamp(floor(Int, (new_latlon[2] - lonmin) / cellsize_deg) + 1, 1, lonres)
    new_lat_idx = clamp(latres - floor(Int, (latmax - new_latlon[1]) / cellsize_deg), 1, latres)

    return new_latlon[1], new_latlon[2], new_lon_idx, new_lat_idx
end

# ===================================================================
# Specialized Movement Kernels (DVM and Diving)
# ===================================================================

@kernel function dvm_action_kernel!(
    alive, mig_status, z, target_z, pool_z, active, gut_fullness, length,
    is_weak_migrator::Bool, swim_speed_bls::Float32,
    maxdepth, depth_res_m, depthres, t, dt
)
    ind = @index(Global)
    @inbounds if alive[ind] == 1.0
        
        my_status = mig_status[ind]
        my_z = z[ind]
        
        agent_length_m = length[ind] / 1000.0f0
        swim_speed = 4.0f0 # Migration velocity in meters per minute
        z_increment = swim_speed * Float32(dt)
        
        is_daytime = (360.0f0 <= t < 1080.0f0)
        
        if is_daytime
            if my_status == 0.0f0
                mig_status[ind] = 2.0f0
                z[ind] = min(target_z[ind], my_z + z_increment)
            elseif my_status == 2.0f0
                z[ind] = min(target_z[ind], my_z + z_increment)
                if z[ind] >= target_z[ind]; mig_status[ind] = -1.0f0; end
            end
        else
            if my_status == -1.0f0 && (t < 1080.0f0 + dt) && (t >= 1080.0f0)
                hunger_threshold = rand(Float32)
                is_hungry = (gut_fullness[ind] < hunger_threshold)
                can_ascend = !is_weak_migrator || is_hungry
                if can_ascend
                    mig_status[ind] = 1.0f0
                    z[ind] = max(1.0f0, max(target_z[ind], my_z - z_increment))
                end
            elseif my_status == 1.0f0
                z[ind] = max(1.0f0, max(target_z[ind], my_z - z_increment))
                if z[ind] <= target_z[ind]; mig_status[ind] = 0.0f0; end
            end
        end
        @inbounds pool_z[ind] = clamp(ceil(Int, z[ind] / depth_res_m), 1, depthres)
        @inbounds active[ind] += dt
    end
end

function dvm_action!(model::MarineModel, sp::Int, is_weak_migrator::Bool)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p_cpu = model.individuals.animals[sp].p
    swim_speed_bls = Float32(p_cpu.Swim_velo.second[sp])
    grid = model.depths.grid
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    depth_res_m = maxdepth / depthres
    t = model.t; dt = model.dt

    if t >= 360.0 && t < 360.0 + dt
        dist_params = model.depths.focal_day[sp, :]
        new_targets = gaussmix(length(data.x), 
                               dist_params.mu1, dist_params.mu2, dist_params.mu3,
                               dist_params.sigma1, dist_params.sigma2, dist_params.sigma3,
                               dist_params.lambda1, dist_params.lambda2)
        copyto!(data.target_z, Float32.(new_targets))
    end

    if t >= 1080.0 && t < 1080.0 + dt
        dist_params = model.depths.focal_night[sp, :]
        new_targets = gaussmix(length(data.x), 
                               dist_params.mu1, dist_params.mu2, dist_params.mu3,
                               dist_params.sigma1, dist_params.sigma2, dist_params.sigma3,
                               dist_params.lambda1, dist_params.lambda2)
        copyto!(data.target_z, Float32.(new_targets))
    end
    
    kernel! = dvm_action_kernel!(device(arch), 256, (length(data.x),))
    kernel!(data.alive, data.mig_status, data.z, data.target_z, data.pool_z, data.active, data.gut_fullness, data.length, is_weak_migrator, swim_speed_bls, maxdepth, depth_res_m, depthres, t, dt)
    KernelAbstractions.synchronize(device(arch))
end

# ===================================================================
# General Habitat-Seeking Movement
# ===================================================================

function movement_toward_habitat!(model::MarineModel, sp::Int, time::AbstractArray)
    arch = model.arch; animal_data = model.individuals.animals[sp].data; animal_param = model.individuals.animals[sp].p
    cpu_time = Array(time); cpu_x, cpu_y, cpu_z = Array(animal_data.x), Array(animal_data.y), Array(animal_data.z)
    cpu_pool_x, cpu_pool_y, cpu_pool_z = Array(animal_data.pool_x), Array(animal_data.pool_y), Array(animal_data.pool_z)
    cpu_length = Array(animal_data.length); cpu_alive = Array(animal_data.alive); cpu_active = Array(animal_data.active)

    new_x, new_y, new_z = copy(cpu_x), copy(cpu_y), copy(cpu_z)
    new_pool_x, new_pool_y, new_pool_z = copy(cpu_pool_x), copy(cpu_pool_y), copy(cpu_pool_z)
    
    month = model.environment.ts; habitat = Array(model.capacities[:, :, month, sp]); grid = model.depths.grid
    latmax = grid[grid.Name .== "yulcorner", :Value][1]; lonmin = grid[grid.Name .== "xllcorner", :Value][1]; cell_size = grid[grid.Name .== "cellsize", :Value][1]
    lonres, latres = size(habitat); maxdepth = grid[grid.Name .== "depthmax", :Value][1]; depthres = Int(grid[grid.Name .== "depthres", :Value][1]); depth_res_m = maxdepth / depthres
    habitat_cells = [(x=i, y=j, value=habitat[i,j]) for i in 1:lonres, j in 1:latres if habitat[i,j] > 0]
    
    if !isempty(habitat_cells)
        sort!(habitat_cells, by = x -> x.value, rev = true)
        cumvals = cumsum(getfield.(habitat_cells, :value)); total_val = cumvals[end]

        # --- OPTIMIZATION: Path Cache for Memoization ---
        # Option 2: Using Tuple of linear indices for faster key hashing
        @inline to_idx(x, y) = (y - 1) * lonres + x
        path_cache = Dict{Tuple{Int, Int}, Vector{Tuple{Int,Int}}}()
        cache_lock = ReentrantLock()

        Threads.@threads for ind in 1:length(cpu_x)
            if cpu_alive[ind] == 1.0 && cpu_time[ind] > 0
                rng = Random.default_rng(); start_pool = (cpu_pool_x[ind], cpu_pool_y[ind])
                cur_pos = (cpu_y[ind], cpu_x[ind])
                
                len_m = cpu_length[ind] / 1000.0; swim_speed_ms = animal_param.Swim_velo[2][sp] * len_m; max_dist = swim_speed_ms * cpu_time[ind]

                # 1. Vertical adjustment
                vertical_shift = (rand(Float32) * 10.0f0) - 5.0f0; vertical_shift = clamp(vertical_shift, -max_dist, max_dist)
                max_horizontal_dist = max(0.0, max_dist - abs(vertical_shift))
                new_z[ind] = clamp(cpu_z[ind] + vertical_shift, 1.0f0, Float32(maxdepth))
                new_pool_z[ind] = clamp(ceil(Int, new_z[ind] / depth_res_m), 1, depthres)

                if !(1 <= start_pool[1] <= lonres && 1 <= start_pool[2] <= latres)
                    @warn "Relocating corrupted agent $(ind) of species $sp."
                end

                if habitat[start_pool...] == 0
                    res = nearest_suitable_habitat(rng, habitat, cur_pos, start_pool, max_horizontal_dist, latmax, lonmin, cell_size)
                    if res !== nothing; new_y[ind], new_x[ind], new_pool_x[ind], new_pool_y[ind] = res; end
                else
                    # 2. Select goal
                    r = rand(rng) * total_val 
                    idx = findfirst(x -> x >= r, cumvals)
                    
                    if idx !== nothing
                        goal_pool = (habitat_cells[idx].x, habitat_cells[idx].y)
                        
                        # --- MEMOIZATION CHECK (Option 2) ---
                        cache_key = (to_idx(start_pool...), to_idx(goal_pool...))
                        path = lock(cache_lock) do
                            get(path_cache, cache_key, nothing)
                        end
                        
                        if isnothing(path)
                            path = find_path(habitat, start_pool, goal_pool)
                            if !isnothing(path)
                                lock(cache_lock) do
                                    path_cache[cache_key] = path
                                end
                            end
                        end
                        
                        # 3. Travel along full path (Option 1 Benefit: Higher Path Quality)
                        if path !== nothing && length(path) > 1
                            res_lat, res_lon, res_pool_y, res_pool_x = reachable_point(rng, cur_pos, path, max_horizontal_dist, latmax, lonmin, cell_size, lonres, latres)
                            new_y[ind], new_x[ind], new_pool_y[ind], new_pool_x[ind] = res_lat, res_lon, res_pool_y, res_pool_x
                        end
                    end
                end
            end
        end
    end

    copyto!(animal_data.x, new_x); copyto!(animal_data.y, new_y); copyto!(animal_data.z, new_z)
    copyto!(animal_data.pool_x, new_pool_x); copyto!(animal_data.pool_y, new_pool_y); copyto!(animal_data.pool_z, new_pool_z)
    return nothing
end

# ===================================================================
# Monthly Resource Redistribution System
# ===================================================================

@kernel function move_resources_kernel!(
    new_biomass_grid,
    new_capacity_grid,
    capacities_for_month,
    total_biomass_per_sp,
    total_suitability_per_sp,
    normalized_vertical_profiles,
    total_capacity_targets
)
    lon, lat, depth_idx, res_sp = @index(Global, NTuple)

    current_total_biomass = total_biomass_per_sp[res_sp]
    total_suitability = total_suitability_per_sp[res_sp]
    
    if total_suitability > 0
        capacity_this_cell = capacities_for_month[lon, lat, res_sp]
        horizontal_proportion = capacity_this_cell / total_suitability
        vertical_proportion = normalized_vertical_profiles[depth_idx, res_sp]
        
        # 1. Redistribute the CURRENT biomass
        if current_total_biomass > 0
            column_biomass = current_total_biomass * horizontal_proportion
            final_biomass = column_biomass * vertical_proportion
            new_biomass_grid[lon, lat, depth_idx, res_sp] = final_biomass
        end
        
        # 2. Redistribute the TOTAL CARRYING CAPACITY (K) to set the new capacity grid
        total_K = total_capacity_targets[res_sp]
        if total_K > 0
            column_capacity = total_K * horizontal_proportion
            final_capacity = column_capacity * vertical_proportion
            new_capacity_grid[lon, lat, depth_idx, res_sp] = final_capacity
        end
    end
end

function move_resources!(model::MarineModel, month::Int)
    arch = model.arch
    g = model.depths.grid
    depthres = Int(g.Value[findfirst(g.Name .== "depthres")])
    lonres = Int(g.Value[findfirst(g.Name .== "lonres")])
    latres = Int(g.Value[findfirst(g.Name .== "latres")])
    max_depth = Int(g.Value[findfirst(g.Name .== "depthmax")])
    n_res = model.n_resource
    n_sp = model.n_species

    total_biomass_per_sp_cpu = reshape(sum(Array(model.resources.biomass), dims=(1,2,3)), n_res)
    capacities_for_month_cpu = Array(@view model.capacities[:, :, month, (n_sp+1):(n_sp+n_res)])
    total_suitability_per_sp_cpu = reshape(sum(capacities_for_month_cpu, dims=(1,2)), n_res)

    lonmax = g[findfirst(g.Name .== "xurcorner"), :Value][1]
    lonmin = g[findfirst(g.Name .== "xllcorner"), :Value][1]
    latmax = g[findfirst(g.Name .== "yulcorner"), :Value][1]
    latmin = g[findfirst(g.Name .== "yllcorner"), :Value][1]
    mean_lat_rad = deg2rad((latmin + latmax) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(mean_lat_rad)
    area_km2 = abs(latmax - latmin) * km_per_deg_lat * abs(lonmax - lonmin) * km_per_deg_lon
    total_capacity_targets_cpu = Float32.(model.resource_trait.Biomass .* area_km2 .* 1e6)

    depth_res_m = max_depth / depthres
    depths_m = [(d - 0.5f0) * depth_res_m for d in 1:depthres]
    
    vertical_profiles_cpu = zeros(Float32, depthres, n_res)
    night_profs = model.depths.resource_night
    for res_sp in 1:n_res
        mu1 = night_profs[res_sp, "mu1"]; mu2 = night_profs[res_sp, "mu2"]
        sigma1 = night_profs[res_sp, "sigma1"]; sigma2 = night_profs[res_sp, "sigma2"]
        lambda1 = night_profs[res_sp, "lambda1"]
        
        profile = lambda1 .* pdf.(Normal(mu1, sigma1), depths_m) .+ (1-lambda1) .* pdf.(Normal(mu2, sigma2), depths_m)
        sum_prof = sum(profile)
        if sum_prof > 0
            vertical_profiles_cpu[:, res_sp] .= profile ./ sum_prof
        end
    end

    total_biomass_per_sp = array_type(arch)(total_biomass_per_sp_cpu)
    total_capacity_targets = array_type(arch)(total_capacity_targets_cpu)
    capacities_for_month = array_type(arch)(capacities_for_month_cpu)
    total_suitability_per_sp = array_type(arch)(total_suitability_per_sp_cpu)
    normalized_vertical_profiles = array_type(arch)(vertical_profiles_cpu)
    
    new_biomass_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res))
    new_capacity_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res))

    kernel_dims = (lonres, latres, depthres, Int64(n_res))
    kernel! = move_resources_kernel!(device(arch), (8, 8, 4, 1), kernel_dims)
    kernel!(new_biomass_grid, new_capacity_grid, capacities_for_month, total_biomass_per_sp, total_suitability_per_sp, normalized_vertical_profiles, total_capacity_targets)
    KernelAbstractions.synchronize(device(arch))

    model.resources.biomass .= new_biomass_grid
    model.resources.capacity .= new_capacity_grid
    return nothing
end

@kernel function sum_by_column_kernel!(column_totals, biomass_grid)
    lon, lat, sp = @index(Global, NTuple)
    total = 0.0f0
    for z in 1:size(biomass_grid, 3)
        total += biomass_grid[lon, lat, z, sp]
    end
    column_totals[lon, lat, sp] = total
end

@kernel function redistribute_by_profile_kernel!(biomass_grid, column_totals, normalized_profiles)
    lon, lat, depth, sp = @index(Global, NTuple)
    total_in_column = column_totals[lon, lat, sp]
    vertical_proportion = normalized_profiles[depth, sp]
    biomass_grid[lon, lat, depth, sp] = total_in_column * vertical_proportion
end

function vertical_resource_movement!(model::MarineModel)
    arch = model.arch; g = model.depths.grid; n_res = Int(model.n_resource)
    depthres = Int(g.Value[findfirst(g.Name .== "depthres")]); lonres = Int(g.Value[findfirst(g.Name .== "lonres")]); latres = Int(g.Value[findfirst(g.Name .== "latres")])
    is_night = model.t < 360 || model.t > 1080 
    profile_source = is_night ? model.depths.resource_night : model.depths.resource_day
    max_depth = Int(g.Value[findfirst(g.Name .== "depthmax")]); depth_res_m = max_depth / depthres
    depths_m = [(d - 0.5f0) * depth_res_m for d in 1:depthres]
    
    vertical_profiles_cpu = zeros(Float32, depthres, n_res)
    for res_sp in 1:n_res
        mu1 = profile_source[res_sp, "mu1"]; mu2 = profile_source[res_sp, "mu2"]
        sigma1 = profile_source[res_sp, "sigma1"]; sigma2 = profile_source[res_sp, "sigma2"]
        lambda1 = profile_source[res_sp, "lambda1"]
        profile = lambda1 .* pdf.(Normal(mu1, sigma1), depths_m) .+ (1-lambda1) .* pdf.(Normal(mu2, sigma2), depths_m)
        sum_prof = sum(profile)
        if sum_prof > 0; vertical_profiles_cpu[:, res_sp] .= profile ./ sum_prof; end
    end

    normalized_profiles_gpu = array_type(arch)(vertical_profiles_cpu)
    column_totals_gpu = array_type(arch)(zeros(Float32, lonres, latres, n_res))
    kernel_sum = sum_by_column_kernel!(device(arch), (8, 8, 1), (lonres, latres, n_res))
    kernel_sum(column_totals_gpu, model.resources.biomass)
    kernel_redist = redistribute_by_profile_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    kernel_redist(model.resources.biomass, column_totals_gpu, normalized_profiles_gpu)
    KernelAbstractions.synchronize(device(arch))
end