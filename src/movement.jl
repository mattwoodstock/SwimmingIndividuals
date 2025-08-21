# ===================================================================
# Pathfinding and Movement Helpers (CPU-based)
# ===================================================================

function find_path(capacity::Matrix{Float32}, start::Tuple{Int,Int}, goal::Tuple{Int,Int})
    open_set = [start]
    came_from = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    visited = Set{Tuple{Int,Int}}()
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    lonres, latres = size(capacity)
    
    while !isempty(open_set)
        current = popfirst!(open_set)
        if current == goal
            path = [current]
            while current in keys(came_from)
                current = came_from[current]
                pushfirst!(path, current)
            end
            return path
        end
        push!(visited, current)
        for (dx, dy) in directions
            nx, ny = current[1] + dx, current[2] + dy
            if 1 <= nx <= lonres && 1 <= ny <= latres
                neighbor = (nx, ny)
                if capacity[nx, ny] > 0 && !(neighbor in visited) && !(neighbor in open_set)
                    push!(open_set, neighbor)
                    came_from[neighbor] = current
                end
            end
        end
    end
    return nothing
end

function reachable_point(rng::AbstractRNG, current_pos::Tuple{Float32, Float32}, path::Vector{Tuple{Int, Int}}, max_distance::Float64, latmax::Float64, lonmin::Float64, cell_size::Float64, lonres::Int, latres::Int)
    function grid_to_coords(cell::Tuple{Int, Int})
        lon_idx, lat_idx = cell
        lon = lonmin + (lon_idx - 1 + rand(rng)) * cell_size
        flipped_lat_idx = (latres - lat_idx) + 1
        lat = latmax - (flipped_lat_idx - 1 + rand(rng)) * cell_size
        return (lat, lon)
    end

    total_distance = 0.0
    prev_lat, prev_lon = current_pos

    for i in 1:length(path)
        lat, lon = grid_to_coords(path[i])
        d = haversine(prev_lat, prev_lon, lat, lon)
        
        # If the animal has reached its maximum distance
        if total_distance + d > max_distance
            # A small epsilon to prevent numerical instability.
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
                # If d is too small, the animal has effectively not moved in this sub-step.
                # Return its previous position to avoid instability.
                grid_x = clamp(Int(floor((prev_lon - lonmin) / cell_size) + 1), 1, lonres)
                grid_y = clamp(latres - Int(floor((latmax - prev_lat) / cell_size)), 1, latres)
                return (prev_lat, prev_lon, grid_y, grid_x)
            end
        end
        
        total_distance += d
        prev_lat, prev_lon = lat, lon
    end

    # If the entire path is shorter than max_distance, move to the end of the path
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
        swim_speed = swim_speed_bls * agent_length_m
        swim_speed = max(0.01f0, swim_speed)
        swim_speed = 2.6f0
        z_increment = swim_speed * Float32(dt)
        
        is_daytime = (360.0f0 <= t < 1080.0f0)
        
        if is_daytime
            # --- DAYTIME: Descent is Mandatory ---
            # This logic is unconditional and remains correct.
            if my_status == 0.0f0      # If at surface, start descending
                mig_status[ind] = 2.0f0
                z[ind] = min(target_z[ind], my_z + z_increment)
            elseif my_status == 2.0f0  # If descending, continue
                z[ind] = min(target_z[ind], my_z + z_increment)
                if z[ind] >= target_z[ind]; mig_status[ind] = -1.0f0; end # Arrived at depth
            end
        else
            # --- NIGHTTIME: Logic is now split by status ---
            
            # 1. Decision Point: Only for animals at depth right at the decision point
            if my_status == -1.0f0 && (t < 1080.0f0 + dt) && (t >= 1080.0f0)
                hunger_threshold = rand(Float32)
                is_hungry = (gut_fullness[ind] < hunger_threshold)
                
                # Strong migrators always ascend. Weak migrators only ascend if hungry.
                can_ascend = !is_weak_migrator || is_hungry
                
                if can_ascend
                    # Begin the ascent
                    mig_status[ind] = 1.0f0
                    z[ind] = max(1.0f0, max(target_z[ind], my_z - z_increment))
                end
                # If can_ascend is false, the animal does nothing and remains at depth.

            # 2. Continuation: For animals already ascending.
            elseif my_status == 1.0f0
                z[ind] = max(1.0f0, max(target_z[ind], my_z - z_increment))
                if z[ind] <= target_z[ind]; mig_status[ind] = 0.0f0; end # Arrived at surface
            end
        end
        
        # Update pool_z regardless of movement
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
    
    t = model.t
    dt = model.dt

    # --- TARGET ASSIGNMENT LOGIC ---
    # A new target is assigned to ALL living agents at the precise transition times.
    
    # Check for Dawn Transition (6 AM)
    if t >= 360.0 && t < 360.0 + dt
        dist_params = model.depths.focal_day[sp, :]
        # Generate new DEEP targets for all living agents
        new_targets = gaussmix(length(data.x), 
                               dist_params.mu1, dist_params.mu2, dist_params.mu3,
                               dist_params.sigma1, dist_params.sigma2, dist_params.sigma3,
                               dist_params.lambda1, dist_params.lambda2)
        # Update the entire target_z array on the device
        copyto!(data.target_z, Float32.(new_targets))
    end

    # Check for Dusk Transition (6 PM)
    if t >= 1080.0 && t < 1080.0 + dt
        dist_params = model.depths.focal_night[sp, :]
        # Generate new SHALLOW targets for all living agents
        new_targets = gaussmix(length(data.x), 
                               dist_params.mu1, dist_params.mu2, dist_params.mu3,
                               dist_params.sigma1, dist_params.sigma2, dist_params.sigma3,
                               dist_params.lambda1, dist_params.lambda2)
        # Update the entire target_z array on the device
        copyto!(data.target_z, Float32.(new_targets))
    end
    
    # --- KERNEL LAUNCH ---
    kernel! = dvm_action_kernel!(device(arch), 256, (length(data.x),))
    
    kernel!(
        data.alive, data.mig_status, data.z, data.target_z, data.pool_z, data.active, data.gut_fullness,
        data.length,
        is_weak_migrator, 
        swim_speed_bls,
        maxdepth, depth_res_m, depthres, t, dt
    )
    
    KernelAbstractions.synchronize(device(arch))
end

# This is a GPU-compliant helper function used by the kernel.
@inline function gaussmix_gpu(
    mu1, mu2, mu3,
    sigma1, sigma2, sigma3,
    lambda1, lambda2
)
    # Generate three random numbers
    u1 = rand(Float32)
    u2 = rand(Float32)

    # Choose which Gaussian distribution to sample from
    if u1 < lambda1
        return mu1 + sigma1 * sqrt(-2.0f0 * log(u2)) * cos(2.0f0 * pi * u1)
    elseif u1 < lambda1 + lambda2
        return mu2 + sigma2 * sqrt(-2.0f0 * log(u2)) * cos(2.0f0 * pi * u1)
    else
        return mu3 + sigma3 * sqrt(-2.0f0 * log(u2)) * cos(2.0f0 * pi * u1)
    end
end

# This is a GPU-compliant helper to get a random value from a uniform distribution.
@inline function uniform_rand_depth_gpu(min_depth::Float32, max_depth::Float32)
    return min_depth + rand(Float32) * (max_depth - min_depth)
end

@kernel function dive_action_kernel!(
    alive, mig_status, interval, z, gut_fullness,
    biomass_school, length_arr, target_z, pool_z, active,
    surface_interval_mean, dive_interval_mean, swim_velo, 
    day_mu1, day_mu2, day_mu3, day_sigma1, day_sigma2, day_sigma3, day_lambda1, day_lambda2,
    night_mu1, night_mu2, night_mu3, night_sigma1, night_sigma2, night_sigma3, night_lambda1, night_lambda2,
    day_min_z, day_max_z, night_min_z, night_max_z,
    depth_res_m, depthres, t, dt
)
    ind = @index(Global)
    @inbounds if alive[ind] == 1.0
        my_status = mig_status[ind]
        my_interval = interval[ind]
        my_z = z[ind]
        my_fullness = gut_fullness[ind]
        
        dive_velo_ms = swim_velo * (length_arr[ind] / 1000.0f0) * 60.0f0
        z_increment = dive_velo_ms * Float32(dt)
        is_daytime = (360.0f0 <= t < 1080.0f0)

        if my_status == 0.0f0
            if my_interval == 0.0f0
                rand_interval = surface_interval_mean + randn(Float32) * (surface_interval_mean * 0.2f0)
                target_z[ind] = max(1.0f0, rand_interval)
            end

            my_interval += dt
            target_surface_interval = target_z[ind]

            hunger_threshold = rand(Float32)
            is_hungry = (my_fullness < hunger_threshold)
                        
            if my_interval >= target_surface_interval && is_hungry
                mig_status[ind] = 1.0f0
                interval[ind] = 0.0f0

                new_target = my_z
                attempts = 0

                while new_target <= my_z && attempts < 10
                    new_target = if is_daytime
                        uniform_rand_depth_gpu(day_min_z,day_max_z)
                    else
                        uniform_rand_depth_gpu(night_min_z,night_max_z)
                    end
                    attempts += 1
                end
                target_z[ind] = max(1.0f0, new_target)
            else
                interval[ind] = my_interval
            end

        elseif my_status == 1.0f0
            z[ind] = min(target_z[ind], my_z + z_increment)
            if z[ind] >= target_z[ind]
                mig_status[ind] = -1.0f0
                interval[ind] = 0.0f0
            end
            @inbounds active[ind] += dt


        elseif my_status == -1.0f0
            if my_interval == 0.0f0
                rand_interval = dive_interval_mean + randn(Float32) * (dive_interval_mean * 0.2f0)
                target_z[ind] = max(1.0f0, rand_interval)
            end

            my_interval += dt
            target_dive_interval = target_z[ind]

            if my_interval >= target_dive_interval
                mig_status[ind] = 2.0f0
                interval[ind] = 0.0f0
            else
                interval[ind] = my_interval
            end

        elseif my_status == 2.0f0
            z[ind] = max(1.0f0, my_z - z_increment)
            if z[ind] <= 1.0f0
                mig_status[ind] = 3.0f0
                z[ind] = 1.0f0
                interval[ind] = 0.0f0
            end
            @inbounds active[ind] += dt
        
        elseif my_status == 3.0f0
            new_resting_depth = is_daytime ?
                gaussmix_gpu(day_mu1, day_mu2, day_mu3, day_sigma1, day_sigma2, day_sigma3, day_lambda1, day_lambda2) :
                gaussmix_gpu(night_mu1, night_mu2, night_mu3, night_sigma1, night_sigma2, night_sigma3, night_lambda1, night_lambda2)
            
            z[ind] = max(1.0f0, new_resting_depth)
            
            mig_status[ind] = 0.0f0
            interval[ind] = 0.0f0
        end
        
        @inbounds pool_z[ind] = clamp(ceil(Int, z[ind] / depth_res_m), 1, depthres)
    end
end

function dive_action!(model::MarineModel, sp::Int)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p_cpu = model.individuals.animals[sp].p
    
    # --- 1. Gather all necessary parameters ---
    surface_interval_mean = p_cpu.Surface_Interval.second[sp]
    dive_interval_mean = p_cpu.Dive_Interval.second[sp]
    swim_velo = p_cpu.Swim_velo.second[sp]

    day_min_z = Float32(p_cpu.Dive_Min_Day.second[sp])
    day_max_z = Float32(p_cpu.Dive_Max_Day.second[sp])
    night_min_z = Float32(p_cpu.Dive_Min_Night.second[sp])
    night_max_z = Float32(p_cpu.Dive_Max_Night.second[sp])
    
    day_dist_params = model.depths.focal_day[sp, :]
    night_dist_params = model.depths.focal_night[sp, :]
    
    # Extract all parameters for the daytime distribution
    day_mu1 = Float32(day_dist_params.mu1); day_mu2 = Float32(day_dist_params.mu2); day_mu3 = Float32(day_dist_params.mu3)
    day_sigma1 = Float32(day_dist_params.sigma1); day_sigma2 = Float32(day_dist_params.sigma2); day_sigma3 = Float32(day_dist_params.sigma3)
    day_lambda1 = Float32(day_dist_params.lambda1); day_lambda2 = Float32(day_dist_params.lambda2)

    # Extract all parameters for the nighttime distribution
    night_mu1 = Float32(night_dist_params.mu1); night_mu2 = Float32(night_dist_params.mu2); night_mu3 = Float32(night_dist_params.mu3)
    night_sigma1 = Float32(night_dist_params.sigma1); night_sigma2 = Float32(night_dist_params.sigma2); night_sigma3 = Float32(night_dist_params.sigma3)
    night_lambda1 = Float32(night_dist_params.lambda1); night_lambda2 = Float32(night_dist_params.lambda2)
    # ==========================================================
    
    grid = model.depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / depthres

    # --- 2. Launch the Dive Action Kernel ---
    kernel! = dive_action_kernel!(device(arch), 256, (length(data.x),))
    kernel!(
        data.alive, data.mig_status, data.interval, data.z,
        data.gut_fullness, data.biomass_school, data.length, data.target_z,
        data.pool_z, data.active,
        surface_interval_mean, dive_interval_mean, swim_velo, 
        # Pass all the unpacked scalar parameters instead of the DataFrameRows
        day_mu1, day_mu2, day_mu3, day_sigma1, day_sigma2, day_sigma3, day_lambda1, day_lambda2,
        night_mu1, night_mu2, night_mu3, night_sigma1, night_sigma2, night_sigma3, night_lambda1, night_lambda2,
        day_min_z, day_max_z, night_min_z, night_max_z,
        depth_res_m, depthres, model.t, model.dt
    )
    KernelAbstractions.synchronize(device(arch))
end

# ===================================================================
# General Habitat-Seeking Movement
# ===================================================================

function movement_toward_habitat!(model::MarineModel, sp::Int, time::AbstractArray)
    # --- 1. GATHER DATA FROM DEVICE TO CPU ---
    arch = model.arch
    animal_data = model.individuals.animals[sp].data
    animal_param = model.individuals.animals[sp].p
    
    # Copy all necessary position data, including Z and the new target coordinates
    cpu_time = Array(time)
    cpu_x, cpu_y, cpu_z = Array(animal_data.x), Array(animal_data.y), Array(animal_data.z)
    cpu_pool_x, cpu_pool_y, cpu_pool_z = Array(animal_data.pool_x), Array(animal_data.pool_y), Array(animal_data.pool_z)
    
    # Assumes these fields have been added to your agent's data structure
    cpu_target_pool_x, cpu_target_pool_y = Array(animal_data.target_pool_x), Array(animal_data.target_pool_y)
    
    cpu_length = Array(animal_data.length)
    cpu_alive = Array(animal_data.alive)
    cpu_active = Array(animal_data.active)

    # Create arrays to store the new positions
    new_x, new_y, new_z = copy(cpu_x), copy(cpu_y), copy(cpu_z)
    new_pool_x, new_pool_y, new_pool_z = copy(cpu_pool_x), copy(cpu_pool_y), copy(cpu_pool_z)
    new_target_pool_x, new_target_pool_y = copy(cpu_target_pool_x), copy(cpu_target_pool_y)
    
    # Get grid parameters
    month = model.environment.ts
    habitat = Array(model.capacities[:, :, month, sp])
    grid = model.depths.grid
    latmax = grid[grid.Name .== "yulcorner", :Value][1]
    lonmin = grid[grid.Name .== "xllcorner", :Value][1]
    cell_size = grid[grid.Name .== "cellsize", :Value][1]
    lonres, latres = size(habitat)
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    depth_res_m = maxdepth / depthres

    habitat_cells = [(x=i, y=j, value=habitat[i,j]) for i in 1:lonres, j in 1:latres if habitat[i,j] > 0]
    
    if !isempty(habitat_cells)
        sort!(habitat_cells, by = x -> x.value, rev = true)
        cumvals = cumsum(getfield.(habitat_cells, :value))
        total_val = cumvals[end]

        rngs = [Random.Xoshiro() for _ in 1:Threads.nthreads()]

        # --- 2. PERFORM MOVEMENT LOGIC ON CPU ---
        Threads.@threads for ind in 1:length(cpu_x)
            if cpu_alive[ind] == 1.0 && cpu_time[ind] > 0
                rng = rngs[Threads.threadid()]

                start_x, start_y = cpu_pool_x[ind], cpu_pool_y[ind]

                cur_x, cur_y = cpu_x[ind], cpu_y[ind]
                len_m = cpu_length[ind] / 1000.0
                swim_speed_ms = animal_param.Swim_velo[2][sp] * len_m
                max_dist = swim_speed_ms * cpu_time[ind]

                vertical_shift = (rand(Float32) * 10.0f0) - 5.0f0
                vertical_shift = clamp(vertical_shift, -max_dist, max_dist)
                max_horizontal_dist = max(0.0, max_dist - abs(vertical_shift))

                potential_new_z = cpu_z[ind] + vertical_shift
                new_z[ind] = clamp(potential_new_z, 1.0f0, Float32(maxdepth))
                new_pool_z[ind] = clamp(ceil(Int, new_z[ind] / depth_res_m), 1, depthres)

                is_invalid_start = !(1 <= start_x <= lonres && 1 <= start_y <= latres)
                if is_invalid_start
                    # 1. Print a warning for diagnosis
                    @warn "Agent $(ind) of species $sp has corrupted coordinates: ($start_x, $start_y). Attempting to relocate."
                end

                if habitat[start_x, start_y] == 0
                    res = nearest_suitable_habitat(rng, habitat, (cur_y, cur_x), (start_x, start_y), max_horizontal_dist, latmax, lonmin, cell_size)
                    if res !== nothing
                        new_y[ind], new_x[ind], new_pool_x[ind], new_pool_y[ind] = res
                    end
                else
                    # 1. Check if the agent needs a new long-term target.
                    has_no_target = (cpu_target_pool_x[ind] < 1 || cpu_target_pool_y[ind] < 1)
                    has_reached_target = (start_x == cpu_target_pool_x[ind] && start_y == cpu_target_pool_y[ind])

                    if has_no_target || has_reached_target
                        # Select a new long-term target and store it.
                        r = (rand(rng)^4) * total_val
                        idx = findfirst(x -> x >= r, cumvals)
                        if idx !== nothing
                            new_target_pool_x[ind] = habitat_cells[idx].x
                            new_target_pool_y[ind] = habitat_cells[idx].y
                        end
                    end
                    
                    # 2. Find the path to the current long-term target.
                    target_x_int = Int(new_target_pool_x[ind])
                    target_y_int = Int(new_target_pool_y[ind])
                    long_term_target = (target_x_int, target_y_int)
                    path = find_path(habitat, (start_x, start_y), long_term_target)
                    
                    # 3. Take a single step along the path.
                    if path !== nothing && length(path) > 1
                        # Create a short path representing just the next step
                        next_step_path = [path[1], path[2]]
                        
                        # Move the maximum possible horizontal distance along this single step
                        res_lat, res_lon, res_pool_y, res_pool_x = reachable_point(rng, (cur_y, cur_x), next_step_path, max_horizontal_dist, latmax, lonmin, cell_size, lonres, latres)
                        
                        new_y[ind] = res_lat
                        new_x[ind] = res_lon
                        new_pool_y[ind] = res_pool_y
                        new_pool_x[ind] = res_pool_x

                    end
                end
            end
        end
    end

    # --- 3. UPDATE DEVICE ARRAYS WITH RESULTS ---
    copyto!(animal_data.x, new_x)
    copyto!(animal_data.y, new_y)
    copyto!(animal_data.z, new_z)
    copyto!(animal_data.pool_x, new_pool_x)
    copyto!(animal_data.pool_y, new_pool_y)
    copyto!(animal_data.pool_z, new_pool_z)
    copyto!(animal_data.target_pool_x, new_target_pool_x)
    copyto!(animal_data.target_pool_y, new_target_pool_y)
    copyto!(animal_data.active, cpu_active)

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

    # --- 1. PREPARE DISTRIBUTION DATA ON CPU ---
    
    # Get the CURRENT total biomass for each species by summing the grid
    total_biomass_per_sp_cpu = reshape(sum(Array(model.resources.biomass), dims=(1,2,3)), n_res)

    # Get the habitat capacity map for the NEW month
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

    # Calculate the normalized vertical distribution profile for each species
    depth_res_m = max_depth / depthres
    depths_m = [(d - 0.5f0) * depth_res_m for d in 1:depthres]
    
    vertical_profiles_cpu = zeros(Float32, depthres, n_res)
    night_profs = model.depths.resource_night # Assuming night profile for redistribution
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

    # --- 2. UPLOAD DATA TO GPU ---
    total_biomass_per_sp = array_type(arch)(total_biomass_per_sp_cpu)
    total_capacity_targets = array_type(arch)(total_capacity_targets_cpu)
    capacities_for_month = array_type(arch)(capacities_for_month_cpu)
    total_suitability_per_sp = array_type(arch)(total_suitability_per_sp_cpu)
    normalized_vertical_profiles = array_type(arch)(vertical_profiles_cpu)
    
    new_biomass_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res))
    new_capacity_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res))

    # --- 3. LAUNCH THE KERNEL ---
    kernel_dims = (lonres, latres, depthres, Int64(n_res))
    kernel! = move_resources_kernel!(device(arch), (8, 8, 4, 1), kernel_dims)
    kernel!(
        new_biomass_grid,
        new_capacity_grid,
        capacities_for_month,
        total_biomass_per_sp,
        total_suitability_per_sp,
        normalized_vertical_profiles,
        total_capacity_targets
    )
    KernelAbstractions.synchronize(device(arch))

    # --- 4. UPDATE THE MODEL STATE ---
    model.resources.biomass .= new_biomass_grid
    model.resources.capacity .= new_capacity_grid
    
    return nothing
end

@kernel function sum_by_column_kernel!(
    column_totals,
    biomass_grid,
)
    lon, lat, sp = @index(Global, NTuple)
    
    # Only perform the sum for species that migrate
    total = 0.0f0
    for z in 1:size(biomass_grid, 3)
        total += biomass_grid[lon, lat, z, sp]
    end
    column_totals[lon, lat, sp] = total
end

@kernel function redistribute_by_profile_kernel!(
    biomass_grid,
    column_totals,
    normalized_profiles,
)
    lon, lat, depth, sp = @index(Global, NTuple)

    # Only perform the redistribution for species that migrate
    total_in_column = column_totals[lon, lat, sp]
    vertical_proportion = normalized_profiles[depth, sp]
        
    # Assign the new biomass based on the total and the profile
    biomass_grid[lon, lat, depth, sp] = total_in_column * vertical_proportion
end

function vertical_resource_movement!(model::MarineModel)
    arch = model.arch
    g = model.depths.grid
    n_res = Int(model.n_resource)
    depthres = Int(g.Value[findfirst(g.Name .== "depthres")])
    lonres = Int(g.Value[findfirst(g.Name .== "lonres")])
    latres = Int(g.Value[findfirst(g.Name .== "latres")])

    # --- 1. Determine Time of Day and Select Profiles ---
    # (Assuming model.t is minutes from midnight, 0-1440)
    is_night = model.t < 360 || model.t > 1080 # (Before 6 AM or after 6 PM)
    
    profile_source = is_night ? model.depths.resource_night : model.depths.resource_day

    # --- 2. Prepare Normalized Profiles on CPU ---
    max_depth = Int(g.Value[findfirst(g.Name .== "depthmax")])
    depth_res_m = max_depth / depthres
    depths_m = [(d - 0.5f0) * depth_res_m for d in 1:depthres]
    
    vertical_profiles_cpu = zeros(Float32, depthres, n_res)
    for res_sp in 1:n_res
        mu1 = profile_source[res_sp, "mu1"]; mu2 = profile_source[res_sp, "mu2"]
        sigma1 = profile_source[res_sp, "sigma1"]; sigma2 = profile_source[res_sp, "sigma2"]
        lambda1 = profile_source[res_sp, "lambda1"]
        
        profile = lambda1 .* pdf.(Normal(mu1, sigma1), depths_m) .+ (1-lambda1) .* pdf.(Normal(mu2, sigma2), depths_m)
        sum_prof = sum(profile)
        if sum_prof > 0
            vertical_profiles_cpu[:, res_sp] .= profile ./ sum_prof
        end
    end

    # --- 3. Upload Data to GPU ---
    normalized_profiles_gpu = array_type(arch)(vertical_profiles_cpu)
    column_totals_gpu = array_type(arch)(zeros(Float32, lonres, latres, n_res))

    # --- 4. Launch Kernels ---
    # Kernel 1: Sum biomass in each column for migrating species
    kernel_sum = sum_by_column_kernel!(device(arch), (8, 8, 1), (lonres, latres, n_res))
    kernel_sum(column_totals_gpu, model.resources.biomass)
    
    # Kernel 2: Redistribute the summed biomass according to the new profile
    kernel_redist = redistribute_by_profile_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    kernel_redist(model.resources.biomass, column_totals_gpu, normalized_profiles_gpu)

    KernelAbstractions.synchronize(device(arch))
end