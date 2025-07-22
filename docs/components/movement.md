## Movement Processes

The `movement.jl` file contains all the functions that govern how agents move through the model world. This includes both the default habitat-seeking behavior and the specialized, archetype-specific movements like Diel Vertical Migration (DVM) and the foraging dives of pelagic animals.

### 1. Pathfinding and Movement Helpers

These are CPU-based helper functions that provide the core logic for the general habitat-seeking behavior. They handle tasks like finding a valid path through the grid and calculating how far an agent can travel along that path in a single timestep.

#### `find_path(...)`
This function implements a Breadth-First Search (BFS) algorithm to find the shortest path between two grid cells (`start` and `goal`). Crucially, the path is constrained to only travel through cells that have a habitat capacity greater than zero, preventing agents from moving across land or other unsuitable areas.

#### `reachable_point(...)`
Once a path has been found, this function determines the exact geographic coordinates an agent can reach along that path within a single timestep. It calculates the total distance of the path and compares it to the maximum distance the agent can travel (based on its swim speed). If the path is longer than the agent can travel, it interpolates to find the precise point along the path where the agent's movement ends.

#### `nearest_suitable_habitat(...)`
This function is a failsafe for agents that find themselves in a grid cell with zero habitat capacity. It performs a search outward from the agent's current location to find the closest valid habitat cell and calculates a new position within that cell.

#### `bl_per_s(...)`
A simple helper function to calculate an agent's swim speed in body lengths per second, based on its size and a species-specific speed parameter.

```julia
# ===================================================================
# Pathfinding and Movement Helpers (CPU-based)
# ===================================================================

function find_path(capacity::Matrix{Float64}, start::Tuple{Int,Int}, goal::Tuple{Int,Int})
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

function reachable_point(current_pos::Tuple{Float64, Float64}, path::Vector{Tuple{Int, Int}}, max_distance::Float64, latmax::Float64, lonmin::Float64, cell_size::Float64, lonres::Int, latres::Int)
    function grid_to_coords(cell::Tuple{Int, Int})
        lon_idx, lat_idx = cell
        lon = lonmin + (lon_idx - 1 + rand()) * cell_size
        lat = latmax - (lat_idx - 1 + rand()) * cell_size 
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
            
            grid_x = clamp(Int(floor((interp_lon - lonmin) / cell_size) + 1), 1, lonres)
            grid_y = clamp(Int(floor((latmax - interp_lat) / cell_size) + 1), 1, latres)
            
            return (interp_lat, interp_lon, grid_y, grid_x)
        end
        prev_lat, prev_lon = lat, lon
    end

    final_cell = path[end]
    final_lat, final_lon = grid_to_coords(final_cell)
    return (final_lat, final_lon, final_cell[2], final_cell[1])
end

function nearest_suitable_habitat(habitat::Matrix{Float64}, start_latlon::Tuple{Float64, Float64}, start_pool::Tuple{Int,Int}, max_distance_m::Float64, latmax::Float64, lonmin::Float64, cellsize_deg::Float64)
    R = 6371000.0
    lonres, latres = size(habitat)
    
    get_neighbors(x, y) = [(x+dx, y+dy) for (dx, dy) in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)) if 1<=x+dx<=lonres && 1<=y+dy<=latres]
    random_point_in_cell(cell) = (latmax - (cell[2]-1+rand())*cellsize_deg, lonmin+(cell[1]-1+rand())*cellsize_deg)

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
    new_lat_idx = clamp(floor(Int, (latmax - new_latlon[1]) / cellsize_deg) + 1, 1, latres)
    
    return new_latlon[1], new_latlon[2], new_lon_idx, new_lat_idx
end

function bl_per_s(length, speed; b=0.35, min_speed = 0.5)
    return max.(speed .* length .^ (-b), min_speed)
end
```

### 2. Specialized Movement Kernels
These are high-performance GPU kernels that manage the state-machine logic for specialized behavioral archetypes.

#### dvm_action!
This system manages Diel Vertical Migration. The launcher function (dvm_action!) first determines if it is day or night and pre-calculates a random target depth for each agent based on the species' day/night vertical distribution profiles. It then calls the dvm_action_kernel!, which runs in parallel for all agents, moving them towards their assigned target depth and updating their migration status.

#### dive_action!
This system manages the behavior of air-breathing divers. The dive_action_kernel! implements a state machine for each agent, cycling through states like "at surface," "descending," "foraging at depth," and "ascending." The decision to dive is probabilistic and is based on the agent's gut fullness and the time it has spent at the surface.

```julia
# ===================================================================
# Specialized Movement Kernels (DVM and Diving)
# ===================================================================

@kernel function dvm_action_kernel!(
    alive, mig_status, z, target_z, pool_z, active,
    p_gpu, maxdepth, depth_res_m, depthres, t, dt, sp
)
    ind = @index(Global)
    @inbounds if alive[ind] == 1.0
        my_status = mig_status[ind]
        my_z = z[ind]
        swim_speed = 2.68f0
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
            if my_status == -1.0f0
                mig_status[ind] = 1.0f0
                z[ind] = max(target_z[ind], my_z - z_increment)
            elseif my_status == 1.0f0
                z[ind] = max(target_z[ind], my_z - z_increment)
                if z[ind] <= target_z[ind]; mig_status[ind] = 0.0f0; end
            end
        end
        
        @inbounds pool_z[ind] = clamp(ceil(Int, z[ind] / depth_res_m), 1, depthres)
        @inbounds active[ind] += dt
    end
end

function dvm_action!(model::MarineModel, sp::Int)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p_cpu = model.individuals.animals[sp].p
    
    p_gpu = (; (k => array_type(arch)(v.second) for (k, v) in pairs(p_cpu))...)
    
    grid = model.depths.grid
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    depth_res_m = maxdepth / depthres
    
    is_daytime = (360.0 <= model.t < 1080.0)
    dist_params = is_daytime ? model.depths.focal_day[sp, :] : model.depths.focal_night[sp, :]
    
    target_z_cpu = gaussmix(length(data.x), dist_params.mu1, dist_params.mu2, dist_params.mu3,
                                           dist_params.sigma1, dist_params.sigma2, dist_params.sigma3,
                                           dist_params.lambda1, dist_params.lambda2)
    copyto!(data.target_z, Float32.(target_z_cpu))

    kernel! = dvm_action_kernel!(device(arch), 256, (length(data.x),))
    kernel!(
        data.alive, data.mig_status, data.z, data.target_z, data.pool_z, data.active,
        p_gpu, maxdepth, depth_res_m, depthres, model.t, model.dt, sp
    )
    KernelAbstractions.synchronize(device(arch))
end

@kernel function dive_action_kernel!(
    alive, mig_status, interval, z, dives_remaining, gut_fullness,
    biomass_school, length_arr, target_z, pool_z, active,
    surface_interval, dive_interval, swim_velo, dive_max, dive_min,
    depth_res_m, depthres, t, dt
)
    ind = @index(Global)
    @inbounds if alive[ind] == 1.0
        my_status = mig_status[ind]
        my_interval = interval[ind]
        my_z = z[ind]
        my_dives_remaining = dives_remaining[ind]
        my_fullness = gut_fullness[ind]
        my_biomass = biomass_school[ind]
        
        dive_velo_ms = swim_velo * (length_arr[ind] / 1000.0f0) * 60.0f0
        z_increment = dive_velo_ms * Float32(dt)

        if my_status == 0.0f0 && my_dives_remaining > 0
            my_interval += dt
            max_fullness = 0.2f0 * my_biomass
            fullness_ratio = max_fullness > 0 ? my_fullness / max_fullness : 1.0f0
            dive_prob = 1.0f0 - (1.0f0 / (1.0f0 + exp(-5.0f0 * (fullness_ratio - 0.5f0))))
            if my_interval >= surface_interval && rand(Float32) < dive_prob
                mig_status[ind] = 1.0f0
                interval[ind] = 0.0f0
                target_z[ind] = rand(Float32) * (dive_max - dive_min) + dive_min
            else
                interval[ind] = my_interval
            end
        elseif my_status == 1.0f0
            z[ind] = min(target_z[ind], my_z + z_increment)
            if z[ind] >= target_z[ind]; mig_status[ind] = -1.0f0; end
        elseif my_status == -1.0f0
            my_interval += dt
            if my_interval >= dive_interval
                mig_status[ind] = 2.0f0
                interval[ind] = 0.0f0
            else
                interval[ind] = my_interval
            end
        elseif my_status == 2.0f0
            z[ind] = max(1.0f0, my_z - z_increment)
            if z[ind] <= 1.0f0
                mig_status[ind] = 0.0f0
                dives_remaining[ind] -= 1
            end
        end
        
        @inbounds pool_z[ind] = clamp(ceil(Int, z[ind] / depth_res_m), 1, depthres)
        @inbounds active[ind] += dt
    end
end

function dive_action!(model::MarineModel, sp::Int)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p_cpu = model.individuals.animals[sp].p
    
    surface_interval = p_cpu.Surface_Interval.second[sp]
    dive_interval = p_cpu.Dive_Interval.second[sp]
    swim_velo = p_cpu.Swim_velo.second[sp]
    dive_max = p_cpu.Dive_Max.second[sp]
    dive_min = p_cpu.Dive_Min.second[sp]
    
    grid = model.depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / depthres

    kernel! = dive_action_kernel!(device(arch), 256, (length(data.x),))
    kernel!(
        data.alive, data.mig_status, data.interval, data.z, data.dives_remaining,
        data.gut_fullness, data.biomass_school, data.length, data.target_z,
        data.pool_z, data.active,
        surface_interval, dive_interval, swim_velo, dive_max, dive_min,
        depth_res_m, depthres, model.t, model.dt
    )
    KernelAbstractions.synchronize(device(arch))
end
```

### 3. General Habitat-Seeking Movement
This is the default movement behavior for agents that are not engaged in a specialized activity. It uses the "Gather, Compute, Update" pattern to remain architecture-compliant.

#### movement_toward_habitat!(model, sp, time)
This function orchestrates the habitat-seeking process. It first gathers all necessary agent and environmental data from the GPU to the CPU. It then uses multi-threading to loop over each agent. For each agent, it determines a target location based on habitat quality and calculates a new position using the pathfinding helper functions. Finally, it copies the updated positions and active time back to the GPU.

```julia
# ===================================================================
# General Habitat-Seeking Movement
# ===================================================================

function movement_toward_habitat!(model::MarineModel, sp::Int, time::AbstractArray)
    # --- 1. GATHER DATA FROM DEVICE TO CPU ---
    arch = model.arch
    animal_data = model.individuals.animals[sp].data
    animal_param = model.individuals.animals[sp].p
    
    cpu_time = Array(time)
    cpu_x, cpu_y = Array(animal_data.x), Array(animal_data.y)
    cpu_pool_x, cpu_pool_y = Array(animal_data.pool_x), Array(animal_data.pool_y)
    cpu_length = Array(animal_data.length)
    cpu_alive = Array(animal_data.alive)
    cpu_active = Array(animal_data.active)

    new_x, new_y = copy(cpu_x), copy(cpu_y)
    new_pool_x, new_pool_y = copy(cpu_pool_x), copy(cpu_pool_y)
    
    month = model.environment.ts
    # The habitat array is (lon, lat)
    habitat = Array(model.capacities[:, :, month, sp])
    grid = model.depths.grid
    latmax = grid[grid.Name .== "yulcorner", :Value][1]
    lonmin = grid[grid.Name .== "xllcorner", :Value][1]
    cell_size = grid[grid.Name .== "cellsize", :Value][1]
    lonres, latres = size(habitat)

    # Precompute list of all valid habitat cells across the entire grid
    habitat_cells = [(x=i, y=j, value=habitat[i,j]) for i in 1:lonres, j in 1:latres if habitat[i,j] > 0]
    
    if !isempty(habitat_cells)
        sort!(habitat_cells, by = x -> x.value, rev = true)
        cumvals = cumsum(getfield.(habitat_cells, :value))
        total_val = cumvals[end]

        # --- 2. PERFORM MOVEMENT LOGIC ON CPU ---
        Threads.@threads for ind in 1:length(cpu_x)
            if cpu_alive[ind] == 1.0 && cpu_time[ind] > 0
                
                cpu_active[ind] += cpu_time[ind] / 60.0

                start_x, start_y = cpu_pool_x[ind], cpu_pool_y[ind]
                cur_x, cur_y = cpu_x[ind], cpu_y[ind]
                len_m = cpu_length[ind] / 1000.0
                swim_speed_ms = bl_per_s(len_m*100, animal_param.Swim_velo[2][sp]) * len_m
                max_dist = swim_speed_ms * cpu_time[ind]

                # If agent is in unsuitable habitat, find the NEAREST valid cell
                if habitat[start_x, start_y] == 0
                    res = nearest_suitable_habitat(habitat, (cur_y, cur_x), (start_x, start_y), max_dist, latmax, lonmin, cell_size)
                    if res !== nothing
                        # Note: nearest_suitable_habitat returns (lat, lon, lon_idx, lat_idx)
                        new_y[ind], new_x[ind], new_pool_x[ind], new_pool_y[ind] = res
                    end
                else
                    # Agent is in suitable habitat, so it will pick a better spot globally
                    r = (rand()^4) * total_val
                    idx = findfirst(x -> x >= r, cumvals)
                    if idx !== nothing
                        target_x, target_y = habitat_cells[idx].x, habitat_cells[idx].y
                        
                        path = find_path(habitat, (start_x, start_y), (target_x, target_y))
                        
                        if path !== nothing && !isempty(path)
                            # reachable_point returns (lat, lon, lat_idx, lon_idx)
                            res_lat, res_lon, res_pool_y, res_pool_x = reachable_point((cur_y, cur_x), path, max_dist, latmax, lonmin, cell_size, lonres, latres)
                            
                            # Assign to the correct variables
                            new_y[ind] = res_lat
                            new_x[ind] = res_lon
                            new_pool_y[ind] = res_pool_y
                            new_pool_x[ind] = res_pool_x
                        end
                    end
                end
            end
        end
    end

    # --- 3. UPDATE DEVICE ARRAYS WITH RESULTS ---
    copyto!(animal_data.x, new_x)
    copyto!(animal_data.y, new_y)
    copyto!(animal_data.pool_x, new_pool_x)
    copyto!(animal_data.pool_y, new_pool_y)
    copyto!(animal_data.active, cpu_active)

    return nothing
end
```

### 4. Monthly Resource Redistribution
This system is responsible for the monthly "shuffling" of the resource grids.

#### move_resources!(model, month)
This high-level function is called from the main TimeStep! loop whenever a new month begins. It first calculates the total biomass for each resource species currently in the model. It then calls the move_resources_kernel! to redistribute this total biomass across the grid according to the habitat capacity map for the new month. This process ensures that resource availability tracks the seasonal changes in the environment.

```julia
# ===================================================================
# Monthly Resource Redistribution System
# ===================================================================

@kernel function move_resources_kernel!(
    new_biomass_grid, capacities_for_month, 
    total_biomass_per_sp, total_capacity_per_sp
)
    lon, lat, depth = @index(Global, NTuple)
    for sp in 1:size(new_biomass_grid, 4)
        total_biomass = total_biomass_per_sp[sp]
        total_capacity = total_capacity_per_sp[sp]
        if total_biomass > 0 && total_capacity > 0
            capacity_this_cell = capacities_for_month[lon, lat, sp]
            if depth <= 5
                new_biomass = (capacity_this_cell / total_capacity) * total_biomass / 5.0
                @inbounds new_biomass_grid[lon, lat, depth, sp] = new_biomass
            else
                @inbounds new_biomass_grid[lon, lat, depth, sp] = 0.0
            end
        else
            @inbounds new_biomass_grid[lon, lat, depth, sp] = 0.0
        end
    end
end

function move_resources!(model::MarineModel, month::Int)
    arch = model.arch
    g = model.depths.grid
    lonres = Int(g[g.Name .== "lonres", :Value][1])
    latres = Int(g[g.Name .== "latres", :Value][1])
    depthres = Int(g[g.Name .== "depthres", :Value][1])
    n_res = model.n_resource
    n_sp = model.n_species

    total_biomass_per_sp = reshape(sum(model.resources.biomass, dims=(1,2,3)), n_res)
    capacities_for_month = array_type(arch)(@view model.capacities[:, :, month, (n_sp+1):(n_sp+n_res)])
    total_capacity_per_sp = reshape(sum(capacities_for_month, dims=(1,2)), n_res)
    new_biomass_grid = array_type(arch)(zeros(Float64, lonres, latres, depthres, n_res))

    kernel! = move_resources_kernel!(device(arch), (8,8,4), (lonres, latres, depthres))
    kernel!(new_biomass_grid, capacities_for_month, total_biomass_per_sp, total_capacity_per_sp)
    KernelAbstractions.synchronize(device(arch))

    model.resources.biomass .= new_biomass_grid
    model.resources.capacity .= new_biomass_grid
    return nothing
end
```