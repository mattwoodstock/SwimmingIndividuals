# ===================================================================
# Pathfinding and Movement Helpers (CPU-based)
# ===================================================================

function find_path(capacity::Matrix{Float64}, start::Tuple{Int,Int}, goal::Tuple{Int,Int})
    open_set = [start]
    came_from = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    visited = Set{Tuple{Int,Int}}()
    # Directions are (d_lon, d_lat)
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
                # Access capacity with correct (lon, lat) order
                if capacity[nx, ny] > 0 && !(neighbor in visited) && !(neighbor in open_set)
                    push!(open_set, neighbor)
                    came_from[neighbor] = current
                end
            end
        end
    end
    return nothing # no path found
end

function reachable_point(current_pos::Tuple{Float64, Float64}, path::Vector{Tuple{Int, Int}}, max_distance::Float64, latmax::Float64, lonmin::Float64, cell_size::Float64, lonres::Int, latres::Int)
    # This helper function now correctly converts from grid index to geographic coordinates,
    # assuming that the first row of the grid corresponds to the maximum latitude.
    function grid_to_coords(cell::Tuple{Int, Int})
        lon_idx, lat_idx = cell
        
        # FIX: Use `latmax` and subtract to correctly handle top-down latitude indexing.
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
            # The latitude index is now calculated from latmax
            grid_y = clamp(Int(floor((latmax - interp_lat) / cell_size) + 1), 1, latres)
            
            return (interp_lat, interp_lon, grid_y, grid_x)
        end
        prev_lat, prev_lon = lat, lon
    end

    final_cell = path[end]
    final_lat, final_lon = grid_to_coords(final_cell)
    # The return order (lat_idx, lon_idx) is correct
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

# ===================================================================
# Specialized Movement Kernels (DVM and Diving)
# ===================================================================
@kernel function dvm_action_kernel!(data, p, grid_params, t, dt, sp)
    ind = @index(Global)
    @inbounds if data.alive[ind] == 1.0
        my_status = data.mig_status[ind]
        my_z = data.z[ind]
        swim_speed = 2.68 # m/min
        z_increment = swim_speed * dt
        is_daytime = (6*60 <= t < 18*60)
        
        if is_daytime
            if my_status == 0.0
                data.mig_status[ind] = 2.0
                data.target_z[ind] = clamp(get_target_z(sp, p.z_day_dist), my_z, grid_params.maxdepth)
                data.z[ind] = min(data.target_z[ind], my_z + z_increment)
            elseif my_status == 2.0
                data.z[ind] = min(data.target_z[ind], my_z + z_increment)
                if data.z[ind] >= data.target_z[ind]; data.mig_status[ind] = -1.0; end
            end
        else # Nighttime
            if my_status == -1.0
                data.mig_status[ind] = 1.0
                data.target_z[ind] = clamp(get_target_z(sp, p.z_night_dist), 1.0, my_z)
                data.z[ind] = max(data.target_z[ind], my_z - z_increment)
            elseif my_status == 1.0
                data.z[ind] = max(data.target_z[ind], my_z - z_increment)
                if data.z[ind] <= data.target_z[ind]; data.mig_status[ind] = 0.0; end
            end
        end
        @inbounds data.pool_z[ind] = ceil(Int, data.z[ind] / grid_params.depth_res_m)
        @inbounds data.active[ind] += dt
    end
end

function dvm_action!(model::MarineModel, sp::Int)
    arch = model.arch
    animal = model.individuals.animals[sp]
    data = animal.data
    p_cpu = animal.p
    
    p_gpu = (; (k => array_type(arch)(v.second) for (k, v) in pairs(p_cpu))...)
    
    grid = model.depths.grid
    grid_params = (
        maxdepth = grid[grid.Name .== "depthmax", :Value][1],
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / grid[grid.Name .== "depthres", :Value][1]
    )

    kernel! = dvm_action_kernel!(device(arch), 256, (length(data.x),))
    kernel!(data, p_gpu, grid_params, model.t, model.dt, sp)
    KernelAbstractions.synchronize(device(arch))
end

@kernel function dive_action_kernel!(data, p_gpu, grid_params, t, dt, sp)
    ind = @index(Global)
    @inbounds if data.alive[ind] == 1.0
        
        my_status = data.mig_status[ind]
        my_interval = data.interval[ind]
        my_z = data.z[ind]
        my_dives_remaining = data.dives_remaining[ind]
        my_fullness = data.gut_fullness[ind]
        my_biomass = data.biomass_school[ind]
        
        surface_interval = p_gpu.Surface_Interval[sp]
        foraging_interval = p_gpu.Dive_Interval[sp]
        dive_velo = p_gpu.Swim_velo[sp] * (data.length[ind] / 1000.0) * 60.0 # m/min
        z_increment = dive_velo * dt

        # State 0: At surface, deciding whether to dive
        if my_status == 0.0 && my_dives_remaining > 0
            my_interval += dt
            max_fullness = 0.2 * my_biomass
            dive_prob = 1.0 - logistic(my_fullness / max_fullness, 5.0, 0.5)
            if my_interval >= surface_interval && rand(Float32) < dive_prob
                data.mig_status[ind] = 1.0 # Start diving
                data.interval[ind] = 0.0
                data.target_z[ind] = rand(Float32) * (p_gpu.Dive_Max[sp] - p_gpu.Dive_Min[sp]) + p_gpu.Dive_Min[sp]
            else
                data.interval[ind] = my_interval
            end
        # State 1: Descending
        elseif my_status == 1.0
            data.z[ind] = min(data.target_z[ind], my_z + z_increment)
            if data.z[ind] >= data.target_z[ind]
                data.mig_status[ind] = -1.0 # Arrived at depth
            end
        # State -1: Foraging at depth
        elseif my_status == -1.0
            my_interval += dt
            if my_interval >= foraging_interval
                data.mig_status[ind] = 2.0 # Time to ascend
                data.interval[ind] = 0.0
            else
                data.interval[ind] = my_interval
            end
        # State 2: Ascending
        elseif my_status == 2.0
            data.z[ind] = max(1.0, my_z - z_increment) # Ascend towards surface
            if data.z[ind] <= 1.0
                data.mig_status[ind] = 0.0 # Arrived at surface
                data.dives_remaining[ind] -= 1
            end
        end
        
        @inbounds data.pool_z[ind] = ceil(Int, data.z[ind] / grid_params.depth_res_m)
        @inbounds data.active[ind] += dt
    end
end

function dive_action!(model::MarineModel, sp::Int)
    arch = model.arch
    animal = model.individuals.animals[sp]
    data = animal.data
    p_cpu = animal.p
    
    p_gpu = (; (k => array_type(arch)(v.second) for (k, v) in pairs(p_cpu))...)
    
    grid = model.depths.grid
    grid_params = (
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / grid[grid.Name .== "depthres", :Value][1]
    )

    kernel! = dive_action_kernel!(device(arch), 256, (length(data.x),))
    kernel!(data, p_gpu, grid_params, model.t, model.dt, sp)
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
