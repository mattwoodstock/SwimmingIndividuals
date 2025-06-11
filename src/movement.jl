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
    return nothing  # no path found
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

function dvm_action(model, sp, ind)
    animal = model.individuals.animals[sp]
    data = animal.data
    params = animal.p
    ΔT = params.t_resolution[2][sp]
    swim_speed = 2.68 # Meters per minute. Current from Bianchi and Mislan 2016

    # Create masks for each condition
    is_daytime = (6*60 <= model.t) && (model.t < 18*60)
    is_nighttime = (model.t >= 18*60) || (model.t < 6*60)

    mig_status_0 = findall(data.mig_status[ind] .== 0.0)
    mig_status_neg1 = findall(data.mig_status[ind] .== -1.0)
    mig_status_1 = findall(data.mig_status[ind] .== 1.0)
    mig_status_2 = findall(data.mig_status[ind] .== 2.0)

    if is_daytime && !isempty(mig_status_0)
        # Daytime descent
        grid = model.depths.grid
        maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
        depthres = grid[findfirst(grid.Name .== "depthres"), :Value]
        z_day_dist = model.depths.focal_day 
        mig_inds = ind[mig_status_0]
        data.mig_status[mig_inds] .= 2
        data.target_z[mig_inds] .= clamp.(get_target_z.(sp, Ref(z_day_dist)), data.z[mig_inds], maxdepth)
        data.mig_rate[mig_inds] .= swim_speed
        data.z[mig_inds] .= min.(data.target_z[mig_inds], data.z[mig_inds] .+ data.mig_rate[mig_inds] .* ΔT)
        data.behavior[mig_inds] .= 2
        data.mig_status[mig_inds[data.z[mig_inds] .>= data.target_z[mig_inds]]] .= -1
        data.active[mig_inds] .= ΔT
        # Update pool_z
        data.pool_z[ind] .= ceil.(Int, data.z[ind] ./ (maxdepth / depthres))
        #Update vision
        data.vis_prey[ind] = visual_range_prey(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
        data.vis_pred[ind] = visual_range_pred(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
    end

    if is_nighttime && !isempty(mig_status_neg1)
        # Nighttime ascent
        z_night_dist = model.depths.focal_night
        grid = model.depths.grid
        maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
        depthres = grid[findfirst(grid.Name .== "depthres"), :Value]
        mig_inds = ind[mig_status_neg1]
        data.mig_status[mig_inds] .= 1
        data.target_z[mig_inds] .= clamp.(get_target_z.(sp, Ref(z_night_dist)), 1, data.z[mig_inds])
        data.mig_rate[mig_inds] .= swim_speed
        data.z[mig_inds] .= max.(data.target_z[mig_inds], data.z[mig_inds] .- data.mig_rate[mig_inds] .* ΔT)
        data.behavior[mig_inds] .= 2
        data.mig_status[mig_inds[data.z[mig_inds] .== data.target_z[mig_inds]]] .= 0
        data.active[mig_inds] .= ΔT
        # Update pool_z
        data.pool_z[ind] .= ceil.(Int, data.z[ind] ./ (maxdepth / depthres))
        #Update vision
        data.vis_prey[ind] = visual_range_prey(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
        data.vis_pred[ind] = visual_range_pred(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
    end

    if !isempty(mig_status_1)
        # Continue ascending
        grid = model.depths.grid
        maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
        depthres = grid[findfirst(grid.Name .== "depthres"), :Value]

        mig_inds = ind[mig_status_1]
        target_z_asc = data.target_z[mig_inds]
        data.z[mig_inds] .= max.(target_z_asc, data.z[mig_inds] .- data.mig_rate[mig_inds] .* ΔT)
        data.mig_status[mig_inds[(data.z[mig_inds] .== target_z_asc) .| (model.t .== 21*60)]] .= 0
        data.active[mig_inds] .= ΔT
        # Update pool_z
        data.pool_z[ind] .= ceil.(Int, data.z[ind] ./ (maxdepth / depthres))
        #Update vision
        data.vis_prey[ind] = visual_range_prey(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
        data.vis_pred[ind] = visual_range_pred(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
    end

    if !isempty(mig_status_2)
        # Continue descending
        grid = model.depths.grid
        maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
        depthres = grid[findfirst(grid.Name .== "depthres"), :Value]

        mig_inds = ind[mig_status_2]
        target_z_desc = data.target_z[mig_inds]
        data.z[mig_inds] .= min.(target_z_desc, data.z[mig_inds] .+ data.mig_rate[mig_inds] .* ΔT)
        data.mig_status[mig_inds[(data.z[mig_inds] .== target_z_desc) .| (model.t .== 9*60)]] .= -1
        data.active[mig_inds] .= ΔT
        # Update pool_z
        data.pool_z[ind] .= ceil.(Int, data.z[ind] ./ (maxdepth / depthres))
        #Update vision
        data.vis_prey[ind] = visual_range_prey(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
        data.vis_pred[ind] = visual_range_pred(model,data.length[ind],data.z[ind],sp,length(ind)) .* ΔT
    end
    return nothing
end

function dive_action(model, sp, inds)
    ##Need to refine active time

    files = model.files
    grid_file = files[files.File .== "grid", :Destination][1]
    grid = CSV.read(grid_file, DataFrame)
    

    # Read grid and resolution data
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]

    animal_data = model.individuals.animals[sp].data
    animal_p = model.individuals.animals[sp].p
    
    # Define behavioral parameters
    surface_interval = animal_p.Surface_Interval[2][sp]
    foraging_interval = animal_p.Dive_Interval[2][sp]
    dive_velocity = animal_p.Swim_velo[2][sp] .* (animal_data.length[inds] ./ 1000) .* 60  # meters per minute
    ΔT = animal_p.t_resolution[2][sp]
    
    is_nighttime = (model.t >= 18*60) || (model.t < 6*60)

    # Array of migration statuses for all individuals
    mig_status = animal_data.mig_status[inds]
    interval = animal_data.interval[inds]
    gut_fullness = animal_data.gut_fullness[inds]
    biomass = animal_data.biomass[inds]
    target_z = animal_data.target_z[inds]
    z = animal_data.z[inds]
    dives_remaining = animal_data.dives_remaining[inds]
    active = animal_data.active[inds]
    spent = fill(0.0,length(inds))

    # Surface interval logic for individuals in `mig_status == 0`
    surface_inds = findall(mig_status .== 0)
    max_fullness = 0.2 .* biomass[surface_inds]
    dive_trigger = gut_fullness[surface_inds] ./ max_fullness
    dive_probability = logistic.(dive_trigger, 5, 0.5)

    to_dive = (interval[surface_inds] .>= surface_interval) .&& (rand(length(surface_inds)) .> dive_probability) .&& (dives_remaining[surface_inds] .> 0)
    interval[surface_inds] .= ifelse.(to_dive, 0, interval[surface_inds] .+ ΔT)
    mig_status[surface_inds[to_dive]] .= 1
    target_z[surface_inds[to_dive]] .= sample_normal(animal_p.Dive_Min[2][sp], animal_p.Dive_Max[2][sp], std=20)[rand(1:end, length(surface_inds[to_dive]))]
    active[surface_inds[to_dive]] .+= ΔT
    spent[surface_inds[to_dive]] .= 1.0

    # Diving logic for individuals in `mig_status == 1`
    dive_inds = findall(mig_status .== 1 .&& spent .== 0.0)
    z[dive_inds] .= min.(target_z[dive_inds], z[dive_inds] .+ dive_velocity[dive_inds] .* ΔT)
    active[dive_inds] .+= ΔT
    mig_status[dive_inds[z[dive_inds] .>= target_z[dive_inds]]] .= -1
    spent[dive_inds] .= 1.0

    # Foraging interval logic for individuals in `mig_status == -1`
    forage_inds = findall(mig_status .== -1 .&& spent .== 0.0)
    interval[forage_inds] .+= ΔT
    to_ascend = interval[forage_inds] .>= foraging_interval
    interval[forage_inds[to_ascend]] .= 0
    mig_status[forage_inds[to_ascend]] .= 2
    spent[forage_inds] .= 1.0

    # Ascending logic for individuals in `mig_status == 2`
    ascent_inds = findall(mig_status .== 2 .&& spent .== 0.0)

    if is_nighttime
        if (model.individuals.animals[sp].p.Taxa[2][sp] == "cetacean")
            target_z[ascent_inds] .= 0
        else
            night_profs = model.depths.focal_night
            target_z[ascent_inds] .= gaussmix(length(ascent_inds),night_profs[sp, "mu1"], night_profs[sp, "mu2"], night_profs[sp, "mu3"],night_profs[sp, "sigma1"], night_profs[sp, "sigma2"], night_profs[sp, "sigma3"],night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])
        end
    else
        if (model.individuals.animals[sp].p.Taxa[2][sp] == "cetacean")
            target_z[ascent_inds] .= 0
        else
            day_profs = model.depths.focal_day
            target_z[ascent_inds] .= gaussmix(length(ascent_inds),day_profs[sp, "mu1"], day_profs[sp, "mu2"], day_profs[sp, "mu3"],day_profs[sp, "sigma1"], day_profs[sp, "sigma2"], day_profs[sp, "sigma3"],day_profs[sp, "lambda1"], day_profs[sp, "lambda2"])
        end
    end

    z[ascent_inds] .= max.(1,target_z[ascent_inds], z[ascent_inds] .- dive_velocity[ascent_inds] .* ΔT)
    active[ascent_inds] .+= ΔT
    mig_status[ascent_inds[z[ascent_inds] .<= 1]] .= 0
    dives_remaining[ascent_inds[z[ascent_inds] .<= 1]] .-= 1

    # Prevent further diving for individuals that have exhausted their daily dives
    max_dives_reached = findall(dives_remaining .<= 0)
    mig_status[max_dives_reached] .= 0

    # Update depth index for grid resolution
    pool_z = ceil.(Int, z ./ (maxdepth / depthres))
    animal_data.pool_z[inds] .= pool_z

    # Update fields in `animal_data`
    animal_data.interval[inds] .= interval
    animal_data.mig_status[inds] .= mig_status
    animal_data.target_z[inds] .= target_z
    animal_data.z[inds] .= z
    animal_data.dives_remaining[inds] .= dives_remaining
    animal_data.active[inds] .= active
    return nothing
end

# Function for the cost function used to triangulate the best distance
function cost_function_prey(prey_location, preds)
    total_distance = sum(norm(prey_location .- predator) for predator in preds)
    return -total_distance
end

function move_to_prey(model, sp, ind,eat_ind, time, prey_list)
    ddt = fill(model.individuals.animals[sp].p.t_resolution[2][sp] * 60.0,length(ind))
    ddt[eat_ind] .= time #Update times from feeding

    distances = model.individuals.animals[sp].p.Swim_velo[2][sp] .* (model.individuals.animals[sp].data.length[ind] / 1000) .* ddt  # meters the animal can swim

    files = model.files
    grid_file = files[files.File .=="grid",:Destination][1]
    grid = CSV.read(grid_file,DataFrame)
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    Threads.@threads for ind_index in 1:length(ind)
        animal = ind[ind_index]
        prey_info = prey_list.preys[ind_index]
        
        if !isempty(prey_info)
            # Extract distances of all preys
            prey_distances = [prey.Distance for prey in prey_info]
            
            # Find the nearest prey
            min_distance, index = findmin(prey_distances)

            if !iszero(index)
                if min_distance <= distances[ind_index]
                    # Directly move to prey location
                    nearest_prey = prey_info[index]
                    model.individuals.animals[sp].data.x[animal] = nearest_prey.x
                    model.individuals.animals[sp].data.y[animal] = nearest_prey.y
                    model.individuals.animals[sp].data.z[animal] = nearest_prey.z

                else
                    habitat = model.capabilities[:,:,model.environment.ts,sp]

                    start = (model.individuals.animals[sp].data.x[animal],model.individuals.animals[sp].data.y[animal])

                    target = (nearest_prey.x,nearest_prey.y)
                    path = find_path(habitat, start, target)

                    if !isempty(path)
                        new_x,new_y,new_pool_x,new_pool_y = reachable_point(current_pos,path, distances[ind_index],latmin,latmax,lonmin,lonmax,nrows,ncols)

                        model.individuals.animals[sp].data.x[animal] = new_x
                        model.individuals.animals[sp].data.y[animal] = new_y
                        model.individuals.animals[sp].data.pool_x[animal] = new_pool_x
                        model.individuals.animals[sp].data.pool_y[animal] = new_pool_y
                    end
                end
            end
        end
    end
end

function movement_toward_habitat(model, time::Vector{Float64}, inds, sp)
    month = model.environment.ts
    habitat = model.capacities[:, :, month, sp]
    grid = model.depths.grid

    nrows, ncols = size(habitat)

    animal_data = model.individuals.animals[sp].data
    animal_param = model.individuals.animals[sp].p

    @views x = animal_data.x[inds]
    @views y = animal_data.y[inds]
    @views pool_x = Int.(animal_data.pool_x[inds])
    @views pool_y = Int.(animal_data.pool_y[inds])
    @views lengths = animal_data.length[inds] ./ 1000

    swim_speed = bl_per_s(lengths*100,animal_param.Swim_velo[2][sp])
    
    max_swim_distance = swim_speed .* lengths .* time[inds]

    latmin = grid[grid.Name .== "yllcorner", :Value][1]
    lonmin = grid[grid.Name .== "xllcorner", :Value][1]
    cell_size = grid[grid.Name .== "cellsize", :Value][1]

    # Precompute list of valid habitat cells
    habitat_cells = Vector{NamedTuple{(:x, :y, :value), Tuple{Int, Int, Float64}}}()
    for i in 1:nrows, j in 1:ncols
        val = habitat[i, j]
        if val > 0
            push!(habitat_cells, (x = j, y = i, value = val))
        end
    end

    sort!(habitat_cells, by = x -> x.value, rev = true)
    cumvals = cumsum(getfield.(habitat_cells, :value))
    total = cumvals[end]

    @Threads.threads for n in 1:length(inds)
        @inbounds ind = inds[n]
        @inbounds start_y = pool_y[n]
        @inbounds start_x = pool_x[n]
        @inbounds cur_y = y[n]
        @inbounds cur_x = x[n]
        @inbounds max_dist = max_swim_distance[n]
        @inbounds dt = time[n]

        if dt > 0
            if habitat[nrows - start_y + 1, start_x] == 0
                new_y, new_x, new_pool_x, new_pool_y = nearest_suitable_habitat(
                    habitat, (cur_y, cur_x), (start_y, start_x), max_dist, latmin, lonmin, cell_size
                )
                if new_y === nothing
                    continue
                end
                @inbounds animal_data.y[ind] = new_y
                @inbounds animal_data.x[ind] = new_x
                @inbounds animal_data.pool_y[ind] = new_pool_y
                @inbounds animal_data.pool_x[ind] = new_pool_x
            else
                count = 0
                path = nothing
                while path === nothing && count < 10
                    count += 1
                    r = (rand()^4) * total
                    idx = findfirst(x -> x ≥ r, cumvals)
                    target_y = habitat_cells[idx].y
                    target_x = habitat_cells[idx].x
                    target = (nrows - target_y + 1, target_x)
                    path = find_path(habitat, (start_y, start_x), target)
                end
                if path === nothing
                    continue
                end
                new_y, new_x, new_pool_y, new_pool_x = reachable_point(
                    (cur_y, cur_x), path, max_dist, latmin, lonmin, cell_size, nrows, ncols
                )
                @inbounds animal_data.y[ind] = new_y
                @inbounds animal_data.x[ind] = new_x
                @inbounds animal_data.pool_y[ind] = new_pool_y
                @inbounds animal_data.pool_x[ind] = new_pool_x
            end
        end
        @inbounds animal_data.active[ind] += time[ind]
    end

    return nothing
end

function move_resources(model,month)
    resources = Vector{resource}()
    depths = model.depths
    grid = depths.grid
    traits = model.resource_trait
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    maxdepth = Int(grid[grid.Name .== "depthmax", :Value][1])

    depth_values = range(0, stop=maxdepth, length=depthres+1) |> collect

    ind = 1
    for sp in 1:model.n_resource
        matching_idxs = findall(r -> r.sp == sp, model.resources)

        biomasses = getfield.(model.resources[matching_idxs], :biomass)
        total_biomass = sum(biomasses)

        caps = model.capacities[:,:,month,model.n_species+sp]
        total_caps = sum(caps)
        n_pack = traits[sp,:Packets]

        night_profs = depths.patch_night
        means = [night_profs[sp, "mu1"], night_profs[sp, "mu2"], night_profs[sp, "mu3"]]
        stds = [night_profs[sp, "sigma1"], night_profs[sp, "sigma2"], night_profs[sp, "sigma3"]]
        weights = [night_profs[sp, "lambda1"], night_profs[sp, "lambda2"], night_profs[sp, "lambda3"]]
        x_values = depth_values[1]:depth_values[end]
        pdf_values = multimodal_distribution.(Ref(x_values), means, stds, weights)
        depth_weights_norm = pdf_values[1] ./ sum(pdf_values[1])
        depth_props = [sum(depth_weights_norm[(x_values .>= depth_values[i]) .& (x_values .< depth_values[i+1])]) for i in 1:length(depth_values)-1]

        for j in 1:latres, k in 1:lonres
            if caps[j,k] > 0
                biomass_target = total_biomass .* (caps[j,k] / total_caps) .* depth_props

                for l in 1:depthres
                    packet_biomass = biomass_target[l] / n_pack
                    x,y,pool_x,pool_y = initial_ind_placement(capacities,model.n_species+sp,grid,n_pack)

                    z = depth_values[l] + rand() * (depth_values[l+1]-depth_values[l])
                    pool_z = l

                    for m in 1:n_pack
                        push!(resources, resource(sp,ind,x[m],y[m],z,pool_x[m],pool_y[m],pool_z,packet_biomass,packet_biomass))

                        ind += 1
                    end
                end
            end
        end
    end
    return resources
end

function bl_per_s(length,speed;b=0.35,min_speed = 0.5)
    return max.(speed .* length .^ (-b),min_speed)
end