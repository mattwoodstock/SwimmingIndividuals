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
        night_profs = model.depths.focal_night
        target_z[ascent_inds] .= gaussmix(length(ascent_inds),night_profs[sp, "mu1"], night_profs[sp, "mu2"], night_profs[sp, "mu3"],night_profs[sp, "sigma1"], night_profs[sp, "sigma2"], night_profs[sp, "sigma3"],night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])
    else
        day_profs = model.depths.focal_day
        target_z[ascent_inds] .= gaussmix(length(ascent_inds),day_profs[sp, "mu1"], day_profs[sp, "mu2"], day_profs[sp, "mu3"],day_profs[sp, "sigma1"], day_profs[sp, "sigma2"], day_profs[sp, "sigma3"],day_profs[sp, "lambda1"], day_profs[sp, "lambda2"])
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

#Optimization function to find the ideal location for prey to go
function predator_avoidance(model, time, ind, to_move, pred_list, sp)
    # Precompute grid values
    files = model.files
    grid_file = files[files.File .=="grid", :Destination][1]
    grid = CSV.read(grid_file, DataFrame)
    
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    # Precompute some other values
    animal = model.individuals.animals[sp]
    animal_data = animal.data
    length_ind = animal_data.length[ind]

    max_dist = model.individuals.animals[sp].p.Swim_velo[2][sp] .* (length_ind / 1000) .* max.(time,0)

    # Threads for parallel processing
    Threads.@threads for ind_index in 1:length(ind)
        pred_list_item = filter(p -> p.Prey == to_move[ind_index], pred_list)        
        # Skip if there are no predators or the animal was consumed
        if model.individuals.animals[sp].data.ac[ind[ind_index]] == 0
            model.individuals.animals[sp].data.behavior[ind[ind_index]] = 0
            continue
        end

        position = SVector(animal_data.x[ind[ind_index]], animal_data.y[ind[ind_index]], animal_data.z[ind[ind_index]])

        if isnothing(pred_list_item)|| isempty(pred_list_item)
            random_direction = normalize(randn(2))
            random_distance = rand() * max_dist[ind_index]/2
            movement_vector = random_distance * random_direction

            current_position = [animal_data.x[ind[ind_index]],animal_data.y[ind[ind_index]]]

            new_position = current_position .+ movement_vector

            animal_data.x[ind[ind_index]] = clamp(new_position[1],lonmin, lonmax)
            animal_data.y[ind[ind_index]] = clamp(new_position[2], latmin, latmax)

            animal_data.pool_x[ind[ind_index]] = clamp(ceil(Int, animal_data.x[ind[ind_index]] / ((lonmax - lonmin) / lonres)), 1, lonres)
            animal_data.pool_y[ind[ind_index]] = clamp(ceil(Int, animal_data.y[ind[ind_index]] / ((latmax - latmin) / latres)), 1, latres)

            model.individuals.animals[sp].data.behavior[ind[ind_index]] = 0
            continue
        end

        predator_position = SVector(pred_list_item[1].x, pred_list_item[1].y, pred_list_item[1].z)

        # Calculate the direction and displacement
        direction_vector = predator_position - position
        direction_magnitude = norm(direction_vector)
        
        if direction_magnitude == 0
            continue
        end
        
        unit_vector = direction_vector / direction_magnitude
        displacement_vector = unit_vector * max_dist[ind_index]
        new_prey_position = position + displacement_vector

        # Apply bounds
        animal_data.x[ind[ind_index]] = clamp(new_prey_position[1], lonmin, lonmax)

        animal_data.y[ind[ind_index]] = clamp(new_prey_position[2], latmin, latmax)
        animal_data.z[ind[ind_index]] = clamp(new_prey_position[3], 1, maxdepth)

        # Update pool indices
        animal_data.pool_x[ind[ind_index]] = max(1, ceil(Int, (animal_data.x[ind[ind_index]] - lonmin) / ((lonmax - lonmin) / lonres)))
        animal_data.pool_y[ind[ind_index]] = max(1, ceil(Int, (animal_data.y[ind[ind_index]] - latmin) / ((latmax - latmin) / latres)))


        animal_data.pool_z[ind[ind_index]] = max(1, clamp(ceil(Int, animal_data.z[ind[ind_index]] / (maxdepth / depthres)), 1, depthres))

        # Update activity in minutes
        animal_data.active[ind[ind_index]] += (time[ind_index]/60)
    end
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
                    # Move in a straight line towards the prey
                    nearest_prey = prey_info[index]
                    dx = nearest_prey.x - model.individuals.animals[sp].data.x[animal]
                    dy = nearest_prey.y - model.individuals.animals[sp].data.y[animal]
                    dz = nearest_prey.z - model.individuals.animals[sp].data.z[animal]

                    if !iszero(dx) && !iszero(dy) && !iszero(dz)
                        norm_factor = sqrt(dx^2 + dy^2 + dz^2)
                        direction = (dx, dy, dz) ./ norm_factor

                        model.individuals.animals[sp].data.x[animal] += direction[1] * distances[ind_index]
                        model.individuals.animals[sp].data.y[animal] += direction[2] * distances[ind_index]
                        model.individuals.animals[sp].data.z[animal] += direction[3] * distances[ind_index]
                    else
                        model.individuals.animals[sp].data.x[animal] = nearest_prey.x
                        model.individuals.animals[sp].data.y[animal] = nearest_prey.y
                        model.individuals.animals[sp].data.z[animal] = nearest_prey.z
                    end
                end
                model.individuals.animals[sp].data.x[animal] = clamp(model.individuals.animals[sp].data.x[animal],lonmin,lonmax)
                model.individuals.animals[sp].data.y[animal] = clamp(model.individuals.animals[sp].data.y[animal],latmin,latmax)
                model.individuals.animals[sp].data.z[animal] = clamp(model.individuals.animals[sp].data.z[animal],1,maxdepth)

                model.individuals.animals[sp].data.pool_x[animal] = max(1,ceil(Int, model.individuals.animals[sp].data.x[animal] / ((lonmax - lonmin) / lonres)))
                model.individuals.animals[sp].data.pool_y[animal] = max(1,ceil(Int, model.individuals.animals[sp].data.y[animal] / ((latmax - latmin) / latres)))
                model.individuals.animals[sp].data.pool_z[animal] = max(1,ceil(Int, model.individuals.animals[sp].data.z[animal] / (maxdepth / depthres)))
                model.individuals.animals[sp].data.pool_z[animal] = clamp(model.individuals.animals[sp].data.pool_z[animal],1,depthres)
            end
        end
    end
end

