function dvm_action(model, sp, ind)
    animal = model.individuals.animals[sp]
    data = animal.data
    params = animal.p
    ΔT = params.t_resolution[2][sp]
    swim_speed = 2.64 # Meters per minute following Bianchi and Mislan 2016. Want to make this size based.
    files = model.files
    z_dist_file_day = files[files.File .== "focal_z_dist_day", :Destination][1]
    z_dist_file_night = files[files.File .== "focal_z_dist_night", :Destination][1]
    grid_file = files[files.File .== "grid", :Destination][1]
    z_day_dist = CSV.read(z_dist_file_day, DataFrame)
    z_night_dist = CSV.read(z_dist_file_night, DataFrame)

    grid = CSV.read(grid_file, DataFrame)
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]

    # Function to get target_z based on distribution
    function get_target_z(sp, dist)
        return gaussmix(1, dist[sp, "mu1"], dist[sp, "mu2"], dist[sp, "mu3"], dist[sp, "sigma1"], dist[sp, "sigma2"], dist[sp, "sigma3"], dist[sp, "lambda1"], dist[sp, "lambda2"])[1]
    end

    # Create masks for each condition
    is_daytime = (6*60 <= model.t) && (model.t < 18*60)
    is_nighttime = model.t >= 18*60

    mig_status_0 = findall(x -> x == 0.0, data.mig_status[ind])
    mig_status_neg1 = findall(x -> x == -1.0, data.mig_status[ind])
    mig_status_1 = findall(x -> x == 1.0, data.mig_status[ind])
    mig_status_2 = findall(x -> x == 2.0, data.mig_status[ind])

    if is_daytime && !isempty(mig_status_0)
        # Daytime descent
        mig_inds = ind[mig_status_0]
        data.mig_status[mig_inds] .= 2
        data.target_z[mig_inds] .= clamp.(get_target_z.(sp, Ref(z_day_dist)), 1, maxdepth)
        data.mig_rate[mig_inds] .= swim_speed
        data.z[mig_inds] .= min.(data.target_z[mig_inds], data.z[mig_inds] .+ data.mig_rate[mig_inds] .* ΔT)
        data.behavior[mig_inds] .= 2
        data.mig_status[mig_inds[data.z[mig_inds] .>= data.target_z[mig_inds]]] .= -1
    end

    if is_nighttime && !isempty(mig_status_neg1)
        # Nighttime ascent
        mig_inds = ind[mig_status_neg1]
        data.mig_status[mig_inds] .= 1
        data.target_z[mig_inds] .= clamp.(get_target_z.(sp, Ref(z_night_dist)), 1, maxdepth)
        data.mig_rate[mig_inds] .= swim_speed
        data.z[mig_inds] .= max.(data.target_z[mig_inds], data.z[mig_inds] .- data.mig_rate[mig_inds] .* ΔT)
        data.behavior[mig_inds] .= 2
        data.mig_status[mig_inds[data.z[mig_inds] .== data.target_z[mig_inds]]] .= 0
    end

    if !isempty(mig_status_1)
        # Continue ascending
        mig_inds = ind[mig_status_1]
        target_z_asc = data.target_z[mig_inds]
        data.z[mig_inds] .= max.(target_z_asc, data.z[mig_inds] .- data.mig_rate[mig_inds] .* ΔT)
        data.mig_status[mig_inds[(data.z[mig_inds] .== target_z_asc) .| (model.t .== 21*60)]] .= 0
    end

    if !isempty(mig_status_2)
        # Continue descending
        mig_inds = ind[mig_status_2]
        target_z_desc = data.target_z[mig_inds]
        data.z[mig_inds] .= min.(target_z_desc, data.z[mig_inds] .+ data.mig_rate[mig_inds] .* ΔT)
        data.mig_status[mig_inds[(data.z[mig_inds] .== target_z_desc) .| (model.t .== 9*60)]] .= -1
    end

    # Update behavior for individuals not migrating
    data.behavior[ind[data.mig_status[ind] .== 0]] .= 1
    data.behavior[ind[data.mig_status[ind] .== -1]] .= 0

    # Update pool_z
    data.pool_z[ind] .= ceil.(Int, data.z[ind] ./ (maxdepth / depthres))

    #Update vision
    data.vis_prey[ind] = visual_range_preys_init(data.length[ind],data.z[ind],length(ind)) .* ΔT
    data.vis_pred[ind] = visual_range_preds_init(arch,data.length[ind],data.z[ind],length(ind)) .* ΔT

    return nothing
end

function surface_dive(model, sp, ind)
    files = model.files
    grid_file = files[files.File .== "grid", :Destination][1]
    grid = CSV.read(grid_file, DataFrame)
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    for i in ind[1]:ind[1]
        animal = model.individuals.animals[sp].data
        animal_p = model.individuals.animals[sp].p
        # Define behavioral parameters
        surface_interval = animal_p.Surface_Interval[2][sp]
        foraging_interval = animal_p.Dive_Interval[2][sp]
        dive_velocity = (animal.length[ind[i]]/1000) * 60 # meters per minute

        # Time resolution
        ΔT = animal_p.t_resolution[2][sp]
        # Progress the animal through the stages
        if animal.mig_status[ind[i]] == 0
            # Surface interval

            animal.interval[ind[i]] += ΔT
            max_fullness = 0.2 * animal.biomass[ind[i]]
            dive_trigger = animal.gut_fullness[ind[i]] / max_fullness
            dist = logistic(dive_trigger, 5, 0.5)
            #Add probability here to start diving

            if animal.interval[ind[i]] >= surface_interval && rand() > dist
                # Reset interval timer and move to the next stage (diving)
                animal.interval[ind[i]] = 0
                animal.mig_status[ind[i]] = 1
                # Randomly select dive depth
                animal.target_z[ind[i]] = sample_normal(animal_p.Dive_depth_min[2][sp], animal_p.Dive_depth_max[2][sp], std=20)[rand(1:end)]
            end

        elseif animal.mig_status[ind[i]] == 1
            # Continue diving

            #Change depth
            if animal.z[ind[i]] >= animal.target_z[ind[i]]
                # Reset active time and move to the next stage (foraging interval)
                animal.mig_status[ind[i]] = -1
            end

        elseif animal.mig_status[ind[i]] == -1
            # Foraging interval

            if animal.interval[ind[i]] >= foraging_interval
                # Reset interval timer and move to the next stage (ascending)
                animal.interval[ind[i]] = 0
                animal.mig_status[ind[i]] = 2
            else
                animal.interval[ind[i]] += ΔT
            end

        elseif animal.mig_status[ind[i]] == 2
            # Ascending

            if animal.z[ind[i]] <= 1
                # Reset active time and move to the next cycle (back to surface interval)
                animal.interval[ind[i]] = 0
                animal.mig_status[ind[i]] = 0

                # Increment dive count for the day
                animal.dives_remaining[ind[i]] -= 1
            end
        end

        # Check if the dive count for the day exceeds the maximum limit
        if animal.dives_remaining[ind[i]] <= 0
            # Prevent further diving
            animal.mig_status[ind[i]] = 0
        end

        # Update the position of the animal based on the dive velocity
        if animal.mig_status[ind[i]] == 1
            animal.z[ind[i]] = min(animal.target_z[ind[i]], animal.z[ind[i]] + dive_velocity * ΔT)
        elseif animal.mig_status[ind[i]] == 2
            animal.z[ind[i]] = max(1, animal.z[ind[i]] - dive_velocity * ΔT)
        end
        model.individuals.animals[sp].data.pool_z[ind[i]] = ceil(Int, model.individuals.animals[sp].data.z[ind[i]] / (maxdepth / depthres))

    end

    return nothing
end

function pelagic_dive(model, sp, ind)

    files = model.files
    grid_file = files[files.File .== "grid", :Destination][1]
    grid = CSV.read(grid_file, DataFrame)
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    z_dist_file_day = files[files.File .== "focal_z_dist_day", :Destination][1]
    z_dist_file_night = files[files.File .== "focal_z_dist_night", :Destination][1]
    z_day_dist = CSV.read(z_dist_file_day, DataFrame)
    z_night_dist = CSV.read(z_dist_file_night, DataFrame)

    for i in ind[1]:ind[1]
        animal = model.individuals.animals[sp].data
        animal_p = model.individuals.animals[sp].p
        # Define behavioral parameters
        surface_interval = animal_p.Surface_Interval[2][sp]
        foraging_interval = animal_p.Dive_Interval[2][sp]
        dive_velocity = (animal.length[ind[i]]/1000) * 60 # meters per minute

        # Time resolution
        ΔT = animal_p.t_resolution[2][sp]

        # Progress the animal through the stages
        if animal.mig_status[ind[i]] == 0
            # Surface interval
            animal.interval[ind[i]] += ΔT
            max_fullness = 0.2 * animal.biomass[ind[i]]
            dive_trigger = animal.gut_fullness[ind[i]] / max_fullness
            dist = logistic(dive_trigger, 5, 0.5)
            #Add probability here to start diving


            if animal.interval[ind[i]] >= surface_interval && rand() > dist
                # Reset interval timer and move to the next stage (diving)
                animal.interval[ind[i]] = 0
                animal.mig_status[ind[i]] = 1
                # Randomly select dive depth
                animal.target_z[ind[i]] = -50
                while animal.z[ind[i]] < animal.target_z[ind[i]]
                    if (6*60 <= model.t < 18*60)
                        animal.target_z[ind[i]] = gaussmix(1, z_day_dist[sp, "mu1"], z_day_dist[sp, "mu2"],z_day_dist[sp, "mu3"], z_day_dist[sp, "sigma1"],z_day_dist[sp, "sigma2"], z_day_dist[sp, "sigma3"],z_day_dist[sp, "lambda1"], z_day_dist[sp, "lambda2"])[1]
                    else
                        animal.target_z[ind[i]] = gaussmix(1, z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])[1]
                    end
                end
            end

        elseif animal.mig_status[ind[i]] == 1
            # Continue diving

            #Change depth
            if animal.z[ind[i]] >= animal.target_z[ind[i]]
                # Reset active time and move to the next stage (foraging interval)
                animal.mig_status[ind[i]] = -1
            end

        elseif animal.mig_status[ind[i]] == -1
            # Foraging interval
            animal.interval[ind[i]] += ΔT

            if animal.interval[ind[i]] >= foraging_interval
                # Reset interval timer and move to the next stage (ascending)
                animal.interval[ind[i]] = 0
                animal.mig_status[ind[i]] = 2

                animal.target_z[ind[i]] = 5e6
                while animal.z[ind[i]] > animal.target_z[ind[i]]
                    if (6*60 <= model.t < 18*60)
                        animal.target_z[ind[i]] = gaussmix(1, z_day_dist[sp, "mu1"], z_day_dist[sp, "mu2"],z_day_dist[sp, "mu3"], z_day_dist[sp, "sigma1"],z_day_dist[sp, "sigma2"], z_day_dist[sp, "sigma3"],z_day_dist[sp, "lambda1"], z_day_dist[sp, "lambda2"])[1]
                    else
                        animal.target_z[ind[i]] = gaussmix(1, z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])[1]
                    end
                end
            end

        elseif animal.mig_status[ind[i]] == 2
            # Ascending

            if animal.z[ind[i]] <= animal.target_z[ind[i]]
                # Reset active time and move to the next cycle (back to surface interval)
                animal.interval[ind[i]] = 0
                animal.mig_status[ind[i]] = 0

                # Increment dive count for the day
                animal.dives_remaining[ind[i]] -= 1
            end
        end

        # Check if the dive count for the day exceeds the maximum limit
        if animal.dives_remaining[ind[i]] <= 0
            # Prevent further diving
            animal.mig_status[ind[i]] = 0
        end

        # Update the position of the animal based on the dive velocity
        if animal.mig_status[ind[i]] == 1
            animal.z[ind[i]] = min(animal.target_z[ind[i]], animal.z[ind[i]] + dive_velocity * ΔT)
        elseif animal.mig_status[ind[i]] == 2
            animal.z[ind[i]] = max(animal.target_z[ind[i]], animal.z[ind[i]] - dive_velocity * ΔT)
        end
        model.individuals.animals[sp].data.pool_z[ind[i]] = ceil(Int, model.individuals.animals[sp].data.z[ind[i]] / (maxdepth / depthres))

        model.individuals.animals[sp].data.pool_z[ind[i]] = clamp(model.individuals.animals[sp].data.pool_z[ind[i]],1,depthres)

    end

    return nothing
end

function cost_function_prey(position, predator_matrix)
    sum_distance = 0.0
    for predator in eachrow(predator_matrix)
        sum_distance += norm(position .- Vector(predator))
    end
    return -sum_distance  # We maximize distance, hence the negative
end

#Optimization function to find the ideal location for prey to go
function predator_avoidance(model,time,ind,to_move,pred_list,sp)
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

    n_ind = length(model.individuals.animals[sp].data.length[ind])
    animal = model.individuals.animals[sp]
    animal_data = animal.data
    length_ind = animal_data.length[ind]
    max_dist = model.individuals.animals[sp].p.Swim_velo[2][sp] * (length_ind / 1000) .* time

    Threads.@threads for ind_index in 1:n_ind
        pred_list_item = pred_list.preds[to_move[ind_index]]
        if isempty(pred_list_item)
            continue
        end

        if model.individuals.animals[sp].data.ac[ind[ind_index]] == 0 #Animal was consumed in by another animal this time.
            continue
        end

        pred_info = pred_list_item[1]
        predator_position = [pred_info.x,pred_info.y,pred_info.z]
        position = [animal_data.x[ind[ind_index]],animal_data.y[ind[ind_index]],animal_data.z[ind[ind_index]]]
        direction_vector = predator_position .- position
        direction_magnitude = sqrt(sum(direction_vector.^2))
        if direction_magnitude == 0
            continue
        else
            unit_vector = direction_vector / direction_magnitude
            displacement_vector = unit_vector .* max_dist[ind_index]
            new_prey_position = position .+ displacement_vector
        end

        x = clamp(new_prey_position[1],lonmin,lonmax)
        y = clamp(new_prey_position[2],latmin,latmax)
        z = clamp(new_prey_position[3],1,maxdepth)

        model.individuals.animals[sp].data.x[ind[ind_index]] = x
        model.individuals.animals[sp].data.y[ind[ind_index]] = y
        model.individuals.animals[sp].data.z[ind[ind_index]] = z

        model.individuals.animals[sp].data.pool_x[ind[ind_index]] = max(1,ceil(Int, model.individuals.animals[sp].data.x[ind[ind_index]] / ((lonmax - lonmin) / lonres)))
        model.individuals.animals[sp].data.pool_y[ind[ind_index]] = max(1,ceil(Int, model.individuals.animals[sp].data.y[ind[ind_index]] / ((latmax - latmin) / latres)))
        model.individuals.animals[sp].data.pool_z[ind[ind_index]] = max(1,ceil(Int, model.individuals.animals[sp].data.z[ind[ind_index]] / (maxdepth / depthres)))
        model.individuals.animals[sp].data.pool_z[ind[ind_index]] = clamp(model.individuals.animals[sp].data.pool_z[ind[ind_index]],1,depthres)

        model.individuals.animals[sp].data.active[ind[ind_index]] += time[ind_index]
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

function move_pool(model,pool_index,ind)
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


    swim_speed = 1 .* (model.pools.pool[pool_index].data.length[ind] ./ 1000) .* model.dt
    z_limit = 10
    x = model.pools.pool[pool_index].data.x[ind]
    y = model.pools.pool[pool_index].data.y[ind]
    z = model.pools.pool[pool_index].data.z[ind]

    theta = 2 .* π .* rand(length(ind))
    phi = acos.(2 .* rand(length(ind)) .- 1)  # Uniform distribution in the z direction

    dx = swim_speed .* sin.(phi) .* cos.(theta)
    dy = swim_speed .* sin.(phi) .* sin.(theta)
    dz = swim_speed .* cos.(phi)

    dz = clamp(z .+ dz, z .- z_limit, z .+ z_limit) .- z

    new_x = x .+ dx
    small_lon = findall(x -> x < lonmin, new_x)
    big_lon = findall(x -> x > lonmax, new_x)

    if length(small_lon) > 0
        new_x[small_lon] = new_x[small_lon] .+ (lonmax-lonmin)
    end
    if length(big_lon) > 0
        new_x[big_lon] = new_x[big_lon] .- (lonmax-lonmin)
    end

    new_y = y .+ dy
    small_lat = findall(x -> x < latmin, new_y)
    big_lat = findall(x -> x > latmax, new_y)

    if length(small_lat) > 0
        new_x[small_lat] = new_x[small_lat] .+ (latmax-latmin)
    end
    if length(big_lat) > 0
        new_x[big_lat] = new_x[big_lat] .- (latmax-latmin)
    end

    new_z = z .+ dz

    new_z = clamp.(new_z,1,maxdepth)

    model.pools.pool[pool_index].data.x[ind] = new_x
    model.pools.pool[pool_index].data.y[ind] = new_y
    model.pools.pool[pool_index].data.z[ind] = new_z

    model.pools.pool[pool_index].data.pool_x[ind] = max.(1,ceil.(Int, new_x ./ ((lonmax - lonmin) / lonres)))
    model.pools.pool[pool_index].data.pool_y[ind] = max.(1,ceil.(Int, new_y ./ ((latmax - latmin) / latres)))
    model.pools.pool[pool_index].data.pool_z[ind] = max.(1,ceil.(Int, new_z ./ (maxdepth / depthres)))

    return nothing
end

