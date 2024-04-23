function random_movement(latitude, longitude, movement_distance)
    # Generate random values for the movement vector
    random_lat_change = rand() - 0.5  # Random value between -0.5 and 0.5
    random_lon_change = rand() - 0.5

    # Normalize the movement vector
    norm_factor = sqrt(random_lat_change^2 + random_lon_change^2)
    norm_random_lat_change = random_lat_change / norm_factor
    norm_random_lon_change = random_lon_change / norm_factor

    # Calculate the changes in coordinates
    lat_change = norm_random_lat_change * movement_distance
    lon_change = norm_random_lon_change * movement_distance

    # Convert degrees to meters for distance calculations
    lat_meters, lon_meters = degrees_to_meters(latitude, longitude)

    # Calculate the new coordinates in meters
    new_lat_meters = lat_meters + lat_change
    new_lon_meters = lon_meters + lon_change

    # Convert the new coordinates back to degrees
    new_lat_deg, new_lon_deg = meters_to_degrees(new_lat_meters, new_lon_meters)

    return new_lat_deg, new_lon_deg
end

function dvm_action(model, sp, ind,outputs)
    animal = model.individuals.animals[sp]
    data = animal.data
    params = animal.p
    ΔT = params.t_resolution[2][sp]

    swim_speed = data.length[ind]/1000 * params.Swim_velo[2][sp] * ΔT

    if (6*60 <= model.t < 18*60) && (data.mig_status[ind] == 0) # Start descent during daytime
        z_dist_file = model.files[model.files.File .=="focal_z_dist_day",:Destination][1]
        grid_file = model.files[model.files.File .=="grid",:Destination][1]
        z_day_dist = CSV.read(z_dist_file,DataFrame)
        grid = CSV.read(grid_file,DataFrame)
        maxdepth = grid[grid.Name .== "depthmax", :Value][1]

        data.mig_status[ind] = 2
        data.target_z[ind] = gaussmix(1,z_day_dist[sp,"mu1"],z_day_dist[sp,"mu2"],z_day_dist[sp,"mu3"],z_day_dist[sp,"sigma1"],z_day_dist[sp,"sigma2"],z_day_dist[sp,"sigma3"],z_day_dist[sp,"lambda1"],z_day_dist[sp,"lambda2"])[1]

        while (data.target_z[ind] <= 0) | (data.target_z[ind] > maxdepth) #Resample if animal is outside of the grid
            data.target_z[ind] = gaussmix(1,z_day_dist[sp,"mu1"],z_day_dist[sp,"mu2"],z_day_dist[sp,"mu3"],z_day_dist[sp,"sigma1"],z_day_dist[sp,"sigma2"],z_day_dist[sp,"sigma3"],z_day_dist[sp,"lambda1"],z_day_dist[sp,"lambda2"])[1]
        end

        data.mig_rate[ind] = swim_speed
        t_adjust = min(ΔT, abs((data.target_z[ind] - data.z[ind]) / data.mig_rate[ind]))
        data.z[ind] = min(data.target_z[ind], data.z[ind] + data.mig_rate[ind] * ΔT)
        data.active_time[ind] += t_adjust
        if data.z[ind] >= data.target_z[ind]
            data.mig_status[ind] = -1
        end
        if sp == 1
            outputs.behavior[ind,3,1] += ΔT
        else
            outputs.behavior[(sum(model.ninds[1:(sp-1)])+ind),3,1] += ΔT
        end
        data.feeding[ind] = 0
    elseif (model.t >= 18*60) && (data.mig_status[ind] == -1) # Start acent during nighttime
        z_dist_file = model.files[model.files.File .=="focal_z_dist_night",:Destination][1]
        grid_file = model.files[model.files.File .=="grid",:Destination][1]
        z_night_dist = CSV.read(z_dist_file,DataFrame)
        grid = CSV.read(grid_file,DataFrame)
        maxdepth = grid[grid.Name .== "depthmax", :Value][1]

        data.mig_status[ind] = 1
        data.target_z[ind] = gaussmix(1,z_night_dist[sp,"mu1"],z_night_dist[sp,"mu2"],z_night_dist[sp,"mu3"],z_night_dist[sp,"sigma1"],z_night_dist[sp,"sigma2"],z_night_dist[sp,"sigma3"],z_night_dist[sp,"lambda1"],z_night_dist[sp,"lambda2"])[1]

        while (data.target_z[ind] <= 0) | (data.target_z[ind] > maxdepth) #Resample if animal is outside of the grid
            data.target_z[ind] = gaussmix(1,z_night_dist[sp,"mu1"],z_night_dist[sp,"mu2"],z_night_dist[sp,"mu3"],z_night_dist[sp,"sigma1"],z_night_dist[sp,"sigma2"],z_night_dist[sp,"sigma3"],z_night_dist[sp,"lambda1"],z_night_dist[sp,"lambda2"])[1]
        end
        data.mig_rate[ind] = swim_speed
        t_adjust = min(ΔT, abs((data.target_z[ind] - data.z[ind]) / data.mig_rate[ind]))
        data.z[ind] = max(data.target_z[ind], data.z[ind] - data.mig_rate[ind] * ΔT)
        if data.z[ind] == data.target_z[ind]
            data.mig_status[ind] = 0
            data.feeding[ind] = 1
        end
        data.active_time[ind] += t_adjust
        if sp == 1
            outputs.behavior[ind,3,1] += ΔT
        else
            outputs.behavior[(sum(model.ninds[1:(sp-1)])+ind),3,1] += ΔT
        end
    elseif data.mig_status[ind] == 1 # Continue ascending
        target_z = data.target_z[ind]
        t_adjust = min(ΔT, abs((target_z - data.z[ind]) / data.mig_rate[ind]))
        data.z[ind] = max(target_z, data.z[ind] - data.mig_rate[ind] * ΔT)
        if (data.z[ind] == target_z) | (model.t == 21*60)
            data.mig_status[ind] = 0
            data.feeding[ind] = 1
        end
        if sp == 1
            outputs.behavior[ind,3,1] += ΔT
        else
            outputs.behavior[(sum(model.ninds[1:(sp-1)])+ind),3,1] += ΔT
        end
        data.active_time[ind] += t_adjust
    elseif data.mig_status[ind] == 2 # Continue descending
        target_z = data.target_z[ind]
        t_adjust = min(ΔT, abs((target_z - data.z[ind]) / data.mig_rate[ind]))
        data.z[ind] = min(target_z, data.z[ind] + data.mig_rate[ind] * ΔT)
        if (data.z[ind] == target_z) | (model.t == 9*60)
            data.mig_status[ind] = -1
        end
        if sp == 1
            outputs.behavior[ind,3,1] += ΔT
        else
            outputs.behavior[(sum(model.ninds[1:(sp-1)])+ind),3,1] += ΔT
        end
        data.active_time[ind] += t_adjust
    end
    return nothing
end


function surface_dive(model, sp, ind)
    animal = model.individuals.animals[sp].data
    animal_p = model.individuals.animals[sp].p
    # Define behavioral parameters
    surface_interval = animal_p.Surface_Interval[2][sp]
    foraging_interval = animal_p.Dive_Interval[2][sp]
    dive_velocity = 1.5 * 60 # meters per minute

    # Time resolution
    ΔT = animal_p.t_resolution[2][sp]

    # Progress the animal through the stages
    if animal.mig_status[ind] == 0
        # Surface interval
        animal.interval[ind] += ΔT
        max_fullness = 0.03 * animal.weight[ind]
        dive_trigger = animal.gut_fullness[ind] / max_fullness
        dist = logistic(dive_trigger, 5, 0.5)
        #Add probability here to start diving

        if animal.interval[ind] >= surface_interval && rand() > dist
            # Reset interval timer and move to the next stage (diving)
            animal.interval[ind] = 0
            animal.mig_status[ind] = 1
            # Randomly select dive depth
            animal.target_z[ind] = sample_normal(animal_p.Dive_depth_min[2][sp], animal_p.Dive_depth_max[2][sp], std=20)[rand(1:end)]
        end

    elseif animal.mig_status[ind] == 1
        # Continue diving
        animal.active_time[ind] += ΔT

        #Change depth
        if animal.z[ind] >= animal.target_z[ind]
            # Reset active time and move to the next stage (foraging interval)
            animal.mig_status[ind] = -1
            animal.feeding[ind] = 1
        end

    elseif animal.mig_status[ind] == -1
        # Foraging interval
        animal.interval[ind] += ΔT

        if animal.interval[ind] >= foraging_interval
            # Reset interval timer and move to the next stage (ascending)
            animal.interval[ind] = 0
            animal.mig_status[ind] = 2
            animal.feeding[ind] = 0
        end

    elseif animal.mig_status[ind] == 2
        # Ascending
        animal.active_time[ind] += ΔT

        if animal.z[ind] <= 0
            # Reset active time and move to the next cycle (back to surface interval)
            animal.interval[ind] = 0
            animal.mig_status[ind] = 0

            # Increment dive count for the day
            animal.dives_remaining[ind] -= 1
        end
    end

    # Check if the dive count for the day exceeds the maximum limit
    if animal.dives_remaining[ind] <= 0
        # Prevent further diving
        animal.mig_status[ind] = 0
    end

    # Update the position of the animal based on the dive velocity
    if animal.mig_status[ind] == 1
        animal.z[ind] = min(animal.target_z[ind], animal.z[ind] + dive_velocity * ΔT)
    elseif animal.mig_status[ind] == 2
        animal.z[ind] = max(0, animal.z[ind] - dive_velocity * ΔT)
    end

    return nothing
end


function pelagic_dive(model, sp, ind)
    animal = model.individuals.animals[sp].data
    animal_p = model.individuals.animals[sp].p
    # Define behavioral parameters
    surface_interval = animal_p.Surface_Interval[2][sp]
    foraging_interval = animal_p.Dive_Interval[2][sp]
    dive_velocity = 1.5 * 60 # meters per minute

    # Time resolution
    ΔT = animal_p.t_resolution[2][sp]

    # Progress the animal through the stages
    if animal.mig_status[ind] == 0
        # Surface interval
        animal.interval[ind] += ΔT
        max_fullness = 0.03 * animal.weight[ind]
        dive_trigger = animal.gut_fullness[ind] / max_fullness
        dist = logistic(dive_trigger, 5, 0.5)
        #Add probability here to start diving

        if animal.interval[ind] >= surface_interval && rand() > dist
            # Reset interval timer and move to the next stage (diving)
            animal.interval[ind] = 0
            animal.mig_status[ind] = 1
            # Randomly select dive depth
            animal.target_z[ind] = sample_normal(animal_p.Dive_depth_min[2][sp], animal_p.Dive_depth_max[2][sp], std=20)[rand(1:end)]
        end

    elseif animal.mig_status[ind] == 1
        # Continue diving
        animal.active_time[ind] += ΔT

        #Change depth
        if animal.z[ind] >= animal.target_z[ind]
            # Reset active time and move to the next stage (foraging interval)
            animal.mig_status[ind] = -1
            animal.feeding[ind] = 1
        end

    elseif animal.mig_status[ind] == -1
        # Foraging interval
        animal.interval[ind] += ΔT

        if animal.interval[ind] >= foraging_interval
            # Reset interval timer and move to the next stage (ascending)
            animal.interval[ind] = 0
            animal.mig_status[ind] = 2
            animal.feeding[ind] = 0
            animal.target_z[ind] = sample_normal(animal_p.Night_depth_min[2][sp], animal_p.Night_depth_max[2][sp], std=20)[rand(1:end)]
        end

    elseif animal.mig_status[ind] == 2
        # Ascending
        animal.active_time[ind] += ΔT

        if animal.z[ind] <= animal.target_z[ind]
            # Reset active time and move to the next cycle (back to surface interval)
            animal.interval[ind] = 0
            animal.mig_status[ind] = 0

            # Increment dive count for the day
            animal.dives_remaining[ind] -= 1
        end
    end

    # Check if the dive count for the day exceeds the maximum limit
    if animal.dives_remaining[ind] <= 0
        # Prevent further diving
        animal.mig_status[ind] = 0
    end

    # Update the position of the animal based on the dive velocity
    if animal.mig_status[ind] == 1
        animal.z[ind] = min(animal.target_z[ind], animal.z[ind] + dive_velocity * ΔT)
    elseif animal.mig_status[ind] == 2
        animal.z[ind] = max(animal.target_z[ind], animal.z[ind] - dive_velocity * ΔT)
    end

    return nothing
end

function cost_function_prey(prey_location, predator_locations)
    total_distance = sum(norm(prey_location .- predator) for predator in 1:nrow(predator_locations))
    return -total_distance
end

#Optimization function to find the ideal location for prey to go
function predator_avoidance(predator_locations,initial_prey_location,max_distance)
    initial_guess = initial_prey_location # Initial guess for prey location

    lower_bound = initial_guess .- max_distance   # Lower bound for prey location
    upper_bound = initial_guess .+ max_distance   # Upper bound for prey location

    result = optimize(p -> cost_function_prey(p, predator_locations), lower_bound, upper_bound, initial_guess)

    return Optim.minimizer(result)
end

function random_movement(original_position,distance)
    # Generate a random direction vector
    random_direction = normalize(randn(3))
    
    # Generate a random distance within the maximum allowed distance
    random_distance = rand() * distance
    
    # Calculate the new position based on the random direction and distance
    new_position = original_position + random_distance * random_direction
    
    return new_position
end
