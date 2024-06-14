function dvm_action(model, sp, ind,outputs)
    for i in ind[1]:ind[1]
        animal = model.individuals.animals[sp]
        data = animal.data
        params = animal.p
        ΔT = params.t_resolution[2][sp]

        swim_speed = data.length[i]/1000 * params.Swim_velo[2][sp] * ΔT * 60

        if (6*60 <= model.t < 18*60) && (any(data.mig_status[i] == 0.0)) # Start descent during daytime

            z_dist_file = model.files[model.files.File .=="focal_z_dist_day",:Destination][1]
            grid_file = model.files[model.files.File .=="grid",:Destination][1]
            z_day_dist = CSV.read(z_dist_file,DataFrame)
            grid = CSV.read(grid_file,DataFrame)
            maxdepth = grid[grid.Name .== "depthmax", :Value][1]

            data.mig_status[i] = 2
            data.target_z[i] = gaussmix(1,z_day_dist[sp,"mu1"],z_day_dist[sp,"mu2"],z_day_dist[sp,"mu3"],z_day_dist[sp,"sigma1"],z_day_dist[sp,"sigma2"],z_day_dist[sp,"sigma3"],z_day_dist[sp,"lambda1"],z_day_dist[sp,"lambda2"])[1]

            while (data.target_z[i] < 1) | (data.target_z[i] > maxdepth) | (data.target_z[i] < data.z[i]) #Resample if animal is outside of the grid
                data.target_z[i] = gaussmix(1,z_day_dist[sp,"mu1"],z_day_dist[sp,"mu2"],z_day_dist[sp,"mu3"],z_day_dist[sp,"sigma1"],z_day_dist[sp,"sigma2"],z_day_dist[sp,"sigma3"],z_day_dist[sp,"lambda1"],z_day_dist[sp,"lambda2"])[1]
            end

            data.mig_rate[i] = swim_speed
            t_adjust = min(ΔT, abs((data.target_z[i] - data.z[i]) / data.mig_rate[i]))
            data.z[i] = min(data.target_z[i], data.z[i] + data.mig_rate[i] * ΔT)
            data.active_time[i] += t_adjust
            if data.z[i] >= data.target_z[i]
                data.mig_status[i] = -1
            end
            if sp == 1
                outputs.behavior[i,3,1] += ΔT
            else
                outputs.behavior[(sum(model.ninds[1:(sp-1)])+i),3,1] += ΔT
            end
            data.feeding[i] = 0
        elseif (model.t >= 18*60) && (data.mig_status[i] == -1) # Start acent during nighttime
            z_dist_file = model.files[model.files.File .=="focal_z_dist_night",:Destination][1]
            grid_file = model.files[model.files.File .=="grid",:Destination][1]
            z_night_dist = CSV.read(z_dist_file,DataFrame)
            grid = CSV.read(grid_file,DataFrame)
            maxdepth = grid[grid.Name .== "depthmax", :Value][1]

            data.mig_status[i] = 1
            data.target_z[i] = gaussmix(1,z_night_dist[sp,"mu1"],z_night_dist[sp,"mu2"],z_night_dist[sp,"mu3"],z_night_dist[sp,"sigma1"],z_night_dist[sp,"sigma2"],z_night_dist[sp,"sigma3"],z_night_dist[sp,"lambda1"],z_night_dist[sp,"lambda2"])[1]

            while (data.target_z[i] < 1) | (data.target_z[i] > maxdepth) | (data.target_z[i] > data.z[i]) #Resample if animal is outside of the grid
                data.target_z[i] = gaussmix(1,z_night_dist[sp,"mu1"],z_night_dist[sp,"mu2"],z_night_dist[sp,"mu3"],z_night_dist[sp,"sigma1"],z_night_dist[sp,"sigma2"],z_night_dist[sp,"sigma3"],z_night_dist[sp,"lambda1"],z_night_dist[sp,"lambda2"])[1]
            end
            data.mig_rate[i] = swim_speed
            t_adjust = min(ΔT, abs((data.target_z[i] - data.z[i]) / data.mig_rate[i]))
            data.z[i] = max(data.target_z[i], data.z[i] - data.mig_rate[i] * ΔT)
            if data.z[i] == data.target_z[i]
                data.mig_status[i] = 0
                data.feeding[i] = 1
            end
            data.active_time[i] += t_adjust
            if sp == 1
                outputs.behavior[i,3,1] += ΔT
            else
                outputs.behavior[(sum(model.ninds[1:(sp-1)])+i),3,1] += ΔT
            end
        elseif data.mig_status[i] == 1 # Continue ascending
            target_z = data.target_z[i]
            t_adjust = min(ΔT, abs((target_z - data.z[i]) / data.mig_rate[i]))
            data.z[i] = max(target_z, data.z[i] - data.mig_rate[i] * ΔT)
            if (data.z[i] == target_z) | (model.t == 21*60)
                data.mig_status[i] = 0
                data.feeding[i] = 1
            end
            if sp == 1
                outputs.behavior[i,3,1] += ΔT
            else
                outputs.behavior[(sum(model.ninds[1:(sp-1)])+i),3,1] += ΔT
            end
            data.active_time[i] += t_adjust
        elseif data.mig_status[i] == 2 # Continue descending
            target_z = data.target_z[i]
            t_adjust = min(ΔT, abs((target_z - data.z[i]) / data.mig_rate[i]))
            data.z[i] = min(target_z, data.z[i] + data.mig_rate[i] * ΔT)
            if (data.z[i] == target_z) | (model.t == 9*60)
                data.mig_status[i] = -1
            end
            if sp == 1
                outputs.behavior[i,3,1] += ΔT
            else
                outputs.behavior[(sum(model.ninds[1:(sp-1)])+i),3,1] += ΔT
            end
            data.active_time[i] += t_adjust
        end
    end
    return nothing
end

function surface_dive(model, sp, ind)
    for i in ind[1]:ind[1]
        animal = model.individuals.animals[sp].data
        animal_p = model.individuals.animals[sp].p
        # Define behavioral parameters
        surface_interval = animal_p.Surface_Interval[2][sp]
        foraging_interval = animal_p.Dive_Interval[2][sp]
        dive_velocity = (animal.length[i]/1000) * 60 # meters per minute

        # Time resolution
        ΔT = animal_p.t_resolution[2][sp]
        # Progress the animal through the stages
        if animal.mig_status[i] == 0
            # Surface interval

            animal.interval[i] += ΔT
            max_fullness = 0.03 * animal.weight[i]
            dive_trigger = animal.gut_fullness[i] / max_fullness
            dist = logistic(dive_trigger, 5, 0.5)
            #Add probability here to start diving

            if animal.interval[i] >= surface_interval && rand() > dist
                # Reset interval timer and move to the next stage (diving)
                animal.interval[i] = 0
                animal.mig_status[i] = 1
                # Randomly select dive depth
                animal.target_z[i] = sample_normal(animal_p.Dive_depth_min[2][sp], animal_p.Dive_depth_max[2][sp], std=20)[rand(1:end)]
            end

        elseif animal.mig_status[i] == 1
            # Continue diving
            animal.active_time[i] += ΔT

            #Change depth
            if animal.z[i] >= animal.target_z[i]
                # Reset active time and move to the next stage (foraging interval)
                animal.mig_status[i] = -1
                animal.feeding[i] = 1
            end

        elseif animal.mig_status[i] == -1
            # Foraging interval

            if animal.interval[i] >= foraging_interval
                # Reset interval timer and move to the next stage (ascending)
                animal.interval[i] = 0
                animal.mig_status[i] = 2
                animal.feeding[i] = 0
            else
                animal.interval[i] += ΔT
            end

        elseif animal.mig_status[i] == 2
            # Ascending
            animal.active_time[i] += ΔT

            if animal.z[i] <= 1
                # Reset active time and move to the next cycle (back to surface interval)
                animal.interval[i] = 0
                animal.mig_status[i] = 0

                # Increment dive count for the day
                animal.dives_remaining[i] -= 1
            end
        end

        # Check if the dive count for the day exceeds the maximum limit
        if animal.dives_remaining[i] <= 0
            # Prevent further diving
            animal.mig_status[i] = 0
        end

        # Update the position of the animal based on the dive velocity
        if animal.mig_status[i] == 1
            animal.z[i] = min(animal.target_z[i], animal.z[i] + dive_velocity * ΔT)
        elseif animal.mig_status[i] == 2
            animal.z[i] = max(1, animal.z[i] - dive_velocity * ΔT)
        end
    end

    return nothing
end

function pelagic_dive(model, sp, ind)

    for i in ind[1]:ind[1]
        animal = model.individuals.animals[sp].data
        animal_p = model.individuals.animals[sp].p
        # Define behavioral parameters
        surface_interval = animal_p.Surface_Interval[2][sp]
        foraging_interval = animal_p.Dive_Interval[2][sp]
        dive_velocity = (animal.length[i]/1000) * 60 # meters per minute

        # Time resolution
        ΔT = animal_p.t_resolution[2][sp]

        # Progress the animal through the stages
        if animal.mig_status[i] == 0
            # Surface interval
            animal.interval[i] += ΔT
            max_fullness = 0.03 * animal.weight[i]
            dive_trigger = animal.gut_fullness[i] / max_fullness
            dist = logistic(dive_trigger, 5, 0.5)
            #Add probability here to start diving


            if animal.interval[i] >= surface_interval && rand() > dist
                # Reset interval timer and move to the next stage (diving)
                animal.interval[i] = 0
                animal.mig_status[i] = 1
                # Randomly select dive depth
                animal.target_z[i] = sample_normal(animal_p.Dive_depth_min[2][sp], animal_p.Dive_depth_max[2][sp], std=20)[rand(1:end)]
            end

        elseif animal.mig_status[i] == 1
            # Continue diving
            animal.active_time[i] += ΔT

            #Change depth
            if animal.z[i] >= animal.target_z[i]
                # Reset active time and move to the next stage (foraging interval)
                animal.mig_status[i] = -1
                animal.feeding[i] = 1
            end

        elseif animal.mig_status[i] == -1
            # Foraging interval
            animal.interval[i] += ΔT

            if animal.interval[i] >= foraging_interval
                # Reset interval timer and move to the next stage (ascending)
                animal.interval[i] = 0
                animal.mig_status[i] = 2
                animal.feeding[i] = 0
                animal.target_z[i] = sample_normal(animal_p.Night_depth_min[2][sp], animal_p.Night_depth_max[2][sp], std=20)[rand(1:end)]
            end

        elseif animal.mig_status[i] == 2
            # Ascending
            animal.active_time[i] += ΔT

            if animal.z[i] <= animal.target_z[i]
                # Reset active time and move to the next cycle (back to surface interval)
                animal.interval[i] = 0
                animal.mig_status[i] = 0

                # Increment dive count for the day
                animal.dives_remaining[i] -= 1
            end
        end

        # Check if the dive count for the day exceeds the maximum limit
        if animal.dives_remaining[i] <= 0
            # Prevent further diving
            animal.mig_status[i] = 0
        end

        # Update the position of the animal based on the dive velocity
        if animal.mig_status[i] == 1
            animal.z[i] = min(animal.target_z[i], animal.z[i] + dive_velocity * ΔT)
        elseif animal.mig_status[i] == 2
            animal.z[i] = max(animal.target_z[i], animal.z[i] - dive_velocity * ΔT)
        end
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

function random_movement(x,y,distance)
    original_position = [x[1],y[1]]

    # Generate a random direction vector
    random_direction = normalize(randn(2))
    
    # Generate a random distance within the maximum allowed distance
    random_distance = rand() * distance * 0.1

    # Calculate the new position based on the random direction and distance
    new_position = original_position .+ random_distance .* random_direction
    
    return new_position
end

function pool_shift(model,pool)

    files = model.files

    if model.t == (7*60)
        z_file = files[files.File .== "nonfocal_z_dist_day", :Destination][1]
    else 
        z_file = files[files.File .== "nonfocal_z_dist_night", :Destination][1]
    end
    
    grid_file = files[files.File .== "grid", :Destination][1]
    state_file = files[files.File .== "state", :Destination][1]

    z_dist = CSV.read(z_file, DataFrame)
    grid = CSV.read(grid_file, DataFrame)
    state = CSV.read(state_file, DataFrame)

    food_limit = parse(Float64, state[state.Name .== "food_exp", :Value][1])

    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]

    z_interval = maxdepth / depthres

    means = [z_dist[pool, "mu1"], z_dist[pool, "mu2"], z_dist[pool, "mu3"]]
    stds = [z_dist[pool, "sigma1"], z_dist[pool, "sigma2"], z_dist[pool, "sigma3"]]
    weights = [z_dist[pool, "lambda1"], z_dist[pool, "lambda2"], z_dist[pool, "lambda3"]]

    x_values = 0:maxdepth
    pdf_values = multimodal_distribution.(Ref(x_values), means, stds, weights)
    
    min_z = round.(Int, z_interval .* (1:g.Nz) .- z_interval .+ 1)
    max_z = round.(Int, z_interval .* (1:g.Nz) .+ 1)

    #Individuals per cubic meter.
    density = [sum(@view pdf_values[1][min_z[k]:max_z[k]]) .* model.pools.pool[pool].characters.Total_density[2][pool] / maxdepth for k in 1:g.Nz]

    max_z_lt_200 = max_z .< 200
    food_limit_arr = fill(food_limit, g.Nx, g.Ny)
    density_num = ifelse.(max_z_lt_200, density .* food_limit_arr, density)

    model.pools.pool[pool].density = reshape(density_num, 1, 1, :)
end
