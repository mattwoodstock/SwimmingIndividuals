function behavior(model, sp, ind, outputs)
    behave_type = model.individuals.animals[sp].p.Type[2][sp]  # A variable that determines the behavioral type of an animal
    
    if behave_type == "dvm_strong"
        dvm_action(model, sp, ind)
        not_migrating = findall(x -> x .<= 0.0,model.individuals.animals[sp].data.mig_status[ind])
        if length(not_migrating) > 0
            decision(model, sp, ind[not_migrating],outputs)
        end
    elseif behave_type == "dvm_weak"
        max_fullness = 0.2 * model.individuals.animals[sp].data.biomass[ind]
        feed_trigger = model.individuals.animals[sp].data.gut_fullness[ind] / max_fullness
        dist = logistic(feed_trigger, 5, 0.5)  # Resample from negative sigmoidal relationship
        
        if model.t >= 18 * 60 && model.individuals.animals[sp].data.mig_status[ind] == -1 && rand() >= dist
            decision(model, sp, ind, outputs)  # Animal does not migrate when it has the chance. Behaves as normal
        else
            dvm_action(model, sp, ind)  # Animal either migrates or continues what it should do
            decision(model, sp, ind, outputs)
        end
    elseif behave_type in ("surface_diver", "pelagic_diver")
        dive_func = behave_type == "surface_diver" ? surface_dive : pelagic_dive
        dive_func(model, sp, ind)  # Function of energy density and dive characteristics to start dive
        decision(model, sp, ind, outputs)
    elseif behave_type == "non_mig"
        decision(model, sp, ind, outputs)
    end
    return nothing
end

function predators(model, sp, ind)
    # Precompute constant values
    min_pred_limit = model.individuals.animals[sp].p.Min_Prey[2][sp]
    max_pred_limit = model.individuals.animals[sp].p.Max_Prey[2][sp]
    # Gather distances
    detection = model.individuals.animals[sp].data.vis_pred[ind]
    calculate_distances_pred(model,sp,ind,min_pred_limit,max_pred_limit,detection)
end

function preys(model, sp, ind)
    # Precompute constant values
    min_prey_limit = model.individuals.animals[sp].p.Min_Prey[2][sp]
    max_prey_limit = model.individuals.animals[sp].p.Max_Prey[2][sp]
    # Gather distances
    detection = model.individuals.animals[sp].data.vis_prey[ind]
    prey = calculate_distances_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)

    return prey
end

function patch_preys(model, sp, ind)
    # Precompute constant values
    min_prey_limit = model.pools.pool[sp].characters.Min_Prey[2][sp]
    max_prey_limit = model.pools.pool[sp].characters.Max_Prey[2][sp]
    # Gather distances
    detection = model.pools.pool[sp].data.vis_prey[ind]
    prey = calculate_distances_patch_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)
    return prey
end

function decision(model, sp, ind, outputs)  
    max_fullness = 0.2 * model.individuals.animals[sp].data.biomass[ind]     
    feed_trigger = model.individuals.animals[sp].data.gut_fullness[ind] ./ max_fullness
    val1 = rand(length(ind))

    # Individual avoids predators if predators exist
    preds = predators(model, sp, ind)  #Create list of predators
    prey = preys(model, sp, ind)  #Create list of preys for all individuals in the species

    to_eat = findall(feed_trigger .<= val1)
    not_eat = findall(feed_trigger .> val1)

    eating = ind[to_eat]
    not_eating = ind[not_eat]

    if length(to_eat) > 0
        time = eat(model, sp, eating,to_eat, prey, outputs)
        predator_avoidance(model,time,eating,to_eat,preds,sp)
    end

    if length(not_eating) > 0
        time = fill(model.individuals.animals[sp].p.t_resolution[2][sp] * 60,length(not_eating))
        predator_avoidance(model,time,not_eating,not_eat,preds,sp)
    end 
    
    #Clear these as they are no longer necessary and take up memory.
    prey = Vector{PreyInfo}
    preds = Vector{PredatorInfo}
end

function visual_range_preds_init(length,depth,min_pred,max_pred,ind)
    pred_contrast = fill(0.3,ind) # Utne-Plam (1999)
    salt = fill(30, ind) # PSU. Add this later if it were to change
    attenuation_coefficient = 0.64 .- 0.016 .* salt
    ind_length = length ./ 1000 # Length in meters
    pred_length = ind_length ./ 0.01 # Largest possible pred-prey size ratio
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30 # The maximum visual range. Currently, this is 1 body length
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(0) # Need real value, in W m^-2
    
    # Light intensity at depth
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
        
    # Constant reflecting fish's visual sensitivity and target size
    pred_size_factor = 1+((min_pred+max_pred)/2) # Based on assumed half prey-pred size relationship
        
    # Visual range as a function of body size and light
    r = max.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* pred_size_factor)))
    return r
end

function visual_range_preys_init(length,depth,min_prey,max_prey,ind)
    pred_contrast = fill(0.3,ind) # Utne-Plam (1999)
    salt = fill(30, ind) # PSU. Add this later if it were to change
    attenuation_coefficient = 0.64 .- 0.016 .* salt
    ind_length = length ./ 1000 # Length in meters
    pred_length = ind_length ./ 0.01 # Largest possible pred-prey size ratio
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30 # The maximum visual range. Currently, this is 1 body length
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(0) # Need real value, in W m^-2

    # Light intensity at depth
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
    
    # Constant reflecting fish's visual sensitivity and target size
    prey_size_factor = (min_prey+max_prey)/2 # Based on assumed half prey-pred size relationship
    
    # Visual range as a function of body size and light
    r = max.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* prey_size_factor)))
    return r
end

function visual_range_pred(model,length,depth,sp,ind)
    min_pred = model.individuals.animals[sp].p.Min_Prey[2][sp]
    max_pred = model.individuals.animals[sp].p.Max_Prey[2][sp]
    pred_contrast = fill(0.3,ind) # Utne-Plam (1999)
    salt = fill(30, ind) # PSU. Add this later if it were to change
    attenuation_coefficient = 0.64 .- 0.016 .* salt
    ind_length = length ./ 1000 # Length in meters
    pred_length = ind_length ./ 0.01 # Largest possible pred-prey size ratio
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30 # The maximum visual range. Currently, this is 1 body length
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(model.t) # Need real value, in W m^-2

    # Light intensity at depth
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
    
    # Constant reflecting fish's visual sensitivity and target size
    pred_size_factor = 1+((min_pred+max_pred)/2) # Based on assumed half prey-pred size relationship
    
    # Visual range as a function of body size and light
    r = max.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* pred_size_factor)))
    return r
end

function visual_range_prey(model,length,depth,sp,ind)
    min_prey = model.individuals.animals[sp].p.Min_Prey[2][sp]
    max_prey = model.individuals.animals[sp].p.Max_Prey[2][sp]
    pred_contrast = fill(0.3,ind) # Utne-Plam (1999)
    salt = fill(30, ind) # PSU. Add this later if it were to change
    attenuation_coefficient = 0.64 .- 0.016 .* salt
    ind_length = length ./ 1000 # Length in meters
    pred_length = ind_length ./ 0.01 # Largest possible pred-prey size ratio
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30 # The maximum visual range. Currently, this is 1 body length
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(model.t) # Need real value, in W m^-2

    # Light intensity at depth
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
    
    # Constant reflecting fish's visual sensitivity and target size
    prey_size_factor = (min_prey+max_prey)/2 # Based on assumed half prey-pred size relationship
    
    # Visual range as a function of body size and light
    r = max.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* prey_size_factor)))
    return r
end

# Function for the cost function used to triangulate the best distance
function cost_function_prey(prey_location, preds)
    total_distance = sum(norm(prey_location .- predator) for predator in preds)
    return -total_distance
end

#Optimization function to find the ideal location for prey to go
function optimize_prey_location(model,sp,ind,preds)
    initial_guess = [model.individuals.animals[sp].data.x[ind],model.individuals.animals[sp].data.y[ind],model.individuals.animals[sp].data.z[ind]] # Initial guess for prey location

    max_distance = model.individuals.animals[sp].p.Swim_velo[2][sp] * model.individuals.animals[sp].data.length[ind] * 60 * model.individuals.animals[sp].p.t_resolution[2][sp] # Calculate maximum swim velocity the animal could move.

    lower_bound = initial_guess .- max_distance   # Lower bound for prey location
    upper_bound = initial_guess .+ max_distance   # Upper bound for prey location

    result = optimize(p -> cost_function_prey(p, preds), lower_bound, upper_bound, initial_guess)

    return Optim.minimizer(result)
end