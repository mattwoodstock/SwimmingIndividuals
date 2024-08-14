function behavior(model, sp, ind, outputs)
    behave_type = model.individuals.animals[sp].p.Type[2][sp]  # A variable that determines the behavioral type of an animal
    
    if behave_type == "dvm_strong"
        dvm_action(model, sp, ind)
        decision(model, sp, ind,outputs)
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
    min_pred_limit = 0.01
    max_pred_limit = 0.2
    # Gather distances
    detection = model.individuals.animals[sp].data.vis_pred[ind]
    calculate_distances_pred(model,sp,ind,min_pred_limit,max_pred_limit,detection)
end

function preys(model, sp, ind)
    # Precompute constant values
    min_prey_limit = 0.01
    max_prey_limit = 0.2
    # Gather distances
    detection = model.individuals.animals[sp].data.vis_prey[ind]
    prey = calculate_distances_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)

    return prey
end

function patch_preys(model, sp, ind)
    # Precompute constant values
    min_prey_limit = 0.01
    max_prey_limit = 0.2
    # Gather distances
    detection = model.pools.pool[sp].data.vis_prey[ind]
    prey = calculate_distances_patch_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)
    return prey
end

function decision(model, sp, ind, outputs)  
    max_fullness = 0.2 *model.individuals.animals[sp].data.biomass[ind]     
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
        time_remaining = findall(x -> x>0,time)
        predator_avoidance(model,time,eating,to_eat,preds,sp)
    end

    if length(not_eating) > 0
        time = fill(model.individuals.animals[sp].p.t_resolution[2][sp] * 60,length(not_eating))
        predator_avoidance(model,time,not_eating,not_eat,preds,sp)
    end 
    
    #Clear these as they are no longer necessary and take up memory.
    prey.preys = NamedTuple()
    preds.preds = NamedTuple()
end

function visual_range_preds_init(arch,length,depth,ind)
    if arch == CPU()
        salt = fill(30,ind) # Needs to be in PSU
        pred_contrast = fill(0.3,ind) # Utne-Plam (1999)
        eye_saturation = fill(4 * 10^-4,ind)
    else
        salt = CUDA.fill(30,ind) # Needs to be in PSU
        pred_contrast = CUDA.fill(0.3,ind) # Utne-Plam (1999)
        eye_saturation = CUDA.fill(4 * 10^-4,ind)
    end
    ind_length = length ./ 1000 # Length in meters
    pred_length = ind_length ./ 0.01 # Largest possible pred-prey size ratio
    surface_irradiance = 300 # Need real value, in W m^-2
    pred_width = pred_length ./ 4
    rmax = ind_length .* 30 # The maximum visual range. Currently, this is 1 body length
    # Light attenuation coefficient
    a_lat = 0.64 .- 0.016 .* salt # Aksnes et al. 2009; per meter
    # Beam Attenuation coefficient
    c_lat = 4.87 .* a_lat # Huse and Fiksen (2010); per meter
    # Compute irradiance based on architecture
    if arch == CPU()
        i_td = surface_irradiance .* exp.(-0.1 .* depth)
    else
        z_cpu = depth # Assuming 'z' is already a Julia array
        z_gpu = CuArray(z_cpu) # Convert 'z' to a CuArray
        # Perform the exponential operation on the GPU
        i_td = surface_irradiance .* CUDA.exp.(-0.1 .* z_gpu) 
    end
    # Prey image area
    pred_image = 0.75 .* pred_length .* pred_width
    # Visual eye sensitivity
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    # Define f(x) and its derivative fp(x)
    f(x) = x.^2 .* exp.(c_lat .* x) .- (pred_contrast .* pred_image .* eye_sensitivity .* (i_td ./ (eye_saturation .+ i_td)))
    fp(x) = 2 .* x .* exp.(c_lat .* x) .+ x.^2 .* c_lat .* exp.(c_lat .* x)
    # Call the Newton-Raphson method
    x = newton_raphson(f, fp)
    # Visual range estimates may be much more complicated when considering bioluminescent organisms. Could incorporate this if we assigned each species a "luminous type"
    ## https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3886326/
    return x
end

function visual_range_preys_init(length,depth,ind)

    salt = fill(30,ind) # Needs to be in PSU
    pred_contrast = fill(0.3,ind) # Utne-Plam (1999)
    eye_saturation = fill(4 * 10^-4,ind)

    ind_length = length ./ 1000 # Length in meters
    pred_length = ind_length .* 0.05 # Largest possible pred-prey size ratio
    surface_irradiance = 300 # Need real value, in W m^-2
    pred_width = pred_length ./ 4
    rmax = ind_length .* 30 # The maximum visual range. Currently, this is 1 body length
    # Light attenuation coefficient
    a_lat = 0.64 .- 0.016 .* salt # Aksnes et al. 2009; per meter
    # Beam Attenuation coefficient
    c_lat = 4.87 .* a_lat # Huse and Fiksen (2010); per meter
    # Compute irradiance based on architecture
    i_td = surface_irradiance .* exp.(-0.1 .* depth)

    # Prey image area
    pred_image = 0.75 .* pred_length .* pred_width
    # Visual eye sensitivity
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    # Define f(x) and its derivative fp(x)
    f(x) = x.^2 .* exp.(c_lat .* x) .- (pred_contrast .* pred_image .* eye_sensitivity .* (i_td ./ (eye_saturation .+ i_td)))
    fp(x) = 2 .* x .* exp.(c_lat .* x) .+ x.^2 .* c_lat .* exp.(c_lat .* x)
    # Call the Newton-Raphson method
    range = newton_raphson(f, fp)
    threshold = 0.1 # Non-visual detection range.
    indices = findall(x -> x < threshold, range)
    if !isempty(indices) > 0
        range[indices] .= 0.01 #1 cm threshold
    end    
    return range
end

function visual_range_preds(model, sp, ind)
    if model.arch == CPU()
        salt = fill(30,length(ind)) # Needs to be in PSU
        pred_contrast = fill(0.3,length(ind)) # Utne-Plam (1999)
        eye_saturation = fill(4 * 10^-4,length(ind))
    else
        salt = CUDA.fill(30,length(ind)) # Needs to be in PSU
        pred_contrast = CUDA.fill(0.3,length(ind)) # Utne-Plam (1999)
        eye_saturation = CUDA.fill(4 * 10^-4,length(ind))
    end
    ind_length = model.individuals.animals[sp].data.length[ind] ./ 1000 # Length in meters
    pred_length = ind_length ./ 0.01 # Largest possible pred-prey size ratio
    surface_irradiance = 300 # Need real value, in W m^-2
    pred_width = pred_length ./ 4
    rmax = ind_length .* 30 # The maximum visual range. Currently, this is 1 body length
    # Light attenuation coefficient
    a_lat = 0.64 .- 0.016 .* salt # Aksnes et al. 2009; per meter
    # Beam Attenuation coefficient
    c_lat = 4.87 .* a_lat # Huse and Fiksen (2010); per meter
    # Compute irradiance based on architecture
    if model.arch == CPU()
        i_td = surface_irradiance .* exp.(-0.1 .* model.individuals.animals[sp].data.z[ind])
    else
        z_cpu = model.individuals.animals[sp].data.z[ind] # Assuming 'z' is already a Julia array
        z_gpu = CuArray(z_cpu) # Convert 'z' to a CuArray
        # Perform the exponential operation on the GPU
        i_td = surface_irradiance .* CUDA.exp.(-0.1 .* z_gpu) 
    end
    # Prey image area
    pred_image = 0.75 .* pred_length .* pred_width
    # Visual eye sensitivity
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    # Define f(x) and its derivative fp(x)
    f(x) = x.^2 .* exp.(c_lat .* x) .- (pred_contrast .* pred_image .* eye_sensitivity .* (i_td ./ (eye_saturation .+ i_td)))
    fp(x) = 2 .* x .* exp.(c_lat .* x) .+ x.^2 .* c_lat .* exp.(c_lat .* x)
    # Call the Newton-Raphson method
    x = newton_raphson(f, fp)
    # Visual range estimates may be much more complicated when considering bioluminescent organisms. Could incorporate this if we assigned each species a "luminous type"
    ## https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3886326/
    return x
end

function visual_range_preys(model,sp,ind)
    ind_length = model.individuals.animals[sp].data.length[ind]/1000 # Length in meters
    prey_length = ind_length * 0.05 #Largest possible pred-prey size ratio (mm)
    salt = 30 #Needs to be in PSU
    surface_irradiance = 300 #Need real value. in W m-2
    prey_width = prey_length/4
    rmax = ind_length * 30 #Currently 30 times the body length of the animal
    prey_contrast = 0.3 #Utne-Plam (1999)   
    eye_saturation = 4 * 10^-4
    #Light attentuation coefficient
    a_lat = 0.64 - 0.016 * salt #Aksnes et al. 2009; per meter
    #Beam Attentuation coefficient
    c_lat = 4.87*a_lat #Huse and Fiksen (2010); per meter
    #Ambient irradiance at foraging depth
    i_td = surface_irradiance*exp(-0.1 * model.individuals.animals[sp].data.z[ind]) #Currently only reflects surface; 
    #Prey image area 
    prey_image = 0.75*prey_length*prey_width
    #Visual eye sensitivity
    eye_sensitivity = (rmax^2)/(prey_image*prey_contrast)
    #Equations for Visual Field
    f(x) =  x^2 * exp(c_lat * x) - (prey_contrast * prey_image * eye_sensitivity * (i_td/(eye_saturation + i_td)))
    fp(x) =  2 * x * exp(c_lat * x) + x^2 * c_lat * exp(c_lat * x)
    x = newton_raphson(f,fp)
    if x < 0.05
        x = sqrt(prey_contrast * prey_image * eye_sensitivity * (i_td / (eye_saturation + i_td)))
    end
    return x
end

function visual_pool(model,pool,depth)
    ind_length = mean(model.pools.pool[pool].characters.Min_Size[2][pool]:model.pools.pool[pool].characters.Max_Size[2][pool])/1000
    prey_length = ind_length * 0.05 #Largest possible pred-prey size ratio (mm)
    salt = 35 #Needs to be in PSU
    surface_irradiance = 300 #Need real value. in W m-2
    prey_width = prey_length/4
    rmax = ind_length * 30 # The maximum visual range. Currently this is 30 x body length
    prey_contrast = 0.3 #Utne-Plam (1999)   
    eye_saturation = 4 * 10^-4
    #Light attentuation coefficient
    a_lat = 0.64 - 0.016 * salt #Aksnes et al. 2009; per meter
    #Beam Attentuation coefficient
    c_lat = 4.87*a_lat #Huse and Fiksen (2010); per meter
    #Ambient irradiance at foraging depth
    i_td = surface_irradiance*exp(-0.1 * depth) #Currently only reflects surface; 
    #Prey image area 
    prey_image = 0.75*prey_length*prey_width
    #Visual eye sensitivity
    eye_sensitivity = (rmax^2)/(prey_image*prey_contrast)
    #Equations for Visual Field
    f(x) =  x^2 * exp(c_lat * x) - (prey_contrast * prey_image * eye_sensitivity * (i_td/(eye_saturation + i_td)))
    fp(x) =  2 * x * exp(c_lat * x) + x^2 * c_lat * exp(c_lat * x)
    x = newton_raphson(f,fp)
    if x < 0.05
        x = sqrt(prey_contrast * prey_image * eye_sensitivity * (i_td / (eye_saturation + i_td)))
    end
    range = (1/2) * (4/3) * pi * x^3
    return range
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