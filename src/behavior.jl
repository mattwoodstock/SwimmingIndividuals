function behavior(model::MarineModel, sp::Int, ind, outputs::MarineOutputs)
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
            decision(model, sp, ind,outputs)  # Animal does not migrate when it has the chance. Behaves as normal
        else
            dvm_action(model, sp, ind)  # Animal either migrates or continues what it should do
            decision(model, sp, ind,outputs)
        end
    elseif behave_type == "pelagic_diver"
        dive_action(model, sp, ind)  # Function of energy density and dive characteristics to start dive
        decision(model, sp, ind,outputs)
    elseif behave_type == "non_mig"
        decision(model, sp, ind,outputs)
    end
    return nothing
end

function predators(model::MarineModel, sp::Int, ind)
    # Precompute constant values
    min_pred_limit = model.individuals.animals[sp].p.Min_Prey[2][sp]
    max_pred_limit = model.individuals.animals[sp].p.Max_Prey[2][sp]
    # Gather distances
    detection = model.individuals.animals[sp].data.vis_pred[ind]
  
    calculate_distances_pred(model,sp,ind,min_pred_limit,max_pred_limit,detection)
end

function decision(model::MarineModel, sp::Int, ind::Vector{Int64},outputs)  
    sp_dat = model.individuals.animals[sp].data

    max_fullness = 0.2 * sp_dat.biomass_school[ind]     
    feed_trigger = sp_dat.gut_fullness[ind] ./ max_fullness
    val1 = rand(length(ind))

    #preds::Vector{PredatorInfo} = predators(model, sp, ind)  #Create list of predators
    time::Vector{Float64} = fill(model.dt * 60,length(sp_dat.alive))

    to_eat = findall(feed_trigger .<= val1)
    eating = ind[to_eat]

    if length(to_eat) > 0
        print("find prey | ")
        prey = calculate_distances_prey(model, sp, eating,time)  #Create list of preys for all individuals in the species
        print("eat | ")
        time = eat(model, sp,eating, prey,time, outputs)
    end
    print("move | ")
    movement_toward_habitat(model,time,ind,sp)
    #Clear these as they are no longer necessary and take up memory.
    prey = Vector{PreyInfo}()
    #preds = Vector{PredatorInfo}()
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
    r = min.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* prey_size_factor)))
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
    r = min.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* prey_size_factor)))
    return r
end