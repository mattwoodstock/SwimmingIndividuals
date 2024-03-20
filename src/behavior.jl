function behavior(model, sp, ind, outputs)
    behave_type = model.individuals.animals[sp].p.Type[2][sp]  # A variable that determines the behavioral type of an animal
    
    if behave_type == "dvm_strong"
        dvm_action(model, sp, ind)
        if model.individuals.animals[sp].data.mig_status[ind] in (0, -1)  # Not currently migrating
            decision(model, sp, ind, outputs)
        end

    elseif behave_type == "dvm_weak"
        max_fullness = 0.1 * model.individuals.animals[sp].data.weight[ind]
        feed_trigger = model.individuals.animals[sp].data.gut_fullness[ind] / max_fullness
        dist = logistic(feed_trigger, 5, 0.5)  # Resample from negative sigmoidal relationship
        
        if model.t >= 18 * 60 && model.individuals.animals[sp].data.mig_status[ind] == -1 && rand() >= dist
            decision(model, sp, ind, outputs)  # Animal does not migrate when it has the chance. Behaves as normal
        else
            dvm_action(model, sp, ind)  # Animal either migrates or continues what it should do
        end
        
        if model.individuals.animals[sp].data.mig_status[ind] in (0, -1)  # Not currently migrating
            decision(model, sp, ind, outputs)
        end
    elseif behave_type == "surface_diver"
        surface_dive(model, sp, ind)

        if model.individuals.animals[sp].data.mig_status[ind] in (0, -1)  # Animal is not currently ascending or descending
            decision(model, sp, ind, outputs)
        end
    elseif behave_type == "pelagic_diver"
        pelagic_dive(model, sp, ind)        
        if model.individuals.animals[sp].data.mig_status[ind] in (0, -1)  # Animal is not currently ascending or descending
            decision(model, sp, ind, outputs)
        end
    elseif behave_type == "nonmig"
        decision(model, sp, ind, outputs)
    end
    return nothing
end


function predator_density(model, sp, ind)
    # Precompute constant values
    min_pred_limit = 0.01
    max_pred_limit = 0.1
    # Gather distances
    detection = visual_range_preds(model, sp, ind)
    pred_list = calculate_distances_pred(model,sp,ind,min_pred_limit,max_pred_limit,detection)
    return pred_list
end

function decision(model, sp, ind, outputs)
    # Precompute constant values
    max_fullness = 0.1 *model.individuals.animals[sp].data.weight[ind] 
    swim_speed = model.individuals.animals[sp].p.Swim_velo[2][sp] * (model.individuals.animals[sp].data.length[ind] / 1000) * 60 * model.individuals.animals[sp].p.t_resolution[2][sp]
    
    feed_trigger = model.individuals.animals[sp].data.gut_fullness[ind] / max_fullness
    val1 = rand()

    # Individual avoids predators if predators exist
    pred_dens = predator_density(model, sp, ind)  # #/m2

    if feed_trigger < val1 && model.individuals.animals[sp].data.feeding[ind] == 1
        eat!(model, sp, ind, outputs)
    elseif nrow(pred_dens) > 0
        prey_loc = [model.individuals.animals[sp].data.x[ind], model.individuals.animals[sp].data.y[ind], model.individuals.animals[sp].data.z[ind]]
        predator_avoidance(pred_dens, prey_loc, swim_speed)  # Optimize movement away from all perceivable predators
    else
        # Random movement
        original_pos = [model.individuals.animals[sp].data.x[ind], model.individuals.animals[sp].data.y[ind], model.individuals.animals[sp].data.z[ind]]

        steady_speed = swim_speed / 2 #Random movement speed at 50% of the burst speed.
        
        new_pos = random_movement(original_pos, steady_speed)

        model.individuals.animals[sp].data.x[ind] = new_pos[1]
        model.individuals.animals[sp].data.y[ind] = new_pos[2]

        #Animal does not randomly change depths since this is such an integral component of community structure.
    end
end


function visual_range_preds(model,sp,ind)
    ind_length = model.individuals.animals[sp].data.length[ind]/1000 # Length in meters

    pred_length = ind_length / 0.01 #Largest possible pred-prey size ratio
    salt = 30 #Needs to be in PSU
    surface_irradiance = 300 #Need real value. in W m-2
    pred_width = pred_length/4
    rmax = ind_length * 30 # The maximum visual range. Currently this is 1 body length
    pred_contrast = 0.3 #Utne-Plam (1999)   
    eye_saturation = 4 * 10^-4
    #Light attentuation coefficient
    a_lat = 0.64 - 0.016 * salt #Aksnes et al. 2009; per meter
    #Beam Attentuation coefficient
    c_lat = 4.87*a_lat #Huse and Fiksen (2010); per meter
    #Ambient irradiance at foraging depth
    i_td = surface_irradiance*exp(-0.1 * model.individuals.animals[sp].data.z[ind]) #Currently only reflects surface; 
    #Prey image area 
    pred_image = 0.75*pred_length*pred_width
    #Visual eye sensitivity
    eye_sensitivity = (rmax^2)/(pred_image*pred_contrast)
    #Equations for Visual Field
    f(x) =  x^2 * exp(c_lat * x) - (pred_contrast * pred_image * eye_sensitivity * (i_td/(eye_saturation + i_td)))
    fp(x) =  2 * x * exp(c_lat * x) + x^2 * c_lat * exp(c_lat * x)
    x = newton_raphson(f,fp)

    if x < 0.05
        x = sqrt(pred_contrast * pred_image * eye_sensitivity * (i_td / (eye_saturation + i_td)))
    end
    #Visual range estimates may be much more complicated when considering bioluminescent organisms. Could incorporate this if we assigned each species a "luminous type"
    ##https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3886326/
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

# Function for the cost function used to triangulate the best distance
function cost_function_prey(prey_location, preds)
    total_distance = sum(norm(prey_location .- predator) for predator in preds)
    return -total_distance
end