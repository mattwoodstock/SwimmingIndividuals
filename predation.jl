function detection_distance(prey_length,pred_df,pred_ind)

    prey_length = prey_length * 1000
    salt = 30 #Needs to be in PSU
    surface_irradiance = 300 #Need real value. in W m-2
    prey_width = prey_length/4
    rmax = pred_df.data.length[pred_ind]*1000 # The maximum visual range. Currently this is 1 body length
    prey_contrast = 0.3 #Utne-Plam (1999)   
    eye_saturation = 1
    
    #Light attentuation coefficient
    a_lat = 0.64 - 0.016 * salt #Aksnes et al. 2009; per meter

    #Beam Attentuation coefficient
    c_lat = 4.87*a_lat #Huse and Fiksen (2010); per meter

    #Ambient irradiance at foraging depth
    i_td = surface_irradiance*exp(-0.1 * pred_df.data.z[pred_ind]) #Currently only reflects surface; 

    #Prey image area 
    prey_image = 0.75*prey_length*prey_width

    #Visual eye sensitivity
    eye_sensitivity = (rmax^2)/(prey_image*prey_contrast)
    
    #Equations for Visual Field
    f(x) = x^2 * exp(c_lat * x) - prey_contrast * prey_image * eye_sensitivity * i_td / (eye_saturation + i_td)
    fp(x) = 2*x * exp(c_lat * x) + c_lat * x^2 * exp(c_lat * x)

    x = NewtonRaphson(f,fp,pred_df.data.length[pred_ind])

    #Visual range estimates may be much more complicated when considering bioluminescent organisms. Could incorporate this if we assigned each species a "luminous type"
    ##https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3886326/
    return x
end

function holling_2(prey_species,prey_length,pred_array,pred_spec,pred_ind)

    visual_range = detection_distance(prey_length,pred_array,pred_spec)
    capture_success = 0.7 #Likelihood a predator consumes individual targetted preys
    ## Would be burst swim speed. May need to convert this to another unit
    swim_speed = pred_array.p.Swim_velo[2][pred_spec]* pred_array.data.length[pred_ind] /100

    #Clearance rate of animal (cubic meters per second per predator)
    clearance = 0.5 * π * visual_range^2 * swim_speed


    for i in 1:nrow(prey_species) # Cycle through number of prey species
        #Enounter rate of predator-prey
        encounter[i] = (clearance * prey_density)/(1+handling_t*prey_density)

        #Consumption rate of predator-prey
        consumption[i] = capture_success * encounter * prey_weight * prey_energy
    end

    return consumption
end

function distance_matrix(lat1, lon1, depth1, lat2, lon2, depth2)
    R = 6371000  # Earth radius in meters
    
    #Uses the haversine formula to calculate x,y distance
    
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    
    a = sin(dlat / 2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2)^2
    c = 2 * atan(sqrt(a), sqrt(1 - a))
    
    x = R * c

    y = depth2 - depth1

    distance = sqrt((y^2)+ (x^2))
    return distance
end

function calculate_distances(model::MarineModel)
    num_animals = sum(model.ninds)
    
    distances = zeros(Float64, num_animals, num_animals)
    count1 = count2 = 0

    for i in 1:model.n_species
        #Find number of individuals in species i
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)
        for j in 1:model.ninds[i]
            count1 += 1
            for k in 1:model.n_species
                #Find number of individuals in species k
                name2 = Symbol("sp"*string(k))
                spec_array2 = getfield(model.individuals.animals, name2)
                for l in 1:model.ninds[k]
                    count2 += 1
                    if (j != l) | (i != k)

                        distances[count1, count2] = distance_matrix(spec_array1.data.y[j], spec_array1.data.x[j], spec_array1.data.z[j], spec_array2.data.y[l], spec_array1.data.x[l], spec_array2.data.z[l])

                    end
                end
            end
            count2 = 0
        end
    end

    return distances
end

function available_prey(model::MarineModel,d_matrix,pred,pred_spec,pred_array,dt)
    
    #Prey limitation. Need to make species-specific
    min_prey_limit = 0.01 #Animals cannot eat anything less than 5% of their body length
    max_prey_limit = 0.05 #Animals cannot eat anything greater than 20% of their body length

    #Prey Detection Distances
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/geb.13782

    #Should this be restricted, or should it be assumed the fishes are constantly swimming?

    swim_velo = pred_array.p.Swim_velo[2][pred_spec] * pred_array.data.length[pred] /100 * 60 * dt

    prey_list = DataFrame(ID = Int[], Sp = Int[], Ind = Int[], x = Float64[], y = Float64[], z = Float64[], Weight = Float64[], Distance = Float64[])

    #Create dataframe for potential preys.
    prey_count = 1
    for i in 1:model.n_species #Cycle through each potential prey species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)

        for j in 1:model.ninds[i] #Cycle through each potential prey
            if (i != pred_spec) | (j != pred) |(spec_array1.data.x[j] != -1) #Animal cannot eat itself, or already dead prey

                prey_length = spec_array1.data.length[j] #Potential prey length

                if (prey_length >= pred_array.data.length[pred] * min_prey_limit) & (prey_length <= pred_array.data.length[pred] * max_prey_limit)

                    prey_distance = d_matrix[pred,prey_count] #Distance to prey item

                    detection = detection_distance(prey_length,pred_array,pred)

                    if (prey_distance <= detection)

                        new_row = Dict("ID" => prey_count, "Sp" => i, "Ind" => j, "x" => spec_array1.data.x[j], "y" => spec_array1.data.y[j], "z" => spec_array1.data.z[j],"Weight" => spec_array1.data.weight[j], "Distance" => prey_distance)
                        #Add individual to prey list
                        push!(prey_list,new_row)
                        prey_count += 1
                    end
                end
            end
        end
    end

    return prey_list
end

function prey_choice(list)
    #selectivity = 1 Currently, all predators do not feed in a taxon-specific fashion.

    prey = list[list.Distance .== minimum(list.Distance),:]

    return prey
end

function move_predator!(pred_df,pred_spec,pred_ind,prey_df,t)

    #Handling time is a function of gut fullness with 2.0 seconds as the base. May want a better source.
    ##https://cdnsciencepub.com/doi/pdf/10.1139/f74-186
    ##seconds of Handling time from Langbehn et al. 2019. Essentially a cool-off period after feeding.

    handling_time_0 = 2.0

    handling_time = (1.19 - 1.24 * pred_df.data.gut_fullness[pred_ind] + 3.6 * pred_df.data.gut_fullness[pred_ind]^2 / handling_time_0) * handling_time_0

    #Identify x,y,z of prey

    pred_df.data.x[pred_ind] = prey_df.x[1]
    pred_df.data.y[pred_ind] = prey_df.y[1]
    #pred_df.data.z[pred_ind] = prey_df.z[1]

    #Calculate time to swim to prey
    swim_velo = pred_df.p.Swim_velo[2][pred_spec] * pred_df.data.length[pred_ind] /100

    time_to_prey = prey_df.Distance[1]/swim_velo

    t = t - ((handling_time + time_to_prey) / 60)

    #Add foraging time to activity time
    pred_df.data.active_time[pred_ind] = pred_df.data.active_time[pred_ind] + time_to_prey/60

    return t
end

function fill_gut!(pred_df,pred_ind,prey_df)

    prop_filled = prey_df.Weight[1]/pred_df.weight[pred_ind]

    pred_df.gut_fullness[pred_ind] = pred_df.gut_fullness[pred_ind] + prop_filled        


    if pred_df.gut_fullness[pred_ind] > 0.03*pred_df.weight[pred_ind]
        pred_df.gut_fullness[pred_ind] = 0.03*pred_df.weight[pred_ind]
    end
    return nothing
end

function prey_density(model)
    #Calculate % of size distribution is available to predation based on size relationships and densities.
end


function eat!(model::MarineModel,d_matrix,i,j,spec_array1,dt)
    ddt = dt #Subset of time

    if (model.dimension == 1) #Running the 1D model

        ##Calculate consumption from a type 2 functional response

        ### Need to calculate prey prey_densities by potential preys
        prey_species = prey_density(model,pooled) #Need to create function
        #prey length = mean length of available preys

        q = holling_2(prey_species,prey_length,spec_array1,i,j)

        return q
    else #Running the 3D model


        prey_list = available_prey(model,d_matrix,j,i,spec_array1,dt)
        
        while ddt > 0 ## Can only eat if there is time left

            if (nrow(prey_list) > 0) && (spec_array1.data.gut_fullness[j] < 1) # There are preys within range. Need to choose one and "remove" it.

                    chosen_prey = prey_choice(prey_list)
                    pred_success = rand()

                    if pred_success >= 0.3 #70% chance of predator success in a feeding event https://onlinelibrary.wiley.com/doi/pdf/10.1111/jfb.14451
                    remove_animal!(model,chosen_prey)
                    ddt = move_predator!(spec_array1,i,j,chosen_prey,ddt)
                    
                    fill_gut!(spec_array1.data,j,chosen_prey)
                    allocate_energy(spec_array1,i,j,chosen_prey)
                end
                    #Still remove animal from prey list as if it goes away
                    deleteat!(prey_list,findall(prey_list.Sp .== chosen_prey.Sp[1] .&& prey_list.Ind .== chosen_prey.Ind[1]))
            else
                chosen_prey = nothing
                ddt = 0 ## No preys within range, therefore we do not need this.
            end
        end
    end
end
