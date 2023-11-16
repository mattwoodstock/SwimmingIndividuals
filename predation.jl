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

function deg2rad(deg)
    return deg * π / 180
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
    min_prey_limit = 0.05 #Animals cannot eat anything less than 5% of their body size
    max_prey_limit = 0.5 #Animals cannot eat anything greater than 50% of their body size

    #Swim speed velocity
    #https://aslopubs.onlinelibrary.wiley.com/doi/pdf/10.1002/lno.12230

    #Should this be restricted, or should it be assumed the fishes are constantly swimming?
    swim_velo = pred_array.p.Swim_velo[2][pred_spec] * pred_array.data.length[pred] /100 * 60 * dt

    prey_list = DataFrame(Sp = Int[], Ind = Int[], x = Float64[], y = Float64[], z = Float64[], Weight = Float64[], Distance = Float64[])

    #Create dataframe for potential preys.
    prey_count = 1
    for i in 1:model.n_species #Cycle through each potential prey species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)

        for j in 1:model.ninds[i] #Cycle through each potential prey
            if (i != pred_spec) | (j != pred) |(spec_array1.data.x[j] != -1) #Animal cannot eat itself, or already dead prey

                prey_length = spec_array1.data.length[j] #Potential prey length
                prey_distance = d_matrix[pred,prey_count] #Distance to prey item

                if (prey_distance <= swim_velo) & (prey_length >= pred_array.data.length[pred] * min_prey_limit) & (prey_length <= pred_array.data.length[pred] * max_prey_limit)

                    new_row = Dict("Sp" => i, "Ind" => j, "x" => spec_array1.data.x[j], "y" => spec_array1.data.y[j], "z" => spec_array1.data.z[j],"Weight" => spec_array1.data.weight[j], "Distance" => prey_distance)
                    #Add individual to prey list
                    push!(prey_list,new_row)
                end
            end
            prey_count += 1
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
    #https://cdnsciencepub.com/doi/pdf/10.1139/f74-186
    handling_time_0 = 2.0

    handling_time = (1.19 - 1.24 * pred_df.data.gut_fullness[pred_ind] + 3.6 * pred_df.data.gut_fullness[pred_ind]^2 / handling_time_0) * handling_time_0

    #seconds of Handling time from Langbehn et al. 2019. Essentially a cool-off period after feeding.
    #Identify x,y,z of prey

    pred_df.data.x[pred_ind] = prey_df.x[1]
    pred_df.data.y[pred_ind] = prey_df.y[1]
    pred_df.data.z[pred_ind] = prey_df.z[1]

    #Calculate time to swim to prey
    swim_velo = pred_df.p.Swim_velo[2][pred_spec] * pred_df.data.length[pred_ind] /100

    time_to_prey = prey_df.Distance[1]/swim_velo

    t = t - ((handling_time + time_to_prey) / 60)

    #Add foraging time to activity time
    pred_df.data.active_time[pred_ind] = pred_df.data.active_time[pred_ind] + time_to_prey

    return t
end

function fill_gut!(pred_df,pred_ind,prey_df)

    prop_filled = prey_df.Weight[1]/pred_df.weight[pred_ind]

    pred_df.gut_fullness[pred_ind] = pred_df.gut_fullness[pred_ind] + prop_filled


    if pred_df.gut_fullness[pred_ind] > 1
        pred_df.gut_fullness[pred_ind] = 1
    end

    return nothing
end

function eat!(model::MarineModel,d_matrix,dt)

    for i in 1:model.n_species #Cycle through each predator species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)

        for j in 1:model.ninds[i] #Cycle through each individual predator
            ddt = dt #Subset of time
            if spec_array1.data.x[j] != -1 #Skip over dead animals
                ## Find potential preys for predator
                prey_list = available_prey(model,d_matrix,j,i,spec_array1,dt)

                while ddt > 0 ## Can only eat if there is time left
                    if size(prey_list,1) > 0 # There are preys within range. Need to choose one and "remove" it.
                        chosen_prey = prey_choice(prey_list)
                        remove_animal!(model,chosen_prey)
                        ddt = move_predator!(spec_array1,i,j,chosen_prey,ddt)
                        fill_gut!(spec_array1.data,j,chosen_prey)
                    else
                        chosen_prey = nothing
                        ddt = 0 ## No preys within range, therefore we do not need this.
                    end
                end
                throw(ErrorException("Stop Here."))
            end
        end
    end
end

