function calculate_distances_prey(model::MarineModel, sp::Int64, inds::SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, min_prey::Float64, max_prey::Float64, detection::Vector{Float64})
    sp_data = model.individuals.animals[sp].data
    sp_length_inds = sp_data.length[inds]
    sp_x_inds = sp_data.x[inds]
    sp_y_inds = sp_data.y[inds]
    sp_z_inds = sp_data.z[inds]

    # This will store the collected PreyInfo for all individuals
    prey_infos_all = PreyInfo[]  # Initialize a single vector to store all prey information

    # Process each individual in `inds`
    for (j_index, j_value) in enumerate(inds)
        min_prey_size = sp_length_inds[j_index] * min_prey
        max_prey_size = sp_length_inds[j_index] * max_prey

        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            species_data = animal.data
            size_range = (species_data.length .>= min_prey_size) .& (species_data.length .<= max_prey_size)
            index1 = findall(size_range)
            prey_infos_all = vcat(prey_infos_all, add_prey(1, sp_data,species_data, j_index, index1, 1, species_index,detection))
        end

        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (pool_data.length .>= min_prey_size) .& (pool_data.length .<= max_prey_size)
            index1 = findall(size_range)
            prey_infos_all = vcat(prey_infos_all, add_prey(2, sp_data,pool_data, j_index, index1, pool_data.abundance, pool_index,detection))
        end
    end
    # Return the collected prey information as AllPreys
    return prey_infos_all
end

function calculate_distances_patch_prey(model::MarineModel, sp::Int64, inds::Vector{Int64}, min_prey::Float64, max_prey::Float64, detection::Vector{Float64})
    sp_data = model.pools.pool[sp].data
    sp_length_inds = sp_data.length[inds]
    sp_x_inds = sp_data.x[inds]
    sp_y_inds = sp_data.y[inds]
    sp_z_inds = sp_data.z[inds]

    # This will store the collected PreyInfo for all individuals
    prey_infos_all = PreyInfo[]  # Initialize a single vector to store all prey information

    # Process each individual in `inds`
    for (j_index, j_value) in enumerate(inds)
        min_prey_size = sp_length_inds[j_index] * min_prey
        max_prey_size = sp_length_inds[j_index] * max_prey

        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            species_data = animal.data
            size_range = (species_data.length .>= min_prey_size) .& (species_data.length .<= max_prey_size)
            index1 = findall(size_range)
            prey_infos_all = vcat(prey_infos_all, add_prey(1, sp_data,species_data, j_index, index1, 1, species_index,detection))
        end

        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (pool_data.length .>= min_prey_size) .& (pool_data.length .<= max_prey_size)
            index1 = findall(size_range)
            prey_infos_all = vcat(prey_infos_all, add_prey(2, sp_data,pool_data, j_index, index1, pool_data.abundance, pool_index,detection))
        end
    end
    # Return the collected prey information as AllPreys
    return prey_infos_all
end

function calculate_distances_pred(model::MarineModel, sp::Int64, inds::SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, min_pred::Float64, max_pred::Float64, detection::Vector{Float64})
    sp_data = model.individuals.animals[sp].data
    sp_length_inds::Vector{Float64} = sp_data.length[inds]
    sp_x_inds::Vector{Float64} = sp_data.x[inds]
    sp_y_inds::Vector{Float64} = sp_data.y[inds]
    sp_z_inds::Vector{Float64} = sp_data.z[inds]

    # This will store the collected PreyInfo for all individuals
    pred_infos_all = PredatorInfo[]  # Initialize a single vector to store all prey information

    # Process each individual in `inds`
    for (j_index, j_value) in enumerate(inds)
        min_pred_size = sp_length_inds[j_index] / max_pred
        max_pred_size = sp_length_inds[j_index] / min_pred

        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            species_data = animal.data
            size_range = (species_data.length .>= min_pred_size) .& (species_data.length .<= max_pred_size)
            index1 = findall(size_range)
            pred_infos_all = vcat(pred_infos_all, add_prey(1, sp_data,species_data, j_index, index1, 1, species_index,detection))
        end

        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (pool_data.length .>= min_pred_size) .& (pool_data.length .<= max_pred_size)
            index1 = findall(size_range)
            pred_infos_all = vcat(pred_infos_all, add_prey(2, sp_data,pool_data, j_index, index1, pool_data.abundance, pool_index,detection))
        end
    end
    # Return the collected prey information as AllPreys
    return pred_infos_all
end

function detect_prey(model::MarineModel,sp,ind)
    #Prey limitation. Need to make species-specific
    min_prey_limit = model.individuals.animals[sp].p.Min_Prey[2][sp] #Animals cannot eat anything less than 1% of their body length
    max_prey_limit = model.individuals.animals[sp].p.Max_Prey[2][sp] #Animals cannot eat anything greater than 5% of their body length
    #Prey Detection Distances
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/geb.13782

    detection = model.individuals.animals[sp].data.vis_prey[ind]

    prey_list = calculate_distances_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)

    searched_volume = (1/2) .* (4/3) .* pi .* detection.^3 #Calculate the maximum searched sphere for the predator (i.e., maximum search volume). Assumed animal can successfuly scan 50% of area
    return prey_list, searched_volume
end

function move_predator(model, sp, inds, index, prey_df,prey_ind)
    swim_velo = model.individuals.animals[sp].p.Swim_velo[2][sp] * model.individuals.animals[sp].data.length[inds[index]] / 1000

    if isa(prey_df,PreyInfo)
        # Handling time and swimming time
        time_to_prey = prey_df.Distance / swim_velo
        # Update predator
        model.individuals.animals[sp].data.x[inds[index]] = prey_df.x
        model.individuals.animals[sp].data.y[inds[index]] = prey_df.y
        model.individuals.animals[sp].data.z[inds[index]] = prey_df.z
    else
        # Handling time and swimming time
        time_to_prey = prey_df[prey_ind].Distance / swim_velo
        # Update predator
        model.individuals.animals[sp].data.x[inds[index]] = prey_df[prey_ind].x
        model.individuals.animals[sp].data.y[inds[index]] = prey_df[prey_ind].y
        model.individuals.animals[sp].data.z[inds[index]] = prey_df[prey_ind].z
    end

    return time_to_prey
end

function move_patch(model, sp, inds, index, prey_df,prey_ind)
    swim_velo = model.pools.pool[sp].characters.Swim_Velo[2][sp] * model.pools.pool[sp].data.length[inds[index]] / 1000 #1 body lengths per second

    # Handling time and swimming time
    time_to_prey = prey_df.Distance / swim_velo

    # Update predator
    model.pools.pool[sp].data.x[inds[index]] = prey_df.x
    model.pools.pool[sp].data.y[inds[index]] = prey_df.y
    model.pools.pool[sp].data.z[inds[index]] = prey_df.z
    return time_to_prey
end

function eat(model::MarineModel, sp, ind,to_eat, prey_list, outputs)
    n_ind = length(model.individuals.animals[sp].data.length[ind])
    ddt = fill(model.individuals.animals[sp].p.t_resolution[2][sp] * 60.0, n_ind)  # Seconds
    animal = model.individuals.animals[sp]
    animal_data = animal.data
    length_ind = animal_data.length[ind]
    gut_fullness_ind = animal_data.gut_fullness[ind]
    max_stomach = animal_data.biomass[ind] * 0.2
    handling_time = model.individuals.animals[sp].p.Handling_Time[2][sp]

    max_dist = model.individuals.animals[sp].p.Swim_velo[2][sp] * (length_ind / 1000) .* ddt

    # Loop through each prey list
    Threads.@threads for ind_index in 1:n_ind
        prey_list_item = filter(p -> p.Predator == to_eat[ind_index], prey_list)

        # Check if prey_list_item is a single PreyInfo or a collection

        if !isa(prey_list_item, PreyInfo) #Is a collection
            if isempty(prey_list_item) #Collection is empty
                continue  # Skip if the collection is empty
            end
            @inbounds sorted_prey = sort!(prey_list_item, by = x -> x.Distance)
        else #Is a single item
            sorted_prey = [prey_list_item]
        end

        if model.individuals.animals[sp].data.ac[ind[ind_index]] == 0 #Animal was consumed by another animal during this timestep and should be skipped.
            continue
        end


        # Continue eating as long as there's time left and the gut is not full
        total_time = 0.0
        prey_index = 1
        while total_time < ddt[ind_index] && gut_fullness_ind[ind_index] < max_stomach[ind_index] && prey_index <= length(sorted_prey)

            prey_info = sorted_prey[prey_index]
            if prey_info.Type == 1
                if model.individuals.animals[prey_info.Sp].data.ac[prey_info.Ind] == 0.0
                    prey_index += 1
                    continue
                end
                if model.individuals.animals[sp].data.ac[ind[ind_index]] == 0.0 #Predator has died somewhere in this process
                    break
                end
            else
                if model.pools.pool[prey_info.Sp].data.biomass[prey_info.Ind] <= 0
                    prey_index += 1
                    continue
                end
            end

            # If closest prey is too far, break the loop
            if prey_info.Distance > max_dist[ind_index]
                break
            end

            # Move towards the closest prey
            move_time = move_predator(model, sp, ind, ind_index, prey_list_item, prey_index)  # Update this call if necessary
            total_time += move_time
            model.individuals.animals[sp].data.active[ind[ind_index]] += (move_time/60)

            # Check if we have enough time left
            if total_time > ddt[ind_index]
                model.individuals.animals[sp].data.active[ind[ind_index]] = (ddt[ind_index]/60)
                break
            end

            # Handle predation based on prey type
            if prey_info.Type == 1
                if prey_info.Biomass > (max_stomach[ind_index] - gut_fullness_ind[ind_index])
                    continue
                end
                # Prey is consumable (e.g., type 1)
                ration = prey_info.Biomass
                model.individuals.animals[sp].data.ration[ind[ind_index]] += ration * model.individuals.animals[sp].p.Energy_density[2][sp] 
                outputs.consumption[sp,prey_info.Sp,Int(animal_data.pool_x[ind_index]),Int(animal_data.pool_y[ind_index]),max(1,Int(animal_data.pool_z[ind_index]))] += ration
                predation_mortality(model, prey_info, outputs)
                total_time += handling_time
            else
                # Prey is in a pool (e.g., type 2)
                prey_biomass = model.pools.pool[prey_info.Sp].data.biomass[prey_info.Ind]
                max_cons = (ddt[ind_index]-total_time) / handling_time

                min_prey_size = model.individuals.animals[sp].data.length[ind[ind_index]] * model.individuals.animals[sp].p.Min_Prey[2][sp]
                max_prey_size = model.individuals.animals[sp].data.length[ind[ind_index]] * model.individuals.animals[sp].p.Max_Prey[2][sp]

                ind_size = (max(model.pools.pool[prey_info.Sp].characters.Min_Size[2][prey_info.Sp],min_prey_size) + min(model.pools.pool[prey_info.Sp].characters.Max_Size[2][prey_info.Sp],max_prey_size))/2

                ind_biomass = model.pools.pool[prey_info.Sp].characters.LWR_a[2][prey_info.Sp] * (ind_size/10) ^ model.pools.pool[prey_info.Sp].characters.LWR_b[2][prey_info.Sp]

                ration = min(prey_biomass,(ind_biomass * max_cons),(max_stomach[ind_index] - gut_fullness_ind[ind_index])) 

                outputs.consumption[sp,(model.n_species+prey_info.Sp),Int(animal_data.pool_x[ind_index]),Int(animal_data.pool_y[ind_index]),max(1,Int(animal_data.pool_z[ind_index]))] += ration * model.pools.pool[prey_info.Sp].characters.Energy_density[2][prey_info.Sp]

                total_time += (ration/ind_biomass) * handling_time

                model.individuals.animals[sp].data.ration[ind[ind_index]] += ration * model.pools.pool[prey_info.Sp].characters.Energy_density[2][prey_info.Sp]

                reduce_pool(model, prey_info.Sp, prey_info.Ind, ration)
                # Continue to the next prey if the pool is depleted
            end

            # Update gut fullness
            model.individuals.animals[sp].data.gut_fullness[ind[ind_index]] += ration
            gut_fullness_ind[ind_index] += ration

            # If the stomach is full, break out of the loop
            if gut_fullness_ind[ind_index] >= max_stomach[ind_index]
                break
            end
            # Update the remaining time
            ddt[ind_index] -= total_time
            prey_index += 1
        end
    end
    return ddt
end

function patches_eat(model::MarineModel, sp, ind, prey_list, outputs)
    n_ind = length(ind)
    ddt = fill(model.dt * 60.0, n_ind)  # Seconds
    animal = model.pools.pool[sp]
    animal_data = animal.data
    length_ind = animal_data.length[ind]
    ration_ts = animal_data.biomass[ind] .* animal.characters.Daily_Ration[2][sp] ./ 1440 .* model.dt  # Assumed 3% of bodyweight for all individuals per day.
    handling_time = animal.characters.Handling_Time[2][sp]
    max_dist = animal.characters.Swim_Velo[2][sp] * (length_ind / 1000) .* ddt #Meters that can be swam.

    Threads.@threads for ind_index in 1:n_ind
        prey_list_item = filter(p -> p.Predator == ind_index, prey_list)

        if isempty(prey_list_item)
            continue
        end
        # Sort prey by distance initially
        sorted_prey = sort!(prey_list_item, by = x -> x.Distance)
        
        total_time = 0.0
        ration = 0.0
        prey_index = 1

        while total_time < ddt[ind_index] && ration < ration_ts[ind_index] && prey_index <= length(sorted_prey)
            prey_info = sorted_prey[prey_index]

            if prey_info.Type == 1
                if model.individuals.animals[prey_info.Sp].data.ac[prey_info.Ind] == 0.0
                    prey_index += 1
                    continue
                end
            else
                if model.pools.pool[prey_info.Sp].data.biomass[prey_info.Ind] <= 0
                    prey_index += 1
                    continue
                end
            end
            # If the closest prey is too far, break the loop
            if prey_info.Distance > max_dist[ind_index]
                break
            end

            # Move towards the closest prey
            move_time = move_patch(model, sp, ind, ind_index, prey_info, prey_index)
            total_time += move_time

            # Check if we have enough time left
            if total_time > ddt[ind_index]
                break
            end
            if prey_info.Type == 1
                if prey_info.Biomass > (ration_ts[ind_index] - ration)
                    prey_index += 1
                    continue
                end
                # Prey is consumable (e.g., type 1)
                ration += prey_info.Biomass
                predation_mortality(model, prey_info, outputs)
                outputs.consumption[(model.n_species+sp),prey_info.Sp,Int(animal_data.pool_x[ind_index]),Int(animal_data.pool_y[ind_index]),max(1,Int(animal_data.pool_z[ind_index]))] += ration

                total_time += handling_time
            else
                # Prey is in a pool (e.g., type 2)
                prey_biomass = model.pools.pool[prey_info.Sp].data.biomass[prey_info.Ind]
                prey_inds = model.pools.pool[prey_info.Sp].data.abundance[prey_info.Ind]
                n_inds = prey_biomass / prey_inds
                max_cons = Int(floor((ddt[ind_index] - total_time) / handling_time))
                ind_biomass = model.pools.pool[prey_info.Sp].characters.LWR_a[2][prey_info.Sp] * (prey_info.Length/10) ^ model.pools.pool[prey_info.Sp].characters.LWR_b[2][prey_info.Sp]

                if max_cons > n_inds
                    cons = min(prey_biomass, (ration_ts[ind_index] - ration))
                    ration += cons
                else
                    cons = min((ind_biomass * max_cons), (ration_ts[ind_index] - ration))
                    ration += cons
                end
                outputs.consumption[(model.n_species+sp),(model.n_species + prey_info.Sp),Int(animal_data.pool_x[ind_index]),Int(animal_data.pool_y[ind_index]),max(1,Int(animal_data.pool_z[ind_index]))] += cons

                total_time += (ration / ind_biomass) * handling_time                    

                reduce_pool(model, prey_info.Sp, prey_info.Ind, cons)
            end

            # Update the remaining time
            ddt[ind_index] -= total_time
            prey_index += 1
        end
    end
    return ddt
end

function pool_predation(model::MarineModel, pool::Int64, inds, outputs::MarineOutputs)
    biomass_values = model.pools.pool[pool].data.biomass[inds]::Vector{Float64}  # Enforce Vector{Float64} type
    ind = findall(x -> x > 0, biomass_values)
    
    if !isempty(ind)
        prey = patch_preys(model, pool, inds[ind])  # Generate prey based on selected indices
        patches_eat(model, pool, inds, prey, outputs)
        return prey
    end

    return Vector{PreyInfo}()
end



