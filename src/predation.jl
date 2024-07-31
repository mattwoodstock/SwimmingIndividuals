function calculate_distances_prey(model::MarineModel, sp, inds, min_prey, max_prey, detection)
    sp_data = model.individuals.animals[sp].data
    sp_length_inds = sp_data.length[inds]
    sp_x_inds = sp_data.x[inds]
    sp_y_inds = sp_data.y[inds]
    sp_z_inds = sp_data.z[inds]

    ind_names = Symbol[]
    ind_data = []

    prey = PreyInfo[]  # Initialize with an empty vector

    function add_prey(prey_type, prey_data, ind, indices,abundances,sp)
        dx = sp_x_inds[ind] .- prey_data.x[indices]
        dy = sp_y_inds[ind] .- prey_data.y[indices]
        dz = sp_z_inds[ind] .- prey_data.z[indices]
        dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
        if prey_type == 2
            dist = max.(0, dist .- (3 .* prey_data.volume[indices] ./ (4 * π)).^(1 / 3))
        end
        within_detection = findall(dist .<= detection[1])
        for (individual,i) in enumerate(within_detection)
            if prey_type == 2
                push!(prey, PreyInfo(prey_type,sp,indices[i], prey_data.x[i], prey_data.y[i], prey_data.z[i],prey_data.biomass[i],prey_data.length[i],abundances[i],dist[individual]))
            else
                push!(prey, PreyInfo(prey_type,sp,indices[i], prey_data.x[i], prey_data.y[i], prey_data.z[i],prey_data.biomass[i],prey_data.length[i],abundances,dist[individual]))
            end
        end
    end

    # Process each individual in `inds` using broadcasting
    for (j,ind) in enumerate(inds)
        min_prey_size = sp_length_inds[j] * min_prey
        max_prey_size = sp_length_inds[j] * max_prey
        name = Symbol("Ind" * string(j))

        prey = PreyInfo[]  # Initialize with an empty vector
        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            if species_index != sp
                species_data = animal.data
                size_range = (min_prey_size .<= species_data.length) .& (species_data.length .<= max_prey_size)
                index1 = findall(size_range)
                add_prey(1, species_data, j, index1,1,species_index)
            end
        end
        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (min_prey_size .<= pool_data.length) .& (pool_data.length .<= max_prey_size)
            index1 = findall(size_range)
            add_prey(2, pool_data, j, index1,pool_data.abundance,pool_index)
        end
        push!(ind_names, name)
        push!(ind_data, prey)
    end
    preys = NamedTuple{Tuple(ind_names)}(ind_data)
    return AllPreys(preys)
end

function calculate_distances_patch_prey(model::MarineModel, sp, inds, min_prey, max_prey, detection)
    sp_data = model.pools.pool[sp].data
    sp_length_inds = sp_data.length[inds]
    sp_x_inds = sp_data.x[inds]
    sp_y_inds = sp_data.y[inds]
    sp_z_inds = sp_data.z[inds]

    ind_names = Symbol[]
    ind_data = []

    prey = PreyInfo[]  # Initialize with an empty vector

    function add_patch_prey(prey_type,sp_data, prey_data, ind, indices,abundances,sp)
        dx = sp_x_inds[ind] .- prey_data.x[indices]
        dy = sp_y_inds[ind] .- prey_data.y[indices]
        dz = sp_z_inds[ind] .- prey_data.z[indices]
        dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
        if prey_type == 1
            dist = max.(0, dist .- (3 .* sp_data.volume[ind] ./ (4 * π)).^(1 / 3))
        else 
            dist = max.(0, dist .- (3 .* prey_data.volume[indices] ./ (4 * π)).^(1 / 3) .- (3 .* sp_data.volume[ind] ./ (4 * π)))
        end
        within_detection = findall(dist .<= detection[1])
        for (individual,i) in enumerate(within_detection)
            if prey_type == 2
                push!(prey, PreyInfo(prey_type,sp,indices[i], prey_data.x[i], prey_data.y[i], prey_data.z[i],prey_data.biomass[i],prey_data.length[i],abundances[i],dist[individual]))
            else
                push!(prey, PreyInfo(prey_type,sp,indices[i], prey_data.x[i], prey_data.y[i], prey_data.z[i],prey_data.biomass[i],prey_data.length[i],abundances,dist[individual]))
            end
        end
    end

    # Process each individual in `inds` using broadcasting
    for (j,ind) in enumerate(inds)
        min_prey_size = sp_length_inds[j] * min_prey
        max_prey_size = sp_length_inds[j] * max_prey
        name = Symbol("Ind" * string(j))

        prey = PreyInfo[]  # Initialize with an empty vector
        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            if species_index != sp
                species_data = animal.data
                size_range = (min_prey_size .<= species_data.length) .& (species_data.length .<= max_prey_size)
                index1 = findall(size_range)
                add_patch_prey(1, sp_data, species_data, j, index1,1,species_index)
            end
        end
        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (min_prey_size .<= pool_data.length) .& (pool_data.length .<= max_prey_size)
            index1 = findall(size_range)
            add_patch_prey(2, sp_data, pool_data, j, index1,pool_data.abundance,pool_index)
        end
        push!(ind_names, name)
        push!(ind_data, prey)
    end
    preys = NamedTuple{Tuple(ind_names)}(ind_data)
    return AllPreys(preys)
end

function calculate_distances_pred(model::MarineModel, sp, inds, min_pred, max_pred, detection)
    sp_data = model.individuals.animals[sp].data
    sp_length_inds = sp_data.length[inds]
    sp_x_inds = sp_data.x[inds]
    sp_y_inds = sp_data.y[inds]
    sp_z_inds = sp_data.z[inds]

    ind_names = Symbol[]
    ind_data = []
    pred = PredatorInfo[]  # Initialize with an empty vector

    function add_pred(pred_type, pred_data, ind, indices)
        dx = sp_x_inds[ind] .- pred_data.x[indices]
        dy = sp_y_inds[ind] .- pred_data.y[indices]
        dz = sp_z_inds[ind] .- pred_data.z[indices]
        dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
        within_detection = dist .<= detection[1]
        if pred_type == 2
            dist = max.(0, dist .- (3 .* pred_data.volume[indices] ./ (4 * π)).^(1 / 3))
        end
        filtered_indices = findall(within_detection)
        for i in filtered_indices
            push!(pred, PredatorInfo(pred_data.x[indices[i]], pred_data.y[indices[i]], pred_data.z[indices[i]], dist[i]))
        end
    end

    # Process each individual in `inds` using broadcasting
    for ind in inds
        name = Symbol("Ind" * string(ind))

        pred = PredatorInfo[]  # Initialize with an empty vector
        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            if species_index != sp
                species_data = animal.data
                size_range = (sp_length_inds[ind] / max_pred .<= species_data.length .<= sp_length_inds[ind] / min_pred)
                index1 = findall(size_range)
                add_pred(1, species_data, ind, index1)
            end
        end

        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (sp_length_inds[ind] / max_pred .<= pool_data.length .<= sp_length_inds[ind] / min_pred)
            index1 = findall(size_range)
            add_pred(2, pool_data, ind, index1)
        end
        push!(ind_names, name)
        push!(ind_data, pred)
    end

    predators = NamedTuple{Tuple(ind_names)}(ind_data)
    return AllPreds(predators)
end

function detect_prey(model::MarineModel,sp,ind)
    #Prey limitation. Need to make species-specific
    min_prey_limit = 0.01 #Animals cannot eat anything less than 1% of their body length
    max_prey_limit = 0.1 #Animals cannot eat anything greater than 5% of their body length
    #Prey Detection Distances
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/geb.13782

    detection = model.individuals.animals[sp].data.vis_prey[ind]

    prey_list = calculate_distances_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)

    searched_volume = (1/2) .* (4/3) .* pi .* detection.^3 #Calculate the maximum searched sphere for the predator (i.e., maximum search volume). Assumed animal can successfuly scan 50% of area
    return prey_list, searched_volume
end

function move_predator(model, sp, inds, index, prey_df,prey_ind)
    swim_velo = model.individuals.animals[sp].p.Swim_velo[2][sp] * model.individuals.animals[sp].data.length[inds[index]] / 1000

    # Handling time and swimming time
    time_to_prey = prey_df[prey_ind].Distance / swim_velo

    # Update predator
    model.individuals.animals[sp].data.x[inds[index]] = prey_df[prey_ind].x
    model.individuals.animals[sp].data.y[inds[index]] = prey_df[prey_ind].y
    model.individuals.animals[sp].data.z[inds[index]] = prey_df[prey_ind].z
    return time_to_prey
end

function move_patch(model, sp, inds, index, prey_df,prey_ind)
    swim_velo = 1 * model.pools.pool[sp].data.length[inds[index]] / 1000 #1 body lengths per second

    # Handling time and swimming time
    time_to_prey = prey_df[prey_ind].Distance / swim_velo

    # Update predator
    model.pools.pool[sp].data.x[inds[index]] = prey_df[prey_ind].x
    model.pools.pool[sp].data.y[inds[index]] = prey_df[prey_ind].y
    model.pools.pool[sp].data.z[inds[index]] = prey_df[prey_ind].z
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
    handling_time = 15.0

    max_dist = 1.5 * (length_ind / 1000) .* ddt
    filtered_prey_list = [prey_list.preys[i] for i in to_eat]

    # Loop through each prey list
    Threads.@threads for ind_index in 1:size(filtered_prey_list,1)
        prey_list_item = filtered_prey_list[ind_index]

        if isempty(prey_list_item)
            continue
        end

        # Continue eating as long as there's time left and the gut is not full
        total_time = 0.0
        while total_time < ddt[ind_index] && gut_fullness_ind[ind_index] < max_stomach[ind_index]
            # Find the closest prey
            min_dist = 5e6
            closest_prey_index = 0
            for j in 1:size(prey_list_item, 1)
                if prey_list_item[j].Distance < min_dist
                    min_dist = prey_list_item[j].Distance
                    closest_prey_index = j
                end
            end

            # If closest prey is too far, break the loop
            if min_dist > max_dist[ind_index]
                break
            end

            # Move towards the closest prey
            move_time = move_predator(model, sp, ind, ind_index, prey_list_item, closest_prey_index)  # Update this call if necessary
            total_time += move_time

            # Check if we have enough time left
            if total_time > ddt[ind_index]
                break
            end

            # Handle predation based on prey type
            prey_info = prey_list_item[closest_prey_index]
            if prey_info.Type == 1
                # Prey is consumable (e.g., type 1)
                ration = prey_info.Biomass
                model.individuals.animals[sp].data.ration[ind[ind_index]] += ration
                predation_mortality(model, prey_info, outputs)
                prey_info = PreyInfo(prey_info.Type,prey_info.Sp,prey_info.Ind,prey_info.x,prey_info.y,prey_info.z,prey_info.Biomass,prey_info.Length,prey_info.Inds,5e6)
                total_time += handling_time
            else
                # Prey is in a pool (e.g., type 2)
                prey_biomass = model.pools.pool[prey_info.Sp].data.biomass[prey_info.Ind]
                prey_inds = model.pools.pool[prey_info.Sp].data.abundance[prey_info.Ind]
                n_inds = prey_biomass/prey_inds
                max_cons = Int(floor((ddt[ind_index]-total_time) / handling_time))
                ind_biomass = model.pools.pool[prey_info.Sp].characters.LWR_a[2][prey_info.Sp] * prey_info.Length ^ model.pools.pool[prey_info.Sp].characters.LWR_b[2][prey_info.Sp]

                if max_cons > n_inds
                    ration = min(prey_biomass, (max_stomach[ind_index] - gut_fullness_ind[ind_index]))
                else
                    ration = min((ind_biomass * max_cons),(max_stomach[ind_index] - gut_fullness_ind[ind_index]))
                end
                total_time += (ration/ind_biomass) * handling_time

                model.individuals.animals[sp].data.ration[ind[ind_index]] += ration
                reduce_pool(model, prey_info.Sp, prey_info.Ind, ration)
                # Continue to the next prey if the pool is depleted
                if model.pools.pool[prey_info.Sp].data.biomass[prey_info.Sp] <= 0
                    prey_info = PreyInfo(prey_info.Type,prey_info.Sp,prey_info.Ind,prey_info.x,prey_info.y,prey_info.z,prey_info.Biomass,prey_info.Length,prey_info.Inds,5e6)
                    continue
                end
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
    ration_ts = animal_data.biomass[ind] .* 0.03 ./ 1440 .* model.dt #Assumed 3% of bodyweight for all individuals per day.
    handling_time = 15.0

    max_dist = 1.5 * (length_ind / 1000) .* ddt

    # Loop through each prey list
    Threads.@threads for ind_index in 1:length(ind)
        prey_list_item = prey_list.preys[ind_index]

        if isempty(prey_list_item)
            continue
        end

        # Continue eating as long as there's time left and the gut is not full
        total_time = 0.0
        ration = 0.0
        while total_time < ddt[ind_index] && ration < ration_ts[ind_index]
            # Find the closest prey
            min_dist = 5e6
            closest_prey_index = 0
            for j in 1:size(prey_list_item, 1)
                if prey_list_item[j].Distance < min_dist
                    min_dist = prey_list_item[j].Distance
                    closest_prey_index = j
                end
            end

            # If closest prey is too far, break the loop
            if min_dist > max_dist[ind_index]
                break
            end

            # Move towards the closest prey
            move_time = move_patch(model, sp, ind, ind_index, prey_list_item, closest_prey_index)  # Update this call if necessary
            total_time += move_time

            # Check if we have enough time left
            if total_time > ddt[ind_index]
                break
            end

            # Handle predation based on prey type
            prey_info = prey_list_item[closest_prey_index]
            if prey_info.Type == 1
                # Prey is consumable (e.g., type 1)
                ration += prey_info.Biomass
                predation_mortality(model, prey_info, outputs)
                prey_info = PreyInfo(prey_info.Type,prey_info.Sp,prey_info.Ind,prey_info.x,prey_info.y,prey_info.z,prey_info.Biomass,prey_info.Length,prey_info.Inds,5e6)
                total_time += handling_time
            else
                # Prey is in a pool (e.g., type 2)
                prey_biomass = model.pools.pool[prey_info.Sp].data.biomass[prey_info.Ind]
                prey_inds = model.pools.pool[prey_info.Sp].data.abundance[prey_info.Ind]
                n_inds = prey_biomass/prey_inds
                max_cons = Int(floor((ddt[ind_index]-total_time) / handling_time))
                ind_biomass = model.pools.pool[prey_info.Sp].characters.LWR_a[2][prey_info.Sp] * prey_info.Length ^ model.pools.pool[prey_info.Sp].characters.LWR_b[2][prey_info.Sp]

                if max_cons > n_inds
                    ration += min(prey_biomass, (ration_ts[ind_index] - ration))
                    cons = min(prey_biomass, (ration_ts[ind_index] - ration))
                else
                    ration += min((ind_biomass * max_cons),(ration_ts[ind_index] - ration))
                    cons = min(prey_biomass, (ration_ts[ind_index] - ration))

                end
                total_time += (ration/ind_biomass) * handling_time

                reduce_pool(model, prey_info.Sp, prey_info.Ind, cons)
                # Continue to the next prey if the pool is depleted
                if model.pools.pool[prey_info.Sp].data.biomass[prey_info.Sp] <= 0
                    prey_info = PreyInfo(prey_info.Type,prey_info.Sp,prey_info.Ind,prey_info.x,prey_info.y,prey_info.z,prey_info.Biomass,prey_info.Length,prey_info.Inds,5e6)
                    continue
                end
            end
            # Update the remaining time
            ddt[ind_index] -= total_time
        end
    end
    return ddt
end

function pool_predation(model, pool,inds,outputs)
    ind = findall(x -> x > 0, model.pools.pool[pool].data.biomass[inds])
    ##Create Prey list
    prey = patch_preys(model,pool,ind)
    patches_eat(model,pool,inds,prey,outputs)
    prey.preys = NamedTuple()

end


