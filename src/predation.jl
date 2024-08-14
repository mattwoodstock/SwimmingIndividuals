function calculate_distances_prey(model::MarineModel, sp, inds, min_prey, max_prey, detection)
    sp_data = model.individuals.animals[sp].data
    sp_length_inds = sp_data.length[inds]
    sp_x_inds = sp_data.x[inds]
    sp_y_inds = sp_data.y[inds]
    sp_z_inds = sp_data.z[inds]

    ind_names = Symbol[]
    ind_data = Vector{PreyInfo}[]  # Preallocate the array of PreyInfo vectors

    function add_prey(prey_type, prey_data, ind, indices, abundances, sp)
        dx = sp_x_inds[ind] .- prey_data.x[indices]
        dy = sp_y_inds[ind] .- prey_data.y[indices]
        dz = sp_z_inds[ind] .- prey_data.z[indices]
        dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
        within_detection = findall(dist .<= detection[ind])

        prey_infos = PreyInfo[]  # Initialize with an empty vector
        for i in within_detection
            prey_info = PreyInfo(prey_type, sp, indices[i], prey_data.x[indices[i]], prey_data.y[indices[i]], prey_data.z[indices[i]], prey_data.biomass[indices[i]], prey_data.length[indices[i]], abundances, dist[i])
            push!(prey_infos, prey_info)
        end
        return prey_infos
    end

    function add_patch_prey(prey_type, prey_data, ind, indices, abundances, pool_index)
        dx = sp_x_inds[ind] .- prey_data.x[indices]
        dy = sp_y_inds[ind] .- prey_data.y[indices]
        dz = sp_z_inds[ind] .- prey_data.z[indices]
        dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
        if prey_type == 2
            dist = max.(0, dist .- (3 .* prey_data.volume[indices] ./ (4 * π)).^(1 / 3))
        end
        within_detection = findall(dist .<= detection[ind])
        prey_infos = PreyInfo[]  # Initialize with an empty vector
        for i in within_detection
            prey_info = PreyInfo(prey_type, pool_index, indices[i], prey_data.x[indices[i]], prey_data.y[indices[i]], prey_data.z[indices[i]], prey_data.biomass[indices[i]], prey_data.length[indices[i]], abundances[i], dist[i])
            push!(prey_infos, prey_info)
        end
        return prey_infos
    end

    # Process each individual in `inds` using broadcasting
    ind_data = map(enumerate(inds)) do (j_index, j_value)
        min_prey_size = sp_length_inds[j_index] * min_prey
        max_prey_size = sp_length_inds[j_index] * max_prey
        name = Symbol("Ind" * string(j_value))

        prey = PreyInfo[]  # Initialize with an empty vector
        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            species_data = animal.data
            size_range = (species_data.length .>= min_prey_size) .& (species_data.length .<= max_prey_size)
            index1 = findall(size_range)
            prey = vcat(prey, add_prey(1, species_data, j_index, index1, 1, species_index))
        end
        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (pool_data.length .>= min_prey_size) .& (pool_data.length .<= max_prey_size)
            index1 = findall(size_range)
            prey = vcat(prey, add_patch_prey(2, pool_data, j_index,index1, pool_data.abundance, pool_index))
        end
        push!(ind_names, name)
        prey
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
    ind_data = Vector{PreyInfo}[]  # Preallocate the array of PreyInfo vectors

    Threads.@threads for j in 1:length(inds)
        min_prey_size = sp_length_inds[j] * min_prey
        max_prey_size = sp_length_inds[j] * max_prey
        name = Symbol("Ind" * string(j))

        prey = PreyInfo[]  # Initialize with an empty vector

        function add_patches(prey_type, sp_data, prey_data, ind, indices, abundances, sp)
            dx = sp_x_inds[ind] .- prey_data.x[indices]
            dy = sp_y_inds[ind] .- prey_data.y[indices]
            dz = sp_z_inds[ind] .- prey_data.z[indices]
            dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            if prey_type == 1
               dist = max.(0, dist .- (3 .* sp_data.volume[ind] ./ (4 * π)).^(1 / 3))
            else
                dist = max.(0, dist .- (3 .* prey_data.volume[indices] ./ (4 * π)).^(1 / 3) .- (3 .* sp_data.volume[ind] ./ (4 * π)))
            end
            within_detection = findall(dist .<= detection[ind])
            for i in 1:length(within_detection)
                prey_info = prey_type == 2 ?
                PreyInfo(prey_type, sp, indices[i], prey_data.x[indices[i]], prey_data.y[indices[i]], prey_data.z[indices[i]], prey_data.biomass[indices[i]], prey_data.length[indices[i]], abundances[indices[i]], dist[within_detection[i]]) :
                PreyInfo(prey_type, sp, indices[i], prey_data.x[indices[i]], prey_data.y[indices[i]], prey_data.z[indices[i]], prey_data.biomass[indices[i]], prey_data.length[indices[i]], abundances, dist[within_detection[i]])
                push!(prey, prey_info)
            end
        end

        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            species_data = animal.data
            size_range = (min_prey_size .<= species_data.length) .& (species_data.length .<= max_prey_size)
            index1 = findall(size_range)
            if !isempty(index1)
                add_patches(1, sp_data, species_data, j, index1, 1, species_index)
            end
        end

        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (min_prey_size .<= pool_data.length) .& (pool_data.length .<= max_prey_size)
            index1 = findall(size_range)
            if !isempty(index1)
                add_patches(2, sp_data, pool_data, j, index1, pool_data.abundance, pool_index)
            end
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
    ind_data = Vector{PreyInfo}[]  # Preallocate the array of PreyInfo vectors

    function add_pred(pred_data, ind, indices,min_dist)
        dx = sp_x_inds[ind] .- pred_data.x[indices]
        dy = sp_y_inds[ind] .- pred_data.y[indices]
        dz = sp_z_inds[ind] .- pred_data.z[indices]
        dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
        within_detection = findall(dist .<= detection[ind])

        pred_infos = PredatorInfo[]  # Initialize with an empty vector
        for i in within_detection
            if dist[i] < min_dist
                min_dist = dist[i]
                pred_info = PredatorInfo(pred_data.x[indices[i]], pred_data.y[indices[i]], pred_data.z[indices[i]], dist[i])
                push!(pred_infos, pred_info)
            end
        end
        return pred_infos
    end

    # Process each individual in `inds` using broadcasting
    ind_data = map(enumerate(inds)) do (j_index, j_value)
        name = Symbol("Ind" * string(j_value))
        min_dist = 5e6
        pred = PredatorInfo[]  # Initialize with an empty vector
        # Process individual animals
        for (species_index, animal) in enumerate(model.individuals.animals)
            species_data = animal.data
            size_range = (sp_length_inds[j_index] / max_pred .<= species_data.length .<= sp_length_inds[j_index] / min_pred)
            index1 = findall(size_range)
            pred = vcat(pred, add_pred(species_data, j_index, index1,min_dist))
        end
        # Process pool animals
        for (pool_index, animal) in enumerate(model.pools.pool)
            pool_data = animal.data
            size_range = (sp_length_inds[j_index] / max_pred .<= pool_data.length .<= sp_length_inds[j_index] / min_pred)
            index1 = findall(size_range)
            pred = vcat(pred, add_pred(pool_data, j_index, index1,min_dist))
        end
        push!(ind_names, name)
        pred
    end
    preds = NamedTuple{Tuple(ind_names)}(ind_data)
    return AllPreds(preds)
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
    handling_time = 60.0

    max_dist = model.individuals.animals[sp].p.Swim_velo[2][sp] * (length_ind / 1000) .* ddt
    # Loop through each prey list
    Threads.@threads for ind_index in 1:n_ind

        prey_list_item = prey_list.preys[to_eat[ind_index]]

        if isempty(prey_list_item)
            continue
        end
        
        if model.individuals.animals[sp].data.ac[ind[ind_index]] == 0 #Animal was consumed in by another animal this time.
            continue
        end

        sorted_prey = sort(prey_list_item, by = x -> x.Biomass)

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
            model.individuals.animals[sp].data.ac[ind[ind_index]] += move_time

            # Check if we have enough time left
            if total_time > ddt[ind_index]
                model.individuals.animals[sp].data.ac[ind[ind_index]] = ddt[ind_index]
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

                min_prey_size = model.individuals.animals[sp].data.length[ind[ind_index]] * 0.01
                max_prey_size = model.individuals.animals[sp].data.length[ind[ind_index]] * 0.1

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
    n_ind = length(prey_list.preys)
    ddt = fill(model.dt * 60.0, n_ind)  # Seconds
    animal = model.pools.pool[sp]
    animal_data = animal.data
    length_ind = animal_data.length[ind]
    ration_ts = animal_data.biomass[ind] .* 0.03 ./ 1440 .* model.dt  # Assumed 3% of bodyweight for all individuals per day.
    handling_time = 60.0
    max_dist = 1 * (length_ind / 1000) .* ddt #Meters that can be swam.

    Threads.@threads for ind_index in 1:n_ind
        prey_list_item = prey_list.preys[ind_index]
        if isempty(prey_list_item)
            continue
        end
        # Sort prey by distance initially
        sorted_prey = sort(prey_list_item, by = x -> x.Biomass)
        
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
                    #prey_index += 1
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

function pool_predation(model, pool,inds,outputs)
    ind = findall(x -> x > 0, model.pools.pool[pool].data.biomass[inds])
    ##Create Prey list
    if length(ind) > 0
        prey = patch_preys(model,pool,inds[ind])
        patches_eat(model,pool,inds,prey,outputs)
        prey.preys = NamedTuple()
    end
end


