function calculate_distances_prey(model::MarineModel, sp, ind, min_prey, max_prey, detection)
    sp_data = model.individuals.animals[sp].data
    sp_length_ind = sp_data.length[ind]
    sp_x_ind = sp_data.x[ind]
    sp_y_ind = sp_data.y[ind]
    sp_z_ind = sp_data.z[ind]

    preys = DataFrame(Type = Int[], Sp = Int[], Ind = Int[], x = Float64[], y = Float64[], z = Float64[], Weight = Float64[], Distance = Float64[])

    # Function to calculate and add prey to the DataFrame
    function add_prey(prey_type, sp_index, prey_data, i)
        dx = sp_x_ind[1] - prey_data.x[i]
        dy = sp_y_ind[1] - prey_data.y[i]
        dz = sp_z_ind[1] - prey_data.z[i]
        dist = sqrt(dx^2 + dy^2 + dz^2)
        if dist <= detection[1]
            push!(preys, (Type = prey_type, Sp = sp_index, Ind = i, x = prey_data.x[i], y = prey_data.y[i], z = prey_data.z[i], Weight = prey_data.biomass[i], Distance = dist))
        end
    end

    # Process individual animals
    for (species_index, animal) in enumerate(model.individuals.animals)
        if species_index != sp
            species_data = animal.data
            size_range = (min_prey * sp_length_ind .<= species_data.length) .& (species_data.length .<= max_prey * sp_length_ind)
            index1 = findall(size_range)
            for i in index1
                add_prey(1, species_index, species_data, i)
            end
        end
    end

    # Process pool animals
    for (pool_index, animal) in enumerate(model.pools.pool)
        pool_data = animal.data
        size_range = (min_prey * sp_length_ind .<= pool_data.length) .& (pool_data.length .<= max_prey * sp_length_ind)
        index1 = findall(size_range)
        for i in index1
            add_prey(2, pool_index, pool_data, i)
        end
    end
    return preys
end


function calculate_distances_pool_prey(model::MarineModel, pool, ind, min_prey, max_prey, detection)
    preys = DataFrame(Type = Int[],Sp = Int[], Ind = Int[], x = Float64[], y = Float64[], z = Float64[], Weight = Float64[], Distance = Float64[])
    
    sp_data = model.pools.pool[pool].data
    for (species_index, animal) in enumerate(model.individuals.animals)
        species_data = animal.data
        size_range = (min_prey * sp_data.length[ind] .<= species_data.length) .& (species_data.length .<= max_prey * sp_data.length[ind])
        index1 = findall(size_range)
        if !isempty(index1)
            for i in index1
                dx = sp_data.x[ind] .- species_data.x[i]
                dy = sp_data.y[ind] .- species_data.y[i]
                dz = sp_data.z[ind] .- species_data.z[i]
                dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
                if dist <= detection
                    push!(preys, (; Type = 1,Sp = species_index, Ind = i, x = species_data.x[i], y = species_data.y[i], z = species_data.z[i], Weight = species_data.biomass[i], Distance = dist[1]))
                end
            end
        end
    end
    for (pool_index, animal) in enumerate(model.pools.pool)
        if pool_index != pool
            pool_data = animal.data
            size_range = (min_prey * sp_data.length[ind] .<= pool_data.length) .& (pool_data.length .<= max_prey * sp_data.length[ind])
            index1 = findall(size_range)
            if !isempty(index1)
                for i in index1
                    dx = sp_data.x[ind] .- pool_data.x[i]
                    dy = sp_data.y[ind] .- pool_data.y[i]
                    dz = sp_data.z[ind] .- pool_data.z[i]

                    dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
                    if dist <= detection
                        push!(preys, (; Type = 2, Sp = pool_index, Ind = i, x = pool_data.x[i], y = pool_data.y[i], z = pool_data.z[i], Weight = pool_data.biomass[i], Distance = dist[1]))
                    end
                end
            end
        end
    end

    return preys
end

function find_nearest_prey(model::MarineModel, sp, ind, min_prey, max_prey)
    preys = DataFrame(x = Float64[], y = Float64[], z = Float64[], Distance = Float64[])
    
    min_distance = 5e6
    sp_data = model.individuals.animals[sp].data
    for (species_index, animal) in enumerate(model.individuals.animals)
        if species_index != sp
            species_data = animal.data
            size_range = (min_prey * sp_data.length[ind] .<= species_data.length) .& (species_data.length .<= max_prey * sp_data.length[ind])
            index1 = findall(size_range)
            if !isempty(index1)
                for i in index1
                    dx = sp_data.x[ind] .- species_data.x[i]
                    dy = sp_data.y[ind] .- species_data.y[i]
                    dz = sp_data.z[ind] .- species_data.z[i]
                    dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)

                    if dist[1] <= min_distance
                        push!(preys, (; x = species_data.x[i], y = species_data.y[i], z = species_data.z[i], Distance = dist[1]))
                        min_distance = dist[1]
                    end
                end
            end
        end
    end
    for (pool_index, animal) in enumerate(model.pools.pool)
        pool_data = animal.data
        size_range = (min_prey * sp_data.length[ind] .<= pool_data.length) .& (pool_data.length .<= max_prey * sp_data.length[ind])
        index1 = findall(size_range)
        if !isempty(index1)
            for i in index1
                dx = sp_data.x[ind] .- pool_data.x[i]
                dy = sp_data.y[ind] .- pool_data.y[i]
                dz = sp_data.z[ind] .- pool_data.z[i]
                dist = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
                if dist[1] <= min_distance
                    push!(preys, (; x = pool_data.x[i], y = pool_data.y[i], z = pool_data.z[i], Distance = dist[1]))
                    min_distance = dist[1]
                end
            end
        end
    end
    if nrow(preys) > 0
        index = argmin(preys.Distance)
        model.individuals.animals[sp].data.x[ind] .+= preys.x[index]
        model.individuals.animals[sp].data.y[ind] .+= preys.y[index]
        model.individuals.animals[sp].data.z[ind] .+= preys.z[index]
    end
    return
end

function calculate_distances_pred(model::MarineModel, sp, ind, min_pred, max_pred, detection)
    preds = DataFrame(Sp = Int[], Ind = Int[], x = Float64[], y = Float64[], z = Float64[], Weight = Float64[], Distance = Float64[])
    
    sp_data = model.individuals.animals[sp].data
    for (species_index, animal) in enumerate(model.individuals.animals)
        if species_index != sp
            species_data = animal.data
            size_range = (min_pred .<= species_data.length) .& (species_data.length .<= max_pred)
            index1 = findall(size_range)
            if !isempty(index1)
                for i in index1
                    dx = sp_data.x[ind][1] - species_data.x[i]
                    dy = sp_data.y[ind][1] - species_data.y[i]
                    dz = sp_data.z[ind][1] - species_data.z[i]
                    dist = sqrt(dx^2 + dy^2 + dz^2)
                    if dist <= detection[1]
                        push!(preds, (; Sp = species_index, Ind = i, x = species_data.x[i], y = species_data.y[i], z = species_data.z[i], Weight = species_data.biomass[i], Distance = 0.0))
                    end
                end
            end
        end
    end
    return preds
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

function move_predator(model, sp, ind, prey_df)
    swim_velo = model.individuals.animals[sp].p.Swim_velo[2][sp] * model.individuals.animals[sp].data.length[ind] / 1000

    # Handling time and swimming time
    handling_time = 2.0 / 60
    time_to_prey = prey_df.Distance[1] / swim_velo

    # Update predator
    model.individuals.animals[sp].data.x[ind] .= prey_df.x[1]
    model.individuals.animals[sp].data.y[ind] .= prey_df.y[1]
    model.individuals.animals[sp].data.z[ind] .= prey_df.z[1]

    # Total time
    t = handling_time .+ time_to_prey
    return t[1]
end

function move_pool(model, pool, ind, prey_df)
    swim_velo = 1.5 * model.pools.pool[pool].data.length[ind] / 1000

    # Handling time and swimming time
    handling_time = 2.0
    time_to_prey = prey_df.Distance[1] / swim_velo

    # Update predator

    model.pools.pool[pool].data.x[ind] = prey_df.x[1]
    model.pools.pool[pool].data.y[ind] = prey_df.y[1]
    model.pools.pool[pool].data.z[ind] = prey_df.z[1]

    # Total time
    t = handling_time .+ time_to_prey
    return t[1]
end

function eat(model::MarineModel, sp, ind, outputs)
    ddt = model.individuals.animals[sp].p.t_resolution[2][sp] * 60  # Seconds
    animal = model.individuals.animals[sp]
    animal_data = animal.data
    length_ind = animal_data.length[ind]
    vis_prey_ind = animal_data.vis_prey[ind]
    gut_fullness_ind = animal_data.gut_fullness[ind]
    max_stomach = animal_data.biomass[ind] * 0.2

    max_dist = 1.5 * length_ind / 1000 * ddt
    detection = min(vis_prey_ind, max_dist)
    min_prey_limit = 0.01
    max_prey_limit = 0.1

    prey_list = calculate_distances_prey(model, sp, ind, min_prey_limit, max_prey_limit, detection)

    while ddt > 0 && nrow(prey_list) > 0
        max_dist = 1.5 * length_ind / 1000 * ddt
        index = argmin(prey_list.Distance)
        chosen_prey = prey_list[index, :]

        if prey_list.Distance[index] <= max_dist[1]
            t = move_predator(model, sp, ind, chosen_prey)
            ddt -= t
            if prey_list.Type[index] == 1
                ration = prey_list.Weight[index]
                model.individuals.animals[sp].data.ration[ind] .+= ration
                predation_mortality(model, chosen_prey, outputs)
                deleteat!(prey_list, index)
            else
                ration = min(0,model.pools.pool[prey_list.Sp[index]].data.biomass[prey_list.Ind[index]])
                model.individuals.animals[sp].data.ration[ind] .+= ration
                if model.pools.pool[prey_list.Sp[index]].data.biomass[prey_list.Ind[index]] <= 0
                    deleteat!(prey_list, index)
                end

                reduce_pool(model, prey_list.Sp[index], prey_list.Ind[index], ration)

                if gut_fullness_ind == max_stomach
                    break
                end
            end
        else
            break
        end
    end
    return ddt, prey_list
end

function pool_predation(model, pool)
    if model.pools.pool[pool].characters.Type[2][pool] == "Predator"
        @Threads.threads for ind in 1:length(model.pools.pool[pool].data.x)
            ddt = 60 #1 minute
            max_dist = 1.5 * model.pools.pool[pool].data.length[ind] / 1000 * model.dt
            detection = min(model.pools.pool[pool].data.vis_prey[ind],max_dist)    
            min_prey_limit = 0.01
            max_prey_limit = 0.1
            prey_list = calculate_distances_pool_prey(model,pool,ind,min_prey_limit,max_prey_limit,detection)
            while ddt > 0
                max_dist = 1.5 * model.pools.pool[pool].data.length[ind] / 1000 * ddt * model.dt
                if nrow(prey_list) > 0
                    index = argmin(prey_list.Distance)
                    chosen_prey = prey_list[index, :]
                    if chosen_prey.Distance[1] <=max_dist
                        t = move_pool(model,pool,ind,chosen_prey)
                        #Consume prey
                        if chosen_prey.Type[1] == 1
                            model.pools.pool[pool].data.ration[ind] += prey_list.Weight[index]
                            predation_mortality(model,chosen_prey, outputs)
                            deleteat!(prey_list, index)
                        else
                            model.pools.pool[pool].data.ration[ind] += min(chosen_prey.Weight[1],model.pools.pool[pool].data.biomass[ind])
                            reduce_pool(model,chosen_prey.Sp[1],chosen_prey.Ind[1],model.pools.pool[pool].data.ration[ind])
                            if model.pools.pool[chosen_prey.Sp[1]].data.biomass[chosen_prey.Ind[1]] <= 0
                                deleteat!(prey_list, index)
                            end
                        end
                        ddt -= t
                        if model.pools.pool[pool].data.ration[ind] == model.pools.pool[pool].data.biomass[ind] #Restrict feeding @ 100% of bodyweight per day
                            ddt = 0
                        end
                    else
                        ddt = 0
                    end
                else
                    ddt = 0
                end
            end
        end
    end
end


