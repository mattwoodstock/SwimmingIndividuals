function holling_2(model, sp, ind, pool_prey, range, outputs)
    # Precompute constants
    t_resolution = model.individuals.animals[sp].p.t_resolution[2][sp]
    lon = Int(ceil(model.individuals.animals[sp].data.pool_x[ind]))
    lat = Int(ceil(model.individuals.animals[sp].data.pool_y[ind]))
    depth = Int(ceil(model.individuals.animals[sp].data.pool_z[ind]))

    handling_time = 2.0 / 60  # 2 seconds scaled to resolution
    attack_rate = range
    density = sum(pool_prey) / model.cell_size
    max_stomach = model.individuals.animals[sp].data.weight[ind] * 0.1
    if density > 0
        # Calculate encounter rate
        encounter = (attack_rate * density) * 60 * t_resolution #Number of expected encounters per time step

        # Include the possibility of predation for pooled preys in low prey densities
        left = encounter % 1
        if encounter > typemax(Int64)
            # Handle the case where encounter is too large
            adj_encounter = typemax(Int64)
        elseif encounter < typemin(Int64)
            # Handle the case where encounter is too small
            adj_encounter = typemin(Int64)
        else
            # Convert encounter to Int64
            adj_encounter = trunc(Int64, encounter)
        end
        distance = cbrt(model.cell_size)/cbrt(density) #Meters to travel in a homogenous prey field.

        swim_velo = model.individuals.animals[sp].p.Swim_velo[2][sp] *model.individuals.animals[sp].data.length[ind]/1000 #meters per second the animal swims.

        time_to_prey = distance/swim_velo #seconds to prey

        t = t_resolution * 60 #Initial seconds remaining

        # Loop over pool_prey
        while (adj_encounter > 0) & (t > 0) & (model.individuals.animals[sp].data.gut_fullness[ind] < max_stomach)
            t -= (time_to_prey + handling_time) #Compute this first, so animal has to get there.
            if t >= 0
                indices = findall(x -> x > 0, pool_prey)
                if length(indices) > 0
                    index = argmax(model.pools.pool[1].characters.Avg_energy[2][indices])

                    ind_size = model.pools.pool[index].characters.Min_Size[2][index] + rand() * (model.pools.pool[index].characters.Max_Size[2][index]-model.pools.pool[index].characters.Min_Size[2][index]) / 10
                    
                    ind_weight = model.pools.pool[index].characters.LWR_a[2][index] * ind_size ^ model.pools.pool[index].characters.LWR_b[2][index]

                    model.individuals.animals[sp].data.gut_fullness[ind] += ind_weight

                    consumed = ind_weight * model.pools.pool[index].characters.Energy_density[2][index]

                    # Update outputs and model data
                    outputs.consumption[sp, model.n_species + index, lon, lat, depth, (model.iteration % model.output_dt) + 1] += ind_weight
                    model.individuals.animals[sp].data.ration[ind] += consumed

                    # Reduce pool size
                    reduce_pool(model, index, lon, lat, depth)
                    pool_prey[index] -= 1
                    adj_encounter -= 1
                end
            end
        end
    end
    return nothing
end

function calculate_distances_prey(model::MarineModel, sp, ind, min_prey, max_prey, detection)
    preys = DataFrame(Sp = Int[], Ind = Int[], x = Float64[], y = Float64[], z = Float64[], Weight = Float64[], Distance = Float64[])
    
    sp_data = model.individuals.animals[sp].data
    for (species_index, animal) in enumerate(model.individuals.animals)
        if species_index != sp
            species_data = animal.data
            size_range = (min_prey * sp_data.length[ind] .<= species_data.length) .& (species_data.length .<= max_prey * sp_data.length[ind])
            index1 = findall(size_range)
            if !isempty(index1)
                for i in index1
                    dx = sp_data.x[ind] - species_data.x[i]
                    dy = sp_data.y[ind] - species_data.y[i]
                    dz = sp_data.z[ind] - species_data.z[i]
                    dist = sqrt(dx^2 + dy^2 + dz^2)
                    if dist <= detection
                        push!(preys, (; Sp = species_index, Ind = i, x = species_data.x[i], y = species_data.y[i], z = species_data.z[i], Weight = species_data.weight[i], Distance = dist))
                    end
                end
            end
        end
    end
    return preys
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
                    dx = sp_data.x[ind] - species_data.x[i]
                    dy = sp_data.y[ind] - species_data.y[i]
                    dz = sp_data.z[ind] - species_data.z[i]
                    dist = sqrt(dx^2 + dy^2 + dz^2)
                    if dist <= detection
                        push!(preds, (; Sp = species_index, Ind = i, x = species_data.x[i], y = species_data.y[i], z = species_data.z[i], Weight = species_data.weight[i], Distance = 0.0))
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

    detection = visual_range_preys(model, sp, ind)

    prey_list = calculate_distances_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)

    searched_volume = (1/2) * (4/3) * pi * detection^3 #Calculate the maximum searched sphere for the predator (i.e., maximum search volume). Assumed animal can successfuly scan 50% of area
    return prey_list, searched_volume
end

function move_predator!(model, sp, ind, prey_df)
    swim_velo = model.individuals.animals[sp].p.Swim_velo[2][sp] * model.individuals.animals[sp].data.length[ind] / 1000

    # Handling time and swimming time
    handling_time = 2.0 / 60
    time_to_prey = prey_df.Distance[1] / swim_velo

    # Update predator position
    model.individuals.animals[sp].data.x[ind] = prey_df.x[1]
    model.individuals.animals[sp].data.y[ind] = prey_df.y[1]
    model.individuals.animals[sp].data.z[ind] = prey_df.z[1]

    # Update activity time
    model.individuals.animals[sp].data.active_time[ind] += time_to_prey / 60

    # Total time
    t = handling_time + time_to_prey

    return t
end

function prey_density(model, sp, ind, preys, area)
    # Precompute constant values
    min_prey_limit = 0.01
    max_prey_limit = 0.1
    min_prey = model.individuals.animals[sp].data.length[ind] * min_prey_limit
    max_prey = model.individuals.animals[sp].data.length[ind] * max_prey_limit
    
    # Calculate density of IBM preys
    IBM_dens = !isempty(preys) ? size(preys, 1) / area : 0
    
    # Calculate density of pooled preys
    pool_dens = 0
    pool_list = Float64[]
    
    for i in 1:model.n_pool
        dens = model.pools.pool[i].density.num[Int(ceil(model.individuals.animals[sp].data.pool_x[ind])), Int(ceil(model.individuals.animals[sp].data.pool_y[ind])), Int(ceil(model.individuals.animals[sp].data.pool_z[ind]))]
        
        if model.pools.pool[i].characters.Min_Size[2][i] < min_prey && model.pools.pool[i].characters.Max_Size[2][i] < max_prey
            first = max_prey - min_prey
            second = model.pools.pool[i].characters.Max_Size[2][i] - model.pools.pool[i].characters.Min_Size[2][i]
            overlap1 = abs(first - second)

            overlap = ifelse(first >= second, overlap1 / first, overlap1 / second)
            overlap = min(overlap, 1)  # Ensure overlap is between 0 and 1
            
            prop_in_range = dens * overlap
            pool_dens += dens * prop_in_range
            push!(pool_list, dens * prop_in_range)
        else
            push!(pool_list, 0.0)
        end
    end
    
    return IBM_dens, pool_list
end

function eat!(model::MarineModel, sp, ind, outputs)
    ddt = model.individuals.animals[sp].p.t_resolution[2][sp]  # Subset of time
    prey_list, search_area = detect_prey(model, sp, ind)

    IBM_dens, pool_prey = prey_density(model, sp, ind, prey_list, search_area)

    lon = Int(round(model.individuals.animals[sp].data.pool_x[ind],digits=0))
    lat = Int(round(model.individuals.animals[sp].data.pool_y[ind],digits=0))
    depth = Int(round(model.individuals.animals[sp].data.pool_z[ind],digits=0))

    density = IBM_dens + sum(pool_prey)
    pool_prop = sum(pool_prey) / density
    max_stomach = model.individuals.animals[sp].data.weight[ind] * 0.1

    if !isempty(prey_list) & (model.individuals.animals[sp].data.gut_fullness[ind] < max_stomach)  # There are preys within range. Need to choose one and "remove" it.
        while ddt > (1 - pool_prop) && !isempty(prey_list)  ## Can only eat if there is time left for IBM species
            # Select the closest prey
            idx_closest = argmin(prey_list.Distance)
            chosen_prey = prey_list[idx_closest, :]
            predation_mortality(model, chosen_prey, outputs)
            ddt = move_predator!(model, sp, ind, chosen_prey)

            model.individuals.animals[sp].data.gut_fullness[ind] += chosen_prey.Weight[1]

            model.individuals.animals[sp].data.ration[ind] += chosen_prey.Weight[1] * model.individuals.animals[chosen_prey.Sp[1]].p.energy_density[2][chosen_prey.Sp[1]]

            outputs.consumption[sp, chosen_prey.Sp[1], lon,lat,depth, (model.iteration % model.output_dt) + 1] += chosen_prey.Weight[1]

            # Remove animal from prey list
            deleteat!(prey_list, idx_closest)
        end

    else
        ddt = 0  # No preys within range, therefore we do not need this.
    end

    # Eat pooled preys
    pool_vis = search_area * pool_prop  # Proportion of search area remaining
    holling_2(model, sp, ind, pool_prey, pool_vis, outputs)
end

function pool_predation(model, pool)
    min_sizes = model.pools.pool[pool].characters.Min_Size[2][pool] * 0.01
    max_sizes = model.pools.pool[pool].characters.Max_Size[2][pool] * 0.1

    avg_pred_size = (model.pools.pool[pool].characters.Min_Size[2][pool] + model.pools.pool[pool].characters.Max_Size[2][pool]) / 2

    avg_pred_weight = model.pools.pool[pool].characters.LWR_a[2][pool] * avg_pred_size ^ model.pools.pool[pool].characters.LWR_b[2][pool]


    # Process depth bins concurrently if possible
    Threads.@threads for i in 1:model.grid.Nx
        for j in 1:model.grid.Ny
            for k in 1:model.grid.Nz
                density = model.pools.pool[i].density.num[i,j,k]
                biomass = density * avg_pred_weight

                q_timestep = biomass * 0.03/1440 #Biomass to consume per minute

                # Initialize IBM_prey outside the loop
                IBM_prey = DataFrame(Sp = Int[], Ind = Int[], Weight = Float64[])
                # Gather Focal species in grid cell and size range
                sp = 1
                for (species_index, animal) in pairs(model.individuals.animals)
                    weights = animal.data.weight
                    indices = findall(x -> q_timestep <= x, weights)
                    if length(indices) > 0
                        lengths = animal.data.length
                        indices1 = findall(x -> min_sizes <= x <= max_sizes, lengths)
                        if length(indices1) > 0
                            pool_z = Int.(animal.data.pool_z)
                            indices2 = findall(x -> x == k, pool_z)
                            if length(indices2) > 0
                                pool_x = Int.(animal.data.pool_x)
                                indices3 = findall(x -> x == i, pool_x)
                                if length(indices3) > 0
                                    pool_y = Int.(animal.data.pool_y)
                                    indices4 = findall(x -> x == j, pool_y)
                                    if length(indices4) > 0
                                        common_indices = intersect(indices,intersect(indices1, intersect(indices2, intersect(indices3, indices4))))

                                        append!(IBM_prey, DataFrame(Sp = fill(sp, length(common_indices)), Ind = common_indices, Weight = animal.data.weight[common_indices]))
                                    end
                                end
                            end
                        end
                    end
                    sp += 1
                end
                IBM_dens = !isempty(IBM_prey) ? size(IBM_prey, 1) / model.cell_size : 0 #Number per m3

                pool_dens = 0
                pool_list = Float64[]

                for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
                    min_weight = model.pools.pool[i].characters.LWR_a[2][i] * model.pools.pool[i].characters.Min_Size[2][i] ^ model.pools.pool[i].characters.LWR_b[2][i] #minimum biomass of one individual

                    if model.pools.pool[i].characters.Min_Size[2][i] < min_sizes && model.pools.pool[i].characters.Max_Size[2][i] < max_sizes && min_weight > q_timestep
                
                        first = max_sizes - min_sizes
                        second = model.pools.pool[i].characters.Max_Size[2][i] - model.pools.pool[i].characters.Min_Size[2][i]
                        overlap1 = abs(first - second)
                
                        overlap = ifelse(first >= second, overlap1 / first, overlap1 / second)
                        overlap = min(overlap, 1)  # Ensure overlap is between 0 and 1
                            
                        prop_in_range = density * overlap
                        pool_dens += density * prop_in_range
                        push!(pool_list, density * prop_in_range)
                    else
                        push!(pool_list, 0.0)
                    end
                end
                IBM_prop = IBM_dens/(IBM_dens + sum(pool_list)) #Abundance proportion
                q_IBM = IBM_prop * q_timestep #Biomass split based on desnity split
                q_pool = 1-q_IBM


                if nrow(IBM_prey) > 0
                    while q_IBM > minimum(IBM_prey.Weight)  # Consume an individual
                        val = Int(round(rand() * size(IBM_prey, 1)))
                        index = IBM_prey[val, :]  # Chosen prey item

                        predation_mortality(model, index, outputs)
                        outputs.consumption[model.n_species + pool, index.Sp[1], i,j,k, (model.iteration % model.output_dt) + 1] += index.Weight[1]

                        deleteat!(IBM_prey,findall(IBM_prey.Sp .== index.Sp[1] .&& IBM_prey.Ind .== index.Ind[1]))
                        q_IBM -= index.Weight[1]
                        IBM_dens -= index.Weight[1]
                    end
                end

                for index in 1:length(pool_list)
                    if pool_list[index] > 0
                        ind_size = (model.pools.pool[index].characters.Min_Size[2][index] + rand() * (model.pools.pool[index].characters.Max_Size[2][index]- model.pools.pool[index].characters.Min_Size[2][index])) / 10

                        ind_biomass = model.pools.pool[index].characters.LWR_a[2][index] * ind_size ^ model.pools.pool[index].characters.LWR_b[2][index]

                        biom_consumed = pool_list[index] * ind_biomass * q_pool

                        num_consumed = biom_consumed / ind_biomass

                        outputs.consumption[model.n_species + pool, model.n_species + index, i,j,k, (model.iteration % model.output_dt) + 1] += biom_consumed

                        model.pools.pool[index].density.num[i,j,k] -= num_consumed

                        pool_list[index] -= num_consumed
                    end
                end
            end
        end
    end
end


