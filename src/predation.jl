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

function holling_2_pool(model, pool, IBM_prey, Pool_prey, IBM_dens, outputs, lon, lat, depth)
    attack_rate = visual_pool(model, pool, depth)
    handling_time = 2.0 / 60  # ind per minute
    density = IBM_dens + sum(Pool_prey)

    encounter = (attack_rate * density) / (1 + attack_rate * handling_time * density)

    # Predator density
    pred_inds = model.pools.pool[pool].density[depth] * model.cell_size
    adjust_inds = round(pred_inds)
    adj_encounter = round(encounter) * adjust_inds

    while adj_encounter > 0
        if (IBM_dens > sum(Pool_prey.Dens)) && !isempty(IBM_prey)  # Consume an individual
            val = rand(1:size(IBM_prey, 1))
            index = IBM_prey[val, :]  # Chosen prey item

            outputs.consumption[model.n_species + pool, index.Sp, lon, lat, depth, (model.iteration % model.output_dt) + 1] += index.Weight
            predation_mortality(model, index, outputs)
            IBM_prey = IBM_prey[1:end .!= val, :]  # Remove consumed individual
        else  # Consume a pool animal
            val = argmax(Pool_prey.Dens)
            index = Pool_prey[val, :]

            avg_size = mean(model.pools.pool[index.Sp].characters.Min_Size[2][index.Sp]:model.pools.pool[index.Sp].characters.Max_Size[2][index.Sp]) / 10
            encounter_biomass = model.pools.pool[index.Sp].characters.LWR_a[2][index.Sp] * avg_size ^ model.pools.pool[index.Sp].characters.LWR_b[2][index.Sp]
            consumed = encounter_biomass * model.pools.pool[index.Sp].characters.Energy_density[2][index.Sp]

            outputs.consumption[model.n_species + pool, model.n_species + index.Sp, lon, lat, depth, (model.iteration % model.output_dt) + 1] += encounter_biomass
            reduce_pool(model, index.Sp, lon, lat, depth)
            Pool_prey = Pool_prey[1:end .!= val, :]  # Remove consumed pool animal
        end
        adj_encounter -= 1
    end
end


function holling_2(model, sp, ind, pool_prey, range, outputs)
    # Precompute constants
    t_resolution = model.individuals.animals[sp].p.t_resolution[2][sp]
    lon = Int(ceil(model.individuals.animals[sp].data.pool_x[ind]))
    lat = Int(ceil(model.individuals.animals[sp].data.pool_y[ind]))
    depth = Int(ceil(model.individuals.animals[sp].data.pool_z[ind]))

    handling_time = 2.0 / 60  # 2 seconds scaled to resolution
    attack_rate = range
    density = sum(pool_prey) / model.cell_size #Check the units of pool_prey
    max_stomach = model.individuals.animals[sp].data.weight[ind] * 0.1
    if density > 0
        # Calculate encounter rate
        encounter = (attack_rate * density) * 60 * t_resolution #Number of expected encounters per time step

        # Include the possibility of predation for pooled preys in low prey densities
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
                    pool_prey[index] -= 1 #Check to assure this is the total number of inds and not a raw abundance
                    adj_encounter -= 1
                end
            end
        end
    end
    return nothing
end



function calculate_distances_matrix(model::MarineModel)
    num_animals = sum(model.ninds)
    distances = zeros(Float64, num_animals, num_animals)
    count1 = 0
    for i in 1:model.n_species
        # Find number of individuals in species i
        spec_array1 = model.individuals.animals[i]
        for j in 1:model.ninds[i]
            count1 += 1
            count2 = 0
            for k in 1:model.n_species
                # Find number of individuals in species k
                spec_array2 = model.individuals.animals[k]
                for l in 1:model.ninds[k]
                    count2 += 1
                    if (j != l) || (i != k)
                        delta_y = spec_array1.data.y[j] - spec_array2.data.y[l]
                        delta_x = spec_array1.data.x[j] - spec_array2.data.x[l]
                        delta_z = spec_array1.data.z[j] - spec_array2.data.z[l]
                        distances[count1, count2] = sqrt(delta_y^2 + delta_x^2 + delta_z^2)
                    else
                        distances[count1, count2] = Inf # Make object far away from itself for modeling purposes
                    end
                end
            end
        end
    end
    return distances
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

    # Update predator
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
        dens = model.pools.pool[i].density[Int(ceil(model.individuals.animals[sp].data.pool_x[ind])), Int(ceil(model.individuals.animals[sp].data.pool_y[ind])), Int(ceil(model.individuals.animals[sp].data.pool_z[ind]))]
        
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

function prey_density_pool(model, sp, lon, lat, depth, min_size, max_size)
    IBM_prey = DataFrame(Sp = Int[], Ind = Int[], Weight = Float64[])

    # Gather Focal species in grid cell and size range
    for (species_index, animal) in pairs(model.individuals.animals)
        lengths = animal.data.length
        pool_z = animal.data.pool_z

        indices = findall(x -> min_size <= x <= max_size, lengths)
        depth_indices = ceil.(Int, pool_z[indices])

        append!(IBM_prey, DataFrame(Sp = fill(species_index, length(indices)), Ind = indices, Weight = animal.data.weight[indices]))
    end

    IBM_dens = nrow(IBM_prey) / model.cell_size  # ind per cubic meter

    Pool_prey = DataFrame(Sp = Int[], Dens = Float64[])

    for (pool_index, pool) in pairs(model.pools.pool)
        density = pool.density[lon, lat, depth]

        if density > 0
            min_sz = pool.characters.Min_Size[2][pool_index]
            max_sz = pool.characters.Max_Size[2][pool_index]
            prop_in_range = count(x -> min_size <= x <= max_size, sample_normal(min_sz, max_sz)) / model.num_samples
            push!(Pool_prey, (; Sp = pool_index, Dens = density * prop_in_range))
        end
    end

    return IBM_prey, IBM_dens, Pool_prey
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
    max_stomach = model.individuals.animals[sp].data.weight[ind] * 0.03

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
                density = model.pools.pool[i].density[i,j,k]
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

                        model.pools.pool[index].density[i,j,k] -= num_consumed

                        pool_list[index] -= num_consumed
                    end
                end
            end
        end
    end
end


