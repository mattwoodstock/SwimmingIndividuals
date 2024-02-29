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

function holling_2_pool(model, pool, IBM_prey, Pool_prey, IBM_dens, outputs, depth, min_size, max_size)

    attack_rate = visual_pool(model,pool,depth)
    handling_time = 2.0 / 60  #ind per second
    density = IBM_dens + sum(Pool_prey[:,"Dens"])
    
    encounter = (attack_rate * density) / (1 + attack_rate * handling_time * density)


    #Predator density
    cell_volume = 1000 * 1 * 1
    pred_inds = model.pools.pool[pool].density.num[depth] * cell_volume
    adjust_inds = Int(round(pred_inds,digits = 0))
    #Number of predators

    adj_encounter = Int(round(encounter,digits = 0)) * adjust_inds
    while adj_encounter > 0
        if (IBM_dens > sum(Pool_prey[:,"Dens"])) && (nrow(IBM_prey) > 0) #Consume an individual
            val = rand(1:nrow(IBM_prey))

            index = IBM_prey[val,:] #Chosen prey item

            outputs.consumption[model.n_species+pool, index.Sp[1], depth, model.iteration] += index.Weight[1]
            predation_mortality(model,index,outputs)
            deleteat!(IBM_prey, val)

        else #Consume a pool animal
            val = argmax(Pool_prey[:,"Dens"])
            index = Pool_prey[val,:]

            avg_size = mean(model.pools.pool[index.Sp[1]].characters.Min_Size[2][index.Sp[1]]:model.pools.pool[index.Sp[1]].characters.Max_Size[2][index.Sp[1]])/10

            encounter_biomass = model.pools.pool[index.Sp[1]].characters.LWR_a[2][index.Sp[1]] * avg_size ^ model.pools.pool[index.Sp[1]].characters.LWR_b[2][index.Sp[1]]
    
            consumed = encounter_biomass * model.pools.pool[index.Sp[1]].characters.Energy_density[2][index.Sp[1]]

            outputs.consumption[model.n_species+pool, model.n_species+index.Sp[1], depth, model.iteration] += consumed

            reduce_pool(model,index.Sp[1],depth)
        end
        adj_encounter -= 1
    end
end

function holling_2(model, sp, ind, pool_prey, range, outputs)
    cell_volume = 1 * 1 * 100
    t_resolution = model.individuals.animals[sp].p.t_resolution[2][sp]

    # Precompute constants
    handling_time = 2.0 / 60 * t_resolution #Scaled to resolution
    attack_rate = range
    capture_success = 0.7 #Langbehn et al. 2019
    density = sum(pool_prey.Dens)
    if density > 0
        encounter = attack_rate * density / (1 + attack_rate * handling_time * density)

        adj_encounter = Int(round(encounter,digits=0))
        # Convert pool_z to integer outside the loop
        depth = ceil(Int, model.individuals.animals[sp].data.pool_z[ind])

        # Loop over pool_prey

        while adj_encounter > 0 #Run until end of time step
            index = argmax(pool_prey[:,"Dens"])
            item = pool_prey[index,:] #Pick highest density value

            avg_size = mean(model.pools.pool[item.Pool[1]].characters.Min_Size[2][item.Pool[1]]:model.pools.pool[item.Pool[1]].characters.Max_Size[2][item.Pool[1]])/10

            encounter_biomass = model.pools.pool[item.Pool[1]].characters.LWR_a[2][item.Pool[1]] * avg_size ^ model.pools.pool[item.Pool[1]].characters.LWR_b[2][item.Pool[1]]

            consumed = capture_success * encounter_biomass * model.pools.pool[item.Pool[1]].characters.Energy_density[2][item.Pool[1]]

            # Update outputs and model data
            outputs.consumption[sp, model.n_species + item.Pool[1], depth, model.iteration] += consumed
            model.individuals.animals[sp].data.ration[ind] += consumed

            # Reduce pool size
            reduce_pool(model, item.Pool[1], depth)
            pool_prey[index,"Dens"] -= 1/cell_volume
            adj_encounter -= 1
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
    max_prey_limit = 0.05 #Animals cannot eat anything greater than 5% of their body length
    #Prey Detection Distances
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/geb.13782

    detection = visual_range_preys(model, sp, ind)

    prey_list = calculate_distances_prey(model,sp,ind,min_prey_limit,max_prey_limit,detection)

    swim_velo = model.individuals.animals[sp].data.length[ind]/1000 * model.individuals.animals[sp].p.Swim_velo[2][sp] * 60 * model.individuals.animals[sp].p.t_resolution[2][sp]

    if model.dimension == 1
        searched_volume = detection #Prey will be condensed. This includes visual range on z-axis.
    else
        searched_volume = ((1/2) * (4/3) * pi * detection^3) * 60 * model.individuals.animals[sp].p.t_resolution[2][sp] #Calculate the maximum searched sphere for the predator (i.e., maximum search volume). Assumed animal can successfuly scan 50% of area
    end
    return prey_list, searched_volume
end

function move_predator!(model,sp,ind,prey_df)
    #Handling time is a function of gut fullness with 2.0 seconds as the base. May want a better source.
    ##https://cdnsciencepub.com/doi/pdf/10.1139/f74-186
    ##seconds of Handling time from Langbehn et al. 2019. Essentially a cool-off period after feeding.

    handling_time_0 = 2.0

    handling_time = (1.19 - 1.24 * model.individuals.animals[sp].data.gut_fullness[ind] + 3.6 * model.individuals.animals[sp].data.gut_fullness[ind]^2 / handling_time_0) * handling_time_0

    #Identify x,y,z of prey

    model.individuals.animals[sp].data.x[ind] = prey_df.x[1]
    model.individuals.animals[sp].data.y[ind] = prey_df.y[1]
    model.individuals.animals[sp].data.z[ind] = prey_df.z[1]

    #Seconds of time to swim to prey
    swim_velo = model.individuals.animals[sp].p.Swim_velo[2][sp] * model.individuals.animals[sp].data.length[ind] / 1000 * model.individuals.animals[sp].p.t_resolution[2][sp] * 60

    time_to_prey = prey_df.Distance[1]/swim_velo

    t = ((handling_time + time_to_prey) / 60)

    #Add foraging time to activity time
    model.individuals.animals[sp].data.active_time[ind] = model.individuals.animals[sp].data.active_time[ind] + time_to_prey/60

    return t
end

function fill_gut!(model,sp,ind,prey_df)
    prop_filled = prey_df.Weight[1]/model.individuals.animals[sp].data.weight[ind]
    model.individuals.animals[sp].data.gut_fullness[ind] += prop_filled        
    return nothing
end

function prey_density(model, sp, ind, preys, area)
    # Precompute constant values
    min_prey_limit = 0.01
    max_prey_limit = 0.05
    min_prey = model.individuals.animals[sp].data.length[ind] * min_prey_limit
    max_prey = model.individuals.animals[sp].data.length[ind] * max_prey_limit
    
    # Calculate density of IBM preys
    IBM_dens = !isempty(preys) ? size(preys, 1) / area : 0
    
    # Calculate density of pooled preys
    pool_list = DataFrame(Pool = Int[], Dens = Float64[])
    pool_dens = 0
    
    for i in 1:model.n_pool
        dens = model.pools.pool[i].density.num[Int(ceil(model.individuals.animals[sp].data.pool_z[ind]))]        

        samples = sample_normal(model.pools.pool[i].characters.Min_Size[2][i], model.pools.pool[i].characters.Max_Size[2][i])

        prop_in_range = count(x -> min_prey <= x <= max_prey, samples) / length(samples)

        pool_dens += dens * prop_in_range
        
        push!(pool_list, Dict("Pool" => i, "Dens" => dens * prop_in_range))
    end

    # Calculate relative abundances for weighted feeding
    pool_list[!,:Relative] .= pool_list.Dens ./ sum(pool_list.Dens)
    
    # Remove rows with only zero
    filter!(row -> row.Relative != 0, pool_list)
    
    return IBM_dens,pool_dens, pool_list
end

function prey_density_pool(model,sp,depth,min_size,max_size)
    cell_volume = 100 * 1 * 1 #Cubic meters. Ned to feed this in.

    IBM_prey = DataFrame(Sp = Int[], Ind = Int[], Weight = Float64[])

    #Gather Focal species in grid cell and size range
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
        indicies = findall(x -> min_size <= x<= max_size, model.individuals.animals[species_index].data.length)

        for i in 1:length(indicies)
            if Int(ceil(model.individuals.animals[species_index].data.pool_z[indicies[i]])) == depth
                push!(IBM_prey, (; Sp = species_index, Ind = indicies[i], Weight = model.individuals.animals[species_index].data.weight[indicies[i]]))
            end
        end
    end
    IBM_dens = nrow(IBM_prey)/cell_volume #ind per cubic meter

    Pool_prey = DataFrame(Sp = Int[], Dens = Float64[])

    for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
        dens = model.pools.pool[pool_index].density.num[depth]        

        samples = sample_normal(model.pools.pool[pool_index].characters.Min_Size[2][pool_index], model.pools.pool[pool_index].characters.Max_Size[2][pool_index])
        prop_in_range = count(x -> min_size <= x <= max_size, samples) / length(samples)

        if dens > 0 #Only save if pool prey necessary
            push!(Pool_prey, (; Sp = pool_index, Dens = dens * prop_in_range))
        end
    end
    return IBM_prey, IBM_dens, Pool_prey
end


function eat!(model::MarineModel,sp,ind,outputs)
    ddt = model.individuals.animals[sp].p.t_resolution[2][sp] #Subset of time

    if (model.dimension == 1) #Running the 1D model. Need to update this at some point.

        #Calculate encounter range (i.e., distance an animal can perceive)
        ## Assumed prey length is 10% of predator

        IBM_prey, search_area = available_prey(model,sp,ind)

        ### Need to calculate prey prey_densities by potential preys
        density, pool_prey = prey_density(model,spec_array1,j,IBM_prey,search_area)

        #prey length = mean length of available preys
        holling_2(model,sp,ind,pool_prey,pool_vis,outputs)
        return nothing
    else #Running the 3D model version
        prey_list, search_area = detect_prey(model,sp,ind)
        IBM_dens, pool_dens, pool_prey = prey_density(model,sp,ind,prey_list,search_area)
        pool_prop = pool_dens / (IBM_dens + pool_dens)
        density = IBM_dens + pool_dens
        if (nrow(prey_list) > 0) && (model.individuals.animals[sp].data.gut_fullness[ind] < 1) # There are preys within range. Need to choose one and "remove" it.
            while (ddt > (1-pool_prop)) && (nrow(prey_list) > 0) ## Can only eat if there is time left for IBM species

            #Select the closest prey
            chosen_prey = prey_list[argmin(prey_list.Distance),:]

            predation_mortality(model,chosen_prey,outputs)
            ddt = move_predator!(model,sp,ind,chosen_prey)
            fill_gut!(model,sp,ind,chosen_prey)
            allocate_energy(model,sp,ind,chosen_prey)

            model.individuals.animals[sp].data.daily_ration[ind] += chosen_prey.Weight[1] * model.individuals.animals[chosen_prey.Sp[1]].p.energy_density[2][chosen_prey.Sp[1]]
            model.individuals.animals[sp].data.ration[ind] += chosen_prey.Weight[1] * model.individuals.animals[chosen_prey.Sp[1]].p.energy_density[2][chosen_prey.Sp[1]]

            #Still remove animal from prey list as if it goes away
            deleteat!(prey_list,findall(prey_list.Sp .== chosen_prey.Sp[1] .&& prey_list.Ind .== chosen_prey.Ind[1]))
            end
        else
            ddt = 0 ## No preys within range, therefore we do not need this.
        end
        # Eat pooled preys
        pool_vis = search_area * pool_prop #Proportion of search area remaining
        if nrow(pool_prey) > 0
            holling_2(model,sp,ind,pool_prey,pool_vis,outputs)
        end
    end
end

function pool_predation(model, pool)
    # Precompute constants
    min_sizes = model.pools.pool[pool].characters.Min_Size[2][pool] * 0.01
    max_sizes = model.pools.pool[pool].characters.Max_Size[2][pool] * 0.05

    # Process depth bins concurrently if possible
    @inbounds Threads.@threads for i in 1:model.grid.Nz
        IBM_prey, IBM_dens, Pool_prey = prey_density_pool(model, pool, i, min_sizes, max_sizes)
        holling_2_pool(model, pool, IBM_prey, Pool_prey, IBM_dens, outputs, i, min_sizes, max_sizes)
    end
end
