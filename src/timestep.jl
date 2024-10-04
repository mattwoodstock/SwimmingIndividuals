function TimeStep!(sim)
    start = now()
    model = sim.model
    envi = model.environment
    outputs = sim.outputs

    model.iteration += 1
    model.t += sim.ΔT
    model.t %= 1440  #Reset the day at midnight

    chunk_size = convert(Int64,1000) #Process 1,000 individuals at a time.
    print(model.t)
    print(":   ")

    #Add the behavioral context for each species
    for (species_index, _) in enumerate(keys(model.individuals.animals))
        species_data = model.individuals.animals[species_index].data
        species_chars = model.individuals.animals[species_index].p
        t_resolution = species_chars.t_resolution[2][species_index]
        if model.t % t_resolution == 0
            alive = findall(species_data.ac .== 1.0)
            model.abund[species_index] = length(alive)
            model.bioms[species_index] = sum(species_data.biomass[alive])
            if length(alive) > 0
                #Divide into chunks for quicker processing and eliminating memory allocation at one time.
                n = convert(Int64,length(alive))
                num_chunks = ceil(Int,n/chunk_size)
                for chunk in 1:num_chunks
                    start_idx = (chunk-1) * chunk_size + 1
                    end_idx = min(chunk*chunk_size,n)
                    chunk_indices = alive[start_idx:end_idx]
                    behavior(model, species_index, chunk_indices, outputs)
                    ind_temp = individual_temp(model, species_index, chunk_indices, envi)
                    energy(model, species_index, ind_temp,chunk_indices)
                end
            end
            #reproduce(model,species_index,alive)
        end 
        species_data.daily_ration[alive] .+= species_data.ration[alive]
    end
    #Non-focal species processing
    for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
        n = convert(Int64,length(model.pools.pool[pool_index].data.length))
        num_chunks = ceil(Int,n/chunk_size)
        for chunk in 1:num_chunks
            start_idx = (chunk-1) * chunk_size + 1
            end_idx = min(chunk*chunk_size,n)
            chunk_indices = start_idx:end_idx
            #pool_predation(model,pool_index,chunk_indices,outputs)
        end
    end

    if (model.t % model.output_dt == 0) & (model.iteration > model.spinup) #Only output results after the spinup is done.
        timestep_results(sim) #Assign and output
    end 
    pool_growth(model) #Grow pool individuals back to carrying capacity (initial biomass)  

    if model.t == 0 #Reset day at midnight
        for (species_index, _) in enumerate(keys(model.individuals.animals))
            model.individuals.animals[species_index].data.daily_ration .= 0
            model.individuals.animals[species_index].data.ac .= 1.0
            model.individuals.animals[species_index].data.behavior .= 1.0
        end
    end

    for (species_index, _) in enumerate(keys(model.individuals.animals))
        model.individuals.animals[species_index].data.ration .= 0
        model.individuals.animals[species_index].data.active .= 0
        model.individuals.animals[species_index].data.consumed .= 0.0
        model.individuals.animals[species_index].data.landscape .= 0.0
    end

    stop = now()
    #println(stop-start)
end
