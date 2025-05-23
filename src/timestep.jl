function TimeStep!(sim::MarineSimulation)
    model = sim.model
    envi = model.environment
    fisheries = model.fishing
    outputs = sim.outputs
    species::Int64 = model.n_species
    pool::Int64 = model.n_pool

    model.iteration += 1
    model.t += sim.ΔT
    model.t %= 1440  #Reset the day at midnight

    #Get right day and month
    origin = DateTime(2025, 1, 1, 0, 0)
    elapsed_minutes = (model.iteration-1) * sim.ΔT
    current_datetime = origin + Minute(round(Int, elapsed_minutes))
    current_date = Date(current_datetime)
    month_index = month(current_date)
    day_index = dayofyear(current_date)
    year_index = year(current_date)
    envi.ts = month_index

    chunk_size::Int64 = 10000 #Process 1,000 individuals at a time.
    print(month_index)
    print(":   ")
    #Add the behavioral context for each species
    for spec in 1:species 
        species_data = model.individuals.animals[spec].data
        species_chars = model.individuals.animals[spec].p
        t_resolution::Float64 = species_chars.t_resolution[2][spec]
        if model.t % t_resolution == 0
            alive::Vector{Int64} = findall(species_data.ac::Vector{Float64} .== 1.0)

            model.abund[spec] = length(alive)
            model.bioms[spec] = sum(species_data.biomass[alive])
            if length(alive) > 0
                #Divide into chunks for quicker processing and eliminating memory allocation at one time.
                n::Int64 = length(alive)
                num_chunks::Int64 = Int(ceil(n/chunk_size))
                for chunk in 1:num_chunks
                    start_idx = (chunk-1) * chunk_size + 1
                    end_idx = min(chunk*chunk_size,n)
                    chunk_indices::Vector{Int64} = view(alive,start_idx:end_idx)
                    print("behave | ")
                    behavior(model, spec, chunk_indices, outputs)
                    ind_temp::Vector{Float64} = individual_temp(model, spec, chunk_indices, envi)
                    print("energy | ")

                    energy(model, spec, ind_temp,chunk_indices)
                    print("fish | ")

                    apply_fishing!(model, fisheries, spec, day_index,chunk_indices)
                end
            end
        end 
        species_data.daily_ration[alive] .+= species_data.ration[alive]
    end

    print("pools | ")

    #Non-focal species processing. Note the "Larval Pool" does not eat
    for spec in 1:pool   
        n::Int64 = length(model.pools.pool[spec].data.length)
        num_chunks::Int = Int(ceil(n/chunk_size))
        for chunk in 1:num_chunks
            start_idx = (chunk-1) * chunk_size + 1
            end_idx = min(chunk*chunk_size,n)
            chunk_indices = start_idx:end_idx
            pool_predation(model,spec,chunk_indices,outputs,sim.ΔT)
            pool_movement(model,chunk_indices,spec)
        end
    end

    print("results | ")


    if (model.t % model.output_dt == 0) & (model.iteration > model.spinup) #Only output results after the spinup is done.
        timestep_results(sim) #Assign and output
    end 

    pool_growth(model) #Grow pool individuals back to carrying capacity (initial biomass)  
    model.pools.pool[pool + 1].data.age .+= sim.ΔT #Larval animals grow by the timestep

    print("recruit | ")

    recruit(model) #Larva of age 
    
    if model.t == 0 #Reset day at midnight
        for spec in 1:species   
            model.individuals.animals[spec].data.daily_ration::Vector{Float64} .= 0.0
            model.individuals.animals[spec].data.ac::Vector{Float64} .= 1.0
            model.individuals.animals[spec].data.behavior::Vector{Float64} .= 1.0
            model.individuals.animals[spec].data.dives_remaining::Vector{Float64} .= 0.0
        end
    end

    for spec in 1:species 
        model.individuals.animals[spec].data.ration::Vector{Float64} .= 0.0
        model.individuals.animals[spec].data.active::Vector{Float64} .= 0.0
        model.individuals.animals[spec].data.consumed::Vector{Float64} .= 0.0
        model.individuals.animals[spec].data.landscape::Vector{Float64} .= 0.0
    end
    println("recruit | ")
end

