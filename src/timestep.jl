function TimeStep!(sim::MarineSimulation)
    model = sim.model
    envi = model.environment
    fisheries = model.fishing
    outputs = sim.outputs
    species::Int64 = model.n_species
    resources::Int64 = model.n_resource

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

    if month_index != envi.ts
        model.resources = move_resources(model,month_index)
    end

    if day_index == 1
        fishery -> (fishery.cumulative_catch = 0; fishery.cumulative_inds = 0).(fisheries)
    end

    envi.ts = month_index

    chunk_size::Int64 = 10000000 #Process XXX individuals at a time. Use this if you run into memory issues.
    print(current_date)
    print(":   ")
    #Add the behavioral context for each species
    for spec in 1:species 
        species_data = model.individuals.animals[spec].data
        species_chars = model.individuals.animals[spec].p

        if model.t % dt == 0
            living::Vector{Int64} = findall(x -> x == 1, species_data.alive)
            to_model = findall(i -> species_data.alive[i] == 1 && species_data.age[i] >= species_chars.Larval_Duration[2][spec], eachindex(species_data.alive))

            model.abund[spec] = sum(species_data.abundance[living])
            model.bioms[spec] = sum(species_data.biomass_school[living])
            print(length(species_data.length[living]))
            print("   Abundance: ")

            print(model.abund[spec])
            print("   Mean Length: ")
            println(mean(species_data.length[living]))

            ## Gather initial abundances for STE mortality calculations
            init_abundances(model,spec,outputs)

            if length(living) > 0
                #Divide into chunks for quicker processing and eliminating memory allocation at one time.
                n::Int64 = length(living)
                num_chunks::Int64 = Int(ceil(n/chunk_size))
                for chunk in 1:num_chunks
                    start_idx = (chunk-1) * chunk_size + 1
                    end_idx = min(chunk*chunk_size,n)
                    chunk_indices::Vector{Int64} = view(living,start_idx:end_idx)
                    print("behave | ")

                    behavior(model, spec, chunk_indices, outputs)

                    ind_temp::Vector{Float64} = individual_temp(model, spec, chunk_indices, envi)

                    print("energy | ")

                    energy(model, spec, ind_temp,chunk_indices)
                    print("fish | ")

                    if (model.iteration > model.spinup) #Restrict fishing until after a no-fishing burn-in
                        fishing(model, fisheries, spec, day_index,chunk_indices,outputs)
                    end
                end
            end
        end 
        species_data.age .+= (model.dt / 1440)
    end

    print("resources | ")

    #Resource procedure
    resource_predation(model,outputs)
    resource_growth(model)
    resource_mortality(model)
    println("results | ")

    if (model.t % model.output_dt == 0) #Only output results after the spinup is done.
        timestep_results(sim) #Assign and output

        if (model.iteration > model.spinup)
            fishery_results(sim,fisheries)
        end
    end 

    #Reset at each time step
    for spec in 1:species 
        model.individuals.animals[spec].data.ration::Vector{Float64} .= 0.0
        model.individuals.animals[spec].data.active::Vector{Float64} .= 0.0
    end

    if day_index == 365 #Reset year at day 365
        for fishery in fisheries
            fishery.cumulative_catch = 0
            fishery.cumulative_inds = 0
        end
    end
end

