function TimeStep!(sim::MarineSimulation)
    model = sim.model
    envi = model.environment
    fisheries = model.fishing
    outputs = sim.outputs
    species::Int32 = model.n_species
    arch = model.arch # Get architecture for checks

    model.iteration += 1
    model.t = (model.t + sim.ΔT) % 1440 # Reset the day at midnight

    # Get right day and month
    origin = DateTime(2025, 1, 1, 0, 0)
    elapsed_minutes = (model.iteration - 1) * sim.ΔT
    current_datetime = origin + Minute(round(Int, elapsed_minutes))
    current_date = Date(current_datetime)
    month_index = month(current_date)
    day_index = dayofyear(current_date)

    if month_index != envi.ts
        move_resources!(model, month_index)
    end

    if day_index == 1
        # This operation is fine as it's on a small CPU array of structs
        (fishery -> (fishery.cumulative_catch = 0; fishery.cumulative_inds = 0)).(fisheries)
    end

    envi.ts = month_index

    print(current_date)
    print(": ")
    
    # Loop over focal species
    for spec in 1:species
        species_data = model.individuals.animals[spec].data
        species_chars = model.individuals.animals[spec].p

        # Find all living agents who are old enough to be modeled
        cpu_alive = Array(species_data.alive)
        cpu_age = Array(species_data.age)
        
        larval_duration = species_chars.Larval_Duration[2][spec]
        living::Vector{Int32} = findall(i -> cpu_alive[i] == 1 && cpu_age[i] >= larval_duration, eachindex(cpu_alive))

        if !isempty(living)
            # --- Population stats (requires GPU->CPU transfer for sum/mean) ---
            model.abund[spec] = sum(Array(species_data.abundance[living]))
            model.bioms[spec] = sum(Array(species_data.biomass_school[living]))
            print(length(living))
            print("  Abundance: ")
            print(model.abund[spec])
            print("  Mean Length: ")
            println(mean(Array(species_data.length[living])))


            # --- Main agent update loop ---
            print("behave | ")
            behavior(model, spec, living, outputs) # 'behavior' dispatches to kernel-based functions

            ind_temp = individual_temp!(model, spec)

            print("energy | ")

            energy!(model, spec, ind_temp, living) # Assuming energy! is the new kernel launcher

            print("fish | ")
            if (model.iteration > model.spinup)
                fishing!(model, spec, day_index, outputs) # Assuming fishing! is the kernel launcher
            end
        end
        
        # Age all individuals (this broadcast is fine)
        species_data.age .+= (model.dt / 1440)
    end

    print("resources | ")
    # --- Resource procedure (using new kernel launchers) ---
    resource_predation!(model, outputs) # This function still needs optimization
    resource_growth!(model)
    resource_mortality!(model)
    
    println("results | ")

    if (model.t % model.output_dt == 0)
        timestep_results(sim)
        if (model.iteration > model.spinup)
            fishery_results(sim)
        end
    end

    if model.plt_diags == 1 ## Gather diagnostic-based results to make sure the model works after code revisions
        assemble_diagnostic_results(model,sim.run,model.iteration)
    end

    # --- Reset per-timestep accumulators ---
    for spec in 1:species
        model.individuals.animals[spec].data.ration .= 0.0
        model.individuals.animals[spec].data.active .= 0.0
    end

    # Annual reset (this is fine)
    if day_index == 365
        (fishery -> (fishery.cumulative_catch = 0; fishery.cumulative_inds = 0)).(fisheries)
    end
end