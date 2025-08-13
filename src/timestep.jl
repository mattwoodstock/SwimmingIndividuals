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

    # Move resource groups, if necessary. Only functoinal at less than 12 hour timestep
    if month_index != envi.ts
        move_resources!(model, month_index)
    end

    is_night_now = model.t < 360 || model.t > 1080
    t_previous = (model.t - sim.ΔT + 1440) % 1440 # Handle midnight wrap-around
    was_night_before = t_previous < 360 || t_previous > 1080
    
    if is_night_now != was_night_before
        vertical_resource_movement!(model)
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
        if model.iteration % 10 == 0
            # Get current population and capacity
            n_alive = count(x -> x == 1.0, Array(model.individuals.animals[spec].data.alive))
            current_maxN = length(model.individuals.animals[spec].data.x)
            resize_threshold = 0.90 # 90% capacity

            # If population exceeds the threshold, resize the storage
            if n_alive > current_maxN * resize_threshold
                new_maxN = floor(Int, current_maxN * 1.5) # Increase capacity by 50%
                resize_agent_storage!(model, spec, new_maxN)
            end
        end

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

            energy!(model, spec, ind_temp, living,outputs) # Assuming energy! is the new kernel launcher

            print("fish | ")
            if (model.iteration > model.spinup)
                fishing!(model, spec, day_index, outputs) # Assuming fishing! is the kernel launcher
            end
        end
        
        # Age all individuals
        species_data.age .+= (model.dt / 1440)
    end

    print("resources | ")
    # --- Resource procedure (using kernel launchers) ---
    if (model.iteration > model.spinup)
        resource_predation!(model, outputs)
    end
    resource_mortality!(model)

    resource_growth!(model)
   
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