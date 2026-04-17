"""
    TimeStep!(sim::MarineSimulation)

The primary orchestration loop for a single model iteration. 
Handles calendar progression, environment updates, agent biology, and I/O.
Includes the application of instantaneous natural mortality as defined in the Canvas.
"""
function TimeStep!(sim::MarineSimulation)
    model = sim.model
    envi = model.environment
    fisheries = model.fishing
    outputs = sim.outputs
    species::Int32 = model.n_species
    arch = model.arch

    # Increment iteration and update internal day-time counter [0, 1440]
    model.iteration += 1
    model.t = (model.t + sim.ΔT) % 1440 

    # --- 1. Calendar & Date Resolution ---
    # Target start date (post-spinup) is Jan 1, 2026.
    # The datetime is offset by the spinup duration so results always begin in 2026.
    target_start_date = DateTime(2026, 1, 1, 0, 0)
    
    elapsed_minutes = (model.iteration - 1) * sim.ΔT
    current_datetime = target_start_date + Minute(round(Int, elapsed_minutes - model.spinup))
    current_date = Date(current_datetime)
    
    # Calculate previous date to handle daily/annual resets
    previous_elapsed = (model.iteration - 2) * sim.ΔT
    previous_datetime = target_start_date + Minute(round(Int, previous_elapsed - model.spinup))
    previous_date = Date(previous_datetime)
    
    month_index = month(current_date)
    day_index = dayofyear(current_date)

    # --- 2. Environmental Dynamics ---
    if month_index != envi.ts
        move_resources!(model, month_index)
        envi.ts = month_index
    end

    # Diel Vertical Migration (DVM) of resource pools
    is_night_now = model.t < 360 || model.t > 1080
    t_prev_step = (model.t - sim.ΔT + 1440) % 1440
    was_night_before = t_prev_step < 360 || t_prev_step > 1080
    
    if is_night_now != was_night_before
        vertical_resource_movement!(model)
    end

    # --- 3. Counters & Accumulator Resets ---
    if current_date != previous_date
        model.daily_birth_counters = zeros(Int, species)
    end

    if day_index == 1 && current_date != previous_date
        for fishery in fisheries
            fishery.cumulative_catch = 0.0
            fishery.cumulative_inds = 0
            fishery.effort_days = 0.0
            fishery.bycatch_tonnage = 0.0
            fishery.bycatch_inds = 0
        end
    end

    # --- 4. Agent Life History & Behavior ---
    print(current_date, ": ")
    
    for spec in 1:species
        # Dynamic storage resizing
        if model.iteration % 10 == 0
            n_alive = count(x -> x == 1.0f0, Array(model.individuals.animals[spec].data.alive))
            current_maxN = length(model.individuals.animals[spec].data.x)
            if n_alive > current_maxN * 0.90
                resize_agent_storage!(model, spec, floor(Int, current_maxN * 1.5))
            end
        end

        species_data = model.individuals.animals[spec].data
        species_chars = model.individuals.animals[spec].p
        larval_duration = species_chars.Larval_Duration[2][spec]

        # Transfer only necessary status data to CPU for index filtering
        cpu_alive = Array(species_data.alive)
        cpu_age = Array(species_data.age)
        
        # Explicit cast to Vector{Int32} for kernel compatibility
        alive_indices = Vector{Int32}(findall(i -> cpu_alive[i] == 1.0f0, eachindex(cpu_alive)))
        modeled_indices = Vector{Int32}(findall(i -> cpu_alive[i] == 1.0f0 && cpu_age[i] >= larval_duration, eachindex(cpu_alive)))

        if !isempty(modeled_indices)
            # Update Population Metrics (summing abundances/biomass on CPU)
            model.abund[spec] = Int64(round(sum(Array(species_data.abundance[alive_indices]))))
            model.bioms[spec] = sum(Array(species_data.biomass_school[alive_indices]))
            
            print(length(alive_indices), " Agents | ")
            print(sum(model.abund[spec]), " Individuals | ")

            # Store current biomass for potential analysis callbacks
            species_data.biomass_init[alive_indices] = species_data.biomass_school[alive_indices] 

            # Movement, Foraging, and Energy
            print("behave | ")
            behavior(model, spec, modeled_indices, outputs)

            ind_temp = individual_temp!(model, spec)
            print("energy | ")
            energy!(model, spec, ind_temp, modeled_indices, outputs, current_date)

            # Fishing (Remains active during spinup to reach harvest equilibrium)
            print("fish | ")
            fishing!(model, spec, day_index, outputs)

            # --- NATURAL MORTALITY ---
            # Applies the instantaneous rate M to the individuals within agents to account for M not in the model.
            # This is called every timestep to provide constant exit pressure.
            print("mort | ")
            natural_mortality!(model, spec)
        end
        
        # Universal aging
        species_data.age .+= (sim.ΔT / 1440.0)
    end

    # --- 5. Trophic Dynamics ---
    print("resources | ")
    # Predators graze during spinup so resources stabilize at 'grazed' levels
    resource_predation!(model, outputs)
    resource_mortality!(model)
    resource_growth!(model, current_date)
   
    # --- 6. Results Export (Gated by Spin-up) ---
    if (model.iteration % model.output_dt == 0)
        if elapsed_minutes > model.spinup
            print("results | ")
            timestep_results(sim)
            resource_results(model, sim.run, model.iteration)
            fishery_results(sim)
            println("saved.")
        else
            println("spinup (no output save).")
        end
    else
        println("") # End console line
    end

    # --- 7. Final State Maintenance ---
    for spec in 1:species
        # Reset per-step consumption and activity buffers
        model.individuals.animals[spec].data.ration_biomass .= 0.0f0
        model.individuals.animals[spec].data.ration_energy .= 0.0f0
        model.individuals.animals[spec].data.active .= 0.0f0
    end
end