## The Simulation Timestep

The `timestep.jl` file contains the `TimeStep!` function, which is the heart of the simulation's engine. This function is called once for every timestep and is responsible for orchestrating the entire sequence of events, from updating the environment to executing agent behaviors and saving the results.

### Overview of `TimeStep!`

The function acts as a master conductor, ensuring that all sub-models are called in the correct logical order. At each step, it performs the following key tasks:

1.  **Advance Time:** It increments the model's internal clock and calculates the current month and day of the year.
2.  **Update Environment:** It checks if a new month has begun. If so, it calls `move_resources!` to redistribute the resource grids according to the new month's habitat capacity. It also handles annual resets for fishery quotas.
3.  **Process Focal Species:** It loops through each focal species and, for every living agent, executes the core sub-models in sequence:
    * `behavior()`: Determines the agent's actions for the timestep.
    * `individual_temp!`: Allows the agent to sense its local temperature.
    * `energy!`: Calculates the agent's full bioenergetics budget.
    * `fishing!`: Applies mortality from any active fisheries.
    * The agent's age is then incremented.
4.  **Process Resources:** It calls the functions that apply background predation, logistic growth, and natural mortality to the resource grids.
5.  **Save Outputs:** It checks if the current timestep is a designated output step and, if so, calls the `timestep_results` and `fishery_results` functions to save the model state.
6.  **Reset Accumulators:** It resets any per-timestep variables (like `ration` and `active` time) to zero in preparation for the next step.

### Full Code Example

```julia
function TimeStep!(sim::MarineSimulation)
    # Get references to the main components of the simulation
    model = sim.model
    envi = model.environment
    fisheries = model.fishing
    outputs = sim.outputs
    species::Int64 = model.n_species
    arch = model.arch

    # Advance the model's internal clock
    model.iteration += 1
    model.t = (model.t + sim.ΔT) % 1440 # Time of day in minutes, resets at midnight

    # --- Update Date and Season ---
    # Calculate the current calendar day and month from the iteration number
    origin = DateTime(2025, 1, 1, 0, 0)
    elapsed_minutes = (model.iteration - 1) * sim.ΔT
    current_datetime = origin + Minute(round(Int, elapsed_minutes))
    current_date = Date(current_datetime)
    month_index = month(current_date)
    day_index = dayofyear(current_date)

    # If the month has changed, redistribute the resource grids
    if month_index != envi.ts
        move_resources!(model, month_index)
    end

    # At the start of a new year, reset fishery quotas
    if day_index == 1
        (fishery -> (fishery.cumulative_catch = 0; fishery.cumulative_inds = 0)).(fisheries)
    end

    # Update the environment's current month
    envi.ts = month_index

    # Print the current date to the console for user feedback
    print(current_date)
    print(": ")
    
    # --- Main Agent Processing Loop ---
    # This loop iterates over each focal species defined in the model
    for spec in 1:species
        species_data = model.individuals.animals[spec].data
        species_chars = model.individuals.animals[spec].p

        # Find all living agents for the current species
        # This requires a copy from GPU to CPU to use `findall`
        cpu_alive = Array(species_data.alive)
        living::Vector{Int64} = findall(x -> x == 1, cpu_alive)
        
        if !isempty(living)
            # --- Calculate and print population summary statistics ---
            model.abund[spec] = sum(Array(species_data.abundance[living]))
            model.bioms[spec] = sum(Array(species_data.biomass_school[living]))
            print(length(living))
            print("  Abundance: ")
            print(model.abund[spec])
            print("  Mean Length: ")
            println(mean(Array(species_data.length[living])))

            # --- Execute Agent Sub-models ---
            print("behave | ")
            behavior(model, spec, living, outputs)

            ind_temp = individual_temp!(model, spec)

            print("energy | ")
            energy!(model, spec, ind_temp, living)

            print("fish | ")
            if (model.iteration > model.spinup)
                fishing!(model, spec, day_index, outputs)
            end
        end
        
        # Age all individuals of the species by one timestep
        species_data.age .+= (model.dt / 1440)
    end

    print("resources | ")
    # --- Update Resource Grids ---
    resource_predation!(model, outputs)
    resource_growth!(model)
    resource_mortality!(model)
    
    println("results | ")

    # --- Save Model Output ---
    # Check if the current timestep is an output step
    if (model.t % model.output_dt == 0)
        timestep_results(sim)
        if (model.iteration > model.spinup)
            fishery_results(sim)
        end
    end

    # --- Reset Per-Timestep Accumulators ---
    # Reset variables like ration and active time to zero for all agents
    for spec in 1:species
        model.individuals.animals[spec].data.ration .= 0.0
        model.individuals.animals[spec].data.active .= 0.0
    end

    # Annual reset for fisheries
    if day_index == 365
        (fishery -> (fishery.cumulative_catch = 0; fishery.cumulative_inds = 0)).(fisheries)
    end
end
