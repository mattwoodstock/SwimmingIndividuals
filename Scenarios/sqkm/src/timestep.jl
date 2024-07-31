function TimeStep!(sim)
    start = now()
    model = sim.model
    temp = sim.temp
    outputs = sim.outputs

    model.iteration += 1
    model.t += sim.ΔT
    model.t %= 1440  #Reset the day at midnight

    print(model.t)
    print(":   ")
    #Add the behavioral context for each species

    for (species_index, _) in enumerate(keys(model.individuals.animals))
        species = model.individuals.animals[species_index]
        t_resolution = species.p.t_resolution[2][species_index]
        if model.t % t_resolution == 0
    
            # Calculate n_ind outside the GPU loop
            n_ind = count(!iszero, species.data.ac) #Number of individuals per species that are active
            if n_ind > 0
                # Transfer necessary data to GPU
                ac = CUDA.copy(species.data.ac)
                ration = CUDA.copy(species.data.ration)
                energy = CUDA.copy(species.data.energy)

                @Threads.threads for j in 1:n_ind
                    j = j:j #This allows the GPU to operate without using scalars

                    if all(ac[j] .== 1.0) #Only runs if individual is alive
                        behavior(model, species_index, j, outputs)
    
                        consumed = model.individuals.animals[species_index].data.ration[j]

                        ind_temp = individual_temp(model, species_index, j, temp)

                        respire = respiration(model, species_index, j, ind_temp)

                        egest, excrete = excretion(model, species_index, consumed)

                        sda = specific_dynamic_action(consumed, egest)

                        growth(model, species_index, j, consumed, sda, respire, egest, excrete)

                        evacuate_gut(model, species_index, j, ind_temp)
    
                        if any(x -> x < 0,energy[j])
                            starvation(model,species, species_index, j, outputs)
                        end

                        if species.data.mature[j] == 1
                            reproduce(model,j,ind)
                        end
                    end
                end
            end
            model.bioms[species_index] = sum(model.individuals.animals[species_index].data.biomass)
        end  
    end
    println(model.abund)
    #Non-focal species processing
    for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
        pool_predation(model,pool_index)
    end

    if (model.t % model.output_dt == 0) & (model.iteration > model.spinup) #Only output results after the spinup is done.
        timestep_results(sim) #Assign and output
        CSV.write("Mortality Counts.csv",Tables.table(outputs.mortalities))
    end 

    if model.t == 0
        for (species_index, _) in enumerate(keys(model.individuals.animals))
            species = model.individuals.animals[species_index]
            species.data.ration .= 0 #Reset ration counter for all individuals
            species.data.consumed .= 0 #Reset consumption counter for all individuals
        end
    end
    pool_growth(model) #Grow pool individuals back to carrying capacity (initial biomass)
    println(now()-start)
end