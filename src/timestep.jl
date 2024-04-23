function TimeStep!(sim)
    model = sim.model
    temp = sim.temp
    outputs = sim.outputs

    model.iteration += 1
    model.t += sim.ΔT
    model.t %= 1440  #Reset the day at midnight

    print(model.iteration)
    print(":   ")
    start1 = now()
    #Add the behavioral context for each species
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
        species = model.individuals.animals[species_index]
        t_resolution = species.p.t_resolution[2][species_index]
        if model.iteration % t_resolution == 0
            fill!(species.data.active_time, 0) # Reset activity time for this time step

            n_ind = count(!iszero, model.individuals.animals[species_index].data.length) #Number of individuals per species
            if n_ind > 0
                @Threads.threads for j in 1:n_ind
                    if model.individuals.animals[species_index].data.ac[j] == 1.0
                        behavior(model, species_index, j,outputs)

                        consumed = species.data.ration[j]

                        ind_temp = individual_temp(model,species_index,j,temp)
                        respire = respiration(model,species_index,j,ind_temp)
                        egest, excrete = excretion(model,species_index,consumed)
                        sda = specific_dynamic_action(consumed,egest)
                        growth(model,species_index,j,consumed,sda,respire,egest,excrete)

                        evacuate_gut(model,species_index,j,ind_temp)

                        if (species.data.energy[j] < 0) & (model.iteration > model.spinup) #Animal starves to death if its energy reserves fall below 0
                            starvation(species,species_index,j,outputs)
                        end

                        #if model.t == model.spinup #Reset energy budget once the real model starts
                        #    species.data.energy = species.data.weight[j] * species.p.energy_density[2][species_index]* 0.2
                        #end
                    end
                end
            end
        end

        model.individuals.animals[species_index].data.ration .= 0 #Reset ration counter for all individuals
    end

    #Non-focal species processing
    for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
        pool_predation(model,pool_index)
    end

    if (model.iteration % model.output_dt == 0) & (model.iteration > model.spinup) #Only output results after the spinup is done.
        timestep_results(model,outputs) #Assign and output
        CSV.write("Mortality Counts.csv",Tables.table(outputs.mortalities))
    end 


    #Save results
    results(sim)
    #plotting(model,outputs)
    #Reset necessary components
    reset(model)
    pool_growth(model) #Grow pool individuals back to carrying capacity
end