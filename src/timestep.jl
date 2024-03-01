function TimeStep!(sim)
    model = sim.model
    temp = sim.temp
    outputs = sim.outputs

    model.iteration += 1
    model.t += sim.ΔT
    model.t %= 1440  #Reset the day at midnight

    print(model.iteration)
    print(":   ")

    counter_file = DataFrame(Value = [model.iteration]) #To keep track during cluster runs
    CSV.write("timestepping.csv",counter_file)
    
    #Add the behavioral context for each species
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
        species = model.individuals.animals[species_index]
        t_resolution = species.p.t_resolution[2][species_index]
        if model.iteration % t_resolution == 0
            fill!(species.data.active_time, 0) # Reset activity time for this time step

            n_ind = count(!iszero, model.individuals.animals[species_index].data.length) #Number of individuals per species
            for j in 1:n_ind
                behavior(model, species_index, j,outputs)

                consumed = model.individuals.animals[species_index].data.ration[j]

                ind_temp = individual_temp(model,species_index,j,temp)
                respire = respiration(model,species_index,j,ind_temp)
                egest, excrete = excretion(model,species_index,consumed)
                sda = specific_dynamic_action(consumed,egest)
                growth(model,species_index,j,consumed,sda,respire,egest,excrete)

                evacuate_gut(model,species_index,j,ind_temp)

                if species.data.energy[j] < 0 #Animal starves to death if its energy reserves fall below 0
                    starvation(species,species_index,j,outputs)
                end    

                timestep_results(model,outputs,respire,ind_temp,species_index,j)

            end
        end
    end

    #Non-focal species processing
    for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
        pool_predation(model,pool_index)
    end

    #Save results
    results(sim)
    #plotting(model,outputs)

    #Reset necessary components
    reset(model)
    pool_growth(model) #Grow pool individuals back to carrying capacity
end