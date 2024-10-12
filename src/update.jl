function update!(sim::MarineSimulation; time_offset = (vels = false, PARF = false, temp = false))
    #Run model
    for i in 1:sim.iterations
        TimeStep!(sim)

        # Open a text file to write the profile results
        #open("profile_output.txt", "w") do file
        #    Profile.print(file)
        #end
        #ProfileView.view()
    end
end

function reset_run(sim::MarineSimulation)
    model = sim.model
    
    for (species_index, _) in enumerate(keys(model.individuals.animals))
        println(species_index)
        println(model.individuals.animals[species_index].data.x)
        model.individuals.animals[species_index].data.x = []
        model.individuals.animals[species_index].p = NamedTuple()
    end
    
    for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
        model.pools.pool[pool_index].data = NamedTuple()
        model.pools.pool[pool_index].characters = NamedTuple()
    end
end