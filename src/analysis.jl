function timestep_results(sim)
    model = sim.model
    outputs = sim.outputs
    population_array = zeros(model.n_species,5) #2D array to append to the Population-Scale Information

    Sp = []
    Ind = []
    x = []
    y = []
    z = []
    lengths = []
    ration = []
    energy = []
    cost = []
    behavior = []
    age = []
    rmr = []

    #Individual-scale
    for (species_index, animal_index) in enumerate(keys(model.individuals.animals))
        alive = findall(x -> x == 1.0, model.individuals.animals[species_index].data.ac) #Number of individuals per species that are active
        pred_mort = findall(x -> x == 4.0, model.individuals.animals[species_index].data.behavior)
        starv_mort = findall(x -> x == 5.0, model.individuals.animals[species_index].data.behavior)

        population_array[species_index,1] = length(alive)
        population_array[species_index,2] = sum(model.individuals.animals[species_index].data.biomass[alive])
        population_array[species_index,3] = length(pred_mort)
        population_array[species_index,4] = length(starv_mort)
        population_array[species_index,5] = mean(model.individuals.animals[species_index].data.daily_ration[alive] ./ 5000 ./ model.individuals.animals[species_index].data.biomass[alive] .* 100)

        specs = fill(species_index,length(alive))
        append!(Sp,specs)
        append!(Ind,alive)
        append!(x,model.individuals.animals[species_index].data.x[alive])
        append!(y,model.individuals.animals[species_index].data.y[alive])
        append!(z,model.individuals.animals[species_index].data.z[alive])
        append!(lengths,model.individuals.animals[species_index].data.length[alive])
        append!(ration,model.individuals.animals[species_index].data.ration[alive])
        append!(energy,model.individuals.animals[species_index].data.energy[alive])
        append!(cost,model.individuals.animals[species_index].data.cost[alive])
        append!(behavior,model.individuals.animals[species_index].data.behavior[alive])
        append!(age,model.individuals.animals[species_index].data.age[alive])
        append!(rmr,model.individuals.animals[species_index].data.rmr[alive])

    end
    individual_array = hcat(Sp,Ind,x,y,z,lengths,ration,energy,cost,behavior,age,rmr)
    column_names = ["Species", "Individual", "X", "Y", "Z", "Length", "Ration", "Energy", "Cost", "Behavior","Age","RMR"]
    df = DataFrame(individual_array, Symbol.(column_names))

    if model.iteration == 1
        outputs.population_results[:,:,1] = population_array
    else
        outputs.population_results = cat(outputs.population_results,population_array,dims=3)
    end
    # Save the results periodically
    ts = Int(model.iteration)
    run = Int(sim.run)
    filename = "Results2/Individual/IndividualResults_$run-$ts.csv"
    CSV.write(filename, df)
    filename2 = "Results2/Population/PopulationResults_$run.jld"
    save(filename2,"population",outputs.population_results)
    filename3 = "Results2/Ecosystem/EcosystemResults$run-$ts.jld"
    save(filename3,"ecosystem",outputs.consumption)
end
