function timestep_results(sim::MarineSimulation)
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
    rmr = []
    active = []
    landscape = []

    #Individual-scale
    for (species_index, animal_index) in enumerate(keys(model.individuals.animals))
        alive = findall(x -> x == 1.0, model.individuals.animals[species_index].data.ac) #Number of individuals per species that are active

        #Parameters necessary for population results
        pred_mort = findall(x -> x == 4.0, model.individuals.animals[species_index].data.behavior)
        starv_mort = findall(x -> x == 5.0, model.individuals.animals[species_index].data.behavior)
        population_array[species_index,1] = length(alive)
        population_array[species_index,2] = sum(model.individuals.animals[species_index].data.biomass[alive])
        population_array[species_index,3] = length(pred_mort)
        population_array[species_index,4] = length(starv_mort)
        population_array[species_index,5] = mean(model.individuals.animals[species_index].data.daily_ration[alive] ./ 5000 ./ model.individuals.animals[species_index].data.biomass[alive] .* 100)
        #

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
        append!(rmr,model.individuals.animals[species_index].data.rmr[alive])
        append!(active,model.individuals.animals[species_index].data.active[alive])
        append!(landscape,model.individuals.animals[species_index].data.landscape[alive])

    end
    individual_array = hcat(Sp,Ind,x,y,z,lengths,ration,energy,cost,behavior,rmr,active,landscape)
    column_names = ["Species", "Individual", "X", "Y", "Z", "Length", "Ration", "Energy", "Cost", "Behavior","RMR","Active","Landscape"]
    df = DataFrame(individual_array, Symbol.(column_names))

    if model.iteration == 1
        outputs.population_results[:,:,1] = population_array
    else
        outputs.population_results = cat(outputs.population_results,population_array,dims=3)
    end
    # Save the results periodically
    ts = Int(model.iteration)
    run = Int(sim.run)
    filename = "results/Individual/IndividualResults_$run-$ts.csv"
    CSV.write(filename, df)

    if ts == 24 #Only want these daily for this run.
        ## Uncomment for population results (Most can be aggregated from individual results and this code takes a long time)
        file_path = "results/Population/PopulationResults$run-$ts.h5"
        # Save arrays in the JLD2 file
        h5open(file_path, "w") do f
            write(f, "population", outputs.population_results)
        end

        #filename2 = "results/Population/PopulationResults_$run.jld"

        #save(filename2,"population",outputs.population_results)

        ## Uncomment for ecosystem results
        file_path = "results/Ecosystem/EcosystemResults$run-$ts.h5"
        # Save arrays in the JLD2 file
        h5open(file_path, "w") do f
            write(f, "diets", outputs.consumption)
            write(f, "encounters", outputs.encounters)
        end
    end
end
