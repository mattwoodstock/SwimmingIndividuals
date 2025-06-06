function timestep_results(sim::MarineSimulation)
    model = sim.model
    outputs = sim.outputs
    population_array = zeros(model.n_species,5) #2D array to append to the Population-Scale Information
    # Save the results periodically
    ts = Int(model.iteration)
    run = Int(sim.run)
    month = model.environment.ts

    Sp = []
    Ind = []
    x = []
    y = []
    z = []
    lengths = []
    ration = []
    energy = []
    cost = []
    active = []
    abundance = []

    #Individual-scale
    for (species_index, animal_index) in enumerate(keys(model.individuals.animals))
        alive = findall(x -> x == 1.0, model.individuals.animals[species_index].data.alive) #Number of individuals per species that are active

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
        append!(active,model.individuals.animals[species_index].data.active[alive])
        append!(abundance,model.individuals.animals[species_index].data.abundance[alive])
    end

    individual_array = hcat(Sp,Ind,x,y,z,lengths,abundance,ration,energy,cost,active)
    column_names = ["Species", "Individual", "X", "Y", "Z", "Length","Abundance", "Ration", "Energy", "Cost","Active"]
    df = DataFrame(individual_array, Symbol.(column_names))

    #if model.iteration == 1
        #outputs.population_results[:,:,1] = population_array
    #else
        #outputs.population_results = cat(outputs.population_results,population_array,dims=3)
    #end

    filename = "results/Individual/IndividualResults_$run-$ts.csv"
    CSV.write(filename, df)

    ## Uncomment for population results (Most can be aggregated from individual results and this code takes a long time)
    #file_path = "results/Population/PopulationResults$run-$ts.h5"
    # Save arrays in the JLD2 file
    #h5open(file_path, "w") do f
        #write(f, "population", outputs.population_results)
    #end

    #filename2 = "results/Population/PopulationResults_$run.jld"

    #save(filename2,"population",outputs.population_results)

    ## Uncomment for ecosystem results
    #file_path = "results/Ecosystem/EcosystemResults$run-$ts.h5"
        # Save arrays in the JLD2 file
    #h5open(file_path, "w") do f
        #write(f, "diets", outputs.consumption)
        #write(f, "encounters", outputs.encounters)
    #end

    #spec = model.n_species+model.n_pool
    #outputs.consumption = zeros(model.n_species, (spec+1), model.grid.Nx,model.grid.Ny,model.grid.Nz) #Subconsumption timestep
    #outputs.encounters = zeros(spec, (spec+1), model.grid.Nx,model.grid.Ny,model.grid.Nz) 
end

function fishery_results(sim,fisheries)
    ts = Int(sim.model.iteration)
    run = Int(sim.run)
    name = []
    quotas = []
    catches_t = []
    catches_ind = []
    for fishery in fisheries
        push!(name,fishery.name)
        push!(quotas,fishery.quota)
        push!(catches_t,fishery.cumulative_catch)
        push!(catches_ind,fishery.cumulative_inds)
    end
    fish_array = hcat(name,quotas,catches_t,catches_ind)
    column_names = ["Name","Quota","Tonnage","Inds"]
    df = DataFrame(fish_array, Symbol.(column_names))

    filename = "results/Fishery/FisheryResults_$run-$ts.csv"
    CSV.write(filename, df)
end