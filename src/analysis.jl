function results(sim)
    model = sim.model
    outputs = sim.outputs

    # Timestep results
    #consumption_biomass(model,outputs)
    #production_biomass(model,outputs)
    pool_density_trend(model,outputs)
    biomasses(model,outputs)

    #Daily results
    if model.t == 0
        #daily_ration(model)
    end

    #End of simulation
    if model.iteration == sim.iterations
        mortality_rates(model,outputs)
        consumption(outputs)
        diet_composition(outputs)
        #q_b(model,outputs)
        #p_b(model,outputs)
        #Export files
    end
end

function timestep_results(model,outputs)
    ts = Int(model.iteration / model.output_dt)
    new_array = zeros(sum(model.ninds),3)


    ## Locations, energy, and ration
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
            if species_index == 1

                #new_array[1:model.ninds[1], 1] .= model.individuals.animals[species_index].data.z
                new_array[1:model.ninds[1], 1] .= model.individuals.animals[species_index].data.length
                new_array[1:model.ninds[1], 2] .= model.individuals.animals[species_index].data.ration
                new_array[1:model.ninds[1], 3] .= model.individuals.animals[species_index].data.energy
                #new_array[1:model.ninds[1], 4] .= model.individuals.animals[species_index].data.ac
            else
                #new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 1] .= model.individuals.animals[species_index].data.z
                new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 1] .= model.individuals.animals[species_index].data.length
                new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 2] .= model.individuals.animals[species_index].data.ration
                new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 3] .= model.individuals.animals[species_index].data.energy
                #new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 4] .= model.individuals.animals[species_index].data.ac
            end
    end

    ## Consumption
    #sum_consumption = sum(outputs.consumption,dims=6)
    #sum_behavior = sum(outputs.behavior,dims=3)

    if ts == 1
        outputs.timestep_array[:,:,1] = new_array
        #outputs.consumption_aggregate[:,:,:,:,:,1] = sum_consumption
        #outputs.behavior_aggregate[:,:,1] = sum_behavior
    else
        outputs.timestep_array = cat(outputs.timestep_array,new_array,dims=3)
        #outputs.consumption_aggregate = cat(outputs.consumption_aggregate,sum_consumption,dims=6)
        #outputs.behavior_aggregate = cat(outputs.behavior_aggregate,sum_behavior,dims=3)
    end

    #outputs.behavior = zeros(sum(model.ninds),4,1) #Subconsumption timestep
    #Reset to zero for next iteration

    filename = "timestep_results.jld"
    #For no behavior
    save(filename,"timestep",outputs.timestep_array)

    #With behavior included
    #save(filename,"timestep",outputs.timestep_array,"behavior",outputs.behavior_aggregate)
end

function biomasses(model, outputs)
    Nz = model.grid.Nz
    n_species = model.n_species
    iteration = model.iteration
    animals = model.individuals.animals

    for i in 1:n_species
        biomass_species = zeros(Float64, Nz)
        weight = animals[i].data.weight
        pool_z = animals[i].data.pool_z

        for depth in 1:Nz
            indices = findall(x -> x == depth, pool_z)
            biomass_species[depth] = sum(weight[indices])
        end

        outputs.biomass[iteration, :, i] = biomass_species
    end

    if iteration == sim.iterations
        # Sum by species across depths
        sum_array = sum(outputs.biomass, dims=2)[:, :, 1]

        # Output file
        CSV.write(joinpath("diags", "Biomasses.csv"), Tables.table(sum_array), writeheader=false)
    end
end

function pool_density_trend(model, outputs)
    # Vectorized operation to calculate the sum of pool densities
    for i in 1:model.n_pool
        outputs.pool_density[model.iteration, i] = sum(model.pools.pool[i].density.num)

        if model.iteration == sim.iterations
            CSV.write(joinpath("diags", "Pool_trend.csv"), Tables.table(outputs.pool_density),writeheader = false)
        end
    end
end

function consumption(outputs)
    sum_array = dropdims(sum(outputs.consumption, dims=(3,4)),dims=(3,4))
    CSV.write(joinpath("diags", "Consumption.csv"), Tables.table(sum_array),writeheader = false)
end



function q_b(model, outputs)
    # Preallocate memory for DataFrame
    q_b = DataFrame(Sp = Int[], n = Int[], QB = Float64[], SD = Float64[])
    # Compute mean and standard deviation outside the loop
    mean_consumption = mean(outputs.consumption_biomass, dims=2)[:]
    std_consumption = std(outputs.consumption_biomass, dims=2)[:]
    # Loop over species
    for i in 1:model.n_species
        # Push column vectors directly into the DataFrame
        push!(q_b, (Sp = i, n = model.ninds[i], QB = mean_consumption[i], SD = std_consumption[i]))
    end
    # Write DataFrame to CSV file
    CSV.write(joinpath("diags", "QB_results.csv"), q_b)
end



function p_b(model, outputs)
    # Preallocate memory for DataFrame
    p_b = DataFrame(Sp = Int[], n = Int[], PB = Float64[], SD = Float64[])
    # Compute mean and standard deviation outside the loop
    mean_pb = mean(outputs.production_biomass, dims=2)[:]
    std_pb = std(outputs.production_biomass, dims=2)[:]
    # Loop over species
    for i in 1:model.n_species
        # Push column vectors directly into the DataFrame
        push!(p_b, (Sp = i, n = model.ninds[i], PB = mean_pb[i], SD = std_pb[i]))
    end
    # Write DataFrame to CSV file
    CSV.write(joinpath("diags", "PB_results.csv"), p_b)
end


function daily_ration(model)
    dr = DataFrame(Sp = Int[], n = Int[], DR = Float64[], SD = Float64[])

    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
        avg = mean(model.individuals.animals[species_index].data.daily_ration)
        dev = std(model.individuals.animals[species_index].data.daily_ration)

        push!(dr, (; Sp = species_index, n = model.ninds[species_index], DR = avg, SD = dev))
    end
    CSV.write(joinpath("diags","Consumption results.csv"),dr)
end

function mortality_rates(model,outputs)
    morts = DataFrame(Sp = Int[], n = Int[], Predation = Float64[], Starvation = Float64[])
    time = model.iteration #Number of minutes in the simulation
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
        pred = outputs.mortalities[species_index,1]
        starv = outputs.mortalities[species_index,2]
        #Calculates deaths per minute
        push!(morts, (; Sp = species_index, n = model.ninds[species_index], Predation = pred, Starvation = starv))
    end
    CSV.write(joinpath("diags","Mortality results.csv"),morts)
end

function consumption_biomass(model,outputs)
    for i in 1:(model.n_species)
        outputs.consumption_biomass[i,model.iteration] = sum(outputs.foodweb.consumption[i,:,:,model.iteration])/sum(model.individuals.animals[i].data.weight)
    end
end

function production_biomass(model,outputs)
    for i in 1:(model.n_species)
        outputs.production_biomass[i,model.iteration] = outputs.production[i,model.iteration]/sum(model.individuals.animals[i].data.weight)
    end
end

function diet_composition(outputs)
    #Create Full Model Diet Matrix for Calculations
    sum_matrix = sum(outputs.consumption, dims=(3,4))
    # Normalize the values so that each column sums to one
    #diet_matrix = dropdims(sum_matrix ./ sum(sum_matrix, dims=1),dims=(3,4))
    diet_matrix = dropdims(sum_matrix ./ sum(sum_matrix, dims=2),dims=(3,4))
    diet_matrix[isnan.(diet_matrix)] .=0 #Replace NaNs that were created with 0s

    CSV.write(joinpath("diags","Diet Composition results.csv"),Tables.table(diet_matrix), writeheader=false)

end