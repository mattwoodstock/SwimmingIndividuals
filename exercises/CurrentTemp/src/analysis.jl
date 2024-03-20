function timestep_results(model,outputs)
    ts = Int(model.iteration / model.output_dt)
    new_array = zeros(sum(model.ninds),4)

    ## Locations, energy, and ration
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
            if species_index == 1

                new_array[1:model.ninds[1], 1] .= model.individuals.animals[species_index].data.energy
                new_array[1:model.ninds[1], 2] .= model.individuals.animals[species_index].data.x
                new_array[1:model.ninds[1], 3] .= model.individuals.animals[species_index].data.y
                new_array[1:model.ninds[1], 4] .= model.individuals.animals[species_index].data.z
            else
                new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 1] .= model.individuals.animals[species_index].data.energy
                new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 2] .= model.individuals.animals[species_index].data.x
                new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 3] .= model.individuals.animals[species_index].data.y
                new_array[(sum(model.ninds[1:(species_index-1)])+1):sum(model.ninds[1:(species_index)]), 4] .= model.individuals.animals[species_index].data.z
            end
    end

    ## Consumption
    sum_consumption = sum(outputs.consumption,dims=6)

    if ts == 1
        outputs.timestep_array[:,:,1] = new_array
        outputs.consumption_aggregate[:,:,:,:,:,1] = sum_consumption
    else
        outputs.timestep_array = cat(outputs.timestep_array,new_array,dims=3)
        outputs.consumption_aggregate = cat(outputs.consumption_aggregate,sum_consumption,dims=6)
    end

    filename = "timestep_results.jld"
    save(filename,"timestep",outputs.timestep_array,"consumption",outputs.consumption_aggregate)
end