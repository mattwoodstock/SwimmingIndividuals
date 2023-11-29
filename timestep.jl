function TimeStep!(model::MarineModel, ΔT)

    # model.t = model.t+ΔT
    model.iteration = model.iteration + 1
    model.t = model.t + ΔT

    #Reset the day at midnight
    if model.t > 1440
        model.t = model.t - 1440
    end


    text1 = string(model.t)
    text2 = " : "
    print(text1*text2)
    #Vertical movement of animals
    for i in eachindex(fieldnames(typeof(model.individuals.animals))) #Loop each species
        #Create array to call species-specific traits
        name = Symbol("sp"*string(i))
        spec_array = getfield(model.individuals.animals, name)
        
        ## Link to iterating through structs
        #https://discourse.julialang.org/t/how-to-write-a-fast-loop-through-structure-fields/22535

        #Reset activity time for this time step
        spec_array.data.active_time .= 0

        # Animal movement
        if spec_array.p.DVM_trigger[2][i] > 0 #Species is a vertical migrator
            dvm_action(spec_array,i,model.t,ΔT)
        end

        if spec_array.p.Dive_Frequency[2][i] > 0 #Species will make dives
            dive_action(spec_array,i,ΔT)
        end
    end

    ##Check to see if species is a diver
    #divers = getfield(spec_array.p, :Dive_Frequency)
    #if divers[2][i] > 0 #Species is a vertical migrator
    #    spec_array = dive_action(spec_array,i,model.t)
    #end

    # Predation Procedure
    ## Calculate distance matrix between individuals
    distance_matrix = calculate_distances(model)
        
    for i in 1:model.n_species #Cycle through each predator species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)

        if spec_array1.data.feeding[1] == 1 ##All individuals of a species should have the same feeding schedule. Skip over if they are not eating.

                
            for j in 1:model.ninds[i] #Cycle through each individual predator
                if spec_array1.data.x[j] != -1 #Skip over dead animals
                    ## Find potential preys for predator
                    eat!(model,distance_matrix,i,j,spec_array1,ΔT)
                end
            end
        end
    end

    #Metabolism
    for i in eachindex(fieldnames(typeof(model.individuals.animals))) #Loop each species
        #Create array to call species-specific traits
        name = Symbol("sp"*string(i))
        spec_array = getfield(model.individuals.animals, name)
        
        for j in 1:model.ninds[i]
            metabolism(spec_array,j,ΔT)
            evacuate_gut!(spec_array,j,ΔT)
            if spec_array.data.energy[j] < 0 #Animal starves to death if its energy reserves fall below 0
                starvation!(spec_array,j)
            end

            ## Put individual at new lat,long for now to test processes
            #spec_array.data.x[j] = rand(-71:70)
            #spec_array.data.y[j] = rand(44:45)

        end
    end

    println(model.individuals.animals.sp1.data.z[1])

    #Replace individuals
    replace_individuals!(model)

    #Reset necessary components at the end of the day
    if (model.t >= 1440)
        reset!(model)
    end



    return nothing
end