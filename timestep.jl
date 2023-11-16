function TimeStep!(model::MarineModel, ΔT)

    # model.t = model.t+ΔT
    model.iteration = model.iteration + 1
    model.t = model.iteration * ΔT 

    #Vertical movement of animals
    for i in eachindex(fieldnames(typeof(model.individuals.animals))) #Loop each species
        #Create array to call species-specific traits
        name = Symbol("sp"*string(i))
        spec_array = getfield(model.individuals.animals, name)
        
        ## Link to iterating through structs
        #https://discourse.julialang.org/t/how-to-write-a-fast-loop-through-structure-fields/22535


        # Animal movement
        ##Check to see if species is a vertical migrator
        mig_rates = getfield(spec_array.p, :Mig_rate)

        if mig_rates[2][i] > 0 #Species is a vertical migrator
            spec_array = dvm_action(spec_array,i,model.t,ΔT)
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
        
    eat!(model,distance_matrix, ΔT)

    #Metabolism
    for i in eachindex(fieldnames(typeof(model.individuals.animals))) #Loop each species
        #Create array to call species-specific traits
        name = Symbol("sp"*string(i))
        spec_array = getfield(model.individuals.animals, name)
        
        for j in 1:model.ninds[i]
            metabolism!(spec_array,j)
            evacuate_gut!(spec_array.data,j,ΔT)
        end
    end

    println(model.individuals.animals.sp1.data.gut_fullness[1])

        
    #Replace individuals
    #Reset things
    #Number of individuals in @model.inds. Currently, this will not change

    return nothing
end