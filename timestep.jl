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



        ##Check to see if species is a vertical migrator
        mig_rates = getfield(spec_array.p, :Mig_rate)

        if mig_rates[2][i] > 0 #Species is a vertical migrator
            spec_array = dvm_action(spec_array,i,model.t,ΔT)
        end

        test1 = string(model.individuals.animals.sp8.data.z[1])
        test2 = " | "

        print(test1 * test2)
        ##Check to see if species is a diver
        #divers = getfield(spec_array.p, :Dive_Frequency)
        #if divers[2][i] > 0 #Species is a vertical migrator
        #    spec_array = dive_action(spec_array,i,model.t)
        #end


    end

    return nothing
end