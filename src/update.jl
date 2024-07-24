function update!(sim::MarineSimulation; time_offset = (vels = false, PARF = false, temp = false))
    #Output Lengths and any other starting information
    start_ind = 1
    for sp in 1:sim.model.n_species
        inds = length(sim.model.individuals.animals[sp].data.length)
        final_ind = start_ind + (inds-1)
        sim.outputs.lengths[start_ind:final_ind] = sim.model.individuals.animals[sp].data.length
        sim.outputs.weights[start_ind:final_ind] = sim.model.individuals.animals[sp].data.biomass

        start_ind = final_ind + 1
    end
    
    CSV.write("Lengths.csv",Tables.table(sim.outputs.lengths))
    CSV.write("Weights.csv",Tables.table(sim.outputs.weights))


    #Run model
    for i in 1:sim.iterations
        TimeStep!(sim)
    end
end