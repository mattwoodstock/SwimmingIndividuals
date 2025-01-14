function generate_outputs(model,spec,iterations,output_dt)
    model.ninds = sum(length(model.individuals.animals[i].data) for i in 1:model.n_species)
    mortalities = zeros(model.n_species,2)
    lengths = zeros(model.ninds)
    weights = zeros(model.ninds)
    consumption = zeros(spec, (spec+1), model.grid.Nx,model.grid.Ny,model.grid.Nz) #Subconsumption timestep
    encounters = zeros(spec, (spec+1), model.grid.Nx,model.grid.Ny,model.grid.Nz) 
    population_results = Array{Float64,3}(undef,model.n_species,5,1)
    return MarineOutputs(mortalities,lengths,weights,consumption,encounters,population_results)
end