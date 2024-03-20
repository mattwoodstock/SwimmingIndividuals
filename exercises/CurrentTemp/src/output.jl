function generate_outputs(model,spec,iterations,output_dt)

    #depth = zeros(iterations)
    #trophic_level = zeros(model.n_species+model.n_pool)
    #daily_ration = copy(trophic_level)
    mortalities = zeros(model.n_species,2)
    biomass = zeros(iterations,model.grid.Nz,model.n_species)
    consumption = zeros(spec, spec, model.grid.Nx,model.grid.Ny,model.grid.Nz,output_dt) #Subconsumption timestep
    consumption_aggregate = zeros(spec, spec, model.grid.Nx,model.grid.Ny,model.grid.Nz,1)
    #consumption_biomass = zeros(model.n_species,iterations)
    production = zeros(iterations,model.n_species)
    #production_biomass = zeros(model.n_species,iterations)
    pool_density = zeros(iterations,model.n_pool)

    timestep_array = Array{Float64,3}(undef,sum(model.ninds),4,1)

    #daily_array = Array{Float64,4}(undef,sum(model.ninds),4,1). Fix later

    return MarineOutputs(mortalities,biomass,consumption,consumption_aggregate,production,pool_density,timestep_array)
end