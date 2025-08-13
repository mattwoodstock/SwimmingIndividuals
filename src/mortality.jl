function resource_mortality!(model::MarineModel)
    arch = model.arch
    # Z rates from the trait table are treated as ANNUAL instantaneous rates
    annual_z_rates = model.resource_trait.Z
    
    # Convert annual instantaneous Z to a per-timestep proportional loss
    minutes_per_year = 365.0f0 * 1440.0f0
    per_minute_z = annual_z_rates ./ minutes_per_year
    
    # Proportion surviving one timestep = exp(-Z*dt)
    proportion_surviving = exp.(-per_minute_z .* model.dt)
    per_timestep_survival_rates = array_type(arch)(Float32.(proportion_surviving))

    # The kernel will now use the survival rates directly
    kernel! = resource_mortality_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    kernel!(model.resources.biomass, per_timestep_survival_rates)
    KernelAbstractions.synchronize(device(arch))
end

@kernel function resource_mortality_kernel!(biomass_grid, per_timestep_survival_rates)
    lon, lat, depth, sp = @index(Global, NTuple)
    biomass = biomass_grid[lon, lat, depth, sp]

    if biomass > 0
        # Apply the pre-calculated per-timestep survival proportion
        @inbounds biomass_grid[lon, lat, depth, sp] = biomass * per_timestep_survival_rates[sp]
    end
end