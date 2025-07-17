@kernel function resource_mortality_kernel!(biomass_grid, resource_trait, dt)
    lon, lat, depth, sp = @index(Global, NTuple)

    biomass = biomass_grid[lon, lat, depth, sp]
    if biomass > 0
        z_rate = resource_trait.Z[sp]
        z_per_minute = z_rate / (365 * 1440)
        proportion_surviving = exp(-z_per_minute * dt)
        @inbounds biomass_grid[lon, lat, depth, sp] = biomass * proportion_surviving
    end
end

function resource_mortality!(model::MarineModel)
    arch = model.arch
    trait_gpu = (; (Symbol(c) => array_type(arch)(model.resource_trait[:, c]) for c in names(model.resource_trait))...)

    kernel! = resource_mortality_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    # Pass only the biomass grid
    kernel!(model.resources.biomass, trait_gpu, model.dt)
    KernelAbstractions.synchronize(device(arch))
end