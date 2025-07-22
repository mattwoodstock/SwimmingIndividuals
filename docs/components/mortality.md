## Mortality & Natural Loss

The `mortality.jl` file is responsible for applying background mortality to the resource populations. This represents natural death from sources other than direct predation by the focal species agents (e.g., disease, predation by un-modeled organisms, old age).

### Resource Mortality

This system uses a high-performance GPU kernel to apply a continuous background mortality rate (`Z`) to every cell in the resource biomass grid at each timestep.

#### `resource_mortality!(model)`
This is the main launcher function for the resource mortality submodel. It prepares the necessary data (the resource trait parameters) in a GPU-compatible format and then calls the `resource_mortality_kernel!` to run in parallel across the entire resource grid.

#### `resource_mortality_kernel!(...)`
This GPU kernel is the core of the natural mortality calculation. Each thread is responsible for a single grid cell `(longitude, latitude, depth)` for a single resource species. It calculates the proportion of biomass that should survive the timestep based on the species' annual natural mortality rate (`Z`) and updates the biomass in that cell accordingly.

```julia
@kernel function resource_mortality_kernel!(biomass_grid, resource_trait, dt)
    lon, lat, depth, sp = @index(Global, NTuple)

    biomass = biomass_grid[lon, lat, depth, sp]
    if biomass > 0
        # Get the annual natural mortality rate for this species
        z_rate = resource_trait.Z[sp]
        # Convert the annual rate to a per-minute rate
        z_per_minute = z_rate / (365 * 1440)
        # Calculate the proportion of biomass surviving this timestep using the continuous mortality equation
        proportion_surviving = exp(-z_per_minute * dt)
        # Update the biomass in the grid cell
        @inbounds biomass_grid[lon, lat, depth, sp] = biomass * proportion_surviving
    end
end

function resource_mortality!(model::MarineModel)
    arch = model.arch
    # Convert the resource trait DataFrame to a GPU-compatible NamedTuple of arrays
    trait_gpu = (; (Symbol(c) => array_type(arch)(model.resource_trait[:, c]) for c in names(model.resource_trait))...)

    # Set up and launch the kernel to run over the entire 4D resource grid
    kernel! = resource_mortality_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    # Pass only the biomass grid to be modified
    kernel!(model.resources.biomass, trait_gpu, model.dt)
    KernelAbstractions.synchronize(device(arch))
end
