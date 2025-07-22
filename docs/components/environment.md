## Environmental Factors

The `environment.jl` file is the cornerstone of the model's physical world. It contains the functions responsible for loading environmental data from external files, calculating habitat suitability for each species, placing agents into the world, and allowing agents to sense their local conditions.

### 1. Environment Loading

The simulation begins by loading all necessary environmental data grids from a single NetCDF file (`environment.nc`).

#### `generate_environment!(arch, nc_file, plt_diags)`
This is the primary function for setting up the model's environment. It reads the specified NetCDF file, loads each variable (e.g., `temp`, `salinity`, `bathymetry`) into memory, and moves the data to the appropriate computational architecture (CPU or GPU). As a diagnostic step, it also generates and saves a plot of the surface layer for each variable to help the user verify that the data has been loaded correctly.

```julia
# ===================================================================
# Data Structures and Environment Loading
# ===================================================================

function generate_environment!(arch::Architecture, nc_file::String,plt_diags)
    ds = NCDataset(nc_file)
    envi_data = Dict{String, AbstractArray}()
    output_dir = "./results/Test/Environment"
    mkpath(output_dir)
    @info "Saving environment plots to: $output_dir"

    for var_name in keys(ds)
        if var_name in ["lon", "lat", "depth", "time"]
            continue
        end
        
        @info "Loading and plotting environmental variable: $var_name"
        
        data_cpu = ds[var_name]
        envi_data[var_name] = array_type(arch)(data_cpu)

        # Plotting logic
        if plt_diags == 1
            local surface_slice
            if ndims(data_cpu) >= 3
                if ndims(data_cpu) == 4
                    surface_slice = data_cpu[:, :, 1, 1]
                else # 3D
                    surface_slice = data_cpu[:, :, 1]
                end
            elseif ndims(data_cpu) == 2
                surface_slice = data_cpu
            else
                @warn "Skipping plot for variable '$var_name' with unsupported dimensions."
                continue
            end

            p = heatmap(
                surface_slice',
                title = "Surface Layer: $var_name (Month 1)",
                xlabel = "Longitude Index",
                ylabel = "Latitude Index",
                c = :viridis
            )
            savefig(p, joinpath(output_dir, "$(var_name)_surface.png"))
        end
    end

    close(ds)
    return MarineEnvironment(envi_data, 1)
end
```
### 2. Agent Temperature Sensing
This system allows each agent to determine the ambient water temperature at its specific location in the 3D grid. It uses a high-performance, two-kernel "Index-Gather" pattern to do this efficiently on the GPU.

#### individual_temp!(model, sp)
This is the main launcher function. It first calls calculate_temp_indices_kernel! to have each agent determine its linear index into the 4D temperature grid. It then calls gather_temps_kernel! to perform a highly efficient lookup using these pre-calculated indices.

```julia
# ===================================================================
# Agent Temperature Sensing System
# ===================================================================

@kernel function calculate_temp_indices_kernel!(
    temp_idx_out, alive, pool_x, pool_y, pool_z,
    envi_ts, lonres, latres, depthres
)
    ind = @index(Global)
    @inbounds if alive[ind] == 1.0
        x, y, z = pool_x[ind], pool_y[ind], pool_z[ind]
        if 1 <= x <= lonres && 1 <= y <= latres && 1 <= z <= depthres
            linear_idx = x + (y-1)*lonres + (z-1)*lonres*latres + (envi_ts-1)*lonres*latres*depthres
            temp_idx_out[ind] = linear_idx
        else
            temp_idx_out[ind] = 0
        end
    else
        temp_idx_out[ind] = 0
    end
end

@kernel function gather_temps_kernel!(temps_out, temp_grid_4d, temp_indices)
    ind = @index(Global)
    idx = temp_indices[ind]
    if idx > 0
        @inbounds temps_out[ind] = temp_grid_4d[idx]
    end
end

function individual_temp!(model::MarineModel, sp::Int)
    arch = model.arch
    envi = model.environment
    agent_data = model.individuals.animals[sp].data
    
    temp_grid_4d = envi.data["temp"]
    lonres, latres, depthres, _ = size(temp_grid_4d)

    kernel1 = calculate_temp_indices_kernel!(device(arch), 256, (length(agent_data.x),))
    kernel1(
        agent_data.temp_idx, agent_data.alive,
        agent_data.pool_x, agent_data.pool_y, agent_data.pool_z,
        envi.ts, lonres, latres, depthres
    )

    temps_out = array_type(arch)(zeros(eltype(temp_grid_4d), length(agent_data.x)))
    kernel2 = gather_temps_kernel!(device(arch), 256, (length(agent_data.x),))
    kernel2(temps_out, temp_grid_4d, agent_data.temp_idx)

    KernelAbstractions.synchronize(device(arch))
    return temps_out
end
```
### 3. Habitat Capacity and Agent Placement
This section contains the functions that determine where agents can live and how they are initially placed in the model world.

#### initial_habitat_capacity(...)
This function is a critical part of the model setup. It runs on the CPU using multi-threading. For each species, it loops through every grid cell and every month, calculating a habitat suitability index from 0 to 1. This index is based on comparing the environmental data for that cell to the species' unique preferences (defined in envi_pref.csv). The function uses a "land mask" derived from the temperature data to ensure that any cell corresponding to land is automatically assigned a capacity of zero, which is a crucial step for model stability.

#### initial_ind_placement(...)
This CPU-based function is used to find a starting location for a new agent. It takes the pre-calculated habitat capacity map for a given species and month and performs a weighted random selection to choose a grid cell, with a higher probability of selecting cells with better habitat. This ensures that agents are placed in ecologically plausible locations at the start of the simulation.

```julia
# ===================================================================
# Habitat Capacity and Agent Placement
# ===================================================================

function initial_habitat_capacity(envi::MarineEnvironment, n_spec::Int, n_resource::Int, files, arch::Architecture, plt_diags)
    prefs_df = CSV.read(files[files.File .== "envi_pref",:Destination][1], DataFrame)
    trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1], DataFrame))))
    resource = Dict(pairs(eachcol(CSV.read(files[files.File .== "resource_trait",:Destination][1], DataFrame))))

    spec_names = vcat(trait[:SpeciesLong][1:n_spec], resource[:SpeciesLong][1:n_resource])
    
    envi_data_cpu = Dict{String, Array}()
    for (name, grid) in envi.data
        envi_data_cpu[name] = Array(grid)
    end
    
    ref_var = envi_data_cpu["temp-surf"]
    lonres, latres, nmonths = size(ref_var)
    
    land_mask = .!ismissing.(ref_var[:,:,1])

    capacities_cpu = ones(Float64, lonres, latres, nmonths, n_spec + n_resource)

    Threads.@threads for i in 1:length(spec_names)
        sp_name = spec_names[i]
        sp_prefs = filter(row -> row.species == sp_name, prefs_df)
        
        if !isempty(sp_prefs)
            for month in 1:nmonths, lat in 1:latres, lon in 1:lonres
                
                if !land_mask[lon, lat]
                    capacities_cpu[lon, lat, month, i] = 0.0
                    continue
                end

                suitability_total = 1.0
                
                for pref_row in eachrow(sp_prefs)
                    var_name = pref_row.variable
                    if !haskey(envi_data_cpu, var_name); continue; end
                    
                    env_var_grid = envi_data_cpu[var_name]
                    val = (ndims(env_var_grid) == 3) ? env_var_grid[lon, lat, month] : env_var_grid[lon, lat]

                    if ismissing(val)
                        suitability_total = 0.0
                        break
                    end
                    
                    pref_min = pref_row.pref_min
                    opt_min = pref_row.opt_min
                    opt_max = pref_row.opt_max
                    pref_max = pref_row.pref_max

                    suitability_i = 0.0
                    if any(ismissing, [pref_min, opt_min, opt_max, pref_max])
                        suitability_i = 1.0 
                    else
                        if val >= opt_min && val <= opt_max
                            suitability_i = 1.0
                        elseif val > pref_min && val < opt_min
                            suitability_i = (val - pref_min) / (opt_min - pref_min)
                        elseif val > opt_max && val < pref_max
                            suitability_i = (pref_max - val) / (pref_max - opt_max)
                        end
                    end
                    
                    suitability_total *= suitability_i
                end
                
                capacities_cpu[lon, lat, month, i] = suitability_total
            end
        end
    end
    
    if plt_diags == 1
        output_dir = "./results/Test/Capacities"
        mkpath(output_dir)

        for i in 1:length(spec_names)
            sp_name = spec_names[i]
            for month in 1:nmonths
                capacity_slice = capacities_cpu[:, :, month, i]
                p = heatmap(
                    capacity_slice', 
                    title = "Habitat Capacity: $sp_name - Month $month",
                    xlabel = "Longitude Index",
                    ylabel = "Latitude Index",
                    c = :viridis,
                    clims = (0, 1)
                )
                savefig(p, joinpath(output_dir, "$(sp_name)_month_$(month)_capacity.png"))
            end
        end
        @info "Habitat capacity maps exported to $output_dir"
    end

    return array_type(arch)(capacities_cpu)
end

function initial_ind_placement(df_cpu, sp, grid, n_selections, month, land_mask)
    latmax = grid[grid.Name .== "yulcorner", :Value][1]
    lonmin = grid[grid.Name .== "xllcorner", :Value][1]
    cell_size = grid[grid.Name .== "cellsize", :Value][1]

    lonres, latres = size(land_mask)

    capacity_slice = @view df_cpu[:, :, month, sp]
    
    valid_cells = DataFrame(x=Int[], y=Int[], value=Float64[])
    for lon in 1:lonres, lat in 1:latres
        if land_mask[lon, lat] && capacity_slice[lon, lat] > 0
            push!(valid_cells, (x=lon, y=lat, value=capacity_slice[lon, lat]))
        end
    end

    if nrow(valid_cells) > 0
        sort!(valid_cells, :value, rev=true)
        cumvals = cumsum(valid_cells.value)
        total = cumvals[end]

        x_values, y_values = Int[], Int[]
        for _ in 1:n_selections
            r = (rand()^2) * total
            selected_idx = findfirst(cumvals .>= r)
            if selected_idx === nothing; selected_idx = 1; end
            push!(x_values, valid_cells.x[selected_idx])
            push!(y_values, valid_cells.y[selected_idx])
        end

        actual_x = lonmin .+ (x_values .- 1) .* cell_size .+ rand(n_selections) .* cell_size
        actual_y = latmax .- (y_values .- 1) .* cell_size .- rand(n_selections) .* cell_size

        return (lons=actual_x, lats=actual_y, grid_x=x_values, grid_y=y_values)
    else
        @warn "No valid water-based habitat for species $sp in month $month. Placing randomly."
        x_values = rand(1:lonres, n_selections)
        y_values = rand(1:latres, n_selections)
        actual_x = lonmin .+ (x_values .- 1) .* cell_size .+ rand(n_selections) .* cell_size
        actual_y = latmax .- (y_values .- 1) .* cell_size .- rand(n_selections) .* cell_size
        return (lons=actual_x, lats=actual_y, grid_x=x_values, grid_y=y_values)
    end
end
```