## Predation Components

The `predation.jl` file contains all the functions that govern how agents find, select, and consume prey. This is one of the most computationally intensive parts of the model and is divided into three distinct systems: a spatial index for fast searching, a detailed agent-on-agent predation model, and a background predation model for diffuse resources.

### 1. Spatial Indexing System

To make the process of finding nearby prey efficient, especially on the GPU, the model first builds a spatial index at the beginning of each timestep. This process sorts all agents based on their grid cell location.

#### `build_spatial_index!(model)`
This function orchestrates the creation of the spatial index. It first calls a GPU kernel (`assign_cell_ids_kernel!`) to assign a unique linear ID to each agent based on its `(pool_x, pool_y, pool_z)` coordinates. It then performs a high-speed sort on these IDs. Finally, it uses a parallel prefix scan algorithm (`thrust.searchsortedfirst` on the GPU) to find the start and end indices in the sorted array that correspond to each grid cell. This creates a set of pointers (`cell_starts`, `cell_ends`) that allow kernels to perform very fast, localized searches by only looking at agents within a specific grid cell.

```julia
# ===================================================================
# Spatial Indexing System
# ===================================================================

# Kernel to assign a cell_id to each agent based on its grid location
@kernel function assign_cell_ids_kernel!(agents, lonres, latres)
    i = @index(Global)
    @inbounds agents.cell_id[i] = get_cell_id(agents.pool_x[i], agents.pool_y[i], agents.pool_z[i], lonres, latres)
end

"""
Builds the spatial index for all species. This should be called once per timestep.
It sorts agents by their location for fast neighborhood searches.
"""
function build_spatial_index!(model::MarineModel)
    arch = model.arch
    g = model.depths.grid
    lonres = Int(g[g.Name .== "lonres", :Value][1])
    latres = Int(g[g.Name .== "latres", :Value][1])
    depthres = Int(g[g.Name .== "depthres", :Value][1])
    n_cells = lonres * latres * depthres

    for sp in 1:model.n_species
        agents = model.individuals.animals[sp].data
        n_agents = length(agents.x)
        if n_agents == 0; continue; end

        if length(agents.cell_starts) < n_cells
            @error "The pre-allocated 'cell_starts' array is too small for the grid size."
            return
        end

        kernel_assign = assign_cell_ids_kernel!(device(arch), 256, (n_agents,))
        kernel_assign(agents, lonres, latres)
        
        sortperm!(agents.sorted_id, agents.cell_id)
        
        if arch isa GPU
            sorted_cell_ids = agents.cell_id[agents.sorted_id]
            cell_range = array_type(arch)(1:n_cells)
            cell_starts_gpu = thrust.searchsortedfirst(sorted_cell_ids, cell_range)
            
            cell_ends_gpu = similar(cell_starts_gpu)
            if n_cells > 1
                cell_ends_gpu[1:end-1] .= cell_starts_gpu[2:end] .- 1
            end
            cell_ends_gpu[end] = n_agents
            
            copyto!(@view(agents.cell_starts[1:n_cells]), cell_starts_gpu)
            copyto!(@view(agents.cell_ends[1:n_cells]), cell_ends_gpu)
        else
            sorted_cell_ids_cpu = agents.cell_id[agents.sorted_id]
            cell_starts_cpu = ones(Int, n_cells)
            cell_ends_cpu = zeros(Int, n_cells)
            
            if !isempty(sorted_cell_ids_cpu)
                for i in 2:n_agents
                    if sorted_cell_ids_cpu[i] != sorted_cell_ids_cpu[i-1]
                        cell_starts_cpu[sorted_cell_ids_cpu[i]] = i
                        cell_ends_cpu[sorted_cell_ids_cpu[i-1]] = i - 1
                    end
                end
                cell_ends_cpu[sorted_cell_ids_cpu[end]] = n_agents
            end
            
            copyto!(@view(agents.cell_starts[1:n_cells]), cell_starts_cpu)
            copyto!(@view(agents.cell_ends[1:n_cells]), cell_ends_cpu)
        end
    end
end
```

### 2. Agent-on-Agent Predation ("Record-and-Resolve")
This system models direct predation between focal species agents and their consumption of the resource grid. It uses a three-step process to handle the complex, parallel interactions efficiently and safely.

#### calculate_distances_prey!(...)
This is the launcher for the first step. It prepares all the necessary data (agent data, prey data, resource grids, trait parameters) and then calls the find_best_prey_kernel! to run on the GPU.

The find_best_prey_kernel! is where each predator "records" its intent to eat. Each predator agent searches its local 3x3x3 grid neighborhood for all possible prey (both other agents and resource grid cells). It evaluates each potential prey based on its size preference and distance. The single best target (the closest, suitably-sized prey) is recorded for each predator. This process uses atomic operations (atomic_cas!) to ensure that multiple threads can safely write their findings without interfering with each other.

####  resolve_consumption!(...)
This function is the second step and acts as a "referee." It runs on the CPU. It gathers the "intent to eat" records from all predators and resolves any conflicts that arise when multiple predators have targeted the same prey. It ensures that the biomass of a single prey is not over-consumed, allocating the available biomass among the competing predators. The result of this process is a successful_ration for each predator.

####  apply_consumption!(...)
This is the final step. The launcher function deconstructs all the necessary data and passes it to the apply_consumption_kernel!. This GPU kernel runs over all predators and "applies" the successful_ration that was determined by the referee. It updates the state of both the predator (increasing its ration and gut fullness) and the prey (decreasing its biomass and abundance).

```julia
# ===================================================================
# Agent-on-Agent Predation System ("Record-and-Resolve")
# ===================================================================

@kernel function find_best_prey_kernel!(
    best_prey_dist, best_prey_idx, best_prey_sp, best_prey_type,
    pred_data, prey_data_all, resource_biomass_grid, resource_trait,
    pred_inds, grid_params, pred_params
)
    j_idx = @index(Global)
    pred_idx = pred_inds[j_idx]

    @inbounds if pred_data.alive[pred_idx] == 1.0
        my_x, my_y, my_z = pred_data.x[pred_idx], pred_data.y[pred_idx], pred_data.z[pred_idx]
        my_pool_x, my_pool_y, my_pool_z = pred_data.pool_x[pred_idx], pred_data.pool_y[pred_idx], pred_data.pool_z[pred_idx]
        min_size = pred_data.length[pred_idx] * pred_params.min_prey_ratio
        max_size = pred_data.length[pred_idx] * pred_params.max_prey_ratio
        detection_radius_sq = pred_data.vis_prey[pred_idx]^2

        for dz in -1:1, dy in -1:1, dx in -1:1
            search_x = my_pool_x + dx
            search_y = my_pool_y + dy
            search_z = my_pool_z + dz

            if 1 <= search_x <= grid_params.lonres && 1 <= search_y <= grid_params.latres && 1 <= search_z <= grid_params.depthres
                for prey_sp_idx in 1:length(prey_data_all)
                    prey_data = prey_data_all[prey_sp_idx]
                    for k in 1:length(prey_data.x)
                        @inbounds if prey_data.alive[k] == 1.0 && min_size <= prey_data.length[k] <= max_size
                            dist_sq = (my_x - prey_data.x[k])^2 + (my_y - prey_data.y[k])^2 + (my_z - prey_data.z[k])^2
                            if dist_sq <= detection_radius_sq && dist_sq < best_prey_dist[pred_idx]
                                old_dist = atomic_cas!(pointer(best_prey_dist, pred_idx), best_prey_dist[pred_idx], Float32(dist_sq))
                                if old_dist >= best_prey_dist[pred_idx]
                                    best_prey_idx[pred_idx] = k
                                    best_prey_sp[pred_idx] = prey_sp_idx
                                    best_prey_type[pred_idx] = 1
                                end
                            end
                        end
                    end
                end
                
                for res_sp in 1:size(resource_biomass_grid, 4)
                    biomass_density = resource_biomass_grid[search_x, search_y, search_z, res_sp]
                    if biomass_density > 0
                        res_min = resource_trait.Min_Size[res_sp]
                        res_max = resource_trait.Max_Size[res_sp]
                        μ, σ = lognormal_params_from_minmax(res_min, res_max)
                        mean_size = exp(μ + 0.5 * σ^2)
                        if min_size <= mean_size <= max_size
                            prey_grid_x = grid_params.lon_min + (search_x - 0.5) * grid_params.cell_size_deg
                            prey_grid_y = grid_params.lat_min + (search_y - 0.5) * grid_params.cell_size_deg
                            prey_grid_z = (search_z - 0.5) * grid_params.depth_res_m
                            dist_sq = (my_x - prey_grid_x)^2 + (my_y - prey_grid_y)^2 + (my_z - prey_grid_z)^2
                            if dist_sq <= detection_radius_sq && dist_sq < best_prey_dist[pred_idx]
                                linear_idx = search_x + (search_y-1)*grid_params.lonres + (search_z-1)*grid_params.lonres*grid_params.latres
                                old_dist = atomic_cas!(pointer(best_prey_dist, pred_idx), best_prey_dist[pred_idx], Float32(dist_sq))
                                if old_dist >= best_prey_dist[pred_idx]
                                    best_prey_idx[pred_idx] = linear_idx
                                    best_prey_sp[pred_idx] = res_sp
                                    best_prey_type[pred_idx] = 2
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function calculate_distances_prey!(model::MarineModel, sp::Int, inds::Vector{Int})
    arch = model.arch
    pred_data = model.individuals.animals[sp].data
    pred_params = (min_prey_ratio = model.individuals.animals[sp].p.Min_Prey[2][sp], max_prey_ratio = model.individuals.animals[sp].p.Max_Prey[2][sp])
    grid = model.depths.grid
    grid_params = (
        lonres = Int(grid[grid.Name .== "lonres", :Value][1]), latres = Int(grid[grid.Name .== "latres", :Value][1]),
        depthres = Int(grid[grid.Name .== "depthres", :Value][1]), lon_min = grid[grid.Name .== "xllcorner", :Value][1],
        lat_min = grid[grid.Name .== "yllcorner", :Value][1], cell_size_deg = grid[grid.Name .== "cellsize", :Value][1],
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / Int(grid[grid.Name .== "depthres", :Value][1])
    )
    trait_df = model.resource_trait
    resource_trait_gpu = (; (Symbol(c) => array_type(arch)(trait_df[:, c]) for c in names(trait_df))...)
    fill!(pred_data.best_prey_dist, Inf32); fill!(pred_data.best_prey_idx, 0);
    fill!(pred_data.best_prey_sp, 0); fill!(pred_data.best_prey_type, 0)
    prey_data_all = Tuple(animal.data for animal in model.individuals.animals)
    kernel! = find_best_prey_kernel!(device(arch), 256, (length(inds),))
    kernel!(
        pred_data.best_prey_dist, pred_data.best_prey_idx, pred_data.best_prey_sp, pred_data.best_prey_type,
        pred_data, prey_data_all, model.resources.biomass, resource_trait_gpu,
        array_type(arch)(inds), grid_params, pred_params
    )
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

function resolve_consumption!(model::MarineModel, sp::Int, to_eat::Vector{Int})
    pred_data = model.individuals.animals[sp].data
    best_prey_idx_cpu = Array(pred_data.best_prey_idx[to_eat]); best_prey_sp_cpu = Array(pred_data.best_prey_sp[to_eat]);
    best_prey_type_cpu = Array(pred_data.best_prey_type[to_eat]); pred_biomass_cpu = Array(pred_data.biomass_school[to_eat]);
    pred_gut_full_cpu = Array(pred_data.gut_fullness[to_eat]);
    prey_biomass_all_cpu = [Array(animal.data.biomass_school) for animal in model.individuals.animals]
    res_biomass_cpu = Array(model.resources.biomass)
    successful_rations_cpu = zeros(Float64, length(pred_data.x))
    prey_claimed = Dict{Tuple{Int, Int}, Float64}()
    for i in 1:length(to_eat)
        pred_idx = to_eat[i]; prey_idx = best_prey_idx_cpu[i]
        if prey_idx == 0; continue; end
        prey_sp = best_prey_sp_cpu[i]; prey_type = best_prey_type_cpu[i]
        prey_key = (prey_type, prey_idx)
        claimed_biomass = get(prey_claimed, prey_key, 0.0)
        total_biomass = (prey_type == 1) ? prey_biomass_all_cpu[prey_sp][prey_idx] : res_biomass_cpu[prey_idx]
        available_biomass = total_biomass - claimed_biomass
        if available_biomass <= 0; continue; end
        max_stomach = 0.2 * pred_biomass_cpu[i]
        current_stomach = pred_gut_full_cpu[i] * pred_biomass_cpu[i]
        stomach_space = max(0.0, max_stomach - current_stomach)
        ration = min(available_biomass, stomach_space)
        if ration > 0
            successful_rations_cpu[pred_idx] = ration
            prey_claimed[prey_key] = get(prey_claimed, prey_key, 0.0) + ration
        end
    end
    copyto!(@view(pred_data.successful_ration[1:end]), successful_rations_cpu)
end

@kernel function apply_consumption_kernel!(
    alive, best_prey_dist, best_prey_idx, best_prey_sp, best_prey_type,
    x, y, z, pool_x, pool_y, pool_z, length_arr, biomass_school, gut_fullness,
    ration, active, successful_ration,
    prey_data_all, resource_biomass_grid,
    pred_chars, time, grid_params, sp
)
    pred_idx = @index(Global)
    @inbounds if alive[pred_idx] == 1.0
        s_ration = successful_ration[pred_idx]
        if s_ration > 0.0
            dist = sqrt(best_prey_dist[pred_idx])
            time_left = time[pred_idx]
            swim_velo = pred_chars.Swim_velo[sp] * (length_arr[pred_idx] / 1000.0)
            time_to_prey = swim_velo > 0 ? dist / swim_velo : Inf32

            if time_to_prey < time_left
                time_left -= time_to_prey
                @atomic active[pred_idx] += time_to_prey / 60.0
                handling_time = pred_chars.Handling_Time[sp]
                num_can_handle = handling_time > 0 ? floor(Int, time_left / handling_time) : 0
                time_spent_handling = num_can_handle * handling_time
                @atomic ration[pred_idx] += s_ration
                @atomic gut_fullness[pred_idx] += s_ration / biomass_school[pred_idx]
                @atomic active[pred_idx] += time_spent_handling / 60.0
                time[pred_idx] -= (time_to_prey + time_spent_handling)
            else
                @atomic active[pred_idx] += time_left / 60.0
                time[pred_idx] = 0.0
            end
            successful_ration[pred_idx] = 0.0
        end
    end
end

function apply_consumption!(model::MarineModel, sp::Int, time::AbstractArray, outputs::MarineOutputs)
    arch = model.arch
    pred_data = model.individuals.animals[sp].data
    pred_chars_gpu = (; (k => array_type(arch)(v.second) for (k, v) in pairs(model.individuals.animals[sp].p))...)
    prey_data_all = Tuple(animal.data for animal in model.individuals.animals)
    grid = model.depths.grid
    grid_params = (
        lonres = Int(grid[grid.Name .== "lonres", :Value][1]), latres = Int(grid[grid.Name .== "latres", :Value][1]),
        depthres = Int(grid[grid.Name .== "depthres", :Value][1]), lon_min = grid[grid.Name .== "xllcorner", :Value][1],
        lat_min = grid[grid.Name .== "yllcorner", :Value][1], cell_size_deg = grid[grid.Name .== "cellsize", :Value][1],
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / Int(grid[grid.Name .== "depthres", :Value][1])
    )
    kernel! = apply_consumption_kernel!(device(arch), 256, (length(pred_data.x),))
    kernel!(
        pred_data.alive, pred_data.best_prey_dist, pred_data.best_prey_idx, 
        pred_data.best_prey_sp, pred_data.best_prey_type,
        pred_data.x, pred_data.y, pred_data.z, pred_data.pool_x, pred_data.pool_y, pred_data.pool_z,
        pred_data.length, pred_data.biomass_school, pred_data.gut_fullness,
        pred_data.ration, pred_data.active, pred_data.successful_ration,
        prey_data_all, model.resources.biomass,
        pred_chars_gpu, time, grid_params, sp
    )
    KernelAbstractions.synchronize(device(arch))
end
```

### 3. Background Resource Predation
This system models the mortality imposed on focal species by diffuse, un-modeled predators (e.g., zooplankton). It is a three-kernel process designed for the GPU.

#### resource_predation!(model, output)
This is the main launcher function. It first calls calculate_prey_biomass_kernel! to sum the biomass of all focal species agents into a 3D grid. It then calls calculate_mortality_rate_kernel!, which uses this prey grid and the resource predator grid (model.resources.biomass) to calculate a mortality rate in each cell based on a Holling Type II functional response. Finally, it launches a kernel for each prey species (apply_resource_mortality_kernel!) to apply this calculated mortality rate to the individual agents.

```julia
# ===================================================================
# Background Resource Predation System
# ===================================================================

@inline function resource_predation_mortality_device(
    prey_biomass::Float64, pred_biomass::Float64, pred_weight::Float64,
    daily_ration::Float64, handling_time::Float64, dt::Float64
)
    (prey_biomass <= 0.0 || pred_biomass <= 0.0) && return 0.0
    N = prey_biomass / 1000.0
    total_intake = (pred_biomass / 1000.0) * (daily_ration / 100.0)
    total_intake <= 0.0 && return 0.0
    denominator = N - total_intake * handling_time * N
    FR_day = (denominator <= 1e-9) ? total_intake : (total_intake / denominator * N) / (1.0 + (total_intake / denominator) * handling_time * N)
    FR_timestep = FR_day * (dt / 1440.0)
    mortality = (FR_timestep * 1000.0) / prey_biomass
    return clamp(mortality, 0.0, 1.0)
end

@kernel function calculate_prey_biomass_kernel!(prey_biomass_grid, animals_all)
    i = @index(Global)
    for sp in 1:length(animals_all)
        animal = animals_all[sp]
        if i <= length(animal.x) && animal.alive[i] == 1.0
            @atomic prey_biomass_grid[animal.pool_x[i], animal.pool_y[i], animal.pool_z[i], sp] += animal.biomass_school[i]
        end
    end
end

@kernel function calculate_mortality_rate_kernel!(mortality_rate_grid, pred_biomass_grid, prey_biomass_grid, resource_trait, dt, cell_area)
    x, y, z, r = @index(Global, NTuple)
    pred_biom = pred_biomass_grid[x, y, z, r] / cell_area
    if pred_biom > 0.0
        min_prey_ratio = resource_trait.Min_Prey[r]
        max_prey_ratio = resource_trait.Max_Prey[r]
        max_ingestion = resource_trait.Daily_Ration[r]
        handling_time = Float64(resource_trait.Handling_Time[r])
        max_pred_size = resource_trait.Max_Size[r]
        μ_pred, σ_pred = lognormal_params_from_maxsize(Int(max_pred_size))
        pred_length_mean = exp(μ_pred + 0.5 * σ_pred^2)
        pred_weight_mean = resource_trait.LWR_a[r] * (pred_length_mean / 10)^resource_trait.LWR_b[r]
        
        total_available_biomass = 0.0
        for sp in 1:size(prey_biomass_grid, 4)
            total_available_biomass += prey_biomass_grid[x, y, z, sp]
        end

        if total_available_biomass > 0.0
            for sp in 1:size(prey_biomass_grid, 4)
                prey_biom = prey_biomass_grid[x, y, z, sp] / cell_area
                if prey_biom > 0
                    prop_ingestion = (prey_biom / (total_available_biomass / cell_area)) * max_ingestion
                    mortality_rate = resource_predation_mortality_device(prey_biom, pred_biom, pred_weight_mean, prop_ingestion, handling_time, dt)
                    mortality_rate_grid[x, y, z, r, sp] = mortality_rate
                end
            end
        end
    end
end

@kernel function apply_resource_mortality_kernel!(animal_data, mortality_rate_grid, sp_idx)
    i = @index(Global)
    if i <= length(animal_data.x) && animal_data.alive[i] == 1.0
        x, y, z = animal_data.pool_x[i], animal_data.pool_y[i], animal_data.pool_z[i]
        total_mortality_rate = 0.0
        for r in 1:size(mortality_rate_grid, 4)
            total_mortality_rate += mortality_rate_grid[x, y, z, r, sp_idx]
        end
        if total_mortality_rate > 0
            total_mortality_rate = min(total_mortality_rate, 1.0)
            biomass_removed = animal_data.biomass_school[i] * total_mortality_rate
            @atomic animal_data.biomass_school[i] -= biomass_removed
            if animal_data.biomass_ind[i] > 0
                inds_removed = floor(Int, biomass_removed / animal_data.biomass_ind[i])
                @atomic animal_data.abundance[i] -= Float64(inds_removed)
            end
            if animal_data.abundance[i] <= 0; animal_data.alive[i] = 0.0; end
        end
    end
end

function resource_predation!(model::MarineModel, output::MarineOutputs)
    arch = model.arch
    g = model.depths.grid
    lonres = Int(g[g.Name .== "lonres", :Value][1])
    latres = Int(g[g.Name .== "latres", :Value][1])
    depthres = Int(g[g.Name .== "depthres", :Value][1])
    n_sp = model.n_species
    n_res = model.n_resource
    
    trait_df = model.resource_trait
    resource_trait_gpu = (; (Symbol(c) => array_type(arch)(trait_df[:, c]) for c in names(trait_df))...)

    prey_biomass_grid = array_type(arch)(zeros(Float64, lonres, latres, depthres, n_sp))
    mortality_rate_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res, n_sp))
    
    animals_all = Tuple(animal.data for animal in model.individuals.animals)
    cell_area = 1.0 # Placeholder

    max_len = 0
    if !isempty(animals_all)
        max_len = maximum(length(a.x) for a in animals_all if !isempty(a.x))
    end
    
    if max_len > 0
        kernel1 = calculate_prey_biomass_kernel!(device(arch), 256, (max_len,))
        kernel1(prey_biomass_grid, animals_all)
    end
    
    kernel2_ndrange = (lonres, latres, depthres, n_res)
    kernel2 = calculate_mortality_rate_kernel!(device(arch), (8,8,4,1), kernel2_ndrange)
    kernel2(mortality_rate_grid, model.resources.biomass, prey_biomass_grid, resource_trait_gpu, model.dt, cell_area)

    for sp_idx in 1:n_sp
        agents = model.individuals.animals[sp_idx].data
        if length(agents.x) > 0
            kernel3 = apply_resource_mortality_kernel!(device(arch), 256, (length(agents.x),))
            kernel3(agents, mortality_rate_grid, sp_idx)
        end
    end
    
    KernelAbstractions.synchronize(device(arch))
    return nothing
end
```