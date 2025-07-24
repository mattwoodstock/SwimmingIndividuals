@kernel function assign_cell_ids_kernel!(agents, lonres, latres)
    i = @index(Global)
    @inbounds agents.cell_id[i] = get_cell_id(agents.pool_x[i], agents.pool_y[i], agents.pool_z[i], lonres, latres)
end

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


# ===================================================================
# High-Performance Predation System
# ===================================================================

# --- STEP 1: Find Best Prey ---

@kernel function find_best_prey_kernel!(
    best_prey_dist, best_prey_idx, best_prey_sp, best_prey_type,
    pred_data, prey_data_all, resource_biomass_grid, resource_trait,
    pred_inds, grid_params, pred_params,
    num_prey_species  # <-- pass it in
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

                # Search Focal Species Prey (num_prey_species passed in)
                for prey_sp_idx in 1:num_prey_species
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
                
                # Search Resource Grid
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

@generated function get_prey(prey_data_all, ::Val{i}) where {i}
    :(getfield(prey_data_all, $(QuoteNode(Symbol("sp$i")))))
end

function calculate_distances_prey!(model::MarineModel, sp::Int, inds::CuArray{Int32})
    arch = model.arch
    pred_data = model.individuals.animals[sp].data

    # Extract predator parameters
    pred_params = (
        min_prey_ratio = Float32(model.individuals.animals[sp].p.Min_Prey[2][sp]),
        max_prey_ratio = Float32(model.individuals.animals[sp].p.Max_Prey[2][sp])
    )

    # Extract and move prey data (as tuple of device structs)
    prey_data_all = Tuple(animal.data for animal in model.individuals.animals)
    num_prey_species = length(prey_data_all)

    # Extract grid parameters and make GPU-safe
    grid = model.depths.grid
    grid_params = (
        lonres = Int32(grid[grid.Name .== "lonres", :Value][1]),
        latres = Int32(grid[grid.Name .== "latres", :Value][1]),
        depthres = Int32(grid[grid.Name .== "depthres", :Value][1]),
        lon_min = Float32(grid[grid.Name .== "xllcorner", :Value][1]),
        lat_min = Float32(grid[grid.Name .== "yllcorner", :Value][1]),
        cell_size_deg = Float32(grid[grid.Name .== "cellsize", :Value][1]),
        depth_res_m = Float32(grid[grid.Name .== "depthmax", :Value][1]) / Float32(grid[grid.Name .== "depthres", :Value][1])
    )

    # Resource traits: convert to GPU-compatible NamedTuple of CuArrays
    trait_df = model.resource_trait
    resource_trait_gpu = (; (Symbol(c) => CuArray(trait_df[:, c]) for c in names(trait_df))...)

    # Ensure GPU-compatible initialization
    CUDA.fill!(pred_data.best_prey_dist, Inf32)
    CUDA.fill!(pred_data.best_prey_idx, 0)
    CUDA.fill!(pred_data.best_prey_sp, 0)
    CUDA.fill!(pred_data.best_prey_type, 0)


    kernel! = find_best_prey_kernel!(device(arch), 256, (length(inds),))
    kernel!(
        pred_data.best_prey_dist, pred_data.best_prey_idx, pred_data.best_prey_sp, pred_data.best_prey_type,
        pred_data, prey_data_all, model.resources.biomass, resource_trait_gpu,
        array_type(arch)(inds), grid_params, pred_params,
        num_prey_species
    )
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

# --- STEP 2: Resolve Consumption ---
function flatten_prey_biomass(model)
    num_prey = length(model.individuals.animals)
    prey_biomass_arrays = [model.individuals.animals[i].data.biomass_school for i in 1:num_prey]

    lengths = Int32.(map(length, prey_biomass_arrays))
    offsets = zeros(Int32, num_prey + 1)
    for i in 1:num_prey
        offsets[i+1] = offsets[i] + lengths[i]
    end
    total_len = offsets[end]

    # Create a single large CuArray on GPU to hold all biomass data
    prey_biomass_flat = CUDA.zeros(Float32, total_len)

    # Copy each prey biomass array into the flattened array at the right offset
    for i in 1:num_prey
        start_idx = offsets[i] + 1
        end_idx = offsets[i+1]
        # Copy data to GPU
        prey_biomass_flat[start_idx:end_idx] .= CuArray(prey_biomass_arrays[i])
    end

    return prey_biomass_flat, CuArray(offsets)
end


@kernel function resolve_consumption_kernel!(
    best_prey_idx::AbstractVector{Int64},
    best_prey_sp::AbstractVector{Int64},
    best_prey_type::AbstractVector{Int64},
    biomass_school::AbstractVector{Float32},
    gut_fullness::AbstractVector{Float32},
    successful_ration::AbstractVector{Float32},
    inds::AbstractVector{Int32},

    prey_biomass_flat::AbstractVector{Float32},
    prey_offsets::AbstractVector{Int32},

    resource_biomass_grid::AbstractArray{Float32,4}
)

    tid = @index(Global)
    if tid <= length(inds)
        pred_idx = inds[tid]

        prey_type = best_prey_type[pred_idx]
        prey_idx = best_prey_idx[pred_idx]

        if prey_idx == 0
            successful_ration[pred_idx] = 0f0
        else
            pred_biomass = biomass_school[pred_idx]
            pred_gut = gut_fullness[pred_idx]
            max_fullness = 0.2f0 * pred_biomass
            current_stomach = pred_gut * pred_biomass
            stomach_space = max(0f0, max_fullness - current_stomach)

            available_biomass = 0f0

            if prey_type == 1
                prey_sp = best_prey_sp[pred_idx]
                # Calculate flat index
                offset = prey_offsets[prey_sp]
                idx_in_flat = offset + prey_idx
                if 1 <= idx_in_flat <= length(prey_biomass_flat)
                    available_biomass = prey_biomass_flat[idx_in_flat]
                else
                    available_biomass = 0f0
                end

            elseif prey_type == 2
                res_sp = best_prey_sp[pred_idx]
                X = size(resource_biomass_grid, 1)
                Y = size(resource_biomass_grid, 2)
                Z = size(resource_biomass_grid, 3)
                linear_3d = prey_idx - 1
                x = linear_3d % X + 1
                y = (linear_3d ÷ X) % Y + 1
                z = (linear_3d ÷ (X*Y)) + 1
                available_biomass = resource_biomass_grid[x, y, z, res_sp]
            end

            successful_ration[pred_idx] = min(stomach_space, available_biomass)
        end
    end
end

function resolve_consumption!(
    model::MarineModel,
    sp::Int64,
    inds::Vector{Int32}
)
    n = length(inds)
    if n == 0
        return
    end

    arch = model.arch
    pred_data = model.individuals.animals[sp].data

    inds_gpu = CuArray(Int32.(inds))

    prey_biomass_flat, prey_offsets = flatten_prey_biomass(model)

    resource_biomass_grid = model.resources.biomass  # CuArray 4D

    threads = 256
    blocks = cld(n, threads)

    kernel = resolve_consumption_kernel!(device(arch), threads, blocks)

    kernel(
        pred_data.best_prey_idx,
        pred_data.best_prey_sp,
        pred_data.best_prey_type,
        pred_data.biomass_school,
        pred_data.gut_fullness,
        pred_data.successful_ration,
        inds_gpu,
        prey_biomass_flat,
        prey_offsets,
        resource_biomass_grid
    )
    KernelAbstractions.synchronize(device(arch))
    return nothing
end


@kernel function apply_consumption_kernel!(
    alive, best_prey_dist, best_prey_idx, best_prey_sp, best_prey_type,
    x, y, z, length_arr, biomass_school, gut_fullness, ration, active, successful_ration,
    
    # Prey data arrays (example fields, add more as needed)
    prey_x, prey_y, prey_z, prey_biomass_school,
    
    # Resource biomass grid (assumed 4D CuDeviceArray)
    resource_biomass_grid,
    
    # Predator traits as scalars
    swim_velo::Float32,
    handling_time::Float32,
    
    # Time array
    time_array,
    
    # Grid parameters tuple with basic Int/Float32 fields
    grid_params,
    
    sp
)
    pred_idx = @index(Global)

    # Basic bounds check
    if pred_idx > length(alive)
    else

        @inbounds if alive[pred_idx] == 1.0
            s_ration = successful_ration[pred_idx]
            
            if s_ration > 0.0
                # 1. Movement logic
                dist = sqrt(best_prey_dist[pred_idx])
                time_left = time_array[pred_idx]
                
                # swim_velo scaled by length (example formula)
                swim_v = swim_velo * (length_arr[pred_idx] / 1000.0)
                time_to_prey = swim_v > 0f0 ? dist / swim_v : Inf32

                if time_to_prey < time_left
                    time_left -= time_to_prey
                    @atomic active[pred_idx] += time_to_prey / 60.0f0

                    # 2. Consumption logic
                    num_can_handle = handling_time > 0f0 ? floor(Int, time_left / handling_time) : 0
                    time_spent_handling = num_can_handle * handling_time

                    # Update predator ration and gut fullness atomically
                    @atomic ration[pred_idx] += s_ration
                    @atomic gut_fullness[pred_idx] += s_ration / biomass_school[pred_idx]

                    # 3. Reduce prey biomass/resource biomass accordingly

                    # Prey is agent or resource?
                    prey_type = best_prey_type[pred_idx]
                    prey_idx = best_prey_idx[pred_idx]

                    if prey_type == 1
                        # prey is agent
                        @atomic prey_biomass_school[prey_idx] -= s_ration
                        if prey_biomass_school[prey_idx] < 0f0
                            prey_biomass_school[prey_idx] = 0f0
                        end
                    elseif prey_type == 2
                        # prey is resource, reduce resource biomass grid
                        # Here, prey_idx is linear index of resource grid cell
                        # atomic subtraction to resource biomass grid at that index (first species dimension assumed)
                        @atomic resource_biomass_grid[prey_idx] -= s_ration
                        if resource_biomass_grid[prey_idx] < 0f0
                            resource_biomass_grid[prey_idx] = 0f0
                        end
                    end

                    # Update time and active time budget
                    @atomic active[pred_idx] += time_spent_handling / 60.0f0
                    time_array[pred_idx] -= (time_to_prey + time_spent_handling)
                else
                    # Prey not reachable, spend remaining time active
                    @atomic active[pred_idx] += time_left / 60.0f0
                    time_array[pred_idx] = 0f0
                end

                # Reset successful ration for next timestep
                successful_ration[pred_idx] = 0f0
            end
        end
    end
end

function apply_consumption!(model::MarineModel, sp::Int, time::CuArray{Float32}, outputs::MarineOutputs)
    arch = model.arch
    pred_data = model.individuals.animals[sp].data

    # Extract predator traits scalars for species sp
    swim_velo = Float32(model.individuals.animals[sp].p.Swim_velo[2][sp])
    handling_time = Float32(model.individuals.animals[sp].p.Handling_Time[2][sp])

    # Extract prey data arrays for all prey species (example only first species for simplicity)
    # You will need to extend this if you have multiple prey species
    prey = model.individuals.animals[1].data  # Example prey species index = 1

    # Assuming prey_biomass_school is a CuArray
    prey_x = prey.x
    prey_y = prey.y
    prey_z = prey.z
    prey_biomass_school = prey.biomass_school

    resource_biomass_grid = model.resources.biomass

    # Grid params (just forwarding)
    grid = model.depths.grid
    grid_params = (
        lonres = Int(grid[grid.Name .== "lonres", :Value][1]),
        latres = Int(grid[grid.Name .== "latres", :Value][1]),
        depthres = Int(grid[grid.Name .== "depthres", :Value][1]),
        lon_min = grid[grid.Name .== "xllcorner", :Value][1],
        lat_min = grid[grid.Name .== "yllcorner", :Value][1],
        cell_size_deg = grid[grid.Name .== "cellsize", :Value][1],
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / Int(grid[grid.Name .== "depthres", :Value][1])
    )

    n = length(pred_data.x)
    threads = 256
    blocks = cld(n, threads)

    kernel = apply_consumption_kernel!(device(arch), threads, blocks)

    kernel(
        pred_data.alive, pred_data.best_prey_dist, pred_data.best_prey_idx,
        pred_data.best_prey_sp, pred_data.best_prey_type,
        pred_data.x, pred_data.y, pred_data.z, pred_data.length,
        pred_data.biomass_school, pred_data.gut_fullness,
        pred_data.ration, pred_data.active, pred_data.successful_ration,
        prey_x, prey_y, prey_z, prey_biomass_school,
        resource_biomass_grid,
        swim_velo,
        handling_time,
        time,
        grid_params,
        sp
    )

    KernelAbstractions.synchronize(device(arch))
    return nothing
end



# ===================================================================
# Background Resource Predation System
# ===================================================================

@inline function resource_predation_mortality_device(
    prey_biomass::Float32, pred_biomass::Float32, pred_weight::Float32,
    daily_ration::Float32, handling_time::Float32, dt::Float32
)
    (prey_biomass <= 0.0f0 || pred_biomass <= 0.0f0) && return 0.0f0
    N = prey_biomass / 1000.0f0
    total_intake = (pred_biomass / 1000.0f0) * (daily_ration / 100.0f0)
    total_intake <= 0.0f0 && return 0.0f0
    denominator = N - total_intake * handling_time * N
    FR_day = (denominator <= 1e-9f0) ? total_intake : (total_intake / denominator * N) / (1.0f0 + (total_intake / denominator) * handling_time * N)
    FR_timestep = FR_day * (dt / 1440.0f0)
    mortality = (FR_timestep * 1000.0f0) / prey_biomass
    return clamp(mortality, 0.0f0, 1.0f0)
end

# -- Kernel 1: Calculate prey biomass grid
@kernel function calculate_prey_biomass_kernel!(prey_biomass_grid, animals_all)
    i = @index(Global)
    for sp in 1:length(animals_all)
        animal = animals_all[sp]
        if i <= length(animal.x) && animal.alive[i] == 1.0f0
            @atomic prey_biomass_grid[animal.pool_x[i], animal.pool_y[i], animal.pool_z[i], sp] += animal.biomass_school[i]
        end
    end
end

# -- Kernel 2: Calculate mortality rate grid using resource traits
@kernel function calculate_mortality_rate_kernel!(
    mortality_rate_grid,
    pred_biomass_grid,
    prey_biomass_grid,
    min_prey,
    max_prey,
    daily_ration,
    handling_time,
    max_size,
    lwr_a,
    lwr_b,
    mu_pred,
    sigma_pred,
    dt::Float32,
    cell_area::Float64
)
    x, y, z, r = @index(Global, NTuple)
    pred_biom = pred_biomass_grid[x, y, z, r] / cell_area

    n_prey_species = size(prey_biomass_grid, 4)

    if pred_biom > 0.0f0
        min_prey_ratio = min_prey[r]
        max_prey_ratio = max_prey[r]
        max_ingestion = daily_ration[r]
        h_time = handling_time[r]
        max_pred_size = max_size[r]

        μ_pred = mu_pred[r]
        σ_pred = sigma_pred[r]

        pred_length_mean = exp(μ_pred + 0.5f0 * σ_pred^2)
        pred_weight_mean = lwr_a[r] * (pred_length_mean / 10.0f0)^lwr_b[r]

        # Step 1: Calculate total available prey biomass within size limits
        total_available_biomass = 0.0f0

        for sp in 1:n_prey_species
            # Placeholder: estimate prey size for sp - you must replace with real prey sizes if available
            prey_size_sp = pred_weight_mean

            # Include only prey within predator's prey size range
            if prey_size_sp >= min_prey_ratio && prey_size_sp <= max_prey_ratio
                total_available_biomass += prey_biomass_grid[x, y, z, sp]
            end
        end

        if total_available_biomass > 0.0f0
            # Step 2: Type II functional response total consumption rate per predator biomass
            a = max_ingestion
            N = total_available_biomass / cell_area

            consumption_rate_per_biomass = (a * N) / (1.0f0 + a * h_time * N)
            total_consumption = consumption_rate_per_biomass * pred_biom

            # Step 3: Limit consumption by max ingestion
            max_consumption = max_ingestion * pred_biom
            total_consumption = min(total_consumption, max_consumption)

            # Step 4: Distribute consumption across prey species proportionally
            for sp in 1:n_prey_species
                prey_size_sp = pred_weight_mean
                if prey_size_sp >= min_prey_ratio && prey_size_sp <= max_prey_ratio
                    prey_biom = prey_biomass_grid[x, y, z, sp] / cell_area
                    if prey_biom > 0.0f0
                        prop = prey_biom / N
                        prey_consumed = total_consumption * prop
                        mortality_rate = (prey_consumed * dt) / (prey_biom * 1440.0f0)  # dt minutes -> fraction days
                        mortality_rate_grid[x, y, z, r, sp] = clamp(mortality_rate, 0.0f0, 1.0f0)
                    else
                        mortality_rate_grid[x, y, z, r, sp] = 0.0f0
                    end
                else
                    mortality_rate_grid[x, y, z, r, sp] = 0.0f0
                end
            end
        else
            # No prey available - zero mortality
            for sp in 1:n_prey_species
                mortality_rate_grid[x, y, z, r, sp] = 0.0f0
            end
        end
    else
        # No predator biomass - zero mortality
        for sp in 1:n_prey_species
            mortality_rate_grid[x, y, z, r, sp] = 0.0f0
        end
    end
end
# -- Kernel 3: Apply mortality to animals
@kernel function apply_resource_mortality_kernel!(animal_data, mortality_rate_grid, sp_idx)
    i = @index(Global)
    if i <= length(animal_data.x) && animal_data.alive[i] == 1.0f0
        x, y, z = animal_data.pool_x[i], animal_data.pool_y[i], animal_data.pool_z[i]
        total_mortality_rate = 0.0f0
        for r in 1:size(mortality_rate_grid, 4)
            total_mortality_rate += mortality_rate_grid[x, y, z, r, sp_idx]
        end
        if total_mortality_rate > 0.0f0
            total_mortality_rate = min(total_mortality_rate, 1.0f0)
            biomass_removed = animal_data.biomass_school[i] * total_mortality_rate
            @atomic animal_data.biomass_school[i] -= biomass_removed
            if animal_data.biomass_ind[i] > 0.0f0
                inds_removed = floor(Int32, biomass_removed / animal_data.biomass_ind[i])
                @atomic animal_data.abundance[i] -= Float32(inds_removed)
            end
            if animal_data.abundance[i] <= 0.0f0
                animal_data.alive[i] = 0.0f0
            end
        end
    end
end

# -- Main resource predation function
function resource_predation!(model::MarineModel, output::MarineOutputs)
    arch = model.arch
    g = model.depths.grid
    lonres = Int(g[g.Name .== "lonres", :Value][1])
    latres = Int(g[g.Name .== "latres", :Value][1])
    depthres = Int(g[g.Name .== "depthres", :Value][1])
    n_sp = model.n_species
    n_res = Int(model.n_resource)

    rt = model.resource_trait

    # Precompute μ and σ arrays for all resources on CPU
    μs_cpu = zeros(Float32, n_res)
    σs_cpu = zeros(Float32, n_res)
    for r in 1:n_res
        μ, σ = lognormal_params_from_minmax(rt.Min_Size[r], rt.Max_Size[r])
        μs_cpu[r] = Float32(μ)
        σs_cpu[r] = Float32(σ)
    end

    # Move resource trait arrays to GPU
    resource_trait_gpu = (
        Min_Prey = array_type(arch)(Float32.(rt.Min_Prey)),
        Max_Prey = array_type(arch)(Float32.(rt.Max_Prey)),
        Daily_Ration = array_type(arch)(Float32.(rt.Daily_Ration)),
        Handling_Time = array_type(arch)(Float32.(rt.Handling_Time)),
        Max_Size = array_type(arch)(Float32.(rt.Max_Size)),
        LWR_a = array_type(arch)(Float32.(rt.LWR_a)),
        LWR_b = array_type(arch)(Float32.(rt.LWR_b)),
        Mu_pred = array_type(arch)(μs_cpu),
        Sigma_pred = array_type(arch)(σs_cpu)
    )

    prey_biomass_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_sp))
    mortality_rate_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res, n_sp))

    animals_all = Tuple(animal.data for animal in model.individuals.animals)
    cell_area = 1.0 # adjust accordingly

    max_len = 0
    if !isempty(animals_all)
        max_len = maximum(length(a.x) for a in animals_all if !isempty(a.x))
    end

    # Kernel 1: Calculate prey biomass in grid cells
    if max_len > 0
        kernel1 = calculate_prey_biomass_kernel!(device(arch), 256, (max_len,))
        kernel1(prey_biomass_grid, animals_all)
    end

    # Kernel 2: Calculate mortality rates grid
    kernel2_ndrange = (lonres, latres, depthres, n_res)
    kernel2 = calculate_mortality_rate_kernel!(device(arch), (8,8,4,1), kernel2_ndrange)
    kernel2(
        mortality_rate_grid,
        model.resources.biomass,
        prey_biomass_grid,
        resource_trait_gpu.Min_Prey,
        resource_trait_gpu.Max_Prey,
        resource_trait_gpu.Daily_Ration,
        resource_trait_gpu.Handling_Time,
        resource_trait_gpu.Max_Size,
        resource_trait_gpu.LWR_a,
        resource_trait_gpu.LWR_b,
        resource_trait_gpu.Mu_pred,
        resource_trait_gpu.Sigma_pred,
        Float32(model.dt),
        cell_area
    )

    # Kernel 3: Apply mortality to individuals for each species
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