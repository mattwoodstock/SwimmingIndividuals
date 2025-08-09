# == Debugging ==
function debug_best_prey(pred_data, sp)
    best_idx = Array(pred_data.best_prey_idx)
    best_type = Array(pred_data.best_prey_type)
    found = count(x -> x > 0, best_idx)
    println("🐟 Species $sp: predators with prey found: $found / $(length(best_idx))")
    
    prey_type_counts = countmap(best_type)
    println("    Prey type counts: ", prey_type_counts)
end

function debug_successful_ration(pred_data, sp)
    ration = Array(pred_data.successful_ration)
    positive = count(x -> x > 0f0, ration)
    max_ration = maximum(ration)
    println("📦 Species $sp: predators with non-zero successful ration: $positive / $(length(ration))")
    println("    Max successful ration: $max_ration")
end

function debug_flattened_prey_biomass(prey_biomass_flat, prey_offsets)
    flat = Array(prey_biomass_flat)
    total = sum(flat)
    println("📉 Total flattened prey biomass: $total")
    println("    Non-zero entries: ", count(x -> x > 0f0, flat))
    
    println("📚 Prey offsets: ", Array(prey_offsets))
end

function debug_resource_biomass(resource_biomass_grid, pred_data)
    idxs = Array(pred_data.best_prey_idx)
    types = Array(pred_data.best_prey_type)
    sps = Array(pred_data.best_prey_sp)

    used_idxs = [idxs[i] for i in 1:length(idxs) if types[i] == 2]
    used_sps = [sps[i] for i in 1:length(sps) if types[i] == 2]

    println("🌊 Resource prey: used grid cells = $(length(used_idxs))")
    
    total_val = 0.0
    for (i, idx) in enumerate(used_idxs)
        sp = used_sps[i]
        X, Y, Z = size(resource_biomass_grid)[1:3]
        lin = idx - 1
        x = lin % X + 1
        y = (lin ÷ X) % Y + 1
        z = (lin ÷ (X * Y)) + 1
        val = resource_biomass_grid[x, y, z, sp]
        total_val += val
    end
    println("    Sum of resource biomass at used prey cells: $total_val")
end

function debug_post_consumption(prey, pred_data, sp)
    prey_biomass = Array(prey.biomass_school)
    pred_ration = Array(pred_data.ration)
    pred_gut = Array(pred_data.gut_fullness)

    println("💀 Post-consumption diagnostics for species $sp")
    println("    Total prey biomass remaining: ", sum(prey_biomass))
    println("    Non-zero predator rations: ", count(x -> x > 0f0, pred_ration))
    println("    Max gut fullness: ", maximum(pred_gut))
    println(" ")
    println(" ")
    println(" ")

end

# GPU-compliant function to calculate the squared distance in meters.
@inline function haversine_distance_sq(lat1, lon1, z1, lat2, lon2, z2)
    R = 6371000.0f0  # Earth's radius in meters
    
    # Convert degrees to radians
    lat1_rad, lon1_rad = lat1 * 0.0174533f0, lon1 * 0.0174533f0
    lat2_rad, lon2_rad = lat2 * 0.0174533f0, lon2 * 0.0174533f0

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula for horizontal distance
    a = CUDA.sin(dlat / 2f0)^2 + CUDA.cos(lat1_rad) * CUDA.cos(lat2_rad) * CUDA.sin(dlon / 2f0)^2
    c = 2f0 * CUDA.atan(CUDA.sqrt(a), CUDA.sqrt(1f0 - a))
    horizontal_dist = R * c

    # Vertical distance
    vertical_dist = z2 - z1
    
    # Return total squared distance in meters
    return horizontal_dist^2 + vertical_dist^2
end

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
# Agent Predation System
# ===================================================================

# --- STEP 1: Find Best Prey ---
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

        for dy in -1:1, dx in -1:1 #Do not search other depth bins for prey to avoid large vertical movements.
            search_x = my_pool_x + dx
            search_y = my_pool_y + dy
            search_z = my_pool_z

            if 1 <= search_x <= grid_params.lonres && 1 <= search_y <= grid_params.latres && 1 <= search_z <= grid_params.depthres
                # Search Focal Species Prey
                for prey_sp_idx in 1:length(prey_data_all)
                    prey_data = prey_data_all[prey_sp_idx]
                    # This simplified search iterates over all prey. A full spatial index implementation would be faster.
                    for k in 1:length(prey_data.x)
                        @inbounds if prey_data.alive[k] == 1.0 && min_size <= prey_data.length[k] <= max_size
                            
                            prey_x, prey_y, prey_z = prey_data.x[k], prey_data.y[k], prey_data.z[k]
                            dist_sq = haversine_distance_sq(my_y, my_x, my_z, prey_y, prey_x, prey_z)

                            if dist_sq <= detection_radius_sq && dist_sq < best_prey_dist[pred_idx]
                                old_dist = atomic_cas!(pointer(best_prey_dist, pred_idx), best_prey_dist[pred_idx], Float32(dist_sq))
                                if old_dist > dist_sq # Check if our update was successful
                                    best_prey_idx[pred_idx] = k
                                    best_prey_sp[pred_idx] = prey_sp_idx
                                    best_prey_type[pred_idx] = 1
                                end
                            end
                        end
                    end
                end
                
                # Search Resource Grid
                if my_pool_x == search_x && my_pool_y == search_y && my_pool_z == search_z
                    for res_sp in 1:size(resource_biomass_grid, 4)
                        biomass_density = resource_biomass_grid[search_x, search_y, search_z, res_sp]
                        
                        if biomass_density > 0f0
                            res_min = resource_trait.Min_Size[res_sp]
                            res_max = resource_trait.Max_Size[res_sp]
                            μ = (log(res_min) + log(res_max)) / 2f0
                            σ = (log(res_max) - log(res_min)) / 4f0
                            mean_size = exp(μ + 0.5f0 * σ^2)

                            if min_size <= mean_size <= max_size
                                a = resource_trait.LWR_a[res_sp]
                                b = resource_trait.LWR_b[res_sp]
                                mean_weight = a * (mean_size / 10f0)^b

                                if mean_weight > 0f0
                                    abundance = biomass_density / mean_weight
                                    if abundance > 0f0
                                        lat_rad = my_y * 0.0174532925f0 
                                        deg2m = 111320.0f0
                                        width_m = grid_params.cell_size_deg * deg2m * cos(lat_rad)
                                        height_m = grid_params.cell_size_deg * deg2m
                                        volume_m3 = width_m * height_m * grid_params.depth_res_m
                                        
                                        volume_per_individual = volume_m3 / abundance
                                        dist_sq = cbrt(volume_per_individual)^2

                                        if dist_sq < best_prey_dist[pred_idx]
                                            linear_idx = search_x + (search_y-1)*grid_params.lonres + (search_z-1)*grid_params.lonres*grid_params.latres
                                            old_dist = atomic_cas!(pointer(best_prey_dist, pred_idx), best_prey_dist[pred_idx], Float32(dist_sq))
                                            
                                            if old_dist > dist_sq
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
        end
    end
end

function calculate_distances_prey!(model::MarineModel, sp::Int, inds::Vector{Int32})
    arch = model.arch
    pred_data = model.individuals.animals[sp].data
    pred_params = (min_prey_ratio = model.individuals.animals[sp].p.Min_Prey[2][sp], max_prey_ratio = model.individuals.animals[sp].p.Max_Prey[2][sp])
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
    trait_df = model.resource_trait
    resource_trait_gpu = (; (Symbol(c) => array_type(arch)(trait_df[:, c]) for c in names(trait_df))...)

    fill!(pred_data.best_prey_dist, Inf32)
    fill!(pred_data.best_prey_idx, 0)
    fill!(pred_data.best_prey_sp, 0)
    fill!(pred_data.best_prey_type, 0)

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


# --- STEP 2: Resolve Consumption Conflicts (CPU Referee) ---
function resolve_consumption!(model::MarineModel, sp::Int, to_eat::Vector{Int32})
    pred_data = model.individuals.animals[sp].data
    
    best_prey_idx_cpu = Array(pred_data.best_prey_idx[to_eat])
    best_prey_sp_cpu = Array(pred_data.best_prey_sp[to_eat])
    best_prey_type_cpu = Array(pred_data.best_prey_type[to_eat])
    pred_biomass_cpu = Array(pred_data.biomass_school[to_eat])
    pred_gut_full_cpu = Array(pred_data.gut_fullness[to_eat])
    
    prey_biomass_all_cpu = [Array(animal.data.biomass_school) for animal in model.individuals.animals]
    res_biomass_cpu = Array(model.resources.biomass)

    agent_energy_densities = [animal.p.Energy_density.second[i] for (i, animal) in enumerate(model.individuals.animals)]
    resource_energy_densities = model.resource_trait.Energy_density

    successful_rations_cpu = zeros(Float64, length(pred_data.x))
    prey_claimed = Dict{Tuple{Int, Int}, Float64}()

    for i in 1:length(to_eat)
        pred_idx = to_eat[i]
        prey_idx = best_prey_idx_cpu[i]
        if prey_idx == 0; continue; end

        prey_sp = best_prey_sp_cpu[i]
        prey_type = best_prey_type_cpu[i]
        
        prey_key = (prey_type, prey_idx)
        claimed_biomass = get(prey_claimed, prey_key, 0.0)
        
        total_biomass = (prey_type == 1) ? prey_biomass_all_cpu[prey_sp][prey_idx] : res_biomass_cpu[prey_idx]
        available_biomass = total_biomass - claimed_biomass
        if available_biomass <= 0; continue; end

        max_stomach = 0.2 * pred_biomass_cpu[i]
        current_stomach = pred_gut_full_cpu[i]
        stomach_space = max(0.0, (1.0 - current_stomach)*max_stomach)
        
        ration_biomass = min(available_biomass, stomach_space)
        
        if ration_biomass > 0
            energy_density = (prey_type == 1) ? agent_energy_densities[prey_sp] : resource_energy_densities[prey_sp]
            ration_joules = ration_biomass * energy_density

            successful_rations_cpu[pred_idx] = ration_joules
            prey_claimed[prey_key] = get(prey_claimed, prey_key, 0.0) + ration_biomass
        end
    end
    
    copyto!(@view(pred_data.successful_ration[1:end]), successful_rations_cpu)
end

@kernel function apply_consumption_kernel!(
    alive, best_prey_dist, best_prey_idx, best_prey_sp, best_prey_type,
    x, y, z, pool_x, pool_y, pool_z, length_arr, biomass_school, gut_fullness, abundance,
    ration, active, successful_ration,
    all_prey_x::NTuple{N, Any}, all_prey_y::NTuple{N, Any}, all_prey_z::NTuple{N, Any},
    all_prey_biomass::NTuple{N, Any}, all_prey_biomass_school::NTuple{N, Any}, all_prey_alive::NTuple{N, Any}, 
    all_prey_abundance::NTuple{N,Any},
    all_prey_energy::NTuple{N,Any},
    all_prey_energy_density::CuDeviceVector{Float32},
    resource_biomass_grid, 
    resource_energy_density,
    resource_trait,
    consumption_array,
    swim_velo::Float32, handling_time::Float32, time_array,
    predator_sp_idx::Int, n_species::Int32,
    grid_params 
) where {N}
    pred_idx = @index(Global)

    @inbounds if pred_idx <= length(alive) && alive[pred_idx] == 1.0
        # s_ration is now in JOULES
        s_ration = successful_ration[pred_idx]
        
        if s_ration > 0.0f0
            dist = sqrt(best_prey_dist[pred_idx])
            time_left = time_array[pred_idx]
            
            swim_v = swim_velo * (length_arr[pred_idx] / 1000.0f0)
            time_to_prey = swim_v > 0f0 ? dist / swim_v : Inf32

            if time_to_prey < time_left
                time_left -= time_to_prey
                @atomic active[pred_idx] += time_to_prey / 60.0f0

                prey_type = best_prey_type[pred_idx]
                prey_idx = best_prey_idx[pred_idx]
                prey_sp_idx = best_prey_sp[pred_idx]

                prey_ind_biomass = 0.0f0
                prey_energy_density = 0.0f0

                if prey_type == 1
                    prey_ind_biomass = all_prey_biomass[prey_sp_idx][prey_idx]
                    prey_energy_density = all_prey_energy_density[prey_sp_idx][prey_sp_idx]
                elseif prey_type == 2
                    res_min = resource_trait.Min_Size[prey_sp_idx]
                    res_max = resource_trait.Max_Size[prey_sp_idx]
                    μ = (log(res_min) + log(res_max)) / 2f0
                    σ = (log(res_max) - log(res_min)) / 4f0
                    mean_size = exp(μ + 0.5f0 * σ^2)
                    a = resource_trait.LWR_a[prey_sp_idx]
                    b = resource_trait.LWR_b[prey_sp_idx]
                    prey_ind_biomass = a * (mean_size / 10f0)^b
                    prey_energy_density = resource_energy_density[prey_sp_idx]
                end

                predator_abundance = abundance[pred_idx]
                num_can_handle = (handling_time > 0f0 && prey_ind_biomass > 0f0) ? floor(Int, (time_left / handling_time) * predator_abundance) : typemax(Int)

                # Calculate the maximum consumable energy based on handling time
                prey_ind_energy = prey_ind_biomass * prey_energy_density
                max_consumable_energy = num_can_handle * prey_ind_energy
                
                # effective_ration is the final amount consumed, in JOULES
                effective_ration = min(s_ration, max_consumable_energy)
                
                if effective_ration > 0f0
                    # Convert the energy ration back to biomass for updates
                    effective_biomass = effective_ration / prey_energy_density

                    px, py, pz = pool_x[pred_idx], pool_y[pred_idx], pool_z[pred_idx]
                    prey_dim_idx = (prey_type == 1) ? prey_sp_idx : n_species + prey_sp_idx
                    if prey_dim_idx > 0
                        # Consumption array should track biomass flow
                        @atomic consumption_array[px, py, pz, predator_sp_idx, prey_dim_idx] += effective_biomass
                    end
                    
                    # --- Predator state updates ---
                    if prey_type == 1
                        x[pred_idx] = all_prey_x[prey_sp_idx][prey_idx]
                        y[pred_idx] = all_prey_y[prey_sp_idx][prey_idx]
                        z[pred_idx] = all_prey_z[prey_sp_idx][prey_idx]
                    elseif prey_type == 2
                        biomass_density = resource_biomass_grid[px, py, pz, prey_sp_idx]
                        mean_weight = prey_ind_biomass
                        abundance_resource = (mean_weight > 0.0f0) ? (biomass_density / mean_weight) : 0.0f0

                        move_distance = 0.0f0
                        if abundance_resource > 0.0f0
                            current_lat_rad = y[pred_idx] * 0.0174532925f0
                            deg2m = 111320.0f0
                            width_m = grid_params.cell_size_deg * deg2m * cos(current_lat_rad)
                            height_m = grid_params.cell_size_deg * deg2m
                            cell_volume_m3 = width_m * height_m * grid_params.depth_res_m
                            volume_per_individual = cell_volume_m3 / abundance_resource
                            move_distance = cbrt(volume_per_individual)
                        end

                        num_consumed = prey_ind_biomass > 0f0 ? floor(Int, effective_biomass / prey_ind_biomass) : 0


                        if move_distance > 0.0f0 && num_consumed > 0
                            total_move_distance = move_distance * num_consumed
                            max_possible_dist = swim_v * time_left
                            total_move_distance = min(total_move_distance, max_possible_dist)

                            rand_x = randn(Float32); rand_y = randn(Float32); rand_z = randn(Float32)
                            norm = sqrt(rand_x*rand_x + rand_y*rand_y + rand_z*rand_z)
                            dir_x, dir_y, dir_z = (norm > 0.0f0) ? (rand_x/norm, rand_y/norm, rand_z/norm) : (1.0f0, 0.0f0, 0.0f0)

                            offset_x_m = dir_x * total_move_distance
                            offset_y_m = dir_y * total_move_distance
                            offset_z_m = dir_z * total_move_distance

                            current_lat = y[pred_idx]
                            current_lat_rad = current_lat * 0.0174532925f0
                            offset_lat_deg = offset_y_m / 111320.0f0
                            offset_lon_deg = offset_x_m / (111320.0f0 * cos(current_lat_rad))
                            
                            new_lon = x[pred_idx] + offset_lon_deg
                            new_lat = clamp(current_lat + offset_lat_deg, -90.0f0, 90.0f0)
                            
                            current_z = z[pred_idx]
                            z_min_boundary = max(1.0f0, current_z - 5.0f0)
                            z_max_boundary = current_z + 5.0f0
                            potential_new_z = current_z + offset_z_m
                            new_z = clamp(potential_new_z, z_min_boundary, z_max_boundary)
                            
                            x[pred_idx] = new_lon; y[pred_idx] = new_lat; z[pred_idx] = new_z

                            pool_x[pred_idx] = clamp(floor(Int32, (new_lon - grid_params.lonmin) / grid_params.cell_size_deg) + 1, 1, grid_params.lonres)
                            pool_y[pred_idx] = clamp(grid_params.latres - floor(Int32, (grid_params.latmax - new_lat) / grid_params.cell_size_deg), 1, grid_params.latres)
                            pool_z[pred_idx] = clamp(ceil(Int32, new_z / grid_params.depth_res_m), 1, grid_params.depthres)
                        end
                    end
                    
                    # Predator's ration is in JOULES
                    @atomic ration[pred_idx] += effective_ration
                    # Gut fullness is a biomass ratio, so use the converted biomass value
                    @atomic gut_fullness[pred_idx] += effective_biomass / (biomass_school[pred_idx] * 0.2f0)

                    # --- Prey state updates (in biomass) ---
                    if prey_type == 1
                        prey_initial_biomass = all_prey_biomass_school[prey_sp_idx][prey_idx]
                            
                        if prey_initial_biomass > 0.0f0
                            biomass_proportion_consumed = effective_biomass / prey_initial_biomass
                            prey_total_energy = all_prey_energy[prey_sp_idx][prey_idx]
                            energy_removed = prey_total_energy * biomass_proportion_consumed
                            
                            @atomic all_prey_energy[prey_sp_idx][prey_idx] -= energy_removed

                            @atomic all_prey_biomass_school[prey_sp_idx][prey_idx] -= effective_biomass
                            prey_ind_biomass_val = all_prey_biomass_school[prey_sp_idx][prey_idx]
                            if prey_ind_biomass_val > 0.0f0
                                inds_removed = floor(Int32, effective_biomass / prey_ind_biomass_val)
                                @atomic all_prey_abundance[prey_sp_idx][prey_idx] -= Float32(inds_removed)
                            end
                        end

                        if all_prey_abundance[prey_sp_idx][prey_idx] <= 0.0f0
                            all_prey_alive[prey_sp_idx][prey_idx] = 0.0f0
                        end

                        if all_prey_biomass_school[prey_sp_idx][prey_idx] <= 0f0
                            all_prey_alive[prey_sp_idx][prey_idx] = 0.0f0
                        end
                    elseif prey_type == 2
                        @atomic resource_biomass_grid[px, py, pz, prey_sp_idx] -= effective_biomass
                    end
                    
                    # --- Time budget update ---
                    num_consumed = prey_ind_biomass > 0f0 ? floor(Int, effective_biomass / prey_ind_biomass) : 0
                    time_spent_handling = num_consumed * handling_time
                    @atomic active[pred_idx] += time_spent_handling / 60.0f0
                    time_array[pred_idx] -= time_spent_handling
                end
            else
                @atomic active[pred_idx] += time_left / 60.0f0
                time_array[pred_idx] = 0f0
            end
            successful_ration[pred_idx] = 0f0
        end
    end
end

function apply_consumption!(model::MarineModel, sp::Int, time::CuArray{Float32}, outputs::MarineOutputs)
    arch = model.arch
    pred_data = model.individuals.animals[sp].data
    n_species = model.n_species

    # Extract predator traits as scalars
    swim_velo = Float32(model.individuals.animals[sp].p.Swim_velo[2][sp])
    handling_time = Float32(model.individuals.animals[sp].p.Handling_Time[2][sp])

    grid = model.depths.grid
    grid_params = (
        lonres = Int(grid[grid.Name .== "lonres", :Value][1]),
        latres = Int(grid[grid.Name .== "latres", :Value][1]),
        depthres = Int(grid[grid.Name .== "depthres", :Value][1]),
        lonmin = grid[grid.Name .== "xllcorner", :Value][1],
        latmax = grid[grid.Name .== "yulcorner", :Value][1],
        cell_size_deg = grid[grid.Name .== "cellsize", :Value][1],
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / Int(grid[grid.Name .== "depthres", :Value][1])
    )

    # Create tuples of all required data arrays from all prey species
    all_prey_x = tuple((animal.data.x for animal in model.individuals.animals)...)
    all_prey_y = tuple((animal.data.y for animal in model.individuals.animals)...)
    all_prey_z = tuple((animal.data.z for animal in model.individuals.animals)...)
    all_prey_biomass = tuple((animal.data.biomass_ind for animal in model.individuals.animals)...)
    all_prey_biomass_school = tuple((animal.data.biomass_school for animal in model.individuals.animals)...)
    all_prey_alive = tuple((animal.data.alive for animal in model.individuals.animals)...)
    all_prey_abundance = tuple((animal.data.abundance for animal in model.individuals.animals)...)
    all_prey_energy_density = tuple((array_type(arch)(animal.p.Energy_density.second) for animal in model.individuals.animals)...)
    all_prey_energy = tuple((animal.data.energy for animal in model.individuals.animals)...)
    agent_energy_densities_cpu = [animal.p.Energy_density.second[i] for (i, animal) in enumerate(model.individuals.animals)]
    all_prey_energy_density = array_type(arch)(Float32.(agent_energy_densities_cpu))
    resource_energy_density = array_type(arch)(model.resource_trait.Energy_density)

    resource_biomass_grid = model.resources.biomass
    trait_df = model.resource_trait
    resource_trait_gpu = (; (Symbol(c) => array_type(arch)(trait_df[:, c]) for c in names(trait_df))...)

    n = length(pred_data.x)
    kernel! = apply_consumption_kernel!(device(arch), 256, (n,))

    kernel!(
        pred_data.alive, pred_data.best_prey_dist, pred_data.best_prey_idx,
        pred_data.best_prey_sp, pred_data.best_prey_type,
        pred_data.x, pred_data.y, pred_data.z, pred_data.pool_x, pred_data.pool_y, pred_data.pool_z, pred_data.length,
        pred_data.biomass_school, pred_data.gut_fullness, pred_data.abundance,
        pred_data.ration, pred_data.active, pred_data.successful_ration,
        all_prey_x, all_prey_y, all_prey_z, all_prey_biomass, all_prey_biomass_school, all_prey_alive,
        all_prey_abundance,
        all_prey_energy,
        all_prey_energy_density, # Pass agent energy densities
        resource_biomass_grid,
        resource_energy_density, # Pass resource energy densities
        resource_trait_gpu,
        outputs.consumption,
        swim_velo,
        handling_time,
        time,
        sp,
        n_species,
        grid_params
    )

    KernelAbstractions.synchronize(device(arch))
    return nothing
end

# ===================================================================
# Background Resource Predation System
# ===================================================================
@inline function find_size_bin(value, bins)
    for i in 1:length(bins)
        if value < bins[i]
            return i
        end
    end
    return length(bins) + 1 # Return the last bin if larger than all bin thresholds
end

@kernel function aggregate_prey_by_size_kernel!(
    prey_biomass_grid_by_size,
    animals_all,
    size_bins
)
    i = @index(Global)
    for sp in 1:length(animals_all)
        animal = animals_all[sp]
        if i <= length(animal.x) && animal.alive[i] == 1.0f0
            px, py, pz = animal.pool_x[i], animal.pool_y[i], animal.pool_z[i]
            
            # Find which size bin this agent belongs to
            bin_idx = find_size_bin(animal.length[i], size_bins)
            
            # Atomically add the agent's biomass to the correct grid cell, species, and size bin
            @atomic prey_biomass_grid_by_size[px, py, pz, sp, bin_idx] += animal.biomass_school[i]
        end
    end
end

# -- Kernel 1: Calculate prey biomass grid
@kernel function aggregate_agent_props_kernel!(
    prey_biomass_grid,
    agent_total_length_grid,
    agent_abundance_grid,
    animals_all
)
    i = @index(Global)
    for sp in 1:length(animals_all)
        animal = animals_all[sp]
        # Ensure we are within the bounds of the current species' data
        if i <= length(animal.x) && animal.alive[i] == 1.0f0
            px, py, pz = animal.pool_x[i], animal.pool_y[i], animal.pool_z[i]
            
            # Atomically add the properties of this agent to its grid cell
            @atomic prey_biomass_grid[px, py, pz, sp] += animal.biomass_school[i]
            # We add the length multiplied by the number of individuals in the school
            @atomic agent_total_length_grid[px, py, pz, sp] += (animal.length[i] * animal.abundance[i])
            @atomic agent_abundance_grid[px, py, pz, sp] += animal.abundance[i]
        end
    end
end

# -- Kernel 2: Calculate mortality rate grid using resource traits
@kernel function calculate_mortality_rate_kernel!(
    mortality_rate_grid_by_size,
    pred_biomass_grid,
    prey_biomass_grid_by_size,
    consumption_array,
    size_bins,
    min_prey,
    max_prey,
    daily_ration, # Now in Joules
    handling_time,
    lwr_a,
    lwr_b,
    resource_energy_density, # NEW: Energy density (J/g) for resources
    agent_energy_density,    # NEW: Energy density (J/g) for agents
    mu_pred,
    sigma_pred,
    dt::Float32,
    grid_params,
    n_species::Int32
)
    x, y, z, r = @index(Global, NTuple)

    # --- Calculate Cell Volume ---
    cell_lat_deg = grid_params.lat_min + (y - 0.5f0) * grid_params.cell_size_deg
    lat_rad = cell_lat_deg * 0.0174532925f0 
    deg2m = 111320.0f0
    width_m = grid_params.cell_size_deg * deg2m * cos(lat_rad)
    height_m = grid_params.cell_size_deg * deg2m
    cell_volume_m3 = width_m * height_m * grid_params.depth_res_m
    
    pred_biom = pred_biomass_grid[x, y, z, r]

    n_prey_species = size(prey_biomass_grid_by_size, 4)
    n_size_bins = size(prey_biomass_grid_by_size, 5)
    n_resource_species = size(pred_biomass_grid, 4)

    if pred_biom <= 0.0f0
        for sp in 1:n_prey_species, bin in 1:n_size_bins
            mortality_rate_grid_by_size[x, y, z, r, sp, bin] = 0.0f0
        end
    else
        min_prey_size = min_prey[r]
        max_prey_size = max_prey[r]
        max_ingestion = daily_ration[r] # This is the attack rate 'a' in Joules
        h_time = handling_time[r]
        
        # Step 1: Calculate total available prey ENERGY
        total_available_energy = 0.0f0
        # Agent prey
        for sp in 1:n_prey_species
            energy_density = agent_energy_density[sp]
            for bin in 1:n_size_bins
                prey_size_in_bin = (bin == 1) ? 0.0f0 : size_bins[bin-1]
                if prey_size_in_bin >= min_prey_size && prey_size_in_bin <= max_prey_size
                    biomass_in_bin = prey_biomass_grid_by_size[x, y, z, sp, bin]
                    total_available_energy += biomass_in_bin * energy_density
                end
            end
        end
        # Resource prey
        for prey_r in 1:n_resource_species
            if r != prey_r
                prey_μ = mu_pred[prey_r]; prey_σ = sigma_pred[prey_r]
                prey_mean_size = exp(prey_μ + 0.5f0 * prey_σ^2)
                if prey_mean_size >= min_prey_size && prey_mean_size <= max_prey_size
                    biomass_of_prey_r = pred_biomass_grid[x, y, z, prey_r]
                    energy_density = resource_energy_density[prey_r]
                    total_available_energy += biomass_of_prey_r * energy_density
                end
            end
        end

        if total_available_energy > 0.0f0
            # Step 2: Calculate total consumption in JOULES
            a = max_ingestion
            N = total_available_energy / cell_volume_m3 # N is now energy density (J/m³)
            
            # Consumption rate per unit of predator biomass (J consumed per gram of predator)
            consumption_rate_per_biomass = (a * N) / (1.0f0 + a * h_time * N)
            total_consumption_J = consumption_rate_per_biomass * pred_biom
            total_consumption_J = min(total_consumption_J, max_ingestion * pred_biom)

            predator_dim_idx = n_species + r

            # Step 3: Distribute mortality to AGENT prey
            for sp in 1:n_prey_species
                energy_density = agent_energy_density[sp]
                total_consumed_biomass_this_sp = 0.0f0
                for bin in 1:n_size_bins
                    mortality_rate_grid_by_size[x, y, z, r, sp, bin] = 0.0f0
                    prey_size_in_bin = (bin == 1) ? 0.0f0 : size_bins[bin-1]
                    if prey_size_in_bin >= min_prey_size && prey_size_in_bin <= max_prey_size
                        prey_biom_in_bin = prey_biomass_grid_by_size[x, y, z, sp, bin]
                        if prey_biom_in_bin > 0.0f0
                            prop_energy = (prey_biom_in_bin * energy_density) / total_available_energy
                            consumed_energy = total_consumption_J * prop_energy
                            consumed_biomass = consumed_energy / energy_density
                            
                            total_consumed_biomass_this_sp += consumed_biomass
                            mortality_rate = (consumed_biomass * dt) / (prey_biom_in_bin * 1440.0f0)
                            mortality_rate_grid_by_size[x, y, z, r, sp, bin] = clamp(mortality_rate, 0.0f0, 1.0f0)
                        end
                    end
                end
                if total_consumed_biomass_this_sp > 0.0f0
                    @atomic consumption_array[x, y, z, predator_dim_idx, sp] += total_consumed_biomass_this_sp
                end
            end

            # Step 4: Apply consumption to RESOURCE patches
            for prey_r in 1:n_resource_species
                if r != prey_r
                    prey_μ = mu_pred[prey_r]; prey_σ = sigma_pred[prey_r]
                    prey_mean_size = exp(prey_μ + 0.5f0 * prey_σ^2)
                    if prey_mean_size >= min_prey_size && prey_mean_size <= max_prey_size
                        resource_prey_biomass = pred_biomass_grid[x, y, z, prey_r]
                        if resource_prey_biomass > 0.0f0
                            energy_density = resource_energy_density[prey_r]
                            prop_energy = (resource_prey_biomass * energy_density) / total_available_energy
                            consumed_energy = total_consumption_J * prop_energy
                            consumed_biomass = consumed_energy / energy_density
                            
                            @atomic pred_biomass_grid[x, y, z, prey_r] -= consumed_biomass
                            
                            prey_dim_idx = n_species + prey_r
                            @atomic consumption_array[x, y, z, predator_dim_idx, prey_dim_idx] += consumed_biomass
                        end
                    end
                end
            end
        else
            # No available prey, set all mortality rates to zero
            for sp in 1:n_prey_species, bin in 1:n_size_bins
                mortality_rate_grid_by_size[x, y, z, r, sp, bin] = 0.0f0
            end
        end
    end
end

@inline function find_size_bin(value, bins)
    # The bins array contains the upper thresholds for each bin
    for i in 1:length(bins)
        if value < bins[i]
            return i # Belongs to bin 1, 2, ..., n
        end
    end
    # If larger than all thresholds, it belongs to the last bin
    return length(bins) + 1 
end

@kernel function apply_resource_mortality_kernel!(
    animal_data,
    mortality_rate_grid_by_size,
    size_bins,
    sp_idx
)
    i = @index(Global)
    if i <= length(animal_data.x) && animal_data.alive[i] == 1.0f0
        # --- Get agent's location and size information ---
        x, y, z = animal_data.pool_x[i], animal_data.pool_y[i], animal_data.pool_z[i]
        my_length = animal_data.length[i]
        
        # --- Determine the agent's specific size bin ---
        my_bin = find_size_bin(my_length, size_bins)
        
        # --- Sum the mortality rate from all resource predators for this specific bin ---
        total_mortality_rate = 0.0f0
        for r in 1:size(mortality_rate_grid_by_size, 4)
            total_mortality_rate += mortality_rate_grid_by_size[x, y, z, r, sp_idx, my_bin]
        end
        
        if total_mortality_rate > 0.0f0
            # Clamp the total rate to a maximum of 1.0 (100% mortality)
            total_mortality_rate = min(total_mortality_rate, 1.0f0)
            
            initial_biomass = animal_data.biomass_school[i]

            if initial_biomass < 0.0f0
                # --- Remove biomass based on the calculated mortality rate ---
                biomass_removed = animal_data.biomass_school[i] * total_mortality_rate

                # Calculate the proportion of the school's biomass being consumed
                biomass_proportion_consumed = biomass_removed / initial_biomass
                
                # Get the prey's total stored energy
                initial_energy = animal_data.energy[i]
                
                # Calculate the amount of energy to remove
                energy_removed = initial_energy * biomass_proportion_consumed
                
                # Atomically update the prey's stored energy
                @atomic animal_data.energy[i] -= energy_removed

                @atomic animal_data.biomass_school[i] -= biomass_removed
                
                # --- Remove a corresponding number of individuals from the school ---
                if animal_data.biomass_ind[i] > 0.0f0
                    inds_removed = floor(Int32, biomass_removed / animal_data.biomass_ind[i])
                    @atomic animal_data.abundance[i] -= Float32(inds_removed)
                end
                
                # --- Check if the agent's school is now depleted ---
                if animal_data.abundance[i] <= 0.0f0
                    animal_data.alive[i] = 0.0f0
                end
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

    # --- 1. Create a grid_params tuple to pass to the kernel ---
    grid_params = (
        lat_min = g[g.Name .== "yllcorner", :Value][1],
        cell_size_deg = g[g.Name .== "cellsize", :Value][1],
        depth_res_m = g[g.Name .== "depthmax", :Value][1] / depthres
    )

    # --- 2. Dynamically Define 5 Size Bins ---
    min_size_global = minimum(rt.Min_Size)
    max_size_global = maximum(rt.Max_Size)
    log_min = log(min_size_global)
    log_max = log(max_size_global)
    log_step = (log_max - log_min) / 5.0f0
    size_bins_cpu = Float32[exp(log_min + i * log_step) for i in 1:4]
    size_bins_gpu = array_type(arch)(size_bins_cpu)
    n_size_bins = 5

    # --- 3. Initialize Grids ---
    prey_biomass_grid_by_size = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_sp, n_size_bins))
    mortality_rate_grid_by_size = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res, n_sp, n_size_bins))

    # --- 4. Prepare Resource and Agent Trait Data ---
    μs_cpu = zeros(Float32, n_res)
    σs_cpu = zeros(Float32, n_res)
    for r in 1:n_res
        μ, σ = lognormal_params_from_minmax(rt.Min_Size[r], rt.Max_Size[r])
        μs_cpu[r] = Float32(μ)
        σs_cpu[r] = Float32(σ)
    end

    resource_trait_gpu = (
        Min_Prey = array_type(arch)(Float32.(rt.Min_Prey)),
        Max_Prey = array_type(arch)(Float32.(rt.Max_Prey)),
        Daily_Ration = array_type(arch)(Float32.(rt.Daily_Ration)), # This is now in Joules
        Handling_Time = array_type(arch)(Float32.(rt.Handling_Time)),
        LWR_a = array_type(arch)(Float32.(rt.LWR_a)),
        LWR_b = array_type(arch)(Float32.(rt.LWR_b)),
        Energy_density = array_type(arch)(Float32.(rt.Energy_density)), # Energy density for resources
        Mu_pred = array_type(arch)(μs_cpu),
        Sigma_pred = array_type(arch)(σs_cpu)
    )
    
    # Gather energy densities for agent prey species
    agent_energy_densities_gpu = array_type(arch)(
        [Float32(animal.p.Energy_density.second[sp]) for (sp, animal) in enumerate(model.individuals.animals)]
    )

    # --- 5. Kernel 1: Aggregate Agent Biomass into Size-Binned Grids ---
    animals_all = Tuple(animal.data for animal in model.individuals.animals)
    max_len = isempty(animals_all) ? 0 : maximum(length(a.x) for a in animals_all if !isempty(a.x))

    if max_len > 0
        kernel1 = aggregate_prey_by_size_kernel!(device(arch), 256, (max_len,))
        kernel1(prey_biomass_grid_by_size, animals_all, size_bins_gpu)
    end

    # --- 6. Kernel 2: Calculate and Apply Predation ---
    kernel2_ndrange = (lonres, latres, depthres, n_res)
    kernel2 = calculate_mortality_rate_kernel!(device(arch), (8,8,4,1), kernel2_ndrange)
    kernel2(
        mortality_rate_grid_by_size,
        model.resources.biomass,
        prey_biomass_grid_by_size,
        output.consumption,
        size_bins_gpu,
        resource_trait_gpu.Min_Prey,
        resource_trait_gpu.Max_Prey,
        resource_trait_gpu.Daily_Ration,
        resource_trait_gpu.Handling_Time,
        resource_trait_gpu.LWR_a,
        resource_trait_gpu.LWR_b,
        resource_trait_gpu.Energy_density, # Pass resource energy densities
        agent_energy_densities_gpu,       # Pass agent energy densities
        resource_trait_gpu.Mu_pred,
        resource_trait_gpu.Sigma_pred,
        Float32(model.dt),
        grid_params,
        n_sp
    )

    # --- 7. Kernel 3: Apply Mortality to Agents ---
    for sp_idx in 1:n_sp
        agents = model.individuals.animals[sp_idx].data
        if length(agents.x) > 0
            kernel3 = apply_resource_mortality_kernel!(device(arch), 256, (length(agents.x),))
            kernel3(agents, mortality_rate_grid_by_size, size_bins_gpu, sp_idx)
        end
    end

    KernelAbstractions.synchronize(device(arch))
    return nothing
end