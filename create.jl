# ===================================================================
# Agent Creation and Initialization
# ===================================================================

function construct_individuals(arch::Architecture, params::Dict, maxN)
    # Defines the full data structure for an agent.
    rawdata = StructArray(
        unique_id = zeros(Int, maxN), x = zeros(Float32,maxN), y = zeros(Float32,maxN), z = zeros(Float32,maxN),
        length = zeros(Float32,maxN), abundance = zeros(Float64,maxN),
        biomass_ind = zeros(Float32,maxN), biomass_school = zeros(Float32,maxN),
        energy = zeros(Float32,maxN), gut_fullness = zeros(Float32,maxN),
        cost = zeros(Float32,maxN), pool_x = zeros(Int,maxN), pool_y = zeros(Int,maxN),
        pool_z = zeros(Int,maxN), active = zeros(Float32,maxN),
        ration = zeros(Float32,maxN), alive = zeros(Float32,maxN),
        vis_prey = zeros(Float32,maxN), mature = zeros(Float32,maxN),
        age=zeros(Float32,maxN),
        generation = zeros(Int32,maxN),
        cell_id = zeros(Int, maxN),
        sorted_id = zeros(Int, maxN),
        repro_energy = zeros(Float32, maxN),
        best_prey_dist = zeros(Float32, maxN),
        best_prey_idx = zeros(Int, maxN),
        best_prey_sp = zeros(Int, maxN),
        best_prey_type = zeros(Int, maxN),
        successful_ration = zeros(Float32, maxN),
        temp_idx = zeros(Int, maxN),
        cell_starts = zeros(Int, maxN),
        cell_ends = zeros(Int, maxN),
        mig_status = zeros(Float32, maxN),
        target_z = zeros(Float32, maxN),
        interval = zeros(Float32, maxN),
        target_pool_x = zeros(Float32, maxN),
        target_pool_y = zeros(Float32, maxN)
    )

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Dive_Interval,:Min_Prey,:LWR_b, :Surface_Interval,:W_mat,:SpeciesLong, :LWR_a, :Larval_Size,:Max_Prey, :Max_Size,:School_Size,:Taxa, :Larval_Duration, :Max_Stomach, :Sex_Ratio,:SpeciesShort,:FLR_b, :Handling_Time,:Dive_Min_Night,:FLR_a,:Energy_density,:Min_Size, :Hatch_Survival, :MR_type, :Dive_Min_Day, :Dive_Max_Day, :Swim_velo, :Biomass,:Dive_Max_Night, :Type)
    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function initialize_individuals(arch, plank, B::Float32, sp::Int, depths::MarineDepths, capacities, dt, envi::MarineEnvironment, params::Dict, start_date::Date)
    # --- 1. Pre-calculate constants and create temporary CPU arrays ---
    grid = depths.grid
    night_profs = depths.focal_night
    depthres = grid[findfirst(grid.Name .== "depthres"), :Value]
    maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
    lonmax = grid[findfirst(grid.Name .== "xurcorner"), :Value]
    lonmin = grid[findfirst(grid.Name .== "xllcorner"), :Value]
    latmax = grid[findfirst(grid.Name .== "yulcorner"), :Value]
    latmin = grid[findfirst(grid.Name .== "yllcorner"), :Value]
    mean_lat_rad = deg2rad((latmin + latmax) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(mean_lat_rad)
    area_km2 = abs(latmax - latmin) * km_per_deg_lat * abs(lonmax - lonmin) * km_per_deg_lon
    target_b = B * 1e6 * area_km2
    school_size = plank.p.School_Size[2][sp]
    max_size = plank.p.Max_Size[2][sp]
    min_size = plank.p.Min_Size[2][sp]

    cpu_lengths, cpu_biomass_ind, cpu_biomass_school = Float32[], Float32[], Float32[]
    
    # --- 2. Generate core data on the CPU until target biomass is met ---
    current_b = 0.0
    μ, σ = lognormal_params_from_minmax(min_size,max_size)
    dist = LogNormal(μ, σ)
    while current_b < target_b
        new_length = rand(dist)
        while new_length > max_size; new_length = rand(dist); end
        ind_biomass = plank.p.LWR_a[2][sp] * (new_length / 10)^plank.p.LWR_b[2][sp]
        school_biomass = ind_biomass * school_size
        push!(cpu_lengths, new_length)
        push!(cpu_biomass_ind, ind_biomass)
        push!(cpu_biomass_school, school_biomass)
        current_b += school_biomass
    end

    n_agents = length(cpu_lengths)
    current_maxN = length(plank.data.x)

    if n_agents > current_maxN
        @info "Initial population ($n_agents) exceeds initial capacity ($current_maxN). Resizing buffer..."
        plank = construct_individuals(arch, params, n_agents)
    end

    # --- 3. Generate remaining data in efficient, vectorized calls on the CPU ---
    if n_agents > 0
        cpu_abundance = fill(Float32(school_size), n_agents)

        # --- Generate unique IDs for the initial population ---
        cpu_unique_ids = zeros(Int64, n_agents)
        date_int = parse(Int, Dates.format(start_date, "mmddyy"))
        species_counter = 0

        for i in 1:n_agents
            species_counter += 1
            # Formula for CCCCSSMMDDYY format
            cpu_unique_ids[i] = (Int64(species_counter) * 100000000) + (Int64(sp) * 1000000) + Int64(date_int)
        end
        
        land_mask = coalesce.(Array(envi.data["bathymetry"]), 0.0) .> 0
        res = initial_ind_placement(Array(capacities), sp, grid, n_agents, 1, land_mask)
        
        cpu_x, cpu_y, cpu_pool_x, cpu_pool_y = res.lons, res.lats, res.grid_x, res.grid_y
        cpu_z = gaussmix(n_agents, night_profs[sp, "mu1"], night_profs[sp, "mu2"], night_profs[sp, "mu3"], 
                                 night_profs[sp, "sigma1"], night_profs[sp, "sigma2"], night_profs[sp, "sigma3"], 
                                 night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])
        cpu_z = clamp.(cpu_z, 1.0, maxdepth)
        cpu_pool_z = max.(1, ceil.(Int, cpu_z ./ (maxdepth / depthres)))
        cpu_pool_z = clamp.(cpu_pool_z, 1, Int(depthres))
        max_weight = plank.p.LWR_a[2][sp] * (max_size / 10)^plank.p.LWR_b[2][sp]
        cpu_mature = min.(1.0, cpu_biomass_ind ./ (plank.p.W_mat[2][sp] * max_weight))
        cpu_vis_prey = visual_range_preys_init(cpu_lengths, cpu_z, plank.p.Min_Prey[2][sp], plank.p.Max_Prey[2][sp], n_agents) .* dt
        
        # --- 4. Copy all data from CPU arrays to the target device (CPU or GPU) in one batch ---
        copyto!(plank.data.unique_id, 1, cpu_unique_ids, 1, n_agents)
        copyto!(plank.data.length, 1, cpu_lengths, 1, n_agents)
        copyto!(plank.data.biomass_ind, 1, cpu_biomass_ind, 1, n_agents)
        copyto!(plank.data.biomass_school, 1, cpu_biomass_school, 1, n_agents)
        copyto!(plank.data.abundance, 1, cpu_abundance, 1, n_agents)
        copyto!(plank.data.x, 1, cpu_x, 1, n_agents)
        copyto!(plank.data.y, 1, cpu_y, 1, n_agents)
        copyto!(plank.data.z, 1, cpu_z, 1, n_agents)
        copyto!(plank.data.pool_x, 1, cpu_pool_x, 1, n_agents)
        copyto!(plank.data.pool_y, 1, cpu_pool_y, 1, n_agents)
        copyto!(plank.data.pool_z, 1, cpu_pool_z, 1, n_agents)
        copyto!(plank.data.mature, 1, cpu_mature, 1, n_agents)
        copyto!(plank.data.vis_prey, 1, cpu_vis_prey, 1, n_agents)
        
        # Initialize other fields
        @views plank.data.alive[1:n_agents] .= 1.0
        @views plank.data.generation[1:n_agents] .= 1
        @views plank.data.energy[1:n_agents] .= plank.data.biomass_school[1:n_agents] .* plank.p.Energy_density[2][sp] .* 0.2
        @views plank.data.gut_fullness[1:n_agents] .= CUDA.rand(Float32, n_agents)
        @views plank.data.age[1:n_agents] .= plank.p.Larval_Duration[2][sp]+1
        
        # Set remaining unused slots to non-alive
        @views plank.data.alive[n_agents+1:end] .= 0.0
    end
    return plank, species_counter
end

function generate_individuals(params::Dict, arch::Architecture, Nsp::Int32, B::Vector{Float32}, maxN::Int64, depths::MarineDepths, capacities, dt::Int32, envi::MarineEnvironment, start_date::Date)
    plank_names = Symbol[]
    plank_data=[]

    daily_birth_counters = zeros(Int,Nsp)

    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_individuals(arch, params, maxN)
        plank, species_counter = initialize_individuals(arch, plank, B[i], i, depths, capacities, dt, envi, params, start_date)
        daily_birth_counters[i] = species_counter
        push!(plank_names, name)
        push!(plank_data, plank)
    end

    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks), daily_birth_counters
end

# ===================================================================
# Resource Creation and Dynamics
# ===================================================================

@kernel function initialize_resources_kernel!(
    resource_biomass, 
    resource_capacity, 
    capacities, 
    total_biomass_targets,
    total_capacity_sum,
    normalized_vertical_profiles,
    n_spec
)
    lon, lat, depth_idx, res_sp = @index(Global, NTuple)

    # Get the total biomass target for this species
    target_biomass = total_biomass_targets[res_sp]
    
    # Get the sum of all habitat suitability values across the map for this species
    total_sum = total_capacity_sum[res_sp]
    
    if target_biomass > 0 && total_sum > 0
        # Calculate the average annual habitat suitability for this specific cell (lon, lat)
        avg_capacity_local = 0.0f0
        for month in 1:size(capacities, 3)
            avg_capacity_local += capacities[lon, lat, month, res_sp + n_spec]
        end
        avg_capacity_local /= size(capacities, 3)

        # Determine this horizontal cell's proportion of the total suitability
        horizontal_proportion = avg_capacity_local / total_sum
        
        # Allocate a portion of the total biomass to this water column (lon, lat)
        column_biomass = target_biomass * horizontal_proportion
        
        # Get the vertical proportion for this specific depth layer
        vertical_proportion = normalized_vertical_profiles[depth_idx, res_sp]
        
        # Calculate the final biomass for this specific 3D cell
        final_biomass = column_biomass * vertical_proportion
        
        if final_biomass > 0
            resource_biomass[lon, lat, depth_idx, res_sp] = final_biomass
            resource_capacity[lon, lat, depth_idx, res_sp] = final_biomass * 2.0f0
        end
    end
end

function initialize_resources(
    traits::DataFrame, n_spec::Int32, n_resource::Int32, 
    depths::MarineDepths, capacities::AbstractArray, arch::Architecture
)
    grid = depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    
    resource_biomass = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_resource))
    resource_capacity = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_resource))
    
    lonmax = grid[findfirst(grid.Name .== "xurcorner"), :Value][1]
    lonmin = grid[findfirst(grid.Name .== "xllcorner"), :Value][1]
    latmax = grid[findfirst(grid.Name .== "yulcorner"), :Value][1]
    latmin = grid[findfirst(grid.Name .== "yllcorner"), :Value][1]

    mean_lat_rad = deg2rad((latmin + latmax) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(mean_lat_rad)
    area_km2 = abs(latmax - latmin) * km_per_deg_lat * abs(lonmax - lonmin) * km_per_deg_lon

    total_biomass_targets_cpu = Float32.(traits.Biomass .* area_km2 .* 1e6)
    # --- 1. PREPARE DISTRIBUTION DATA ON CPU ---

    total_capacity_sum_cpu = zeros(Float32, n_resource)
    capacities_cpu = Array(capacities)
    for res_sp in 1:n_resource
        avg_capacity_map = mean(capacities_cpu[:, :, :, res_sp + n_spec], dims=3)[:,:,1]
        total_capacity_sum_cpu[res_sp] = sum(avg_capacity_map)
    end

    max_depth = Int(grid[grid.Name .== "depthmax", :Value][1])
    depth_res_m = max_depth / depthres
    depths_m = [(d - 0.5f0) * depth_res_m for d in 1:depthres] # Center of each depth bin
    
    # Create a matrix to hold the profile for each resource species
    vertical_profiles_cpu = zeros(Float32, depthres, n_resource)
    night_profs = depths.resource_night # Assumes this field exists

    for res_sp in 1:n_resource
        # Use a Gaussian mixture model for the profile, similar to agent initialization
        mu1 = night_profs[res_sp, "mu1"]; mu2 = night_profs[res_sp, "mu2"]
        sigma1 = night_profs[res_sp, "sigma1"]; sigma2 = night_profs[res_sp, "sigma2"]
        lambda1 = night_profs[res_sp, "lambda1"]
        
        # Calculate the profile for this species
        profile = lambda1 .* pdf.(Normal(mu1, sigma1), depths_m) .+ (1-lambda1) .* pdf.(Normal(mu2, sigma2), depths_m)
        
        # Normalize the profile so it sums to 1
        sum_prof = sum(profile)
        if sum_prof > 0
            vertical_profiles_cpu[:, res_sp] .= profile ./ sum_prof
        end
    end

    # --- 2. UPLOAD DATA TO GPU ---
    total_biomass_targets = array_type(arch)(total_biomass_targets_cpu)
    total_capacity_sum = array_type(arch)(total_capacity_sum_cpu)
    normalized_vertical_profiles = array_type(arch)(vertical_profiles_cpu)

    # --- 3. LAUNCH THE KERNEL ---
    # FIX: Ensure all launch dimensions are Int64 to prevent MethodError
    kernel_dims = (lonres, latres, depthres, Int64(n_resource))
    kernel! = initialize_resources_kernel!(device(arch), (8, 8, 4, 1), kernel_dims)
    kernel!(
        resource_biomass, 
        resource_capacity, 
        capacities, 
        total_biomass_targets,
        total_capacity_sum,
        normalized_vertical_profiles,
        n_spec
    )
    
    KernelAbstractions.synchronize(device(arch))
    return (biomass = resource_biomass, capacity = resource_capacity)
end

# Kernel for resource growth
@kernel function resource_growth_kernel!(biomass_grid, capacity_grid, per_timestep_rates)
    lon, lat, depth, sp = @index(Global, NTuple)
    biomass = biomass_grid[lon, lat, depth, sp]
    capacity = capacity_grid[lon, lat, depth, sp]
    rate = per_timestep_rates[sp]

    if biomass > 0 && capacity > 0
        @inbounds biomass_grid[lon, lat, depth, sp] = biomass + rate * biomass * (1.0 - biomass / capacity)
    elseif biomass <= 1E-9 && capacity > 0 #Assure that biomass should not be 0 for future growth
        @inbounds biomass_grid[lon, lat, depth, sp] = 1E-9
    end

    @inbounds biomass_grid[lon, lat, depth, sp] = capacity / 2f0 # keep these population sizes static for now.

end

# Launcher for resource growth
function resource_growth!(model::MarineModel)
    arch = model.arch
    # Growth rates from the trait table are treated as ANNUAL rates
    annual_growth_rates = model.resource_trait.Growth
    
    # Convert annual rate to a per-timestep rate
    minutes_per_year = 365.0f0 * 1440.0f0
    per_timestep_rates = array_type(arch)(Float32.(annual_growth_rates ./ (minutes_per_year / model.dt)))

    kernel! = resource_growth_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    kernel!(model.resources.biomass, model.resources.capacity, per_timestep_rates)
    KernelAbstractions.synchronize(device(arch))
end

# ===================================================================
# Reproduction System
# ===================================================================
function calculate_new_offspring_cpu(p_cpu, parent_data, repro_energy_list, spawn_val, sp, current_date::Date,daily_births::Int)
    # --- 1. Calculate eggs per parent (Unchanged) ---
    egg_volume = 0.15 .* parent_data.biomass_ind .^ 0.14
    egg_energy = 2.15 .* egg_volume .^ 0.77
    spent_energy = repro_energy_list .* spawn_val
    num_eggs_per_parent = floor.(Int, spent_energy ./ (egg_energy .+ 1f-9) .* p_cpu.Sex_Ratio.second[sp] .* p_cpu.Hatch_Survival.second[sp])

    # --- 2. NEW: Group parent indices by their grid cell ---
    # The dictionary key is the cell (px,py,pz), the value is a list of parent indices in that cell.
    cell_groups = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
    for i in 1:length(parent_data.x)
        # Only consider parents that will actually produce eggs
        if num_eggs_per_parent[i] > 0
            cell_key = (parent_data.pool_x[i], parent_data.pool_y[i], parent_data.pool_z[i])
            if !haskey(cell_groups, cell_key)
                cell_groups[cell_key] = Int[]
            end
            push!(cell_groups[cell_key], i)
        end
    end

    if isempty(cell_groups); return nothing; end

    # --- 3. Initialize empty vectors for the new MERGED agents ---
    new_unique_ids = Int64[]
    new_x, new_y, new_z = Float32[], Float32[], Float32[]
    new_pool_x, new_pool_y, new_pool_z = Int[], Int[], Int[]
    new_generation = Int[]
    new_abundance = Float32[]
    new_biomass_school = Float32[]
    new_energy = Float32[]
    
    # --- 4. Setup for ID generation ---
    date_int = parse(Int, Dates.format(current_date, "mmddyy"))
    species_counter = 0

    for (cell_key, parent_indices_in_cell) in cell_groups
        
        # Sum the total eggs from all parents in this cell
        total_eggs_in_cell = sum(num_eggs_per_parent[parent_indices_in_cell])

        if total_eggs_in_cell > 0
            # The new agent's properties are based on the first parent in the group
            first_parent_idx = parent_indices_in_cell[1]

            # A. Generate ONE unique ID for this new merged agent
            species_counter += 1
            new_id = (Int64(species_counter) * 100000000) + (Int64(sp) * 1000000) + Int64(date_int)
            push!(new_unique_ids, new_id)

            # B. Add the location/generation from the first parent
            push!(new_x, parent_data.x[first_parent_idx])
            push!(new_y, parent_data.y[first_parent_idx])
            push!(new_z, parent_data.z[first_parent_idx])
            push!(new_pool_x, parent_data.pool_x[first_parent_idx])
            push!(new_pool_y, parent_data.pool_y[first_parent_idx])
            push!(new_pool_z, parent_data.pool_z[first_parent_idx])
            push!(new_generation, parent_data.generation[first_parent_idx] + 1)

            # C. The new agent's abundance is the SUM of all eggs from this cell
            push!(new_abundance, Float32(total_eggs_in_cell))
            
            # D. Calculate biomass and energy for this agent based on its total abundance
            larval_size = Float32(p_cpu.Larval_Size.second[sp])
            lwr_a = p_cpu.LWR_a.second[sp]
            lwr_b = p_cpu.LWR_b.second[sp]
            energy_ed = p_cpu.Energy_density.second[sp]
            
            biomass_ind = lwr_a * (larval_size / 10.0f0) ^ lwr_b
            biomass_sch = biomass_ind * total_eggs_in_cell
            
            push!(new_biomass_school, biomass_sch)
            push!(new_energy, biomass_sch * energy_ed * 0.2f0)
        end
    end

    daily_births += species_counter

    if isempty(new_unique_ids); return nothing; end

    # --- 6. Calculate remaining properties and return the NamedTuple ---
    num_new_agents = length(new_unique_ids)
    larval_size = Float32(p_cpu.Larval_Size.second[sp])
    lwr_a = p_cpu.LWR_a.second[sp]
    lwr_b = p_cpu.LWR_b.second[sp]
    
    return (
        unique_id = new_unique_ids,
        x = new_x, y = new_y, z = new_z,
        length = fill(larval_size, num_new_agents),
        abundance = new_abundance,
        biomass_ind = fill(lwr_a * (larval_size / 10.0f0) ^ lwr_b, num_new_agents),
        biomass_school = new_biomass_school,
        energy = new_energy,
        gut_fullness = fill(1.0f0, num_new_agents),
        age = zeros(Float32, num_new_agents),
        pool_x = new_pool_x, pool_y = new_pool_y, pool_z = new_pool_z,
        generation = new_generation
    ), daily_births
end

# Manages the GPU/CPU data transfer for reproduction
function process_reproduction!(model::MarineModel, sp::Int32, spawn_val::Real)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p_cpu = model.individuals.animals[sp].p

    repro_energy_all = Array(data.repro_energy)
    repro_inds = findall(repro_energy_all .> 0)
    if isempty(repro_inds); return; end

    parent_data_cpu = (
        x = Array(data.x[repro_inds]), y = Array(data.y[repro_inds]), z = Array(data.z[repro_inds]),
        pool_x = Array(data.pool_x[repro_inds]), pool_y = Array(data.pool_y[repro_inds]), pool_z = Array(data.pool_z[repro_inds]),
        biomass_ind = Array(data.biomass_ind[repro_inds]),generation = Array(data.generation[repro_inds])
    )

    new_offspring = calculate_new_offspring_cpu(p_cpu, parent_data_cpu, repro_energy_all[repro_inds], spawn_val, sp,current_date)
    if new_offspring === nothing; return; end
    
    num_new = length(new_offspring.x)
    if num_new == 0; return; end

    dead_slots = findall(Array(data.alive) .== 0)
    num_to_add = min(num_new, length(dead_slots))
    if num_to_add < num_new
        @warn "Not enough space for all new offspring. Discarding $(num_new - num_to_add) individuals."
    end

    if num_to_add > 0
        slots_to_fill = @view dead_slots[1:num_to_add]
        for (field, values) in pairs(new_offspring)
            dest_view = @view getproperty(data, field)[slots_to_fill]
            source_view = @view values[1:num_to_add]
            copyto!(dest_view, source_view)
        end
        @view(data.alive[slots_to_fill]) .= 1.0
    end
    @view(data.repro_energy[repro_inds]) .= 0.0
end