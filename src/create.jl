# ===================================================================
# Agent Creation and Initialization
# ===================================================================

function construct_individuals(arch::Architecture, params::Dict, maxN)
    # Defines the full data structure for an agent, including all state variables
    # and fields needed for spatial indexing and other calculations.
    rawdata = StructArray(
        x = zeros(Float64,maxN), y = zeros(Float64,maxN), z = zeros(Float64,maxN),
        length = zeros(Float64,maxN), abundance = zeros(Float64,maxN),
        biomass_ind = zeros(Float64,maxN), biomass_school = zeros(Float64,maxN),
        energy = zeros(Float64,maxN), gut_fullness = zeros(Float64,maxN),
        cost = zeros(Float64,maxN), pool_x = zeros(Int,maxN), pool_y = zeros(Int,maxN),
        pool_z = zeros(Int,maxN), active = zeros(Float64,maxN),
        ration = zeros(Float64,maxN), alive = zeros(Float64,maxN),
        vis_prey = zeros(Float64,maxN), mature = zeros(Float64,maxN),
        age=zeros(Float64,maxN),
        cell_id = zeros(Int, maxN),
        sorted_id = zeros(Int, maxN),
        repro_energy = zeros(Float64, maxN),
        best_prey_dist = zeros(Float32, maxN),
        best_prey_idx = zeros(Int, maxN),
        best_prey_sp = zeros(Int, maxN),
        best_prey_type = zeros(Int, maxN),
        successful_ration = zeros(Float64, maxN),
        temp_idx = zeros(Int, maxN),
        cell_starts = zeros(Int, maxN),
        cell_ends = zeros(Int, maxN),
        mig_status = zeros(Float64, maxN),
        target_z = zeros(Float64, maxN),
        interval = zeros(Float64, maxN),
        dives_remaining = zeros(Int, maxN)
    )

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Dive_Interval,:Min_Prey,:LWR_b, :Surface_Interval,:W_mat,:SpeciesLong, 
                 :LWR_a, :Larval_Size,:Max_Prey, :Max_Size, :Dive_Max,:School_Size,:Taxa,
                 :Larval_Duration, :Sex_Ratio,:SpeciesShort,:FLR_b, :Dive_Min,:Handling_Time,
                 :FLR_a,:Energy_density, :Hatch_Survival, :MR_type, :Swim_velo, :Biomass, :Type)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function initialize_individuals(plank, B::Float64, sp::Int, depths::MarineDepths, capacities, dt, envi::MarineEnvironment)
    # This function uses the "Gather, Compute, Update" pattern to be architecture-compliant.
    
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

    cpu_lengths, cpu_biomass_ind, cpu_biomass_school = Float64[], Float64[], Float64[]
    
    # --- 2. Generate core data on the CPU until target biomass is met ---
    current_b = 0.0
    μ, σ = lognormal_params_from_maxsize(max_size)
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

    # --- 3. Generate remaining data in efficient, vectorized calls on the CPU ---
    if n_agents > 0
        if n_agents > length(plank.data.x)
            @warn "Generated $n_agents agents, but only have space for $(length(plank.data.x)). Truncating."
            n_agents = length(plank.data.x)
        end

        cpu_abundance = fill(Float64(school_size), n_agents)
        
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
        remaining_dives = 1440/(plank.p.Surface_Interval[2][sp] + plank.p.Dive_Interval[2][sp])
        
        # --- 4. Copy all data from CPU arrays to the target device (CPU or GPU) in one batch ---
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
        @views plank.data.energy[1:n_agents] .= plank.data.biomass_school[1:n_agents] .* plank.p.Energy_density[2][sp] .* 0.2
        @views plank.data.gut_fullness[1:n_agents] .= 0.2
        @views plank.data.age[1:n_agents] .= plank.p.Larval_Duration[2][sp]
        @views plank.data.dives_remaining[1:n_agents] .= remaining_dives
        
        # Set remaining unused slots to non-alive
        @views plank.data.alive[n_agents+1:end] .= 0.0
    end
    return plank.data
end

function generate_individuals(params::Dict, arch::Architecture, Nsp::Int, B, maxN::Int, depths::MarineDepths, capacities, dt, envi::MarineEnvironment)
    plank_names = Symbol[]
    plank_data=[]
    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_individuals(arch, params, maxN)
        initialize_individuals(plank, B[i], i, depths, capacities, dt, envi)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end

# ===================================================================
# Resource Creation and Dynamics
# ===================================================================

@kernel function initialize_resources_kernel!(
    resource_biomass, resource_capacity, capacities, traits_gpu, 
    n_spec, max_depth, depth_res_m
)
    lon, lat, depth_idx = @index(Global, NTuple)

    # --- Primary Guard: Check if the cell is a water cell ---
    for res_sp in 1:size(resource_biomass, 4)
        # Habitat capacity for this resource in this cell (average over months)
        avg_capacity = 0.0
        for month in 1:size(capacities, 3)
            avg_capacity += capacities[lon, lat, month, res_sp + n_spec]
        end
        avg_capacity /= size(capacities, 3)
        
        if avg_capacity > 0
            depth_m = (depth_idx - 1) * depth_res_m
            depth_pref = exp(-0.005 * depth_m)
            density = traits_gpu.Biomass[res_sp] * 1_000_000
            biomass = density * avg_capacity * depth_pref
            
            resource_biomass[lon, lat, depth_idx, res_sp] = biomass
            resource_capacity[lon, lat, depth_idx, res_sp] = biomass
        end
    end
end

function initialize_resources(
    traits::DataFrame, n_spec::Int, n_resource::Int, 
    depths::MarineDepths, capacities::AbstractArray, arch::Architecture
)
    grid = depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    
    resource_biomass = array_type(arch)(zeros(Float64, lonres, latres, depthres, n_resource))
    resource_capacity = array_type(arch)(zeros(Float64, lonres, latres, depthres, n_resource))
    
    traits_gpu = (; (Symbol(c) => array_type(arch)(traits[:, c]) for c in names(traits))...)
    max_depth = Int(grid[grid.Name .== "depthmax", :Value][1])
    depth_res_m = max_depth / depthres

    # --- Create the land mask on the correct device ---
    # Assumes bathymetry > 0 is water. Adjust if your convention is different.

    kernel! = initialize_resources_kernel!(device(arch), (8, 8, 4), (lonres, latres, depthres))
    kernel!(
        resource_biomass, 
        resource_capacity, 
        capacities, 
        traits_gpu, 
        n_spec, 
        max_depth, 
        depth_res_m
    )
    
    KernelAbstractions.synchronize(device(arch))
    return (biomass = resource_biomass, capacity = resource_capacity)
end

# Kernel for resource growth
@kernel function resource_growth_kernel!(biomass_grid, capacity_grid, resource_trait)
    lon, lat, depth, sp = @index(Global, NTuple)
    biomass = biomass_grid[lon, lat, depth, sp]
    capacity = capacity_grid[lon, lat, depth, sp]
    if biomass > 0 && capacity > 0
        rate = resource_trait.Growth[sp]
        @inbounds biomass_grid[lon, lat, depth, sp] = biomass + rate * biomass * (1.0 - biomass / capacity)
    end
end

# Launcher for resource growth
function resource_growth!(model::MarineModel)
    arch = model.arch
    trait_gpu = (; (Symbol(c) => array_type(arch)(model.resource_trait[:, c]) for c in names(model.resource_trait))...)
    kernel! = resource_growth_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    kernel!(model.resources.biomass, model.resources.capacity, trait_gpu)
    KernelAbstractions.synchronize(device(arch))
end

# ===================================================================
# Reproduction System
# ===================================================================

# CPU-only function to calculate new offspring properties
function calculate_new_offspring_cpu(p_cpu, parent_data, repro_energy_list, spawn_val, sp)
    egg_volume = 0.15 .* parent_data.biomass_ind .^ 0.14
    egg_energy = 2.15 .* egg_volume .^ 0.77
    spent_energy = repro_energy_list .* spawn_val .* 0.9
    num_eggs_per_parent = floor.(Int, spent_energy ./ egg_energy .* p_cpu.Sex_Ratio.second[sp] .* p_cpu.Hatch_Survival.second[sp])

    total_eggs = sum(num_eggs_per_parent)
    if total_eggs == 0; return nothing; end

    new_x, new_y, new_z = zeros(total_eggs), zeros(total_eggs), zeros(total_eggs)
    new_pool_x, new_pool_y, new_pool_z = zeros(Int, total_eggs), zeros(Int, total_eggs), zeros(Int, total_eggs)
    new_length = zeros(total_eggs)
    
    current_idx = 1
    for i in 1:length(parent_data.x)
        num_eggs = num_eggs_per_parent[i]
        if num_eggs > 0
            indices = current_idx:(current_idx + num_eggs - 1)
            new_x[indices] .= parent_data.x[i]
            new_y[indices] .= parent_data.y[i]
            new_z[indices] .= parent_data.z[i]
            new_pool_x[indices] .= parent_data.pool_x[i]
            new_pool_y[indices] .= parent_data.pool_y[i]
            new_pool_z[indices] .= parent_data.pool_z[i]
            rand_lengths = rand(Float32, num_eggs) .* p_cpu.Larval_Size.second[sp]
            new_length[indices] .= rand_lengths
            current_idx += num_eggs
        end
    end

    new_biomass_ind = p_cpu.LWR_a.second[sp] .* (new_length ./ 10.0) .^ p_cpu.LWR_b.second[sp]
    
    return (
        x=new_x, y=new_y, z=new_z, length=new_length,
        abundance=fill(Float64(p_cpu.School_Size.second[sp]), total_eggs),
        biomass_ind=new_biomass_ind,
        biomass_school=new_biomass_ind .* p_cpu.School_Size.second[sp],
        energy = new_biomass_ind .* p_cpu.School_Size.second[sp] .* p_cpu.Energy_density.second[sp] .* 0.2,
        gut_fullness = fill(0.2, total_eggs),
        age = zeros(total_eggs),
        pool_x=new_pool_x, pool_y=new_pool_y, pool_z=new_pool_z
    )
end

# Manages the GPU/CPU data transfer for reproduction
function process_reproduction!(model::MarineModel, sp::Int, spawn_val::Real)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p_cpu = model.individuals.animals[sp].p

    repro_energy_all = Array(data.repro_energy)
    repro_inds = findall(repro_energy_all .> 0)
    if isempty(repro_inds); return; end

    parent_data_cpu = (
        x = Array(data.x[repro_inds]), y = Array(data.y[repro_inds]), z = Array(data.z[repro_inds]),
        pool_x = Array(data.pool_x[repro_inds]), pool_y = Array(data.pool_y[repro_inds]), pool_z = Array(data.pool_z[repro_inds]),
        biomass_ind = Array(data.biomass_ind[repro_inds])
    )

    new_offspring = calculate_new_offspring_cpu(p_cpu, parent_data_cpu, repro_energy_all[repro_inds], spawn_val, sp)
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
