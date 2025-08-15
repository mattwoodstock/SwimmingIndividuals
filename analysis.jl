# ===================================================================
# Top-Level Output Functions
# ===================================================================

function timestep_results(sim::MarineSimulation)
    model = sim.model
    outputs = sim.outputs
    arch = model.arch
    ts = Int(model.iteration)
    run = Int(sim.run)

    # Get the base results directory from the model's files DataFrame
    files_df = model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]

    # Construct full paths for output subdirectories
    individual_dir = joinpath(res_dir, "Individual")
    population_dir = joinpath(res_dir, "Population")

    if ts == 1 && run == 1
        # Recreate directories at the start of a run
        isdir(individual_dir) && rm(individual_dir, recursive=true)
        mkpath(individual_dir)
        isdir(population_dir) && rm(population_dir, recursive=true)
        mkpath(population_dir)
    end

    # --- Gather individual data for CSV output ---
    Sp, Ind, x, y, z, lengths, abundance, biomass, gut_fullness, ration,energy,cost = [],[],[],[],[],[],[],[],[],[],[],[],[]

    for (species_index, animal) in enumerate(model.individuals.animals)
        spec_dat = animal.data
        
        cpu_alive_mask = Array(spec_dat.alive) .== 1.0
        alive_indices = findall(cpu_alive_mask)
        if isempty(alive_indices); continue; end

        append!(Sp, fill(species_index, length(alive_indices)))
        append!(Ind, spec_dat.unique_id[alive_indices])
        append!(x, Array(spec_dat.x[alive_indices]))
        append!(y, Array(spec_dat.y[alive_indices]))
        append!(z, Array(spec_dat.z[alive_indices]))
        append!(lengths, Array(spec_dat.length[alive_indices]))
        append!(abundance, Array(spec_dat.abundance[alive_indices]))
        append!(biomass, Array(spec_dat.biomass_school[alive_indices]))
        append!(gut_fullness, Array(spec_dat.gut_fullness[alive_indices]))
        append!(ration, Array(spec_dat.ration[alive_indices]))
        append!(energy, Array(spec_dat.energy[alive_indices]))
        append!(cost, Array(spec_dat.cost[alive_indices]))

    end

    if !isempty(Sp)
        df = DataFrame(Species=Sp, Individual=Ind, X=x, Y=y, Z=z, Length=lengths, Abundance=abundance, Biomass=biomass,Fullness = gut_fullness,Ration = ration, Energy = energy,Cost = cost)
        # Use the constructed path for writing the CSV file
        csv_path = joinpath(individual_dir, "IndividualResults_$run-$ts.csv")
        CSV.write(csv_path, df)
    end
    
    # --- Calculate and save population-scale results ---
    # 1. Calculate size-resolved biomass first
    init_biomass_by_size!(model, outputs) 
    
    # 2. Calculate all three mortality types
    M = instantaneous_mortality(outputs, arch)
    F = fishing_mortality(outputs, arch)
    S = starvation_mortality(outputs, arch) # New call
    
    # 3. Copy results to CPU for saving
    cpu_M = Array(M)
    cpu_F = Array(F)
    cpu_S = Array(S) # New data
    cpu_DC = Array(outputs.consumption)
    cpu_biomass = Array(outputs.biomass) # Save new biomass grid

    # 4. Save all results to the HDF5 file
    population_dir = joinpath(model.files[model.files.File .== "res_dir", :Destination][1], "Population")
    h5_path = joinpath(population_dir, "Population_Results_$(run)-$(ts).h5")
    h5open(h5_path, "w") do file
        write(file, "M", cpu_M)
        write(file, "F", cpu_F)
        write(file, "S", cpu_S)
        write(file, "Diet", cpu_DC)
        write(file, "Biomass", cpu_biomass)
    end

    # --- Reset output arrays on the device ---
    fill!(outputs.mortalities, 0)
    fill!(outputs.Fmort, 0)
    fill!(outputs.consumption, 0.0f0)
    fill!(outputs.Smort, 0)
    
    return nothing
end

# This function is entirely CPU-based and does not need modification.
function fishery_results(sim::MarineSimulation)
    ts = Int(sim.model.iteration)
    run = Int(sim.run)
    fisheries = sim.model.fishing

    # Get the base results directory from the model's files DataFrame
    files_df = sim.model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]
    
    # Construct full path for the fishery output subdirectory
    fish_dir = joinpath(res_dir, "Fishery")

    if ts == 1 && run == 1
        isdir(fish_dir) && rm(fish_dir, recursive=true)
        mkpath(fish_dir)
    end

    name, quotas, catches_t, catches_ind = [], [], [], []
    effort, cpue, mean_len, bycatch_t, bycatch_n = [], [], [], [], []  

    for fishery in sim.model.fishing
        push!(name, fishery.name)
        push!(quotas, fishery.quota)
        push!(catches_t, fishery.cumulative_catch)
        push!(catches_ind, fishery.cumulative_inds)
        
        push!(effort, fishery.effort_days)
        current_cpue = fishery.effort_days > 0 ? fishery.cumulative_catch / fishery.effort_days : 0.0
        push!(cpue, current_cpue)
        push!(mean_len, fishery.mean_length_catch)
        push!(bycatch_t, fishery.bycatch_tonnage)
        push!(bycatch_n, fishery.bycatch_inds)
    end

    df = DataFrame(
        Name=name, Quota=quotas, Tonnage=catches_t, Individuals=catches_ind,
        Effort_Days=effort, CPUE_T_per_Day=cpue, Mean_Length_mm=mean_len,
        Bycatch_Tonnage=bycatch_t, Bycatch_Individuals=bycatch_n
    )    # Use the constructed path for writing the CSV file
    csv_path = joinpath(fish_dir, "FisheryResults_$run-$ts.csv")
    CSV.write(csv_path, df)
end

function assemble_diagnostic_results(model::MarineModel, run::Int32, ts::Int32)
    
    # Get the base results directory from the model's files DataFrame
    files_df = model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]

    # Construct the full path for the run-specific diagnostics directory
    output_dir = joinpath(res_dir, "diags", "run_$run")
    
    if ts == 1 && isdir(output_dir)
        rm(output_dir, recursive=true)
    end
    mkpath(output_dir)

    # --- 2. Assemble Agent-Based Results ---
    sp_ids, agent_ids = Int[], Int[]
    x_coords, y_coords, z_coords = Float32[], Float32[], Float32[]
    pool_x_coords, pool_y_coords = Int[], Int[]
    daily_rations, repro_energy, cost = Float32[], Float32[], Float32[]
    biomass_ind, biomass_school = Float32[], Float32[]

    for sp in 1:model.n_species
        agents = model.individuals.animals[sp].data
        
        cpu_alive_mask = Array(agents.alive) .== 1.0
        alive_indices = findall(cpu_alive_mask)
        if isempty(alive_indices); continue; end

        append!(sp_ids, fill(sp, length(alive_indices)))
        append!(agent_ids, Array(agents.unique_id[alive_indices]))
        append!(x_coords, Array(agents.x[alive_indices]))
        append!(y_coords, Array(agents.y[alive_indices]))
        append!(z_coords, Array(agents.z[alive_indices]))
        append!(pool_x_coords, Array(agents.pool_x[alive_indices]))
        append!(pool_y_coords, Array(agents.pool_y[alive_indices]))
        append!(daily_rations, Array(agents.ration[alive_indices]))
        append!(repro_energy, Array(agents.repro_energy[alive_indices]))
        append!(cost, Array(agents.cost[alive_indices]))
        append!(biomass_ind, Array(agents.biomass_ind[alive_indices]))
        append!(biomass_school, Array(agents.biomass_school[alive_indices]))
    end

    agent_results_df = DataFrame(
        SpeciesID = sp_ids, AgentID = agent_ids, 
        X = x_coords, Y = y_coords, Z = z_coords,
        pool_x = pool_x_coords, pool_y = pool_y_coords,
        DailyRation = daily_rations, ReproEnergy = repro_energy, Cost = cost,
        IndividualBiomass = biomass_ind, SchoolBiomass = biomass_school
    )
    CSV.write(joinpath(output_dir, "agent_results_$ts.csv"), agent_results_df)


    # --- 3. Assemble Spatially Explicit Resource Biomass ---
    resource_biomass_cpu = Array(model.resources.biomass)
    lonres, latres, depthres, nres = size(resource_biomass_cpu)
    lon_indices, lat_indices, depth_indices = Int[], Int[], Int[]
    res_sp_ids, biomass_densities = Int[], Float32[]

    for sp in 1:nres, d in 1:depthres, l in 1:latres, lon in 1:lonres
        biomass = resource_biomass_cpu[lon, l, d, sp]
        if biomass > 0
            push!(lon_indices, lon)
            push!(lat_indices, l)
            push!(depth_indices, d)
            push!(res_sp_ids, sp)
            push!(biomass_densities, biomass)
        end
    end

    resource_results_df = DataFrame(
        LonIndex = lon_indices, LatIndex = lat_indices, DepthIndex = depth_indices,
        ResourceID = res_sp_ids, BiomassDensity = biomass_densities
    )
    CSV.write(joinpath(output_dir, "resource_results_$ts.csv"), resource_results_df)

    return nothing
end

function test_prey_detection(model::MarineModel)
    @info "--- Running Prey Detection Test ---"
    
    # --- 1. Setup: Select a single predator and prey ---
    pred_sp = 1
    prey_sp = 2
    res_sp = 1
    
    pred_data = model.individuals.animals[pred_sp].data
    prey_data = model.individuals.animals[prey_sp].data
    pred_params = model.individuals.animals[pred_sp].p
    prey_params = model.individuals.animals[prey_sp].p
    
    pred_idx = findfirst(Array(pred_data.alive) .== 1.0)
    prey_idx = findfirst(Array(prey_data.alive) .== 1.0)
    
    if pred_idx === nothing || prey_idx === nothing
        @warn "Could not find a living predator or prey to conduct the test. Skipping."
        return
    end

    # --- 2. FOCAL PREY TEST ---
    @info "--- Testing for FOCAL prey detection... ---"
    
    pred_data_cpu = StructArray(NamedTuple(k => Array(v) for (k, v) in pairs(StructArrays.components(pred_data))))
    prey_data_cpu = StructArray(NamedTuple(k => Array(v) for (k, v) in pairs(StructArrays.components(prey_data))))

    # --- FIX: Ensure prey is a valid size for the predator ---
    pred_len = pred_data_cpu.length[pred_idx]
    min_size = pred_len * pred_params.Min_Prey.second[pred_sp]
    max_size = pred_len * pred_params.Max_Prey.second[pred_sp]
    
    # Set prey length to be in the middle of the acceptable range
    new_prey_len = (min_size + max_size) / 2.0
    prey_data_cpu.length[prey_idx] = new_prey_len
    
    # Update prey biomass to match its new length
    new_prey_biomass_ind = prey_params.LWR_a.second[prey_sp] * (new_prey_len / 10.0)^prey_params.LWR_b.second[prey_sp]
    prey_data_cpu.biomass_ind[prey_idx] = new_prey_biomass_ind
    prey_data_cpu.biomass_school[prey_idx] = new_prey_biomass_ind * prey_params.School_Size.second[prey_sp]
    
    @info "Test Predator Length: $(pred_len)mm. Valid prey size range: [$(min_size)mm, $(max_size)mm]."
    @info "Set Test Prey ID $prey_idx to length $(new_prey_len)mm."

    # Place the correctly-sized prey 1 meter in front of predator
    pred_x, pred_y, pred_z = pred_data_cpu.x[pred_idx], pred_data_cpu.y[pred_idx], pred_data_cpu.z[pred_idx]
    prey_data_cpu.x[prey_idx] = pred_x + 0.00001
    prey_data_cpu.y[prey_idx] = pred_y
    prey_data_cpu.z[prey_idx] = pred_z
    prey_data_cpu.pool_x[prey_idx] = pred_data_cpu.pool_x[pred_idx]
    prey_data_cpu.pool_y[prey_idx] = pred_data_cpu.pool_y[pred_idx]
    prey_data_cpu.pool_z[prey_idx] = pred_data_cpu.pool_z[pred_idx]
    
    copyto!(pred_data, pred_data_cpu)
    copyto!(prey_data, prey_data_cpu)
    
    @info "Placed Focal Prey ID $prey_idx at the location of Predator ID $pred_idx."

    calculate_distances_prey!(model, pred_sp, [pred_idx])

    best_idx = Array(pred_data.best_prey_idx[[pred_idx]])[1]
    best_sp = Array(pred_data.best_prey_sp[[pred_idx]])[1]
    best_dist = Array(pred_data.best_prey_dist[[pred_idx]])[1]

    if best_idx > 0 && best_sp == prey_sp
        println("✅ SUCCESS: Predator $pred_idx correctly identified Focal Prey $best_idx.")
        println("   - Distance: $(sqrt(best_dist)) meters")
    else
        println("❌ FAILURE: Predator $pred_idx did NOT find the focal prey.")
    end
    
    # --- 3. RESOURCE PREY TEST ---
    @info "\n--- Testing for RESOURCE prey detection... ---"

    res_biomass_cpu = Array(model.resources.biomass)
    pred_pool_x = pred_data_cpu.pool_x[pred_idx]
    pred_pool_y = pred_data_cpu.pool_y[pred_idx]
    pred_pool_z = pred_data_cpu.pool_z[pred_idx]
    
    res_biomass_cpu[pred_pool_x, pred_pool_y, pred_pool_z, res_sp] = 5000.0
    copyto!(model.resources.biomass, res_biomass_cpu)

    @info "Placed a dense patch of Resource ID $res_sp in cell ($pred_pool_x, $pred_pool_y, $pred_pool_z)."

    calculate_distances_prey!(model, pred_sp, [pred_idx])

    best_idx_res = Array(pred_data.best_prey_idx[[pred_idx]])[1]
    best_sp_res = Array(pred_data.best_prey_sp[[pred_idx]])[1]
    best_type_res = Array(pred_data.best_prey_type[[pred_idx]])[1]
    best_dist_res = Array(pred_data.best_prey_dist[[pred_idx]])[1]
    
    println("\n--- TEST RESULTS ---")
    if best_idx_res > 0 && best_sp_res == res_sp && best_type_res == 2
        println("✅ SUCCESS: Predator $pred_idx correctly identified Resource Prey patch $best_sp_res.")
        println("   - Nearest-Neighbor Distance: $(sqrt(best_dist_res)) meters")
    else
        println("❌ FAILURE: Predator $pred_idx did NOT find the resource prey patch.")
        println("   - This confirms a bug within the `find_best_prey_kernel!`'s resource search logic.")
    end
    println("--------------------\n")
    
    res_biomass_cpu[pred_pool_x, pred_pool_y, pred_pool_z, res_sp] = 0.0
    copyto!(model.resources.biomass, res_biomass_cpu)
end

# ===================================================================
# GPU-Compliant Analysis Kernels
# ===================================================================

"""
Calculates the abundance of agents in each grid cell AND for each size bin.
"""
@kernel function init_abundances_by_size_kernel!(
    abundance_out, 
    agents, 
    size_bin_thresholds, 
    sp_idx
)
    i = @index(Global) # Each thread handles one agent

    @inbounds if agents.alive[i] == 1.0f0
        # Get the agent's grid cell coordinates
        x = agents.pool_x[i]
        y = agents.pool_y[i]
        z = agents.pool_z[i]
        
        # Determine which size bin this agent belongs to
        size_bin = find_species_size_bin(agents.length[i], sp_idx, size_bin_thresholds)
        
        # Atomically add this agent's abundance to its cell and size bin
        if size_bin > 0
            @atomic abundance_out[x, y, z, sp_idx, size_bin] += agents.abundance[i]
        end
    end
end

"""
Calculates the BIOMASS of agents in each grid cell and size bin.
"""
@kernel function init_biomass_by_size_kernel!(
    biomass_out, 
    agents, 
    size_bin_thresholds, 
    sp_idx
)
    i = @index(Global) # Each thread handles one agent

    @inbounds if agents.alive[i] == 1.0f0
        x = agents.pool_x[i]; y = agents.pool_y[i]; z = agents.pool_z[i]
        size_bin = find_species_size_bin(agents.length[i], sp_idx, size_bin_thresholds)
        
        if size_bin > 0
            @atomic biomass_out[x, y, z, sp_idx, size_bin] += agents.biomass_school[i]
        end
    end
end

@kernel function sum_predation_mortality_kernel!(summed_mortality, raw_mortality)
    lon, lat, depth, pred_sp, prey_sp, pred_bin, prey_bin = @index(Global, NTuple)
    
    mort_val = raw_mortality[lon, lat, depth, pred_sp, prey_sp, pred_bin, prey_bin]
    if mort_val > 0
        # Atomically add the mortality to the 5D summary array, using the prey's index
        @atomic summed_mortality[lon, lat, depth, prey_sp, prey_bin] += mort_val
    end
end

@kernel function sum_fishing_mortality_kernel!(summed_mortality, raw_mortality)
    # Each thread now gets a 5D index for the OUTPUT array
    lon, lat, depth, prey_sp, prey_bin = @index(Global, NTuple)
    
    total_mort_for_this_cell = 0
    
    # This prevents multiple threads from writing to the same memory location.
    for fishery_idx in 1:size(raw_mortality, 4)
        mort_val = raw_mortality[lon, lat, depth, fishery_idx, prey_sp, prey_bin]
        if mort_val > 0
                @cuprintf("  -> Found %d mortality from fishery %d\n", mort_val, fishery_idx)
             total_mort_for_this_cell += mort_val
        end
    end
    
    if total_mort_for_this_cell > 0
        # A single, safe write operation for each thread
        summed_mortality[lon, lat, depth, prey_sp, prey_bin] = total_mort_for_this_cell
    end
end

function init_biomass_by_size!(model::MarineModel, outputs::MarineOutputs)
    arch = model.arch
    fill!(outputs.biomass, 0.0f0)

    for sp in 1:model.n_species
        agents = model.individuals.animals[sp].data
        n_agents = length(agents.x)
        if n_agents > 0
            kernel! = init_biomass_by_size_kernel!(device(arch), 256, (n_agents,))
            kernel!(outputs.biomass, agents, model.size_bin_thresholds, sp)
        end
    end
    
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

"""
Calculates instantaneous mortality rates (M and F) from the 7D mortality arrays.
This kernel now correctly uses the BIOMASS grid for its calculation.
"""
@kernel function mortality_rate_kernel!(Rate, biomass_by_size, summed_mortality)
    lon, lat, depth, prey_sp, prey_bin = @index(Global, NTuple)
    
    @inbounds mort_val = summed_mortality[lon, lat, depth, prey_sp, prey_bin]
    if mort_val > 0
        biomass_val = biomass_by_size[lon, lat, depth, prey_sp, prey_bin]
        
        @inbounds if biomass_val > 0
            FT = eltype(Rate)
            # mort_val is now total biomass lost for this size class
            mort_frac = FT(mort_val) / biomass_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            if mort_frac < FT(1.0)
                Rate[lon, lat, depth, prey_sp, prey_bin] = -log(FT(1.0) - mort_frac)
            else
                Rate[lon, lat, depth, prey_sp, prey_bin] = FT(10.0)
            end
        end
    end
end

@kernel function fishing_mortality_rate_kernel!(
    Rate, 
    biomass_by_size, 
    fishing_mortality
)
    # Each thread gets a 6D index for the Fmort array:
    # (lon, lat, depth, fishery, prey_sp, prey_bin)
    lon, lat, depth, fishery, prey_sp, prey_bin = @index(Global, NTuple)
    
    @inbounds mort_val = fishing_mortality[lon, lat, depth, fishery, prey_sp, prey_bin]
    if mort_val > 0
        # Biomass is indexed by the PREY's species and size bin (a 5D lookup)
        biomass_val = biomass_by_size[lon, lat, depth, prey_sp, prey_bin]
        
        @inbounds if biomass_val > 0
            FT = eltype(Rate)
            mort_frac = FT(mort_val) / biomass_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            if mort_frac < FT(1.0)
                Rate[lon, lat, depth, fishery, prey_sp, prey_bin] = -log(FT(1.0) - mort_frac)
            else
                # Use a large, finite number for 100% mortality
                Rate[lon, lat, depth, fishery, prey_sp, prey_bin] = FT(10.0)
            end
        end
    end
end

"""
Calculates instantaneous starvation mortality (S) from the 5D Smort array.
This kernel now correctly uses the BIOMASS grid for its calculation.
"""
@kernel function starvation_mortality_kernel!(Rate, biomass_by_size, starvation_mortality)
    lon, lat, depth, sp, size_bin = @index(Global, NTuple)
    
    @inbounds if starvation_mortality[lon, lat, depth, sp, size_bin] > 0
        biomass_val = biomass_by_size[lon, lat, depth, sp, size_bin]
        
        @inbounds if biomass_val > 0
            FT = eltype(Rate)
            mort_val = FT(starvation_mortality[lon, lat, depth, sp, size_bin])
            mort_frac = mort_val / biomass_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            if mort_frac < FT(1.0)
                Rate[lon, lat, depth, sp, size_bin] = -log(FT(1.0) - mort_frac)
            else
                Rate[lon, lat, depth, sp, size_bin] = FT(10.0)
            end
        end
    end
end

# ===================================================================
# Launcher Functions for Analysis
# ===================================================================


"""
This launcher now calculates both size-resolved abundances AND biomass.
"""
function init_abundances_and_biomass!(model::MarineModel, outputs::MarineOutputs)
    arch = model.arch
    
    # Reset both arrays before calculating
    fill!(outputs.abundance, 0.0f0)
    fill!(outputs.biomass, 0.0f0)

    for sp in 1:model.n_species
        agents = model.individuals.animals[sp].data
        n_agents = length(agents.x)
        if n_agents > 0
            # Launch abundance kernel
            abund_kernel! = init_abundances_by_size_kernel!(device(arch), 256, (n_agents,))
            abund_kernel!(outputs.abundance, agents, model.size_bin_thresholds, sp)
            
            # Launch biomass kernel
            biomass_kernel! = init_biomass_by_size_kernel!(device(arch), 256, (n_agents,))
            biomass_kernel!(outputs.biomass, agents, model.size_bin_thresholds, sp)
        end
    end
    
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

"""
Launcher for instantaneous predation mortality (M).
"""
function instantaneous_mortality(outputs::MarineOutputs, arch)
    # Create the 5D output array for the final rates
    M_dims = size(outputs.biomass)
    M = array_type(arch)(zeros(Float32, M_dims...))
    
    # Create a temporary 5D array to hold the summed mortality
    summed_M = array_type(arch)(zeros(Int32, M_dims...))
    
    # 1. Launch the summation kernel
    sum_kernel! = sum_predation_mortality_kernel!(device(arch))
    sum_kernel!(summed_M, outputs.mortalities, ndrange=size(outputs.mortalities))
    
    # 2. Launch the rate calculation kernel
    rate_kernel! = mortality_rate_kernel!(device(arch))
    rate_kernel!(M, outputs.biomass, summed_M, ndrange=size(M))
    
    KernelAbstractions.synchronize(device(arch))
    return M
end
"""
Launcher for instantaneous fishing mortality (F).
"""
function fishing_mortality(outputs::MarineOutputs, arch)
    # The output F array is 6D, the same size as the input Fmort
    F = array_type(arch)(zeros(Float32, size(outputs.Fmort)...))
    
    # --- FIX: Launch the new, dedicated 6D kernel ---
    kernel! = fishing_mortality_rate_kernel!(device(arch))
    # The ndrange is the size of the 6D Fmort array
    kernel!(F, outputs.biomass, outputs.Fmort, ndrange=size(F))
    # --- END FIX ---
    
    KernelAbstractions.synchronize(device(arch))
    return F
end

"""
Launcher for instantaneous starvation mortality (S).
"""
function starvation_mortality(outputs::MarineOutputs, arch)
    S = array_type(arch)(zeros(Float32, size(outputs.Smort)...))
    kernel! = starvation_mortality_kernel!(device(arch))
    kernel!(S, outputs.biomass, outputs.Smort, ndrange=size(S))
    KernelAbstractions.synchronize(device(arch))
    return S
end