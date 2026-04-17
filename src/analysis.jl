# ===================================================================
# Top-Level Output Functions
# ===================================================================

"""
    timestep_results(sim::MarineSimulation)

Gather individual-level data and population-level matrices for the 
current timestep and export them to CSV and HDF5 formats.
"""
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

    # FIX: Ensure directories exist before writing. 
    # The previous 'ts == 1' check failed because the first output occurs at ts = 4.
    !isdir(individual_dir) && mkpath(individual_dir)
    !isdir(population_dir) && mkpath(population_dir)

    # --- 1. Gather individual data for CSV output ---
    Sp, Ind, x, y, z, lengths, abundance, biomass, gut_fullness, ration_biomass, ration_energy, energy, cost, age, generation = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    for (species_index, animal) in enumerate(model.individuals.animals)
        spec_dat = animal.data
        
        # Pull data from the device to the CPU and filter for living agents
        alive_mask = Array(spec_dat.alive) .== 1.0
        
        append!(Sp, fill(species_index, count(alive_mask)))
        # Match unique_id as defined in the agent StructArray
        append!(Ind, Array(spec_dat.unique_id)[alive_mask])
        append!(x, Array(spec_dat.x)[alive_mask])
        append!(y, Array(spec_dat.y)[alive_mask])
        append!(z, Array(spec_dat.z)[alive_mask])
        append!(lengths, Array(spec_dat.length)[alive_mask])
        append!(abundance, Array(spec_dat.abundance)[alive_mask])
        # Match biomass_school as defined in the agent StructArray
        append!(biomass, Array(spec_dat.biomass_school)[alive_mask])
        append!(gut_fullness, Array(spec_dat.gut_fullness)[alive_mask])
        append!(ration_biomass, Array(spec_dat.ration_biomass)[alive_mask])
        append!(ration_energy, Array(spec_dat.ration_energy)[alive_mask])
        append!(energy, Array(spec_dat.energy)[alive_mask])
        append!(cost, Array(spec_dat.cost)[alive_mask])
        append!(age, Array(spec_dat.age)[alive_mask])
        append!(generation, Array(spec_dat.generation)[alive_mask])
    end
    
    # Save individual data to CSV
    # FIX: Changed naming convention to use underscores to match analysis scripts
    ind_df = DataFrame(
        Species = Sp, Individual = Ind, X = x, Y = y, Z = z, 
        Length = lengths, Abundance = abundance, Biomass = biomass, 
        Fullness = gut_fullness, Ration_b = ration_biomass, Ration_e = ration_energy, 
        Energy = energy, Cost = cost, Age = age, Generation = generation
    )
    CSV.write(joinpath(individual_dir, "IndividualResults_$(run)_$(ts).csv"), ind_df)

    # --- 2. Generate Population Arrays ---
    init_biomass_by_size!(model, outputs) 
    
    cpu_F = Array(outputs.Fmort)
    cpu_S = Array(outputs.Smort)
    cpu_DC = Array(outputs.consumption)
    cpu_biomass = Array(outputs.biomass)

    # --- 3. Save Population Results to HDF5 ---
    h5_path = joinpath(population_dir, "Population_Results_$(run)_$(ts).h5")
    h5open(h5_path, "w") do file
        file["F", deflate=3] = cpu_F
        file["S", deflate=3] = cpu_S
        file["Diet", deflate=3] = cpu_DC
        file["Biomass", deflate=3] = cpu_biomass
    end

    # --- 4. Reset output arrays on the device ---
    # Use 0.0f0 to match the 32-bit Float architecture on the GPU
    fill!(outputs.Fmort, 0.0f0)
    fill!(outputs.Smort, 0.0f0)
    fill!(outputs.consumption, 0.0f0)
end

"""
    resource_results(model::MarineModel, run::Integer, ts::Integer)

Export the current spatial biomass density of all resource groups.
Uses Integer type to handle both Int32 and Int64 inputs safely.
"""
function resource_results(model::MarineModel, run::Integer, ts::Integer)
    files_df = model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]
    
    resource_dir = joinpath(res_dir, "Resource")
    !isdir(resource_dir) && mkpath(resource_dir)
    
    # Gathering data for Resource results
    cpu_res = Array(model.resources.biomass)
    lonres, latres, depthres, n_res = size(cpu_res)
    
    lons, lats, depths, res_ids, biomass_density = [], [], [], [], []
    
    for r in 1:n_res, k in 1:depthres, j in 1:latres, i in 1:lonres
        val = cpu_res[i, j, k, r]
        if val > 1e-6 # Sparse threshold for CSV output
            push!(lons, i); push!(lats, j); push!(depths, k)
            push!(res_ids, r); push!(biomass_density, val)
        end
    end
    
    res_df = DataFrame(LonIndex=lons, LatIndex=lats, DepthIndex=depths, ResourceID=res_ids, BiomassDensity=biomass_density)
    CSV.write(joinpath(resource_dir, "resource_results_$(run)_$(ts).csv"), res_df)
end

"""
    fishery_results(sim::MarineSimulation)

Export catch and effort metrics for all active fishing fleets.
"""
function fishery_results(sim::MarineSimulation)
    ts = Int(sim.model.iteration)
    run = Int(sim.run)
    fisheries = sim.model.fishing

    # Get the base results directory from the model's files DataFrame
    files_df = sim.model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]
    
    # Construct full path for the fishery output subdirectory
    fish_dir = joinpath(res_dir, "Fishery")
    !isdir(fish_dir) && mkpath(fish_dir)

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

    for fishery in fisheries
        fishery.mean_length_catch = 0.0
        fishery.mean_weight_catch = 0.0
    end
end

# --- Post-Processing Helpers & Kernels ---

"""
    init_biomass_by_size!(model, outputs)

Aggregate individual biomass into spatially explicit size-structured population grids.
"""
function init_biomass_by_size!(model, outputs)
    fill!(outputs.biomass, 0.0f0)
    # The actual aggregation logic is launched separately via kernels in update.jl
end

"""
Calculates instantaneous fishing mortality (F) from the 6D fishing mortality array.
"""
@kernel function fishing_mortality_kernel!(Rate, biomass_by_size, fishing_mortality)
    lon, lat, depth, fishery, prey_sp, prey_bin = @index(Global, NTuple)
    
    @inbounds if fishing_mortality[lon, lat, depth, fishery, prey_sp, prey_bin] > 0
        biomass_val = biomass_by_size[lon, lat, depth, prey_sp, prey_bin]
        mort_val = fishing_mortality[lon, lat, depth, fishery, prey_sp, prey_bin]
        
        @inbounds if biomass_val > 0
            FT = eltype(Rate)
            mort_frac = FT(mort_val) / biomass_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            if mort_frac < FT(1.0)
                Rate[lon, lat, depth, fishery, prey_sp, prey_bin] = -log(FT(1.0) - mort_frac)
            else
                Rate[lon, lat, depth, fishery, prey_sp, prey_bin] = FT(10.0)
            end
        end
    end
end

"""
Calculates instantaneous starvation mortality (S) from the 5D Smort array.
"""
@kernel function starvation_mortality_kernel!(Rate, biomass_by_size, starvation_mortality)
    lon, lat, depth, sp, size_bin = @index(Global, NTuple)
    
    @inbounds if starvation_mortality[lon, lat, depth, sp, size_bin] > 0
        biomass_val = biomass_by_size[lon, lat, depth, sp, size_bin]
        mort_val = starvation_mortality[lon, lat, depth, sp, size_bin]
        
        @inbounds if biomass_val > 0
            FT = eltype(Rate)
            mort_frac = FT(mort_val) / biomass_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            if mort_frac < FT(1.0)
                Rate[lon, lat, depth, sp, size_bin] = -log(FT(1.0) - mort_frac)
            else
                Rate[lon, lat, depth, sp, size_bin] = FT(10.0)
            end
        end
    end
end