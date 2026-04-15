# ===================================================================
# High-Level Model Setup and Execution
# ===================================================================

"""
    setup_and_run_model(config_filename="files.csv")

The primary entry point for setting up the environment, initializing 
agents/resources, and driving the simulation loop.
"""
function setup_and_run_model(config_filename="files.csv")
    # Identify the path to the config file located in the root directory
    # (One level up from this file's location in src/)
    base_path = joinpath(@__DIR__, "..")
    config_path = joinpath(base_path, config_filename)

    if !isfile(config_path)
        error("Configuration file not found at: $config_path")
    end

    ## 1. Load configuration databases
    files = CSV.read(config_path, DataFrame)
    
    # Resolve scenario and results directories
    scen_dir_row = filter(row -> row.File == "scen_dir", files)
    scen_dir = scen_dir_row[1, :Destination]
    
    # Update destinations to be absolute or scenario-relative
    files.Destination = [
        row.File == "scen_dir" ? row.Destination : joinpath(scen_dir, row.Destination) 
        for row in eachrow(files)
    ]
    
    res_dir_row = filter(row -> row.File == "res_dir", files)
    res_dir_name = res_dir_row[1, :Destination]
    full_res_path = joinpath(scen_dir, res_dir_name)
    mkpath(full_res_path)

    # Load traits, parameters, and grid settings
    trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait", :Destination][1], DataFrame))))
    resource_trait = CSV.read(files[files.File .== "resource_trait", :Destination][1], DataFrame)
    params = CSV.read(files[files.File .== "params", :Destination][1], DataFrame)
    grid = CSV.read(files[files.File .== "grid", :Destination][1], DataFrame)
    fisheries_df = CSV.read(files[files.File .== "fisheries", :Destination][1], DataFrame)
    envi_file = files[files.File .== "environment", :Destination][1]

    ## 2. Global simulation settings
    Nsp = parse(Int32, params[params.Name .== "numspec", :Value][1])
    Nresource = parse(Int32, params[params.Name .== "numresource", :Value][1])
    output_dt = parse(Int32, params[params.Name .== "output_dt", :Value][1])
    spinup = parse(Int32, params[params.Name .== "spinup", :Value][1])
    plt_diags = parse(Int32, params[params.Name .== "plt_diags", :Value][1])
    foraging_attempts = parse(Int32, params[params.Name .== "num_foraging_attempts", :Value][1])
    n_iteration = parse(Int32, params[params.Name .== "nts", :Value][1])
    dt = parse(Int32, params[params.Name .== "model_dt", :Value][1])
    n_iters = parse(Int16, params[params.Name .== "n_iter", :Value][1])
    
    maxN = Int64(500000) 
    arch_str = params[params.Name .== "architecture", :Value][1]

    # Handle Hardware Architecture
    if arch_str == "GPU"
        if CUDA.functional()
            arch = GPU()
            @info "✅ Architecture successfully set to GPU."
        else
            @warn "GPU specified but CUDA is not functional. Falling back to CPU."
            arch = CPU()
        end
    else
        arch = CPU()
        @info "✅ Architecture successfully set to CPU."
    end

    t = 0.0 # Initial simulation time
    start_date = Date(2023, 1, 1) # Placeholder start date

    ## 3. Environment and Infrastructure Initialization
    envi = generate_environment!(arch, envi_file, plt_diags, files)
    depths = generate_depths(files)
    capacities = initial_habitat_capacity(envi, Nsp, Nresource, files, arch, plt_diags)

    ## 4. Multi-Run Simulation Loop
    for iter in 1:n_iters
        @info "--- Starting Simulation Run $iter ---"
        
        B = Float32.(trait[:Biomass][1:Nsp])

        # Initialize Agents and Resources
        inds, daily_birth_counters = generate_individuals(trait, arch, Nsp, B, maxN, depths, capacities, dt, envi, start_date)
        resources = initialize_resources(resource_trait, Nsp, Nresource, depths, capacities, arch)
        fishery_fleet = load_fisheries(fisheries_df, dt)

        # Pre-simulation summary
        init_abund = fill(0, Nsp)
        bioms = fill(0.0, Nsp)
        for sp in 1:Nsp
            init_abund[sp] = sum(inds.animals[sp].data.abundance)
            bioms[sp] = sum(inds.animals[sp].data.biomass_school)
        end

        # Create the high-level model object
        model = MarineModel(
            arch, envi, depths, fishery_fleet, t, 0, dt, 
            inds, resources, resource_trait, capacities, 
            maxN, Nsp, Nresource, init_abund, bioms, init_abund, 
            files, output_dt, spinup, foraging_attempts, plt_diags,
            daily_birth_counters
        )

        # Setup outputs
        # Note: Added default bin size of 10 for size-based outputs
        outputs = generate_outputs(model, Int32(10))

        # Setup and Run Simulation
        sim = MarineSimulation(model, dt, n_iteration, iter, outputs)
        runSI(sim)
    end
    
    @info "Simulation sequence complete."
    return nothing
end