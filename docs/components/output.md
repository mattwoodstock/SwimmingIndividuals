## Model Outputs & Analysis

These two files work together to handle all aspects of saving and processing the results of a simulation. The `output.jl` file is responsible for creating the data structures that hold the results, while the `analysis.jl` file contains the functions that process the raw model state into meaningful scientific outputs.

### 1. Output Data Structures (`output.jl`)

This file defines the `MarineOutputs` struct and the function that initializes it.

#### `generate_outputs(model)`
This function is called once at the beginning of a simulation to create all the large, multi-dimensional arrays needed to store the results. It is architecture-aware, meaning it will create standard `Array`s for a CPU simulation and `CuArray`s for a GPU simulation. This ensures that the output arrays reside on the same device as the agent data, which is crucial for performance. The function creates grids to store spatially explicit data on mortalities, fishing mortality, consumption, and abundance.

```julia
# In output.jl

# ===================================================================
# Data Structures and Functions for Model Output
# ===================================================================

"""
    generate_outputs(model::MarineModel)
This function creates the data structures for storing simulation results.
It is architecture-aware and will create arrays on the GPU if the model
is configured to run on a GPU.
"""
function generate_outputs(model::MarineModel)
    arch = model.arch
    grid = model.depths.grid
    
    # Get grid dimensions
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    n_fish = length(model.fishing)

    # --- Create output arrays on the correct device (CPU or GPU) ---
    mortalities = array_type(arch)(zeros(Int64, lonres, latres, depthres, model.n_species + model.n_resource, model.n_species))
    Fmort = array_type(arch)(zeros(Int64, lonres, latres, depthres, n_fish, model.n_species))
    consumption = array_type(arch)(zeros(Float64, lonres, latres, depthres, model.n_species + model.n_resource, model.n_species + model.n_resource))
    abundance = array_type(arch)(zeros(Float64, lonres, latres, depthres, model.n_species + model.n_resource))
    
    return MarineOutputs(mortalities, Fmort, consumption, abundance)
end
```

### 2. Analysis and Saving (analysis.jl)
This file contains the functions that perform the analysis at each output step, converting raw data (like the number of deaths) into standard scientific metrics (like instantaneous mortality rates) and saving the results to files.

#### GPU-Compliant Analysis Kernels
These are the high-performance GPU kernels that form the core of the analysis system.

#### mortality_rate_kernel!(...): This kernel runs in parallel over the 5D mortality grid. Each thread takes the number of deaths in a cell and the total abundance in that cell and calculates the instantaneous mortality rate (M or F).

#### init_abundances_kernel!(...): This kernel runs in parallel over all agents of a species. Each agent atomically adds its abundance to the correct cell in the 4D abundance grid, allowing for a highly efficient calculation of the spatial distribution of the population.

```julia
# ===================================================================
# GPU-Compliant Analysis Kernels
# ===================================================================

@kernel function mortality_rate_kernel!(Rate, abundance, mortality)
    # Each thread gets a multi-dimensional index for the mortality array
    idx = @index(Global, Cartesian)
    
    @inbounds if mortality[idx] > 0
        abund_val = abundance[idx[1], idx[2], idx[3], idx[5]]
        
        @inbounds if abund_val > 0
            # Use the element type of the output array for all calculations
            FT = eltype(Rate)
            
            mort_val::FT = mortality[idx]
            
            mort_frac = mort_val / abund_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            # Avoid log(0) which results in -Inf
            if mort_frac < FT(1.0)
                Rate[idx] = -log(FT(1.0) - mort_frac)
            else
                Rate[idx] = FT(10.0) # Use a large, finite number for 100% mortality
            end
        end
    end
end

# Kernel to calculate the abundance of agents in each grid cell.
@kernel function init_abundances_kernel!(abundance_out, agents, sp_idx)
    i = @index(Global) # Each thread handles one agent

    @inbounds if agents.alive[i] == 1.0
        # Get the agent's grid cell coordinates
        x = agents.pool_x[i]
        y = agents.pool_y[i]
        z = agents.pool_z[i]
        
        # Atomically add this agent's abundance to its cell in the 4D grid
        @atomic abundance_out[x, y, z, sp_idx] += agents.abundance[i]
    end
end
```

### Launcher Functions for Analysis
These functions are the high-level wrappers that prepare the data and launch the analysis kernels.

```julia
# ===================================================================
# Launcher Functions for Analysis
# ===================================================================

function instantaneous_mortality(outputs::MarineOutputs, arch)
    # Allocate the output array on the correct device
    M = array_type(arch)(zeros(Float32, size(outputs.mortalities)...))
    
    # Launch the kernel with a 5D grid matching the mortality array
    kernel! = mortality_rate_kernel!(device(arch), (8,8,4,4,1), size(M))
    kernel!(M, outputs.abundance, outputs.mortalities)
    KernelAbstractions.synchronize(device(arch))
    
    return M
end

function fishing_mortality(outputs::MarineOutputs, arch)
    # Allocate the output array on the correct device
    F = array_type(arch)(zeros(Float32, size(outputs.Fmort)...))

    # The same kernel can be used for fishing mortality
    kernel! = mortality_rate_kernel!(device(arch), (8,8,4,4,1), size(F))
    kernel!(F, outputs.abundance, outputs.Fmort)
    KernelAbstractions.synchronize(device(arch))

    return F
end

# This launcher calculates abundances for ALL species at once.
function init_abundances!(model::MarineModel, outputs::MarineOutputs)
    arch = model.arch
    
    # Reset the abundance array on the device before calculating
    fill!(outputs.abundance, 0.0)

    # Launch a separate kernel for each species
    for sp in 1:model.n_species
        agents = model.individuals.animals[sp].data
        n_agents = length(agents.x)
        
        if n_agents > 0
            kernel! = init_abundances_kernel!(device(arch), 256, (n_agents,))
            kernel!(outputs.abundance, agents, sp)
        end
    end
    
    # Wait for all kernels to finish before proceeding
    KernelAbstractions.synchronize(device(arch))
    return nothing
end
```

### Top-Level Output Functions
These are the main functions called from the TimeStep! loop to save the results.

#### timestep_results(sim): This function orchestrates the saving of all main outputs. It first gathers the data for all living individual agents from the GPU to the CPU and saves it as a CSV file. It then calls the analysis launcher functions (init_abundances!, instantaneous_mortality, fishing_mortality) to calculate the population-level metrics. Finally, it gathers these processed grids and saves them to an HDF5 file.

#### fishery_results(sim): This is a simpler, CPU-only function that loops through the fishery objects, gathers their summary statistics (e.g., cumulative catch), and saves them to a CSV file.

```julia
# ===================================================================
# Top-Level Output Functions
# ===================================================================

function timestep_results(sim::MarineSimulation)
    model = sim.model
    outputs = sim.outputs
    arch = model.arch
    ts = Int(model.iteration)
    run = Int(sim.run)

    # --- Gather individual data for CSV output ---
    # This logic is inherently CPU-based and requires copying data.
    Sp, Ind, x, y, z, lengths, abundance, biomass = [],[],[],[],[],[],[],[]

    for (species_index, animal) in enumerate(model.individuals.animals)
        spec_dat = animal.data
        
        # Findall must be on a CPU array
        cpu_alive_mask = Array(spec_dat.alive) .== 1.0
        alive_indices = findall(cpu_alive_mask)
        if isempty(alive_indices); continue; end

        # Copy only the data for living individuals
        append!(Sp, fill(species_index, length(alive_indices)))
        append!(Ind, alive_indices)
        append!(x, Array(spec_dat.x[alive_indices]))
        append!(y, Array(spec_dat.y[alive_indices]))
        append!(z, Array(spec_dat.z[alive_indices]))
        append!(lengths, Array(spec_dat.length[alive_indices]))
        append!(abundance, Array(spec_dat.abundance[alive_indices]))
        append!(biomass, Array(spec_dat.biomass_school[alive_indices]))
    end

    if !isempty(Sp)
        # Create and write DataFrame on CPU
        df = DataFrame(Species=Sp, Individual=Ind, X=x, Y=y, Z=z, Length=lengths, Abundance=abundance, Biomass=biomass)
        CSV.write("results/Individual/IndividualResults_$run-$ts.csv", df)
    end
    
    # --- Calculate and save population-scale results ---
    init_abundances!(model, outputs) # Calculate abundances before mortality rates
    M = instantaneous_mortality(outputs, arch)
    F = fishing_mortality(outputs, arch)
    
    # Copy results to CPU for saving to HDF5
    cpu_M = Array(M)
    cpu_F = Array(F)
    cpu_DC = Array(outputs.consumption)

    h5open("results/Population/Instantaneous_Mort_$(run)-$(ts).h5", "w") do file
        write(file, "M", cpu_M)
        write(file, "F", cpu_F)
        write(file, "Diet", cpu_DC)
    end

    # --- Reset output arrays on the device ---
    fill!(outputs.mortalities, 0)
    fill!(outputs.Fmort, 0)
    fill!(outputs.consumption, 0.0)
    
    return nothing
end

# This function is entirely CPU-based and does not need modification.
function fishery_results(sim::MarineSimulation)
    ts = Int(sim.model.iteration)
    run = Int(sim.run)
    fisheries = sim.model.fishing

    name, quotas, catches_t, catches_ind = [], [], [], []
    
    for fishery in fisheries
        push!(name, fishery.name)
        push!(quotas, fishery.quota)
        push!(catches_t, fishery.cumulative_catch)
        push!(catches_ind, fishery.cumulative_inds)
    end

    df = DataFrame(Name=name, Quota=quotas, Tonnage=catches_t, Individuals=catches_ind)
    CSV.write("results/Fishery/FisheryResults_$run-$ts.csv", df)
end
```