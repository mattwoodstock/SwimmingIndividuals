## Fisheries

The `fisheries.jl` file contains all the functions related to the impact of human fishing activity on the simulated populations. This module is responsible for loading the complex regulations that define each fishery and for applying fishing mortality to the agents in a high-performance, parallel manner.

### 1. Loading Fishery Regulations

At the beginning of the simulation, the model needs to load and parse the rules for all the fisheries that will be active in the model world.

#### `load_fisheries(df)`
This function reads the `fisheries.csv` DataFrame, groups the data by the unique `FisheryName`, and constructs a `Fishery` object for each one. This object is a container for all the rules that govern that specific fishery, including its target and bycatch species, its annual quota, the start and end days of its season, the geographic area of its operation, and the gear selectivity parameters that determine which sizes of fish are most likely to be caught.

```julia
function load_fisheries(df::DataFrame)
    grouped = groupby(df, :FisheryName)
    fisheries = Fishery[]

    for g in grouped
        name = g.FisheryName[1]
        quota = g.Quota[1]
        season = (g.StartDay[1], g.EndDay[1])
        area = ((g.XMin[1], g.XMax[1]), (g.YMin[1], g.YMax[1]), (g.ZMin[1], g.ZMax[1]))
        slot_limit = (g.SlotMin[1], g.SlotMax[1])
        daily_bag_limit = g.BagLimit[1] * g.NVessel[1]

        targets = g.Species[g.Role .== "target"]
        bycatch = g.Species[g.Role .== "bycatch"]

        selectivities = Dict{String, Selectivity}()
        for row in eachrow(g)
            selectivities[row.Species] = Selectivity(row.Species, row.L50, row.Slope)
        end

        push!(fisheries, Fishery(
            name,
            collect(targets),
            collect(bycatch),
            selectivities,
            quota,
            0.0,  # initialize cumulative catch
            0,
            season,
            area,
            slot_limit,
            daily_bag_limit
        ))
    end
    return fisheries
end
```

### 2. Applying Fishing Pressure
At each timestep, the model must determine which agents are caught by the active fisheries. This is a computationally intensive task that is handled by a high-performance GPU kernel.

#### fishing!(model, sp, day, outputs)
This is the main launcher function for the fishing submodel. It loops through each active fishery. For each one, it first performs a series of checks on the CPU to see if the fishery is open (i.e., within its season and under its quota). If it is, the function deconstructs the complex fishery object into a simple, GPU-compatible NamedTuple of parameters. It then calls the fishing_kernel! to run in parallel for all agents of the target species.

#### fishing_kernel!(...)
This GPU kernel is the core of the fishing model. Each thread handles a single agent. The kernel efficiently checks the agent against all the fishery's regulations: its location (in_area), its size (in_slot), and a probabilistic gear selectivity curve. If an agent passes all these filters, the kernel calculates the number of individuals caught from its school (incorporating a density-dependent effect) and uses robust atomic operations to safely subtract the removed biomass and abundance from the agent's state.

```julia
@kernel function fishing_kernel!(
    # Deconstructed agent data arrays
    alive, x, y, z, length_arr, abundance, biomass_ind, biomass_school,
    # Other arrays and parameters
    Fmort_inds, spec_char_gpu, fishery_params_gpu, sp::Int
)
    ind = @index(Global)

    @inbounds if alive[ind] == 1.0
        # --- Filter 1: Is the agent in the right place? ---
        in_area = (fishery_params_gpu.area[1][1] <= x[ind] <= fishery_params_gpu.area[1][2] &&
                   fishery_params_gpu.area[2][1] <= y[ind] <= fishery_params_gpu.area[2][2] &&
                   fishery_params_gpu.area[3][1] <= z[ind] <= fishery_params_gpu.area[3][2])

        if in_area
            # --- Filter 2: Is the agent the right size? ---
            in_slot = fishery_params_gpu.slot_limit[1] <= length_arr[ind] <= fishery_params_gpu.slot_limit[2]

            if in_slot
                # --- Filter 3: Gear Selectivity (probabilistic) ---
                selectivity = 1.0 / (1.0 + exp(-fishery_params_gpu.slope * (length_arr[ind] - fishery_params_gpu.l50)))
                
                if rand(Float32) <= selectivity
                    # --- ALL FILTERS PASSED: Calculate Catch ---
                    abund = abundance[ind]
                    k = spec_char_gpu.School_Size[sp] / 2.0
                    density_effect = abund^2 / (abund^2 + k^2)
                    
                    possible_inds = floor(Int, abund)
                    catch_inds = floor(Int, rand(Float32) * possible_inds * density_effect)

                    if catch_inds > 0
                        biomass_removed = catch_inds * biomass_ind[ind]
                        
                        # --- FIX: Use a robust atomic subtraction and correction pattern ---
                        # This avoids the complex CAS loop and is guaranteed to be GPU-compliant.
                        
                        # Atomically subtract the desired amount. This returns the ORIGINAL value.
                        old_biomass = @atomic biomass_school[ind] -= biomass_removed
                        
                        # Now, check if we overdrew the account (subtracted more than was available)
                        if old_biomass < biomass_removed
                            # We took too much. Atomically add back the difference to set biomass to zero.
                            biomass_to_return = biomass_removed - old_biomass
                            @atomic biomass_school[ind] += biomass_to_return
                        end
                        
                        # The actual amount of biomass successfully removed is the minimum of what we wanted and what was there.
                        actual_biomass_removed = min(biomass_removed, old_biomass)

                        if actual_biomass_removed > 0
                            actual_inds_caught = floor(Int, actual_biomass_removed / biomass_ind[ind])
                            
                            if actual_inds_caught > 0
                                old_abund = @atomic abundance[ind] -= Float64(actual_inds_caught)
                                if old_abund - actual_inds_caught <= 0
                                    alive[ind] = 0.0
                                end
                                @atomic Fmort_inds[ind] += actual_inds_caught
                            end
                        end
                    end
                end
            end
        end
    end
end


# LAUNCHER: Deconstructs complex objects before calling the kernel
function fishing!(model::MarineModel, sp::Int, day::Int, outputs::MarineOutputs)
    arch = model.arch
    spec_dat = model.individuals.animals[sp].data
    spec_char_cpu = model.individuals.animals[sp].p
    spec_name = spec_char_cpu.SpeciesLong.second[sp]

    Fmort_inds = array_type(arch)(zeros(Int, length(spec_dat.x)))
    spec_char_gpu = (; (k => array_type(arch)(v.second) for (k, v) in pairs(spec_char_cpu))...)

    for (fish_idx, fishery) in enumerate(model.fishing)
        if !(fishery.season[1] <= day <= fishery.season[2]) || fishery.cumulative_catch >= fishery.quota
            continue
        end
        if !(spec_name in fishery.target_species || spec_name in fishery.bycatch_species)
            continue
        end

        sel = fishery.selectivities[spec_name]
        fishery_params_gpu = (
            area = fishery.area,
            slot_limit = fishery.slot_limit,
            l50 = sel.L50,
            slope = sel.slope
        )

        kernel! = fishing_kernel!(device(arch), 256, (length(spec_dat.x),))
        
        # Launch the kernel with deconstructed arrays
        kernel!(
            spec_dat.alive, spec_dat.x, spec_dat.y, spec_dat.z, spec_dat.length,
            spec_dat.abundance, spec_dat.biomass_ind, spec_dat.biomass_school,
            Fmort_inds, spec_char_gpu, fishery_params_gpu, sp
        )

        total_inds_caught = Int(sum(Fmort_inds))
        if total_inds_caught > 0
            cpu_biomass_ind = Array(spec_dat.biomass_ind)
            cpu_fmort_inds = Array(Fmort_inds)
            avg_biomass_ind = mean(cpu_biomass_ind[cpu_fmort_inds .> 0])
            total_biomass_caught = total_inds_caught * avg_biomass_ind
            
            fishery.cumulative_inds += total_inds_caught
            fishery.cumulative_catch += total_biomass_caught / 1e6
            
            fill!(Fmort_inds, 0)
        end
    end
    KernelAbstractions.synchronize(device(arch))
end
```