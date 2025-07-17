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

@kernel function fishing_kernel!(
    spec_dat, Fmort_inds, # Main data arrays
    spec_char_gpu, fishery_params_gpu, # GPU-compatible parameters
    sp::Int # Pass species index directly
)
    ind = @index(Global) # My index

    @inbounds if spec_dat.alive[ind] == 1.0
        # --- Filter 1: Is the agent in the right place? ---
        # Note: Accessing fields from the simple fishery_params_gpu tuple
        in_area = (fishery_params_gpu.area[1][1] <= spec_dat.x[ind] <= fishery_params_gpu.area[1][2] &&
                   fishery_params_gpu.area[2][1] <= spec_dat.y[ind] <= fishery_params_gpu.area[2][2] &&
                   fishery_params_gpu.area[3][1] <= spec_dat.z[ind] <= fishery_params_gpu.area[3][2])

        if in_area
            # --- Filter 2: Is the agent the right size? ---
            in_slot = fishery_params_gpu.slot_limit[1] <= spec_dat.length[ind] <= fishery_params_gpu.slot_limit[2]

            if in_slot
                # --- Filter 3: Gear Selectivity (probabilistic) ---
                selectivity = 1.0 / (1.0 + exp(-fishery_params_gpu.slope * (spec_dat.length[ind] - fishery_params_gpu.l50)))
                
                if rand(Float32) <= selectivity
                    # --- ALL FILTERS PASSED: Calculate Catch ---
                    abundance = spec_dat.abundance[ind]
                    # Note: Accessing traits from the GPU-compatible spec_char_gpu
                    k = spec_char_gpu.School_Size[sp] / 2.0
                    density_effect = abundance^2 / (abundance^2 + k^2)
                    
                    possible_inds = floor(Int, abundance)
                    catch_inds = floor(Int, rand(Float32) * possible_inds * density_effect)

                    if catch_inds > 0
                        # Atomically update agent state to prevent race conditions
                        @atomic spec_dat.biomass_school[ind] -= catch_inds * spec_dat.biomass_ind[ind]
                        old_abund = @atomic spec_dat.abundance[ind] -= Float64(catch_inds)

                        if old_abund - catch_inds <= 0
                            spec_dat.alive[ind] = 0.0
                        end
                        
                        # Atomically record the number of individuals caught
                        @atomic Fmort_inds[ind] += catch_inds
                    end
                end
            end
        end
    end
end

function fishing!(model::MarineModel, sp::Int, day::Int, outputs::MarineOutputs)
    arch = model.arch
    spec_dat = model.individuals.animals[sp].data
    spec_char_cpu = model.individuals.animals[sp].p
    spec_name = spec_char_cpu.SpeciesLong.second[sp]

    # Temporary array to store individuals caught per agent
    Fmort_inds = array_type(arch)(zeros(Int, length(spec_dat.x)))

    spec_char_gpu = (; (k => array_type(arch)(v.second) for (k, v) in pairs(spec_char_cpu))...)

    for (fish_idx, fishery) in enumerate(model.fishing)
        # --- Main Fishery Checks (done once on CPU) ---
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

        # --- Launch Kernel with only GPU-compatible arguments ---
        kernel! = fishing_kernel!(device(arch), 256, (length(spec_dat.x),))
        kernel!(spec_dat, Fmort_inds, spec_char_gpu, fishery_params_gpu, sp)

        # --- Post-Kernel Processing (on CPU) ---
        total_inds_caught = Int(sum(Fmort_inds)) # Copy scalar result from GPU
        if total_inds_caught > 0
            # (This logic to update cumulative totals is fine on the CPU)
            # You may want a more efficient way to get avg_biomass_ind if this becomes a bottleneck
            cpu_biomass_ind = Array(spec_dat.biomass_ind)
            cpu_fmort_inds = Array(Fmort_inds)
            avg_biomass_ind = mean(cpu_biomass_ind[cpu_fmort_inds .> 0])
            total_biomass_caught = total_inds_caught * avg_biomass_ind
            
            fishery.cumulative_inds += total_inds_caught
            fishery.cumulative_catch += total_biomass_caught / 1e6
            
            # Reset temp array for next fishery
            fill!(Fmort_inds, 0)
        end
    end
    # Synchronize to ensure all fishing kernels are done before proceeding
    KernelAbstractions.synchronize(device(arch))
end
