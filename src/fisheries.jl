function load_fisheries(df::DataFrame,dt::Int32)
    grouped = groupby(df, :FisheryName)
    fisheries = Fishery[]

    for g in grouped
        name = g.FisheryName[1]
        bag_limit = Int32(ceil(g.BagLimit[1] * g.NVessel[1] * (dt/1440)))

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
            g.Quota[1],
            0.0,  # cumulative_catch
            0,    # cumulative_inds
            (g.StartDay[1], g.EndDay[1]),
            ((g.XMin[1], g.XMax[1]), (g.YMin[1], g.YMax[1]), (g.ZMin[1], g.ZMax[1])),
            (g.SlotMin[1], g.SlotMax[1]),
            bag_limit,
            0,    # effort_days
            0.0,  # mean_length_catch
            0.0,  # mean_weight_catch
            0.0,  # bycatch_tonnage
            0     # bycatch_inds
        ))
    end
    return fisheries
end

@kernel function fishing_kernel!(
    alive, x, y, z, length, abundance, biomass_ind,
    caught_inds, caught_lengths, caught_is_bycatch,
    fishery_params,
    remaining_quota,
    remaining_bag_limit
)
    i = @index(Global) # Each thread handles one agent

    if alive[i] == 1.0f0
        # Check if agent is within the fishery's geographic area
        # Note: This is a simplified check; a real model might use a polygon check
        if fishery_params.area[1] <= x[i] <= fishery_params.area[2] &&
           fishery_params.area[3] <= y[i] <= fishery_params.area[4]

            # Calculate selectivity (capture probability based on size)
            selectivity = 1.0f0 / (1.0f0 + exp(-fishery_params.slope * (length[i] - fishery_params.l50)))

            # Perform a random draw to see if this school is caught
            if rand(Float32) < selectivity
                # Check if the fish is within the legal slot limit
                if fishery_params.slot_limit[1] <= length[i] <= fishery_params.slot_limit[2]
                    
                    # 1. Determine how many individuals from this school could be caught
                    #    This is the minimum of the school's size and the remaining bag limit.
                    #    The result is a float because abundance is a float.
                    potential_catch = min(abundance[i], Float32(remaining_bag_limit[1]))

                    # 2. Round DOWN to the nearest whole number to get the actual number to catch.
                    inds_to_catch = floor(Int32, potential_catch)

                    if inds_to_catch > 0
                        # Check if this catch exceeds the total quota
                        biomass_of_catch = inds_to_catch * biomass_ind[i]
                        
                        # Use atomic operations to safely "claim" the catch
                        quota_before = @atomic remaining_quota[1] -= biomass_of_catch
                        
                        # If we didn't bust the quota, the catch is successful
                        if quota_before >= biomass_of_catch
                            @atomic remaining_bag_limit[1] -= inds_to_catch
                            
                            # Record the successful catch
                            caught_inds[i] = inds_to_catch
                            caught_lengths[i] = length[i]
                            caught_is_bycatch[i] = fishery_params.is_bycatch
                        else
                            # We busted the quota, so undo the subtraction
                            @atomic remaining_quota[1] += biomass_of_catch
                        end
                    end
                end
            end
        end
    end
end
# LAUNCHER: Deconstructs complex objects before calling the kernel
# LAUNCHER: Deconstructs complex objects before calling the kernel
function fishing!(model::MarineModel, sp::Int, day::Int, outputs::MarineOutputs)
    arch = model.arch
    spec_dat = model.individuals.animals[sp].data
    spec_char_cpu = model.individuals.animals[sp].p
    spec_name = spec_char_cpu.SpeciesLong.second[sp]

    # These temporary arrays are correct
    caught_inds = array_type(arch)(zeros(Int32, length(spec_dat.x))) # Use Int32
    caught_lengths = array_type(arch)(zeros(Float32, length(spec_dat.x)))
    caught_is_bycatch = array_type(arch)(zeros(Int8, length(spec_dat.x)))
    
    spec_char_gpu = (; (k => array_type(arch)(v.second) for (k, v) in pairs(spec_char_cpu))...)

    for fishery in model.fishing
        is_active_today = (fishery.season[1] <= day <= fishery.season[2] && fishery.cumulative_catch < fishery.quota)
        
        if is_active_today
            is_target = spec_name in fishery.target_species
            is_bycatch_flag = spec_name in fishery.bycatch_species
            
            if is_target || is_bycatch_flag
                fishery.effort_days += 1
                
                remaining_quota_biomass = (fishery.quota - fishery.cumulative_catch) * 1e6
                remaining_quota_gpu = array_type(arch)([Float32(remaining_quota_biomass)])
                remaining_bag_limit_gpu = array_type(arch)([Int32(fishery.bag_limit)])

                sel = fishery.selectivities[spec_name]
                fishery_params_gpu = (
                    area = (
                        Float32(fishery.area[1][1]), # min_lon
                        Float32(fishery.area[1][2]), # max_lon
                        Float32(fishery.area[2][1]), # min_lat
                        Float32(fishery.area[2][2])  # max_lat
                    ),
                    slot_limit = (Float32(fishery.slot_limit[1]), Float32(fishery.slot_limit[2])),
                    l50 = Float32(sel.L50),
                    slope = Float32(sel.slope),
                    is_bycatch = Int8(is_bycatch_flag)
                )

                kernel! = fishing_kernel!(device(arch), 256, (length(spec_dat.x),))
                
                kernel!(
                    spec_dat.alive, spec_dat.x, spec_dat.y, spec_dat.z, spec_dat.length,
                    spec_dat.abundance, spec_dat.biomass_ind,
                    caught_inds, caught_lengths, caught_is_bycatch,
                    fishery_params_gpu,
                    remaining_quota_gpu,
                    remaining_bag_limit_gpu 
                )
                
                # --- Post-kernel processing ---
                total_inds_caught = Int(round(sum(caught_inds)))
                if total_inds_caught > 0
                    cpu_biomass_ind = Array(spec_dat.biomass_ind)
                    cpu_caught_inds = Array(caught_inds)
                    cpu_caught_lengths = Array(caught_lengths)
                    cpu_caught_is_bycatch = Array(caught_is_bycatch)
                    
                    caught_mask = cpu_caught_inds .> 0
                    
                    total_biomass_caught_g = sum(cpu_caught_inds[caught_mask] .* cpu_biomass_ind[caught_mask])
                    total_biomass_caught_t = total_biomass_caught_g / 1e6

                    fishery.cumulative_inds += total_inds_caught
                    fishery.cumulative_catch += total_biomass_caught_t

                    current_total_len = fishery.mean_length_catch * (fishery.cumulative_inds - total_inds_caught)
                    new_total_len = sum(cpu_caught_lengths[caught_mask] .* cpu_caught_inds[caught_mask])
                    fishery.mean_length_catch = (current_total_len + new_total_len) / fishery.cumulative_inds

                    bycatch_inds_this_step = sum(cpu_caught_inds[caught_mask .& (cpu_caught_is_bycatch .== 1)])
                    bycatch_biomass_this_step_g = sum(cpu_caught_inds[caught_mask .& (cpu_caught_is_bycatch .== 1)] .* cpu_biomass_ind[caught_mask .& (cpu_caught_is_bycatch .== 1)])
                    
                    fishery.bycatch_inds += bycatch_inds_this_step
                    fishery.bycatch_tonnage += bycatch_biomass_this_step_g / 1e6
                end
                
                fill!(caught_inds, 0)
                fill!(caught_lengths, 0.0f0)
                fill!(caught_is_bycatch, 0)
            end
        end
    end
    KernelAbstractions.synchronize(device(arch))
end