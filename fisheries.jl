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
    alive, x, y, z, length, abundance, biomass_ind, biomass_school,
    pool_x, pool_y, pool_z,
    Fmort,
    size_bin_thresholds::CuDeviceMatrix{Float32},
    biomass_caught_per_fishery,
    inds_caught_per_fishery, # NEW: Temporary array to sum total individuals
    fishery_params,
    remaining_quota,
    remaining_bag_limit,
    sp_idx::Int32,
    fishery_idx::Int32
)
    i = @index(Global)

    if alive[i] == 1.0f0
        if fishery_params.area[1] <= x[i] <= fishery_params.area[2] &&
           fishery_params.area[3] <= y[i] <= fishery_params.area[4]

            selectivity = 1.0f0 / (1.0f0 + exp(-fishery_params.slope * (length[i] - fishery_params.l50)))

            if rand(Float32) < selectivity
                if fishery_params.slot_limit[1] <= length[i] <= fishery_params.slot_limit[2]
                    
                    potential_catch = min(abundance[i], Float32(remaining_bag_limit[1]))
                    inds_to_catch = floor(Int32, potential_catch)

                    if inds_to_catch > 0
                        biomass_of_catch = inds_to_catch * biomass_ind[i]
                        quota_before = @atomic remaining_quota[1] -= biomass_of_catch
                        
                        if quota_before >= biomass_of_catch
                            @atomic remaining_bag_limit[1] -= inds_to_catch
                            
                            px, py, pz = pool_x[i], pool_y[i], pool_z[i]
                            size_bin = find_species_size_bin(length[i], sp_idx, size_bin_thresholds)
                            
                            # Log biomass to the Fmort grid
                            @atomic Fmort[px, py, pz, fishery_idx, sp_idx, size_bin] += biomass_of_catch
                            
                            @atomic biomass_caught_per_fishery[fishery_idx] += biomass_of_catch
                            @atomic inds_caught_per_fishery[fishery_idx] += inds_to_catch
                            
                            # Directly update the agent's state
                            @atomic abundance[i] -= Float32(inds_to_catch)
                            @atomic biomass_school[i] -= biomass_of_catch
                            
                            if abundance[i] <= 0.0f0
                                alive[i] = 0.0f0
                            end
                        else
                            @atomic remaining_quota[1] += biomass_of_catch
                        end
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

    size_bin_thresholds = model.size_bin_thresholds

    biomass_caught_per_fishery = array_type(arch)(zeros(Float32, length(model.fishing)))
    inds_caught_per_fishery = array_type(arch)(zeros(Int32, length(model.fishing)))

    for (fishery_id, fishery) in enumerate(model.fishing)
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
                    spec_dat.abundance, spec_dat.biomass_ind, spec_dat.biomass_school, 
                    spec_dat.pool_x, spec_dat.pool_y, spec_dat.pool_z,
                    outputs.Fmort, 
                    size_bin_thresholds,
                    biomass_caught_per_fishery,
                    inds_caught_per_fishery,
                    fishery_params_gpu,
                    remaining_quota_gpu,
                    remaining_bag_limit_gpu,
                    Int32(sp),
                    Int32(fishery_id)
                )
            end
        end
    end
    KernelAbstractions.synchronize(device(arch))

    # --- Post-kernel processing for summary statistics ---
    total_biomass_caught_cpu = Array(biomass_caught_per_fishery)
    total_inds_caught_cpu = Array(inds_caught_per_fishery)
    
    for (fishery_id, fishery) in enumerate(model.fishing)
        biomass_caught_g = total_biomass_caught_cpu[fishery_id]
        inds_caught = total_inds_caught_cpu[fishery_id]
        
        if biomass_caught_g > 0
            fishery.cumulative_catch += biomass_caught_g / 1e6
        end
        if inds_caught > 0
            fishery.cumulative_inds += inds_caught
        end
    end
end