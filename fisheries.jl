function load_fisheries(df::DataFrame, dt::Int32)
    grouped = groupby(df, :FisheryName)
    fisheries = Fishery[]

    for g in grouped
        name = g.FisheryName[1]

        # Fleet-wide bag limit (sum across all vessels)
        bag_limit = Int32(g.BagLimit[1] * g.NVessel[1])

        targets = g.Species[g.Role .== "target"]
        bycatch = g.Species[g.Role .== "bycatch"]

        selectivities = Dict{String, Selectivity}()
        for row in eachrow(g)
            sel_type_str = row.SelectivityType
            sel_type_code = sel_type_str == "logistic" ? 1 :
                            sel_type_str == "knife_edge" ? 2 :
                            sel_type_str == "dome_shaped" ? 3 : 1

            p1 = Float32(get(row, :L50, 0.0))
            p2 = Float32(get(row, :Slope, 0.0))
            p3 = Float32(get(row, :L50_2, 0.0))
            p4 = Float32(get(row, :Slope_2, 0.0))
            
            selectivities[row.Species] = Selectivity(row.Species, sel_type_code, p1, p2, p3, p4)
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
            bag_limit,  # fleet-wide
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
    pool_x, pool_y, pool_z,energy,
    Fmort,
    size_bin_thresholds::CuDeviceMatrix{Float32},
    biomass_caught_per_fishery,
    inds_caught_per_fishery,
    length_caught_per_fishery,
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

            local selectivity::Float32
            sel_type = fishery_params.sel_type
            len = length[i]

            if sel_type == 1 # Logistic
                l50 = fishery_params.p1
                slope = fishery_params.p2
                selectivity = 1.0f0 / (1.0f0 + exp(-slope * (len - l50)))
            
            elseif sel_type == 2 # Knife-edge
                l_knife = fishery_params.p1
                selectivity = ifelse(len >= l_knife, 1.0f0, 0.0f0)

            elseif sel_type == 3 # Dome-shaped (double logistic)
                l50_1 = fishery_params.p1
                slope1 = fishery_params.p2
                l50_2 = fishery_params.p3
                slope2 = fishery_params.p4
                
                ascending = 1.0f0 / (1.0f0 + exp(-slope1 * (len - l50_1)))
                descending = 1.0f0 - (1.0f0 / (1.0f0 + exp(-slope2 * (len - l50_2))))
                selectivity = ascending * descending
            else
                selectivity = 0.0f0 # Default to zero if type is unknown
            end

            if rand(Float32) < selectivity
                if fishery_params.slot_limit[1] <= length[i] <= fishery_params.slot_limit[2]
                    
                    inds_to_catch = min(abundance[i], remaining_bag_limit[1])

                    if inds_to_catch > 0
                        # Atomically try to reserve fish from the fleet-wide bag limit
                        bag_before = @atomic remaining_bag_limit[1] -= inds_to_catch

                        if bag_before >= inds_to_catch
                            biomass_of_catch = inds_to_catch * biomass_ind[i]

                            # Check quota
                            quota_before = @atomic remaining_quota[1] -= biomass_of_catch

                            if quota_before >= biomass_of_catch
                                # Log the catch
                                px, py, pz = pool_x[i], pool_y[i], pool_z[i]
                                size_bin = find_species_size_bin(length[i], sp_idx, size_bin_thresholds)

                                @atomic Fmort[px, py, pz, fishery_idx, sp_idx, size_bin] += biomass_of_catch
                                @atomic biomass_caught_per_fishery[fishery_idx] += biomass_of_catch
                                @atomic inds_caught_per_fishery[fishery_idx] += inds_to_catch
                                @atomic length_caught_per_fishery[fishery_idx] += length[i] * inds_to_catch

                                # Remove from school
                                proportion_removed = biomass_of_catch / biomass_school[i]
                                @atomic energy[i] -= energy[i] * proportion_removed
                                @atomic abundance[i] -= Float32(inds_to_catch)
                                @atomic biomass_school[i] -= biomass_of_catch

                                if abundance[i] <= 0.0f0
                                    alive[i] = 0.0f0
                                end
                            else
                                @atomic remaining_quota[1] += biomass_of_catch
                                @atomic remaining_bag_limit[1] += inds_to_catch
                            end
                        else
                            # Not enough left in fleet-wide bag limit
                            @atomic remaining_bag_limit[1] += inds_to_catch
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

    # Create temporary arrays for summarizing catch
    biomass_caught_per_fishery = array_type(arch)(zeros(Float32, length(model.fishing)))
    inds_caught_per_fishery = array_type(arch)(zeros(Int32, length(model.fishing)))
    length_caught_per_fishery = array_type(arch)(zeros(Float32, length(model.fishing)))

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
                        Float32(fishery.area[1][1]),
                        Float32(fishery.area[1][2]),
                        Float32(fishery.area[2][1]),
                        Float32(fishery.area[2][2])
                    ),
                    slot_limit = map(Float32, fishery.slot_limit),
                    sel_type = sel.sel_type,
                    p1 = sel.p1,
                    p2 = sel.p2,
                    p3 = sel.p3,
                    p4 = sel.p4,
                    is_bycatch = Int8(is_bycatch_flag)
                )

                kernel! = fishing_kernel!(device(arch), 256, (length(spec_dat.x),))
                
                kernel!(
                    spec_dat.alive, spec_dat.x, spec_dat.y, spec_dat.z, spec_dat.length,
                    spec_dat.abundance, spec_dat.biomass_ind, spec_dat.biomass_school, 
                    spec_dat.pool_x, spec_dat.pool_y, spec_dat.pool_z, spec_dat.energy,
                    outputs.Fmort, 
                    size_bin_thresholds,
                    biomass_caught_per_fishery,
                    inds_caught_per_fishery,
                    length_caught_per_fishery,
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
    total_length_caught_cpu = Array(length_caught_per_fishery)
    
    for (fishery_id, fishery) in enumerate(model.fishing)
        biomass_caught_g = total_biomass_caught_cpu[fishery_id]
        inds_caught = total_inds_caught_cpu[fishery_id]
        total_len_caught = total_length_caught_cpu[fishery_id]
        
        if inds_caught > 0
            # 1. Calculate the mean length of the catch from this timestep's data.
            mean_len_this_step = total_len_caught / inds_caught
            
            # 2. Calculate the mean weight of the catch from this timestep's data.
            mean_weight_this_step = biomass_caught_g / inds_caught

            # 3. Assign these non-cumulative values directly to the fishery object.
            fishery.mean_length_catch = mean_len_this_step
            fishery.mean_weight_catch = mean_weight_this_step

            # Update cumulative totals (this part is correct)
            fishery.cumulative_catch += biomass_caught_g / 1e6
            fishery.cumulative_inds += inds_caught
        else
            # If no fish were caught this timestep, set the means to 0
            fishery.mean_length_catch = 0.0
            fishery.mean_weight_catch = 0.0
        end
    end
end