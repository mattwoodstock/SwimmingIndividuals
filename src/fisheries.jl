function apply_fishing!(model,fisheries::Vector{Fishery},sp, day,inds)
    spec = model.individuals.animals[sp].p.SpeciesLong[2][sp]
    spec_dat = model.individuals.animals[sp].data

    for fishery in 1:size(fisheries)[1]
        if !(fisheries[fishery].season[1] ≤ day ≤ fisheries[fishery].season[2]) ||
            fisheries[fishery].cumulative_catch ≥ fisheries[fishery].quota
            continue
        end

        in_target_or_bycatch = spec in fisheries[1].target_species || sp in fisheries[1].bycatch_species

        if !in_target_or_bycatch
            continue
        end

        for ind in inds
            # Check spatial area
            in_area = (fisheries[fishery].area[1][1] ≤ spec_dat.x[ind] ≤ fisheries[fishery].area[1][2] && fisheries[fishery].area[2][1] ≤ spec_dat.y[ind] ≤ fisheries[fishery].area[2][2] && fisheries[fishery].area[3][1] ≤ spec_dat.z[ind] ≤ fisheries[fishery].area[3][2])

            if !in_area
                continue
            end

            # Check slot limit
            in_slot = fisheries[fishery].slot_limit[1] ≤ spec_dat.length[ind] ≤ fisheries[fishery].slot_limit[2]

            if !in_slot
                continue
            end

            # Apply gear selectivity
            l50 = fisheries[fishery].selectivities[string(spec)].L50
            slope = fisheries[fishery].selectivities[string(spec)].slope
            selectivity = 1 / (1 + exp(-slope * (spec_dat.length[ind] - l50)))

            if rand() > selectivity
                continue
            end

            # Attempt catch
            catch_weight = spec_dat.biomass[ind]
            if fisheries[fishery].cumulative_catch + catch_weight ≤ fisheries[fishery].quota
                spec_dat.ac[ind] = 0
                fisheries[fishery].cumulative_catch += catch_weight
            end
        end
    end
end