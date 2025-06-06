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

function fishing(model, fisheries::Vector{Fishery}, sp, day, inds)
    spec = model.individuals.animals[sp].p.SpeciesLong[2][sp]
    spec_dat = model.individuals.animals[sp].data
    spec_char = model.individuals.animals[sp].p

    for fishery in fisheries
        if !(fishery.season[1] ≤ day ≤ fishery.season[2]) || fishery.cumulative_catch ≥ fishery.quota
            continue
        end

        in_target_or_bycatch = spec in fishery.target_species || sp in fishery.bycatch_species
        if !in_target_or_bycatch
            continue
        end

        daily_catch = 0

        for ind in inds
            in_area = (fishery.area[1][1] ≤ spec_dat.x[ind] ≤ fishery.area[1][2] &&
                       fishery.area[2][1] ≤ spec_dat.y[ind] ≤ fishery.area[2][2] &&
                       fishery.area[3][1] ≤ spec_dat.z[ind] ≤ fishery.area[3][2])

            if !in_area || daily_catch >= fishery.bag_limit
                continue
            end

            in_slot = fishery.slot_limit[1] ≤ spec_dat.length[ind] ≤ fishery.slot_limit[2]
            if !in_slot
                continue
            end

            l50 = fishery.selectivities[string(spec)].L50
            slope = fishery.selectivities[string(spec)].slope
            selectivity = 1 / (1 + exp(-slope * (spec_dat.length[ind] - l50)))

            if rand() > selectivity
                continue
            end

            # Density-dependent adjustment
            abundance = spec_dat.abundance[ind]
            k = spec_char.School_Size[2][sp] / 2 #Density at which catchability is 50%. Assumed to be 1/2 school size
            density_effect = abundance^2 / (abundance^2 + k^2)

            available_biomass = spec_dat.biomass_school[ind]
            available_tons = available_biomass / 10e6
            remaining_quota = fishery.quota - fishery.cumulative_catch

            if remaining_quota ≤ 0 || available_tons ≤ 0
                break
            end

            biomass_ind = spec_char.LWR_a[2][sp] * (spec_dat.length[ind]/10)^spec_char.LWR_b[2][sp]
            max_catchable_inds = floor(Int, (remaining_quota * 10e6) / biomass_ind)
            max_inds_by_biomass = floor(Int, available_biomass / biomass_ind)

            possible_inds = min(fishery.bag_limit - daily_catch, max_catchable_inds, max_inds_by_biomass)
            # Apply density compensation to reduce catch
            catch_inds = floor(Int, rand() * possible_inds * density_effect)

            if catch_inds ≤ 0
                continue
            end

            adj_catch = catch_inds * biomass_ind
            adj_catch_tons = adj_catch / 10e6

            spec_dat.biomass_school[ind] -= adj_catch
            spec_dat.abundance[ind] -= catch_inds
            fishery.cumulative_catch += adj_catch_tons
            fishery.cumulative_inds += catch_inds
            daily_catch += catch_inds

            if spec_dat.biomass_school[ind] ≤ 0
                spec_dat.alive[ind] = 0
            end
        end
    end
end

