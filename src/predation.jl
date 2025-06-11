function calculate_distances_prey(model::MarineModel, sp::Int64, inds, dt)
    sp_data = model.individuals.animals[sp].data
    sp_chars = model.individuals.animals[sp].p

    grid = model.depths.grid
    cell_size = grid[grid.Name .== "cellsize", :Value][1]

    min_prey = sp_chars.Min_Prey[2][sp]
    max_prey = sp_chars.Max_Prey[2][sp]
    handling_time = sp_chars.Handling_Time[2][sp]

    detection = @view sp_data.vis_prey[inds]
    sp_length_inds = @view sp_data.length[inds]
    x = @view sp_data.x[inds]
    y = @view sp_data.y[inds]
    z = @view sp_data.z[inds]
    sp_pool_x = @view sp_data.pool_x[inds]
    sp_pool_y = @view sp_data.pool_y[inds]

    swim_speed = bl_per_s(sp_length_inds/10,sp_chars.Swim_velo[2][sp])

    adj_length = sp_length_inds ./ 1000
    max_swim_dist = swim_speed .* adj_length .* @view(dt[inds])

    max_num = Int(dt[1] / handling_time)

    cell_size_m_lat = cell_size * 111320
    cell_size_m_lon = cell_size .* 111320 .* cosd.(y)
    cell_size_min = min.(cell_size_m_lat, cell_size_m_lon)
    n_cell = Int.(ceil.(cell_size_min ./ max_swim_dist)) .+ 1

    min_prey_sizes = sp_length_inds .* min_prey
    max_prey_sizes = sp_length_inds .* max_prey

    n = length(inds)
    thread_results = Vector{Vector{PreyInfo}}(undef, n)

    n_resources = length(getfield.(model.resources, :x))

    @Threads.threads for j in 1:n
        @inbounds begin
            ind = inds[j]
            max_dist = detection[j]
            j_min_prey_size = min_prey_sizes[j]
            j_max_prey_size = max_prey_sizes[j]
            j_ncell = n_cell[j]

            prey_x = Float64[]
            prey_y = Float64[]
            prey_z = Float64[]
            prey_id = Int[]
            prey_spec = Int[]
            prey_type = Int[]
            prey_biomass = Float64[]
            prey_length = Float64[]

            for (species_index, animal) in enumerate(model.individuals.animals)
                species_data = animal.data
                prey_lengths = species_data.length

                size_mask = (prey_lengths .>= j_min_prey_size) .& (prey_lengths .<= j_max_prey_size)
                in_x = (species_data.pool_x .>= (sp_pool_x[j] - j_ncell)) .& (species_data.pool_x .<= (sp_pool_x[j] + j_ncell))
                in_y = (species_data.pool_y .>= (sp_pool_y[j] - j_ncell)) .& (species_data.pool_y .<= (sp_pool_y[j] + j_ncell))
                mask = size_mask .& in_x .& in_y

                index = findall(mask)
                append!(prey_x, @view species_data.x[index])
                append!(prey_y, @view species_data.y[index])
                append!(prey_z, @view species_data.z[index])
                append!(prey_id, index)
                append!(prey_spec, fill(species_index, length(index)))
                append!(prey_type, fill(1,length(index)))
                append!(prey_biomass, @view species_data.biomass_school[index])
                append!(prey_length, @view species_data.length[index])
            end

            pool_x = getfield.(model.resources, :pool_x)
            pool_y = getfield.(model.resources, :pool_y)

            #Gather resource sizes.
            in_x = (pool_x .>= (sp_pool_x[j] - j_ncell)) .& (pool_x .<= (sp_pool_x[j] + j_ncell))
            in_y = (pool_y .>= (sp_pool_y[j] - j_ncell)) .& (pool_y .<= (sp_pool_y[j] + j_ncell))
            mask = in_x .& in_y
            masked_indices = findall(mask)

            if length(masked_indices) > 0
                mean_lengths = Float64[]
                spec = 1
                max_size = model.resource_trait[spec,:Max_Size]
            
                μ,σ = lognormal_params_from_maxsize(max_size)
                dist = LogNormal(μ, σ)
                prop = cdf(dist, j_max_prey_size) - cdf(dist, j_min_prey_size)
                numerator, _ = quadgk(x -> x * pdf(dist, x), j_min_prey_size, j_max_prey_size)

                for k in masked_indices
                    this_spec = getfield(model.resources[k], :sp)
                    if this_spec == spec #Same as last species, so all data is the same
                    else #Next species
                        spec += 1
                        max_size = model.resource_trait[spec,:Max_Size]
            
                        μ,σ = lognormal_params_from_maxsize(max_size)
                        dist = LogNormal(μ, σ)
                        prop = cdf(dist, j_max_prey_size) - cdf(dist, j_min_prey_size)
                        numerator, _ = quadgk(x -> x * pdf(dist, x), j_min_prey_size, j_max_prey_size)
                    end
                    push!(mean_lengths, numerator/prop)
                end
        
                append!(prey_x, getfield.(model.resources[masked_indices], :x))
                append!(prey_y, getfield.(model.resources[masked_indices], :y))
                append!(prey_z, getfield.(model.resources[masked_indices], :z))
                append!(prey_id, getfield.(model.resources[masked_indices], :ind))
                append!(prey_spec, getfield.(model.resources[masked_indices], :sp))
                append!(prey_type, fill(2,length(masked_indices)))
                append!(prey_biomass, getfield.(model.resources[masked_indices], :biomass)*prop)
                append!(prey_length,mean_lengths)
            end

            prey_infos = PreyInfo[]
            if !isempty(prey_id)
                prey_coords = hcat(prey_x, prey_y)'  # shape: 2 x N
                prey_tree = KDTree(prey_coords)
                prey_inds, prey_specs,prey_type,prey_biomasses, distances, prey_lengths = knn_haversine(prey_tree, [x[j], y[j],z[j]],prey_z, max_num, prey_spec, prey_id,prey_type,prey_biomass, prey_length, max_dist)

                for i in 1:length(prey_inds)
                    push!(prey_infos, PreyInfo(ind, prey_specs[i], prey_inds[i],prey_type[i], prey_lengths[i], prey_biomasses[i], distances[i]))
                end
            end
            thread_results[j] = prey_infos
        end
    end
    return vcat(thread_results...)
end

function move_predator(model,sp,sp_data,sp_char, ind, prey_df,time_pred)
    swim_velo = sp_char.Swim_velo[2][sp] * (sp_data.length[ind] / 1000)
    predator_x = sp_data.x[ind]
    predator_y = sp_data.y[ind]
    predator_z = sp_data.z[ind]
    if prey_df.Type == 1
        x = model.individuals.animals[prey_df.Sp].data.x[prey_df.Ind]
        y = model.individuals.animals[prey_df.Sp].data.y[prey_df.Ind]
        z = model.individuals.animals[prey_df.Sp].data.z[prey_df.Ind]
    else
        x = getfield(model.resources[prey_df.Ind], :x)
        y = getfield(model.resources[prey_df.Ind], :y)
        z = getfield(model.resources[prey_df.Ind], :z)
    end

    horizontal_distance = haversine(predator_y, predator_x, y, x)  # haversine(lat1, lon1, lat2, lon2)
    vertical_distance = abs(z - predator_z)
    total_distance = sqrt(horizontal_distance^2 + vertical_distance^2)

    time_to_prey = total_distance  / swim_velo

    if time_to_prey > time_pred
        frac = time_pred/time_to_prey
        sp_data.x[ind] = predator_x + (x - predator_x) * frac
        sp_data.y[ind] = predator_y + (y - predator_y) * frac
        sp_data.z[ind] = predator_z + (z - predator_z) * frac
        return time_pred
    else
        sp_data.x[ind] = x
        sp_data.y[ind] = y
        sp_data.z[ind] = z
        return time_to_prey
    end
end

function move_resource(model,sp, ind, mean_size,prey_df,time_pred)
    swim_velo = model.resource_trait[sp,:Swim_velo] * (mean_size / 1000)

    predator_x = getfield(model.resources[prey_df.Ind], :x)
    predator_y = getfield(model.resources[prey_df.Ind], :y)
    predator_z = getfield(model.resources[prey_df.Ind], :z)
    if prey_df.Type == 1
        x = model.individuals.animals[prey_df.Sp].data.x[prey_df.Ind]
        y = model.individuals.animals[prey_df.Sp].data.y[prey_df.Ind]
        z = model.individuals.animals[prey_df.Sp].data.z[prey_df.Ind]
    else
        x = getfield(model.resources[prey_df.Ind], :x)
        y = getfield(model.resources[prey_df.Ind], :y)
        z = getfield(model.resources[prey_df.Ind], :z)
    end

    horizontal_distance = haversine(predator_y, predator_x, y, x)  # haversine(lat1, lon1, lat2, lon2)
    vertical_distance = abs(z - predator_z)
    total_distance = sqrt(horizontal_distance^2 + vertical_distance^2)

    time_to_prey = total_distance  / swim_velo
    
    if time_to_prey > time_pred[ind]
        frac = time_pred[ind]/time_to_prey
        model.resources[ind].x = predator_x + (x - predator_x) * frac
        model.resources[ind].y = predator_y + (y - predator_y) * frac
        model.resources[ind].z = predator_z + (z - predator_z) * frac
        return time_pred[ind]
    else
        model.resources[ind].x = x
        model.resources[ind].y = y
        model.resources[ind].z = z
        return time_to_prey
    end
end

function eat(model::MarineModel, sp, to_eat, prey_list, time, outputs)
    animal = model.individuals.animals[sp]
    data = animal.data
    chars = animal.p

    @views length_ind = data.length[to_eat]
    @views gut_fullness_ind = data.gut_fullness[to_eat]
    @views biomass_school_ind = data.biomass_school[to_eat]

    n_ind = length(to_eat)
    max_stomachs = 0.2
    max_dists = chars.Swim_velo[2][sp] .* (length_ind ./ 1000) .* time[to_eat]

    for j in 1:n_ind
        sorted_prey_buffer = PreyInfo[]
        @inbounds ind = to_eat[j]
        if data.alive[ind] == 0
            continue
        end

        empty!(sorted_prey_buffer)
        for p in prey_list
            @inbounds if p.Predator == ind
                push!(sorted_prey_buffer, p)
            end
        end
        if isempty(sorted_prey_buffer)
            continue
        end

        sort!(sorted_prey_buffer, by = x -> x.Distance)

        max_dist = max_dists[j]
        total_time = 0.0
        prey_index = 1

        while total_time < time[ind] && gut_fullness_ind[j] < max_stomachs && prey_index <= length(sorted_prey_buffer)
            
            prey_info = sorted_prey_buffer[prey_index]

            if prey_info.Distance > max_dist
                break
            end

            move_time = move_predator(model, sp, data, chars, ind, prey_info, time[ind])
            move_time = 0
            total_time += move_time
            if total_time >= time[ind]
                break
            end

            data.active[ind] += move_time / 60
            time_left = time[ind] - total_time
            max_dist = chars.Swim_velo[2][sp] * (length_ind[j] / 1000) * time_left

            ration = 0.0
            n_consumed = 0

            if prey_info.Type == 1
                prey_data = model.individuals.animals[prey_info.Sp].data
                prey_chars = model.individuals.animals[prey_info.Sp].p

                if prey_data.alive[prey_info.Ind] == 0.0
                    prey_index += 1
                    continue
                end

                # Fetch fresh biomass and abundance from model, not from prey_info
                prey_biomass = prey_data.biomass_school[prey_info.Ind]
                ind_biomass = prey_data.biomass_ind[prey_info.Ind]
                abundance = prey_data.abundance[prey_info.Ind]
                max_cons = min(abundance, floor(Int, time_left / chars.Handling_Time[2][sp]))

                x = floor(Int,prey_data.pool_x[prey_info.Ind])
                y = floor(Int,prey_data.pool_y[prey_info.Ind])
                z = floor(Int,prey_data.pool_z[prey_info.Ind])

                ration_max = (max_stomachs * biomass_school_ind[j]) - (gut_fullness_ind[j] * biomass_school_ind[j])
                ration = min(prey_biomass, ind_biomass * max_cons, ration_max)
                ration = max(0.0, min(ration, prey_biomass))  # Final safety clamp
                n_consumed = floor(Int, ration / ind_biomass)
                data.ration[ind] += ration * prey_chars.Energy_density[2][sp]
                predation_mortality(model, prey_info, outputs, n_consumed, ration)

                outputs.mortalities[x,y,z,sp,prey_info.Sp] += n_consumed
                outputs.consumption[x,y,z,sp,prey_info.Sp] += ration

                # Update local prey_info biomass to keep current predator's loop consistent
                prey_info.Biomass = max(0.0, prey_biomass - ration)
            else
                # For non-animal prey (resources)
                # Always fetch fresh biomass from model.resources
                prey_spec = prey_info.Sp
                prey_biomass = model.resources[prey_info.Ind].biomass

                x = model.resources[prey_info.Ind].pool_x
                y = model.resources[prey_info.Ind].pool_y
                z = model.resources[prey_info.Ind].pool_z

                if prey_biomass > 0
                    prey_length = prey_info.Length[1] / 10
                    ind_biomass = model.resource_trait[prey_spec, :LWR_a] * prey_length ^ model.resource_trait[prey_spec, :LWR_b]

                    n_inds = ceil(Int, prey_biomass / ind_biomass)
                    max_cons = min(n_inds, floor(Int, time_left / chars.Handling_Time[2][sp]))

                    ration_max = (max_stomachs * biomass_school_ind[j]) - (gut_fullness_ind[j] * biomass_school_ind[j])
                    ration = min(prey_biomass, ind_biomass * max_cons, ration_max)

                    ration = max(0.0, min(ration, prey_biomass))  # Final safety clamp
                    n_consumed = floor(Int, ration / ind_biomass)

                    data.ration[ind] += ration * model.resource_trait[prey_spec, :Energy_density]
                    #model.resources[prey_info.Ind].biomass -= ration

                    # Update local prey_info biomass to keep current predator's loop consistent
                    #prey_info.Biomass = max(0.0, prey_biomass - ration)
                    outputs.consumption[x,y,z,sp,model.n_species + prey_spec] += ration

                end
            end
                total_time < time[ind] && gut_fullness_ind[j] < max_stomachs && prey_index <= length(sorted_prey_buffer)
            if ration > 0

                data.gut_fullness[ind] += (ration / biomass_school_ind[j])
            end

            update_prey_distances(model, sp, j, sorted_prey_buffer, max_dist, 1)
            sort!(sorted_prey_buffer, by = x -> x.Distance)
            prey_index += 1

        end
        time[ind] = max(0.0, time[ind] - total_time)
    end

    return time
end

function resource_predation(model::MarineModel,output::MarineOutputs)
    grid = model.depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    Z_daily_sum = zeros(Float64, lonres, latres, depthres, model.n_species)

    latmax = grid[grid.Name .== "yulcorner", :Value][1]
    cell_size_deg = grid[grid.Name .== "cellsize", :Value][1]

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(deg2rad(latmax))
    cell_area = km_per_deg_lat * km_per_deg_lon * cell_size_deg^2  # km²

    dt = model.dt  # timestep in minutes

    for r in 1:model.n_resource
        min_prey_ratio = model.resource_trait[r, :Min_Prey]
        max_prey_ratio = model.resource_trait[r, :Max_Prey]
        max_ingestion = model.resource_trait[r, :Daily_Ration]
        handling_time = Float64(model.resource_trait[r, :Handling_Time])
        max_pred_size = model.resource_trait[r, :Max_Size]
        pred_a = model.resource_trait[r, :LWR_a]
        pred_b = model.resource_trait[r, :LWR_b]

        mean_pred_size, sd_pred_size = lognormal_params_from_maxsize(max_pred_size)
        pred_length_mean = exp(mean_pred_size + 0.5 * sd_pred_size^2)  # mm
        pred_weight_mean = pred_a * (pred_length_mean / 10)^pred_b  # g

        for x in 1:lonres, y in 1:latres, z in 1:depthres
            pred_idxs = findall(p -> p.sp == r && p.pool_x == x && p.pool_y == y && p.pool_z == z, model.resources)
            if isempty(pred_idxs)
                continue
            end

            pred_biom = sum(getfield.(model.resources[pred_idxs], :biomass)) / cell_area  # g/km²

            if pred_biom <= 0
                continue
            end

            predator_biomass = pred_biom  # g/km²

            all_biomasses = zeros(Float64, model.n_species + model.n_resource)
            species_prey_idxs = Vector{Vector{Int}}(undef, model.n_species)
            species_prey_filters = Vector{BitVector}(undef, model.n_species)

            for sp in 1:model.n_species
                animal_dat = model.individuals.animals[sp].data
                animal_chars = model.individuals.animals[sp].p
                prey_a = animal_chars.LWR_a[2][sp]
                prey_b = animal_chars.LWR_b[2][sp]

                prey_idxs = findall(p -> p.pool_x == x && p.pool_y == y && p.pool_z == z, animal_dat)
                species_prey_idxs[sp] = prey_idxs
                if isempty(prey_idxs)
                    all_biomasses[sp] = 0.0
                    species_prey_filters[sp] = falses(0)
                    continue
                end

                prey_lengths = animal_dat.length[prey_idxs]
                prey_weights = prey_a .* (prey_lengths ./ 10).^prey_b
                prey_biomasses = animal_dat.biomass_school[prey_idxs] ./ cell_area

                prey_filter = (prey_lengths .>= min_prey_ratio * pred_length_mean) .&
                              (prey_lengths .<= max_prey_ratio * pred_length_mean)
                species_prey_filters[sp] = prey_filter
                species_prey_idxs[sp] = prey_idxs[prey_filter]

                if all(.!prey_filter)
                    all_biomasses[sp] = 0.0
                    continue
                end

                total_prey_biomass = sum(prey_biomasses[prey_filter])
                all_biomasses[sp] = total_prey_biomass
            end

            if sum(all_biomasses) == 0
                continue 
            end

            for rr in 1:model.n_resource
                idx = findall(res -> res.sp == rr && res.pool_x == x && res.pool_y == y && res.pool_z == z, model.resources)
                res_biomass = sum(getfield.(model.resources[idx], :biomass)) / cell_area

                if res_biomass <= 0
                    all_biomasses[model.n_species + rr] = 0.0
                    continue
                end

                res_max_length = model.resource_trait[rr, :Max_Size]
                μ, σ = lognormal_params_from_maxsize(res_max_length)

                size_edges = exp.(range(μ - 3σ, μ + 3σ; length=11))
                size_mids = 0.5 .* (size_edges[1:end-1] + size_edges[2:end])
                min_prey_length = min_prey_ratio * pred_length_mean
                max_prey_length = max_prey_ratio * pred_length_mean

                p_edges = cdf.(LogNormal(μ, σ), size_edges)
                p_bins = p_edges[2:end] .- p_edges[1:end-1]
                prey_bin_mask = (size_mids .>= min_prey_length) .& (size_mids .<= max_prey_length)
                res_prey_biomass = sum(p_bins[prey_bin_mask]) * res_biomass

                all_biomasses[model.n_species + rr] = res_prey_biomass
            end

            prop_biomass = all_biomasses ./ sum(all_biomasses)
            for sp in 1:(model.n_species)
                animal_dat = model.individuals.animals[sp].data
                prop_ingestion = prop_biomass[sp] * max_ingestion
                total_prey_biomass = all_biomasses[sp]

                mortality = resource_predation_mortality(
                    total_prey_biomass,
                    predator_biomass,
                    pred_weight_mean,
                    prop_ingestion,
                    handling_time,
                    dt
                )

                kept_idxs = species_prey_idxs[sp]
                kept_biomasses = animal_dat.biomass_school[kept_idxs]
                total_kept_biomass = sum(kept_biomasses)

                if total_kept_biomass > 0
                    biomass_proportions = kept_biomasses ./ total_kept_biomass
                    total_biomass_removed = total_kept_biomass * mortality
                    biomass_removed = biomass_proportions .* total_biomass_removed

                    for (i, idx) in enumerate(kept_idxs)
                        ind_biom = animal_dat.biomass_ind[idx]
                        inds_removed = floor(Int, biomass_removed[i] / ind_biom)
                        animal_dat.biomass_school[idx] -= biomass_removed[i]
                        animal_dat.biomass_school[idx] = max(animal_dat.biomass_school[idx], 0.0)
                        animal_dat.abundance[idx] -= inds_removed
                        if animal_dat.abundance[idx] <= 0
                            animal_dat.alive[idx] = 0
                        end
                        output.mortalities[x,y,z,model.n_species+r,sp] += inds_removed #Add removed individuals to mortality frame for future caluculation of Z
                    end
                end
            end
        end
    end
end


function resource_predation_mortality(
    prey_biomass::Float64,     # g/km²
    pred_biomass::Float64,     # g/km²
    pred_weight::Float64,      # g
    daily_ration::Float64,     # % body weight per day
    handling_time::Float64,    # days·kg
    timestep::Float64          # minutes
)
    N = prey_biomass / 1000  # kg/km²
    total_intake = daily_ration * pred_weight * (pred_biomass / pred_weight) / 1000  # kg/day/km²

    function intake_error(a)
        fr = (a * N) / (1 + a * handling_time * N)
        return (fr - total_intake)^2
    end

    result = optimize(intake_error, 1e-6, 10.0)
    a_opt = Optim.minimizer(result)

    FR_day = (a_opt * N) / (1 + a_opt * handling_time * N)  # kg/day/km²
    timestep_frac = timestep / 1440  # day fraction
    FR_timestep = FR_day * timestep_frac  # kg/km² per timestep

    # Convert to proportion of prey removed
    mortality = (FR_timestep * 1000) / prey_biomass  # unitless fraction
    return clamp(mortality, 0.0, 1.0)
end
