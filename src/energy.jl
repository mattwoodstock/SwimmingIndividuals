function energy(model::MarineModel, sp::Int, temp::Vector{Float64}, indices)
    # Extract animal data and parameters
    animal_data = model.individuals.animals[sp].data
    animal_chars = model.individuals.animals[sp].p
    ind = collect(indices)

    animal_ed = animal_chars.Energy_density[2][sp]
    consumed = animal_data.ration[ind]                     # C in Wisconsin model
    t_res = model.dt
    weight = animal_data.biomass_school[ind]
    taxa = animal_chars.Taxa[2][sp]
    energy_type = animal_chars.MR_type[2][sp]
    school_size = animal_chars.School_Size[2][sp]
    max_size = animal_chars.Max_Size[2][sp]
    abundance = animal_data.abundance[ind]

    #Test
    percent_bw = (consumed ./ (animal_ed .* animal_data.biomass_school[ind])) .* 100

    active_time = min.(1, animal_data.active[ind] ./ t_res)

    # Constants
    egestion_coeff = 0.1  # F
    excretion_coeff = 0.05 # U
    sda_coeff = 0.05      # SDA
    n_hours = t_res / 60
    evac_prop = min.(1.0, 0.053 .* exp.(0.073 .* temp))

    ## --- RESPIRATION (R) --- ##
    if energy_type == "deepsea"
        oxy = 13.6 # kj/g
        depth = max.(1, animal_data.z[ind])
        log_weight = log.(weight)
        inv_temp = 1000 ./ (273.15 .+ temp)
        log_depth = log.(depth)

        lnr = @. (taxa == "Fish" ? 19.491 + 0.885 * log_weight - 5.770 * inv_temp - 0.261 * log_depth :
                  taxa == "Cephalopod" ? 28.326 + 0.779 * log_weight - 7.903 * inv_temp - 0.365 * log_depth :
                  18.775 + 0.766 * log_weight - 5.265 * inv_temp - 0.113 * log_depth)

        R = (exp.(lnr) / 1140 * (oxy * 1000)) / 60 * t_res
    elseif energy_type == "cetacean"
        min_fmr = (350 .* (weight ./ 1000) .^ 0.75) .* (t_res / 1440) ./ 4184
        max_fmr = (420 .* (weight ./ 1000) .^ 0.75) .* (t_res / 1440) ./ 4184
        R = min_fmr .+ (max_fmr .- min_fmr) .* active_time
    else
        R0 = 0.02
        k = 8.617e-5
        TK = temp .+ 273.15
        rmr = R0 .* weight .^ 0.75 .* exp.(-0.65 ./ (k .* TK))
        R = ((rmr / 2) .* (1 .- active_time)) .+ (rmr .* active_time)
    end

    ## --- Wisconsin Components --- ##
    F = consumed .* egestion_coeff
    U = consumed .* excretion_coeff
    SDA = consumed .* sda_coeff

    ## --- Net Production Energy --- ##
    net_energy = consumed .- (R .+ SDA .+ F .+ U)
    animal_data.gut_fullness[ind] = max.(0.0, animal_data.gut_fullness[ind] .* (1 .- evac_prop) .^ n_hours)

    animal_data.cost[ind] .= R .+ SDA .+ F .+ U
    animal_data.energy[ind] .+= net_energy

    ## --- Growth --- ##
    r_max = weight .* animal_ed .* 0.2
    excess = max.(0, animal_data.energy[ind] .- r_max)
    animal_data.energy[ind] .= min.(r_max, animal_data.energy[ind])

    can_grow = (animal_data.length[ind] .< max_size) .& (excess .> 0)
    growth_prop = exp.(-5* animal_data.length[ind] / max_size)

    if any(can_grow)
        growing_inds = findall(can_grow)
        growth_energy = excess[growing_inds] .* growth_prop[growing_inds]

        # Save current state
        first_biomass_ind = copy(animal_data.biomass_ind[ind[growing_inds]])
        first_length = copy(animal_data.length[ind[growing_inds]])

        proposed_biomass_school = growth_energy ./ animal_ed
        proposed_biomass_ind = proposed_biomass_school ./ abundance[growing_inds]

        proposed_total_biomass = first_biomass_ind .+ proposed_biomass_ind
        proposed_length = ((proposed_total_biomass ./ animal_chars.LWR_a[2][sp]) .^ (1 / animal_chars.LWR_b[2][sp])) .* 10
        growth_ratio = (proposed_length .- first_length) ./ first_length

        final_length = first_length .* (1 .+ growth_ratio)
        final_biomass_ind = animal_chars.LWR_a[2][sp] .* (final_length ./ 10) .^ animal_chars.LWR_b[2][sp]
        final_biomass_school = final_biomass_ind .* school_size

        energy_used = (final_biomass_school .- animal_data.biomass_school[ind[growing_inds]]) .* animal_ed
        animal_data.energy[ind[growing_inds]] .-= energy_used

        animal_data.biomass_school[ind[growing_inds]] .= final_biomass_school
        animal_data.biomass_ind[ind[growing_inds]] .= final_biomass_ind
        animal_data.length[ind[growing_inds]] .= final_length

        # Maturation
        mature_weight = animal_chars.W_mat[2][sp] * (animal_chars.LWR_a[2][sp] * (max_size / 10)^animal_chars.LWR_b[2][sp])
        mature_prob = logistic(animal_data.biomass_ind[ind[growing_inds]], 5, mature_weight)
        trigger = rand(length(ind[growing_inds]))
        to_mature = findall(mature_prob .> trigger)
        animal_data.mature[ind[growing_inds[to_mature]]] .= 1.0
    end

    ## --- Reproduction --- ##
    can_repro = (animal_data.mature[ind] .== 1.0) .& (excess .> 0)
    spawn_season = CSV.read(model.files[model.files.File .== "reproduction", :Destination][1], DataFrame)
    species_name = animal_chars.SpeciesLong[2][sp]
    row_idx = findfirst(==(species_name), spawn_season.Species)
    val = spawn_season[row_idx, model.environment.ts + 1]

    if any(can_repro) && (val > 0)
        repro_inds = findall(can_repro)
        repro_energy = excess[repro_inds] .* (1 .- growth_prop[repro_inds])

        reproduce(model, sp, ind[repro_inds], repro_energy, val)
    end

    ## --- Starvation --- ##
    starve = animal_data.energy[ind] .< 0
    if model.iteration > model.spinup && any(starve)
        animal_data.alive[ind[findall(starve)]] .= 0.0
    end

    return nothing
end
