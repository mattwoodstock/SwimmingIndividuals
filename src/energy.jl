function energy(model, sp, temp, ind)
    # Extract animal data and parameters
    animal = model.individuals.animals[sp]
    animal_data = animal.data
    animal_chars = animal.p
    animal_ed = animal.p.Energy_density[2][sp]
    consumed = animal_data.ration[ind]

    # Precompute constants
    oxy = 13.6 # kj/g
    t_res = animal_chars.t_resolution[2][sp]
    weight = animal_data.biomass[ind]
    depth = max.(1,animal_data.z[ind]) #Somehow depth can get below 0. Fix this bug.
    taxa = animal_chars.Taxa[2][sp]

    # Respire calculation using taxa-specific equations
    log_weight = log.(weight)
    inv_temp = 1000 ./ (273.15 .+ temp)
    log_depth = log.(depth)
    
    lnr = zeros(length(weight)) # Preallocate

    if taxa == "Fish"
        lnr .= 19.491 .+ 0.885 .* log_weight .- 5.770 .* inv_temp .- 0.261 .* log_depth
    elseif taxa == "Cephalopod"
        lnr .= 28.326 .+ 0.779 .* log_weight .- 7.903 .* inv_temp .- 0.365 .* log_depth
    elseif taxa == "Crustacean"
        lnr .= 18.775 .+ 0.766 .* log_weight .- 5.265 .* inv_temp .- 0.113 .* log_depth
    end

    rmr = (exp.(lnr) / 1140 * (oxy * 1000)) / 60 * t_res

    # Active time proportion
    active_time = min.(1, animal_data.active[ind] ./ model.dt)

    # Total Respiration (R)
    R = ((rmr / 2) .* (1 .- active_time)) .+ (rmr .* active_time)
    animal_data.rmr[ind] .= R

    # Egestion & Excretion
    egestion_coefficient = 0.2
    excretion_coefficient = 0.2
    egest = consumed .* egestion_coefficient
    excrete = excretion_coefficient .* (consumed .- egest)

    # SDA calculation
    sda_coefficient = 0.2
    sda = sda_coefficient .* (consumed .- egest)

    # Gut evacuation calculation
    ts = animal_chars.t_resolution[2][sp]
    e = (0.0942 .* exp.(0.0708 .* temp)) .* (ts / 60)
    animal_data.gut_fullness[ind] .= max.(0, animal_data.gut_fullness[ind] .- (animal_data.gut_fullness[ind] .* e))

    # Energy calculation
    leftover = consumed .- (R .+ sda .+ egest .+ excrete)
    animal_data.cost[ind] .= (R .+ sda .+ egest .+ excrete)
    animal_data.energy[ind] .= animal_data.energy[ind] .+ leftover

    # Growth
    r_max = animal_data.biomass[ind] .* animal_ed .* 0.2
    animal_data.energy[ind] .= min.(r_max, animal_data.energy[ind])

    excess = max.(0, animal_data.energy[ind] .- r_max)
    have_excess = excess .> 0
    growth_prop = 1 .- (animal_data.length[ind] ./ animal.p.Max_Size[2][sp])

    can_grow = animal_data.length[ind] .< animal_chars.Max_Size[2][sp]
    growing_inds = have_excess .& can_grow

    animal_data.biomass[ind[growing_inds]] .+= (excess[growing_inds] .* growth_prop[growing_inds]) ./ animal_ed
    animal_data.length[ind[growing_inds]] .= ((animal_data.biomass[ind[growing_inds]] ./ animal_chars.LWR_a[2][sp]) .^ (1 / animal_chars.LWR_b[2][sp])) .* 10

    #starvation
    starve = findall(x -> x < 0,animal_data.energy[ind])
    if model.iteration > model.spinup .& length(starve) > 0
        animal_data.ac[ind[starve]] .= 0.0
        animal_data.behavior[ind[starve]] .= 5
    end
    nothing
end
