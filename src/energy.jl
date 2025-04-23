function energy(model::MarineModel, sp::Int, temp::Vector{Float64}, indices)
    # Extract animal data and parameters
    animal_data = model.individuals.animals[sp].data
    animal_chars = model.individuals.animals[sp].p
    ind = collect(indices)  # Convert SubArray to an Array

    animal_ed::Float64 = animal_chars.Energy_density[2][sp]
    consumed::Vector{Float64} = animal_data.ration[ind]
    t_res::Float64 = animal_chars.t_resolution[2][sp]
    weight::Vector{Float64} = animal_data.biomass[ind]
    taxa::String15 = animal_chars.Taxa[2][sp]
    energy_type::String15 = animal_chars.MR_type[2][sp]

    # Active time proportion
    active_time::Vector{Float64} = min.(1, animal_data.active[ind] ./ model.dt)

    # Precompute constants
    egestion_coefficient = 0.2
    excretion_coefficient = 0.2
    sda_coefficient = 0.2
    ts::Float64 = animal_chars.t_resolution[2][sp]
    e::Vector{Float64} = (0.0942 .* exp.(0.0708 .* temp)) .* (ts / 60)

    if energy_type == "deepsea"
        oxy = 13.6 # kj/g
        depth::Vector{Float64} = max.(1, animal_data.z[ind]) # Fix depth below 0

        # Respiration calculation
        log_weight::Vector{Float64} = log.(weight)
        inv_temp::Vector{Float64} = 1000 ./ (273.15 .+ temp)
        log_depth::Vector{Float64} = log.(depth)

        lnr = @. (taxa == "Fish" ? 19.491 + 0.885 * log_weight - 5.770 * inv_temp - 0.261 * log_depth :
                  taxa == "Cephalopod" ? 28.326 + 0.779 * log_weight - 7.903 * inv_temp - 0.365 * log_depth :
                  18.775 + 0.766 * log_weight - 5.265 * inv_temp - 0.113 * log_depth)

        R::Vector{Float64} = (exp.(lnr) / 1140 * (oxy * 1000)) / 60 * t_res
    elseif energy_type == "cetacean"
        min_fmr::Vector{Float64} = (350 .* (weight ./ 1000) .^ 0.75) .* (t_res / 1440) ./ 4184
        max_fmr::Vector{Float64} = (420 .* (weight ./ 1000) .^ 0.75) .* (t_res / 1440) ./ 4184
        R = min_fmr .+ (max_fmr .- min_fmr) .* active_time
    else 
        rmr = weight .^ (3/4) .* exp.(-0.65 ./ (8.617e-5 .* (temp .+ 273.15)))
        R = ((rmr / 2) .* (1 .- active_time)) .+ (rmr .* active_time)
    end

    # Total Respiration (R)
    animal_data.rmr[ind] .= R

    # Egestion & Excretion
    egest::Vector{Float64} = consumed .* egestion_coefficient
    excrete::Vector{Float64} = excretion_coefficient .* (consumed .- egest)

    # SDA calculation
    sda::Vector{Float64} = sda_coefficient .* (consumed .- egest)

    # Gut evacuation calculation
    animal_data.gut_fullness[ind] .= max.(0, animal_data.gut_fullness[ind] .- (animal_data.gut_fullness[ind] .* e))

    # Energy calculation
    leftover::Vector{Float64} = consumed .- (R .+ sda .+ egest .+ excrete)
    animal_data.cost[ind] .= (R .+ sda .+ egest .+ excrete)
    animal_data.energy[ind] .= animal_data.energy[ind] .+ leftover

    # Growth
    r_max::Vector{Float64} = animal_data.biomass[ind] .* animal_ed .* 0.2
    animal_data.energy[ind] .= min.(r_max, animal_data.energy[ind])

    excess::Vector{Float64} = max.(0, animal_data.energy[ind] .- r_max)

    # Check for growing individuals directly
    can_grow::BitVector = (animal_data.length[ind] .< animal_chars.Max_Size[2][sp]) .& (excess .> 0)
    
    if any(can_grow)

        growth_prop::Vector{Float64} = 1 .- (animal_data.length[ind] ./ animal.p.Max_Size[2][sp])
        growing_inds = findall(can_grow)

        growth_energy::Vector{Float64} = excess[growing_inds] .* growth_prop[growing_inds]
        animal_data.energy[ind[growing_inds]] .-= growth_energy
        animal_data.biomass[ind[growing_inds]] .+= growth_energy ./ animal_ed
        animal_data.length[ind[growing_inds]] .= ((animal_data.biomass[ind[growing_inds]] ./ animal_chars.LWR_a[2][sp]) .^ (1 / animal_chars.LWR_b[2][sp])) .* 10

        #Calculate probability of reaching maturity as a probablistic function
        mature_weight::Float64 = animal_chars.WMat_trigger[2][sp] * (animal_chars.LWR_a[2][sp] * (animal_chars.Max_Size/10)^animal_chars.LWR_b[2][sp])
        mature_prob::Vector{Float64} = logistic(animal_data.length[ind[growing_inds]],5,mature_weight)

        trigger = rand(length(ind[growing_inds]))
        to_mature = findall(x -> x > trigger,mature_prob)

        animal_data.mature[ind[growing_inds[to_mature]]] .= 1.0

        println(animal_data.mature)
        stop
    end

    # Reproduction
    is_mature::BitVector = (animal_data.mature[ind] .== 1.0)
    can_repro::BitVector = is_mature .& (excess .> 0)

    spawn_season = CSV.read(model.files[model.files.File .== "reproduction",:Destination][1],DataFrame) #Database of grid variables

    species_name = animal_chars.SpeciesLong[2][sp]
    row_idx = findfirst(==(species_name), spawn_season.Species)
    val = spawn_season[row_idx,model.environment.ts+1]

    if any(can_repro) && (val > 0)
        repro_energy::BitVector = max.(0, excess[findall(can_grow)] .* (1 .- growth_prop[findall(can_grow)]))
        reproduce(model, sp, ind[findall(can_repro)], repro_energy,val)
    end

    # Starvation
    starve::BitVector = (animal_data.energy[ind] .< 0)
    if model.iteration > model.spinup && any(starve)
        animal_data.ac[ind[findall(starve)]] .= 0.0
        animal_data.behavior[ind[findall(starve)]] .= 5
    end

    nothing
end
