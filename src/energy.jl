function respiration(model,sp,ind,temp)
    #Uses Davison et al. 2013 for equation of Routine Metabolic Rate
    oxy = 13.6 #kj/g

    #From Davison et al. 2013
    #rmr = (0.001*exp(14.47)*model.individuals.animals[sp].data.weight[ind]^0.75*exp((1000*-5.020)/(273.15+temp))) * 1000 * model.individuals.animals[sp].p.t_resolution[2][sp]

    weight = Array(model.individuals.animals[sp].data.weight[ind])
    t_res = model.individuals.animals[sp].p.t_resolution[2][sp]
    depth = Array(model.individuals.animals[sp].data.z[ind])
    taxa = model.individuals.animals[sp].p.Taxa[2][sp]

    #Langbehn et al. 2019
    #x_prime = -0.655
    #x = -0.4568
    #B = 8.62*10^-5
    #component1 = 8.52*10^10
    #component2 = (weight/1000)^0.83
    #component3 = exp(x_prime/(B*(273.15+temp)))
    #component4 = exp(x/(B*273.15))/exp(x_prime/(B*273.15))
    #smr = (component1 * component2 * component3 * component4) / 60 * t_res #J per timestep

    ## Ikeda et al. 2016
    if taxa == "Fish"
        lnr = 19.491 .+ 0.885*log.(weight).-5.770 .*(1000/(273.15 .+temp)) .-0.261 .*log.(depth)
    elseif taxa == "Cephalopod"
        lnr = 28.326 .+ 0.779*log.(weight).-7.903 .*(1000/(273.15 .+temp)) .-0.365 .*log.(depth)
    elseif taxa == "Crustacean"
        # Ikeda et al. 2014: Representative of copepods. Other crustaceans would have a dummy variable depicting specific taxa (e.g., decapods, euphausiids)
        lnr = 18.775 .+ 0.766*log.(weight).-5.265 .*(1000/(273.15 .+temp)) .-0.113.*log.(depth)
    end
    rmr = (exp(lnr) / 1140 * (oxy*1000)) / 60 * t_res

    smr = rmr / 2
    amr = smr * 4 #Winberg adjustment

    prop_time = model.individuals.animals[sp].data.active_time[ind] / model.individuals.animals[sp].p.t_resolution[2][sp]

    R = first(amr) .* prop_time .+ first(smr)*(1 .-prop_time) 
    model.individuals.animals[sp].data.rmr[ind] = R
    return R
end

function specific_dynamic_action(consumed,egest)
    sda_coefficient = 0.2 #General value used is either between 0.15 or 0.2
    sda = sda_coefficient * (consumed - egest)
    return sda
end

function excretion(model,sp,consumed)
    egestion_coefficient = (1-0.8)
    excretion_coefficient = (1-0.8)

    egest = consumed * egestion_coefficient
    excrete = excretion_coefficient * (consumed - egest)

    return egest, excrete
end

function evacuate_gut(model,sp,ind,temp)
    ts = model.individuals.animals[sp].p.t_resolution[2][sp]

    #Pakhomov et al. 1996
    #Gut evacuation per hour converted to per time step.
    e = (0.0942*exp(0.0708*getindex(temp,1))) * (ts/60)

    model.individuals.animals[sp].data.gut_fullness[ind] = model.individuals.animals[sp].data.gut_fullness[ind] - (model.individuals.animals[sp].data.gut_fullness[ind] * e) 

    #Gut cannot be less than empty.
    if any(x -> x < 0, model.individuals.animals[sp].data.gut_fullness[ind])
        model.individuals.animals[sp].data.gut_fullness[ind] = 0
    end

    return nothing
end

function growth(model, sp, ind, consumed, sda, respire, egest, excrete)
    animal = model.individuals.animals[sp]
    animal_data = animal.data
    animal_ed = animal.p.Energy_density[2][sp]

    leftover = Array(consumed - (respire + sda + egest + excrete))

    r_max = Array(animal_data.weight[ind] * animal_ed * 0.2)
    excess = Array(animal_data.energy[ind]) + leftover - r_max

    animal_data.energy[ind] = min(r_max, Array(animal_data.energy[ind]) + leftover)

    if any(x -> x>0,excess)
        #model.individuals.animals[sp].data.weight[ind] += excess / animal_ed
    end

    nothing
end


function metabolism(model,sp,ind,temp)
    ts = model.individuals.animals[sp].p.t_resolution[2][sp]/60
    ## Equation from Davison et al. 2013

    rmr = 0.0001* exp(14.47)*model.individuals.animals[sp].data.weight[ind]^0.75*exp((1000*-5.02)/(273.15+ind_temp))

    smr = 0.5 * rmr
    amr = 4 * rmr

    full_mr = amr*(model.individuals.animals[sp].data.active_time[ind]/ts) + smr*(1-(model.individuals.animals[sp].data.active_time[ind]/ts))
    model.individuals.animals[sp].data.energy[ind] -= full_mr
    return nothing 
end