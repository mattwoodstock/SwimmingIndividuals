function respiration(model,sp,ind,temp)
    #Uses Davison et al. 2013 for equation of Routine Metabolic Rate
    oxy = 13.6 #kj/g

    #From Davison et al. 2013
    #rmr = (0.001*exp(14.47)*model.individuals.animals[sp].data.weight[ind]^0.75*exp((1000*-5.020)/(273.15+temp))) * 1000 * model.individuals.animals[sp].p.t_resolution[2][sp]

    weight = Array(model.individuals.animals[sp].data.biomass[ind])
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
    if model.individuals.animals[sp].data.behavior[ind][1] == 0
        R = rmr/2 #Animal is resting
    elseif model.individuals.animals[sp].data.behavior[ind][1] == 2
        R = rmr*4 #Animal is migrating/diving
    else
        R = rmr #Animal is foraging
    end
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

    r_max = Array(animal_data.biomass[ind] * animal_ed * 0.2)
    excess = Array(animal_data.energy[ind]) + leftover - r_max

    animal_data.energy[ind] = min(r_max, Array(animal_data.energy[ind]) .+ leftover[1])
    if any(x -> x>0,excess)
        growth_prop = min(0.5,animal_data.length[ind][1]/animal.p.Max_Size[2][sp]) #Scale this with distance of length to maximum length? Assume a certain proportion (minimum 50%) is always going towards maturity or reproduction.
        animal_data.biomass[ind] += (excess * growth_prop)/animal_ed
        animal_data.length[ind] .= (animal_data.biomass[ind][1]/animal.p.LWR_a[2][sp]) ^ (1/animal.p.LWR_b[2][sp])*10
        animal_data.mature[ind] .= min(1,animal_data.length[ind][1]/(0.5*(animal.p.Max_Size[2][sp])))
    end
    nothing
end