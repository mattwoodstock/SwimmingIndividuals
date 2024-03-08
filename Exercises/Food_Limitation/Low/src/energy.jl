function allocate_energy(model,sp,ind,prey)
    #Calculate energy that was gained

    ## energetic content of fishes:
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/jfb.15331
    energy = prey.Weight[1] * model.individuals.animals[prey.Sp[1]].p.energy_density[2][prey.Sp[1]]

    egestion = energy * 0.16
    excretion = (energy-egestion) * 0.1
    sda = (energy-egestion) * 0.175

    # Change in energy = consumed energy - (rmr + sda) - (egestion + excretion). RMR will be calculated later.
    model.individuals.animals[sp].data.energy[ind] += (energy - sda - (egestion + excretion))

    if model.individuals.animals[sp].data.energy[ind] >= model.individuals.animals[sp].data.weight[ind] * model.individuals.animals[sp].p.energy_density[2][sp] * 0.2 #Reaches maximum energy storage
        
        #Activate growth protocol. Does not currently work.
        #growth!(pred_df,pred_sp,pred_ind)

        #Return energy calculation to storage
        model.individuals.animals[sp].data.energy[ind] = model.individuals.animals[sp].data.weight[ind] * model.individuals.animals[sp].p.energy_density[2][sp] .* 0.2
    end
end

function respiration(model,sp,ind,temp)

    #Uses Davison et al. 2013 for equation of Routine Metabolic Rate

    #rmr = (0.001*exp(14.47)*model.individuals.animals[sp].data.weight[ind]^0.75*exp((1000*-5.020)/(273.15+temp))) * 1000 * model.individuals.animals[sp].p.t_resolution[2][sp]
    weight = model.individuals.animals[sp].data.weight[ind]
    t_res = model.individuals.animals[sp].p.t_resolution[2][sp]
    x_prime = -0.655
    x = -0.4568
    B = 8.62*10^-5
    component1 = 8.52*10^10
    component2 = (weight/1000)^0.83
    component3 = exp(x_prime/(B*(273.15+temp)))
    component4 = exp(x/(B*273.15))/exp(x_prime/(B*273.15))
    smr = (component1 * component2 * component3 * component4) / 60 * t_res #J per timestep

    #smr = rmr / 2
    amr = smr * 4 #Winberg adjustment

    prop_time = model.individuals.animals[sp].data.active_time[ind] / model.individuals.animals[sp].p.t_resolution[2][sp]

    R = amr * prop_time + smr*(1-prop_time) 

    return R
end

function specific_dynamic_action(consumed,egest)
    sda_coefficient = 0.2 #General value used is either between 0.15 or 0.2
    sda = sda_coefficient * (consumed - egest)
    return sda
end

function excretion(model,sp,consumed)
    egestion_coefficient = (1-model.individuals.animals[sp].p.Assimilation_eff[2][sp])
    excretion_coefficient = (1-model.individuals.animals[sp].p.Assimilation_eff[2][sp])

    egest = consumed * egestion_coefficient
    excrete = excretion_coefficient * (consumed - egest)

    return egest, excrete
end



function evacuate_gut(model,sp,ind,temp)

    ts = model.individuals.animals[sp].p.t_resolution[2][sp]

    #Pakhomov et al. 1996
    #Gut evacuation per hour converted to per time step.
    e = (0.0942*exp(0.0708*temp)) * (ts/60)

    model.individuals.animals[sp].data.gut_fullness[ind] = model.individuals.animals[sp].data.gut_fullness[ind] - (model.individuals.animals[sp].data.gut_fullness[ind] * e) 

    #Gut cannot be less than empty.
    if model.individuals.animals[sp].data.gut_fullness[ind] < 0
        model.individuals.animals[sp].data.gut_fullness[ind] = 0
    end

    return nothing
end

function growth(model, sp, ind, consumed, sda, respire, egest, excrete)
    animal = model.individuals.animals[sp]
    animal_data = animal.data
    animal_ed = animal.p.energy_density[2][sp]

    leftover = consumed - (respire + sda + egest + excrete)

    r_max = animal_data.weight[ind] * animal_ed * 0.2
    excess = animal_data.energy[ind] + leftover - r_max

    animal_data.energy[ind] = min(r_max, animal_data.energy[ind] + leftover)

    if excess > 0
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