function allocate_energy(pred_df,pred_sp,pred_ind,prey)
    #Calculate energy that was gained

    ## energetic content of fishes:
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/jfb.15331
    energy = prey.Weight[1] * pred_df.p.energy_density[2][prey.Sp[1]]

    egestion = energy * 0.16
    excretion = (energy-egestion) * 0.1
    sda = (energy-egestion) * 0.175

    # Change in energy = consumed energy - (rmr + sda) - (egestion + excretion). RMR will be calculated later.
    pred_df.data.energy[pred_ind] = pred_df.data.energy[pred_ind] + (energy - sda - (egestion + excretion))

    if pred_df.data.energy[pred_ind] >= pred_df.data.weight[pred_ind] * pred_df.p.energy_density[2][pred_sp] * 0.2 #Reaches maximum energy storage
        
        #Activate growth protocol. Does not currently work.
        #growth!(pred_df,pred_sp,pred_ind)

        #Return energy calculation to storage
        pred_df.data.energy[pred_ind] = pred_df.data.weight[pred_ind] * pred_df.p.energy_density[2][pred_sp] .* 0.2
    end
end

function evacuate_gut!(pred_df,pred_ind,dt,temp)

    ind_temp = individual_environment_1D(temp,pred_df,pred_ind)

    #Pakhomov et al. 1996
    #Gut evacuation per hour converted to per time step.
    e = (0.0942*exp(0.0708*ind_temp)) * (dt/60)

    pred_df.data.gut_fullness[pred_ind] = pred_df.data.gut_fullness[pred_ind] - (pred_df.data.gut_fullness[pred_ind] * e) 

    #Gut cannot be less than empty.
    if pred_df.data.gut_fullness[pred_ind] < 0
        pred_df.data.gut_fullness[pred_ind] = 0
    end

    return nothing
end

function growth(df,sp,ind)
        maximum = df.data.weight[ind] * df.p.energy_density[2][sp] * 0.2

        surplus = max(0,df.data.energy[ind]-maximum)

        energy_coupler = 1.288 * 10^-4
        
        weight = length^3 + energy_coupler*(E + Er)

end

function metabolism(pred_df,pred_ind,dt,temp)

    ind_temp = individual_environment_1D(temp,pred_df,pred_ind)

    ts = dt / 60
    
    ## Equation from Davison et al. 2013

    #In Julia, log = natural log, while log10 = log base 10.
    smr = ((8.52*10^10) * ((pred_df.data.weight[pred_ind]/1000)^ 0.83) * (exp(-0.655/((8.62*10^-5)*(273.15+ind_temp)))) * ((exp(-0.4568/((8.62*10^-5)*273.15)))/(exp(-0.655/((8.62*10^-5)*273.15))))) * ts

    #Active type adds to SMR
    rmr = smr * (1 + (pred_df.data.active_time[pred_ind]/dt))

    pred_df.data.energy[pred_ind] = pred_df.data.energy[pred_ind] - rmr

    return nothing 
end


function deb(reserve, ingested_energy,temp,length,weight)
    zoom = 1
    assimilation_efficiency = 0.8
    frac_growth = 0.8
    pam = 22.5*zoom
    pm = 18
    max_volumetric_length = frac_growth*pam/pm
    v_dot = 0.02 #cm per day
    v_cost = 2800 # joules per cubic centimeter
    volume = length ^ 3
    pt = 0 #surface-area somatic maintenance rate
    dv = #Structure density
    uv = #Chemical potential of structure
    wv = #Molar weight of structure
    kj = 0.002 #Maturity maintenance coefficient
    eh = #Cumulated energy into development. Will need to initialize according to maturity storage
    kr = 0.95 #Fraction of reproduction buffer fixed into eggs
    pr = 1

    t = temp #Kelvins
    t1 = 293 #Kelvins

    ta = 5300 #Arhennius temperature
    adj_temp = exp((ta/t1)-(ta/t))

    pa = 0.8 * ingested_energy

    g = (v_dot*v_cost)/(frac_growth*pam)
    e = (reserve/volume)*(v_dot/pam)

    pc = adj_temp*pam*length^2*((g*e)/(g+e))*(1+(length/g*max_volumetric_length))

    reserve = pa - pc


    ps = adj_temp*((pm*length^3)+(pt*length^2))
    pg = (1/v_cost)*(frac_growth*pc - ps)

    kg = (uv*dv/(wv*v_cost))

    pj = adj_temp*kj*eh

    pd = ps + (pj + (1-kr)*pr)
    growth = (1/weight)*(pa - (1-kg)*pg-pd)


    return growth
end