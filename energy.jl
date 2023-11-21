
function allocate_energy(pred_df,pred_ind,prey)
    #Calculate energy that was gained

    ## energetic content of fishes:
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/jfb.15331
    energy = prey.Weight[1] * pred_df.p.energy_density[2][prey.Sp[1]]

    egestion = energy * 0.16
    excretion = (energy-egestion) * 0.1
    sda = (energy-egestion) * 0.175

    # Change in energy = consumed energy - (rmr + sda) - (egestion + excretion). RMR will be calculated later.
    pred_df.data.energy[pred_ind] = pred_df.data.energy[pred_ind] + (energy - sda - (egestion + excretion))

end

function evacuate_gut!(pred_df,pred_ind,dt)

    temp = 10
    #Pakhomov et al. 1996
    #Gut evacuation per hour converted to per time step.
    e = (0.0942*exp(0.0708*temp)) * (dt/60)

    pred_df.data.gut_fullness[pred_ind] = pred_df.data.gut_fullness[pred_ind] - e

    #Gut cannot be less than empty.
    if pred_df.data.gut_fullness[pred_ind] < 0
        pred_df.data.gut_fullness[pred_ind] = 0
    end

    return nothing
end

function metabolism(pred_df,pred_ind,dt)

    ts = dt / 60
    temp = 10 #Will need to add temp
    
    ## Equation from Davison et al. 2013

    #In Julia, log = natural log, while log10 = log base 10.
    smr = ((8.52*10^10) * ((pred_df.data.weight[pred_ind]/1000)^ 0.83) * (exp(-0.655/((8.62*10^-5)*(273.15+temp)))) * ((exp(-0.4568/((8.62*10^-5)*273.15)))/(exp(-0.655/((8.62*10^-5)*273.15))))) * ts

    #Active type adds to SMR
    rmr = smr * (1 + (pred_df.data.active_time[pred_ind]/dt))

    pred_df.data.energy[pred_ind] = pred_df.data.energy[pred_ind] - rmr

    return nothing 
end

