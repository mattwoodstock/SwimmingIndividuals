function allocate_energy(prey_df,prey_spec,prey_ind,pred_df,pred_ind)
    #Calculate energy that was gained

    ## energetic content of fishes:
    #https://onlinelibrary.wiley.com/doi/pdf/10.1111/jfb.15331
    energy = prey_df.data.weight[prey_ind] * prey_df.p.energy_density[2][prey_spec]

    #Update predator energy content
    pred_df.data.energy[pred_ind] = pred_df.data.energy[pred_ind] + energy
end

function evacuate_gut!(pred_df,pred_ind,dt)

    temp = 10
    #Pakhomov et al. 1996
    #Gut evacuation per hour.
    e = (0.0942*exp(0.0708*temp)) / dt

    pred_df.gut_fullness[pred_ind] = pred_df.gut_fullness[pred_ind] * e

    #Gut cannot be less than empty.
    if pred_df.gut_fullness[pred_ind] < 0
        pred_df.gut_fullness[pred_ind] = 0
    end

    return nothing
end

function metabolism!(pred_df,pred_ind)

    temp = 10 #Will need to add temp

    ## Equation from Ikeda et al. 2016
    #https://www.sciencedirect.com/science/article/pii/S0022098116300478

    #In Julia, log = natural log, while log10 = log base 10.
    rmr = exp(19.491 + 0.885*log(pred_df.data.weight[pred_ind]) -5.770 *1000/temp -0.261* log(pred_df.data.z[pred_ind]))

    return rmr 
end

