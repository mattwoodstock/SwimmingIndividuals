function generate_temperature()
    #Create environmental parameters for 1D model
    numpoints = 1000
    decay_rate = 0.003
    initial_temp = 22.0

    temperature = DataFrame(Depth = Int[], Value = Float64[])
    for i in 1:numpoints
        newrow = Dict("Depth" => i, "Value" => initial_temp * exp.(-decay_rate * i))
        push!(temperature,newrow)
    end
    return temperature
end


function individual_environment_1D(df,pred_df,pred_ind)
    ## Find depth of animal for temperature estimates
    df_sub = df[df.Depth .<= pred_df.data.z[pred_ind],:]

    if nrow(df_sub) == 0
        ind_val = df[1,"Value"]
    else
        ind_val = df_sub[nrow(df_sub),"Value"]
    end

    return ind_val
end