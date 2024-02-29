function generate_temperature(max_depth::Int; surface_temp::Float64=20.0, deep_temp::Float64=8.0)
    # Calculate temperature gradient
    decay_rate = log(deep_temp / surface_temp) / 300  # Using exponential decay
    # Calculate the depths and temperatures vectorized
    depths = collect(0:3000)
    temperatures = surface_temp .* exp.(decay_rate .* depths)  # Exponential decay

    plt = plot(temperatures, -1*depths, ylabel="Depth (m)", xlabel="Temperature (°C)", label="Temperature profile", lw=2)
    #savefig(plt,joinpath("diags", "Temperature Profile.png"))
    
    return temperatures
end

function individual_temp(model,sp,ind,temp)
    ## Find depth of animal for temperature estimates
    depth = Int(round(model.individuals.animals[sp].data.z[ind],digits=0))
    
    if depth < 0
        depth = 0
    end

    if depth > 1000
        depth = 1000
    end
    ind_val = temp[depth+1] #Currently, depth is each meter
    return ind_val
end