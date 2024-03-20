function generate_temperature(files::DataFrame; surface_temp::Float64=26.0, deep_temp::Float64=4.0)

    #Will want to add real data.
    state_file = files[files.File .=="state",:Destination][1]

    state = CSV.read(state_file,DataFrame)

    temp_adjust = parse(Float64,state[state.Name .== "temp_adjust", :Value][1]) #Number of model iterations (i.e., timesteps) to run

    # Calculate temperature gradient
    decay_rate = log(deep_temp / surface_temp) / 300  # Using exponential decay
    # Calculate the depths and temperatures vectorized
    depths = collect(0:3000)
    temperatures = surface_temp .* exp.(decay_rate .* depths) .+ temp_adjust # Exponential decay

    #plt = plot(temperatures, -1*depths, ylabel="Depth (m)", xlabel="Temperature (°C)", label="Temperature profile", lw=2)
    #savefig(plt,joinpath("diags", "Temperature Profile.png"))
    
    return temperatures
end

function individual_temp(model,sp,ind,temp)
    ## Find depth of animal for temperature estimates
    depth = Int(round(model.individuals.animals[sp].data.z[ind],digits=0))

    ind_val = temp[depth+5] #Currently, depth is each meter
    return ind_val
end