function generate_temperature(files::DataFrame; surface_temp::Float64=26.0, deep_temp::Float64=4.0)

    #Will want to add real data.
    grid_file = files[files.File .=="grid",:Destination][1]

    grid = CSV.read(grid_file,DataFrame)

    # Calculate temperature gradient
    decay_rate = log(deep_temp / surface_temp) / 300  # Using exponential decay
    # Calculate the depths and temperatures vectorized
    depths = collect(0:3000)
    temperatures = surface_temp .* exp.(decay_rate .* depths)  # Exponential decay

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