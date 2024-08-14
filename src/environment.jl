function generate_environment(files::DataFrame)
    environment = files[files.File .=="environment",:Destination][1]   
    ds = Dataset(environment, "r")
    temperature_var = copy(ds["temperature"])
    z_t = copy(ds["z"][1:35])

    chl_var = copy(ds["chl"])
    close(ds)
    return MarineEnvironment(temperature_var, z_t,chl_var)
end

function individual_temp(model,sp,ind,environment)
    files = model.files
    grid_file = files[files.File .=="grid",:Destination][1]
    grid = CSV.read(grid_file,DataFrame)
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    envi_ts = 1
    sub_temp = environment.temp[:,:,:,envi_ts]

    x = model.individuals.animals[sp].data.x[ind] / (lonmax-lonmin) * size(sub_temp)[1]
    y =  model.individuals.animals[sp].data.y[ind] / (latmax-latmin) * size(sub_temp)[2]
    z = model.individuals.animals[sp].data.z[ind]

    ind_val = trilinear_interpolation_irregular_z(sub_temp,x,y,z,environment.temp_z)
    return ind_val
end

function smart_placement(matrix,ts,n_selections)
    nrows, ncols = size(matrix[:,:,ts])
    # Flatten the matrix and normalize to create a probability distribution
    flattened_mat = reshape(matrix[:,:,ts], nrows * ncols)
    prob_dist = flattened_mat / sum(flattened_mat)
    # Create cumulative probabilities
    cumulative_probs = cumsum(prob_dist)    
    # Sample indices based on cumulative probabilities
    indices = 1:(nrows * ncols)
    sampled_indices = Vector{Int}(undef, n_selections)
    for i in 1:n_selections
        rand_val = rand()
        sampled_index = searchsortedfirst(cumulative_probs, rand_val)
        # Debug: Print the sampled index
        sampled_indices[i] = indices[sampled_index]
    end
    # Convert flat indices to matrix indices
    xy_coords = [(divrem(index - 1, ncols) .+ 1) for index in sampled_indices]
    # Access the first tuple in the vector
    first_tuple = xy_coords[1]

    # Access the first and second values of the first tuple
    first_value = first_tuple[1]
    second_value = first_tuple[2]
    return first_value,second_value
end