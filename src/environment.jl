function generate_environment!()
    # List of months as strings
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    # Mapping of months to indices (1 for "jan", 2 for "feb", ...)
    month_indices = Dict("jan" => 1, "feb" => 2, "mar" => 3, "apr" => 4, "may" => 5, "jun" => 6,"jul" => 7, "aug" => 8, "sep" => 9, "oct" => 10, "nov" => 11, "dec" => 12)

    #Load temperature
    files = readdir("./envi_data/temp")
    temp_data = Array{Float64}(undef, 53, 131, length(files))
    # Assuming temp_data is initialized appropriately

    # Loop over each month
    for month in months
        file_path = "./envi_data/temp/$(month).asc"
        
        # Get the index for the current month (e.g., 1 for "jan", 2 for "feb", etc.)
        month_index = month_indices[month]
        
        # Load the ASCII raster and store it in the corresponding month index of temp_data
        temp_data[:,:,month_index] = load_ascii_raster(file_path)
        temp_data[:,:,month_index] = reverse(temp_data[:,:,month_index], dims=1)

        max_temp = maximum(temp_data[:, :, month_index])
        p1 = heatmap(temp_data[:, :, month_index], title="Temperature $(month)", xlabel="X", ylabel="Y", color=:viridis,clim=(0, max_temp))
        plot(p1)
        savefig("./results/Test/Environment/$(month) temperature plots.png")
    end

    #Load salinity
    files = readdir("./envi_data/sal")
    salt_data = Array{Float64}(undef, 53, 131, length(files))
    # Assuming temp_data is initialized appropriately
    
    # Loop over each month
    for month in months
        file_path = "./envi_data/sal/$(month).asc"
            
        # Get the index for the current month (e.g., 1 for "jan", 2 for "feb", etc.)
        month_index = month_indices[month]
            
        # Load the ASCII raster and store it in the corresponding month index of temp_data
        salt_data[:,:,month_index] = load_ascii_raster(file_path)
        salt_data[:,:,month_index] = reverse(salt_data[:,:,month_index], dims=1)
    
        max_sal = maximum(salt_data[:, :, month_index])
        p1 = heatmap(salt_data[:, :, month_index], title="Temperature $(month)", xlabel="X", ylabel="Y", color=:viridis,clim=(0, max_sal))
        plot(p1)
        savefig("./results/Test/Environment/$(month) salninity plots.png")
    end

    # Load chl 
    files = readdir("./envi_data/chl")
    chl_data = Array{Float64}(undef, 53, 131, length(files))

    for month in months
        file_path = "./envi_data/chl/$(month).asc"
        
        # Get the index for the current month (e.g., 1 for "jan", 2 for "feb", etc.)
        month_index = month_indices[month]
        
        # Load the ASCII raster and store it in the corresponding month index of temp_data
        chl_data[:,:,month_index] = load_ascii_raster(file_path)
        chl_data[:,:,month_index] = reverse(chl_data[:,:,month_index], dims=1)
        max_chl = maximum(chl_data[:, :, month_index])

        p1 = heatmap(chl_data[:, :, month_index], title="chlA $(month)", xlabel="X", ylabel="Y", color=:viridis,clim=(0, max_chl))
        plot(p1)
        savefig("./results/Test/Environment/$(month) chlA plots.png")
    end

    #Load bathymetry
    file_path = "./envi_data/bathymetry/bathymetry.asc"
    bathymetry = load_ascii_raster(file_path)
    bathymetry = reverse(bathymetry, dims=1)

    p1 = heatmap(bathymetry, title="bathymetry", xlabel="X", ylabel="Y", color=:viridis)
    plot(p1)
    savefig("./results/Test/Environment/bathymetry plots.png")

    return MarineEnvironment(bathymetry,temp_data,salt_data,chl_data,1)
end

function ipar_curve(time, peak_ipar=450, peak_time=12, width=4)
    adj_time = time/60
    return peak_ipar * exp(-((adj_time - peak_time)^2) / (2 * width^2))
end

function individual_temp(model::MarineModel, sp::Int, ind::Vector{Int64}, environment::MarineEnvironment)
    envi_ts = environment.ts
    #sub_temp = Array(environment.temp[:, :, envi_ts])

    #clean_temp_grid = coalesce.(sub_temp, NaN)
    #clean_temp_grid = Array(clean_temp_grid)
    #itp = interpolate(clean_temp_grid, BSpline(Linear()))

    x = round.(Int, model.individuals.animals[sp].data.pool_x[ind])
    y = round.(Int, model.individuals.animals[sp].data.pool_y[ind])
    #z = round.(Int, model.individuals.animals[sp].data.pool_z[ind])

    #Trilinear interpolation
    #ind_val = [itp[xi, yi, zi] for (xi, yi, zi) in zip(y, x, z)]
    ind_val = []
    for i in 1:length(y)
        push!(ind_val,environment.temp[y[i], x[i], envi_ts])
    end

    return ind_val
end

function initial_ind_placement(df, sp, grid, n_selections)
    # Grid boundaries
    latmin = grid[grid.Name .== "latmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    capacity = Matrix(df[:,:,1,sp])
    nrows, ncols = size(df[:, :,1, sp])
    data = DataFrame(x=Int[], y=Int[], value=Float64[])
    for i in 1:nrows
        for j in 1:ncols
            if capacity[i,j] > 0.05
                push!(data, (x=j, y=i, value=capacity[i, j]))
            end
        end
    end

    # Sort descending by value
    sort!(data, :value, rev=true)
    # Compute cumulative sum
    data.cumulative_value = cumsum(data.value)
    # Sample a random value between 0 and the total

    total = sum(data.value)
    x_values = []
    y_values = []
    for ind in 1:n_selections
        r = (rand()^2) * total
        # Find the first row where cumulative_value >= r
        selected = findfirst(data.cumulative_value .>= r)
        push!(x_values,data.x[selected])
        push!(y_values,data.y[selected])
    end
    # Return the selected row
    # Cell size
    cell_height = (latmax - latmin) / nrows
    cell_width  = (lonmax - lonmin) / ncols

    # Corrected Y-coordinate flip to match habitat map orientation
    actual_x = lonmin .+ (x_values .- 1) .* cell_width .+ rand(n_selections) .* cell_width
    actual_y = latmin .+ (nrows .- y_values) .* cell_height .+ rand(n_selections) .* cell_height
    if n_selections == 1
        return actual_x[1], actual_y[1], x_values[1], y_values[1]
    end

    pool_x = Int.(x_values)
    pool_y = Int.(y_values)
    capacity = []

    for i in 1:length(pool_x)
        push!(capacity,df[pool_y[i],pool_x[i],sp])
    end
    return actual_x, actual_y, x_values, y_values
end

function pool_placement(df, sp, grid, n_selections)
    # Grid boundaries
    latmin = grid[grid.Name .== "latmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    capacity = Matrix(df[:,:,1,sp])
    nrows, ncols = size(df[:, :, 1,sp])
    data = DataFrame(x=Int[], y=Int[], value=Float64[])
    for i in 1:nrows
        for j in 1:ncols
            if capacity[i,j] > 0.05
                push!(data, (x=j, y=i, value=capacity[i, j]))
            end        
        end
    end

    # Sort descending by value
    sort!(data, :value, rev=true)
    # Compute cumulative sum
    data.cumulative_value = cumsum(data.value)
    # Sample a random value between 0 and the total

    total = sum(data.value)
    x_values = []
    y_values = []
    for ind in 1:n_selections
        r = (rand()^2) * total
        # Find the first row where cumulative_value >= r
        selected = findfirst(data.cumulative_value .>= r)
        push!(x_values,data.x[selected])
        push!(y_values,data.y[selected])
    end
    # Return the selected row
    # Cell size
    cell_height = (latmax - latmin) / nrows
    cell_width  = (lonmax - lonmin) / ncols

    # Corrected Y-coordinate flip to match habitat map orientation
    actual_x = lonmin .+ (x_values .- 1) .* cell_width .+ rand(n_selections) .* cell_width
    actual_y = latmin .+ (nrows .- y_values) .* cell_height .+ rand(n_selections) .* cell_height
    if n_selections == 1
        return actual_x[1], actual_y[1], x_values[1], y_values[1]
    end

    pool_x = Int.(x_values)
    pool_y = Int.(y_values)
    capacity = []

    for i in 1:length(pool_x)
        push!(capacity,df[pool_y[i],pool_x[i],sp])
    end
    return actual_x, actual_y, x_values, y_values
end

function habitat_capacity_multi(env_vars,prefs,sp,type)
    nrows, ncols, nmonths = size(env_vars[1])
    shape = size(env_vars[1])
    suitability_total = ones(Float64, shape...)

    for month in 1:12
        for row in 1:nrows
            for col in 1:ncols
                for env in 1:length(env_vars)
                    if ndims(env_vars[env]) == 3
                        val = env_vars[env][row,col,month]
                    else
                        val = env_vars[env][row,col]
                    end

                    if val < prefs[env].pref_min || val > prefs[env].pref_max
                        suitability = 0.0
                    elseif val <= prefs[env].opt_min
                        suitability = (val - prefs[env].pref_min) / (prefs[env].opt_min - prefs[env].pref_min)
                    elseif val <= prefs[env].opt_max
                        suitability = 1.0
                    else
                        suitability = (prefs[env].pref_max - val) / (prefs[env].pref_max - prefs[env].opt_max)
                    end
                    suitability_total[row,col,month] *= suitability
                end
            end
        end
        heatmap(suitability_total[:,:,month], aspect_ratio=1, c=:viridis, title="Habitat Capacity & Path",xlabel="X", ylabel="Y", colorbar_title="Capacity")
        filename = "./results/Test/Capacities/$type - $sp - $month Capacity.png"
        savefig(filename)
    end
    return suitability_total
end

function initial_habitat_capacity(envi,n_spec,files)
    prefs_df = CSV.read(files[files.File .== "envi_pref",:Destination][1],DataFrame) #Database of grid variables
    trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1],DataFrame)))) #Database of IBM species traits 

    spec_names = trait[:SpeciesLong]

    shape = (size(envi.temp)[1:3]..., n_spec)
    capacities = fill(1.0, shape...)

    total_capacity = ones(Float64, size(envi.temp)...)

    for spec in 1:n_spec
        sp_prefs = prefs_df[prefs_df.species .== spec_names[spec], :]

        sp_env_vars = Vector{AbstractArray}()
        sp_var_prefs = Vector{NamedTuple}()
    
        for row in eachrow(sp_prefs)
            varname = row.variable
        
            # Map variable names to fields in the MarineEnvironment
            env_array = getfield(envi, Symbol(varname))
            @assert env_array !== nothing "Variable '$varname' not found in MarineEnvironment"
        
            push!(sp_env_vars, env_array)
            push!(sp_var_prefs, (pref_min = row.pref_min,opt_min  = row.opt_min,opt_max  = row.opt_max,pref_max = row.pref_max))
        end

        if nrow(sp_prefs) > 0
            species_capacity = habitat_capacity_multi(sp_env_vars, sp_var_prefs,spec,"focal")
            total_capacity .*= species_capacity
        end
        capacities[:,:,:,spec] = total_capacity
    end

    return capacities
end

function pool_habitat_capacity(envi,n_spec,files)
    prefs_df = CSV.read(files[files.File .== "envi_pref",:Destination][1],DataFrame) #Database of grid variables
    trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "nonfocal_trait",:Destination][1],DataFrame)))) #Database of IBM species traits 
    spec_names = trait[:Group]
    shape = (size(envi.temp)[1:3]..., (n_spec+1))
    capacities = fill(1.0, shape...)
    total_capacity = ones(Float64, size(envi.temp)...)
    for spec in 1:(n_spec+1)
        sp_prefs = prefs_df[prefs_df.species .== spec_names[spec], :]
        sp_env_vars = Vector{AbstractArray}()
        sp_var_prefs = Vector{NamedTuple}()
    
        for row in eachrow(sp_prefs)
            varname = row.variable
        
            # Map variable names to fields in the MarineEnvironment
            env_array = getfield(envi, Symbol(varname))
            @assert env_array !== nothing "Variable '$varname' not found in MarineEnvironment"
        
            push!(sp_env_vars, env_array)
            push!(sp_var_prefs, (pref_min = row.pref_min,opt_min  = row.opt_min,opt_max  = row.opt_max,pref_max = row.pref_max))
        end
        if nrow(sp_prefs) > 0
            species_capacity = habitat_capacity_multi(sp_env_vars, sp_var_prefs,spec,"pool")
            total_capacity .*= species_capacity
        end
        capacities[:,:,:,spec] = total_capacity
    end
    return capacities
end