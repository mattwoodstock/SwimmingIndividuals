function predation_mortality(model::MarineModel,df,outputs)
    if model.iteration > model.spinup
        model.individuals.animals[df.Sp[1]].data.ac[df.Ind[1]] = 0.0
        model.individuals.animals[df.Sp[1]].data.behavior[df.Ind[1]] = 4
        outputs.mortalities[df.Sp[1],1] += 1 #Add one to the predation mortality column
    end
    return nothing
end

function starvation(model,dead_sp, sp, i, outputs)
    starve = findall(x -> x < 0,dead_sp.data.energy[i])
    if model.iteration > model.spinup .& length(starve) > 0
        dead_sp.data.ac[i[starve]] .= 0.0
        dead_sp.data.behavior[i[starve]] .= 5
        #outputs.production[model.iteration,sp] .+= model.individuals.animals[sp].data.weight[i] #For P/B iteration
    end
    return nothing
end

function reduce_pool(model,pool,ind,ration)
    model.pools.pool[pool].data.biomass[ind] -= ration[1]

    if model.pools.pool[pool].data.biomass[ind] <= 0 #Make sure pool actually stays alive
        new_patch_location(model,pool,ind)
    end
    return nothing
end

function new_patch_location(model,patch,ind)
    environment = model.environment
    g = model.grid
    grid_file = files[files.File .== "grid", :Destination][1]
    grid = CSV.read(grid_file, DataFrame)

    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    z_interval = maxdepth / depthres

    min_z = round.(Int, z_interval .* (1:g.Nz) .- z_interval .+ 1)
    max_z = round.(Int, z_interval .* (1:g.Nz) .+ 1)

    pool_z = Int(model.pools.pool[patch].data.pool_z[ind])

    x_part, y_part = smart_placement(environment.chl,1,1)
    lon_chl_res = size(environment.chl,1)
    lat_chl_res = size(environment.chl,2)

    x = lonmin + rand() * ((lonmax - lonmin)*(x_part/lon_chl_res))
    y = latmin + rand() * (latmax - latmin)*(y_part/lat_chl_res)

    z = min_z[pool_z] + rand() * (max_z[pool_z] - min_z[pool_z])

    x = clamp(x,lonmin,lonmax)
    y = clamp(y,latmin,latmax)
    z = clamp(z,1,maxdepth)

    pool_x = max(1,ceil(Int, x / ((lonmax - lonmin) / lonres)))
    pool_y = max(1,ceil(Int, y / ((latmax - latmin) / latres)))

    model.pools.pool[patch].data.biomass[ind] = model.pools.pool[patch].data.init_biomass[ind]
    model.pools.pool[patch].data.x[ind] = x
    model.pools.pool[patch].data.y[ind] = y
    model.pools.pool[patch].data.z[ind] = z
    model.pools.pool[patch].data.pool_x[ind] = pool_x
    model.pools.pool[patch].data.pool_y[ind] = pool_y
end