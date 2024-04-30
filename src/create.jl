function generate_individuals(params::Dict, arch::Architecture, Nsp, N, maxN, g::AbstractGrid,files)
    plank_names = Symbol[]
    plank_data=[]


    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_plankton(arch, params, maxN)
        generate_plankton!(plank, N[i], g, arch,i, maxN,files)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end

function generate_pools(arch::Architecture, params::Dict, Npool, g::AbstractGrid,files)
    pool_names = Symbol[]
    pool_data=[]

    for i in 1:Npool
        name = Symbol("pool"*string(i))
        pool = construct_pool(arch,params,g)
        generate_pool(pool, g ,i, files)
        push!(pool_names, name)
        push!(pool_data, pool)
    end
    groups = NamedTuple{Tuple(pool_names)}(pool_data)
    return pools(groups)
end

function generate_particle(params::Dict,arch::Architecture,max_particle)
    ## Can add more as more particles become necessary
    eDNA = construct_eDNA(arch,params,max_particle)

    return particles(eDNA)
end

function construct_eDNA(arch::Architecture,params,max_particle)
    rawdata = StructArray(species = zeros(max_particle), ind = zeros(max_particle),x = zeros(max_particle),y = zeros(max_particle), z = zeros(max_particle), lifespan = zeros(max_particle)) 

    data = replace_storage(array_type(arch),rawdata)

    param_names=(:Shed_rate,:Decay_rate,:Type)

    state = NamedTuple{param_names}(params)

    #Set x,y,and z of all eDNA particles to equal -1 since they are not yet in the system.
    data.x .= 5e6
    data.y .= 5e6
    data.z .= 5e6

    return eDNA(data,state)
end

function construct_plankton(arch::Architecture, params::Dict, maxN)
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),length = zeros(maxN), weight = zeros(maxN), energy = zeros(maxN), target_z = zeros(maxN), mig_status = zeros(maxN), mig_rate = zeros(maxN), rmr = zeros(maxN), active_time = zeros(maxN),gut_fullness = zeros(maxN),feeding = zeros(maxN),dives_remaining = zeros(maxN),interval = zeros(maxN), dive_capable = zeros(maxN), daily_ration = zeros(maxN), pool_x = zeros(maxN), pool_y = zeros(maxN), pool_z = zeros(maxN),eDNA_shed = zeros(maxN), ration = zeros(maxN), ac = zeros(maxN)) 

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Sex_rat,:Dive_Interval, :Day_depth_min, :Daily_ration, :Day_depth_max, :Fecundity, :LWR_b, :Surface_Interval, :SpeciesLong, :LWR_a, :VBG_K, :VBG_t0, :Max_Size, :t_resolution, :SpeciesShort, :M_const, :Dive_depth_max,:Night_depth_min, :energy_density, :Abundance, :Dive_depth_min, :Min_Size, :Dive_Frequency, :N_conc, :Night_depth_max, :Assimilation_eff, :Swim_velo, :VBG_LOO,:Type)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function construct_pool(arch::Architecture, params::Dict, g)
    rawdata = StructArray(num = zeros(Float64,g.Nx,g.Ny,g.Nz),capacity = zeros(Float64,g.Nx,g.Ny,g.Nz)) 

    density = replace_storage(array_type(arch), rawdata)

    param_names=(:LWR_a,:Avg_energy,:Trophic_Level , :Group, :Max_Size, :LWR_b, :Total_density,:Growth,:Min_Size,:Energy_density)

    characters = NamedTuple{param_names}(params)

    return groups(density, characters)
end

function generate_plankton!(plank, N::Int64, g::AbstractGrid, arch::Architecture,sp, maxN, files)
    grid_file = files[files.File .=="grid",:Destination][1]
    z_dist_file = files[files.File .=="focal_z_dist_night",:Destination][1]

    grid = CSV.read(grid_file,DataFrame)
    z_night_dist = CSV.read(z_dist_file,DataFrame)

    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    ## Optimize this? Want all individuals to be different
    # Set plank data values
    # Generate random numbers on the CPU
    rand_gpu = CUDA.rand(Float64, N)

    plank.data.ac[1:N] .= 1.0
    if arch == GPU()
        plank.data.x .= lonmin .+ rand_gpu .* (lonmax .- lonmin)
        plank.data.y .= latmin .+ rand_gpu .* (latmax .- latmin)
        plank.data.z .= 0.0
        plank.data.length .= plank.p.Min_Size[2][sp] .+ rand_gpu .* (plank.p.Max_Size[2][sp] .- plank.p.Min_Size[2][sp])
        plank.data.weight .= plank.p.LWR_a[2][sp] .* plank.data.length ./ 10 .* plank.p.LWR_b[2][sp]

        plank.data.gut_fullness .= rand_gpu .* 0.03 .* plank.data.weight
        plank.data.interval .= rand_gpu .* plank.p.Surface_Interval[2][sp]
    else
        plank.data.x .= lonmin .+ rand(N) .* (lonmax .- lonmin)
        plank.data.y .= latmin .+ rand(N) .* (latmax .- latmin)
        plank.data.z .= gaussmix(N, z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],
        z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],
        z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],
        z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])
        plank.data.length .= plank.p.Min_Size[2][sp] .+ rand(N) .* (plank.p.Max_Size[2][sp] .- plank.p.Min_Size[2][sp])
        plank.data.weight .= plank.p.LWR_a[2][sp] .* plank.data.length ./ 10 .* plank.p.LWR_b[2][sp]

        plank.data.gut_fullness .= rand(N) .* 0.03 .* plank.data.weight
        plank.data.interval .= rand(N) .* plank.p.Surface_Interval[2][sp]
    end

    # Loop to resample values until they meet the criteria
    while any(plank.data.z .<= 1) || any(plank.data.z .> maxdepth)
        # Resample z values for points outside the grid
        outside_indices = findall((plank.data.z .<= 1) .| (plank.data.z .> maxdepth))
        
        # Resample the values for the resampled indices
        new_values = gaussmix(length(outside_indices), z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],
                            z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],
                            z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],
                            z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])
        
        # Assign the resampled values to the corresponding indices in plank.data.z

        plank.data.z[outside_indices] = new_values
    end

    # Calculate pool indices
    plank.data.pool_x .= ceil.(Int, plank.data.x ./ ((lonmax - lonmin) / lonres))
    plank.data.pool_y .= ceil.(Int, plank.data.y ./ ((latmax - latmin) / latres))
    plank.data.pool_z .= ceil.(Int, plank.data.z ./ (maxdepth / depthres))

    plank.data.energy  .= plank.data.weight * plank.p.energy_density[2][sp] .* 0.2   # Initial reserve energy = Rmax

    plank.data.target_z .= copy(plank.data.z)
    plank.data.dive_capable .= 1

    plank.data.feeding .= 1 #Animal can feed. 0 if the animal is not in a feeding cycle

    if N != maxN #Remove the individuals that are not created out of the model domain
        plank.data.x[N+1:maxN]   .= 5e6
        plank.data.y[N+1:maxN]   .= 5e6
        plank.data.z[N+1:maxN]   .= 5e6
    end

    plank.data.dives_remaining .= plank.p.Dive_Frequency[2][sp]
    plank.data.eDNA_shed .= 0

    return plank.data
end

function generate_pool(groups, g::AbstractGrid, sp, files)
    z_night_file = files[files.File .== "nonfocal_z_dist_night", :Destination][1]
    grid_file = files[files.File .== "grid", :Destination][1]
    state_file = files[files.File .== "state", :Destination][1]

    z_night_dist = CSV.read(z_night_file, DataFrame)
    grid = CSV.read(grid_file, DataFrame)
    state = CSV.read(state_file, DataFrame)

    food_limit = parse(Float64, state[state.Name .== "food_exp", :Value][1])

    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]

    z_interval = maxdepth / depthres

    horiz_cell_size = ((latmax - latmin) / latres) * ((lonmax - lonmin) / lonres) # Square meters of grid cell
    cell_size = horiz_cell_size * (maxdepth / depthres) # Cubic meters of water in each grid cell

    means = [z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"], z_night_dist[sp, "mu3"]]
    stds = [z_night_dist[sp, "sigma1"], z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"]]
    weights = [z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"], z_night_dist[sp, "lambda3"]]

    x_values = 0:maxdepth
    pdf_values = multimodal_distribution.(Ref(x_values), means, stds, weights)
    
    min_z = round.(Int, z_interval .* (1:g.Nz) .- z_interval .+ 1)
    max_z = round.(Int, z_interval .* (1:g.Nz) .+ 1)

    density = [sum(@view pdf_values[1][min_z[k]:max_z[k]]) .* groups.characters.Total_density[2][sp] / maxdepth * horiz_cell_size * (max_z[k] - min_z[k]) for k in 1:g.Nz]

    max_z_lt_200 = max_z .< 200
    food_limit_arr = fill(food_limit, g.Nx, g.Ny)
    density_num = ifelse.(max_z_lt_200, density .* food_limit_arr, density)

    groups.density = reshape(density_num, 1, 1, :)
end

function reset(model::MarineModel)
    files = model.files
    grid_file = files[files.File .=="grid",:Destination][1]
    z_night_dist_file = files[files.File .=="focal_z_dist_night",:Destination][1]
    z_day_dist_file = files[files.File .=="focal_z_dist_day",:Destination][1]

    grid = CSV.read(grid_file,DataFrame)
    z_night_dist = CSV.read(z_night_dist_file,DataFrame)
    z_day_dist = CSV.read(z_day_dist_file,DataFrame)

    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    #Replace dead individuals and necessary components at the end of the day.
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
        species = model.individuals.animals[species_index]
        n_ind = count(!iszero, model.individuals.animals[species_index].data.length) #Number of individuals per species

        model.individuals.animals[species_index].data.ration .= 0 #Reset timestep ration for next one.
        for j in 1:n_ind
            if species.data.x[j] == 5e6 #Need to replace individual
                species.data.ac[j] = 1.0

                species.data.x[j] = lonmin + rand() * (lonmax-lonmin)
                species.data.y[j] = latmin + rand() * (latmax-latmin)

                species.data.pool_x[j] = Int(ceil(species.data.x[j]/((lonmax-lonmin)/lonres),digits = 0))  
                species.data.pool_y[j] = Int(ceil(species.data.y[j]/((latmax-latmin)/latres),digits = 0))  

                species.data.length[j] = species.p.Min_Size[2][species_index] + rand() * (species.p.Max_Size[2][species_index]-species.p.Min_Size[2][species_index])
                species.data.weight[j]  = species.p.LWR_a[2][species_index] * species.data.length[j]/10 * species.p.LWR_b[2][species_index]   # Bm
                species.data.gut_fullness[j] = rand() * 0.03 * species.data.weight[j] #Proportion of gut that is full. Start with a random value between empty and 3% of predator diet.
                species.data.interval[j] = rand() * species.p.Surface_Interval[2][species_index]

                if 6*60 <= model.t < 18*60
                    species.data.z[j] = gaussmix(1,z_day_dist[species_index,"mu1"],z_day_dist[species_index,"mu2"],z_day_dist[species_index,"mu3"],z_day_dist[species_index,"sigma1"],z_day_dist[species_index,"sigma2"],z_day_dist[species_index,"sigma3"],z_day_dist[species_index,"lambda1"],z_day_dist[species_index,"lambda2"])[1]

                    while (species.data.z[j] <= 0) | (species.data.z[j] > maxdepth) #Resample if animal is outside of the grid
                        species.data.z[j] = gaussmix(1,z_day_dist[species_index,"mu1"],z_day_dist[species_index,"mu2"],z_day_dist[species_index,"mu3"],z_day_dist[species_index,"sigma1"],z_day_dist[species_index,"sigma2"],z_day_dist[species_index,"sigma3"],z_day_dist[species_index,"lambda1"],z_day_dist[species_index,"lambda2"])[1]
                    end
                else
                    species.data.z[j] = gaussmix(1,z_night_dist[species_index,"mu1"],z_night_dist[species_index,"mu2"],z_night_dist[species_index,"mu3"],z_night_dist[species_index,"sigma1"],z_night_dist[species_index,"sigma2"],z_night_dist[species_index,"sigma3"],z_night_dist[species_index,"lambda1"],z_night_dist[species_index,"lambda2"])[1]

                    while (species.data.z[j] <= 0) | (species.data.z[j] > maxdepth) #Resample if animal is outside of the grid
                        species.data.z[j] = gaussmix(1,z_night_dist[species_index,"mu1"],z_night_dist[species_index,"mu2"],z_night_dist[species_index,"mu3"],z_night_dist[species_index,"sigma1"],z_night_dist[species_index,"sigma2"],z_night_dist[species_index,"sigma3"],z_night_dist[species_index,"lambda1"],z_night_dist[species_index,"lambda2"])[1]
                    end
                end
                species.data.pool_z[j] = Int(ceil(species.data.z[j]/(maxdepth/depthres),digits=0))

                species.data.energy[j] = species.data.weight[j] * species.p.energy_density[2][species_index]* 0.2   # Initial reserve energy = Rmax

                species.data.gut_fullness[j] = rand() * 0.1 *species.data.weight[j] #Proportion of gut that is full. Start with a random value.
                species.data.daily_ration[j] = 0
                species.data.ration[j] = 0
            end
            if (model.t == 0) #Only reset values for living animals at the end of the day
                species.data.daily_ration .= 0
            end
        end
    end
end



function pool_growth(model)
    #Function that controls the growth of a population back to its carrying capacity
    for (species_index,animal_index) in enumerate(keys(model.pools.pool))
        for i in 1:model.grid.Nz
            growth_rate = model.pools.pool[species_index].characters.Growth[2][species_index]/1440
            population = model.pools.pool[species_index].density[i,]

            model.pools.pool[species_index].density[i,] += growth_rate * population
        end
    end
    return nothing
end