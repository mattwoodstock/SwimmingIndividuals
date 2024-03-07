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
    data.x .= -1
    data.y .= -1
    data.z .= -1

    return eDNA(data,state)
end

function construct_plankton(arch::Architecture, params::Dict, maxN)
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),length = zeros(maxN), weight = zeros(maxN), energy = zeros(maxN), target_z = zeros(maxN), mig_status = zeros(maxN), mig_rate = zeros(maxN), rmr = zeros(maxN), active_time = zeros(maxN),gut_fullness = zeros(maxN),feeding = zeros(maxN),dives_remaining = zeros(maxN),interval = zeros(maxN), dive_capable = zeros(maxN), daily_ration = zeros(maxN), pool_x = zeros(maxN), pool_y = zeros(maxN), pool_z = zeros(maxN),eDNA_shed = zeros(maxN), ration = zeros(maxN)) 

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Sex_rat,:Dive_Interval, :Day_depth_min, :Daily_ration, :Day_depth_max, :Fecundity, :LWR_b, :Surface_Interval, :SpeciesLong, :LWR_a, :VBG_K, :VBG_t0, :Max_Size, :t_resolution, :SpeciesShort, :M_const, :Dive_depth_max,:Night_depth_min, :energy_density, :Abundance, :Dive_depth_min, :Min_Size, :Dive_Frequency, :N_conc, :Night_depth_max, :Assimilation_eff, :Swim_velo, :VBG_LOO,:Type)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function construct_pool(arch::Architecture, params::Dict, g)
    rawdata = StructArray(num = zeros(Float64,g.Nx,g.Ny,g.Nz),capacity = zeros(Float64,g.Nx,g.Ny,g.Nz)) 

    density = replace_storage(array_type(arch), rawdata)

    param_names=(:LWR_a,:Trophic_Level , :Group, :Max_Size, :LWR_b, :Total_density,:Growth,:Min_Size,:Energy_density)

    characters = NamedTuple{param_names}(params)
    return groups(density, characters)
end

function generate_plankton!(plank, N::Int64, g::AbstractGrid, arch::Architecture,sp, maxN, files)
    grid_file = files[files.File .=="grid",:Destination][1]

    grid = CSV.read(grid_file,DataFrame)

    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]

    samples = sample_normal(plank.p.Night_depth_min[2][sp],plank.p.Night_depth_max[2][sp],std = 20)

    ## Optimize this? Want all individuals to be different
    for i in 1:N
        index = Int(ceil(rand() * length(samples),digits = 0))

        plank.data.x[i] = lonmin + rand() * (lonmax-lonmin)
        plank.data.y[i] = latmin + rand() * (latmax-latmin)
        plank.data.z[i] = plank.p.Night_depth_min[2][sp] + rand() * (plank.p.Night_depth_max[2][sp] - plank.p.Night_depth_min[2][sp])

        plank.data.pool_x[i] = Int(ceil(plank.data.x[i]/((lonmax-lonmin)/lonres),digits = 0))  
        plank.data.pool_y[i] = Int(ceil(plank.data.y[i]/((latmax-latmin)/latres),digits = 0))  
        plank.data.pool_z[i] = Int(ceil(plank.data.z[i]/(maxdepth/depthres),digits=0))


        plank.data.length[i] = rand(plank.p.Min_Size[2][sp]:plank.p.Max_Size[2][sp])
        plank.data.weight[i]  = plank.p.LWR_a[2][sp] * plank.data.length[i]/10 * plank.p.LWR_b[2][sp]   # Bm
        plank.data.gut_fullness[i] = rand() * 0.03 * plank.data.weight[i] #Proportion of gut that is full. Start with a random value between empty and 3% of predator diet.
        plank.data.interval[i] = rand(0.0:plank.p.Surface_Interval[2][sp])
    end

    plank.data.energy  .= plank.data.weight * plank.p.energy_density[2][sp] .* 0.2   # Initial reserve energy = Rmax
    plank.data.target_z .= copy(plank.data.z)
    plank.data.mig_status .= 0
    plank.data.mig_rate .= 0
    plank.data.dive_capable .= 1
    plank.data.ration .= 0

    plank.data.feeding .= 1 #Animal can feed. 0 if the animal is not in a feeding cycle

    plank.data.daily_ration .= 0

    if N != maxN #Remove the individuals that are not created out of the model domain
        plank.data.x[N+1:maxN]   .= -1
        plank.data.y[N+1:maxN]   .= -1
    end

    plank.data.dives_remaining .= plank.p.Dive_Frequency[2][sp]
    plank.data.eDNA_shed .= 0

    mask_individuals!(plank.data, g, N, arch)
end

function generate_pool(groups, g::AbstractGrid,sp, files)
    z_night_file = files[files.File .=="nonfocal_z_dist_night",:Destination][1]
    grid_file = files[files.File .=="grid",:Destination][1]

    z_night_dist = CSV.read(z_night_file,DataFrame)
    grid = CSV.read(grid_file,DataFrame)

    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]

    z_interval = maxdepth/depthres

    cell_size = ((latmax-latmin)/latres) * ((lonmax-lonmin)/lonres) * (maxdepth/depthres) #cubic meters of water in each grid cell

    for pool in 1:sp
        # Example parameters for the multimodal distribution
        means = [z_night_dist[pool,"mu1"],z_night_dist[pool,"mu2"],z_night_dist[pool,"mu3"]]
        stds = [z_night_dist[pool,"sigma1"],z_night_dist[pool,"sigma2"],z_night_dist[pool,"sigma3"]]
        weights = [z_night_dist[pool,"lambda1"],z_night_dist[pool,"lambda2"],z_night_dist[pool,"lambda3"]]
                        
        x_values = collect(0:maxdepth)
        for i in 1:g.Nx
            for j in 1:g.Ny
                pdf_values = [multimodal_distribution(x, means, stds, weights) for x in x_values]
                pdf_values .= pdf_values/sum(pdf_values) #Normalize

                for k in 1:g.Nz

                    min_z = round(Int,z_interval * k - z_interval + 1)
                    max_z = round(Int,z_interval * k + 1)
                    density = sum(pdf_values[min_z:max_z]) .* groups.characters.Total_density[2][sp] * cell_size

                    groups.density.num[i,j,k] = density
                    groups.density.capacity[i,j,k] = density
                end
            end
        end
    end
end

function reset(model::MarineModel)
    grid_file = model.files[model.files.File .=="grid",:Destination][1]
    grid = CSV.read(grid_file,DataFrame)

    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    depthres = grid[grid.Name .== "depthres", :Value][1]
    lonmax = grid[grid.Name .== "lonmax", :Value][1]
    lonmin = grid[grid.Name .== "lonmin", :Value][1]
    latmax = grid[grid.Name .== "latmax", :Value][1]
    latmin = grid[grid.Name .== "latmin", :Value][1]
    lonres = grid[grid.Name .== "lonres", :Value][1]
    latres = grid[grid.Name .== "latres", :Value][1]
    #Replace dead individuals and necessary components at the end of the day.
    for (species_index,animal_index) in enumerate(keys(model.individuals.animals))
        species = model.individuals.animals[species_index]
        n_ind = count(!iszero, model.individuals.animals[species_index].data.length) #Number of individuals per species

        if 6*60 <= model.t < 18*60
            samples = species.p.Day_depth_min[2][species_index] + rand() * (species.p.Day_depth_max[2][species_index] - species.p.Day_depth_min[2][species_index])
        else
            samples = species.p.Night_depth_min[2][species_index] + rand() * (species.p.Night_depth_max[2][species_index] - species.p.Night_depth_min[2][species_index])
        end

        model.individuals.animals[species_index].data.ration .= 0 #Reset timestep ration for next one.
        for j in 1:n_ind
            if species.data.x[j] == -1 #Need to replace individual
                species.data.length[j] = rand(species.p.Min_Size[2][species_index]:species.p.Max_Size[2][species_index])
                species.data.length[j] = 40 #For experimental purposes.
                species.data.weight[j]  = species.p.LWR_a[2][species_index] * species.data.length[j]/10 * species.p.LWR_b[2][species_index]   # Bm after converting to cm

                index = Int(ceil(rand() * length(samples),digits = 0))
                species.data.z[j] = samples[index]
                species.data.pool_z[j] = Int(ceil(species.data.z[j]/(maxdepth/depthres),digits=0))

                species.data.x[j] = lonmin + rand() * (lonmax-lonmin)
                species.data.y[j] = latmin + rand() * (latmax-latmin)

                species.data.pool_x[j] = Int(ceil(species.data.x[j]/((lonmax-lonmin)/lonres),digits=0))
                species.data.pool_y[j] = Int(ceil(species.data.y[j]/((latmax-latmin)/latres),digits=0))

                species.data.energy[j] = species.data.weight[j] * species.p.energy_density[2][species_index]* 0.2   # Initial reserve energy = Rmax

                species.data.gut_fullness[j] = rand() * 0.03 *species.data.weight[j] #Proportion of gut that is full. Start with a random value.
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
        carrying_capacity = model.pools.pool[species_index].density.capacity[i,]### Create as intitial variables
        growth_rate = model.pools.pool[species_index].characters.Growth[2][species_index]/1440
        population = model.pools.pool[species_index].density.num[i,]

        growth_rate = growth_rate * (1 - population / carrying_capacity)
        model.pools.pool[species_index].density.num[i,] *= 1 + growth_rate
        end
    end
    return nothing
end