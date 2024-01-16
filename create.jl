function generate_individuals(params::Dict, arch::Architecture, Nsp, N, maxN, g::AbstractGrid,z_night_file)
    plank_names = Symbol[]
    plank_data=[]

    if length(N) ≠ Nsp
        throw(ArgumentError("The length of `N_individual` must be $(Nsp), the same as `N_species`, each species has its own initial condition"))
    end

    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_plankton(arch, params, maxN)
        generate_plankton!(plank, N[i], g, arch,i, maxN,z_night_file)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end


function generate_pools(arch::Architecture, params::Dict, Npool, g::AbstractGrid,z_pool_night_file,grid)
    pool_names = Symbol[]
    pool_data=[]

    for i in 1:Npool
        name = Symbol("pool"*string(i))
        pool = construct_pool(arch,params,g)
        generate_pool!(pool, g ,i, z_pool_night_file,grid)
        push!(pool_names, name)
        push!(pool_data, pool)
    end
    groups = NamedTuple{Tuple(pool_names)}(pool_data)
    return pools(groups)
end

function construct_plankton(arch::Architecture, params::Dict, maxN)
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),length = zeros(maxN), weight = zeros(maxN), energy = zeros(maxN), generation = zeros(maxN), target_z = zeros(maxN), mig_status = zeros(maxN), mig_rate = zeros(maxN), rmr = zeros(maxN), active_time = zeros(maxN),gut_fullness = zeros(maxN),feeding = zeros(maxN),dives_remaining = zeros(maxN),interval = zeros(maxN), daily_ration = zeros(maxN), pool_z = zeros(maxN)) 

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:DVM_trigger, :Dive_Interval, :Day_depth_min, :Daily_ration, :Day_depth_max, :Fecundity, :LWR_b, :Surface_Interval, :SpeciesLong, :LWR_a, :VBG_K, :VBG_t0, :Max_Size, :SpeciesShort, :M_const, :Dive_depth_max,:Night_depth_min, :energy_density, :Abundance, :Dive_depth_min, :Min_Size, :Dive_Frequency, :N_conc, :Night_depth_max, :Assimilation_eff, :Swim_velo, :VBG_LOO, :Sex_rat)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function construct_pool(arch::Architecture, params::Dict, g)
    rawdata = StructArray(num = zeros(Float64,g.Nx,g.Ny,g.Nz)) 

    density = replace_storage(array_type(arch), rawdata)

    param_names=(:LWR_a,:Trophic_Level , :Group, :Max_Size, :LWR_b, :Total_density, :Min_Size)

    characters = NamedTuple{param_names}(params)

    return groups(density, characters)
end

function generate_plankton!(plank, N::Int64, g::AbstractGrid, arch::Architecture,sp, maxN,z_night_file)

    z_night_dist = CSV.read(z_night_file,DataFrame)

    plank.data.generation[1:N] .= 1.0   # generation
    
    if z_night_dist[sp,"Type"] == "Distribution"
        plank.data.z[1:N] .= gaussmix(N,z_night_dist[sp,"mu1"],z_night_dist[sp,"mu2"],z_night_dist[sp,"mu3"],z_night_dist[sp,"sigma1"],z_night_dist[sp,"sigma2"],z_night_dist[sp,"sigma3"],z_night_dist[sp,"lambda1"],z_night_dist[sp,"lambda2"])
    else #Sample between min and max
        plank.data.z[1:N] .= rand(plank.p.Night_depth_min[2][sp]:plank.p.Night_depth_max[2][sp])
    end

    plank.data.pool_z[1:N] .= 1

    ## Optimize this? Want all individuals to be different
    for i in 1:N
        plank.data.length[i] = rand(plank.p.Min_Size[2][sp]:plank.p.Max_Size[2][sp])
        plank.data.weight[i]  = plank.p.LWR_a[2][sp] * plank.data.length[i] * plank.p.LWR_b[2][sp]   # Bm
        plank.data.gut_fullness[i] = rand() * 0.03 * plank.data.weight[i] #Proportion of gut that is full. Start with a random value between empty and 3% of predator diet.
        plank.data.interval[i] = rand(0.0:plank.p.Surface_Interval[2][sp])

        while plank.data.z[i] < 0 #Resample if animal is at negative depth
            plank.data.z[i] = gaussmix(1,z_night_dist[sp,"mu1"],z_night_dist[sp,"mu2"],z_night_dist[sp,"mu3"],z_night_dist[sp,"sigma1"],z_night_dist[sp,"sigma2"],z_night_dist[sp,"sigma3"],z_night_dist[sp,"lambda1"],z_night_dist[sp,"lambda2"])[1]
        end
    end

    plank.data.x[1:N] .= [rand(-70:-70) for i in 1:N]
    plank.data.y[1:N] .= [rand(44:44) for i in 1:N]

    plank.data.energy  .= plank.data.weight * plank.p.energy_density[2][sp] .* 0.2   # Initial reserve energy = Rmax
    plank.data.target_z .= copy(plank.data.z)
    plank.data.mig_status .= 0
    plank.data.mig_rate .= 0

    plank.data.feeding .= 1 #Animal can feed. 0 if the animal is not in a feeding cycle

    plank.data.daily_ration .= 0

    if N != maxN #Remove the individuals that are not created out of the model domain
        plank.data.x[N+1:maxN]   .= -1
        plank.data.y[N+1:maxN]   .= -1
    end

    plank.data.dives_remaining .= plank.p.Dive_Frequency[2][sp]

    mask_individuals!(plank.data, g, N, arch)
end

function generate_pool!(groups, g::AbstractGrid,sp, z_night_file,grid)

    z_night_dist = CSV.read(z_night_file,DataFrame)

    maxdepth = grid[grid.Name .== "depthmax", :Value][1]

    depthres = grid[grid.Name .== "depthres", :Value][1]

    z_interval = maxdepth/depthres

    for pool in 1:sp
        for i in 1:g.Nx
            for j in 1:g.Ny

                # Example parameters for the multimodal distribution
                means = [z_night_dist[pool,"mu1"],z_night_dist[pool,"mu2"],z_night_dist[pool,"mu3"]]
                stds = [z_night_dist[pool,"sigma1"],z_night_dist[pool,"sigma2"],z_night_dist[pool,"sigma3"]]
                weights = [z_night_dist[pool,"lambda1"],z_night_dist[pool,"lambda2"],z_night_dist[pool,"lambda3"]]

                x_values = collect(0:maxdepth)
                pdf_values = [multimodal_distribution(x, means, stds, weights) for x in x_values]

                for k in 1:g.Nz
                    min_z = round(Int,z_interval * k - z_interval + 1)
                    max_z = round(Int,z_interval * k)

                    density = sum(pdf_values[min_z:max_z]) .* groups.characters.Total_density[2][sp]

                    groups.density.num[k,] = density
                end
            end
        end
    end
end


function replace_individuals!(model::MarineModel)
    for i in 1:model.n_species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)
        for j in 1:model.ninds[i]
            if spec_array1.data.x[j] == -1 #Need to replace individual

                #Check to see if first individual can die
                if (i == 1) & (j == 1)
                    println("Dead")
                end

                spec_array1.data.generation[j] = 1.0   # generation

                spec_array1.data.length[j] = rand(spec_array1.p.Min_Size[2][i]:spec_array1.p.Max_Size[2][i])
                spec_array1.data.weight[j]  = spec_array1.p.LWR_a[2][i] * spec_array1.data.length[j] * spec_array1.p.LWR_b[2][i]   # Bm

                while spec_array1.data.z[j] < 0 #Resample if animal is at negative depth
                    spec_array1.data.z[j] = gaussmix(1,z_night_dist[i,"mu1"],z_night_dist[i,"mu2"],z_night_dist[i,"mu3"],z_night_dist[i,"sigma1"],z_night_dist[i,"sigma2"],z_night_dist[i,"sigma3"],z_night_dist[i,"lambda1"],z_night_dist[i,"lambda2"])[1]
                end

                spec_array1.data.x[j]   = rand(-70:70)
                spec_array1.data.y[j]   = rand(44:44)
                spec_array1.data.energy[j]  = spec_array1.data.weight[j] * spec_array1.p.energy_density[2][i]* 0.2   # Initial reserve energy = Rmax

                spec_array1.data.target_z[j] = copy(spec_array1.data.z[j])
                spec_array1.data.mig_status[j] = 0
                spec_array1.data.mig_rate[j] = 0
                spec_array1.data.gut_fullness[j] = rand() * 0.03 *spec_array1.data.weight[j] #Proportion of gut that is full. Start with a random value.
                spec_array1.data.daily_ration[j] = 0
            end
        end
    end
end

function reset!(model::MarineModel)
    for i in 1:model.n_species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)
        for j in 1:model.ninds[i]
            if spec_array1.data.x != -1 #Only reset values for living animals
                #spec_array1.data.daily_ration[j] = 0
            end
        end
    end
end