function generate_individuals(params::Dict, arch::Architecture, Nsp, N, maxN, g::AbstractGrid)
    plank_names = Symbol[]
    plank_data=[]

    if length(N) ≠ Nsp
        throw(ArgumentError("The length of `N_individual` must be $(Nsp), the same as `N_species`, each species has its own initial condition"))
    end

    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_plankton(arch, params, maxN)
        #println(plank.p.Daily_ration[2][1]) #Calling structs!!!!
        generate_plankton!(plank, N[i], g, arch,i, maxN)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end

function construct_plankton(arch::Architecture, params::Dict, maxN)
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),length = zeros(maxN), weight = zeros(maxN), energy = zeros(maxN), generation = zeros(maxN), target_z = zeros(maxN), mig_status = zeros(maxN), mig_rate = zeros(maxN), rmr = zeros(maxN), active_time = zeros(maxN),gut_fullness = zeros(maxN),feeding = zeros(maxN),dives_remaining = zeros(maxN),interval = zeros(maxN)) 

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:DVM_trigger, :Dive_Interval, :Day_depth_min, :Daily_ration, :Day_depth_max, :Fecundity, :LWR_b, :Surface_Interval, :SpeciesLong, :LWR_a, :VBG_K, :VBG_t0, :Max_Size, :SpeciesShort, :M_const, :Dive_depth_max,:Night_depth_min, :energy_density, :Abundance, :Dive_depth_min, :Min_Size, :Dive_Frequency, :N_conc, :Night_depth_max, :Assimilation_eff, :Swim_velo, :VBG_LOO, :Sex_rat)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end


function generate_plankton!(plank, N::Int64, g::AbstractGrid, arch::Architecture,sp, maxN)
    plank.data.generation[1:N] .= 1.0   # generation

    ## Optimize this? Want all individuals to be different
    for i in 1:N
        plank.data.length[i] = rand(plank.p.Min_Size[2][sp]:plank.p.Max_Size[2][sp])
        plank.data.weight[i]  = plank.p.LWR_a[2][sp] * plank.data.length[i] * plank.p.LWR_b[2][sp]   # Bm
        plank.data.z[i] = rand(plank.p.Night_depth_min[2][sp]:plank.p.Night_depth_max[2][sp])
        plank.data.interval[i] = rand(0.0:plank.p.Surface_Interval[2][sp])
    end

    plank.data.x   .= -70
    plank.data.y   .= 45
    plank.data.energy  .= plank.data.weight * plank.p.energy_density[2][sp] .* 0.2   # Initial reserve energy = Rmax
    plank.data.target_z .= copy(plank.data.z)
    plank.data.mig_status .= 0
    plank.data.mig_rate .= 0
    plank.data.gut_fullness .= rand() #Proportion of gut that is full. Start with a random value.

    plank.data.feeding .= 1 #Animal can feed. 0 if the animal is not in a feeding cycle

    if N != maxN #Remove the individuals that are not created out of the model domain
        plank.data.x[N+1:maxN]   .= -1
        plank.data.y[N+1:maxN]   .= -1
    end

    plank.data.dives_remaining .= plank.p.Dive_Frequency[2][sp]


    mask_individuals!(plank.data, g, N, arch)
end


function replace_individuals(model::MarineModel)
    for i in 1:model.n_species
        name = Symbol("sp"*string(i))
        spec_array1 = getfield(model.individuals.animals, name)
        for j in 1:model.ninds[i]
            if spec_array1.data.x == -1 #Need to replace individual
                spec_array1.data.generation[j] = 1.0   # generation

                spec_array1.data.length[j] = rand(spec_array1.p.Min_Size[2][sp]:spec_array1.p.Max_Size[2][sp])
                spec_array1.data.weight[j]  = spec_array1.p.LWR_a[2][sp] * spec_array1.data.length[j] * spec_array1.p.LWR_b[2][sp]   # Bm
                spec_array1.data.z[j] = rand(spec_array1.p.Night_depth_min[2][sp]:spec_array1.p.Night_depth_max[2][sp])

                spec_array1.data.x[j]   = -70
                spec_array1.data.y[j]   = 45
                spec_array1.data.energy[j]  = plank.data.weight .* 0.2   # Initial reserve energy = Rmax
                spec_array1.data.target_z[j] = copy(spec_array1.data.z[j])
                spec_array1.data.mig_status[j] = 0
                spec_array1.data.mig_rate[j] = 0
                spec_array1.data.gut_fullness[j] = rand() #Proportion of gut that is full. Start with a random value.
            end
        end
    end
end