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
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),length = zeros(maxN), weight = zeros(maxN), volume = zeros(maxN), energy = zeros(maxN), generation = zeros(maxN), target_z = zeros(maxN), mig_status = zeros(maxN), rmr = zeros(maxN), active_time = zeros(maxN),gut_fullness = zeros(maxN)) 

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:DVM_trigger, :Dive_Interval, :Day_depth_min, :Daily_ration, :Day_depth_max, :Fecundity, :LWR_b, :Surface_Interval, :SpeciesLong, :LWR_a, :VBG_K, :VBG_t0, :Max_Size, :SpeciesShort, :M_const, :Dive_depth_max,:Night_depth_min, :energy_density, :Abundance, :Dive_depth_min, :Min_Size, :Mig_rate, :Dive_Frequency, :N_conc, :Night_depth_max, :Assimilation_eff, :Swim_velo, :VBG_LOO, :Sex_rat)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end


function generate_plankton!(plank, N::Int64, g::AbstractGrid, arch::Architecture,sp, maxN)
    plank.data.generation[1:N] .= 1.0   # generation

    ## Optimize this? Want all individuals to be different
    for i in 1:N
        plank.data.length[i] = rand() * (plank.p.Max_Size[2][sp] - plank.p.Min_Size[2][sp]) + plank.p.Min_Size[2][sp]
        plank.data.weight[i]  = plank.p.LWR_a[2][sp] * plank.data.length[i] * plank.p.LWR_b[2][sp]   # Bm
        plank.data.volume[i] = (π*(plank.data.length[i]/2)^2 * plank.data.length[i]) * 0.75 #Assume animal is a cyclindrical shape that occupies 3/4 of the full cyclinder.
        plank.data.z[i] = rand() * (plank.p.Night_depth_max[2][sp] - plank.p.Night_depth_min[2][sp]) + plank.p.Night_depth_min[2][sp]
    end

    plank.data.x   .= -70
    plank.data.y   .= 45
    plank.data.energy  .= plank.data.weight .* 0.2   # Initial reserve energy = Rmax
    plank.data.target_z .= copy(plank.data.z)
    plank.data.mig_status .= 0
    plank.data.gut_fullness .= rand() #Proportion of gut that is full. Start with a random value.


    if N != maxN #Remove the individuals that are not created out of the model domain
        plank.data.x[N+1:maxN]   .= -1
        plank.data.y[N+1:maxN]   .= -1
    end


    mask_individuals!(plank.data, g, N, arch)
end