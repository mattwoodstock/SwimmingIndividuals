function generate_individuals(params::Dict, arch::Architecture, Nsp, N, maxN, g::AbstractGrid)
    plank_names = Symbol[]
    plank_data=[]

    if length(N) ≠ Nsp
        throw(ArgumentError("The length of `N_individual` must be $(Nsp), the same as `N_species`, each species has its own initial condition"))
    end

    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_plankton(arch, i, params, maxN)
        #println(plank.p.Daily_ration[2][1]) #Calling structs!!!!
        generate_plankton!(plank, N[i], g, arch,i)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end

function construct_plankton(arch::Architecture, sp::Int64, params::Dict, maxN)
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),length = zeros(maxN), weight = zeros(maxN), energy = zeros(maxN), age = zeros(maxN), generation = zeros(maxN)) 
    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Dive_Interval,:Daily_ration, :Day_depth_min, :Constant_biomass,:Tmax, :Day_depth_max, :Target_biomass, :Fecundity, :LWR_b, :Surface_Interval, :SpeciesLong, :LWR_a, :VBG_K, :VBG_t0, :Movement_type, :Mat_age, :LF_dist1, :SpeciesShort, :M_type, :VBG_LOO, :M_const, :Night_depth_min,:Dive_depth_max, :Abundance, :Dive_depth_min, :Mig_rate, :Dive_Frequency, :Repro_season, :N_conc,  :Taxon, :Night_depth_max, :LF_dist,  :Assimilation_eff, :Swim_velo, :LF_dist2, :Sex_rat              
    )

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function generate_plankton!(plank, N::Int64, g::AbstractGrid, arch::Architecture,sp)
    tmax = plank.p.Tmax
    L_inf = plank.p.VBG_LOO
    k = plank.p.VBG_K
    t0 = plank.p.VBG_t0
    condition = plank.p.LWR_a
    power = plank.p.LWR_b
    z_min = plank.p.Night_depth_min
    z_max = plank.p.Night_depth_max

    plank.data.generation[1:N] .= 1.0   # generation

    ## Optimize this? Want all individuals to be different
    for i in 1:N
        plank.data.age[i] = rand() * tmax[2][sp] #age
        plank.data.length[i] = L_inf[2][sp] * (1 - exp(-1 *k[2][sp] * (plank.data.age[i] - t0[2][sp])))
        plank.data.weight[i]  = condition[2][sp] * plank.data.length[i] * power[2][sp]   # Bm
        plank.data.z[i] = rand() * (z_max[2][sp] - z_min[2][sp]) + z_min[2][1]
    end

    plank.data.x   .= 1
    plank.data.y   .= 1
    plank.data.energy  .= plank.data.weight .* 0.2   # Initial reserve energy = Rmax

    mask_individuals!(plank.data, g, N, arch)
end