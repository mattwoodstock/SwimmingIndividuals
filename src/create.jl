function generate_individuals(params::Dict, arch::Architecture, Nsp::Int, B, maxN::Int,depths::MarineDepths)
    plank_names = Symbol[]
    plank_data=[]
    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_plankton(arch, params, maxN)
        generate_plankton!(plank, B[i],i,depths)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end

function generate_pools(arch::Architecture, params::Dict, Npool::Int, g::AbstractGrid,maxN::Int,dt::Int,environment::MarineEnvironment,depths::MarineDepths)
    pool_names = Symbol[]
    pool_data=[]

    for i in 1:(Npool+1)
        name = Symbol("pool"*string(i))
        pool = construct_pool(arch,params,g,maxN)
        generate_pool(pool,i,dt,environment,depths)
        push!(pool_names, name)
        push!(pool_data, pool)
    end
    groups = NamedTuple{Tuple(pool_names)}(pool_data)
    return pools(groups)
end

function construct_plankton(arch::Architecture, params::Dict, maxN)
    rawdata = StructArray(x = zeros(Float64,maxN), y = zeros(Float64,maxN), z = zeros(Float64,maxN),length = zeros(Float64,maxN), biomass = zeros(Float64,maxN), energy = zeros(Float64,maxN), target_z = zeros(Float64,maxN), mig_status = zeros(Float64,maxN), mig_rate = zeros(Float64,maxN), rmr = zeros(Float64,maxN), behavior = zeros(Float64,maxN),gut_fullness = zeros(Float64,maxN),cost = zeros(Float64,maxN),dives_remaining = zeros(Float64,maxN),interval = zeros(Float64,maxN), dive_capable = zeros(Float64,maxN), daily_ration = zeros(Float64,maxN), consumed = zeros(Float64,maxN), pool_x = zeros(Float64,maxN), pool_y = zeros(Float64,maxN), pool_z = zeros(Float64,maxN),active = zeros(Float64,maxN), ration = zeros(Float64,maxN), ac = zeros(Float64,maxN), vis_prey = zeros(Float64,maxN), vis_pred = zeros(Float64,maxN),mature = zeros(Float64,maxN),landscape = zeros(Float64,maxN))

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Dive_Interval, :Sex_Ratio,:SpeciesShort,:Min_Prey,:Handling_Time,:LWR_b, :Surface_Interval,:Energy_density,:SpeciesLong, :LWR_a, :Larval_Size, :Hatch_Survival,:Max_Prey, :Max_Size, :t_resolution, :MR_type,  :Swim_velo, :Biomass,:Taxa, :Larval_Duration, :Type)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function construct_pool(arch::Architecture, params::Dict, g,maxN)
    rawdata = StructArray(x = zeros(Float64,maxN), y = zeros(Float64,maxN), z = zeros(Float64,maxN),abundance=zeros(Float64,maxN),biomass = zeros(Float64,maxN),init_biomass= zeros(Float64,maxN), length = zeros(Float64,maxN),vis_prey = zeros(Float64,maxN), vis_pred = zeros(Float64,maxN),ration = zeros(Float64,maxN),pool_x = zeros(Float64,maxN),pool_y = zeros(Float64,maxN),pool_z = zeros(Float64,maxN),focal = zeros(Float64,maxN),age = zeros(Float64,maxN),active = zeros(Float64,maxN)) 

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Trophic_Level,:Min_Prey,:Handling_Time, :LWR_b, :Total_density,:Growth,:Min_Size,:Energy_density,:LWR_a,:Daily_Ration,:Max_Prey, :Group, :Max_Size,:Swim_Velo,:Type)

    characters = NamedTuple{param_names}(params)
    return patch(data, characters)
end

function generate_plankton!(plank, B::Float64,sp::Int,depths::MarineDepths)
    grid = depths.grid
    night_profs = depths.focal_night

    depthres = grid[findfirst(grid.Name .== "depthres"), :Value]
    lonres = grid[findfirst(grid.Name .== "lonres"), :Value]
    latres = grid[findfirst(grid.Name .== "latres"), :Value]
    maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
    lonmax = grid[findfirst(grid.Name .== "lonmax"), :Value]
    lonmin = grid[findfirst(grid.Name .== "lonmin"), :Value]
    latmax = grid[findfirst(grid.Name .== "latmax"), :Value]
    latmin = grid[findfirst(grid.Name .== "latmin"), :Value]

    cell_size = ((latmax - latmin) / latres) * ((lonmax - lonmin) / lonres) # Square meters of grid cell
    target_b = B*cell_size #Total grams to create
    # Set plank data values
    current_b = 0
    ind = 0
    
    while current_b < target_b
        ind += 1
        if ind > length(plank.data.length)
            push!(plank.data.length,rand() .* (plank.p.Max_Size[2][sp]))
            push!(plank.data.biomass,plank.p.LWR_a[2][sp] .* (plank.data.length[ind] ./ 10) .^ plank.p.LWR_b[2][sp])
        else
            plank.data.length[ind] = rand() .* (plank.p.Max_Size[2][sp])
            plank.data.biomass[ind] = plank.p.LWR_a[2][sp] .* (plank.data.length[ind] ./ 10) .^ plank.p.LWR_b[2][sp]
            plank.data.ac[ind] = 1.0
            plank.data.x[ind] = lonmin + rand() * (lonmax - lonmin)
            plank.data.y[ind] = latmin + rand() * (latmax - latmin)
            plank.data.z[ind] = gaussmix(1, night_profs[sp, "mu1"], night_profs[sp, "mu2"],night_profs[sp, "mu3"], night_profs[sp, "sigma1"],night_profs[sp, "sigma2"], night_profs[sp, "sigma3"],night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])[1]
            plank.data.gut_fullness[ind] = rand() * 0.2 * plank.data.biomass[ind]
            plank.data.vis_prey[ind] = visual_range_preys_init(plank.data.length[ind],plank.data.z[ind],plank.p.Min_Prey[2][sp],plank.p.Max_Prey[2][sp],1)[1] * plank.p.t_resolution[2][sp]
            plank.data.vis_pred[ind] = visual_range_preds_init(plank.data.length[ind],plank.data.z[ind],plank.p.Min_Prey[2][sp],plank.p.Max_Prey[2][sp],1)[1] * plank.p.t_resolution[2][sp]
            # Calculate pool indices
            plank.data.pool_x[ind] = max(1,ceil(Int, plank.data.x[ind] / ((lonmax - lonmin) / lonres)))
            plank.data.pool_y[ind] = max(1,ceil(Int, plank.data.y[ind] / ((latmax - latmin) / latres)))
            plank.data.pool_z[ind] = max(1,ceil(Int, plank.data.z[ind] / (maxdepth / depthres)))
            plank.data.pool_z[ind] = clamp(plank.data.pool_z[ind],1,depthres)

            plank.data.energy[ind]  = plank.data.biomass[ind] * plank.p.Energy_density[2][sp] * 0.2   # Initial reserve energy = Rmax
            plank.data.behavior[ind] = 1.0
            plank.data.target_z[ind] = copy(plank.data.z[ind])
            plank.data.dive_capable[ind] = 1
            plank.data.cost[ind] = 0
            plank.data.consumed[ind] = 0
            plank.data.active[ind] = 0
            plank.data.mature[ind] = 0.0 #All animals are immature and will become mature once it is necessary. Does not affect model processes/results.
            plank.data.landscape[ind] = 0.0

        end
        current_b += plank.data.biomass[ind]
    end
    to_append = ind - 1

    x = lonmin .+ rand(to_append) * (lonmax - lonmin)
    y = latmin .+ rand(to_append) * (latmax - latmin)
    z = gaussmix(to_append, night_profs[sp, "mu1"], night_profs[sp, "mu2"],night_profs[sp, "mu3"], night_profs[sp, "sigma1"],night_profs[sp, "sigma2"], night_profs[sp, "sigma3"],
    night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])

    x = clamp.(x,lonmin,lonmax)
    y = clamp.(y,latmin,latmax)
    z = clamp.(z,1,maxdepth)

    pool_x = max.(1,ceil.(Int,x ./ ((lonmax - lonmin) / lonres)))
    pool_y = max.(1,ceil.(Int,y ./ ((latmax - latmin) / latres)))
    pool_z = max.(1,ceil.(Int,z ./ (maxdepth/depthres)))
    pool_z = clamp.(pool_z,1,depthres)
    append!(plank.data.ac, fill(1.0,to_append))
    append!(plank.data.x, x)
    append!(plank.data.y, y)
    append!(plank.data.z, z)
    append!(plank.data.gut_fullness, rand(to_append) .* 0.2 .* plank.data.biomass[(2:ind)])
    append!(plank.data.vis_prey, visual_range_preys_init(plank.data.length[(2:ind)],z,plank.p.Min_Prey[2][sp],plank.p.Max_Prey[2][sp],to_append) .* plank.p.t_resolution[2][sp])
    append!(plank.data.vis_pred, visual_range_preds_init(plank.data.length[(2:ind)],z,plank.p.Min_Prey[2][sp],plank.p.Max_Prey[2][sp],to_append) .* plank.p.t_resolution[2][sp])
    append!(plank.data.pool_x, pool_x)
    append!(plank.data.pool_y, pool_y)
    append!(plank.data.pool_z, pool_z)
    append!(plank.data.energy, plank.data.biomass[(2:ind)] .* plank.p.Energy_density[2][sp] .* 0.2)
    append!(plank.data.behavior, fill(1.0,to_append))
    append!(plank.data.target_z, z)
    append!(plank.data.dive_capable, fill(1,to_append))
    append!(plank.data.cost, fill(0,to_append))
    append!(plank.data.consumed, fill(0,to_append))
    append!(plank.data.active, fill(0,to_append))
    append!(plank.data.mig_status, fill(0,to_append))
    append!(plank.data.mig_rate, fill(0,to_append))
    append!(plank.data.rmr, fill(0,to_append))
    append!(plank.data.dives_remaining, fill(0,to_append))
    append!(plank.data.interval, fill(0,to_append))
    append!(plank.data.daily_ration, fill(0,to_append))
    append!(plank.data.ration, fill(0,to_append))
    append!(plank.data.mature, min.(1,plank.data.length[(2:ind)] ./ (0.5*(plank.p.Max_Size[2][sp]))))
    append!(plank.data.landscape, fill(0.0,to_append))
    return plank.data
end

function generate_pool(group, sp::Int,dt::Int,environment::MarineEnvironment,depths::MarineDepths)
    grid = depths.grid
    night_profs = depths.patch_night

    depthres = grid[findfirst(grid.Name .== "depthres"), :Value]
    lonres = grid[findfirst(grid.Name .== "lonres"), :Value]
    latres = grid[findfirst(grid.Name .== "latres"), :Value]
    maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
    lonmax = grid[findfirst(grid.Name .== "lonmax"), :Value]
    lonmin = grid[findfirst(grid.Name .== "lonmin"), :Value]
    latmax = grid[findfirst(grid.Name .== "latmax"), :Value]
    latmin = grid[findfirst(grid.Name .== "latmin"), :Value]

    z_interval = maxdepth / depthres

    horiz_cell_size = ((latmax - latmin) / latres) * ((lonmax - lonmin) / lonres) # Square meters of grid cell
    cell_size = horiz_cell_size * z_interval # Cubic meters of water in each grid cell

    means = [night_profs[sp, "mu1"], night_profs[sp, "mu2"], night_profs[sp, "mu3"]]
    stds = [night_profs[sp, "sigma1"], night_profs[sp, "sigma2"], night_profs[sp, "sigma3"]]
    weights = [night_profs[sp, "lambda1"], night_profs[sp, "lambda2"], night_profs[sp, "lambda3"]]

    x_values = 0:maxdepth
    pdf_values = multimodal_distribution.(Ref(x_values), means, stds, weights)
    
    min_z = round.(Int, z_interval .* (1:depthres) .- z_interval .+ 1)
    max_z = round.(Int, z_interval .* (1:depthres) .+ 1)

    #g per cubic meter.
    density = [sum(@view pdf_values[1][min_z[k]:max_z[k]])/sum(pdf_values[1]) .* group.characters.Total_density[2][sp] ./ z_interval for k in 1:depthres]

    #Randomly calculate the individuals
    ## Total biomass in grid cell as target
    ## Number of individuals 
    ## Calculate density

    num_patches = parse(Int16,state[state.Name .== "num_patches",:Value][1]) #### User controlled
    patch_num = 0
    for k in 1:depthres
        target_b = density[k][1] * cell_size
        b_remaining = target_b
        for i in 1:num_patches
            den = b_remaining / (num_patches - (i-1))
            patch_num += 1
            ind_size = (group.characters.Min_Size[2][sp] + group.characters.Max_Size[2][sp])/2
            size_sd = (group.characters.Max_Size[2][sp] - group.characters.Min_Size[2][sp])/4
            ind_biom =  group.characters.LWR_a[2][sp] * (ind_size/10) ^ group.characters.LWR_b[2][sp]

            if (den < ind_biom) & (patch_num > 1)
                break
            end
            if i == num_patches
                den = b_remaining
            end
            inds = ceil(den / ind_biom)

            if inds > typemax(Int64)
                patch_inds = BigInt(inds)
            else
                patch_inds = Int64(inds)
            end            
            x_part, y_part = smart_placement(environment.chl,1,1)
            lon_chl_res = size(environment.chl,1)
            lat_chl_res = size(environment.chl,2)

            x = lonmin + rand() * ((lonmax - lonmin)*(x_part/lon_chl_res))
            y = latmin + rand() * (latmax - latmin)*(y_part/lat_chl_res)

            z = min_z[k] + rand() * (max_z[k] - min_z[k])

            x = clamp(x,lonmin,lonmax)
            y = clamp(y,latmin,latmax)
            z = clamp(z,1,maxdepth)

            if patch_num > length(group.data.x)
                push!(group.data.x, x)
                push!(group.data.y, y)
                push!(group.data.z, z)
                push!(group.data.pool_x, max(1,ceil(Int, x / ((lonmax - lonmin) / lonres))))
                push!(group.data.pool_y, max(1,ceil(Int, y / ((latmax - latmin) / latres))))
                push!(group.data.pool_z, max(1,ceil(Int, z / (maxdepth / depthres))))
                push!(group.data.abundance, patch_inds)
                push!(group.data.biomass, den)
                push!(group.data.init_biomass, den)
                push!(group.data.length, ind_size)
                push!(group.data.ration, 0)
                push!(group.data.vis_prey, visual_range_preys_init(ind_size,z,group.characters.Min_Prey[2][sp],group.characters.Max_Prey[2][sp],1)[1]*dt)
                push!(group.data.vis_pred, visual_range_preds_init(ind_size,z,group.characters.Min_Prey[2][sp],group.characters.Max_Prey[2][sp],1)[1]*dt)
                push!(group.data.age,0)
                push!(group.data.active,1.0)
                push!(group.data.focal,0)
            else
                group.data.x[1] = x
                group.data.y[1] = y
                group.data.z[1] = z
                group.data.pool_x[1] = max(1,ceil(Int, x / ((lonmax - lonmin) / lonres)))
                group.data.pool_y[1] = max(1,ceil(Int, y / ((latmax - latmin) / latres)))
                group.data.pool_z[1] = max(1,ceil(Int, z / (maxdepth / depthres)))
                group.data.abundance[1] = patch_inds
                group.data.biomass[1] = den
                group.data.init_biomass[1] = den
                group.data.length[1] = ind_size
                group.data.vis_prey[1] = visual_range_preys_init(ind_size,z,group.characters.Min_Prey[2][sp],group.characters.Max_Prey[2][sp],1)[1]*dt
                group.data.vis_pred[1] = visual_range_preds_init(ind_size,z,group.characters.Min_Prey[2][sp],group.characters.Max_Prey[2][sp],1)[1]*dt
                group.data.ration[1] = 0
                group.data.age[1] = 0
                group.data.active[1] = 1.0
                group.data.focal[1] = 0
            end
            b_remaining -= den
        end
    end
end

function pool_growth(model::MarineModel)
    # Function that controls the growth of a population back to its carrying capacity
    for (pool_index, animal) in enumerate(model.pools.pool)
        growth_rate = animal.characters.Growth[2][pool_index] / (1440/model.dt)  # Daily growth rate adjusted to per minute
        for i in 1:length(animal.data.x)
            population = animal.data.biomass[i]
            carrying_capacity = animal.data.init_biomass[i]  # Assuming carrying capacity is defined here
            # Logistic growth model
            model.pools.pool[pool_index].data.biomass[i] = population + growth_rate * population * (1 - population / carrying_capacity)
        end
    end
    return nothing
end

function reproduce(model,sp,ind,energy)
    sp_dat = model.individuals.animals[sp].data
    sp_char = model.individuals.animals[sp].p

    grid = depths.grid

    depthres = grid[findfirst(grid.Name .== "depthres"), :Value]
    lonres = grid[findfirst(grid.Name .== "lonres"), :Value]
    latres = grid[findfirst(grid.Name .== "latres"), :Value]
    maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
    lonmax = grid[findfirst(grid.Name .== "lonmax"), :Value]
    lonmin = grid[findfirst(grid.Name .== "lonmin"), :Value]
    latmax = grid[findfirst(grid.Name .== "latmax"), :Value]
    latmin = grid[findfirst(grid.Name .== "latmin"), :Value]

    egg_volume = 0.15 .* sp_dat.biomass[ind] .^ 0.14 #From Barneche et al. 2018
    egg_energy = 2.15 .* egg_volume .^ 0.77

    num_eggs = max.(0,Int.(ceil.(energy / egg_energy .* sp_char.Sex_Ratio[2][sp] .* sp_char.Hatch_Survival[2][sp])))

    parent_x = sp_dat.x[ind]
    parent_y = sp_dat.y[ind]
    parent_z = sp_dat.z[ind]

    to_replace = findall(x -> x == 0,model.pools.pool[model.n_pool+1].data.active)
    
    larval_dat = model.pools.pool[model.n_pool+1].data
    for i in 1:length(num_eggs)
        if length(to_replace) > 0 #Replace non-active pools to save space 
            larval_dat.x[to_replace[i]] = parent_x[i]
            larval_dat.y[to_replace[i]] = parent_y[i]
            larval_dat.z[to_replace[i]] = parent_z[i]
            larval_dat.abundance[to_replace[i]] = num_eggs[i]
            larval_dat.length[to_replace[i]] = rand() * sp_char.Larval_Size[2][sp]
            larval_dat.biomass[to_replace[i]] = sp_char.LWR_a[2][sp] * (larval_dat.length[to_replace[i]]/10) ^ sp_char.LWR_b[2][sp]
            larval_dat.biomass[to_replace[i]] = larval_dat.init_biomass[to_replace[i]]
            larval_dat.vis_prey[to_replace[i]] = 0.0
            larval_dat.vis_pred[to_replace[i]] = 0.0
            larval_dat.ration[to_replace[i]] = 0.0
            larval_dat.pool_x[to_replace[i]] = max(1,ceil(Int, larval_dat.x[to_replace[i]] / ((lonmax - lonmin) / lonres)))
            larval_dat.pool_y[to_replace[i]] = max(1,ceil(Int, larval_dat.y[to_replace[i]] / ((latmax - latmin) / latres)))
            larval_dat.pool_z[to_replace[i]] = max(1,ceil(Int, larval_dat.z[to_replace[i]] / (maxdepth / depthres)))
            larval_dat.focal[to_replace[i]] = sp
            larval_dat.age[to_replace[i]] = 0.0
            larval_dat.active[to_replace[i]] = 1.0
        else
            push!(larval_dat.x, parent_x[i])
            push!(larval_dat.y, parent_y[i])
            push!(larval_dat.z, parent_z[i])
            push!(larval_dat.abundance, num_eggs[i])
            new_length = rand() * sp_char.Larval_Size[2][sp]
            push!(larval_dat.length, new_length)
            new_biomass = sp_char.LWR_a[2][sp] * (new_length/10) ^ sp_char.LWR_b[2][sp]
            push!(larval_dat.biomass, new_biomass)
            push!(larval_dat.init_biomass, new_biomass)
            push!(larval_dat.vis_prey, 0.0)
            push!(larval_dat.vis_pred, 0.0)
            push!(larval_dat.ration, 0.0)
            push!(larval_dat.pool_x, max(1,ceil(Int, parent_x[i] / ((lonmax - lonmin) / lonres))))
            push!(larval_dat.pool_y, max(1,ceil(Int, parent_y[i] / ((latmax - latmin) / latres))))
            push!(larval_dat.pool_z, max(1,ceil(Int, parent_z[i] / (maxdepth / depthres))))
            push!(larval_dat.focal, sp)
            push!(larval_dat.age, 0.0)
            push!(larval_dat.active, 1.0)
        end
    end
end

function recruit(model::MarineModel)
    larvae = model.pools.pool[model.n_pool + 1].data

    for sp in 1:model.n_species
        sp_dat = model.individuals.animals[sp].data
        sp_char = model.individuals.animals[sp].p
        spec_larvae = findall(x -> x == sp, larvae.focal)
        of_age = findall(x -> x >= sp_char.Larval_Duration[2][sp], larvae[spec_larvae])

        if isempty(of_age)
            continue # Skip if no larvae of this age
        end

        num_create = larvae.abundance[spec_larvae[of_age]]
        currently_dead = findall(x -> x == 0.0, sp_dat.ac) # Species animals that are currently dead

        if !isempty(currently_dead)
            to_replace = min(length(currently_dead), num_create)
            for j in 1:to_replace
                update_individual!(sp_dat, larvae, spec_larvae[of_age], sp_char, currently_dead[j], j)
            end
            if num_create > to_replace
                to_make = num_create - to_replace
                create_new_individuals!(sp_dat, larvae, spec_larvae[of_age], sp_char, to_make)
            end
        else
            create_new_individuals!(sp_dat, larvae, spec_larvae[of_age], sp_char, num_create)
        end
    end
end

function update_individual!(sp_dat, larvae, spec_larvae, sp_char, idx, age_index)
    grid = model.depths.grid
    depthres::Int64 = grid[findfirst(grid.Name .== "depthres"), :Value]
    lonres::Int64 = grid[findfirst(grid.Name .== "lonres"), :Value]
    latres::Int64 = grid[findfirst(grid.Name .== "latres"), :Value]
    maxdepth::Int64 = grid[findfirst(grid.Name .== "depthmax"), :Value]
    lonmax::Int64 = grid[findfirst(grid.Name .== "lonmax"), :Value]
    lonmin::Int64 = grid[findfirst(grid.Name .== "lonmin"), :Value]
    latmax::Int64 = grid[findfirst(grid.Name .== "latmax"), :Value]
    latmin::Int64 = grid[findfirst(grid.Name .== "latmin"), :Value]
    
    sp_dat.x[idx] = larvae.x[spec_larvae[age_index]]
    sp_dat.y[idx] = larvae.y[spec_larvae[age_index]]
    sp_dat.z[idx] = larvae.z[spec_larvae[age_index]]
    sp_dat.length[idx] = sp_char.Larval_Size[2][sp]
    new_biomass = sp_char.LWR_a[2][sp] * (sp_char.Larval_Size[2][sp]/10) ^ sp_char.LWR_b[2][sp]
    sp_dat.biomass[idx] = new_biomass
    sp_dat.energy[idx] = new_biomass * plank.p.Energy_density[2][sp] * 0.2
    sp_dat.target_z[idx] = larvae.z[spec_larvae[age_index]]

    # Additional initialization
    sp_dat.mig_status[idx] = 0.0
    sp_dat.mig_rate[idx] = 0.0
    sp_dat.rmr[idx] = 0.0
    sp_dat.behavior[idx] = 1.0
    sp_dat.active[idx] = 0.0
    sp_dat.gut_fullness[idx] = rand() * 0.2 * new_biomass
    sp_dat.cost[idx] = 0.0
    sp_dat.dives_remaining[idx] = 0.0
    sp_dat.dive_capable[idx] = 1.0
    sp_dat.daily_ration[idx] = 0.0
    sp_dat.consumed[idx] = 0.0
    sp_dat.pool_x[idx] = max(1, ceil(Int, sp_dat.x[idx] / ((lonmax - lonmin) / lonres)))
    sp_dat.pool_y[idx] = max(1, ceil(Int, sp_dat.y[idx] / ((latmax - latmin) / latres)))
    sp_dat.pool_z[idx] = max(1, ceil(Int, sp_dat.z[idx] / (maxdepth / depthres)))
    sp_dat.ration[idx] = 0.0
    sp_dat.ac[idx] = 1.0
    sp_dat.vis_prey[idx] = visual_range_prey(sp_char.Larval_Size[2][sp], sp_dat.z[idx], sp_char.Min_Prey[2][sp], sp_char.Max_Prey[2][sp], 1)[1] * sp_char.t_resolution[2][sp]
    sp_dat.vis_pred[idx] = visual_range_pred(sp_char.Larval_Size[2][sp], sp_dat.z[idx], sp_char.Min_Prey[2][sp], sp_char.Max_Prey[2][sp], 1)[1] * sp_char.t_resolution[2][sp]
    sp_dat.mature[idx] = 0.0
    sp_dat.landscape[idx] = 0.0
end

function create_new_individuals!(sp_dat, larvae, spec_larvae, sp_char, num_new)
    grid = model.depths.grid
    depthres::Int64 = grid[findfirst(grid.Name .== "depthres"), :Value]
    lonres::Int64 = grid[findfirst(grid.Name .== "lonres"), :Value]
    latres::Int64 = grid[findfirst(grid.Name .== "latres"), :Value]
    maxdepth::Int64 = grid[findfirst(grid.Name .== "depthmax"), :Value]
    lonmax::Int64 = grid[findfirst(grid.Name .== "lonmax"), :Value]
    lonmin::Int64 = grid[findfirst(grid.Name .== "lonmin"), :Value]
    latmax::Int64 = grid[findfirst(grid.Name .== "latmax"), :Value]
    latmin::Int64 = grid[findfirst(grid.Name .== "latmin"), :Value]
    
    # Preallocate new data
    push!(sp_dat.x, fill(larvae.x[spec_larvae], num_new)...)
    push!(sp_dat.y, fill(larvae.y[spec_larvae], num_new)...)
    push!(sp_dat.z, fill(larvae.z[spec_larvae], num_new)...)
    push!(sp_dat.length, fill(sp_char.Larval_Size[2][sp], num_new)...)

    new_biomass = sp_char.LWR_a[2][sp] * (sp_char.Larval_Size[2][sp]/10) ^ sp_char.LWR_b[2][sp]
    new_energy = new_biomass * plank.p.Energy_density[2][sp] * 0.2

    push!(sp_dat.biomass, fill(new_biomass, num_new)...)
    push!(sp_dat.energy, fill(new_energy, num_new)...)

    sp_dat.target_z = append!(sp_dat.target_z, fill(larvae.z[spec_larvae], num_new)...)
    sp_dat.mig_status = append!(sp_dat.mig_status, fill(0.0, num_new)...)
    sp_dat.mig_rate = append!(sp_dat.mig_rate, fill(0.0, num_new)...)
    sp_dat.rmr = append!(sp_dat.rmr, fill(0.0, num_new)...)
    sp_dat.behavior = append!(sp_dat.behavior, fill(1.0, num_new)...)

    new_fullness = rand(num_new) * 0.2 * new_biomass
    push!(sp_dat.gut_fullness, new_fullness...)
    push!(sp_dat.cost, fill(0.0, num_new)...)
    push!(sp_dat.dives_remaining, fill(0.0, num_new)...)
    push!(sp_dat.dive_capable, fill(1.0, num_new)...)
    push!(sp_dat.daily_ration, fill(0.0, num_new)...)
    push!(sp_dat.consumed, fill(0.0, num_new)...)
    push!(sp_dat.active, fill(0.0, num_new)...)

    # Calculate new pools
    new_pool_x = max(1, ceil(Int, sp_dat.x[end] / ((lonmax - lonmin) / lonres)))
    new_pool_y = max(1, ceil(Int, sp_dat.y[end] / ((latmax - latmin) / latres)))
    new_pool_z = max(1, ceil(Int, sp_dat.z[end] / (maxdepth / depthres)))

    push!(sp_dat.pool_x, fill(new_pool_x, num_new)...)
    push!(sp_dat.pool_y, fill(new_pool_y, num_new)...)
    push!(sp_dat.pool_z, fill(new_pool_z, num_new)...)

    push!(sp_dat.ration, fill(0.0, num_new)...)
    push!(sp_dat.ac, fill(1.0, num_new)...)

    new_vis_prey = visual_range_prey(sp_char.Larval_Size[2][sp], sp_dat.z[end], sp_char.Min_Prey[2][sp], sp_char.Max_Prey[2][sp], 1)[1] * sp_char.t_resolution[2][sp]
    new_vis_pred = visual_range_pred(sp_char.Larval_Size[2][sp], sp_dat.z[end], sp_char.Min_Prey[2][sp], sp_char.Max_Prey[2][sp], 1)[1] * sp_char.t_resolution[2][sp]

    push!(sp_dat.vis_prey, fill(new_vis_prey, num_new)...)
    push!(sp_dat.vis_pred, fill(new_vis_pred, num_new)...)
    push!(sp_dat.mature, fill(0.0, num_new)...)
    push!(sp_dat.landscape, fill(0.0, num_new)...)
end

