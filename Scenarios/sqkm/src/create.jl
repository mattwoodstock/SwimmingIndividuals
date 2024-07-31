function generate_individuals(params::Dict, arch::Architecture, Nsp, B, maxN, g::AbstractGrid,files)
    plank_names = Symbol[]
    plank_data=[]
    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_plankton(arch, params, maxN)
        generate_plankton!(plank, B[i], g, arch,i, maxN,files)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end

function generate_pools(arch::Architecture, params::Dict, Npool, g::AbstractGrid,files,maxN,dt)
    pool_names = Symbol[]
    pool_data=[]

    for i in 1:Npool
        name = Symbol("pool"*string(i))
        pool = construct_pool(arch,params,g,maxN)
        generate_pool(pool, g ,i, files,dt)
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
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),length = zeros(maxN), biomass = zeros(maxN), energy = zeros(maxN), target_z = zeros(maxN), mig_status = zeros(maxN), mig_rate = zeros(maxN), rmr = zeros(maxN), behavior = zeros(maxN),gut_fullness = zeros(maxN),feeding = zeros(maxN),dives_remaining = zeros(maxN),interval = zeros(maxN), dive_capable = zeros(maxN), daily_ration = zeros(maxN), consumed = zeros(maxN), pool_x = zeros(maxN), pool_y = zeros(maxN), pool_z = zeros(maxN),eDNA_shed = zeros(maxN), ration = zeros(maxN), ac = zeros(maxN), vis_prey = zeros(maxN), vis_pred = zeros(maxN),mature = zeros(maxN))

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Dive_Interval,:SpeciesShort,:LWR_b, :Surface_Interval,:Energy_density,:SpeciesLong, :LWR_a, :Max_Size, :t_resolution,  :Swim_velo, :Biomass,:Taxa, :Type)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function construct_pool(arch::Architecture, params::Dict, g,maxN)
    rawdata = StructArray(x = zeros(maxN), y = zeros(maxN), z = zeros(maxN),abundance=zeros(maxN),volume = zeros(maxN),biomass = zeros(maxN),init_biomass= zeros(maxN), length = zeros(maxN),vis_prey = zeros(maxN), vis_pred = zeros(maxN),ration = zeros(maxN)) 

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Trophic_Level, :LWR_b, :Total_density,:Growth,:Min_Size,:Energy_density,:LWR_a,:Avg_energy, :Group, :Max_Size,:Max_patch,:Type)

    characters = NamedTuple{param_names}(params)
    return patch(data, characters)
end

function generate_plankton!(plank, B::Float64, g::AbstractGrid, arch::Architecture,sp, maxN, files)
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
            push!(plank.data.ac, 1.0)
            push!(plank.data.x, lonmin + rand() * (lonmax - lonmin))
            push!(plank.data.y, latmin + rand() * (latmax - latmin))
            push!(plank.data.z, gaussmix(1, z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],
            z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])[1])
            push!(plank.data.gut_fullness, rand() * 0.2 * plank.data.biomass[ind])
            push!(plank.data.vis_prey, visual_range_preys_init(arch,plank.data.length[ind],plank.data.z[ind],1)[1] * plank.p.t_resolution[2][sp])
            push!(plank.data.vis_pred, visual_range_preds_init(arch,plank.data.length[ind],plank.data.z[ind],1)[1] * plank.p.t_resolution[2][sp])
            push!(plank.data.pool_x, ceil(Int, plank.data.x[ind] / ((lonmax - lonmin) / lonres)))
            push!(plank.data.pool_y, ceil(Int, plank.data.y[ind] / ((latmax - latmin) / latres)))
            push!(plank.data.pool_z, ceil(Int, plank.data.z[ind] / (maxdepth / depthres)))
            push!(plank.data.energy, plank.data.biomass[ind] * plank.p.Energy_density[2][sp] * 0.2)
            push!(plank.data.behavior, 1.0)
            push!(plank.data.target_z, copy(plank.data.z[ind]))
            push!(plank.data.dive_capable, 1)
            push!(plank.data.feeding, 1)
            push!(plank.data.consumed, 0)
            push!(plank.data.eDNA_shed, 0)
            push!(plank.data.mig_status, 0)
            push!(plank.data.mig_rate, 0)
            push!(plank.data.rmr, 0)
            push!(plank.data.dives_remaining, 0)
            push!(plank.data.interval, 0)
            push!(plank.data.daily_ration, 0)
            push!(plank.data.ration, 0)
            push!(plank.data.mature,min(1,plank.data.length[ind]/(0.5*(plank.p.Max_Size[2][sp]))))
        else
            plank.data.length[ind] = rand() .* (plank.p.Max_Size[2][sp])
            plank.data.biomass[ind] = plank.p.LWR_a[2][sp] .* (plank.data.length[ind] ./ 10) .^ plank.p.LWR_b[2][sp]
            plank.data.ac[ind] = 1.0
            plank.data.x[ind] = lonmin + rand() * (lonmax - lonmin)
            plank.data.y[ind] = latmin + rand() * (latmax - latmin)
            plank.data.z[ind] = gaussmix(1, z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])[1]
            plank.data.gut_fullness[ind] = rand() * 0.2 * plank.data.biomass[ind]
            plank.data.vis_prey[ind] = visual_range_preys_init(arch,plank.data.length[ind],plank.data.z[ind],1)[1] * plank.p.t_resolution[2][sp]
            plank.data.vis_pred[ind] = visual_range_preds_init(arch,plank.data.length[ind],plank.data.z[ind],1)[1] * plank.p.t_resolution[2][sp]
            # Calculate pool indices
            plank.data.pool_x[ind] = ceil(Int, plank.data.x[ind] / ((lonmax - lonmin) / lonres))
            plank.data.pool_y[ind] = ceil(Int, plank.data.y[ind] / ((latmax - latmin) / latres))
            plank.data.pool_z[ind] = ceil(Int, plank.data.z[ind] / (maxdepth / depthres))
            plank.data.energy[ind]  = plank.data.biomass[ind] * plank.p.Energy_density[2][sp] * 0.2   # Initial reserve energy = Rmax
            plank.data.behavior[ind] = 1.0
            plank.data.target_z[ind] = copy(plank.data.z[ind])
            plank.data.dive_capable[ind] = 1
            plank.data.feeding[ind] = 1 #Animal can feed. 0 if the animal is not in a feeding cycle
            plank.data.consumed[ind] = 0
            plank.data.eDNA_shed[ind] = 0
            plank.data.mature[ind] = min(1,plank.data.length[ind]/(0.5*(plank.p.Max_Size[2][sp])))
        end
        current_b += plank.data.biomass[ind]
    end

    # Loop to resample values until they meet the criteria
    while any(plank.data.z .<= 1) || any(plank.data.z .> maxdepth)
        # Resample z values for points outside the grid
        outside_indices = findall((plank.data.z .<= 1) .| (plank.data.z .> maxdepth))
        # Resample the values for the resampled indices
        new_values = gaussmix(length(outside_indices), z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])
        
        # Assign the resampled values to the corresponding indices in plank.data.z
        plank.data.z[outside_indices] = new_values

        plank.data.pool_z[outside_indices] = ceil.(Int, plank.data.z[outside_indices] ./ (maxdepth / depthres))
    end
    if ind != maxN #Remove the individuals that are not created out of the model domain
        plank.data.x[ind+1:maxN] .= 5e6
        plank.data.y[ind+1:maxN] .= 5e6
        plank.data.z[ind+1:maxN] .= 5e6
    end
    return plank.data
end

function generate_pool(group, g::AbstractGrid, sp, files,dt)
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

    #g per cubic meter.
    density = [sum(@view pdf_values[1][min_z[k]:max_z[k]])/sum(pdf_values[1]) .* group.characters.Total_density[2][sp] ./ z_interval for k in 1:g.Nz]

    #Randomly calculate the individuals
    ## Total biomass in grid cell as target
    ## Number of individuals 
    ## Calculate density
    patch_num = 0
    for k in 1:g.Nz
        num_patches = Int(ceil(rand() * cell_size)) #Maximum number of individuals in a patch
        target_b = density[k][1] * cell_size
        b_remaining = target_b
        for i in 1:num_patches
            patch_num += 1
            den = b_remaining * rand()
            ind_size = group.characters.Min_Size[2][sp] + rand() * (group.characters.Max_Size[2][sp] - group.characters.Min_Size[2][sp])
            ind_biom =  group.characters.LWR_a[2][sp] * (ind_size/10) ^ group.characters.LWR_b[2][sp]
            if (den < ind_biom) & (patch_num > 1)
                continue
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
            patch_vol = rand() * (group.characters.Max_patch[2][sp] / 2)^3 * π * (4 / 3)
            x = lonmin + rand() * (lonmax - lonmin)
            y = latmin + rand() * (latmax - latmin)
            z = min_z[k] + rand() * (max_z[k] - min_z[k])
            if patch_num > length(group.data.x)
                push!(group.data.x, x)
                push!(group.data.y, y)
                push!(group.data.z, z)
                push!(group.data.abundance, patch_inds)
                push!(group.data.volume, patch_vol)
                push!(group.data.biomass, den)
                push!(group.data.init_biomass, den)
                push!(group.data.length, ind_size)
                push!(group.data.ration, 0)
                push!(group.data.vis_prey, visual_range_preys_init(arch,ind_size,z,1)[1]*dt)
                push!(group.data.vis_pred, visual_range_preds_init(arch,ind_size,z,1)[1]*dt)
            else
                group.data.x[patch_num] = x
                group.data.y[patch_num] = y
                group.data.z[patch_num] = z
                group.data.abundance[patch_num] = patch_inds
                group.data.volume[patch_num] = patch_vol
                group.data.biomass[patch_num] = den
                group.data.init_biomass[patch_num] = den
                group.data.length[patch_num] = ind_size
                group.data.vis_prey[patch_num] = visual_range_preys_init(arch,ind_size,z,1)[1]*dt
                group.data.vis_pred[patch_num] = visual_range_preds_init(arch,ind_size,z,1)[1]*dt
                group.data.ration[patch_num] = 0
            end
            b_remaining -= den
        end
    end
end

function pool_growth(model)
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

function reproduce(model,sp,ind)
    #Current parameters from Fishbase oceanic species
    a = 0.03
    b = 3.3
    num_eggs = Int(round(a * model.individuals.animals[sp].data.length[ind][1] ^ b))
    sp_dat = model.individuals.animals[sp].data
    sp_char = model.individuals.animals[sp].p
    for i in 1:num_eggs
        size = 0.01 *sp_char.Max_Size[2][sp]
        biomass = sp_char.LWR_a[2][sp] .* (size ./ 10) .^ sp_char.LWR_b[2][sp]
        push!(sp_dat.length,size)
        push!(sp_dat.biomass,biomass)
        push!(sp_dat.ac, 1.0)
        push!(sp_dat.x, model.individuals.animals[sp].data.x[ind])
        push!(sp_dat.y, model.individuals.animals[sp].data.y[ind])
        push!(sp_dat.z, model.individuals.animals[sp].data.z[ind])
        push!(sp_dat.gut_fullness, 0)
        push!(sp_dat.vis_prey, visual_range_preys_init(arch,size,model.individuals.animals[sp].data.z[ind],1)[1] * sp_char.t_resolution[2][sp])
        push!(sp_dat.vis_pred, visual_range_preds_init(arch,size,model.individuals.animals[sp].data.z[ind],1)[1] * sp_char.t_resolution[2][sp])
        push!(sp_dat.pool_x, model.individuals.animals[sp].data.pool_x[ind])
        push!(sp_dat.pool_y, model.individuals.animals[sp].data.pool_y[ind])
        push!(sp_dat.pool_z, model.individuals.animals[sp].data.pool_z[ind])
        push!(sp_dat.energy, biomass * sp_char.Energy_density[2][sp] * 0.2)
        push!(sp_dat.behavior, 1.0)
        push!(sp_dat.target_z, model.individuals.animals[sp].data.z[ind])
        push!(sp_dat.dive_capable, 1)
        push!(sp_dat.feeding, 1)
        push!(sp_dat.consumed, 0)
        push!(sp_dat.eDNA_shed, 0)
        push!(sp_dat.mig_status, 0)
        push!(sp_dat.mig_rate, 0)
        push!(sp_dat.rmr, 0)
        push!(sp_dat.dives_remaining, 0)
        push!(sp_dat.interval, 0)
        push!(sp_dat.daily_ration, 0)
        push!(sp_dat.ration, 0)
        push!(sp_dat.mature,0)
    end
    model.abund[sp] += num_eggs
end