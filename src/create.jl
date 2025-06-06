function generate_individuals(params::Dict, arch::Architecture, Nsp::Int, B, maxN::Int,depths::MarineDepths,capacities)
    plank_names = Symbol[]
    plank_data=[]
    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_individuals(arch, params, maxN)
        initialize_individiduals(plank, B[i],i,depths,capacities)
        push!(plank_names, name)
        push!(plank_data, plank)
    end
    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks)
end

function construct_individuals(arch::Architecture, params::Dict, maxN)
    rawdata = StructArray(x = zeros(Float64,maxN), y = zeros(Float64,maxN), z = zeros(Float64,maxN),length = zeros(Float64,maxN), abundance = zeros(Float64,maxN),biomass_ind = zeros(Float64,maxN), biomass_school = zeros(Float64,maxN), energy = zeros(Float64,maxN),gut_fullness = zeros(Float64,maxN),cost = zeros(Float64,maxN), pool_x = zeros(Float64,maxN), pool_y = zeros(Float64,maxN), pool_z = zeros(Float64,maxN),active = zeros(Float64,maxN), ration = zeros(Float64,maxN), alive = zeros(Float64,maxN), vis_prey = zeros(Float64,maxN), mature = zeros(Float64,maxN),age=zeros(Float64,maxN))

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Dive_Interval,:Min_Prey,:LWR_b, :Surface_Interval,:W_mat,:SpeciesLong, :LWR_a, :Larval_Size,:Max_Prey, :Max_Size, :Dive_Max,:School_Size,:Taxa,:Larval_Duration, :Sex_Ratio,:SpeciesShort, :Dive_Min,:Handling_Time,:Energy_density, :Hatch_Survival, :MR_type,  :Swim_velo, :Biomass, :Type)

    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function initialize_individiduals(plank, B::Float64,sp::Int,depths::MarineDepths,capacities)
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
    target_b = B* 1_000_000 * cell_size #Total grams to create
    # Set plank data values
    current_b = 0
    ind = 0
    school_size = plank.p.School_Size[2][sp]
    max_size = plank.p.Max_Size[2][sp]
    max_weight = plank.p.LWR_a[2][sp] .* (max_size ./ 10) .^ plank.p.LWR_b[2][sp]

    while current_b < target_b
        ind += 1

        if ind > length(plank.data.length)
            μ,σ = lognormal_params_from_maxsize(max_size)
            dist = LogNormal(μ, σ)
            new_length = rand(dist)
            # Repeat until within allowed size
            while new_length > max_size
                new_length = rand(dist)
            end
            ind_biomass = plank.p.LWR_a[2][sp] * (new_length / 10) ^ plank.p.LWR_b[2][sp]
            school_biomass = ind_biomass * school_size

            push!(plank.data.length,new_length)
            push!(plank.data.biomass_ind,ind_biomass)
            push!(plank.data.biomass_school,school_biomass)
            push!(plank.data.abundance,school_size)
        else
            #Lognormal distribution selection from a given max size (this is a default and should be adjusted with appropriate data)
            μ,σ = lognormal_params_from_maxsize(max_size)
            dist = LogNormal(μ, σ)
            new_length = rand(dist)
            # Repeat until within allowed size
            while new_length > max_size
                new_length = rand(dist)
            end
            plank.data.length[ind] = new_length

            #Random length between 0 and max size (worse default)
            #plank.data.length[ind] = rand() .* (plank.p.Max_Size[2][sp])

            ind_biomass = plank.p.LWR_a[2][sp] .* (plank.data.length[ind] ./ 10) .^ plank.p.LWR_b[2][sp]
            school_biomass = ind_biomass .* school_size

            plank.data.biomass_ind[ind] = ind_biomass
            plank.data.biomass_school[ind] = school_biomass
            plank.data.abundance[ind] = school_size

            plank.data.alive[ind] = 1.0

            plank.data.x[ind], plank.data.y[ind], plank.data.pool_x[ind],plank.data.pool_y[ind] = initial_ind_placement(capacities,sp,grid,1)

            plank.data.z[ind] = gaussmix(1, night_profs[sp, "mu1"], night_profs[sp, "mu2"],night_profs[sp, "mu3"], night_profs[sp, "sigma1"],night_profs[sp, "sigma2"], night_profs[sp, "sigma3"],night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])[1]
            plank.data.gut_fullness[ind] = (rand() * 0.2 * plank.data.biomass_school[ind]) / plank.data.biomass_school[ind]
            plank.data.vis_prey[ind] = visual_range_preys_init(plank.data.length[ind],plank.data.z[ind],plank.p.Min_Prey[2][sp],plank.p.Max_Prey[2][sp],1)[1] * dt
            # Calculate pool indices
            
            plank.data.pool_z[ind] = max(1,ceil(Int, plank.data.z[ind] / (maxdepth / depthres)))
            plank.data.pool_z[ind] = clamp(plank.data.pool_z[ind],1,depthres)
            plank.data.energy[ind]  = plank.data.biomass_school[ind] * plank.p.Energy_density[2][sp] * 0.2   # Initial reserve energy = Rmax
            plank.data.cost[ind] = 0
            plank.data.active[ind] = 0
            plank.data.mature[ind] = min(1,plank.data.biomass_ind[ind] / (plank.p.W_mat[2][sp]*(max_weight)))
            plank.data.age[ind] = plank.p.Larval_Duration[2][sp]
        end
        current_b += plank.data.biomass_school[ind]
    end

    to_append = ind - 1
    x, y, pool_x, pool_y = initial_ind_placement(capacities,sp,grid,to_append)
    z = gaussmix(to_append, night_profs[sp, "mu1"], night_profs[sp, "mu2"],night_profs[sp, "mu3"], night_profs[sp, "sigma1"],night_profs[sp, "sigma2"], night_profs[sp, "sigma3"],
    night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])

    z = clamp.(z,1,maxdepth)

    pool_z = max.(1,ceil.(Int,z ./ (maxdepth/depthres)))
    pool_z = clamp.(pool_z,1,depthres)
    append!(plank.data.alive, fill(1.0,to_append))
    append!(plank.data.x, x)
    append!(plank.data.y, y)
    append!(plank.data.z, z)
    append!(plank.data.gut_fullness, (rand(to_append) .* 0.2 .* plank.data.biomass_school[(2:ind)]) ./ plank.data.biomass_school[(2:ind)])
    append!(plank.data.vis_prey, visual_range_preys_init(plank.data.length[(2:ind)],z,plank.p.Min_Prey[2][sp],plank.p.Max_Prey[2][sp],to_append) .* dt)
    append!(plank.data.pool_x, pool_x)
    append!(plank.data.pool_y, pool_y)
    append!(plank.data.pool_z, pool_z)
    append!(plank.data.energy, plank.data.biomass_school[(2:ind)] .* plank.p.Energy_density[2][sp] .* 0.2)
    append!(plank.data.cost, fill(0,to_append))
    append!(plank.data.active, fill(0,to_append))
    append!(plank.data.ration, fill(0,to_append))
    append!(plank.data.mature, min.(1,plank.data.biomass_ind[(2:ind)] ./ (plank.p.W_mat[2][sp]*(max_size))))
    append!(plank.data.age,fill(plank.p.Larval_Duration[2][sp],to_append))
    return plank.data
end

function reproduce(model,sp,ind,energy,val)
    sp_dat = model.individuals.animals[sp].data
    sp_char = model.individuals.animals[sp].p

    total_abundance = sum(sp_dat.abundance[findall(x -> x == 1.0, sp_dat.alive)])
    K = model.init_abund[sp]
    density_factor = 1 / (1 + total_abundance / K)

    egg_volume = 0.15 .* sp_dat.biomass_ind[ind] .^ 0.14 #From Barneche et al. 2018
    egg_energy = 2.15 .* egg_volume .^ 0.77

    spent_energy = energy .* val

    sp_dat.energy[ind] .-= spent_energy

    num_eggs = ceil.(Int, (spent_energy ./ egg_energy) .* sp_char.Sex_Ratio[2][sp] .* sp_char.Hatch_Survival[2][sp] .* density_factor)

    parent_x = sp_dat.x[ind]
    parent_y = sp_dat.y[ind]
    parent_z = sp_dat.z[ind]

    parent_pool_x = sp_dat.pool_x[ind]
    parent_pool_y = sp_dat.pool_y[ind]
    parent_pool_z = sp_dat.pool_z[ind]

    to_replace = findall(x -> x == 0,sp_dat.alive)
    
    for i in 1:length(num_eggs)
        if i <= length(to_replace) #Replace non-active pools to save space 
            sp_dat.x[to_replace[i]] = parent_x[i]
            sp_dat.y[to_replace[i]] = parent_y[i]
            sp_dat.z[to_replace[i]] = parent_z[i]
            sp_dat.abundance[to_replace[i]] = num_eggs[i]
            sp_dat.length[to_replace[i]] = rand() * sp_char.Larval_Size[2][sp]
            sp_dat.biomass_ind[to_replace[i]] = sp_char.LWR_a[2][sp] * (sp_dat.length[to_replace[i]]/10) ^ sp_char.LWR_b[2][sp]
            sp_dat.biomass_school[to_replace[i]] = sp_dat.biomass_ind[to_replace[i]] * sp_char.School_Size[2][sp]
            sp_dat.vis_prey[to_replace[i]] = visual_range_prey(model,sp_dat.length[to_replace[i]], sp_dat.z[to_replace[i]],sp, 1)[1] * model.dt
            sp_dat.ration[to_replace[i]] = 0.0
            sp_dat.pool_x[to_replace[i]] = parent_pool_x[i]
            sp_dat.pool_y[to_replace[i]] = parent_pool_y[i]
            sp_dat.pool_z[to_replace[i]] = parent_pool_z[i]

            sp_dat.age[to_replace[i]] = 0.0
            sp_dat.alive[to_replace[i]] = 1.0
            sp_dat.energy[to_replace[i]] = sp_dat.biomass_school[to_replace[i]] * sp_char.Energy_density[2][sp] * 0.2
            sp_dat.gut_fullness[to_replace[i]] = (0.2 * sp_dat.biomass_school[to_replace[i]])/sp_dat.biomass_school[to_replace[i]]
            sp_dat.cost[to_replace[i]] = 0.0
            sp_dat.active[to_replace[i]] = 0.0
            sp_dat.mature[to_replace[i]] = 0.0
        else
            push!(sp_dat.x, parent_x[i])
            push!(sp_dat.y, parent_y[i])
            push!(sp_dat.z, parent_z[i])
            push!(sp_dat.abundance, num_eggs[i])
            new_length = rand() * sp_char.Larval_Size[2][sp]
            push!(sp_dat.length, new_length)
            new_biomass = sp_char.LWR_a[2][sp] * (new_length/10) ^ sp_char.LWR_b[2][sp]
            new_school_biom = new_biomass * sp_char.School_Size[2][sp]
            push!(sp_dat.biomass_ind, new_biomass)
            push!(sp_dat.biomass_school, new_school_biom)
            push!(sp_dat.vis_prey, visual_range_prey(model,new_length, parent_z[i],sp, 1)[1] * model.dt)
            push!(sp_dat.ration, 0.0)
            push!(sp_dat.pool_x, parent_pool_x[i])
            push!(sp_dat.pool_y, parent_pool_y[i])
            push!(sp_dat.pool_z, parent_pool_z[i])
            push!(sp_dat.age, 0.0)
            push!(sp_dat.alive, 1.0)
            push!(sp_dat.energy, new_school_biom * sp_char.Energy_density[2][sp] * 0.2)
            new_fullness = (0.2 * new_school_biom) / new_school_biom
            push!(sp_dat.gut_fullness, new_fullness) #Start with a full stomach
            push!(sp_dat.cost, 0.0)
            push!(sp_dat.active, 0.0)
            push!(sp_dat.mature, 0.0)
        end
    end
end

function initialize_resources(traits,n_spec,n_resource,depths,capacities)
    grid = depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    latmax = grid[grid.Name .== "yulcorner", :Value][1]

    maxdepth = Int(grid[grid.Name .== "depthmax", :Value][1])
    cell_size = grid[grid.Name .== "cellsize", :Value][1]

    lat_rad = deg2rad(latmax)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(lat_rad)
    cell_area = (km_per_deg_lat * cell_size) * (km_per_deg_lon * cell_size)

    depth_values = range(0, stop=maxdepth, length=depthres+1) |> collect
    resources = resource[]
    ind = 1
    for i in 1:n_resource
        caps = capacities[:,:,1,i+n_spec]
        sum_caps = sum(caps[:,:])
        n_pack = traits[i,:Packets]
        density = traits[i,:Biomass] * 1_000_000 #biomass density in grams

        night_profs = depths.patch_night

        means = [night_profs[i, "mu1"], night_profs[i, "mu2"], night_profs[i, "mu3"]]
        stds = [night_profs[i, "sigma1"], night_profs[i, "sigma2"], night_profs[i, "sigma3"]]
        weights = [night_profs[i, "lambda1"], night_profs[i, "lambda2"], night_profs[i, "lambda3"]]

        x_values = depth_values[1]:depth_values[end]
        pdf_values = multimodal_distribution.(Ref(x_values), means, stds, weights)

        depth_weights_norm = pdf_values[1] ./ sum(pdf_values[1])

        depth_props = [sum(depth_weights_norm[(x_values .>= depth_values[i]) .& (x_values .< depth_values[i+1])]) for i in 1:length(depth_values)-1]

        for j in 1:latres, k in 1:lonres

            if caps[j,k] > 0
                area_prop = caps[j,k] / sum_caps
                biomass_total = area_prop * density * cell_area
                biomass_target = biomass_total .* depth_props
                for l in 1:depthres
                    packet_biomass = biomass_target[l] / n_pack
                    x,y,pool_x,pool_y = initial_ind_placement(capacities,n_spec+i,grid,n_pack)

                    z = depth_values[l] + rand() * (depth_values[l+1]-depth_values[l])
                    pool_z = l

                    for m in 1:n_pack
                        push!(resources, resource(i,ind,x[m],y[m],z,pool_x[m],pool_y[m],pool_z,packet_biomass,packet_biomass))
                        ind += 1
                    end
                end
            end
        end
    end
    return resources
end

function resource_growth(model)
    for i in 1:model.n_resource
        rate = model.resource_trait[i,:Growth]
        matching_idxs = findall(r -> r.sp == i, model.resources)

        population = getfield.(model.resources[matching_idxs], :biomass)
        carrying_capacity = getfield.(model.resources[matching_idxs], :capacity)

        for (i, idx) in enumerate(matching_idxs)
            model.resources[idx].biomass = population[i] + rate * population[i] * (1 - population[i] / carrying_capacity[i])
        end
    end
end
