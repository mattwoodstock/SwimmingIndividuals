function timestep_results(sim::MarineSimulation)
    model = sim.model
    outputs = sim.outputs
    # Save the results periodically
    ts = Int(model.iteration)
    run = Int(sim.run)

    grid = model.depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    n_fish = length(model.fishing)

    Sp = []
    Ind = []
    x = []
    y = []
    z = []
    lengths = []
    ration = []
    energy = []
    cost = []
    active = []
    abundance = []
    biomass = []

    #Individual-scale
    for (species_index, animal_index) in enumerate(keys(model.individuals.animals))
        spec_dat = model.individuals.animals[species_index].data
        alive = findall(x -> x == 1.0, spec_dat.alive) #Number of individuals per species that are active

        specs = fill(species_index,length(alive))
        append!(Sp,specs)
        append!(Ind,alive)
        append!(x,spec_dat.x[alive])
        append!(y,spec_dat.y[alive])
        append!(z,spec_dat.z[alive])
        append!(lengths,spec_dat.length[alive])
        append!(ration,spec_dat.ration[alive])
        append!(energy,spec_dat.energy[alive])
        append!(cost,spec_dat.cost[alive])
        append!(active,spec_dat.active[alive])
        append!(abundance,spec_dat.abundance[alive])
        append!(biomass,spec_dat.biomass_school[alive])
    end

    individual_array = hcat(Sp,Ind,x,y,z,lengths,abundance,biomass,ration,energy,cost,active)
    column_names = ["Species", "Individual", "X", "Y", "Z", "Length","Abundance","Biomass", "Ration", "Energy", "Cost","Active"]
    df = DataFrame(individual_array, Symbol.(column_names))

    filename = "results/Individual/IndividualResults_$run-$ts.csv"
    CSV.write(filename, df)

    # Calculate materials for population-scale results
    M = instantaneous_mortality(outputs)
    F = fishing_mortality(outputs)
    DC = outputs.consumption

    filename = "results/Population/Instantaneous_Mort_$(run)-$(ts).h5"

    # Save variables to HDF5 file
    h5open(filename, "w") do file
        write(file, "M", M)
        write(file, "F", F)
        write(file, "Diet", DC)
    end

    #Reset outputs for next run
    outputs.Fmort = zeros(lonres,latres,depthres,n_fish,model.n_species)
    outputs.consumption = zeros(lonres,latres,depthres,model.n_species+model.n_resource,model.n_species+model.n_resource)
    outputs.mortalities = zeros(lonres,latres,depthres,model.n_species+model.n_resource,model.n_species)

    return nothing
end

function fishery_results(sim,fisheries)
    ts = Int(sim.model.iteration)
    run = Int(sim.run)
    name = []
    quotas = []
    catches_t = []
    catches_ind = []
    for fishery in fisheries
        push!(name,fishery.name)
        push!(quotas,fishery.quota)
        push!(catches_t,fishery.cumulative_catch)
        push!(catches_ind,fishery.cumulative_inds)
    end
    fish_array = hcat(name,quotas,catches_t,catches_ind)
    column_names = ["Name","Quota","Tonnage","Inds"]
    df = DataFrame(fish_array, Symbol.(column_names))

    filename = "results/Fishery/FisheryResults_$run-$ts.csv"
    CSV.write(filename, df)
end

function init_abundances(model,sp,output)
    grid = model.depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])

    spec_dat = model.individuals.animals[sp].data

    for x in 1:lonres, y in 1:latres, z in 1:depthres 
        idx = findall(p -> p.pool_x == x && p.pool_y == y && p.pool_z == z, spec_dat)

        output.abundance[x,y,z,sp] = sum(spec_dat.abundance[idx]) 
    end
    return nothing
end

function instantaneous_mortality(outputs)
    abundance = outputs.abundance
    mortality = outputs.mortalities 

    M = zeros(size(mortality))

    for sp in 1:size(mortality,5), r in 1:size(mortality,4), x in 1:size(mortality,1), y in 1:size(mortality,2), z in 1:size(mortality,3)
        if mortality[x,y,z,r,sp] == 0 || abundance[x,y,z,sp] == 0
            continue 
        end

        mort = mortality[x,y,z,r,sp] / abundance[x,y,z,sp]
        mort = clamp(mort,0,1)
        M[x,y,z,r,sp] = -log(1-mort)
    end

    return M
end

function fishing_mortality(outputs)
    abundance = outputs.abundance
    mortality = outputs.Fmort 

    F = zeros(size(mortality))

    for sp in 1:size(mortality,5), f in 1:size(mortality,4), x in 1:size(mortality,1), y in 1:size(mortality,2), z in 1:size(mortality,3)
        if mortality[x,y,z,f,sp] == 0 || abundance[x,y,z,sp] == 0
            continue 
        end

        mort = mortality[x,y,z,f,sp] / abundance[x,y,z,sp]
        mort = clamp(mort,0,1)
        F[x,y,z,f,sp] = -log(1-mort)
    end

    return F
end

function diet_matrix(outputs)
    consumption = outputs.consumption 

    Z = zeros(size(mortality))

    for sp in 1:size(mortality,5), r in 1:size(mortality,4), x in 1:size(mortality,1), y in 1:size(mortality,2), z in 1:size(mortality,3)
        if mortality[x,y,z,r,sp] == 0 || abundance[x,y,z,sp] == 0
            continue 
        end

        mort = mortality[x,y,z,r,sp] / abundance[x,y,z,sp]
        Z[x,y,z,r,sp] = -log(1-mort) #Instantaneous rate of mortality at the temporal resolution of the model

        #println(Z[x,y,z,sp])
        println(Z[x,y,z,r,sp] * 365)
    end
    stop
    return Z
end