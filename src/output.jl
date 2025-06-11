function generate_outputs(model)
    grid = model.depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    fish = model.fishing
    n_fish = length(fish)

    model.ninds = sum(length(model.individuals.animals[i].data) for i in 1:model.n_species)
    mortalities = zeros(lonres,latres,depthres,model.n_species+model.n_resource,model.n_species)
    Fmort = zeros(lonres,latres,depthres,n_fish,model.n_species)

    lengths = zeros(model.ninds)
    weights = zeros(model.ninds)
    consumption = zeros(lonres,latres,depthres,model.n_species+model.n_resource,model.n_species+model.n_resource)
    abundance = zeros(lonres,latres,depthres,model.n_species)

    return MarineOutputs(mortalities,Fmort,lengths,weights,consumption,abundance)
end