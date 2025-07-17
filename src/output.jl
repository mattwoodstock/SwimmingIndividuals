# ===================================================================
# Data Structures and Functions for Model Output
# ===================================================================

"""
    generate_outputs(model::MarineModel)
This function creates the data structures for storing simulation results.
It is architecture-aware and will create arrays on the GPU if the model
is configured to run on a GPU.
"""
function generate_outputs(model::MarineModel)
    arch = model.arch
    grid = model.depths.grid
    
    # Get grid dimensions
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    n_fish = length(model.fishing)

    # --- Create output arrays on the correct device (CPU or GPU) ---
    mortalities = array_type(arch)(zeros(Int64, lonres, latres, depthres, model.n_species + model.n_resource, model.n_species))
    Fmort = array_type(arch)(zeros(Int64, lonres, latres, depthres, n_fish, model.n_species))
    consumption = array_type(arch)(zeros(Float64, lonres, latres, depthres, model.n_species + model.n_resource, model.n_species + model.n_resource))
    abundance = array_type(arch)(zeros(Float64, lonres, latres, depthres, model.n_species + model.n_resource))
    
    return MarineOutputs(mortalities, Fmort, consumption, abundance)
end
