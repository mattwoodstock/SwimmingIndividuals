# ===================================================================
# Data Structures for Simulation and Output
# ===================================================================

"""
    MarineOutputs
A mutable struct to hold all the data generated during a simulation run.
The fields are defined as `AbstractArray` to be compatible with both
CPU `Array`s and GPU `CuArray`s.
"""
mutable struct MarineOutputs
    mortalities::AbstractArray{Float32,7}
    Fmort::AbstractArray{Float32,6}
    Smort::AbstractArray{Float32,5}
    consumption::AbstractArray{Float32,7}
    abundance::AbstractArray{Float32,5}
    biomass::AbstractArray{Float32,5}
end


"""
    MarineSimulation
The main "runner" object for a single simulation experiment. It bundles
the model state with run-specific parameters
"""
mutable struct MarineSimulation
    model::MarineModel
    ΔT::Float32
    iterations::Int32
    run::Int32
    outputs::MarineOutputs
end
