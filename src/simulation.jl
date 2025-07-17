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
    mortalities::AbstractArray{Int64,5}
    Fmort::AbstractArray{Int64,5}
    consumption::AbstractArray{Float64,5}
    abundance::AbstractArray{Float64,4}
end


"""
    MarineSimulation
The main "runner" object for a single simulation experiment. It bundles
the model state with run-specific parameters
"""
mutable struct MarineSimulation
    model::MarineModel
    ΔT::Float64
    iterations::Int64
    run::Int64
    outputs::MarineOutputs
end
