mutable struct MarineOutputs
    mortalities::Array{Int64,5}
    Fmort::Array{Int64,5}
    lengths::Vector{Float64}
    weights::Vector{Float64}
    consumption::Array{Float64,5}
    abundance::Array{Float64,4}
end

mutable struct MarineSimulation
    model::MarineModel                       # Model object
    ΔT::Float64                                  # model time step
    iterations::Int64                          # run the simulation for this number of iterations
    run::Int64                                  #Model run ID
    outputs::MarineOutputs
    #output_writer::Union{MarineOutputWriter,Nothing} # Output writer
end